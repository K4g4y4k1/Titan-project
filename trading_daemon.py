import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from aiohttp import web

# --- CONFIGURATION (OPENROUTER & OMNI-SIGHT) ---
load_dotenv()
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")
# Note: L'environnement fournira la clé si configuré, sinon vide pour le mode test

CONFIG = {
    "VERSION": "7.9.6-Omni-Sight-Router",
    "DB_PATH": "titan_v7_omni.db",
    "AI_MODEL": "google/gemini-2.0-flash-exp:free",
    "RISK_PER_TRADE_PCT": 0.01,
    "MAX_OPEN_POSITIONS": 5,
    "DAILY_LOSS_LIMIT_PCT": 0.02,
    "CONSECUTIVE_SL_HALT": 3,
    "COOLDOWN_MIN": 15,
    "MAX_TRADES_PER_CYCLE": 2,
    "MIN_CONF_LIVE": 82,
    "SCAN_INTERVAL_SEC": 180,
    "WATCHLIST": ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "COIN", "PLTR", "MARA"]
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Titan-OmniRouter")

# --- PERSISTANCE ET AUDIT (IDENTIQUE V7.9.5) ---
class AbsoluteDB:
    def __init__(self, path):
        self.path = path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            # Table des décisions (L'intention IA)
            conn.execute("""CREATE TABLE IF NOT EXISTS ai_decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                confidence INTEGER,
                thesis TEXT,
                decision TEXT,                -- EXECUTED / REJECTED
                rejection_reason TEXT,
                entry_price_at_decision REAL,
                tp_price REAL,
                sl_price REAL,
                market_context TEXT
            )""")
            # Table des trades (L'exécution moteur)
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_decision_id INTEGER,
                symbol TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_time DATETIME,
                result TEXT,
                confidence INTEGER,
                tp_target REAL,
                sl_target REAL,
                max_favorable_price REAL,
                max_adverse_price REAL,
                status TEXT DEFAULT 'OPEN',
                FOREIGN KEY(ai_decision_id) REFERENCES ai_decision_log(id)
            )""")
            conn.execute("CREATE TABLE IF NOT EXISTS daily_stats (date TEXT PRIMARY KEY, start_equity REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()

    def log_decision(self, data):
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("""INSERT INTO ai_decision_log 
                (symbol, confidence, thesis, decision, rejection_reason, entry_price_at_decision, tp_price, sl_price, market_context) 
                VALUES (?,?,?,?,?,?,?,?,?)""", (
                    data['symbol'], data['confidence'], data['thesis'], data['decision'], 
                    data['rejection_reason'], data['entry_price'], data['tp_price'], data['sl_price'],
                    json.dumps(data.get('context', {}))
                ))
            return cursor.lastrowid

    def log_trade_start(self, decision_id, symbol, qty, price, conf, tp, sl):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""INSERT INTO trades 
                (ai_decision_id, symbol, qty, entry_price, confidence, tp_target, sl_target, max_favorable_price, max_adverse_price) 
                VALUES (?,?,?,?,?,?,?,?,?)""", (decision_id, symbol, qty, price, conf, tp, sl, price, price))

    def update_excursion(self, trade_id, current_price):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""UPDATE trades SET 
                max_favorable_price = MAX(max_favorable_price, ?),
                max_adverse_price = MIN(max_adverse_price, ?)
                WHERE id = ?""", (current_price, current_price, trade_id))

    def close_trade_record(self, symbol, exit_price, result):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""UPDATE trades SET 
                exit_price = ?, exit_time = CURRENT_TIMESTAMP, result = ?, status = 'CLOSED'
                WHERE symbol = ? AND status = 'OPEN'""", (exit_price, result, symbol))

    def get_system_halt(self):
        with sqlite3.connect(self.path) as conn:
            res = conn.execute("SELECT value FROM system_state WHERE key='halted'").fetchone()
            return res[0] == '1' if res else False

# --- MOTEUR TITAN AVEC OPENROUTER ---
class TitanOmniRouter:
    def __init__(self):
        self.alpaca = tradeapi.REST(os.getenv('ALPACA_KEY'), os.getenv('ALPACA_SECRET'), "https://paper-api.alpaca.markets")
        self.db = AbsoluteDB(CONFIG["DB_PATH"])
        self.last_trade_time = {}
        self.daily_start_equity = 0

    async def get_market_context(self, symbol):
        try:
            bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=5)
            if not bars: return {}
            return {
                "volatility": round((max(b.h for b in bars) - min(b.l for b in bars)) / bars[-1].c * 100, 3),
                "volume": bars[-1].v
            }
        except: return {}

    async def get_ia_picks(self):
        """Appel IA via OpenRouter avec format Chat Completion"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "HTTP-Referer": "https://titan-trading.system",
            "X-Title": "Titan Omni-Sight",
            "Content-Type": "application/json"
        }
        
        prompt = (f"Analyze these stocks: {CONFIG['WATCHLIST']}. "
                  "Identify high-conviction momentum breakouts. "
                  "Return a JSON object with a key 'picks' containing a list of objects: "
                  "{ 'symbol': string, 'confidence': integer 0-100, 'thesis': string (max 20 words) }")

        payload = {
            "model": CONFIG["AI_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"OpenRouter Error {resp.status}: {await resp.text()}")
                        return []
                    data = await resp.json()
                    content = data['choices'][0]['message']['content']
                    return json.loads(content).get("picks", [])
        except Exception as e:
            logger.error(f"OpenRouter Connection Error: {e}")
            return []

    async def execute_cycle(self):
        acc = self.alpaca.get_account()
        equity = float(acc.equity)
        is_halted = self.db.get_system_halt()
        
        picks = await self.get_ia_picks()
        trades_count = 0

        for pick in picks:
            symbol = pick.get('symbol')
            conf = pick.get('confidence', 0)
            thesis = pick.get('thesis', 'N/A')
            
            context = await self.get_market_context(symbol)
            try:
                price = self.alpaca.get_latest_trade(symbol).price
            except: price = 0
            
            # Paramètres de sortie théoriques pour l'audit
            tp, sl = round(price * 1.05, 2), round(price * 0.97, 2)

            decision_data = {
                "symbol": symbol, "confidence": conf, "thesis": thesis,
                "entry_price": price, "tp_price": tp, "sl_price": sl,
                "context": context, "decision": "REJECTED", "rejection_reason": "NONE"
            }

            # CHAÎNE DE DÉCISION
            if is_halted:
                decision_data["rejection_reason"] = "SYSTEM_HALTED"
            elif conf < CONFIG["MIN_CONF_LIVE"]:
                decision_data["rejection_reason"] = "LOW_CONFIDENCE"
            elif symbol in self.last_trade_time and (datetime.now() - self.last_trade_time[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_MIN"]):
                decision_data["rejection_reason"] = "COOLDOWN_ACTIVE"
            elif len(self.alpaca.list_positions()) >= CONFIG["MAX_OPEN_POSITIONS"]:
                decision_data["rejection_reason"] = "MAX_POSITIONS_REACHED"
            elif trades_count >= CONFIG["MAX_TRADES_PER_CYCLE"]:
                decision_data["rejection_reason"] = "CYCLE_LIMIT"
            else:
                # ÉLIGIBLE
                try:
                    risk = abs(price - sl)
                    qty = round((equity * CONFIG["RISK_PER_TRADE_PCT"]) / risk, 2)
                    if qty > 0:
                        self.alpaca.submit_order(symbol=symbol, qty=qty, side='buy', type='market', 
                                                time_in_force='gtc', order_class='bracket',
                                                take_profit={'limit_price': tp}, stop_loss={'stop_price': sl})
                        decision_data["decision"] = "EXECUTED"
                except Exception as e:
                    decision_data["rejection_reason"] = f"EXEC_FAIL: {str(e)[:30]}"

            # LOG DE LA DÉCISION (Audit intégral)
            d_id = self.db.log_decision(decision_data)

            if decision_data["decision"] == "EXECUTED":
                self.db.log_trade_start(d_id, symbol, qty, price, conf, tp, sl)
                self.last_trade_time[symbol] = datetime.now()
                trades_count += 1
                logger.info(f"TRADE EXECUTED: {symbol} at {price}")

    async def monitor_and_reconcile(self):
        """Réconcilie les positions fermées par le broker avec la base locale"""
        try:
            positions = {p.symbol: p for p in self.alpaca.list_positions()}
            with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
                conn.row_factory = sqlite3.Row
                open_db_trades = conn.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()

            for t in open_db_trades:
                symbol = t['symbol']
                if symbol not in positions:
                    # Le trade est fermé côté broker
                    last_px = self.alpaca.get_latest_trade(symbol).price
                    result = "TP" if last_px >= t['tp_target'] else "SL"
                    if last_px < t['tp_target'] and last_px > t['sl_target']: result = "MANUAL_OR_OTHER"
                    
                    self.db.close_trade_record(symbol, last_px, result)
                    logger.info(f"RECONCILED: {symbol} closed as {result}")
                else:
                    # Update excursions
                    curr_px = float(positions[symbol].current_price)
                    self.db.update_excursion(t['id'], curr_px)
        except Exception as e:
            logger.error(f"Reconciliation Error: {e}")

    async def main_loop(self):
        logger.info(f"TITAN {CONFIG['VERSION']} ON (Model: {CONFIG['AI_MODEL']})")
        while True:
            if self.alpaca.get_clock().is_open:
                await self.execute_cycle()
                await self.monitor_and_reconcile()
                await asyncio.sleep(CONFIG["SCAN_INTERVAL_SEC"])
            else:
                logger.info("Market Closed. Monitoring only.")
                await asyncio.sleep(60)

# --- DASHBOARD MINIMAL ---
async def start_api(bot):
    app = web.Application()
    async def get_status(request):
        with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
            conn.row_factory = sqlite3.Row
            audit_log = conn.execute("SELECT * FROM ai_decision_log ORDER BY timestamp DESC LIMIT 20").fetchall()
            return web.json_response({
                "bot": CONFIG["VERSION"],
                "model": CONFIG["AI_MODEL"],
                "audit": [dict(r) for r in audit_log]
            })
    app.router.add_get('/status', get_status)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', 8080).start()

if __name__ == "__main__":
    titan = TitanOmniRouter()
    loop = asyncio.get_event_loop()
    loop.create_task(start_api(titan))
    loop.run_until_complete(titan.main_loop())
