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

# --- CONFIGURATION (ABSOLUTE-VISIBILITY) ---
load_dotenv()
GEMINI_API_KEY = "" # Géré par l'environnement
API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN', "admin_secret_token")

CONFIG = {
    "VERSION": "7.9.5-Absolute-Visibility",
    "DB_PATH": "titan_v7_absolute.db",
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
logger = logging.getLogger("Titan-Absolute")

# --- COUCHE DE DONNÉES (FULL AUDIT) ---
class AbsoluteDB:
    def __init__(self, path):
        self.path = path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            # Table des décisions (L'intention)
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
                market_context TEXT           -- JSON: Volatilité, Volume
            )""")
            # Table des trades (L'exécution)
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_decision_id INTEGER,       -- Lien vers l'audit
                symbol TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_time DATETIME,
                result TEXT,                  -- TP / SL / MANUAL
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

# --- MOTEUR TITAN ---
class TitanOmniSight:
    def __init__(self):
        self.alpaca = tradeapi.REST(os.getenv('ALPACA_KEY'), os.getenv('ALPACA_SECRET'), "https://paper-api.alpaca.markets")
        self.db = AbsoluteDB(CONFIG["DB_PATH"])
        self.last_trade_time = {}
        self.daily_start_equity = 0

    async def get_market_context(self, symbol):
        """Récupère un snapshot rapide du marché pour l'audit"""
        try:
            bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=5)
            if not bars: return {}
            return {
                "volatility_15m": round((max(b.h for b in bars) - min(b.l for b in bars)) / bars[-1].c * 100, 3),
                "last_volume": bars[-1].v
            }
        except: return {}

    async def get_ia_picks(self):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
        prompt = (f"Analyze {CONFIG['WATCHLIST']}. Identify momentum breakouts. "
                 "Return JSON: { 'picks': [{ 'symbol': 'XYZ', 'confidence': 85, 'thesis': 'Reasoning' }] }")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}) as resp:
                    data = await resp.json()
                    return json.loads(data['candidates'][0]['content']['parts'][0]['text']).get("picks", [])
        except Exception as e:
            logger.error(f"IA Error: {e}")
            return []

    async def execute_cycle(self):
        acc = self.alpaca.get_account()
        equity = float(acc.equity)
        
        # Check Circuit Breakers
        is_halted = self.db.get_system_halt()
        picks = await self.get_ia_picks()
        trades_count = 0

        for pick in picks:
            symbol = pick['symbol']
            conf = pick['confidence']
            thesis = pick['thesis']
            
            # Contexte et prix
            context = await self.get_market_context(symbol)
            try:
                price = self.alpaca.get_latest_trade(symbol).price
            except: price = 0
            
            tp, sl = round(price * 1.04, 2), round(price * 0.98, 2)

            # Préparation du Log d'Audit
            decision_data = {
                "symbol": symbol, "confidence": conf, "thesis": thesis,
                "entry_price": price, "tp_price": tp, "sl_price": sl,
                "context": context, "decision": "REJECTED", "rejection_reason": "NONE"
            }

            # FILTRAGE LOGIQUE (Documenté)
            if is_halted:
                decision_data["rejection_reason"] = "SYSTEM_HALTED"
            elif conf < CONFIG["MIN_CONF_LIVE"]:
                decision_data["rejection_reason"] = "LOW_CONFIDENCE"
            elif symbol in self.last_trade_time and (datetime.now() - self.last_trade_time[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_MIN"]):
                decision_data["rejection_reason"] = "COOLDOWN_ACTIVE"
            elif len(self.alpaca.list_positions()) >= CONFIG["MAX_OPEN_POSITIONS"]:
                decision_data["rejection_reason"] = "MAX_POSITIONS_REACHED"
            elif trades_count >= CONFIG["MAX_TRADES_PER_CYCLE"]:
                decision_data["rejection_reason"] = "CYCLE_LIMIT_REACHED"
            else:
                # ÉLIGIBLE À L'EXÉCUTION
                try:
                    risk = abs(price - sl)
                    qty = round((equity * CONFIG["RISK_PER_TRADE_PCT"]) / risk, 2)
                    if qty > 0:
                        self.alpaca.submit_order(symbol=symbol, qty=qty, side='buy', type='market', 
                                                time_in_force='gtc', order_class='bracket',
                                                take_profit={'limit_price': tp}, stop_loss={'stop_price': sl})
                        decision_data["decision"] = "EXECUTED"
                        logger.info(f"ORDER SENT: {symbol}")
                except Exception as e:
                    decision_data["rejection_reason"] = f"EXEC_ERR: {str(e)[:40]}"

            # ENREGISTREMENT DE LA DÉCISION (Même si rejeté)
            d_id = self.db.log_decision(decision_data)

            # Si exécuté, lier au trade
            if decision_data["decision"] == "EXECUTED":
                self.db.log_trade_start(d_id, symbol, qty, price, conf, tp, sl)
                self.last_trade_time[symbol] = datetime.now()
                trades_count += 1

    async def monitor_and_reconcile(self):
        """Surveille les positions et réconcilie avec la DB lors de la fermeture"""
        try:
            positions = {p.symbol: p for p in self.alpaca.list_positions()}
            with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
                conn.row_factory = sqlite3.Row
                open_db_trades = conn.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()

            for t in open_db_trades:
                symbol = t['symbol']
                # Si le trade est en DB mais plus chez Alpaca -> Fermé par TP/SL
                if symbol not in positions:
                    last_px = self.alpaca.get_latest_trade(symbol).price
                    result = "TP" if last_px >= t['tp_target'] else "SL"
                    if last_px < t['tp_target'] and last_px > t['sl_target']: result = "MANUAL_OR_OTHER"
                    
                    self.db.close_trade_record(symbol, last_px, result)
                    logger.info(f"TRADE RECONCILED: {symbol} closed as {result} at {last_px}")
                else:
                    # Toujours ouvert -> Update MFE/MAE
                    curr_px = float(positions[symbol].current_price)
                    self.db.update_excursion(t['id'], curr_px)
        except Exception as e:
            logger.error(f"Monitor error: {e}")

    async def run(self):
        logger.info(f"TITAN {CONFIG['VERSION']} is LIVE.")
        while True:
            if self.alpaca.get_clock().is_open:
                await self.execute_cycle()
                await self.monitor_and_reconcile()
                await asyncio.sleep(CONFIG["SCAN_INTERVAL_SEC"])
            else:
                await asyncio.sleep(60)

# --- DASHBOARD API ---
async def start_dashboard(bot):
    app = web.Application()
    async def get_stats(request):
        with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
            conn.row_factory = sqlite3.Row
            audit = conn.execute("SELECT * FROM ai_decision_log ORDER BY timestamp DESC LIMIT 10").fetchall()
            trades = conn.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10").fetchall()
            return web.json_response({
                "bot": CONFIG["VERSION"],
                "decisions": [dict(r) for r in audit],
                "last_trades": [dict(r) for r in trades]
            })
    app.router.add_get('/audit', get_stats)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', 8080).start()

if __name__ == "__main__":
    titan = TitanOmniSight()
    loop = asyncio.get_event_loop()
    loop.create_task(start_dashboard(titan))
    loop.run_until_complete(titan.run())
