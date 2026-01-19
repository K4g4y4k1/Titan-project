import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import re
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from aiohttp import web
import aiohttp_cors

# --- CONFIGURATION ---
load_dotenv()
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")
API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN', "change_me_immediately")

CONFIG = {
    "VERSION": "6.6.5-Apex",
    "PORT": 8080,
    "DB_PATH": "titan_v6_ultimate.db",
    "MAX_TRADES_PER_DAY": 15,
    "MAX_OPEN_POSITIONS": 3,
    "RISK_PER_TRADE_PCT": 0.02,
    "MARKET_STRESS_THRESHOLD": 0.8, 
    "COOLDOWN_PER_SYMBOL_MIN": 15, 
    "TP_PCT": 1.03,
    "SL_PCT": 0.98,
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "google/gemini-2.0-flash-exp:free"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Titan-Sentinel")

# --- PERSISTANCE ---
class TitanDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, qty REAL, entry_price REAL, exit_price REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, exit_time DATETIME,
                    result TEXT, confidence INTEGER, thesis TEXT, mode TEXT,
                    tp_price REAL, sl_price REAL, status TEXT DEFAULT 'OPEN'
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS daily_stats (day DATE PRIMARY KEY, start_equity REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('is_halted', '0')")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('halt_reason', '')")
            conn.commit()

    def get_or_create_daily_stats(self, current_equity):
        today = date.today().isoformat()
        if current_equity <= 10.0: return 0.0
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT start_equity FROM daily_stats WHERE day = ?", (today,)).fetchone()
            if not res:
                conn.execute("INSERT INTO daily_stats (day, start_equity) VALUES (?, ?)", (today, current_equity))
                return current_equity
            return res[0]

    def set_halt_state(self, halted, reason=""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'is_halted'", ('1' if halted else '0',))
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'halt_reason'", (reason,))
            conn.commit()

    def get_halt_state(self):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT value FROM system_state WHERE key IN ('is_halted', 'halt_reason') ORDER BY key DESC").fetchall()
            return (res[0][0] == '1', res[1][0])

    def log_trade_start(self, symbol, qty, price, conf, thesis, mode, tp, sl):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status) 
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN'))
            conn.commit()

    def get_open_shadow_trades(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM trades WHERE mode='SHADOW' AND status='OPEN'").fetchall()]

    def close_trade(self, trade_id, exit_price, result):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""UPDATE trades SET exit_price=?, exit_time=CURRENT_TIMESTAMP, 
                         result=?, status='CLOSED' WHERE id=?""", (exit_price, result, trade_id))
            conn.commit()

    def get_consecutive_losses(self):
        with sqlite3.connect(self.db_path) as conn:
            trades = conn.execute("SELECT result FROM trades WHERE mode='LIVE' AND status='CLOSED' ORDER BY id DESC LIMIT 5").fetchall()
            count = 0
            for (res,) in trades:
                if res == "SL": count += 1
                else: break
            return count

    def get_recent_decisions(self, limit=5):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT entry_time, symbol, mode, confidence, thesis, status FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- CORRECTION BRD 1.1 : STATS JOURNALIÈRES ---
    def get_daily_shadow_performance(self):
        with sqlite3.connect(self.db_path) as conn:
            # On filtre sur la date du jour (exit_time)
            query_base = "FROM trades WHERE mode='SHADOW' AND status='CLOSED' AND date(exit_time) = date('now')"
            total = conn.execute(f"SELECT COUNT(*) {query_base}").fetchone()[0]
            tp = conn.execute(f"SELECT COUNT(*) {query_base} AND result='TP'").fetchone()[0]
            sl = conn.execute(f"SELECT COUNT(*) {query_base} AND result='SL'").fetchone()[0]
            winrate = (tp / total * 100) if total > 0 else 0
            return {"total": total, "tp": tp, "sl": sl, "winrate": round(winrate, 2)}

# --- MOTEUR TITAN ---
class TitanEngine:
    def __init__(self):
        self.alpaca_key = os.getenv('ALPACA_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET')
        base_url = "https://paper-api.alpaca.markets" if CONFIG["ENV_MODE"] == 'PAPER' else "https://api.alpaca.markets"
        self.alpaca = tradeapi.REST(self.alpaca_key, self.alpaca_secret, base_url)
        self.db = TitanDatabase(CONFIG["DB_PATH"])
        self.session = None
        self.last_trade_per_symbol = {} 
        self.equity_stabilized = False 
        
        is_halted, reason = self.db.get_halt_state()
        
        # STRUCTURE GELÉE + ADDITIONS BRD
        self.status = {
            "version": CONFIG["VERSION"],
            "state": "HALTED" if is_halted else "INIT",
            "market": "CLOSED",
            "halt_reason": reason,
            "equity": {"current": 0.0, "pnl_pct": 0.0},
            "safety": {"consecutive_sl": 0, "market_stress": False},
            "positions": {"live": 0, "shadow_open": 0},
            
            # --- CORRECTION BRD 1.1 : trades.last ---
            "trades": {"last": None},
            "decisions": {"recent": []},
            "shadow": {"open": [], "stats": {"total": 0, "tp": 0, "sl": 0, "winrate": 0}},
            "insights": {"last_rejection": None}
        }

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def emergency_halt(self, reason):
        logger.critical(f"HALT D'URGENCE : {reason}")
        self.db.set_halt_state(True, reason)
        self.status["state"] = "HALTED"
        self.status["halt_reason"] = reason
        try:
            for pos in self.alpaca.list_positions():
                self.alpaca.submit_order(symbol=pos.symbol, qty=pos.qty, side='sell', type='market', time_in_force='gtc')
        except Exception as e: logger.error(f"Erreur liquidation : {e}")

    async def sync_daily_equity(self):
        try:
            acc = self.alpaca.get_account()
            current_eq = float(acc.equity)
            clock = self.alpaca.get_clock()
            
            if not clock.is_open:
                start_eq = current_eq 
            else:
                history = self.alpaca.get_portfolio_history(period='1D', timeframe='1Min')
                start_eq = float(history.equity[0]) if history and history.equity and history.equity[0] > 100 else current_eq
            
            stored_start = self.db.get_or_create_daily_stats(start_eq)
            raw_drawdown = 0.0
            if stored_start > 10.0:
                raw_drawdown = ((current_eq - stored_start) / stored_start) * 100
            
            self.status["equity"]["current"] = round(current_eq, 2)
            self.status["equity"]["pnl_pct"] = round(max(min(raw_drawdown, 10.0), -15.0), 2)
            self.status["market"] = "OPEN" if clock.is_open else "CLOSED"
            self.status["positions"]["live"] = len(self.alpaca.list_positions())
            
            # --- BRD 1.1 SYNC ---
            recent = self.db.get_recent_decisions(5)
            self.status["decisions"]["recent"] = recent
            # Correction : trades.last
            self.status["trades"]["last"] = recent[0] if recent else None
            # Correction : Stats Daily
            self.status["shadow"]["stats"] = self.db.get_daily_shadow_performance()
            
            return raw_drawdown
        except Exception as e:
            logger.error(f"Sync Error: {e}")
            return 0.0

    async def manage_shadow_trades(self):
        open_shadows = self.db.get_open_shadow_trades()
        self.status["positions"]["shadow_open"] = len(open_shadows)
        
        detailed_shadows = []
        for trade in open_shadows:
            try:
                bars = self.alpaca.get_bars(trade['symbol'], TimeFrame.Minute, limit=1)
                if not bars: continue
                price = bars[0].c
                logger.info(
                    f"SHADOW_CHECK | {trade['symbol']} | "
                    f"price={round(price,2)} | "
                    f"tp={trade['tp_price']} | sl={trade['sl_price']}"
                )
                
                entry_dt = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                duration = int((datetime.now() - entry_dt).total_seconds() / 60)
                
                detailed_shadows.append({
                    "symbol": trade['symbol'],
                    "entry_price": trade['entry_price'],
                    "tp": trade['tp_price'],
                    "sl": trade['sl_price'],
                    "confidence": trade['confidence'],
                    "duration_min": duration
                })

                if price >= trade['tp_price']: self.db.close_trade(trade['id'], price, "TP")
                elif price <= trade['sl_price']: self.db.close_trade(trade['id'], price, "SL")
            except: pass
        
        self.status["shadow"]["open"] = detailed_shadows

    async def run_trading_logic(self):
        spy = self.alpaca.get_bars("SPY", TimeFrame.Minute, limit=15)
        if len(spy) >= 15:
            vol = ((max([b.h for b in spy]) - min([b.l for b in spy])) / min([b.l for b in spy])) * 100
            self.status["safety"]["market_stress"] = vol > CONFIG["MARKET_STRESS_THRESHOLD"]
            if self.status["safety"]["market_stress"]: return

        picks = await self.fetch_ai_picks()
        equity = float(self.alpaca.get_account().equity)
        # --- LOG IA ---
        if not picks.get("picks"):
            logger.info("AI_DECISION | no viable picks returned")
        else:
            logger.info(f"AI_DECISION | {len(picks['picks'])} candidates received")
        
        executed_in_scan = 0
        for p in picks.get("picks", []):
            symbol = p['symbol'].upper()
            conf = p['confidence']
            if conf < 80:
                logger.info(f"REJECT | {symbol} | confidence={conf} (<80)")
                continue

            
            # Insight rejection
            if conf < 88:
                self.status["insights"]["last_rejection"] = {"symbol": symbol, "confidence": conf, "required": 88, "reason": "CONFIDENCE_THRESHOLD"}
            elif len(self.alpaca.list_positions()) >= CONFIG["MAX_OPEN_POSITIONS"]:
                self.status["insights"]["last_rejection"] = {"symbol": symbol, "confidence": conf, "required": 88, "reason": "MAX_POSITIONS"}

            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                logger.info(f"REJECT | {symbol} | cooldown active")
                continue

            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
                if not bars: continue
                price = bars[0].c
                qty = round((equity * CONFIG["RISK_PER_TRADE_PCT"]) / price, 2)
                tp, sl = round(price * CONFIG['TP_PCT'], 2), round(price * CONFIG['SL_PCT'], 2)

                if conf >= 88 and executed_in_scan < 1 and len(self.alpaca.list_positions()) < CONFIG["MAX_OPEN_POSITIONS"]:
                    self.alpaca.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc',
                                             order_class='bracket', take_profit={'limit_price': tp}, stop_loss={'stop_price': sl})
                    self.db.log_trade_start(symbol, qty, price, conf, p['reason'], "LIVE", tp, sl)
                    self.last_trade_per_symbol[symbol] = datetime.now()
                    executed_in_scan += 1
                else:
                    self.db.log_trade_start(symbol, qty, price, conf, p['reason'], "SHADOW", tp, sl)
                    self.last_trade_per_symbol[symbol] = datetime.now()
            except: pass

    async def fetch_ai_picks(self):
        s = await self.get_session()
        try:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": CONFIG["AI_MODEL"], "messages": [{"role": "user", "content": "Return 5 US tickers with momentum in JSON: {'picks': [{'symbol': '...', 'confidence': 92, 'reason': '...'}]}"}]}) as r:
                d = await r.json()
                match = re.search(r"\{.*\}", d['choices'][0]['message']['content'], re.DOTALL)
                return json.loads(match.group(0)) if match else {"picks": []}
        except: return {"picks": []}

    async def main_loop(self):
        while True:
            try:
                is_halted, _ = self.db.get_halt_state()
                raw_drawdown = await self.sync_daily_equity()
                
                if not self.equity_stabilized:
                    await asyncio.sleep(5)
                    self.equity_stabilized = True
                    continue

                if is_halted:
                    self.status["state"] = "HALTED"
                    await asyncio.sleep(30)
                    continue

                if self.status["market"] == "OPEN" and raw_drawdown <= -2.0: # Hardcoded fallback
                    await self.emergency_halt(f"DRAWDOWN ({round(raw_drawdown,2)}%)")
                    continue
                
                if self.db.get_consecutive_losses() >= 2:
                    await self.emergency_halt("MAX_CONSECUTIVE_SL")
                    continue

                await self.manage_shadow_trades()

                if self.status["market"] == "OPEN":
                    self.status["state"] = "SCANNING"
                    await self.run_trading_logic()
                else:
                    self.status["state"] = "SLEEPING"
                # --- HEARTBEAT OBSERVABILITÉ ---
                logger.info(
                    f"HEARTBEAT | state={self.status['state']} | "
                    f"market={self.status['market']} | "
                    f"live={self.status['positions']['live']} | "
                    f"shadow={self.status['positions']['shadow_open']} | "
                    f"equity={self.status['equity']['current']}"
                )
                await asyncio.sleep(600)
            except Exception as e:
                logger.error(f"Global Loop Error: {e}")
                await asyncio.sleep(60)

# --- API ---
async def api_status(request): return web.json_response(request.app['titan'].status)
async def api_resume(request):
    request.app['titan'].db.set_halt_state(False)
    request.app['titan'].status["state"] = "INIT"
    return web.json_response({"status": "resumed"})

@web.middleware
async def auth_middleware(request, handler):
    if request.method == "OPTIONS": return await handler(request)
    if request.headers.get("Authorization") != f"Bearer {API_TOKEN}": return web.json_response({"error": "Auth"}, status=401)
    return await handler(request)

async def main():
    titan = TitanEngine()
    app = web.Application(middlewares=[auth_middleware])
    app['titan'] = titan
    app.router.add_get('/status', api_status)
    app.router.add_post('/resume', api_resume)
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=False, expose_headers="*", allow_headers=["Authorization", "Content-Type"], allow_methods=["GET", "POST", "OPTIONS"])})
    for r in list(app.router.routes()): cors.add(r)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', CONFIG["PORT"]).start()
    logger.info(f"Titan Sentinel {CONFIG['VERSION']} Online (BRD Final Compliant).")
    await titan.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
