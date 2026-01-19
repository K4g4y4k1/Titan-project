import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import re
import math
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from aiohttp import web
import aiohttp_cors

# --- CONFIGURATION ---
load_dotenv()

API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN')
if not API_TOKEN:
    print("FATAL ERROR: TITAN_DASHBOARD_TOKEN must be defined in your environment.")
    exit(1)

OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")

CONFIG = {
    "VERSION": "7.9.9-OmniAccess",
    "PORT": 8080,
    "DB_PATH": "titan_v7_omni.db",
    "MAX_OPEN_POSITIONS": 3,
    "DOLLAR_RISK_PER_TRADE": 50.0,
    "LIVE_THRESHOLD": 82,
    "MARKET_STRESS_THRESHOLD": 0.75,
    "COOLDOWN_PER_SYMBOL_MIN": 15,
    "TP_PCT": 1.025,
    "SL_PCT": 0.985,
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "google/gemini-2.0-flash-exp:free"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("titan_omni.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Titan-Omni")

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
                    tp_price REAL, sl_price REAL, status TEXT DEFAULT 'OPEN',
                    decision_id INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT, confidence INTEGER, reason TEXT,
                    action TEXT, filter_rejected TEXT
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS daily_stats (day DATE PRIMARY KEY, start_equity REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('is_halted', '0')")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('halt_reason', '')")
            conn.commit()

    def get_halt_state(self):
        with sqlite3.connect(self.db_path) as conn:
            is_halted = conn.execute("SELECT value FROM system_state WHERE key = 'is_halted'").fetchone()[0]
            reason = conn.execute("SELECT value FROM system_state WHERE key = 'halt_reason'").fetchone()[0]
            return (is_halted == '1', reason)

    def set_halt_state(self, halted, reason=""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'is_halted'", ('1' if halted else '0',))
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'halt_reason'", (reason,))
            conn.commit()

    def get_or_create_daily_stats(self, current_equity):
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT start_equity FROM daily_stats WHERE day = ?", (today,)).fetchone()
            if not res:
                conn.execute("INSERT INTO daily_stats (day, start_equity) VALUES (?, ?)", (today, current_equity))
                return current_equity
            return res[0]

    def log_decision(self, symbol, conf, reason, action, filter_rejected=""):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO ai_decision_log (symbol, confidence, reason, action, filter_rejected) VALUES (?,?,?,?,?)",
                (symbol, conf, reason, action, filter_rejected)
            )
            return cursor.lastrowid

    def log_trade(self, symbol, qty, price, conf, thesis, mode, tp, sl, dec_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status, decision_id) 
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN', dec_id))
            conn.commit()

    def get_open_shadow_trades(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM trades WHERE mode='SHADOW' AND status='OPEN'").fetchall()]

    def close_trade(self, trade_id, exit_price, result):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE trades SET exit_price=?, exit_time=CURRENT_TIMESTAMP, result=?, status='CLOSED' WHERE id=?", 
                        (exit_price, result, trade_id))
            conn.commit()

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            res = {'consecutive_sl': 0}
            trades = conn.execute("SELECT result FROM trades WHERE mode='LIVE' AND status='CLOSED' ORDER BY id DESC LIMIT 5").fetchall()
            for (r,) in trades:
                if r == "SL": res['consecutive_sl'] += 1
                else: break
            res['decisions_today'] = conn.execute("SELECT COUNT(*) FROM ai_decision_log WHERE date(timestamp) = date('now')").fetchone()[0]
            shadows = conn.execute("SELECT result FROM trades WHERE mode='SHADOW' AND status='CLOSED' AND date(exit_time) = date('now')").fetchall()
            res['shadow_winrate'] = round((len([r for (r,) in shadows if r == 'TP']) / len(shadows) * 100), 2) if shadows else 0
            return res

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
        
        is_halted, reason = self.db.get_halt_state()
        self.status = {
            "version": CONFIG["VERSION"],
            "state": "HALTED" if is_halted else "INIT",
            "market": "CLOSED",
            "halt_reason": reason,
            "equity": {"current": 0.0, "pnl_pct": 0.0},
            "safety": {"consecutive_sl": 0, "market_stress": False},
            "positions": {"live": 0, "shadow_open": 0},
            "omni": {"decisions_today": 0, "shadow_winrate": 0}
        }

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def calculate_risk_qty(self, entry, sl):
        risk_per_share = abs(entry - sl)
        return math.floor(CONFIG["DOLLAR_RISK_PER_TRADE"] / risk_per_share) if risk_per_share > 0 else 0

    async def sync_data(self):
        try:
            acc = self.alpaca.get_account()
            eq = float(acc.equity)
            clock = self.alpaca.get_clock()
            start_eq = self.db.get_or_create_daily_stats(eq)
            pnl = ((eq - start_eq) / start_eq * 100) if start_eq > 0 else 0
            stats = self.db.get_stats()
            self.status.update({
                "market": "OPEN" if clock.is_open else "CLOSED",
                "equity": {"current": round(eq, 2), "pnl_pct": round(pnl, 2)},
                "positions": {"live": len(self.alpaca.list_positions())},
                "safety": {"consecutive_sl": stats['consecutive_sl']},
                "omni": {"decisions_today": stats['decisions_today'], "shadow_winrate": stats['shadow_winrate']}
            })
            return pnl
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return 0.0

    async def manage_shadows(self):
        shadows = self.db.get_open_shadow_trades()
        self.status["positions"]["shadow_open"] = len(shadows)
        for s in shadows:
            try:
                bars = self.alpaca.get_bars(s['symbol'], TimeFrame.Minute, limit=1)
                if not bars: continue
                p = bars[0].c
                if p >= s['tp_price']: self.db.close_trade(s['id'], p, "TP")
                elif p <= s['sl_price']: self.db.close_trade(s['id'], p, "SL")
            except: pass

    async def run_logic(self):
        spy = self.alpaca.get_bars("SPY", TimeFrame.Minute, limit=15)
        if len(spy) >= 15:
            vol = ((max([b.h for b in spy]) - min([b.l for b in spy])) / min([b.l for b in spy])) * 100
            self.status["safety"]["market_stress"] = vol > CONFIG["MARKET_STRESS_THRESHOLD"]
            if self.status["safety"]["market_stress"]: return

        picks = await self.fetch_ai_picks()
        for p in picks.get("picks", []):
            symbol, conf, thesis = p['symbol'].upper(), p['confidence'], p['reason']
            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                self.db.log_decision(symbol, conf, thesis, "SKIP", "COOLDOWN")
                continue
            if conf < 80:
                self.db.log_decision(symbol, conf, thesis, "SKIP", "LOW_CONFIDENCE")
                continue
            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
                if not bars: continue
                entry = bars[0].c
                tp, sl = round(entry * CONFIG["TP_PCT"], 2), round(entry * CONFIG["SL_PCT"], 2)
                qty = self.calculate_risk_qty(entry, sl)
                if qty < 1:
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "RISK_SIZE_LOW")
                    continue
                if conf >= CONFIG["LIVE_THRESHOLD"] and self.status["positions"]["live"] < CONFIG["MAX_OPEN_POSITIONS"]:
                    dec_id = self.db.log_decision(symbol, conf, thesis, "LIVE")
                    self.alpaca.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc',
                                             order_class='bracket', take_profit={'limit_price': tp}, stop_loss={'stop_price': sl})
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "LIVE", tp, sl, dec_id)
                    self.last_trade_per_symbol[symbol] = datetime.now()
                else:
                    reason_s = "THRESHOLD" if conf < CONFIG["LIVE_THRESHOLD"] else "MAX_POSITIONS"
                    dec_id = self.db.log_decision(symbol, conf, thesis, "SHADOW", reason_s)
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "SHADOW", tp, sl, dec_id)
                    self.last_trade_per_symbol[symbol] = datetime.now()
            except Exception as e:
                logger.error(f"Execution Error {symbol}: {e}")

    async def fetch_ai_picks(self):
        s = await self.get_session()
        try:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": CONFIG["AI_MODEL"], "messages": [{"role": "user", "content": "JSON 5 momentum tickers: {'picks': [{'symbol': '...', 'confidence': 92, 'reason': '...'}]}"}]},
                timeout=30) as r:
                d = await r.json()
                match = re.search(r"\{.*\}", d['choices'][0]['message']['content'], re.DOTALL)
                return json.loads(match.group(0)) if match else {"picks": []}
        except: return {"picks": []}

    async def main_loop(self):
        while True:
            try:
                pnl = await self.sync_data()
                is_halted, _ = self.db.get_halt_state()
                if is_halted:
                    self.status["state"] = "HALTED"
                elif pnl <= -3.0 or self.status["safety"]["consecutive_sl"] >= 3:
                    await self.emergency_halt("CRITICAL_PROTECTION")
                else:
                    await self.manage_shadows()
                    if self.status["market"] == "OPEN":
                        self.status["state"] = "SCANNING"
                        await self.run_logic()
                    else:
                        self.status["state"] = "IDLE"
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(60)

    async def emergency_halt(self, reason):
        self.db.set_halt_state(True, reason)
        self.status["state"] = "HALTED"
        try:
            for p in self.alpaca.list_positions():
                self.alpaca.submit_order(symbol=p.symbol, qty=p.qty, side='sell', type='market', time_in_force='gtc')
        except: pass

# --- API ---
async def api_status(request): 
    return web.json_response(request.app['titan'].status)

async def api_resume(request):
    request.app['titan'].db.set_halt_state(False)
    return web.json_response({"status": "resumed"})

@web.middleware
async def auth_middleware(request, handler):
    if request.method == "OPTIONS": return await handler(request)
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_TOKEN}":
        return web.json_response({"error": "Auth"}, status=401)
    return await handler(request)

async def main():
    titan = TitanEngine()
    app = web.Application(middlewares=[auth_middleware])
    app['titan'] = titan
    # Routes
    app.router.add_get('/', api_status)      # Racine pointe vers status
    app.router.add_get('/status', api_status)
    app.router.add_post('/resume', api_resume)
    
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=False, expose_headers="*", allow_headers=["Authorization", "Content-Type"], allow_methods=["GET", "POST", "OPTIONS"])})
    for r in list(app.router.routes()): cors.add(r)
    
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', CONFIG["PORT"]).start()
    logger.info(f"Titan {CONFIG['VERSION']} Online on port {CONFIG['PORT']}.")
    await titan.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
