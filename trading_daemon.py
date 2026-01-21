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
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")

CONFIG = {
    "VERSION": "7.9.12-OmniAudit-Plus",
    "PORT": 8080,
    "DB_PATH": "titan_v7_omni.db",
    "MAX_OPEN_POSITIONS": 3,
    "DOLLAR_RISK_PER_TRADE": 50.0,
    "LIVE_THRESHOLD": 82,
    "MARKET_STRESS_THRESHOLD": 1.5,
    "COOLDOWN_PER_SYMBOL_MIN": 15,
    "TP_PCT": 1.025,
    "SL_PCT": 0.985,
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "deepseek/deepseek-v3.2",
    "SCAN_INTERVAL": 60 
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("titan_omni.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Titan-Omni")

# --- UTILITAIRES DE PARSING ---
def clean_deepseek_json(raw_text: str):
    """
    Nettoyage robuste des sorties DeepSeek (suppression <think> et markdown).
    """
    if not raw_text: return None
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

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
                    action TEXT, filter_rejected TEXT,
                    ai_raw TEXT
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS daily_stats (day DATE PRIMARY KEY, start_equity REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('is_halted', '0')")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('halt_reason', '')")
            conn.commit()

    def get_or_create_daily_stats(self, current_equity):
        """Gère l'équité de départ pour le calcul du PnL journalier."""
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT start_equity FROM daily_stats WHERE day = ?", (today,)).fetchone()
            if row:
                return row[0]
            else:
                conn.execute("INSERT INTO daily_stats (day, start_equity) VALUES (?, ?)", (today, current_equity))
                conn.commit()
                return current_equity

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

    def log_decision(self, symbol, conf, reason, action, filter_rejected="", ai_raw=""):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO ai_decision_log (symbol, confidence, reason, action, filter_rejected, ai_raw) VALUES (?,?,?,?,?,?)",
                (symbol, conf, reason, action, filter_rejected, ai_raw)
            )
            conn.commit()
            return cursor.lastrowid

    def log_trade(self, symbol, qty, price, conf, thesis, mode, tp, sl, dec_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status, decision_id) 
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN', dec_id))
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
            last_dec = conn.execute("SELECT symbol, action, reason FROM ai_decision_log ORDER BY id DESC LIMIT 1").fetchone()
            res['last_action'] = f"{last_dec[0]}: {last_dec[1]}" if last_dec else "N/A"
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
            "omni": {"decisions_today": 0, "shadow_winrate": 0, "last_action": "N/A"}
        }

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def sync_data(self):
        try:
            acc = self.alpaca.get_account()
            eq = float(acc.equity)
            
            # --- RÉCUPÉRATION PNL JOURNALIER ---
            start_eq = self.db.get_or_create_daily_stats(eq)
            pnl_pct = ((eq - start_eq) / start_eq * 100) if start_eq > 0 else 0.0
            
            clock = self.alpaca.get_clock()
            stats = self.db.get_stats()
            self.status.update({
                "market": "OPEN" if clock.is_open else "CLOSED",
                "equity": {"current": round(eq, 2), "pnl_pct": round(pnl_pct, 2)},
                "positions": {"live": len(self.alpaca.list_positions())},
                "safety": {"consecutive_sl": stats['consecutive_sl']},
                "omni": stats
            })
        except Exception as e: logger.error(f"Sync error: {e}")

    async def fetch_ai_picks(self):
        s = await self.get_session()
        prompt = (
            "You are a quantitative trading assistant. Analyze current US market trends. "
            "Return ONLY a JSON object with this structure: "
            "{'picks': [{'symbol': 'TICKER', 'confidence': 95, 'reason': 'short thesis'}]}. "
            "If no clear opportunities, return {'picks': []}."
        )
        try:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={
                    "model": CONFIG["AI_MODEL"], 
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30) as r:
                
                resp_json = await r.json()
                if 'choices' not in resp_json:
                    return {"picks": [], "raw": str(resp_json), "error": "API_ERROR"}
                
                raw_content = resp_json['choices'][0]['message']['content']
                parsed = clean_deepseek_json(raw_content)
                
                if parsed and "picks" in parsed:
                    return {"picks": parsed["picks"], "raw": raw_content, "error": None}
                else:
                    return {"picks": [], "raw": raw_content, "error": "PARSE_ERROR"}
        except Exception as e:
            return {"picks": [], "raw": str(e), "error": "CONN_ERROR"}

    async def run_logic(self):
        spy_vol = 0.0
        try:
            spy = self.alpaca.get_bars("SPY", TimeFrame.Minute, limit=15)
            if len(spy) >= 15:
                spy_vol = ((max([b.h for b in spy]) - min([b.l for b in spy])) / min([b.l for b in spy])) * 100
                self.status["safety"]["market_stress"] = spy_vol > CONFIG["MARKET_STRESS_THRESHOLD"]
        except Exception as e: logger.error(f"Vol check error: {e}")
        
        ai_data = await self.fetch_ai_picks()
        picks = ai_data.get("picks", [])
        raw_text = ai_data.get("raw", "")
        ai_err = ai_data.get("error")

        # --- HEARTBEAT AMÉLIORÉ ---
        if not picks:
            reason = "NO_PICKS" if not ai_err else ai_err
            self.db.log_decision("SYSTEM", 0, f"Heartbeat (Vol: {round(spy_vol,2)}%). Reason: {reason}", "IDLE", ai_raw=raw_text)
            return

        for p in picks:
            symbol = p.get('symbol', '').upper()
            conf = p.get('confidence', 0)
            thesis = p.get('reason', 'N/A')
            if not symbol: continue

            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                self.db.log_decision(symbol, conf, thesis, "SKIP", "COOLDOWN", ai_raw=raw_text)
                continue

            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
                if not bars: continue
                
                entry = bars[0].c
                tp, sl = round(entry * CONFIG["TP_PCT"], 2), round(entry * CONFIG["SL_PCT"], 2)
                diff = abs(entry - sl)
                qty = math.floor(CONFIG["DOLLAR_RISK_PER_TRADE"] / diff) if diff > 0 else 0

                if qty < 1:
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "RISK_SIZE", ai_raw=raw_text)
                    continue

                can_live = (conf >= CONFIG["LIVE_THRESHOLD"] and self.status["positions"]["live"] < CONFIG["MAX_OPEN_POSITIONS"] and not self.status["safety"]["market_stress"])

                if can_live:
                    dec_id = self.db.log_decision(symbol, conf, thesis, "LIVE", ai_raw=raw_text)
                    self.alpaca.submit_order(
                        symbol=symbol, qty=qty, side='buy', type='market', 
                        time_in_force='gtc', order_class='bracket', 
                        take_profit={'limit_price': tp}, stop_loss={'stop_price': sl}
                    )
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "LIVE", tp, sl, dec_id)
                else:
                    rej = "STRESS" if self.status["safety"]["market_stress"] else "CONF/LIMIT"
                    dec_id = self.db.log_decision(symbol, conf, thesis, "SHADOW", rej, ai_raw=raw_text)
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "SHADOW", tp, sl, dec_id)
                
                self.last_trade_per_symbol[symbol] = datetime.now()
            except Exception as e: logger.error(f"Trade error {symbol}: {e}")

    async def main_loop(self):
        while True:
            try:
                await self.sync_data()
                is_halted, _ = self.db.get_halt_state()
                if not is_halted:
                    if self.status["market"] == "OPEN":
                        self.status["state"] = "SCANNING"
                        await self.run_logic()
                    else: 
                        self.status["state"] = "IDLE"
                else: 
                    self.status["state"] = "HALTED"
            except Exception as e:
                logger.error(f"Loop crash: {e}")
            await asyncio.sleep(CONFIG["SCAN_INTERVAL"])

# --- API ENDPOINTS ---
async def api_status(request): 
    return web.json_response(request.app['titan'].status)

async def api_decisions(request):
    limit = int(request.query.get("limit", 20))
    with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM ai_decision_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return web.json_response([dict(r) for r in rows])

async def api_resume(request):
    request.app['titan'].db.set_halt_state(False)
    return web.json_response({"status": "resumed"})

@web.middleware
async def auth_middleware(request, handler):
    if request.method == "OPTIONS": return await handler(request)
    if request.headers.get("Authorization") != f"Bearer {API_TOKEN}":
        return web.json_response({"error": "Auth"}, status=401)
    return await handler(request)

async def main():
    titan = TitanEngine()
    app = web.Application(middlewares=[auth_middleware])
    app['titan'] = titan
    app.router.add_get('/status', api_status)
    app.router.add_get('/decisions', api_decisions)
    app.router.add_post('/resume', api_resume)
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_headers=["Authorization", "Content-Type"], allow_methods=["GET", "POST", "OPTIONS"])})
    for r in list(app.router.routes()): cors.add(r)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', CONFIG["PORT"]).start()
    logger.info(f"Titan-Omni Plus v7.9.12 Ready.")
    await titan.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
