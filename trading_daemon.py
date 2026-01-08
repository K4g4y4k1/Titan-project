import asyncio
import sqlite3
import logging
import logging.handlers
import os
import json
import uuid
import threading
import pandas as pd
import io
import yfinance as yf
import hmac
import aiohttp
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import alpaca_trade_api as tradeapi

# --- CONFIGURATION v5.7.3 "GEL TOTAL" ---
DB_PATH = "titan_prod_v5.db"
LOG_FILE = "titan_system.log"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN", "c3stun3s4cr334v3ntur3")
ENV_MODE = os.getenv("ENV_MODE", "PAPER")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_KEY") 
OR_KEY = os.getenv("OPENROUTER_API_KEY")

AI_MODEL = "x-ai/grok-4.1-fast" 

GOUVERNANCE = {
    "ENV_MODE": ENV_MODE,
    "MAX_SECTOR_EXPOSURE_PCT": 0.20,
    "MAX_POSITION_SIZE_PCT": 0.08, 
    "MAX_TRADES_PER_DAY": 8,
    "MAX_DAILY_DRAWDOWN_PCT": 0.015,
    "PEAD_WINDOW_DAYS": 21,
    "MODES": {
        "EXPLOITATION": { "MIN_SCORE": 88, "BASE_RISK": 0.008 },
        "EXPLORATION": { "MIN_SCORE": 75, "BASE_RISK": 0.002 }
    },
    "BASE_TP_PCT": 0.05,
    "BASE_SL_PCT": 0.025,
    "BLACKLIST": ["GME", "AMC", "BBBY", "DJT", "SPCE", "FFIE", "LUMN"]
}

SYSTEM_STATE = {
    "status": "initializing",
    "equity": 0.0,
    "engine_version": "5.7.3 (Final)",
    "trades_today": 0,
    "drawdown_pct": 0.0,
    "health": {"db": "connected", "last_cycle": None},
    "positions": []
}

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanGrok")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
            fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
            self.logger.addHandler(fh)

    def log(self, event_type, level="info", **kwargs):
        payload = {"ts": datetime.now().isoformat(), "lvl": level.upper(), "ev": event_type, **kwargs}
        self.logger.info(json.dumps(payload))

SLOG = StructuredLogger()

class SecureMetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
    def do_GET(self):
        auth = self.headers.get('Authorization', "")
        if not hmac.compare_digest(auth, f"Bearer {DASHBOARD_TOKEN}"):
            self._set_headers(401); return
        self._set_headers(200)
        self.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode())

class TitanEngine:
    def __init__(self):
        base_url = "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets"
        self.alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            # Ajout de reasoning_category à la table
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, 
                qty REAL, entry_price REAL, exit_price REAL, status TEXT, 
                pnl REAL, mode TEXT, consensus REAL, sector TEXT, 
                reasoning_category TEXT, ai_reason TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            # Migration ultra-légère si colonne manquante
            try: conn.execute("ALTER TABLE trades ADD COLUMN reasoning_category TEXT")
            except: pass

    async def get_ai_score(self, session, sym, price):
        now_str = datetime.now().strftime('%Y-%m-%d')
        prompt = (f"Today: {now_str}. Analyze PEAD for {sym} at {price}$. "
                  f"Rule 1: If earnings > {GOUVERNANCE['PEAD_WINDOW_DAYS']} days ago, score = 0. "
                  f"Return JSON: {{\"score\": 0-100, \"category\": \"PEAD/REVERSAL/NOISE\", \"reason\": \"...\"}}")
        
        headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
        payload = {"model": AI_MODEL, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
        
        try:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    content = json.loads(res['choices'][0]['message']['content'])
                    return float(content.get('score', 0)), content.get('category', 'UNKNOWN'), content.get('reason', 'N/A')
        except: pass
        return 0, "UNKNOWN", "AI_FAILURE"

    async def sync_reconcile_full(self, positions):
        alpaca_symbols = [p.symbol for p in positions]
        open_orders = self.alpaca.list_orders(status='open')
        pending_symbols = [o.symbol for o in open_orders]
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            db_open_trades = cursor.execute("SELECT client_id, symbol, qty, entry_price FROM trades WHERE status='OPEN'").fetchall()
            
            for cid, sym, qty, entry in db_open_trades:
                if sym not in alpaca_symbols and sym not in pending_symbols:
                    activities = self.alpaca.get_activities(activity_types='FILL')
                    exit_price = None
                    for act in sorted(activities, key=lambda x: x.created_at, reverse=True):
                        if act.symbol == sym and act.side.startswith('sell'):
                            exit_price = float(act.price)
                            break
                    
                    if exit_price:
                        pnl = (exit_price - entry) * qty
                        cursor.execute("UPDATE trades SET status='CLOSED', exit_price=?, pnl=? WHERE client_id=?", (exit_price, pnl, cid))
                    else:
                        cursor.execute("UPDATE trades SET status='ORPHAN', pnl=NULL WHERE client_id=?", (cid,))
            conn.commit()

        SYSTEM_STATE["positions"] = [{"s": p.symbol, "q": p.qty, "upnl": float(p.unrealized_pl)} for p in positions]
        with sqlite3.connect(DB_PATH) as conn:
            db_active = [r[0] for r in conn.execute("SELECT symbol FROM trades WHERE status='OPEN'").fetchall()]
        
        return set(alpaca_symbols + pending_symbols + db_active)

    def check_discipline_veto(self, sym):
        with sqlite3.connect(DB_PATH) as conn:
            last = conn.execute("SELECT status, pnl, timestamp FROM trades WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (sym,)).fetchone()
            if last:
                status, pnl, ts_str = last
                if status == 'ORPHAN' or (status == 'CLOSED' and pnl is None):
                    return True
                if pnl is not None and pnl < 0:
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    if datetime.now() - ts < timedelta(hours=24):
                        return True
        return False

    async def run_cycle(self):
        try:
            SYSTEM_STATE["health"]["last_cycle"] = datetime.now().isoformat()
            acc = self.alpaca.get_account()
            SYSTEM_STATE["equity"] = float(acc.equity)
            
            # CORRECTIF 1 : Recalcul trades_today depuis la DB (Audit 3.3)
            with sqlite3.connect(DB_PATH) as conn:
                count = conn.execute("SELECT COUNT(*) FROM trades WHERE date(timestamp) = date('now')").fetchone()[0]
                SYSTEM_STATE["trades_today"] = count

            forbidden_symbols = await self.sync_reconcile_full(self.alpaca.list_positions())

            daily_loss_pct = abs(float(acc.equity) - float(acc.last_equity)) / float(acc.last_equity)
            if daily_loss_pct > GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"] and float(acc.equity) < float(acc.last_equity):
                SYSTEM_STATE["status"] = "HALTED_DRAWDOWN"; return

            if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
                SYSTEM_STATE["status"] = "standby"; return

            SYSTEM_STATE["status"] = "scanning"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={AV_KEY}") as resp:
                    if resp.status != 200: return
                    df = pd.read_csv(io.BytesIO(await resp.read()))
                    candidates = df[df['reportDate'] == datetime.now().strftime('%Y-%m-%d')]

                for _, row in candidates.iterrows():
                    sym = row.get('symbol')
                    if not sym or sym in forbidden_symbols or sym in GOUVERNANCE["BLACKLIST"]: continue
                    if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break
                    if self.check_discipline_veto(sym): continue

                    try:
                        price = float(self.alpaca.get_latest_trade(sym).price)
                        # CORRECTIF 2 : Capture de la catégorie de raisonnement (Audit 4.1)
                        score, category, reason = await self.get_ai_score(session, sym, price)
                        
                        mode = "EXPLOITATION" if score >= GOUVERNANCE["MODES"]["EXPLOITATION"]["MIN_SCORE"] else "EXPLORATION" if score >= GOUVERNANCE["MODES"]["EXPLORATION"]["MIN_SCORE"] else None
                        
                        if mode:
                            risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"]
                            qty = int((SYSTEM_STATE["equity"] * risk) / (price * GOUVERNANCE["BASE_SL_PCT"]))
                            
                            if qty > 0:
                                cid = f"tx_{uuid.uuid4().hex[:8]}"
                                self.alpaca.submit_order(
                                    symbol=sym, qty=qty, side='buy', type='limit',
                                    limit_price=round(price * 1.002, 2),
                                    time_in_force='gtc', order_class='bracket',
                                    take_profit={'limit_price': round(price * (1+GOUVERNANCE["BASE_TP_PCT"]), 2)},
                                    stop_loss={'stop_price': round(price * (1-GOUVERNANCE["BASE_SL_PCT"]), 2)},
                                    client_order_id=cid
                                )
                                with sqlite3.connect(DB_PATH) as conn:
                                    conn.execute("""INSERT INTO trades 
                                        (client_id, symbol, qty, entry_price, status, mode, consensus, reasoning_category, ai_reason) 
                                        VALUES (?,?,?,?,?,?,?,?,?)""",
                                        (cid, sym, qty, price, 'OPEN', mode, score, category, reason))
                                SYSTEM_STATE["trades_today"] += 1
                                forbidden_symbols.add(sym)
                                SLOG.log("trade_sent", symbol=sym, cat=category, score=score)
                    except Exception as e:
                        SLOG.log("error_sym", symbol=sym, err=str(e))

        except Exception as e:
            SLOG.log("critical_crash", error=str(e))

async def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), SecureMetricsHandler).serve_forever, daemon=True).start()
    engine = TitanEngine()
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
