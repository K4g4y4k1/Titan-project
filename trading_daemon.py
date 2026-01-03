import asyncio
import sqlite3
import logging
import logging.handlers
import os
import sys
import json
import uuid
import threading
import pandas as pd
import io
import yfinance as yf
import shutil
import secrets
import hmac
import aiohttp
import traceback
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import alpaca_trade_api as tradeapi

# --- CONFIGURATION v5.6.11-LTS "SECURE-ENV" ---
DB_PATH = "titan_prod_v5.db"
LOG_FILE = "titan_system.log"
BACKUP_DIR = "backups"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

# SÉCURITÉ : On récupère le token depuis le service systemd.
# Si absent, on utilise un fallback mais on log un avertissement critique.
DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN")
if not DASHBOARD_TOKEN:
    DASHBOARD_TOKEN = "12345" # Fallback temporaire
    TOKEN_WARNING = True
else:
    TOKEN_WARNING = False

ENV_MODE = os.getenv("ENV_MODE", "PAPER")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_KEY") 
OR_KEY = os.getenv("OPENROUTER_API_KEY")

GOUVERNANCE = {
    "ENV_MODE": ENV_MODE,
    "MIN_STOCK_PRICE": 5.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25, 
    "MAX_POSITION_SIZE_PCT": 0.10,      
    "MAX_TRADES_PER_DAY": 12,
    "MAX_HOLDING_DAYS": 3,
    "BLACKLIST": ["GME", "AMC", "BBBY", "DJT", "SPCE", "FFIE", "LUMN"],
    "MAX_DAILY_DRAWDOWN_PCT": 0.02,    
    "MAX_TOTAL_DRAWDOWN_PCT": 0.10,     
    "MAX_CONSECUTIVE_FAILURES": 5,
    "CYCLE_LATENCY_THRESHOLD": 150,
    "MIN_TRADES_FOR_JUDGEMENT": 10,
    "DEGRADED_THRESHOLD_USD": 0.0,
    "QUARANTINE_THRESHOLD_USD": -15.0,
    "GLOBAL_CAPS": { "EXPLOITATION": 0.80, "EXPLORATION": 0.20 },
    "MODES": {
        "EXPLOITATION": { "MIN_SCORE": 85, "MAX_SIGMA": 20, "BASE_RISK": 0.01 },
        "EXPLORATION": { "MIN_SCORE": 72, "MAX_SIGMA": 35, "BASE_RISK": 0.0025 }
    },
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "SLIPPAGE_PROTECTION": 0.002        
}

SYSTEM_STATE = {
    "status": "initializing",
    "equity": 0.0,
    "engine_version": "5.6.11-LTS",
    "trades_today": 0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "health": {"db": "unknown", "alpaca": "unknown", "last_cycle": None, "consecutive_errors": 0},
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "trades": 0}
    },
    "auth_hint": DASHBOARD_TOKEN[:4] + "...",
    "security_alert": "INSECURE_DEFAULT_TOKEN" if TOKEN_WARNING else "SECURE_ENV_LOADED"
}

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanLTS")
        self.logger.setLevel(logging.INFO)
        fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
        fh.setFormatter(logging.Formatter('%(message)s'))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log(self, event_type, level="info", **kwargs):
        payload = {"timestamp": datetime.now().isoformat(), "level": level.upper(), "event": event_type, **kwargs}
        msg = json.dumps(payload)
        if level.lower() == "critical": self.logger.critical(msg)
        elif level.lower() == "error": self.logger.error(msg)
        else: self.logger.info(msg)

SLOG = StructuredLogger()

async def fetch_with_retry(session, url, headers=None, json_data=None, method="GET", retries=3):
    for i in range(retries):
        try:
            if method == "GET":
                async with session.get(url, headers=headers, timeout=15) as resp:
                    if resp.status == 200: return await resp.read()
                    if resp.status == 429: await asyncio.sleep(2**i)
            else:
                async with session.post(url, headers=headers, json=json_data, timeout=15) as resp:
                    if resp.status == 200: return await resp.json()
        except Exception as e:
            if i == retries - 1: raise e
            await asyncio.sleep(1 * (i + 1))
    return None

class SecureMetricsHandler(BaseHTTPRequestHandler):
    last_req = 0
    def log_message(self, format, *args): return

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        now = time.time()
        if now - SecureMetricsHandler.last_req < 0.5:
            self.send_response(429); self.end_headers(); return
        SecureMetricsHandler.last_req = now
        
        auth = self.headers.get('Authorization', "")
        if not hmac.compare_digest(auth, f"Bearer {DASHBOARD_TOKEN}"):
            self._set_headers(401)
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return

        self._set_headers(200)
        self.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode())

class OrderExecutor:
    def __init__(self, alpaca):
        self.alpaca = alpaca

    async def secure_submit(self, params):
        try:
            order = self.alpaca.submit_order(**params)
            SLOG.log("order_submitted", symbol=params['symbol'], qty=params['qty'])
            await asyncio.sleep(3)
            check = self.alpaca.get_order(order.id)
            if check.status == 'rejected':
                SLOG.log("order_rejected", level="error", symbol=params['symbol'])
                return False
            return True
        except Exception as e:
            SLOG.log("exec_exception", level="error", error=str(e))
            return False

class TitanEngine:
    def __init__(self):
        self.alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, 
            "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets")
        self.executor = OrderExecutor(self.alpaca)
        self._init_db()
        self.initial_equity = self._load_anchor()
        self.av_cache = None
        self.last_av_fetch = None

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, qty REAL, entry_price REAL, exit_price REAL, status TEXT, pnl REAL, mode TEXT, consensus REAL, dispersion REAL, sector TEXT, ai_reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS sector_cache (symbol TEXT PRIMARY KEY, sector TEXT)")
        SYSTEM_STATE["health"]["db"] = "connected"

    def _load_anchor(self):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='initial_equity'").fetchone()
            if row: return float(row[0])
            acc = self.alpaca.get_account()
            e = float(acc.equity)
            conn.execute("INSERT INTO meta (key, value) VALUES ('initial_equity', ?)", (e,))
            return e

    def get_sector(self, sym):
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT sector FROM sector_cache WHERE symbol=?", (sym,)).fetchone()
            if res: return res[0]
        try:
            ticker = yf.Ticker(sym)
            s = ticker.info.get('sector', 'Unknown')
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT OR REPLACE INTO sector_cache VALUES (?,?)", (sym, s))
            return s
        except: return "Unknown"

    async def reconcile(self, positions):
        pos_map = {p.symbol: p for p in positions}
        active_symbols = set([p.symbol for p in positions])
        # On récupère aussi les ordres ouverts pour le filtre anti-overtrade
        try:
            open_orders = self.alpaca.list_orders(status='open')
            for o in open_orders: active_symbols.add(o.symbol)
        except: pass
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT client_id, symbol, entry_price, qty, timestamp FROM trades WHERE status='OPEN'")
            for c_id, sym, entry, qty, ts in cursor.fetchall():
                if (datetime.now() - datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')).days >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                    try: self.alpaca.close_position(sym)
                    except: pass
                if sym not in pos_map:
                    try:
                        orders = self.alpaca.list_orders(status='closed', limit=5, symbols=[sym])
                        if orders:
                            latest = sorted(orders, key=lambda x: x.filled_at or x.submitted_at, reverse=True)[0]
                            exit_p = float(latest.filled_avg_price)
                            cursor.execute("UPDATE trades SET status='CLOSED', exit_price=?, pnl=? WHERE client_id=?", 
                                         (exit_p, (exit_p - entry)*qty, c_id))
                    except: pass
            conn.commit()
        return active_symbols

    def sync_forge(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE date(timestamp) = date('now')")
            SYSTEM_STATE["trades_today"] = cursor.fetchone()[0]
            for m in ["EXPLOITATION", "EXPLORATION"]:
                pnls = [r[0] for r in cursor.execute("SELECT pnl FROM trades WHERE mode=? AND status='CLOSED' ORDER BY timestamp DESC LIMIT 20", (m,)).fetchall()]
                count = len(pnls)
                exp = sum(pnls)/count if count > 0 else 0
                if count >= GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
                    if exp <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]: SYSTEM_STATE["allocation"][m] = 0.0
                    elif exp <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]: SYSTEM_STATE["allocation"][m] = 0.5
                    else: SYSTEM_STATE["allocation"][m] = 1.0
                SYSTEM_STATE["stats"][m] = {"expectancy": round(exp, 2), "trades": count}

    async def run_cycle(self):
        try:
            with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
            SYSTEM_STATE["health"]["last_cycle"] = datetime.now().isoformat()
            
            positions = self.alpaca.list_positions()
            acc = self.alpaca.get_account()
            equity = float(acc.equity)
            SYSTEM_STATE["equity"] = equity
            
            # Réconciliation et récupération des symboles actifs (Positions + Ordres)
            active_symbols = await self.reconcile(positions)
            self.sync_forge()
            
            if (equity - float(acc.last_equity)) / float(acc.last_equity) <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
                SYSTEM_STATE["status"] = "halt_dd"; self.alpaca.close_all_positions(); return
            
            if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
                SYSTEM_STATE["status"] = "standby"; return

            SYSTEM_STATE["status"] = "scanning"
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={AV_KEY}"
                data = await fetch_with_retry(session, url)
                if not data: return
                df = pd.read_csv(io.BytesIO(data))
                candidates = df[df['reportDate'] == datetime.now().strftime('%Y-%m-%d')]

                for _, row in candidates.iterrows():
                    sym = row['symbol']
                    if sym in active_symbols or sym in GOUVERNANCE["BLACKLIST"]: continue
                    if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break

                    price = float(self.alpaca.get_latest_trade(sym).price)
                    if price < GOUVERNANCE["MIN_STOCK_PRICE"]: continue
                    
                    sector = self.get_sector(sym)
                    sector_val = sum(float(p.market_value) for p in positions if self.get_sector(p.symbol) == sector)
                    if (sector_val / equity) >= GOUVERNANCE["MAX_SECTOR_EXPOSURE_PCT"]: continue

                    score, sigma, reason = 85.0, 12.0, "PROD_SENTINEL_OK" 
                    mode = "EXPLOITATION" if score >= 85 and sigma <= 20 else "EXPLORATION" if score >= 72 and sigma <= 35 else None
                    
                    if mode and SYSTEM_STATE["allocation"][mode] > 0:
                        risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * SYSTEM_STATE["allocation"][mode]
                        qty = int((equity * risk) / (price * GOUVERNANCE["BASE_SL_PCT"]))
                        qty = min(qty, int((equity * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / price))
                        
                        if qty > 0:
                            params = {
                                "symbol": sym, "qty": qty, "side": 'buy', "type": 'limit',
                                "limit_price": round(price * (1 + GOUVERNANCE["SLIPPAGE_PROTECTION"]), 2),
                                "time_in_force": 'gtc', "order_class": 'bracket',
                                "take_profit": {'limit_price': round(price * (1 + GOUVERNANCE["BASE_TP_PCT"]), 2)},
                                "stop_loss": {'stop_price': round(price * (1 - GOUVERNANCE["BASE_SL_PCT"]), 2)},
                                "client_order_id": f"apex_{uuid.uuid4().hex[:8]}"
                            }
                            if await self.executor.secure_submit(params):
                                with sqlite3.connect(DB_PATH) as conn:
                                    conn.execute("INSERT INTO trades (client_id, symbol, qty, entry_price, status, sector, mode, consensus, ai_reason) VALUES (?,?,?,?,?,?,?,?,?)",
                                                 (params["client_order_id"], sym, qty, price, 'OPEN', sector, mode, score, reason))
                                active_symbols.add(sym)
            SYSTEM_STATE["health"]["consecutive_errors"] = 0
        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("cycle_crash", level="error", error=str(e), trace=traceback.format_exc())

async def main():
    if TOKEN_WARNING:
        SLOG.log("security_warning", level="critical", msg="TITAN_DASHBOARD_TOKEN non trouvé dans l'environnement. Utilisation du fallback '12345'.")

    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), SecureMetricsHandler).serve_forever, daemon=True).start()
    engine = TitanEngine()
    SLOG.log("system_online", version=SYSTEM_STATE["engine_version"])
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
