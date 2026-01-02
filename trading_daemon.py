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

# --- CONFIGURATION v5.6.5.3 "APEX-ULTIMATE-CORS" ---
DB_PATH = "titan_prod_v5.db"
LOG_FILE = "titan_system.log"
BACKUP_DIR = "backups"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN", secrets.token_urlsafe(32))

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
        "EXPLOITATION": { "MIN_AVG": 85, "MAX_SIGMA": 20, "BASE_RISK": 0.01 },
        "EXPLORATION": { "MIN_AVG": 72, "MAX_SIGMA": 35, "BASE_RISK": 0.0025 }
    },
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "SLIPPAGE_PROTECTION": 0.002        
}

SYSTEM_STATE = {
    "status": "initializing",
    "equity": 0.0,
    "engine_version": "5.6.5.3-ApexUltimate-CORS",
    "trades_today": 0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "health": {
        "db": "unknown", 
        "alpaca": "unknown", 
        "av_api": "ok",
        "last_cycle": None, 
        "consecutive_errors": 0
    },
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "trades": 0}
    },
    "auth_hint": DASHBOARD_TOKEN[:8] + "..."
}

# --- LOGGING (Fichier + Console) ---
class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanUltimate")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, event_type, level="info", **kwargs):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "event": event_type,
            **kwargs
        }
        msg = json.dumps(payload)
        if level.lower() == "critical": self.logger.critical(msg)
        elif level.lower() == "error": self.logger.error(msg)
        elif level.lower() == "warning": self.logger.warning(msg)
        else: self.logger.info(msg)

SLOG = StructuredLogger()

# --- RETRY LOGIC ---
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

# --- HEALTH MONITOR ---
class HealthMonitor:
    def check_all(self, alpaca_api):
        checks = {"latency": True, "errors": True, "alpaca": True}
        
        if SYSTEM_STATE["health"]["last_cycle"]:
            latency = (datetime.now() - datetime.fromisoformat(SYSTEM_STATE["health"]["last_cycle"])).total_seconds()
            if latency > GOUVERNANCE["CYCLE_LATENCY_THRESHOLD"]: checks["latency"] = False
            
        if SYSTEM_STATE["health"]["consecutive_errors"] >= GOUVERNANCE["MAX_CONSECUTIVE_FAILURES"]:
            checks["errors"] = False
            
        try:
            alpaca_api.get_clock()
            SYSTEM_STATE["health"]["alpaca"] = "connected"
        except:
            checks["alpaca"] = False
            SYSTEM_STATE["health"]["alpaca"] = "error"
            
        return checks

# --- ORDER EXECUTOR ---
class OrderExecutor:
    def __init__(self, alpaca):
        self.alpaca = alpaca

    async def secure_submit(self, params):
        try:
            order = self.alpaca.submit_order(**params)
            SLOG.log("order_sent", symbol=params['symbol'], qty=params['qty'])
            
            await asyncio.sleep(3)
            check = self.alpaca.get_order(order.id)
            
            if check.status == 'rejected':
                SLOG.log("order_rejected", level="error", symbol=params['symbol'])
                SYSTEM_STATE["health"]["consecutive_errors"] += 1
                return False
            
            SYSTEM_STATE["health"]["consecutive_errors"] = 0 
            return True
        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("exec_exception", level="error", error=str(e))
            return False

# --- SECURE DASHBOARD (CORS-Fixed) ---
class SecureMetricsHandler(BaseHTTPRequestHandler):
    last_request_time = 0

    def log_message(self, format, *args): 
        return

    def _set_cors_headers(self):
        """CORS headers pour cross-origin requests."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "3600")

    def do_OPTIONS(self):
        """Preflight CORS."""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        # Rate limit
        now = time.time()
        if now - SecureMetricsHandler.last_request_time < 0.5:
            self.send_response(429)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Rate limit"}).encode())
            return
        SecureMetricsHandler.last_request_time = now

        # Auth
        auth = self.headers.get('Authorization', "")
        if not hmac.compare_digest(auth, f"Bearer {DASHBOARD_TOKEN}"):
            self.send_response(401)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return
        
        # Response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        
        response = {"metrics": SYSTEM_STATE, "timestamp": datetime.now().isoformat()}
        self.wfile.write(json.dumps(response).encode())

# --- TITAN ENGINE ---
class TitanEngine:
    def __init__(self):
        self._validate_env()
        self.alpaca = tradeapi.REST(
            ALPACA_KEY, ALPACA_SECRET, 
            "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets"
        )
        self.executor = OrderExecutor(self.alpaca)
        self.health = HealthMonitor()
        self._init_db()
        self.initial_equity = self._load_anchor()
        self.av_cache = None
        self.last_av_fetch = None
        self._backup_db()

    def _validate_env(self):
        needed = {"ALPACA_KEY": ALPACA_KEY, "ALPACA_SECRET": ALPACA_SECRET, "AV_KEY": AV_KEY}
        missing = [k for k, v in needed.items() if not v]
        if missing:
            SLOG.log("critical_init_error", level="critical", missing=missing)
            sys.exit(1)

    def _init_db(self):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, 
                    qty REAL, entry_price REAL, exit_price REAL, status TEXT, 
                    pnl REAL, mode TEXT, consensus REAL, dispersion REAL, 
                    sector TEXT, ai_reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
                conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute("CREATE TABLE IF NOT EXISTS sector_cache (symbol TEXT PRIMARY KEY, sector TEXT)")
            SYSTEM_STATE["health"]["db"] = "connected"
        except Exception as e:
            SLOG.log("db_error", level="critical", error=str(e)); sys.exit(1)

    def _load_anchor(self):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='initial_equity'").fetchone()
            if row: return float(row[0])
            try:
                equity = float(self.alpaca.get_account().equity)
                conn.execute("INSERT INTO meta (key, value) VALUES ('initial_equity', ?)", (equity,))
                return equity
            except: sys.exit(1)

    def _backup_db(self):
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        target = os.path.join(BACKUP_DIR, f"titan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        try:
            shutil.copy2(DB_PATH, target)
            with sqlite3.connect(target) as conn:
                if conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok":
                    SLOG.log("backup_ok", file=target)
        except Exception as e: SLOG.log("backup_fail", level="error", error=str(e))

    def get_sector_with_cache(self, symbol):
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT sector FROM sector_cache WHERE symbol=?", (symbol,)).fetchone()
            if res: return res[0]
        try:
            sector = yf.Ticker(symbol).info.get('sector', 'Unknown')
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT OR REPLACE INTO sector_cache VALUES (?,?)", (symbol, sector))
            return sector
        except: return "Unknown"

    async def get_signals_cached(self, session):
        """Cache AV avec fix DataFrame ambiguity."""
        now = datetime.now()
        
        if self.av_cache is not None and not self.av_cache.empty and self.last_av_fetch:
            if (now - self.last_av_fetch).total_seconds() < 3600:
                return self.av_cache
            
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={AV_KEY}"
        try:
            data = await fetch_with_retry(session, url)
            if data:
                df = pd.read_csv(io.BytesIO(data))
                filtered = df[df['reportDate'] == now.strftime('%Y-%m-%d')]
                self.av_cache = filtered
                self.last_av_fetch = now
                SYSTEM_STATE["health"]["av_api"] = "ok"
                SLOG.log("av_fetch", count=len(filtered))
                return self.av_cache
        except Exception as e:
            SLOG.log("av_error", level="warning", error=str(e))
            SYSTEM_STATE["health"]["av_api"] = "error"
        
        return pd.DataFrame()

    async def reconcile_trades(self, positions):
        """Sync DB <> Broker."""
        pos_map = {p.symbol: p for p in positions}
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT client_id, symbol, entry_price, qty, timestamp FROM trades WHERE status='OPEN'")
            
            for c_id, sym, entry, qty, ts in cursor.fetchall():
                entry_dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                
                if (datetime.now() - entry_dt).days >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                    try:
                        self.alpaca.close_position(sym)
                        SLOG.log("time_exit", symbol=sym)
                    except: pass
                    continue
                
                if sym not in pos_map:
                    try:
                        orders = self.alpaca.list_orders(status='closed', limit=5, symbols=[sym])
                        if orders:
                            latest = sorted(orders, key=lambda x: x.filled_at or x.submitted_at, reverse=True)[0]
                            exit_p = float(latest.filled_avg_price)
                            pnl = (exit_p - entry) * qty
                            cursor.execute(
                                "UPDATE trades SET status='CLOSED', exit_price=?, pnl=? WHERE client_id=?", 
                                (exit_p, pnl, c_id)
                            )
                            SLOG.log("trade_closed", symbol=sym, pnl=round(pnl, 2))
                    except: pass
            
            conn.commit()

    def sync_forge(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE date(timestamp) = date('now')")
            SYSTEM_STATE["trades_today"] = cursor.fetchone()[0]
            
            for m in ["EXPLOITATION", "EXPLORATION"]:
                pnls = [r[0] for r in cursor.execute(
                    "SELECT pnl FROM trades WHERE mode=? AND status='CLOSED' ORDER BY timestamp DESC LIMIT 20", 
                    (m,)
                ).fetchall()]
                count = len(pnls)
                exp = sum(pnls)/count if count > 0 else 0
                
                if count >= GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
                    if exp <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]: 
                        SYSTEM_STATE["allocation"][m] = 0.0
                    elif exp <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]: 
                        SYSTEM_STATE["allocation"][m] = 0.5
                    else: 
                        SYSTEM_STATE["allocation"][m] = 1.0
                
                SYSTEM_STATE["stats"][m] = {"expectancy": round(exp, 2), "trades": count}

    async def run_cycle(self):
        try:
            with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
            SYSTEM_STATE["health"]["last_cycle"] = datetime.now().isoformat()
            
            h_status = self.health.check_all(self.alpaca)
            if not all(h_status.values()):
                SYSTEM_STATE["status"] = "unhealthy_halt"
                SLOG.log("unhealthy", level="critical", checks=h_status)
                if ENV_MODE == "LIVE": self.alpaca.close_all_positions()
                return

            positions = self.alpaca.list_positions()
            acc = self.alpaca.get_account()
            equity = float(acc.equity)
            SYSTEM_STATE["equity"] = equity
            
            await self.reconcile_trades(positions)
            self.sync_forge()
            
            daily_dd = (equity - float(acc.last_equity)) / float(acc.last_equity)
            if daily_dd <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
                SYSTEM_STATE["status"] = "halt_dd"
                self.alpaca.close_all_positions(); return

            if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
                SYSTEM_STATE["status"] = "standby"; return

            SYSTEM_STATE["status"] = "scanning"
            open_symbols = set(p.symbol for p in positions)
            sector_map = {p.symbol: self.get_sector_with_cache(p.symbol) for p in positions}
            
            async with aiohttp.ClientSession() as session:
                candidates = await self.get_signals_cached(session)
                
                for _, row in candidates.iterrows():
                    sym = row['symbol']
                    if sym in open_symbols or sym in GOUVERNANCE["BLACKLIST"]: continue
                    if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break

                    try:
                        price = float(self.alpaca.get_latest_trade(sym).price)
                        if price < GOUVERNANCE["MIN_STOCK_PRICE"]: continue
                        
                        sector = self.get_sector_with_cache(sym)
                        exp_val = sum(float(p.market_value) for p in positions if sector_map.get(p.symbol) == sector)
                        if (exp_val / equity) >= GOUVERNANCE["MAX_SECTOR_EXPOSURE_PCT"]: continue

                        score, sigma, reason = await self._call_ia(session, sym, sector)
                        mode = "EXPLOITATION" if score >= 85 and sigma <= 20 else "EXPLORATION" if score >= 72 and sigma <= 35 else None

                        if mode and SYSTEM_STATE["allocation"][mode] > 0:
                            risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * SYSTEM_STATE["allocation"][mode]
                            qty = int((equity * risk) / (price * GOUVERNANCE["BASE_SL_PCT"]))
                            
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
                                        conn.execute(
                                            "INSERT INTO trades (client_id, symbol, qty, entry_price, status, sector, mode, consensus, ai_reason) VALUES (?,?,?,?,?,?,?,?,?)",
                                            (params["client_order_id"], sym, qty, price, 'OPEN', sector, mode, score, reason)
                                        )
                    except: 
                        continue

        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("cycle_crash", level="critical", error=str(e), trace=traceback.format_exc())

    async def _call_ia(self, session, sym, sector):
        if not OR_KEY: return 82.0, 12.0, "SIMULATION"
        prompt = f"PEAD expert. {sym} ({sector}). JSON: score, sigma, reason."
        try:
            res = await fetch_with_retry(
                session, "https://openrouter.ai/api/v1/chat/completions", 
                headers={"Authorization": f"Bearer {OR_KEY}"},
                json_data={
                    "model": "google/gemini-2.0-flash-001", 
                    "messages": [{"role": "user", "content": prompt}], 
                    "response_format": {"type": "json_object"}
                },
                method="POST"
            )
            if res:
                data = json.loads(res['choices'][0]['message']['content'])
                return float(data.get('score', 80)), float(data.get('sigma', 15)), data.get('reason', 'N/A')
        except: pass
        return 80.0, 15.0, "FALLBACK"

async def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), SecureMetricsHandler).serve_forever, daemon=True).start()
    engine = TitanEngine()
    SLOG.log("online", version=SYSTEM_STATE["engine_version"], token=SYSTEM_STATE["auth_hint"])
    
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    try: 
        asyncio.run(main())
    except KeyboardInterrupt: 
        SLOG.log("shutdown"); 
        sys.exit(0)
