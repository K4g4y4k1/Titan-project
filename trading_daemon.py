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

# --- CONFIGURATION v5.6.5 "APEX-ULTIMATE" ---
DB_PATH = "titan_prod_v5.db"
LOG_FILE = "titan_system.log"
BACKUP_DIR = "backups"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

# Sécurité Dashboard
DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN", "12345") # Token par défaut ou via env

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
    "engine_version": "5.6.5-ApexUltimate",
    "trades_today": 0,
    "daily_pnl_usd": 0.0,
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
    "last_trade": None,
    "auth_hint": DASHBOARD_TOKEN[:4] + "..."
}

# --- OBSERVABILITÉ ---
class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanUltimate")
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5
        )
        self.logger.addHandler(handler)

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

# --- MONITORING SANTÉ ---
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

# --- EXÉCUTION (LIMIT ORDERS) ---
class OrderExecutor:
    def __init__(self, alpaca):
        self.alpaca = alpaca

    async def secure_submit(self, params):
        try:
            params["type"] = "limit"
            order = self.alpaca.submit_order(**params)
            SLOG.log("order_sent", symbol=params['symbol'], qty=params['qty'], price=params['limit_price'])
            
            await asyncio.sleep(3)
            check = self.alpaca.get_order(order.id)
            
            if check.status == 'rejected':
                SLOG.log("order_rejected", level="error", symbol=params['symbol'], reason=getattr(check, 'rejection_reason', 'unknown'))
                SYSTEM_STATE["health"]["consecutive_errors"] += 1
                return False
            
            SYSTEM_STATE["health"]["consecutive_errors"] = 0 
            return True
        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("exec_exception", level="error", error=str(e))
            return False

# --- SERVEUR MÉTRIQUES AVEC FIX CORS ROBUSTE ---
class SecureMetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return

    def _set_cors_headers(self):
        """Ajoute les headers nécessaires pour autoriser les navigateurs externes."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, POST")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept, Origin")
        self.send_header("Access-Control-Max-Age", "86400") # Cache du preflight pour 24h

    def do_OPTIONS(self):
        """Répond aux requêtes de pré-vérification (Preflight) des navigateurs."""
        self.send_response(204) # No Content, standard pour OPTIONS
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        # Autoriser le health check sans token pour le debug
        if self.path == '/health':
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b"OK")
            return

        auth = self.headers.get('Authorization', "")
        
        # Vérification du Token
        if not hmac.compare_digest(auth, f"Bearer {DASHBOARD_TOKEN}"):
            self.send_response(401)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return
            
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        
        # Envoi des métriques
        self.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode())

# --- MOTEUR TITAN APEX ---
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

    def _validate_env(self):
        needed = {"ALPACA_KEY": ALPACA_KEY, "AV_KEY": AV_KEY}
        missing = [k for k, v in needed.items() if not v]
        if missing:
            SLOG.log("startup_failed", level="critical", missing=missing)
            sys.exit(1)

    def _init_db(self):
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

    def _load_anchor(self):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='initial_equity'").fetchone()
            if row: return float(row[0])
            try:
                equity = float(self.alpaca.get_account().equity)
                conn.execute("INSERT INTO meta (key, value) VALUES ('initial_equity', ?)", (equity,))
                return equity
            except: sys.exit(1)

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
        now = datetime.now()
        if self.av_cache and self.last_av_fetch and (now - self.last_av_fetch).total_seconds() < 3600:
            return self.av_cache
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={AV_KEY}"
        try:
            async with session.get(url, timeout=20) as resp:
                if resp.status == 200:
                    df = pd.read_csv(io.BytesIO(await resp.read()))
                    self.av_cache = df[df['reportDate'] == now.strftime('%Y-%m-%d')]
                    self.last_av_fetch = now
                    SYSTEM_STATE["health"]["av_api"] = "ok"
                    return self.av_cache
        except: pass
        return pd.DataFrame()

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

    async def reconcile_trades(self, positions):
        pos_map = {p.symbol: p for p in positions}
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT client_id, symbol, entry_price, qty, timestamp FROM trades WHERE status='OPEN'")
            for c_id, sym, entry, qty, ts in cursor.fetchall():
                if (datetime.now() - datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')).days >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                    self.alpaca.close_position(sym); continue
                if sym not in pos_map:
                    orders = self.alpaca.list_orders(status='closed', limit=5, symbols=[sym])
                    if orders:
                        latest = sorted(orders, key=lambda x: x.filled_at if x.filled_at else x.submitted_at, reverse=True)[0]
                        exit_p = float(latest.filled_avg_price)
                        cursor.execute("UPDATE trades SET status='CLOSED', exit_price=?, pnl=? WHERE client_id=?", 
                                     (exit_p, (exit_p - entry) * qty, c_id))
            conn.commit()

    async def get_ai_score(self, session, c):
        if not OR_KEY: return 82.0, 12.0, "SIMULATION: Clé manquante"
        prompt = f"Analyste PEAD. {c['symbol']} ({c['sector']}). JSON: score(0-100), sigma(0-50), reason [FUND]|[TECH]|[VETO]."
        try:
            async with session.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OR_KEY}"},
                json={"model": "google/gemini-2.0-flash-001", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
                timeout=12) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    data = json.loads(res['choices'][0]['message']['content'])
                    return float(data.get('score', 80)), float(data.get('sigma', 15)), data.get('reason', 'N/A')
        except: pass
        return 80.0, 15.0, "FALLBACK: Erreur IA"

    async def run_cycle(self):
        try:
            with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
            SYSTEM_STATE["health"]["last_cycle"] = datetime.now().isoformat()
            
            h_status = self.health.check_all(self.alpaca)
            if not all(h_status.values()):
                SYSTEM_STATE["status"] = "unhealthy_halt"
                if ENV_MODE == "LIVE": self.alpaca.close_all_positions()
                return

            positions = self.alpaca.list_positions()
            acc = self.alpaca.get_account()
            equity = float(acc.equity)
            SYSTEM_STATE["equity"] = equity
            SYSTEM_STATE["daily_pnl_usd"] = equity - float(acc.last_equity)
            
            await self.reconcile_trades(positions)
            self.sync_forge()
            
            if (equity - float(acc.last_equity)) / float(acc.last_equity) <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
                self.alpaca.close_all_positions(); SYSTEM_STATE["status"] = "halt_daily_dd"; return

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

                        score, sigma, reason = await self.get_ai_score(session, {"symbol": sym, "sector": sector})
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
                                    SYSTEM_STATE["last_trade"] = {"symbol": sym, "consensus": score, "ai_reason": reason}
                                    with sqlite3.connect(DB_PATH) as conn:
                                        conn.execute("INSERT INTO trades (client_id, symbol, qty, entry_price, status, sector, mode, consensus, ai_reason) VALUES (?,?,?,?,?,?,?,?,?)",
                                                     (params["client_order_id"], sym, qty, price, 'OPEN', sector, mode, score, reason))
                    except: continue
        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("cycle_crash", level="critical", error=str(e))

async def main():
    # Démarrage du serveur de métriques
    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), SecureMetricsHandler).serve_forever, daemon=True).start()
    engine = TitanEngine()
    SLOG.log("system_online", version=SYSTEM_STATE["engine_version"])
    while True:
        await engine.run_cycle(); await asyncio.sleep(60)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: sys.exit(0)
