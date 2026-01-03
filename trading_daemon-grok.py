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

# --- CONFIGURATION v5.6.11-LTS "GROK-SENTINEL" ---
DB_PATH = "titan_prod_v5.db"
LOG_FILE = "titan_system.log"
BACKUP_DIR = "backups"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

# SÉCURITÉ : Récupération du token via l'environnement
DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN")
if not DASHBOARD_TOKEN:
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
    "engine_version": "5.6.11-LTS (Grok)",
    "trades_today": 0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "health": {"db": "unknown", "alpaca": "unknown", "last_cycle": None, "consecutive_errors": 0},
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "trades": 0}
    },
    "ai_brain": "x-ai/grok-2-1212",
    "security_alert": "INSECURE_DEFAULT_TOKEN" if TOKEN_WARNING else "SECURE_ENV_LOADED",
    "positions": [],
    "orders": []
}

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanGrok")
        self.logger.setLevel(logging.INFO)
        fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

    def log(self, event_type, level="info", **kwargs):
        payload = {"timestamp": datetime.now().isoformat(), "level": level.upper(), "event": event_type, **kwargs}
        self.logger.info(json.dumps(payload))

SLOG = StructuredLogger()

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

    def do_OPTIONS(self): self._set_headers(200)

    def do_GET(self):
        now = time.time()
        if now - SecureMetricsHandler.last_req < 0.2:
            self.send_response(429); self.end_headers(); return
        SecureMetricsHandler.last_req = now
        
        auth = self.headers.get('Authorization', "")
        if not hmac.compare_digest(auth, f"Bearer {DASHBOARD_TOKEN}"):
            self._set_headers(401)
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return

        self._set_headers(200)
        self.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode())

class TitanEngine:
    def __init__(self):
        self.alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, 
            "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, qty REAL, entry_price REAL, exit_price REAL, status TEXT, pnl REAL, mode TEXT, consensus REAL, dispersion REAL, sector TEXT, ai_reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS sector_cache (symbol TEXT PRIMARY KEY, sector TEXT)")
        SYSTEM_STATE["health"]["db"] = "connected"

    def get_sector(self, sym):
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT sector FROM sector_cache WHERE symbol=?", (sym,)).fetchone()
            if res: return res[0]
        try:
            s = yf.Ticker(sym).info.get('sector', 'Unknown')
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT OR REPLACE INTO sector_cache VALUES (?,?)", (sym, s))
            return s
        except: return "Unknown"

    async def get_ai_score(self, session, sym, sector, price):
        """Validation cognitive via OpenRouter (Grok 2)."""
        prompt = (
            f"Analyse quantitative PEAD pour {sym} (Secteur: {sector}). "
            f"Prix actuel: {price}$. "
            f"Évalue le potentiel de drift à 3 jours après les résultats. "
            f"Réponds uniquement au format JSON : {{\"score\": 0-100, \"sigma\": 0-50, \"reason\": \"...\"}}"
        )
        headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
        payload = {"model": "x-ai/grok-2-1212", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}

        for i in range(5): # Retry Logic Robuste (Recommandation Audit #3)
            try:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        content = json.loads(res['choices'][0]['message']['content'])
                        return float(content.get('score', 80)), float(content.get('sigma', 15)), content.get('reason', 'N/A')
                    elif resp.status == 429: await asyncio.sleep(2**i)
                    else: break
            except: await asyncio.sleep(2**i)
        return 80.0, 15.0, "IA_TIMEOUT_FALLBACK"

    async def reconcile(self, positions):
        """Audit #4 : Active symbols centralisés et synchronisation UI."""
        active_symbols = set([p.symbol for p in positions])
        SYSTEM_STATE["positions"] = [{
            "symbol": p.symbol, "qty": p.qty, "market_price": p.current_price, 
            "avg_entry": p.avg_entry_price, "unrealized_pnl": float(p.unrealized_pl)
        } for p in positions]
        
        try:
            orders = self.alpaca.list_orders(status='open')
            SYSTEM_STATE["orders"] = [{
                "symbol": o.symbol, "type": o.type, "side": o.side, "qty": o.qty, 
                "status": o.status, "submitted_at": o.submitted_at.strftime("%b %d, %H:%M")
            } for o in orders]
            for o in orders: active_symbols.add(o.symbol)
        except: pass
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
                    SYSTEM_STATE["allocation"][m] = 0.0 if exp <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"] else 0.5 if exp <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"] else 1.0
                SYSTEM_STATE["stats"][m] = {"expectancy": round(exp, 2), "trades": count}

    async def run_cycle(self):
        try:
            SYSTEM_STATE["health"]["last_cycle"] = datetime.now().isoformat()
            positions = self.alpaca.list_positions()
            acc = self.alpaca.get_account()
            SYSTEM_STATE["equity"] = float(acc.equity)
            active_symbols = await self.reconcile(positions)
            self.sync_forge()

            if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
                SYSTEM_STATE["status"] = "standby"; return

            # AUDIT #1 : Optimisation Sector Cache Performance
            # On pré-calcule le map des secteurs des positions actuelles AVANT la boucle
            sector_map = {p.symbol: self.get_sector(p.symbol) for p in positions}

            SYSTEM_STATE["status"] = "scanning"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={AV_KEY}") as resp:
                    if resp.status != 200: return
                    candidates = pd.read_csv(io.BytesIO(await resp.read()))
                    candidates = candidates[candidates['reportDate'] == datetime.now().strftime('%Y-%m-%d')]

                for _, row in candidates.iterrows():
                    sym = row['symbol']
                    if sym in active_symbols or sym in GOUVERNANCE["BLACKLIST"]: continue
                    if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break

                    price = float(self.alpaca.get_latest_trade(sym).price)
                    sector = self.get_sector(sym)
                    
                    # Utilisation du sector_map optimisé (Audit #1)
                    sector_val = sum(float(p.market_value) for p in positions if sector_map.get(p.symbol) == sector)
                    if (sector_val / SYSTEM_STATE["equity"]) >= GOUVERNANCE["MAX_SECTOR_EXPOSURE_PCT"]: continue

                    score, sigma, reason = await self.get_ai_score(session, sym, sector, price)
                    mode = "EXPLOITATION" if score >= 85 and sigma <= 20 else "EXPLORATION" if score >= 72 and sigma <= 35 else None
                    
                    if mode and SYSTEM_STATE["allocation"][mode] > 0:
                        risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * SYSTEM_STATE["allocation"][mode]
                        qty = min(int((SYSTEM_STATE["equity"] * risk) / (price * GOUVERNANCE["BASE_SL_PCT"])), 
                                  int((SYSTEM_STATE["equity"] * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / price))
                        
                        if qty > 0:
                            # AUDIT #2 : Order Verification & Exception Handling
                            try:
                                params = {
                                    "symbol": sym, "qty": qty, "side": 'buy', "type": 'limit',
                                    "limit_price": round(price * (1 + GOUVERNANCE["SLIPPAGE_PROTECTION"]), 2),
                                    "time_in_force": 'gtc', "order_class": 'bracket',
                                    "take_profit": {'limit_price': round(price * (1 + GOUVERNANCE["BASE_TP_PCT"]), 2)},
                                    "stop_loss": {'stop_price': round(price * (1 - GOUVERNANCE["BASE_SL_PCT"]), 2)},
                                    "client_order_id": f"apex_{uuid.uuid4().hex[:8]}"
                                }
                                order = self.alpaca.submit_order(**params)
                                SLOG.log("order_submitted", id=order.id, symbol=sym)

                                # Vérification post-ordre (Audit #2)
                                await asyncio.sleep(3)
                                check = self.alpaca.get_order(order.id)
                                if check.status == 'rejected':
                                    SLOG.log("order_rejected", level="error", symbol=sym)
                                    SYSTEM_STATE["health"]["consecutive_errors"] += 1
                                    continue

                                with sqlite3.connect(DB_PATH) as conn:
                                    conn.execute("INSERT INTO trades (client_id, symbol, qty, entry_price, status, sector, mode, consensus, dispersion, ai_reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                                 (params["client_order_id"], sym, qty, price, 'OPEN', sector, mode, score, sigma, reason))
                                active_symbols.add(sym)
                                SLOG.log("trade_opened", symbol=sym, qty=qty, mode=mode, score=score)
                            except Exception as e:
                                SLOG.log("order_exception", level="error", error=str(e))
                                SYSTEM_STATE["health"]["consecutive_errors"] += 1
                                continue

            SYSTEM_STATE["health"]["consecutive_errors"] = 0
        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("cycle_crash", level="error", error=str(e))

async def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), SecureMetricsHandler).serve_forever, daemon=True).start()
    engine = TitanEngine()
    SLOG.log("system_online", version=SYSTEM_STATE["engine_version"])
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
