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
import hmac
import aiohttp
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import alpaca_trade_api as tradeapi

# --- CONFIGURATION v5.6.21 "APEX-ULTIMATE: HARDENED" ---
DB_PATH = "titan_prod_v5.db"
LOG_FILE = "titan_system.log"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

# Sécurité : Le token doit être défini en ENV, sinon on génère un jeton unique par session
DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN")
if not DASHBOARD_TOKEN:
    import secrets
    DASHBOARD_TOKEN = secrets.token_hex(16)

ENV_MODE = os.getenv("ENV_MODE", "PAPER")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_KEY") 
OR_KEY = os.getenv("OPENROUTER_API_KEY")

AI_MODEL = "x-ai/grok-4.1-fast" 

GOUVERNANCE = {
    "ENV_MODE": ENV_MODE,
    "MIN_STOCK_PRICE": 5.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25, 
    "MAX_POSITION_SIZE_PCT": 0.10,      
    "MAX_TRADES_PER_DAY": 12,
    "MAX_DAILY_DRAWDOWN_PCT": 0.02,    
    "MODES": {
        "EXPLOITATION": { "MIN_SCORE": 85, "MAX_SIGMA": 20, "BASE_RISK": 0.01 },
        "EXPLORATION": { "MIN_SCORE": 72, "MAX_SIGMA": 35, "BASE_RISK": 0.0025 }
    },
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "BLACKLIST": ["GME", "AMC", "BBBY", "DJT", "SPCE", "FFIE", "LUMN"]
}

SYSTEM_STATE = {
    "status": "initializing",
    "equity": 0.0,
    "engine_version": "5.6.21 (Hardened)",
    "trades_today": 0,
    "drawdown_pct": 0.0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "health": {"db": "unknown", "last_cycle": None, "consecutive_errors": 0},
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "trades": 0}
    },
    "positions": [],
    "orders": []
}

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanGrok")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
            fh.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(fh)

    def log(self, event_type, level="info", **kwargs):
        payload = {"timestamp": datetime.now().isoformat(), "level": level.upper(), "event": event_type, **kwargs}
        self.logger.info(json.dumps(payload))

SLOG = StructuredLogger()

class SecureMetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
    def do_OPTIONS(self): self._set_headers(204)
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
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, 
                qty REAL, entry_price REAL, exit_price REAL, status TEXT, 
                pnl REAL, mode TEXT, consensus REAL, dispersion REAL, 
                sector TEXT, ai_reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("CREATE TABLE IF NOT EXISTS sector_cache (symbol TEXT PRIMARY KEY, sector TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        SYSTEM_STATE["health"]["db"] = "connected"

    async def get_sector_async(self, symbol):
        """Correctif Régression 1 : Cache SQLite + Appel Threadé pour éviter de bloquer l'Event Loop"""
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT sector FROM sector_cache WHERE symbol=?", (symbol,)).fetchone()
            if res: return res[0]
        
        try:
            # On délègue l'appel bloquant yfinance à un thread séparé
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: yf.Ticker(symbol).info)
            sector = info.get('sector', 'Unknown')
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT OR REPLACE INTO sector_cache (symbol, sector) VALUES (?,?)", (symbol, sector))
            return sector
        except:
            return "Unknown"

    def sync_forge(self):
        """Amélioration Vigilance 5 : Rétablissement de la dégradation progressive"""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            for m in ["EXPLOITATION", "EXPLORATION"]:
                rows = cursor.execute("SELECT pnl FROM trades WHERE mode=? AND status='CLOSED' ORDER BY timestamp DESC LIMIT 20", (m,)).fetchall()
                pnls = [r[0] for r in rows if r[0] is not None]
                count = len(pnls)
                exp = sum(pnls)/count if count > 0 else 0
                
                # Dégradation progressive rétablie
                if count >= 10:
                    if exp <= -15.0: SYSTEM_STATE["allocation"][m] = 0.0 # Quarantaine
                    elif exp <= 0.0: SYSTEM_STATE["allocation"][m] = 0.5 # Dégradé
                    else: SYSTEM_STATE["allocation"][m] = 1.0
                
                SYSTEM_STATE["stats"][m] = {"expectancy": round(exp, 2), "trades": count}

    async def get_ai_score(self, session, sym, price):
        prompt = f"PEAD Analysis for {sym} at {price}$. JSON only: {{\"score\": 0-100, \"sigma\": 0-50, \"reason\": \"...\"}}"
        headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
        payload = {"model": AI_MODEL, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
        try:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    content = json.loads(res['choices'][0]['message']['content'])
                    return float(content.get('score', 80)), float(content.get('sigma', 15)), content.get('reason', 'N/A')
        except Exception as e:
            SLOG.log("ai_timeout", error=str(e), symbol=sym)
        return 80.0, 15.0, "IA_FALLBACK"

    async def reconcile(self, positions):
        active_symbols = set([p.symbol for p in positions])
        open_orders = self.alpaca.list_orders(status='open', limit=50)
        
        # On calcule l'exposition sectorielle réelle ici (Correctif Régression 2)
        sector_exposure = {}
        for p in positions:
            sym = p.symbol
            # On récupère le secteur (via cache si possible)
            sector = await self.get_sector_async(sym)
            val = float(p.market_value)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + val

        SYSTEM_STATE["positions"] = [{"symbol": p.symbol, "qty": p.qty, "unrealized_pnl": float(p.unrealized_pl)} for p in positions]
        SYSTEM_STATE["orders"] = []
        for o in open_orders:
            if o.status in ['new', 'partially_filled', 'accepted', 'pending_new']:
                active_symbols.add(o.symbol)
                SYSTEM_STATE["orders"].append({"symbol": o.symbol, "status": o.status})
        
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute("SELECT COUNT(*) FROM trades WHERE date(timestamp) = date('now')").fetchone()[0]
            SYSTEM_STATE["trades_today"] = count
            
        return active_symbols, sector_exposure

    async def run_cycle(self):
        try:
            SYSTEM_STATE["health"]["last_cycle"] = datetime.now().isoformat()
            with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())

            acc = self.alpaca.get_account()
            SYSTEM_STATE["equity"] = float(acc.equity)
            active_symbols, sector_exposure = await self.reconcile(self.alpaca.list_positions())
            self.sync_forge()

            daily_loss = float(acc.equity) - float(acc.last_equity)
            SYSTEM_STATE["drawdown_pct"] = abs(daily_loss) / float(acc.last_equity) if daily_loss < 0 else 0.0
            if SYSTEM_STATE["drawdown_pct"] > GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
                SYSTEM_STATE["status"] = "HALTED_BY_DRAWDOWN"; return

            if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
                SYSTEM_STATE["status"] = "standby"; return

            SYSTEM_STATE["status"] = "scanning"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={AV_KEY}") as resp:
                    if resp.status != 200: return
                    data = await resp.read()
                    try:
                        df = pd.read_csv(io.BytesIO(data))
                        # Vigilance 4 : Validation stricte des colonnes
                        if 'symbol' not in df.columns or 'reportDate' not in df.columns:
                            SLOG.log("av_invalid_csv", content=str(data[:100]))
                            return
                        candidates = df[df['reportDate'] == datetime.now().strftime('%Y-%m-%d')]
                    except Exception as e:
                        SLOG.log("av_parse_error", error=str(e))
                        return

                for _, row in candidates.iterrows():
                    sym = row.get('symbol')
                    if not sym or sym in active_symbols or sym in GOUVERNANCE["BLACKLIST"]: continue
                    if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break

                    try:
                        price = float(self.alpaca.get_latest_trade(sym).price)
                        score, sigma, reason = await self.get_ai_score(session, sym, price)
                        mode = "EXPLOITATION" if score >= 85 and sigma <= 20 else "EXPLORATION" if score >= 72 and sigma <= 35 else None
                        
                        if mode:
                            alloc_factor = SYSTEM_STATE["allocation"][mode]
                            if alloc_factor <= 0: continue

                            # Correctif Régression 2 : Veto Sectoriel Effectif
                            sector = await self.get_sector_async(sym)
                            current_sector_val = sector_exposure.get(sector, 0)
                            if (current_sector_val / SYSTEM_STATE["equity"]) >= GOUVERNANCE["MAX_SECTOR_EXPOSURE_PCT"]:
                                SLOG.log("sector_veto", symbol=sym, sector=sector)
                                continue

                            risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * alloc_factor
                            qty = int((SYSTEM_STATE["equity"] * risk) / (price * GOUVERNANCE["BASE_SL_PCT"]))
                            
                            if qty > 0:
                                cid = f"apex_{uuid.uuid4().hex[:8]}"
                                self.alpaca.submit_order(
                                    symbol=sym, qty=qty, side='buy', type='limit',
                                    limit_price=round(price * 1.002, 2),
                                    time_in_force='gtc', order_class='bracket',
                                    take_profit={'limit_price': round(price * 1.06, 2)},
                                    stop_loss={'stop_price': round(price * 0.97, 2)},
                                    client_order_id=cid
                                )
                                with sqlite3.connect(DB_PATH) as conn:
                                    conn.execute("INSERT INTO trades (client_id, symbol, qty, entry_price, status, mode, consensus, dispersion, sector, ai_reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                                 (cid, sym, qty, price, 'OPEN', mode, score, sigma, sector, reason))
                                SYSTEM_STATE["trades_today"] += 1
                                SLOG.log("trade_executed", symbol=sym, mode=mode, score=score)
                    except Exception as e:
                        SLOG.log("execution_error", symbol=sym, error=str(e))

            SYSTEM_STATE["health"]["consecutive_errors"] = 0
        except Exception as e:
            SYSTEM_STATE["health"]["consecutive_errors"] += 1
            SLOG.log("cycle_crash", error=str(e))

async def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), SecureMetricsHandler).serve_forever, daemon=True).start()
    engine = TitanEngine()
    SLOG.log("system_online", token_hint=DASHBOARD_TOKEN[:4] + "...")
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
