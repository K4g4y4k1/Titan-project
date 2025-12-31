import asyncio
import sqlite3
import logging
import os
import sys
import json
import uuid
import statistics
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import alpaca_trade_api as tradeapi
import aiohttp

# --- CONFIGURATION v4.9.6 "THE FINAL VANGUARD" ---
DB_PATH = "titan_prod_v4_9.db"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"
LOG_FILE = "titan_engine.log"

ENV_MODE = os.getenv("ENV_MODE", "PAPER")
LIVE_AFFIRMATION = os.getenv("LIVE_AFFIRMATION", "False") == "True"
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
OR_KEY = os.getenv("OPENROUTER_API_KEY")

GOUVERNANCE = {
    "ENV_MODE": ENV_MODE,
    "MIN_STOCK_PRICE": 5.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25, 
    "MAX_POSITION_SIZE_PCT": 0.10,
    "MAX_TRADES_PER_DAY": 12,
    "BLACKLIST": ["GME", "AMC", "BBBY", "DJT", "SPCE", "FFIE", "LUMN"],
    "MAX_DAILY_DRAWDOWN_PCT": 0.02,
    "MAX_TOTAL_DRAWDOWN_PCT": 0.10,
    
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
    "status": "starting",
    "equity": 0.0,
    "engine_version": "4.9.6-Vanguard",
    "is_promoted": False,
    "trades_today": 0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "status": "ACTIVE", "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "status": "ACTIVE", "trades": 0}
    }
}

# --- MÉTRIQUES ROBUSTES ---
class TitanMetricsHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
    def do_GET(self):
        status = 200 if not os.path.exists(HALT_FILE) else 503
        self._set_headers(status)
        self.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode())
    def log_message(self, format, *args): return

class TitanEngine:
    def __init__(self):
        if ENV_MODE == "LIVE" and not LIVE_AFFIRMATION:
            print("🚨 ERREUR: Mode LIVE sans LIVE_AFFIRMATION=True.")
            sys.exit(1)
        
        self.alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, 
                                    "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets")
        self._init_db()
        self.initial_equity = self._load_anchor()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, qty REAL, entry_price REAL, exit_price REAL, status TEXT, pnl REAL, mode TEXT, consensus REAL, dispersion REAL, sector TEXT, votes_json TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    def _load_anchor(self):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='initial_equity'").fetchone()
            if row: return float(row[0])
            equity = float(self.alpaca.get_account().equity)
            conn.execute("INSERT INTO meta (key, value) VALUES ('initial_equity', ?)", (equity,))
            return equity

    def reconcile_trades(self):
        try:
            positions = {p.symbol: p for p in self.alpaca.list_positions()}
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT client_id, symbol, entry_price, qty FROM trades WHERE status='OPEN'")
                for c_id, sym, entry, qty in cursor.fetchall():
                    if sym not in positions:
                        orders = self.alpaca.list_orders(status='closed', limit=1, symbols=[sym])
                        exit_p = float(orders[0].filled_avg_price) if orders else entry
                        pnl = (exit_p - entry) * qty
                        cursor.execute("UPDATE trades SET status='CLOSED', exit_price=?, pnl=? WHERE client_id=?", (exit_p, pnl, c_id))
                conn.commit()
        except Exception as e: logging.error(f"Reconcile error: {e}")

    def sync_governance_forge(self):
        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE date(timestamp) = ?", (today,))
            SYSTEM_STATE["trades_today"] = cursor.fetchone()[0]
            
            res = {}
            for m in ["EXPLOITATION", "EXPLORATION"]:
                pnls = [r[0] for r in cursor.execute("SELECT pnl FROM trades WHERE mode=? AND status='CLOSED' ORDER BY timestamp DESC LIMIT 20", (m,)).fetchall()]
                count = len(pnls)
                exp = sum(pnls)/count if count > 0 else 0
                res[m] = exp
                if count >= GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
                    if exp <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]: SYSTEM_STATE["allocation"][m] = 0.0
                    elif exp <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]: SYSTEM_STATE["allocation"][m] = 0.5
                    else: SYSTEM_STATE["allocation"][m] = 1.0
                SYSTEM_STATE["stats"][m] = {"expectancy": round(exp, 2), "trades": count}

            # Promotion Logic (v4.9.6)
            if res.get("EXPLORATION", 0) > res.get("EXPLOITATION", 0) and res.get("EXPLORATION", 0) > 0:
                SYSTEM_STATE["is_promoted"] = True
            else: SYSTEM_STATE["is_promoted"] = False

    async def run_cycle(self):
        with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
        self.reconcile_trades()
        self.sync_governance_forge()
        
        acc = self.alpaca.get_account()
        SYSTEM_STATE["equity"] = float(acc.equity)
        
        # Disjoncteurs
        daily_dd = (SYSTEM_STATE["equity"] - float(acc.last_equity)) / float(acc.last_equity)
        total_dd = (SYSTEM_STATE["equity"] - self.initial_equity) / self.initial_equity
        
        if total_dd <= -GOUVERNANCE["MAX_TOTAL_DRAWDOWN_PCT"]:
            with open(HALT_FILE, "w") as f: f.write(f"HALT_TOTAL_DD_{datetime.now()}")
            self.alpaca.close_all_positions()
            return

        if daily_dd <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
            SYSTEM_STATE["status"] = "veto_daily_dd"; return

        if os.path.exists(HALT_FILE):
            SYSTEM_STATE["status"] = "halted"; return

        if not self.alpaca.get_clock().is_open:
            SYSTEM_STATE["status"] = "market_closed"; return
        
        SYSTEM_STATE["status"] = "running"
        # IA & Scanning...

async def main():
    server = HTTPServer(('0.0.0.0', 8080), TitanMetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine = TitanEngine()
    logging.info(f"🏰 Titan Engine v{SYSTEM_STATE['engine_version']} Online")
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    asyncio.run(main())
