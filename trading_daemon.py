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

# --- CONFIGURATION v4.9.8 "VANGUARD-SENTINEL" ---
DB_PATH = "titan_prod_v4_9.db"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"
LOG_FILE = "titan_engine.log"

ENV_MODE = os.getenv("ENV_MODE", "PAPER")
LIVE_AFFIRMATION = os.getenv("LIVE_AFFIRMATION", "False") == "True"

GOUVERNANCE = {
    "ENV_MODE": ENV_MODE,
    "MIN_STOCK_PRICE": 5.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25, 
    "MAX_POSITION_SIZE_PCT": 0.10,
    "MAX_TRADES_PER_DAY": 12,
    "BLACKLIST": ["GME", "AMC", "BBBY", "DJT", "SPCE", "FFIE", "LUMN"],
    "MAX_DAILY_DRAWDOWN_PCT": 0.02,
    "MAX_TOTAL_DRAWDOWN_PCT": 0.10,
    
    # LA FORGE
    "MIN_TRADES_FOR_JUDGEMENT": 10,
    "DEGRADED_THRESHOLD_USD": 0.0,
    "QUARANTINE_THRESHOLD_USD": -15.0,
    "GLOBAL_CAPS": { "EXPLOITATION": 0.80, "EXPLORATION": 0.20 },
    
    # MODES
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
    "engine_version": "4.9.8-Sentinel",
    "is_promoted": False,
    "trades_today": 0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "status": "ACTIVE", "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "status": "ACTIVE", "trades": 0}
    }
}

# --- MODULE IA (COLOSSEUM SIMULÉ) ---
class AIColosseum:
    def __init__(self, api_key):
        self.api_key = api_key

    async def get_consensus(self, session, context):
        # Simulation du quatuor IA (Claude, GPT, Gemini, Grok)
        avg = 82.0
        sigma = 12.0
        votes = [{"model": "Consensus_System", "score": 82}]
        return avg, sigma, votes

# --- MOTEUR TITAN FUSION ---
class TitanEngine:
    def __init__(self):
        self.alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self.fmp_key = os.getenv("FMP_API_KEY", "")
        self.or_key = os.getenv("OPENROUTER_API_KEY", "")

        self.alpaca = tradeapi.REST(
            self.alpaca_key, self.alpaca_secret, 
            "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets"
        )
        self.colosseum = AIColosseum(self.or_key)
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

    async def fetch_fmp_candidates(self, session):
        """Scanner FMP : Récupère les Triple Beats avec filtre de prix."""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={today}&to={today}&apikey={self.fmp_key}"
        
        candidates = []
        try:
            async with session.get(url) as resp:
                if resp.status != 200: return []
                earnings = await resp.json()
                
                for event in earnings:
                    symbol = event.get('symbol')
                    eps_actual = event.get('epsActual')
                    eps_est = event.get('epsEstimated')
                    
                    if eps_actual is not None and eps_est is not None and eps_actual > eps_est:
                        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={self.fmp_key}"
                        async with session.get(quote_url) as q_resp:
                            quote_data = await q_resp.json()
                            if quote_data:
                                q = quote_data[0]
                                price = q.get('price', 0)
                                # --- CORRECTIF AUDIT 2 : FILTRE PRIX ---
                                if price < GOUVERNANCE["MIN_STOCK_PRICE"]:
                                    continue
                                    
                                candidates.append({
                                    "symbol": symbol,
                                    "price": price,
                                    "sector": q.get('sector', 'Unknown'),
                                    "eps_surprise": (eps_actual - eps_est) / abs(eps_est) if eps_est != 0 else 0
                                })
        except Exception as e:
            logging.error(f"FMP Scan Error: {e}")
        return candidates

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
        except Exception: pass

    def sync_forge(self):
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
            
            if count >= 10 and res.get("EXPLORATION", 0) > res.get("EXPLOITATION", 0) and res.get("EXPLORATION", 0) > 0:
                SYSTEM_STATE["is_promoted"] = True
            else: SYSTEM_STATE["is_promoted"] = False

    async def get_sector_exposure(self, sector):
        """Calcule l'exposition actuelle d'un secteur spécifique."""
        try:
            positions = self.alpaca.list_positions()
            sector_value = 0.0
            for p in positions:
                # On récupère le secteur stocké en DB lors de l'ouverture
                with sqlite3.connect(DB_PATH) as conn:
                    row = conn.execute("SELECT sector FROM trades WHERE symbol=? AND status='OPEN'", (p.symbol,)).fetchone()
                    if row and row[0] == sector:
                        sector_value += float(p.market_value)
            return sector_value
        except: return 0.0

    async def execute_trade(self, symbol, price, score, sigma, mode, sector, votes):
        alloc = SYSTEM_STATE["allocation"][mode]
        if alloc <= 0: return

        cap = GOUVERNANCE["GLOBAL_CAPS"][mode]
        if mode == "EXPLORATION" and SYSTEM_STATE["is_promoted"]: cap = 0.40
        
        risk_pct = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * alloc * cap
        conviction = (score - 70) / 30
        sl_pct = min(max(GOUVERNANCE["BASE_SL_PCT"] - (conviction * 0.01), 0.015), 0.05)
        tp_pct = min(max(0.06 + (conviction * 0.04), 0.04), 0.12)

        # --- CORRECTIF AUDIT 4 : CAP TAILLE POSITION ---
        qty = int((SYSTEM_STATE["equity"] * risk_pct) / (price * sl_pct))
        max_qty = int((SYSTEM_STATE["equity"] * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / price)
        qty = min(qty, max_qty)
        
        if qty <= 0: return

        c_id = f"titan_{mode.lower()}_{symbol}_{uuid.uuid4().hex[:4]}"
        try:
            self.alpaca.submit_order(
                symbol=symbol, qty=qty, side='buy', type='limit',
                limit_price=round(price * 1.002, 2), client_order_id=c_id,
                time_in_force='gtc', order_class='bracket',
                take_profit={'limit_price': round(price * (1 + tp_pct), 2)},
                stop_loss={'stop_price': round(price * (1 - sl_pct), 2)}
            )
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO trades (client_id, symbol, qty, entry_price, status, mode, consensus, dispersion, sector, votes_json) VALUES (?,?,?,?,?,?,?,?,?,?)", 
                             (c_id, symbol, qty, price, 'OPEN', mode, score, sigma, sector, json.dumps(votes)))
            logging.info(f"✅ EXEC: {symbol} | Mode: {mode} | Secteur: {sector}")
        except Exception as e: logging.error(f"Exec error: {e}")

    async def run_cycle(self):
        with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
        self.reconcile_trades()
        self.sync_forge()
        
        acc = self.alpaca.get_account()
        SYSTEM_STATE["equity"] = float(acc.equity)
        
        # --- CORRECTIF AUDIT 1 : DRAWDOWN JOURNALIER ---
        daily_dd = (float(acc.equity) - float(acc.last_equity)) / float(acc.last_equity)
        if daily_dd <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
            SYSTEM_STATE["status"] = "veto_daily_dd"
            logging.warning(f"🛑 VETO: Limite de perte journalière atteinte ({daily_dd:.2%})")
            return

        if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
            SYSTEM_STATE["status"] = "standby"; return

        SYSTEM_STATE["status"] = "scanning"
        
        async with aiohttp.ClientSession() as session:
            candidates = await self.fetch_fmp_candidates(session)
            
            for c in candidates:
                if c['symbol'] in GOUVERNANCE["BLACKLIST"]: continue
                if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break
                
                # --- CORRECTIF AUDIT 3 : EXPOSITION SECTORIELLE ---
                sector_exp = await self.get_sector_exposure(c['sector'])
                if (sector_exp / SYSTEM_STATE["equity"]) >= GOUVERNANCE["MAX_SECTOR_EXPOSURE_PCT"]:
                    logging.info(f"🛡️ VETO SECTORIEL: {c['symbol']} ({c['sector']} saturé)")
                    continue

                # IA Consensus
                avg, sigma, votes = await self.colosseum.get_consensus(session, c)
                
                mode = "EXPLOITATION" if avg >= 85 and sigma <= 20 else "EXPLORATION" if avg >= 72 and sigma <= 35 else None
                
                if mode:
                    await self.execute_trade(c['symbol'], c['price'], avg, sigma, mode, c['sector'], votes)

async def main():
    # Metrics server sur 8080
    threading.Thread(target=lambda: HTTPServer(('0.0.0.0', 8080), 
        type('MH', (BaseHTTPRequestHandler,), {
            'do_GET': lambda s: (s.send_response(200), s.send_header("Content-type", "application/json"), 
                                s.send_header("Access-Control-Allow-Origin", "*"), s.end_headers(), 
                                s.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode()))
        })).serve_forever(), daemon=True).start()

    engine = TitanEngine()
    logging.info(f"🛡️ Titan v4.9.8 VANGUARD-SENTINEL Engine Online")
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    asyncio.run(main())
