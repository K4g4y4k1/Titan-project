import asyncio
import sqlite3
import logging
import os
import sys
import json
import uuid
import threading
import pandas as pd
import io
import yfinance as yf
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import alpaca_trade_api as tradeapi
import aiohttp

# --- CONFIGURATION v5.6.1 "APEX-GUARDIAN" (PROMPT OPTIMIZED) ---
DB_PATH = "titan_prod_v5.db"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"

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
    "engine_version": "5.6.1-SeniorAnalyst",
    "trades_today": 0,
    "allocation": {"EXPLOITATION": 1.0, "EXPLORATION": 1.0},
    "stats": {
        "EXPLOITATION": {"expectancy": 0.0, "trades": 0},
        "EXPLORATION": {"expectancy": 0.0, "trades": 0}
    }
}

class TitanEngine:
    def __init__(self):
        self.alpaca = tradeapi.REST(
            ALPACA_KEY, ALPACA_SECRET, 
            "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets"
        )
        self._init_db()
        self.initial_equity = self._load_anchor()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, symbol TEXT, 
                qty REAL, entry_price REAL, exit_price REAL, status TEXT, 
                pnl REAL, mode TEXT, consensus REAL, dispersion REAL, 
                sector TEXT, surprise_pct REAL, ai_reason TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS sector_cache (symbol TEXT PRIMARY KEY, sector TEXT)")

    def _load_anchor(self):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='initial_equity'").fetchone()
            if row: return float(row[0])
            try:
                equity = float(self.alpaca.get_account().equity)
                conn.execute("INSERT INTO meta (key, value) VALUES ('initial_equity', ?)", (equity,))
                return equity
            except: return 100000.0

    def get_sector_with_cache(self, symbol):
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT sector FROM sector_cache WHERE symbol=?", (symbol,)).fetchone()
            if res: return res[0]
        try:
            ticker = yf.Ticker(symbol)
            sector = ticker.info.get('sector', 'Unknown')
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT OR REPLACE INTO sector_cache VALUES (?,?)", (symbol, sector))
            return sector
        except: return "Unknown"

    async def reconcile_trades(self):
        try:
            positions = {p.symbol: p for p in self.alpaca.list_positions()}
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT client_id, symbol, entry_price, qty, timestamp FROM trades WHERE status='OPEN'")
                for c_id, sym, entry, qty, ts in cursor.fetchall():
                    entry_dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    if (datetime.now() - entry_dt).days >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                        self.alpaca.close_position(sym)
                        logging.info(f"⏳ Time-Exit J+{GOUVERNANCE['MAX_HOLDING_DAYS']} : {sym}")
                        continue
                    if sym not in positions:
                        orders = self.alpaca.list_orders(status='closed', limit=10, symbols=[sym])
                        if orders:
                            orders.sort(key=lambda x: x.filled_at if x.filled_at else x.submitted_at, reverse=True)
                            latest_order = orders[0]
                            exit_p = float(latest_order.filled_avg_price)
                            cursor.execute("UPDATE trades SET status='CLOSED', exit_price=?, pnl=? WHERE client_id=?", 
                                         (exit_p, (exit_p - entry) * qty, c_id))
                conn.commit()
        except Exception as e: logging.error(f"Reconcile Error: {e}")

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

    async def fetch_signals(self, session):
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={AV_KEY}"
        candidates = []
        try:
            async with session.get(url) as resp:
                if resp.status != 200: return []
                df = pd.read_csv(io.BytesIO(await resp.read()))
                today_earnings = df[df['reportDate'] == datetime.now().strftime('%Y-%m-%d')]
                open_syms = [p.symbol for p in self.alpaca.list_positions()]
                for _, row in today_earnings.iterrows():
                    sym = row['symbol']
                    if sym in GOUVERNANCE["BLACKLIST"] or sym in open_syms: continue
                    try:
                        trade = self.alpaca.get_latest_trade(sym)
                        if trade.price < GOUVERNANCE["MIN_STOCK_PRICE"]: continue
                        sector = self.get_sector_with_cache(sym)
                        candidates.append({
                            "symbol": sym, "price": trade.price, "sector": sector,
                            "surprise_pct": 10.0 # On garde un buffer de surprise estimé
                        })
                    except: continue
        except: pass
        return candidates

    async def get_ai_score(self, session, c):
        """Optimisation IA v5.6.1 : Format Senior Analyst [FUND][TECH][VETO]."""
        if not OR_KEY: return 82.0, 12.0, "SIMULATION: Clé OpenRouter manquante."
        
        prompt = (
            f"En tant qu'analyste quantitatif expert en PEAD (Post-Earnings Announcement Drift), "
            f"évalue : Symbole: {c['symbol']}, Secteur: {c['sector']}, Prix: {c['price']}$. "
            f"Surprise estimée: {c.get('surprise_pct', 'N/A')}%. "
            f"Réponds en JSON avec 'score' (0-100), 'sigma' (0-50) et "
            f"'reason' au format : [FUND] <analyse> | [TECH] <analyse> | [VETO] <risque_majeur>"
        )
        
        try:
            async with session.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OR_KEY}"},
                json={"model": "google/gemini-2.0-flash-001", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
                timeout=12) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    content = json.loads(res['choices'][0]['message']['content'])
                    return float(content.get('score', 80)), float(content.get('sigma', 15)), content.get('reason', 'N/A')
        except: pass
        return 80.0, 15.0, "FALLBACK: Erreur de communication IA."

    async def run_cycle(self):
        with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
        await self.reconcile_trades()
        self.sync_forge()
        acc = self.alpaca.get_account()
        equity = float(acc.equity)
        SYSTEM_STATE["equity"] = equity
        
        # Kill-Switches
        if (equity - float(acc.last_equity)) / float(acc.last_equity) <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
            self.alpaca.close_all_positions(); SYSTEM_STATE["status"] = "CRITICAL_HALT"; return
        if (equity - self.initial_equity) / self.initial_equity <= -GOUVERNANCE["MAX_TOTAL_DRAWDOWN_PCT"]:
            self.alpaca.close_all_positions(); SYSTEM_STATE["status"] = "CRITICAL_HALT"; return

        if not self.alpaca.get_clock().is_open or os.path.exists(HALT_FILE):
            SYSTEM_STATE["status"] = "standby"; return

        SYSTEM_STATE["status"] = "scanning"
        async with aiohttp.ClientSession() as session:
            candidates = await self.fetch_signals(session)
            for c in candidates:
                if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break
                
                positions = self.alpaca.list_positions()
                sector_exp = sum(float(p.market_value) for p in positions if self.get_sector_with_cache(p.symbol) == c['sector'])
                if (sector_exp / equity) >= GOUVERNANCE["MAX_SECTOR_EXPOSURE_PCT"]: continue

                score, sigma, reason = await self.get_ai_score(session, c)
                mode = "EXPLOITATION" if score >= 85 and sigma <= 20 else "EXPLORATION" if score >= 72 and sigma <= 35 else None
                
                if mode:
                    alloc = SYSTEM_STATE["allocation"][mode]
                    if alloc <= 0: continue
                    risk_pct = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * alloc * GOUVERNANCE["GLOBAL_CAPS"][mode]
                    qty = int((equity * risk_pct) / (c['price'] * GOUVERNANCE["BASE_SL_PCT"]))
                    qty = min(qty, int((equity * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / c['price']))
                    
                    if qty > 0:
                        try:
                            self.alpaca.submit_order(
                                symbol=c['symbol'], qty=qty, side='buy', type='limit',
                                limit_price=round(c['price'] * (1 + GOUVERNANCE["SLIPPAGE_PROTECTION"]), 2),
                                time_in_force='gtc', order_class='bracket',
                                take_profit={'limit_price': round(c['price'] * (1 + GOUVERNANCE["BASE_TP_PCT"]), 2)},
                                stop_loss={'stop_price': round(c['price'] * (1 - GOUVERNANCE["BASE_SL_PCT"]), 2)},
                                client_order_id=f"apex_{uuid.uuid4().hex[:6]}"
                            )
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.execute("INSERT INTO trades (client_id, symbol, qty, entry_price, status, mode, consensus, dispersion, sector, ai_reason) VALUES (?,?,?,?,?,?,?,?,?,?)", 
                                             (uuid.uuid4().hex, c['symbol'], qty, c['price'], 'OPEN', mode, score, sigma, c['sector'], reason))
                            logging.info(f"✅ APEX [{mode}] : {c['symbol']} | Reason: {reason}")
                        except Exception as e: logging.error(f"Exec Error: {e}")

async def main():
    threading.Thread(target=lambda: HTTPServer(('0.0.0.0', 8080), type('MH', (BaseHTTPRequestHandler,), {'do_GET': lambda s: (s.send_response(200), s.send_header("Access-Control-Allow-Origin", "*"), s.end_headers(), s.wfile.write(json.dumps({"metrics": SYSTEM_STATE}).encode()))})).serve_forever(), daemon=True).start()
    engine = TitanEngine()
    logging.info(f"🛡️ Titan v5.6.1 Apex-Guardian Online (Analyste Senior Mode)")
    while True:
        await engine.run_cycle(); await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    asyncio.run(main())
