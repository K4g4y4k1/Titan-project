import asyncio
import sqlite3
import logging
import os
import sys
import json
import uuid
import statistics
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import alpaca_trade_api as tradeapi
import aiohttp

# --- CONFIGURATION DE RÉSILIENCE v4.5 "SENTINEL-ELITE" ---
DB_PATH = "titan_prod_v4_5.db"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"
LOG_FILE = "titan_engine.log"

# VERROUILLAGE DE SÉCURITÉ MULTI-FACTEURS (Audit v4.4)
ENV_MODE = "PAPER"  # "PAPER" ou "LIVE"
LIVE_AFFIRMATION = False # DOIT être True pour le mode LIVE réel

GOUVERNANCE = {
    "ENV_MODE": ENV_MODE,
    "MIN_STOCK_PRICE": 5.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.20,
    "MAX_POSITION_SIZE_PCT": 0.10,
    "BLACKLIST": ["GME", "AMC", "BBBY", "DJT", "SPCE", "FFIE", "LUMN"],
    "MAX_DAILY_DRAWDOWN_PCT": 0.02,
    "MAX_TOTAL_DRAWDOWN_PCT": 0.10,
    "DAILY_LOSS_BUDGET_USD": 2000,    # Limite monétaire (Audit v4.4)
    "BASE_RISK_PER_TRADE_PCT": 0.01,
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "MIN_SL_PCT": 0.015,              # Plancher SL (Audit v4.4)
    "MAX_SL_PCT": 0.05,               # Plafond SL
    "SLIPPAGE_PROTECTION": 0.002,
    # DISJONCTEURS COMPORTEMENTAUX
    "MAX_TRADES_PER_DAY": 5,
    "AI_DISPERSION_THRESHOLD": 25,    # Écart-type max des votes
    "MAX_CONSECUTIVE_LOSSES": 3,
    "COOLDOWN_HOURS": 4               # Durée réelle de pause (Audit v4.4)
}

AI_MODELS = {
    "The_Strategist": "anthropic/claude-3.5-sonnet",
    "The_RiskManager": "openai/gpt-4o",
    "The_Visionary": "google/gemini-pro-1.5"
}

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(module)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

SYSTEM_STATE = {
    "status": "starting",
    "mode": ENV_MODE,
    "last_cycle": None,
    "equity": 0,
    "daily_pnl_usd": 0,
    "trades_today": 0,
    "consecutive_losses": 0,
    "cooldown_until": None, # Horodatage de fin de pause
    "active_positions": 0,
    "engine_version": "4.5.0"
}

# --- INITIALISATION DB ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_order_id TEXT UNIQUE,
                symbol TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                status TEXT, 
                pnl_realized REAL,
                ai_champion TEXT,
                ai_consensus_score REAL,
                ai_dispersion REAL,
                sector TEXT,
                risk_scaling REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                model_name TEXT,
                vote_score INTEGER,
                reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(trade_id) REFERENCES audit_trades(id)
            )
        """)

# --- SERVEUR DE MÉTRIQUES ---
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        is_critical = "critical" in SYSTEM_STATE["status"]
        self.send_response(503 if is_critical else 200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        payload = {
            "server_time": datetime.now().isoformat(),
            "metrics": {**SYSTEM_STATE, "cooldown_until": str(SYSTEM_STATE["cooldown_until"])}
        }
        self.wfile.write(json.dumps(payload, indent=2).encode())
    def log_message(self, format, *args): return

def start_metrics_server():
    server = HTTPServer(('0.0.0.0', 8080), MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

# --- LE COLISÉE IA ---
class AIColosseum:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def _consult_oracle(self, session, model_key, model_id, context):
        prompt = f"""
        ANALYSE PEAD: {context['symbol']} ({context['sector']}). 
        Surprise EPS: {context['eps']:.1%}, Réaction J0: {context['j0']:.1%}.
        MISSION: Score 0-100 et thèse.
        FORMAT JSON: {{"score": 0-100, "reason": "..."}}
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with session.post(self.url, headers=headers, json={
                "model": model_id, 
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }, timeout=15) as r:
                if r.status == 200:
                    res = await r.json()
                    return {"model": model_key, **json.loads(res['choices'][0]['message']['content'])}
        except: return {"model": model_key, "score": 0, "reason": "Error"}

        return {"model": model_key, "score": 0, "reason": "Invalid"}

    async def get_consensus(self, session, context):
        tasks = [self._consult_oracle(session, k, v, context) for k, v in AI_MODELS.items()]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r['score'] > 0]
        if len(valid) < 2: return 0, 0, None, []
        scores = [v['score'] for v in valid]
        avg = sum(scores) / len(scores)
        sigma = statistics.stdev(scores) if len(scores) > 1 else 0
        return avg, sigma, max(valid, key=lambda x: x['score']), valid

# --- MOTEUR TITAN v4.5 ---
class TitanEngine:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.fmp_key = os.getenv("FMP_API_KEY")
        
        # DOUBLE VERIFICATION LIVE (Audit Critique 1)
        if GOUVERNANCE["ENV_MODE"] == "LIVE" and not LIVE_AFFIRMATION:
            logging.critical("🚨 ERREUR SÉCURITÉ: Mode LIVE sans affirmation explicite.")
            sys.exit(1)
        
        self.base_url = "https://api.alpaca.markets" if GOUVERNANCE["ENV_MODE"] == "LIVE" else "https://paper-api.alpaca.markets"
        
        try:
            self.alpaca = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
            self.account = self.alpaca.get_account()
            self.initial_equity = float(self.account.equity)
            init_db()
        except Exception as e:
            logging.error(f"Erreur Init Broker: {e}"); sys.exit(1)

        self.colosseum = AIColosseum(os.getenv("OPENROUTER_API_KEY"))

    def check_resilience(self):
        """Vérifie les barrières de sécurité (Audit v4.4)."""
        if os.path.exists(HALT_FILE): return "MANUAL_HALT"
        
        self.account = self.alpaca.get_account()
        equity = float(self.account.equity)
        day_pnl_usd = equity - float(self.account.last_equity)
        day_pnl_pct = day_pnl_usd / float(self.account.last_equity)
        total_pnl_pct = (equity - self.initial_equity) / self.initial_equity
        
        SYSTEM_STATE.update({
            "equity": equity, 
            "drawdown": day_pnl_pct,
            "daily_pnl_usd": day_pnl_usd
        })

        # 1. Kill Switch Drawdown %
        if day_pnl_pct <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN_PCT"]:
            self.liquidate_all("DAILY_DRAWDOWN_LIMIT")
            return "KILL_DAILY_PCT"
        
        # 2. Kill Switch Budget USD (Audit v4.4)
        if day_pnl_usd <= -GOUVERNANCE["DAILY_LOSS_BUDGET_USD"]:
            self.liquidate_all("DAILY_USD_BUDGET_EXCEEDED")
            return "KILL_DAILY_USD"

        # 3. Cooldown Temporel Réel (Audit v4.4)
        if SYSTEM_STATE["cooldown_until"] and datetime.now() < SYSTEM_STATE["cooldown_until"]:
            return "TEMPORAL_COOLDOWN"

        return None

    def liquidate_all(self, reason):
        logging.critical(f"🛑 LIQUIDATION : {reason}")
        self.alpaca.cancel_all_orders()
        self.alpaca.close_all_positions()
        with open(HALT_FILE, "w") as f: f.write(f"HALT_{reason}_{datetime.now()}")

    async def update_stats(self):
        """Audit R4 et gestion comportementale."""
        positions = self.alpaca.list_positions()
        actual = {p.symbol: float(p.qty) for p in positions}
        SYSTEM_STATE["active_positions"] = len(positions)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, symbol, qty, entry_price, client_order_id FROM audit_trades WHERE status='OPEN'")
            for row in cursor.fetchall():
                t_id, sym, qty, entry, c_id = row
                if sym not in actual:
                    # Fermeture détectée
                    orders = self.alpaca.list_orders(status='closed', limit=10, symbols=[sym])
                    exit_p = entry
                    for o in orders:
                        if o.client_order_id == c_id:
                            exit_p = float(o.filled_avg_price) if o.filled_avg_price else entry
                            break
                    pnl = (exit_p - entry) * qty
                    cursor.execute("UPDATE audit_trades SET status='CLOSED', exit_price=?, pnl_realized=? WHERE id=?", (exit_p, pnl, t_id))
                    
                    # Logic Cooldown (Audit v4.4)
                    if pnl < 0:
                        SYSTEM_STATE["consecutive_losses"] += 1
                        if SYSTEM_STATE["consecutive_losses"] >= GOUVERNANCE["MAX_CONSECUTIVE_LOSSES"]:
                            SYSTEM_STATE["cooldown_until"] = datetime.now() + timedelta(hours=GOUVERNANCE["COOLDOWN_HOURS"])
                            logging.warning(f"❄️ Cooldown activé jusqu'à {SYSTEM_STATE['cooldown_until']}")
                    else:
                        SYSTEM_STATE["consecutive_losses"] = 0
            
            # Count Today
            t_start = datetime.now().strftime("%Y-%m-%d 00:00:00")
            cursor.execute("SELECT COUNT(*) FROM audit_trades WHERE timestamp >= ? AND status != 'REJECTED'", (t_start,))
            SYSTEM_STATE["trades_today"] = cursor.fetchone()[0]
            conn.commit()

    async def execute_trade(self, symbol, price, sector, score, sigma, champion, votes):
        """Exécution Adaptative v4.5 (Audit v4.4)."""
        
        # 1. Réduction Dynamique du Risque (De-risking)
        # Si drawdown total > 5%, on divise le risque par 2
        total_dd = (SYSTEM_STATE["equity"] - self.initial_equity) / self.initial_equity
        risk_scaling = 1.0
        if total_dd < -0.05: risk_scaling = 0.5
        
        risk_pct = GOUVERNANCE["BASE_RISK_PER_TRADE_PCT"] * risk_scaling
        
        # 2. SL/TP Adaptatif avec Bornes (Clamping) (Audit v4.4)
        conviction_factor = (score - 80) / 20 
        
        tp_raw = GOUVERNANCE["BASE_TP_PCT"] + (conviction_factor * 0.04)
        sl_raw = GOUVERNANCE["BASE_SL_PCT"] - (conviction_factor * 0.01)
        
        # Bornes de sécurité
        tp_final = min(max(tp_raw, 0.04), 0.12) # Entre 4% et 12%
        sl_final = min(max(sl_raw, GOUVERNANCE["MIN_SL_PCT"]), GOUVERNANCE["MAX_SL_PCT"])
        
        risk_amt = SYSTEM_STATE["equity"] * risk_pct
        qty = int(risk_amt / (price * sl_final))
        qty = min(qty, int((SYSTEM_STATE["equity"] * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / price))
        
        if qty <= 0: return

        c_id = f"titan_{symbol}_{uuid.uuid4().hex[:8]}"
        try:
            self.alpaca.submit_order(
                symbol=symbol, qty=qty, side='buy', type='limit',
                limit_price=round(price * (1 + GOUVERNANCE["SLIPPAGE_PROTECTION"]), 2),
                client_order_id=c_id, time_in_force='gtc', order_class='bracket',
                take_profit={'limit_price': round(price * (1 + tp_final), 2)},
                stop_loss={'stop_price': round(price * (1 - sl_final), 2)}
            )
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_trades (client_order_id, symbol, qty, entry_price, status, ai_champion, ai_consensus_score, ai_dispersion, sector, risk_scaling)
                    VALUES (?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
                """, (c_id, symbol, qty, price, champion['model'], score, sigma, sector, risk_scaling))
                t_id = cursor.lastrowid
                for v in votes:
                    cursor.execute("INSERT INTO ai_votes (trade_id, model_name, vote_score, reason) VALUES (?, ?, ?, ?)", (t_id, v['model'], v['score'], v['reason']))
                conn.commit()
            logging.info(f"✅ TRADE v4.5: {symbol} (Scaling: {risk_scaling}, SL: {sl_final:.1%})")
        except Exception as e: logging.error(f"Execution Error: {e}")

    async def run_cycle(self):
        with open(HEARTBEAT_FILE, "w") as f: f.write(datetime.now().isoformat())
        
        if not self.alpaca.get_clock().is_open:
            SYSTEM_STATE["status"] = "market_closed"; return
        
        res_error = self.check_resilience()
        if res_error:
            SYSTEM_STATE["status"] = f"critical_{res_error}"; return

        await self.update_stats()
        if SYSTEM_STATE["trades_today"] >= GOUVERNANCE["MAX_TRADES_PER_DAY"]:
            SYSTEM_STATE["status"] = "daily_limit_reached"; return

        SYSTEM_STATE["status"] = "ok"
        
        async with aiohttp.ClientSession() as session:
            # Scanner simulé (Remplacer par FMP Calendar API en production)
            for sym in ["AAPL", "NVDA", "GOOGL"][:2]:
                with sqlite3.connect(DB_PATH) as conn:
                    if conn.execute("SELECT 1 FROM audit_trades WHERE symbol=? AND status='OPEN'", (sym,)).fetchone(): continue
                
                # Fetch Data
                url = f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={self.fmp_key}"
                async with session.get(url) as r:
                    data = (await r.json())[0]
                    price, sector = float(data['price']), data['sector']
                
                avg, sigma, champion, votes = await self.colosseum.get_consensus(session, {'symbol': sym, 'sector': sector, 'eps': 0.1, 'j0': 0.05})
                
                if sigma < GOUVERNANCE["AI_DISPERSION_THRESHOLD"] and avg >= 80:
                    await self.execute_trade(sym, price, sector, avg, sigma, champion, votes)

async def main():
    start_metrics_server()
    engine = TitanEngine()
    logging.info("🏰 Titan Engine v4.5 (Sentinel-Elite) Operational")
    while True:
        try:
            await engine.run_cycle()
            await asyncio.sleep(60)
        except Exception as e:
            logging.error(f"Fatal Loop Error: {e}"); await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
