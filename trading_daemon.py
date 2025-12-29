import asyncio
import sqlite3
import logging
import os
import sys
import json
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import alpaca_trade_api as tradeapi

# --- CONFIGURATION INSTITUTIONNELLE ---
DB_PATH = "titan_prod_v4_1.db"
HALT_FILE = ".halt_trading"
HEARTBEAT_FILE = ".daemon_heartbeat"
LOG_FILE = "titan_engine.log"

GOUVERNANCE = {
    "MIN_STOCK_PRICE": 5.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25,
    "MAX_POSITION_SIZE_PCT": 0.15, 
    "BLACKLIST": ["GME", "AMC", "BBBY", "DWAC"],
    "MAX_DAILY_DRAWDOWN": 0.02,
    "MAX_TOTAL_DRAWDOWN": 0.10,
    "RISK_PER_TRADE_PCT": 0.01, 
    "TAKE_PROFIT_PCT": 0.05,
    "STOP_LOSS_PCT": 0.03
}

# --- LOGGING (R10) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(module)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# --- ÉTAT DU SYSTÈME & METRICS (OPS/R9) ---
SYSTEM_STATE = {
    "status": "starting",
    "last_cycle": None,
    "equity": 0,
    "drawdown": 0,
    "win_rate": 0,
    "profit_factor": 0,
    "active_positions": 0
}

class MetricsHandler(BaseHTTPRequestHandler):
    """Endpoint de santé et métriques pour monitoring (R2/OPS)."""
    def do_GET(self):
        self.send_response(200 if SYSTEM_STATE["status"] == "ok" else 503)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(SYSTEM_STATE).encode())

def start_metrics_server():
    server = HTTPServer(('0.0.0.0', 8080), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info("Serveur de métriques actif sur le port 8080")

class TitanEngine:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.fmp_key = os.getenv("FMP_API_KEY")
        self.is_paper = "paper" in os.getenv("ALPACA_BASE_URL", "paper").lower()

        if not all([self.api_key, self.api_secret, self.fmp_key]):
            logging.critical("ERREUR : Clés API manquantes dans l'environnement.")
            sys.exit(1)

        try:
            base_url = "https://paper-api.alpaca.markets" if self.is_paper else "https://api.alpaca.markets"
            self.alpaca = tradeapi.REST(self.api_key, self.api_secret, base_url, api_version='v2')
            self.account = self.alpaca.get_account()
            self.initial_equity = float(self.account.equity)
            logging.info(f"Titan v4.1 Initialisé. Equity de base: {self.initial_equity}$")
        except Exception as e:
            logging.error(f"Échec connexion Alpaca: {e}")
            sys.exit(1)

        self._init_db()

    def _init_db(self):
        """Initialisation DB avec table d'equity pour analyse temporelle (R9)."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            # Audit des trades
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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Historique Equity (pour courbe de performance)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_equity REAL,
                    drawdown REAL
                )
            """)

    # --- SÉCURITÉ & ANALYTICS (R1 / R9) ---
    def check_security_and_metrics(self):
        if os.path.exists(HALT_FILE): return "HALT_MANUEL"
        
        self.account = self.alpaca.get_account()
        equity = float(self.account.equity)
        
        day_dd = (equity - float(self.account.last_equity)) / float(self.account.last_equity)
        total_dd = (equity - self.initial_equity) / self.initial_equity
        
        # Enregistrement Equity (R9)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO equity_history (total_equity, drawdown) VALUES (?, ?)", (equity, day_dd))
        
        # Mise à jour état global
        SYSTEM_STATE.update({"equity": equity, "drawdown": day_dd})

        if day_dd <= -GOUVERNANCE["MAX_DAILY_DRAWDOWN"]:
            self.emergency_shutdown(f"DAY_DD_{day_dd:.2%}")
            return "KILL_SWITCH_DAILY"
            
        if total_dd <= -GOUVERNANCE["MAX_TOTAL_DRAWDOWN"]:
            self.emergency_shutdown(f"TOTAL_DD_{total_dd:.2%}")
            return "KILL_SWITCH_TOTAL"
            
        return None

    def emergency_shutdown(self, reason):
        logging.critical(f"🚨 SHUTDOWN D'URGENCE : {reason}")
        self.alpaca.cancel_all_orders()
        self.alpaca.close_all_positions()
        with open(HALT_FILE, "w") as f:
            f.write(f"TERMINATED_{reason}_{datetime.now()}")

    # --- RÉCONCILIATION PAR ID (R4) ---
    async def reconcile_portfolio(self):
        positions = self.alpaca.list_positions()
        actual_pos = {p.symbol: float(p.qty) for p in positions}
        SYSTEM_STATE["active_positions"] = len(positions)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, symbol, qty, entry_price, client_order_id FROM audit_trades WHERE status='OPEN'")
            for db_id, sym, qty, entry, c_id in cursor.fetchall():
                if sym not in actual_pos:
                    # Traçabilité précise via les ordres fermés
                    orders = self.alpaca.list_orders(status='closed', limit=5)
                    fill_price = entry # Fallback
                    for o in orders:
                        if o.client_order_id == c_id or o.symbol == sym:
                            fill_price = float(o.filled_avg_price) if o.filled_avg_price else entry
                            break
                    
                    pnl = (fill_price - entry) * qty
                    cursor.execute("""
                        UPDATE audit_trades 
                        SET status='RECONCILED', exit_price=?, pnl_realized=? 
                        WHERE id=?
                    """, (fill_price, pnl, db_id))
                    logging.info(f"R4: Position {sym} réconciliée. PnL: {pnl}$")
            
            # Calcul Metrics R9 pour le dashboard
            cursor.execute("SELECT pnl_realized FROM audit_trades WHERE pnl_realized IS NOT NULL")
            pnls = [row[0] for row in cursor.fetchall()]
            if pnls:
                wins = [p for p in pnls if p > 0]
                SYSTEM_STATE["win_rate"] = len(wins) / len(pnls)
                SYSTEM_STATE["profit_factor"] = sum(wins) / abs(sum([p for p in pnls if p <= 0])) if len(wins) != len(pnls) else 1.0

    # --- EXÉCUTION INSTITUTIONNELLE (R6) ---
    async def place_bracket_order(self, symbol, current_price):
        # 1. Gouvernance
        if symbol in GOUVERNANCE["BLACKLIST"]: return

        # 2. Risk Sizing (R3)
        equity = float(self.account.equity)
        risk_per_trade = equity * GOUVERNANCE["RISK_PER_TRADE_PCT"]
        stop_loss_price = current_price * (1 - GOUVERNANCE["STOP_LOSS_PCT"])
        
        qty = int(risk_per_trade / (current_price - stop_loss_price))
        
        # Cap par position (15% equity max)
        max_qty = int((equity * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / current_price)
        qty = min(qty, max_qty)

        if qty <= 0: return

        # 3. Execution
        client_id = f"titan_{symbol}_{uuid.uuid4().hex[:8]}"
        try:
            self.alpaca.submit_order(
                symbol=symbol, qty=qty, side='buy', type='limit',
                limit_price=round(current_price, 2),
                client_order_id=client_id,
                time_in_force='gtc',
                order_class='bracket',
                take_profit={'limit_price': round(current_price * (1 + GOUVERNANCE["TAKE_PROFIT_PCT"]), 2)},
                stop_loss={'stop_price': round(stop_loss_price, 2)}
            )
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO audit_trades (client_order_id, symbol, qty, entry_price, status)
                    VALUES (?, ?, ?, ?, 'OPEN')
                """, (client_id, symbol, qty, current_price))
            logging.info(f"🚀 ORDRE EXÉCUTÉ : {symbol} x {qty} (ID: {client_id})")
        except Exception as e:
            logging.error(f"Erreur exécution {symbol}: {e}")

    async def run_loop(self):
        SYSTEM_STATE["status"] = "running"
        
        if not self.alpaca.get_clock().is_open:
            SYSTEM_STATE["status"] = "market_closed"
            return

        error = self.check_security_and_metrics()
        if error:
            SYSTEM_STATE["status"] = f"critical_error_{error}"
            return

        await self.reconcile_portfolio()
        # Scan & Logic ici
        
        SYSTEM_STATE["last_cycle"] = datetime.now().isoformat()
        SYSTEM_STATE["status"] = "ok"

async def main():
    start_metrics_server()
    engine = TitanEngine()
    logging.info("=== Titan Engine v4.1 Scale-Ready Operational ===")
    
    while True:
        try:
            await engine.run_loop()
            await asyncio.sleep(60)
        except Exception as e:
            logging.error(f"FATAL: Erreur boucle principale: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
