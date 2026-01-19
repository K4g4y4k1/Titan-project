import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import re
import sys
import numpy as np
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from aiohttp import web
import aiohttp_cors

# --- CONFIGURATION ---
load_dotenv()
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")
API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN', "change_me_immediately")

CONFIG = {
    "VERSION": "7.1.0-Secure",
    "PORT": 8080,
    "DB_PATH": "titan_v7.db",
    "MAX_OPEN_POSITIONS": 5,
    "MARKET_STRESS_THRESHOLD": 1.2,
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "google/gemini-2.0-flash-exp:free"
}

# Configuration robuste des logs pour systemd/journald
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Titan-V7")

# --- PERSISTANCE ---
class TitanDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialisation sécurisée de toutes les tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Table des trades
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, qty REAL, entry_price REAL, exit_price REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, exit_time DATETIME,
                    result TEXT, confidence INTEGER, thesis TEXT, mode TEXT,
                    tp_price REAL, sl_price REAL, status TEXT DEFAULT 'OPEN'
                )
            """)
            # Table des stats journalières
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    day DATE PRIMARY KEY, 
                    start_equity REAL, 
                    risk_consumed REAL DEFAULT 0
                )
            """)
            # Table d'état (Correction Bug 1)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY, 
                    value TEXT
                )
            """)
            # Valeurs par défaut
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('is_halted', '0')")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('halt_reason', '')")
            conn.commit()

    def get_symbol_shadow_performance(self, symbol):
        """Calcule le winrate récent en Shadow pour promotion Live"""
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN result='TP' THEN 1 ELSE 0 END) 
                FROM trades WHERE symbol=? AND mode='SHADOW' AND status='CLOSED'
                ORDER BY exit_time DESC LIMIT 10
            """, (symbol,)).fetchone()
            total, wins = res if (res and res[0]) else (0, 0)
            winrate = (wins / total) if total > 0 else 0
            return total, winrate

    def log_trade_start(self, symbol, qty, price, conf, thesis, mode, tp, sl):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status) 
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN'))
            conn.commit()

# --- MOTEUR TITAN ---
class TitanEngine:
    def __init__(self):
        self.alpaca_key = os.getenv('ALPACA_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET')
        base_url = "https://paper-api.alpaca.markets" if CONFIG["ENV_MODE"] == 'PAPER' else "https://api.alpaca.markets"
        self.alpaca = tradeapi.REST(self.alpaca_key, self.alpaca_secret, base_url)
        self.db = TitanDatabase(CONFIG["DB_PATH"])
        self.session = None
        self.status = {
            "version": CONFIG["VERSION"],
            "state": "INIT",
            "market": "CLOSED",
            "equity": {"current": 0.0, "pnl_pct": 0.0},
            "positions": {"live": 0, "shadow_open": 0}
        }

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def calculate_atr(self, symbol, period=14):
        """Calcul robuste de l'ATR sans dépendance pandas/df (Correction Bug 2)"""
        try:
            bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=30)
            if not bars or len(bars) < period:
                return None
            
            # Utilisation de numpy pour la rapidité
            highs = np.array([b.h for b in bars])
            lows = np.array([b.l for b in bars])
            closes = np.array([b.c for b in bars])
            
            # Calcul simplifié du True Range (High - Low) sur les 30 dernières minutes
            tr = highs - lows
            atr = np.mean(tr[-period:])
            return float(atr)
        except Exception as e:
            logger.error(f"Erreur ATR {symbol}: {e}")
            return None

    async def get_risk_params(self, confidence, price, atr, equity):
        """Calcul du sizing basé sur le risque réel (Audit Axe 1)"""
        if confidence < 85: return None
        
        # Mapping Confiance -> % Risque de l'Equity
        risk_map = [(95, 0.02), (92, 0.015), (88, 0.01), (85, 0.005)]
        risk_pct = next(r for c, r in risk_map if confidence >= c)
        
        # SL = 1.5 * ATR (Distance sécurisée)
        sl_dist = 1.5 * atr
        if sl_dist <= 0: return None
        
        sl_price = price - sl_dist
        tp_price = price + (2.5 * atr) # RR ~1.66
        
        risk_amount = equity * risk_pct
        qty = int(risk_amount / sl_dist)
        
        # Correction Bug 3 : Eviter qty=0 ou ordres impossibles
        if qty < 1:
            logger.warning(f"Sizing rejeté : Qty calculated to 0 (Risk: ${round(risk_amount, 2)}, SL Dist: {round(sl_dist, 4)})")
            return None
            
        return {
            "qty": qty, 
            "sl": round(sl_price, 2), 
            "tp": round(tp_price, 2), 
            "risk_pct": risk_pct,
            "sl_dist": sl_dist
        }

    async def run_trading_logic(self):
        """Boucle principale de décision"""
        try:
            acc = self.alpaca.get_account()
            equity = float(acc.equity)
            self.status["equity"]["current"] = equity
            
            picks = await self.fetch_ai_picks()
            if not picks.get("picks"):
                logger.info("SCAN | Aucun signal IA valide.")
                return

            for p in picks["picks"]:
                symbol = p['symbol'].upper()
                conf = p['confidence']
                
                atr = self.calculate_atr(symbol)
                if not atr: continue
                
                # Récupération du prix actuel
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
                if not bars: continue
                price = bars[0].c
                
                params = await self.get_risk_params(conf, price, atr, equity)
                if not params: continue
                
                # Check Shadow Performance pour promotion
                total_sh, wr_sh = self.db.get_symbol_shadow_performance(symbol)
                is_live_ready = (total_sh >= 3 and wr_sh >= 0.60)
                
                mode = "LIVE" if (is_live_ready and len(self.alpaca.list_positions()) < CONFIG["MAX_OPEN_POSITIONS"]) else "SHADOW"
                
                if mode == "LIVE":
                    logger.info(f"🔥 LIVE ORDER | {symbol} | Qty: {params['qty']} | Conf: {conf}%")
                    self.alpaca.submit_order(
                        symbol=symbol, qty=params['qty'], side='buy', type='market', time_in_force='gtc',
                        order_class='bracket', 
                        take_profit={'limit_price': params['tp']}, 
                        stop_loss={'stop_price': params['sl']}
                    )
                else:
                    logger.info(f"💡 SHADOW | {symbol} | Conf: {conf}% | Reason: {'Live Limit' if is_live_ready else 'Not Promoted'}")
                
                self.db.log_trade_start(symbol, params['qty'], price, conf, p.get('reason', 'v7.1'), mode, params['tp'], params['sl'])
                
        except Exception as e:
            logger.error(f"Erreur Trading Logic: {e}")

    async def fetch_ai_picks(self):
        """Interface avec OpenRouter/Gemini"""
        s = await self.get_session()
        try:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={
                    "model": CONFIG["AI_MODEL"], 
                    "messages": [{"role": "user", "content": "Return 5 US tickers with momentum in JSON: {'picks': [{'symbol': '...', 'confidence': 92, 'reason': '...'}]}"}]
                }) as r:
                d = await r.json()
                content = d['choices'][0]['message']['content']
                match = re.search(r"\{.*\}", content, re.DOTALL)
                return json.loads(match.group(0)) if match else {"picks": []}
        except Exception as e:
            logger.error(f"AI Fetch Error: {e}")
            return {"picks": []}

    async def main_loop(self):
        """Boucle infinie robuste avec flush des logs"""
        logger.info(f"Démarrage du coeur Titan {CONFIG['VERSION']}...")
        while True:
            try:
                clock = self.alpaca.get_clock()
                self.status["market"] = "OPEN" if clock.is_open else "CLOSED"
                
                if clock.is_open:
                    self.status["state"] = "SCANNING"
                    await self.run_trading_logic()
                else:
                    self.status["state"] = "SLEEPING"
                
                # Heartbeat
                logger.info(f"HEARTBEAT | State: {self.status['state']} | Equity: ${self.status['equity']['current']}")
                
                # Force le vidage du buffer pour journalctl
                sys.stdout.flush()
                
                await asyncio.sleep(600) # Scan toutes les 10 min
            except Exception as e:
                logger.error(f"Global Loop Error: {e}")
                sys.stdout.flush()
                await asyncio.sleep(60)

# --- API ---
async def api_status(request): 
    return web.json_response(request.app['titan'].status)

@web.middleware
async def cors_middleware(request, handler):
    # Réponse aux preflight OPTIONS
    if request.method == "OPTIONS":
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Max-Age": "86400",
            },
        )

    response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"

    return response

async def main():
    titan = TitanEngine()
    app = web.Application(middlewares=[cors_middleware])
    app['titan'] = titan
    app.router.add_get('/status', api_status)
    
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=False, 
            expose_headers="*", 
            allow_headers=["*"], 
            allow_methods=["GET", "POST", "OPTIONS"])
    })
    for r in list(app.router.routes()): cors.add(r)
    
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', CONFIG["PORT"]).start()
    
    logger.info(f"API Serveur Titan V7.1 actif sur port {CONFIG['PORT']}")
    sys.stdout.flush()
    
    # Entrée dans la boucle de trading
    await titan.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
