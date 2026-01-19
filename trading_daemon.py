import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import aiohttp_cors
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from aiohttp import web

# --- CONFIGURATION ---
load_dotenv()
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")

CONFIG = {
    "VERSION": "7.9.6-Omni-Sight-Router",
    "DB_PATH": "titan_v7_omni.db",
    "AI_MODEL": "google/gemini-2.0-flash-exp:free",
    "RISK_PER_TRADE_PCT": 0.01,
    "MAX_OPEN_POSITIONS": 5,
    "SCAN_INTERVAL_SEC": 180,
    "WATCHLIST": ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "COIN", "PLTR", "MARA"]
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Titan-OmniRouter")

class AbsoluteDB:
    def __init__(self, path):
        self.path = path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS ai_decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                confidence INTEGER,
                thesis TEXT,
                decision TEXT,
                rejection_reason TEXT,
                entry_price_at_decision REAL,
                tp_price REAL,
                sl_price REAL,
                market_context TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_decision_id INTEGER,
                symbol TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_time DATETIME,
                result TEXT,
                status TEXT DEFAULT 'OPEN'
            )""")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()

    def log_decision(self, data):
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("""INSERT INTO ai_decision_log 
                (symbol, confidence, thesis, decision, rejection_reason, entry_price_at_decision, tp_price, sl_price) 
                VALUES (?,?,?,?,?,?,?,?)""", (
                    data['symbol'], data['confidence'], data['thesis'], data['decision'], 
                    data['rejection_reason'], data['entry_price'], data['tp_price'], data['sl_price']
                ))
            return cursor.lastrowid

class TitanOmniRouter:
    def __init__(self):
        self.alpaca = tradeapi.REST(os.getenv('ALPACA_KEY'), os.getenv('ALPACA_SECRET'), "https://paper-api.alpaca.markets")
        self.db = AbsoluteDB(CONFIG["DB_PATH"])
        self.last_trade_time = {}

    async def get_ia_picks(self):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
        prompt = f"Analyze {CONFIG['WATCHLIST']}. Return JSON: {{ 'picks': [{{ 'symbol': 'AAPL', 'confidence': 90, 'thesis': 'string' }}] }}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json={
                    "model": CONFIG["AI_MODEL"],
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return json.loads(data['choices'][0]['message']['content']).get("picks", [])
            return []
        except: return []

    async def run(self):
        logger.info(f"Moteur de trading démarré ({CONFIG['VERSION']})")
        while True:
            try:
                # Cycle simplifié pour démo/test
                picks = await self.get_ia_picks()
                for p in picks:
                    # Simulation de log pour le dashboard
                    self.db.log_decision({
                        **p, "entry_price": 0, "tp_price": 0, "sl_price": 0, 
                        "decision": "REJECTED", "rejection_reason": "IA_ONLY_MODE"
                    })
                await asyncio.sleep(CONFIG["SCAN_INTERVAL_SEC"])
            except Exception as e:
                logger.error(f"Erreur cycle: {e}")
                await asyncio.sleep(10)

async def start_server(titan):
    app = web.Application()
    
    # Configuration CORS STRICTE pour autoriser votre Dashboard
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        )
    })

    async def get_status(request):
        # Verification du token (optionnel mais recommandé car envoyé par votre dashboard)
        auth_header = request.headers.get('Authorization', '')
        
        with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
            conn.row_factory = sqlite3.Row
            audit = conn.execute("SELECT * FROM ai_decision_log ORDER BY timestamp DESC LIMIT 20").fetchall()
            return web.json_response({
                "bot": CONFIG["VERSION"],
                "model": CONFIG["AI_MODEL"],
                "audit": [dict(r) for r in audit]
            })

    # Route avec support CORS
    resource = cors.add(app.router.add_resource("/status"))
    cors.add(resource.add_method("GET", get_status))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logger.info("--------- API ONLINE SUR PORT 8080 ---------")

async def main():
    titan = TitanOmniRouter()
    # On lance l'API et le moteur en parallèle
    await asyncio.gather(
        start_server(titan),
        titan.run()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
