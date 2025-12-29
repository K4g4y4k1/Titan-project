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
import aiohttp

# --- CONFIGURATION INSTITUTIONNELLE ---
DB_PATH = "titan_prod_v4_2.db"
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

# Modèles en compétition
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

# --- DB INIT ---
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
                ai_champion TEXT, -- Qui a validé ce trade ?
                ai_consensus_score REAL,
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

# --- LE COLISÉE (Multi-IA) ---
class AIColosseum:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def _consult_oracle(self, session, model_key, model_id, context):
        """Interroge un modèle spécifique."""
        prompt = f"""
        Rôle: {model_key}. Analyse l'action {context['symbol']} (Secteur: {context['sector']}).
        Données: EPS Surprise {context['eps']:.1%}, Réaction J0 {context['j0']:.1%}.
        
        Ta mission: Donner un score de conviction (0-100) et une raison.
        Format JSON: {{"score": 0-100, "reason": "..."}}
        """
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": { "type": "json_object" }
        }
        
        try:
            async with session.post(self.url, headers=headers, json=payload, timeout=20) as r:
                if r.status == 200:
                    res = await r.json()
                    content = json.loads(res['choices'][0]['message']['content'])
                    return {
                        "model": model_key,
                        "score": content.get('score', 0),
                        "reason": content.get('reason', 'N/A')
                    }
        except Exception as e:
            logging.error(f"Erreur IA {model_key}: {e}")
        
        return {"model": model_key, "score": 0, "reason": "Error"}

    async def get_consensus(self, session, context):
        """Lance la compétition et renvoie le vainqueur ou le consensus."""
        tasks = [self._consult_oracle(session, k, v, context) for k, v in AI_MODELS.items()]
        results = await asyncio.gather(*tasks)
        
        # Logique de Consensus : Moyenne
        valid_votes = [r for r in results if r['score'] > 0]
        if not valid_votes: return 0, "Aucune IA disponible", []

        avg_score = sum(v['score'] for v in valid_votes) / len(valid_votes)
        
        # Le "Champion" est celui qui a le score le plus haut (pour l'attribution)
        champion = max(valid_votes, key=lambda x: x['score'])
        
        return avg_score, champion['model'], valid_votes

# --- MOTEUR TITAN v4.2 ---
class TitanEngine:
    def __init__(self):
        # ... (Initialisation identique v4.1) ...
        # ... (Connexion Broker & DB) ...
        self.colosseum = AIColosseum(os.getenv("OPENROUTER_API_KEY"))
        init_db()

    # ... (Méthodes Check Security & Reconcile identiques v4.1) ...

    async def process_signal(self, symbol, price, sector, metrics):
        # 1. Gouvernance Quant (R8)
        if symbol in GOUVERNANCE["BLACKLIST"]: return
        
        # 2. Le Colisée (IA)
        context = {"symbol": symbol, "sector": sector, "eps": metrics['eps'], "j0": metrics['j0']}
        
        # On utilise la session aiohttp passée en paramètre ou créée à la volée (simplifié ici)
        async with aiohttp.ClientSession() as session:
            consensus_score, champion_name, votes = await self.colosseum.get_consensus(session, context)

        logging.info(f"🏛️ Consensus pour {symbol}: {consensus_score:.1f}/100 (Champion: {champion_name})")

        # Seuil de déclenchement (Consensus > 75)
        if consensus_score < 75:
            return

        # 3. Exécution
        await self.execute_trade(symbol, price, champion_name, consensus_score, votes)

    async def execute_trade(self, symbol, price, champion, score, votes):
        # ... (Calcul Risk Sizing R3 identique v4.1) ...
        
        # Simulation ordre pour l'exemple
        qty = 10 
        client_id = f"titan_{symbol}_{uuid.uuid4().hex[:8]}"
        
        # ... (Envoi ordre Alpaca) ...

        # Enregistrement enrichi (R9+)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_trades (client_order_id, symbol, qty, entry_price, status, ai_champion, ai_consensus_score)
                VALUES (?, ?, ?, ?, 'OPEN', ?, ?)
            """, (client_id, symbol, qty, price, champion, score))
            
            trade_id = cursor.lastrowid
            
            # Enregistrement des votes individuels
            for v in votes:
                cursor.execute("""
                    INSERT INTO ai_votes (trade_id, model_name, vote_score, reason)
                    VALUES (?, ?, ?, ?)
                """, (trade_id, v['model'], v['score'], v['reason']))
            conn.commit()

    # ... (Reste du code run_loop identique v4.1) ...

# ... (Main loop identique) ...
