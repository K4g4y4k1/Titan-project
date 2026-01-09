import asyncio
import sqlite3
import logging
import logging.handlers
import os
import json
import threading
import pandas as pd
import io
import hmac
import aiohttp
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import alpaca_trade_api as tradeapi

# --- CONFIGURATION v9.0.0 "UNSTOPPABLE-ALPHA" ---
DB_PATH = "titan_prod_v9.db"
LOG_FILE = "titan_unstoppable.log"

DASHBOARD_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN", "alpha_force_2026")
ENV_MODE = os.getenv("ENV_MODE", "PAPER")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_KEY") 
OR_KEY = os.getenv("OPENROUTER_API_KEY")

AI_MODEL = "x-ai/grok-4.1-fast" 

GOUVERNANCE = {
    "MIN_NOTIONAL_ABS": 500.0,        # Seuil minimal pour l'exécution technique
    "MAX_SECTOR_EXPOSURE_PCT": 0.25, 
    "MAX_POSITION_SIZE_PCT": 0.06,    
    "DRAWDOWN_HARD_PCT": 0.040,       
    
    # BRD-13: Risk Floor Absolu (Empêche la stérilisation par calcul)
    "ABS_MIN_RISK_VAL_USD": 50.0,     # On accepte de perdre au moins 50$ par trade quoi qu'il arrive
    
    "MODES": {
        "EXPLOITATION": { "MIN_SCORE": 85, "BASE_RISK": 0.015 },
        "EXPLORATION": { "MIN_SCORE": 72, "BASE_RISK": 0.008 },
        "PROBE": { "MIN_SCORE": 60, "BASE_RISK": 0.003 }
    },
    
    "DATA_CONFIDENCE_WEIGHTS": {
        "HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3
    },
    
    # BRD-14: Quota de sondes
    "DAILY_PROBE_QUOTA": 2, 
    
    # BRD-15: Shadow Mode Thresholds
    "SHADOW_GAP_MIN": 0.035,
    "SHADOW_RVOL_MIN": 2.5
}

SYSTEM_STATE = {
    "status": "active",
    "equity": 0.0,
    "mode_shadow_active": False,
    "daily_stats": {
        "detected": 0,
        "executed": 0,
        "forced_probes": 0,
        "shadow_found": 0
    },
    "health": {"alpaca": "ok", "av": "ok", "ai": "ok"}
}

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("TitanApex")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=30*1024*1024, backupCount=10)
            fh.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(fh)

    def log(self, event_type, **kwargs):
        payload = {"ts": datetime.now().isoformat(), "event": event_type, **kwargs}
        self.logger.info(json.dumps(payload))

SLOG = StructuredLogger()

class MetricsServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return
    def do_GET(self):
        if not hmac.compare_digest(self.headers.get('Authorization', ""), f"Bearer {DASHBOARD_TOKEN}"):
            self.send_response(401); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(SYSTEM_STATE).encode())

class TitanEngine:
    def __init__(self):
        base_url = "https://api.alpaca.markets" if ENV_MODE == "LIVE" else "https://paper-api.alpaca.markets"
        self.alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, symbol TEXT, qty REAL, mode TEXT, forced INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")

    async def get_market_discovery(self, session):
        """BRD-15: Shadow Mode Discovery (Multi-source)"""
        symbols = []
        
        # Source 1: Alpha Vantage (Standard)
        try:
            async with session.get(f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={AV_KEY}") as resp:
                if resp.status == 200:
                    df = pd.read_csv(io.BytesIO(await resp.read()))
                    symbols = df[df['reportDate'] == datetime.now().strftime('%Y-%m-%d')]['symbol'].tolist()
                    SYSTEM_STATE["health"]["av"] = "ok"
        except:
            SYSTEM_STATE["health"]["av"] = "degraded"

        # Source 2: Shadow Mode (Alpaca Snapshots pour les Top Movers)
        # On scanne un univers large pour trouver des anomalies de Gap/Volume sans calendrier
        if not symbols or len(symbols) < 3:
            SYSTEM_STATE["mode_shadow_active"] = True
            # Note: En prod réelle, on utiliserait une liste de 500 symboles liquides
            # Ici on simule par les actifs les plus actifs du jour via Alpaca
            try:
                assets = self.alpaca.list_assets(status='active', asset_class='us_equity')
                # Limitation pour l'exemple: on prend un échantillon
                sample = [a.symbol for a in assets if a.tradable and a.shortable][:200]
                snapshots = self.alpaca.get_snapshots(sample)
                for sym, snap in snapshots.items():
                    gap = (snap.latest_trade.price - snap.prev_daily_bar.close) / snap.prev_daily_bar.close
                    if abs(gap) > GOUVERNANCE["SHADOW_GAP_MIN"]:
                        symbols.append(sym)
                        SYSTEM_STATE["daily_stats"]["shadow_found"] += 1
            except Exception as e:
                SLOG.log("shadow_discovery_failed", error=str(e))
        else:
            SYSTEM_STATE["mode_shadow_active"] = False
            
        return list(set(symbols))

    async def execute_trade(self, sym, ctx, alpha, mode, is_forced=False):
        """BRD-13: Execution avec Floor de Risque Absolu"""
        try:
            equity = SYSTEM_STATE["equity"]
            
            # Calcul du risque de base
            base_risk_val = equity * GOUVERNANCE["MODES"][mode]["BASE_RISK"]
            
            # Application de la pénalité data
            conf_weight = GOUVERNANCE["DATA_CONFIDENCE_WEIGHTS"].get(ctx['conf'], 0.3)
            calculated_risk = base_risk_val * conf_weight
            
            # --- BRD-13: Risk Floor ---
            # On ne laisse pas le risque descendre sous un montant qui rendrait le trade impossible
            final_risk_val = max(calculated_risk, GOUVERNANCE["ABS_MIN_RISK_VAL_USD"])
            
            # Sizing par volatilité (ATR)
            sl_dist = ctx['price'] * max(ctx['atr_pct'], 0.02) * 1.5
            qty = int(final_risk_val / sl_dist) if sl_dist > 0 else 0
            
            # Vérification Notional
            notional = qty * ctx['price']
            if notional < GOUVERNANCE["MIN_NOTIONAL_ABS"]:
                # Si on est en mode FORCED ou PROBE, on ajuste au minimum notional
                qty = int(GOUVERNANCE["MIN_NOTIONAL_ABS"] / ctx['price']) + 1
                notional = qty * ctx['price']

            # Hard Cap Equity
            if notional > (equity * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]):
                qty = int((equity * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / ctx['price'])

            if qty <= 0: return False

            # Submission
            self.alpaca.submit_order(
                symbol=sym, qty=qty, side='buy', type='limit',
                limit_price=round(ctx['price'] * 1.002, 2),
                time_in_force='day', order_class='bracket',
                take_profit={'limit_price': round(ctx['price'] + (sl_dist * 2.3), 2)},
                stop_loss={'stop_price': round(ctx['price'] - sl_dist, 2)}
            )
            
            SYSTEM_STATE["daily_stats"]["executed"] += 1
            if is_forced: SYSTEM_STATE["daily_stats"]["forced_probes"] += 1
            
            SLOG.log("trade_executed", symbol=sym, mode=mode, forced=is_forced, notional=notional)
            return True
        except Exception as e:
            SLOG.log("exec_failure", symbol=sym, error=str(e))
            return False

    async def run_cycle(self):
        try:
            acc = self.alpaca.get_account()
            SYSTEM_STATE["equity"] = float(acc.equity)
            if float(acc.equity) < float(acc.last_equity) * (1 - GOUVERNANCE["DRAWDOWN_HARD_PCT"]):
                SYSTEM_STATE["status"] = "HALTED"; return

            async with aiohttp.ClientSession() as session:
                # 1. Discovery (AV + Shadow BRD-15)
                symbols = await self.get_market_discovery(session)
                SYSTEM_STATE["daily_stats"]["detected"] = len(symbols)
                
                candidates_for_quota = []

                for sym in symbols:
                    # 2. Context & AI (Repris de v8)
                    # [Simulé pour la clarté : ctx = get_resilient_context, alpha = get_ai_score]
                    # On suppose ici qu'on récupère ctx et alpha via les méthodes v8
                    # Pour la v9, on simule un candidat viable
                    ctx = {"price": 150.0, "atr_pct": 0.03, "conf": "LOW"} 
                    alpha = {"score": 65, "sigma": 20} # Un score qui normalement "bloquerait"
                    
                    mode = "PROBE" if alpha['score'] >= 60 else None
                    if mode:
                        success = await self.execute_trade(sym, ctx, alpha, mode)
                        if not success:
                            candidates_for_quota.append((sym, ctx, alpha))

                # 3. BRD-14: Guaranteed Probe Quota
                # Si on n'a rien exécuté mais qu'on a des candidats, on en force un
                if SYSTEM_STATE["daily_stats"]["executed"] == 0 and candidates_for_quota:
                    # On prend le meilleur candidat (ici le premier pour l'exemple)
                    best_sym, best_ctx, best_alpha = candidates_for_quota[0]
                    await self.execute_trade(best_sym, best_ctx, best_alpha, "PROBE", is_forced=True)

        except Exception as e:
            SLOG.log("critical_error", error=str(e))

async def main():
    threading.Thread(target=HTTPServer(('0.0.0.0', 8080), MetricsServer).serve_forever, daemon=True).start()
    engine = TitanEngine()
    SLOG.log("system_online", version="v9.0.0-Apex")
    while True:
        await engine.run_cycle()
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
