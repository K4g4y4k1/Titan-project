import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import re
import math
import statistics
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, APIError
from aiohttp import web
import aiohttp_cors

# --- CONFIGURATION V9.0 (CHIMERA INTEGRATION) ---
load_dotenv()

API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN', 'admin-token')
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")
TITAN_WEBHOOK_URL = os.getenv('TITAN_WEBHOOK_URL', "")

CONFIG = {
    "VERSION": "9.0-Chimera",
    "LEARNING_EPOCH": 2,                
    "PORT": 8080,
    "DB_PATH": "titan_v9_chimera.db",
    
    # --- STRATEGY: SMR-26 SETTINGS ---
    "WATCHLIST": ["SPY", "QQQ", "IWM", "DIA", "AMD", "NVDA"], # Univers de scan
    "SMR_Z_THRESH": -2.0,
    "SMR_RSI_THRESH": 30,
    "SMR_LOOKBACK": 100, # Jours d'historique pour le calcul
    
    # --- RISQUE & CAPITAL ---
    "MAX_OPEN_POSITIONS": 5,            
    "MAX_MACRO_POSITIONS": 1,           
    "RISK_PCT_PER_TRADE": 2.0,        
    "MAX_TOTAL_RISK_PCT": 6.0,
    
    # --- SOLVENCY GUARDS ---
    "MAX_CAPITAL_PER_TRADE_PCT": 20.0, 
    "MAX_TOTAL_EXPOSURE_PCT": 60.0,     

    "LIVE_THRESHOLD": 76,
    "MIN_TRADE_AMOUNT_USD": 150.0,
    "MIN_SL_DISTANCE_USD": 0.05,        
    
    # --- KILL SWITCH 2-STAGES ---
    "MIN_WINRATE_THRESHOLD": 60.0,      
    "PNL_WARN_THRESHOLD": -2.0,         
    "PNL_CRIT_THRESHOLD": -4.0,         
    "WINRATE_LOOKBACK_TRADES": 20,      
    "MARKET_OPEN_BLACKOUT_MIN": 15,     
    
    # --- SCOUT MODE ---
    "ALLOW_SCOUT_TRADE": True,          
    "SCOUT_RISK_FACTOR": 0.5,           

    # --- SHADOW PROMOTION ---
    "SHADOW_PROMOTION_MIN_WINS": 2,     
    "SHADOW_PROMOTION_HIGH_CONF": 80,   
    "SHADOW_PROMOTION_WINDOW_HOURS": 24, 

    "MARKET_STRESS_THRESHOLD": 2.0, # Relaxé légèrement pour le swing
    "COOLDOWN_PER_SYMBOL_MIN": 60,
    "ATR_PERIOD": 14,
    "ALLOW_SHORTS": True,
    
    # --- CALIBRATION REGIMES ---
    "REGIME_CONFIG": {
        "TREND": {"TP_MULT": 3.0, "SL_MULT": 1.5, "DESC": "Trend Following", "TRAILING": True},
        "RANGE": {"TP_MULT": 1.0, "SL_MULT": 1.5, "DESC": "Mean Reversion (SMR)", "TRAILING": False}, # Optimisé pour SMR
        "CHOP":  {"TP_MULT": 0.0, "SL_MULT": 0.0, "DESC": "No trade zone", "TRAILING": False},
        "MACRO": {"TP_MULT": 6.0, "SL_MULT": 4.0, "DESC": "Macro Structural", "TRAILING": False} 
    },
    
    "BE_TRIGGER_ATR_STD": 0.8,
    "TRAILING_STEP_ATR": 0.5,
    "ENABLE_ADAPTIVE_GOVERNOR": True,
    "ENABLE_ADAPTIVE_SHADOW": False,
    "MAX_DAILY_DRAWDOWN_PCT": -4.0,
    "MAX_LOSSES_PER_SYMBOL": 2,
    "MAX_SL_UPDATES_PER_TRADE": 20,
    "HEARTBEAT_INTERVAL_MIN": 60,
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "openai/gpt-4o", # Modèle plus rapide requis pour Veto
    "SCAN_INTERVAL": 60 
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("titan_v9.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Titan-Chimera")

# --- UTILITAIRES ---
def clean_deepseek_json(raw_text: str):
    if not raw_text: return None
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

# --- NOTIFICATION MANAGER ---
class NotificationManager:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.session = None

    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send(self, title, message, color=0x3498db, priority="INFO"):
        if not self.webhook_url: return
        if priority == "CRITICAL": color = 0xe74c3c 
        elif priority == "TRADE": color = 0x2ecc71     
        elif priority == "WARNING": color = 0xf1c40f
        elif priority == "ERROR": color = 0xe67e22 
        elif priority == "DEGRADED": color = 0x9b59b6

        payload = {
            "embeds": [{
                "title": f"[{CONFIG['ENV_MODE']}] {title}",
                "description": message,
                "color": color,
                "footer": {"text": f"Titan v{CONFIG['VERSION']} | Epoch {CONFIG['LEARNING_EPOCH']}"},
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        try:
            session = await self._get_session()
            async with session.post(self.webhook_url, json=payload) as resp:
                if resp.status >= 400: logger.error(f"Webhook failed: {resp.status}")
        except Exception as e: logger.error(f"Notification error: {e}")

    async def send_trade_entry(self, symbol, side, qty, price, thesis, regime, scout=False):
        emoji = "🔭" if scout else ("📈" if side == "BUY" else "📉")
        title_sfx = " [SCOUT]" if scout else ""
        msg = f"**Symbol:** {symbol}\n**Side:** {side}\n**Qty:** {qty}\n**Price:** ${price}\n**Regime:** {regime}\n**Thesis:** {thesis}"
        await self.send(f"{emoji} Trade Entry{title_sfx}: {symbol}", msg, priority="TRADE")

    async def send_trade_exit(self, symbol, result, entry, exit_price, pnl_pct, mae=0.0, mfe=0.0):
        emoji = "💰" if result == "TP" else "🛑"
        stats = f"\n*MAE: {mae}% | MFE: {mfe}%*"
        msg = f"**Result:** {result}\n**Entry:** ${entry}\n**Exit:** ${exit_price}\n**Move:** {pnl_pct}%{stats}"
        await self.send(f"{emoji} Trade Closed: {symbol}", msg, priority="TRADE")

    async def send_heartbeat(self, status, equity, pnl_day, spy_vol, buying_power):
        msg = f"**State:** {status}\n**Equity:** ${equity}\n**Buying Power:** ${buying_power}\n**Day PnL:** {pnl_day}%\n**SPY Vol:** {spy_vol}%"
        await self.send("💓 System Heartbeat", msg, priority="INFO")

    async def send_halt(self, reason):
        await self.send("🛑 SYSTEM HALTED", f"@here **Risk Governor Triggered**\nReason: {reason}", priority="CRITICAL")

    async def send_degraded(self, reason):
        await self.send("🛡️ SYSTEM DEGRADED", f"**Mode Survie Activé**\nReason: {reason}\n*Risk reduced. Max pos = 1.*", priority="DEGRADED")
        
    async def send_rejection(self, symbol, status, reason):
        await self.send(f"⚠️ Order Rejected: {symbol}", f"**Status:** {status}\n**Reason:** {reason}", priority="ERROR")
    
    async def send_emergency(self, symbol, reason):
        await self.send("🚨 SENTINEL INTERVENTION", f"**EMERGENCY CLOSE TRIGGERED**\nSymbol: {symbol}\nReason: {reason}\n*System Halted for Audit.*", priority="CRITICAL")

# --- PERSISTANCE ---
class TitanDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, qty REAL, entry_price REAL, exit_price REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, exit_time DATETIME,
                    result TEXT, confidence INTEGER, thesis TEXT, mode TEXT,
                    tp_price REAL, sl_price REAL, status TEXT DEFAULT 'OPEN',
                    decision_id INTEGER, alpaca_order_id TEXT,
                    atr_at_entry REAL, market_regime TEXT,
                    highest_price REAL, lowest_price REAL, 
                    is_be_active INTEGER DEFAULT 0,
                    sl_updates_count INTEGER DEFAULT 0,
                    adaptive_code TEXT,
                    adaptive_reason TEXT,
                    side TEXT DEFAULT 'BUY',
                    epoch_id INTEGER DEFAULT 0,
                    mae REAL DEFAULT 0.0,
                    mfe REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT, confidence INTEGER, reason TEXT,
                    action TEXT, filter_rejected TEXT,
                    ai_raw TEXT
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS daily_stats (day DATE PRIMARY KEY, start_equity REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('is_halted', '0')")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('halt_reason', '')")
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('system_health', 'HEALTHY')")
            conn.commit()

    def log_trade(self, symbol, qty, price, conf, thesis, mode, tp, sl, dec_id, order_id, atr, regime, side):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status, decision_id, alpaca_order_id, atr_at_entry, market_regime, highest_price, lowest_price, sl_updates_count, adaptive_code, adaptive_reason, side, epoch_id, mae, mfe) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,'INIT','Initialized',?,?,0.0,0.0)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN', dec_id, order_id, atr, regime, price, price, side, CONFIG["LEARNING_EPOCH"]))
            conn.commit()

    def update_trade_stats(self, trade_id, highest_price, lowest_price, new_sl, be_active, update_count, mae, mfe, adaptive_code=None, adaptive_reason=None):
        with sqlite3.connect(self.db_path) as conn:
            sql = "UPDATE trades SET highest_price=?, lowest_price=?, sl_price=?, is_be_active=?, sl_updates_count=?, mae=?, mfe=?"
            params = [highest_price, lowest_price, new_sl, 1 if be_active else 0, update_count, mae, mfe]
            if adaptive_code:
                sql += ", adaptive_code=?, adaptive_reason=?"
                params.extend([adaptive_code, adaptive_reason])
            sql += " WHERE id=?"
            params.append(trade_id)
            conn.execute(sql, tuple(params))
            conn.commit()

    def close_trade(self, trade_id, exit_price, result, final_mae, final_mfe):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE trades SET status='CLOSED', exit_price=?, result=?, mae=?, mfe=?, exit_time=CURRENT_TIMESTAMP 
                WHERE id=?
            """, (exit_price, result, final_mae, final_mfe, trade_id))
            conn.commit()

    def get_open_trades(self, mode=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM trades WHERE status='OPEN'"
            if mode: query += f" AND mode='{mode}'"
            return conn.execute(query).fetchall()

    def get_symbol_daily_losses(self, symbol):
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE symbol=? AND result='SL' AND date(exit_time) = date('now')
            """, (symbol,)).fetchone()[0]
            return count
    
    def get_daily_live_trade_count(self):
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE mode='LIVE' AND date(entry_time) = date('now')
            """, ()).fetchone()[0]
            return count

    def get_rolling_live_stats(self, limit=20):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            trades = conn.execute("""
                SELECT result, entry_price, exit_price, side FROM trades 
                WHERE mode='LIVE' AND status='CLOSED' AND epoch_id >= ? 
                ORDER BY exit_time DESC LIMIT ?
            """, (CONFIG["LEARNING_EPOCH"], limit,)).fetchall()
            
            if not trades: return None, 0.0, 0
            total = len(trades)
            wins = 0
            cum_pnl_pct = 0.0
            for t in trades:
                if t['result'] == 'TP': wins += 1
                if t['side'] == 'BUY':
                    pnl = (t['exit_price'] - t['entry_price']) / t['entry_price']
                else:
                    pnl = (t['entry_price'] - t['exit_price']) / t['entry_price']
                cum_pnl_pct += (pnl * 100.0)
            winrate = (wins / total) * 100
            return winrate, cum_pnl_pct, total

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            res = {'consecutive_sl': 0}
            trades = conn.execute("SELECT result FROM trades WHERE mode='LIVE' AND status='CLOSED' ORDER BY id DESC LIMIT 5").fetchall()
            for (r,) in trades:
                if r == "SL": res['consecutive_sl'] += 1
                else: break
            res['decisions_today'] = conn.execute("SELECT COUNT(*) FROM ai_decision_log WHERE date(timestamp) = date('now')").fetchone()[0]
            res['live_open_count'] = conn.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN' AND mode='LIVE'").fetchone()[0]
            return res

    def get_system_state(self):
        with sqlite3.connect(self.db_path) as conn:
            is_halted = conn.execute("SELECT value FROM system_state WHERE key = 'is_halted'").fetchone()[0]
            reason = conn.execute("SELECT value FROM system_state WHERE key = 'halt_reason'").fetchone()[0]
            health = conn.execute("SELECT value FROM system_state WHERE key = 'system_health'").fetchone()[0]
            return (is_halted == '1', reason, health)

    def set_system_state(self, halted, reason="", health="HEALTHY"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'is_halted'", ('1' if halted else '0',))
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'halt_reason'", (reason,))
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'system_health'", (health,))
            conn.commit()

    def get_or_create_daily_stats(self, current_equity):
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT start_equity FROM daily_stats WHERE day = ?", (today,)).fetchone()
            if row: return row[0]
            conn.execute("INSERT INTO daily_stats (day, start_equity) VALUES (?, ?)", (today, current_equity))
            conn.commit()
            return current_equity

    def log_decision(self, symbol, conf, reason, action, filter_rejected="", ai_raw=""):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO ai_decision_log (symbol, confidence, reason, action, filter_rejected, ai_raw) VALUES (?,?,?,?,?,?)",
                (symbol, conf, reason, action, filter_rejected, ai_raw)
            )
            conn.commit()
            return cursor.lastrowid

# --- MOTEUR TITAN CHIMERA ---
class TitanEngine:
    def __init__(self):
        self.alpaca_key = os.getenv('ALPACA_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET')
        base_url = "https://paper-api.alpaca.markets" if CONFIG["ENV_MODE"] == 'PAPER' else "https://api.alpaca.markets"
        self.alpaca = tradeapi.REST(self.alpaca_key, self.alpaca_secret, base_url)
        self.db = TitanDatabase(CONFIG["DB_PATH"])
        self.notifier = NotificationManager(TITAN_WEBHOOK_URL)
        self.session = None
        self.last_trade_per_symbol = {}
        self.last_heartbeat = datetime.min
        
        is_halted, reason, health = self.db.get_system_state()
        self.status = {
            "version": CONFIG["VERSION"],
            "state": "HALTED" if is_halted else "INIT",
            "health": health,
            "equity": {"current": 0.0, "pnl_pct": 0.0, "buying_power": 0.0},
            "safety": {"consecutive_sl": 0, "market_stress": False},
        }

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def calculate_technicals_df(self, bars, period_rsi=14, period_z=20):
        """
        Nouveau module de calcul (ex-SMR logic) intégré à Titan.
        Utilise Pandas pour la robustesse.
        """
        if len(bars) < max(period_rsi, period_z) + 1: return None
        
        data = {
            'close': [b.c for b in bars],
            'high': [b.h for b in bars],
            'low': [b.l for b in bars]
        }
        df = pd.DataFrame(data)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period_rsi).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period_rsi).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Z-Score
        df['mean'] = df['close'].rolling(window=period_z).mean()
        df['std'] = df['close'].rolling(window=period_z).std()
        df['z_score'] = (df['close'] - df['mean']) / df['std']
        
        # ATR (pour sizing Titan)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=CONFIG['ATR_PERIOD']).mean()

        return df.iloc[-1]

    async def check_system_health(self):
        # ... (Logique identique à Titan v8.8.1 - conservée pour brièveté) ...
        # Se référer au fichier précédent pour la logique complète du kill-switch
        return True

    async def sync_data(self):
        try:
            acc = self.alpaca.get_account()
            eq = float(acc.equity)
            bp = float(acc.buying_power)
            start_eq = self.db.get_or_create_daily_stats(eq)
            pnl_pct = ((eq - start_eq) / start_eq * 100) if start_eq > 0 else 0.0
            
            # Mise à jour status
            self.status["equity"] = {"current": round(eq, 2), "pnl_pct": round(pnl_pct, 2), "buying_power": round(bp, 2)}
            
            # Market Stress Check (SPY Volatility)
            try:
                spy = self.alpaca.get_bars("SPY", TimeFrame.Minute, limit=20)
                if len(spy) >= 15:
                    spy_vol = ((max([b.h for b in spy]) - min([b.l for b in spy])) / min([b.l for b in spy])) * 100
                    self.status["safety"]["market_stress"] = spy_vol > CONFIG["MARKET_STRESS_THRESHOLD"]
            except: pass

        except Exception as e: logger.error(f"Sync error: {e}")

    # --- NEWS SIMULATION MODULE ---
    def fetch_market_news(self, symbol):
        """
        Placeholder pour Alpaca News API ou Polygon.
        Pour l'instant, retourne un contexte générique ou vide pour forcer l'IA à utiliser sa base interne.
        """
        # Dans une prod réelle, faire : self.alpaca.get_news(symbol)
        return f"Latest market data and news for {symbol}. Check for earnings, fraud, or macro events."

    # --- SMR AGENT (VETO IA) ---
    async def query_smr_veto(self, symbol, z_score, rsi, news_context):
        """
        L'Intelligence Sémantique de SMR-26, encapsulée dans une fonction asynchrone Titan.
        """
        s = await self.get_session()
        
        prompt = f"""
        Rôle : Analyste de Risque Senior (Titan Chimera System).
        Contexte : Le module technique a détecté un setup de 'Mean Reversion' extrême sur {symbol}.
        Données : Z-Score = {round(z_score, 2)} (Anomalie statistique), RSI = {round(rsi, 2)} (Survente).
        News Context : "{news_context}"
        
        Tâche : Analyse le risque structurel.
        1. S'agit-il d'une panique temporaire (opportunité) ?
        2. Ou d'un changement fondamental (fraude, faillite, procès, perte de contrat) ?
        
        Réponds UNIQUEMENT en JSON :
        {{
            "reasoning": "Ton analyse courte (1 phrase max).",
            "decision": "ALLOW" ou "BLOCK",
            "confidence": 0-100
        }}
        """
        
        try:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": CONFIG["AI_MODEL"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=15) as r:
                resp_json = await r.json()
                raw_content = resp_json['choices'][0]['message']['content']
                return clean_deepseek_json(raw_content), raw_content
        except Exception as e:
            logger.error(f"Veto AI Error: {e}")
            return None, str(e)

    async def scan_smr_opportunities(self):
        """
        Le nouveau CŒUR de la stratégie.
        Remplace le 'fetch_ai_picks' aléatoire par un scan méthodique.
        """
        picks = []
        
        for symbol in CONFIG["WATCHLIST"]:
            # Filtre cooldown
            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                continue

            # 1. Acquisition Données (Daily pour SMR)
            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Day, limit=CONFIG["SMR_LOOKBACK"]+10)
                if not bars: continue
                
                # 2. Calcul Indicateurs Pandas
                tech = self.calculate_technicals_df(bars, period_z=20, period_rsi=14)
                if tech is None: continue
                
                z_score = tech['z_score']
                rsi = tech['rsi']
                atr = tech['atr']
                price = tech['close']
                
                # 3. Filtre Technique (SMR Logic)
                is_oversold = (z_score < CONFIG["SMR_Z_THRESH"]) and (rsi < CONFIG["SMR_RSI_THRESH"])
                
                if is_oversold:
                    logger.info(f"🔎 SMR CANDIDATE: {symbol} (Z:{z_score:.2f} RSI:{rsi:.2f}) - Requesting AI Veto...")
                    
                    # 4. Filtre Sémantique (AI Veto)
                    news = self.fetch_market_news(symbol)
                    ai_resp, raw_ai = await self.query_smr_veto(symbol, z_score, rsi, news)
                    
                    if ai_resp:
                        decision = ai_resp.get("decision", "BLOCK")
                        reason = ai_resp.get("reasoning", "No reason")
                        conf = ai_resp.get("confidence", 0)
                        
                        self.db.log_decision(symbol, conf, reason, decision, "", raw_ai)
                        
                        if decision == "ALLOW":
                            # Conversion au format "Pick" standard de Titan
                            picks.append({
                                'symbol': symbol,
                                'side': 'BUY', # SMR est Mean Reversion Long pour l'instant
                                'confidence': conf,
                                'reason': f"[SMR-26] Tech(Z:{z_score:.1f}/RSI:{rsi:.0f}) + AI Veto: {reason}",
                                'atr_cache': atr, # On passe l'ATR calculé pour éviter un recalcul
                                'regime_cache': 'RANGE' # SMR est une strat de Range/Reversion
                            })
                        else:
                            logger.info(f"🛡️ AI VETO BLOCKED {symbol}: {reason}")
                    else:
                        logger.warning(f"⚠️ AI Veto failed for {symbol}")

            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                
        return picks

    async def run_logic(self):
        # 1. Maintenance des trades existants (Stop suiveur, BE, etc.)
        # Utilise la logique Titan existante (reconcile_trades est supposé être dans la classe, non copié ici pour brièveté mais requis)
        # await self.reconcile_trades() 
        
        # 2. Vérification Santé
        if self.status["safety"]["market_stress"]:
            logger.warning("Market Stress High - Skipping Scan")
            return

        # 3. SCAN HYBRIDE (Technique -> Sémantique)
        picks = await self.scan_smr_opportunities()
        
        if not picks:
            self.db.log_decision("SYSTEM", 0, "Scan SMR Finished. No Candidates.", "IDLE", "")
            return

        # 4. EXÉCUTION (Pipeline Titan Standard)
        # Ici on reprend la logique de "risk sizing" de Titan
        # Pour chaque pick validé par SMR, on le passe au Risk Governor
        
        current_equity = self.status["equity"]["current"]
        if current_equity <= 0: current_equity = 10000.0 # Fallback paper

        for p in picks:
            symbol = p['symbol']
            side = p['side']
            conf = p['confidence']
            thesis = p['reason']
            atr = p['atr_cache']
            
            # --- RISK GOVERNOR TITAN ---
            # Calcul dynamique du SL basé sur l'ATR (hérité de Titan)
            regime = "RANGE" # Force le régime SMR
            regime_cfg = CONFIG["REGIME_CONFIG"][regime]
            
            latest_price = self.alpaca.get_latest_trade(symbol).price
            entry = float(latest_price)
            
            sl_dist = atr * regime_cfg["SL_MULT"]
            tp_dist = atr * regime_cfg["TP_MULT"]
            
            sl = round(entry - sl_dist, 2)
            tp = round(entry + tp_dist, 2)
            
            # Sizing (1% à 2% du capital selon confiance)
            risk_pct = CONFIG["RISK_PCT_PER_TRADE"]
            if conf < 80: risk_pct *= 0.75 # Réduit taille si confiance moyenne
            
            risk_usd = current_equity * (risk_pct / 100.0)
            qty = math.floor(risk_usd / sl_dist)
            
            # Validation finale
            if qty < 1: 
                logger.info(f"Skipping {symbol}: Insufficient capital for risk sizing.")
                continue
                
            cost = qty * entry
            if cost > (current_equity * CONFIG["MAX_CAPITAL_PER_TRADE_PCT"] / 100.0):
                qty = math.floor((current_equity * CONFIG["MAX_CAPITAL_PER_TRADE_PCT"] / 100.0) / entry)
            
            # Envoi Ordre
            try:
                logger.info(f"🚀 EXECUTING TITAN-SMR: {symbol} Qty:{qty} Entry:{entry} TP:{tp} SL:{sl}")
                order = self.alpaca.submit_order(
                    symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc',
                    order_class='bracket',
                    take_profit={'limit_price': tp},
                    stop_loss={'stop_price': sl}
                )
                
                # Log DB
                # Note: decision_id=0 pour simplifier ici
                self.db.log_trade(symbol, qty, entry, conf, thesis, "LIVE", tp, sl, 0, order.id, atr, regime, side)
                await self.notifier.send_trade_entry(symbol, side, qty, entry, thesis, "SMR-HYBRID")
                
                # Cooldown update
                self.last_trade_per_symbol[symbol] = datetime.now()
                
            except Exception as e:
                logger.error(f"Execution Error {symbol}: {e}")

    async def main_loop(self):
        logger.info("🔥 Titan Chimera Engine Started.")
        while True:
            try:
                await self.sync_data()
                is_halted, _, _ = self.db.get_system_state()
                
                # Logique simplifiée d'heure de marché
                clock = self.alpaca.get_clock()
                if clock.is_open and not is_halted:
                    await self.run_logic()
                else:
                    if is_halted: logger.warning("System HALTED.")
                    else: logger.info("Market Closed. Sleeping...")
                    
            except Exception as e: logger.error(f"Loop error: {e}")
            
            await asyncio.sleep(CONFIG["SCAN_INTERVAL"])

# --- ENTRY POINT ---
if __name__ == "__main__":
    titan = TitanEngine()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(titan.main_loop())
