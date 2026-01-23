import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import re
import math
import statistics
from datetime import datetime, date, timedelta, timezone
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, APIError
from aiohttp import web
import aiohttp_cors

# --- CONFIGURATION V8.6.8 (ACTIVE LEARNER) ---
load_dotenv()

API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN')
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")
TITAN_WEBHOOK_URL = os.getenv('TITAN_WEBHOOK_URL', "")

CONFIG = {
    "VERSION": "8.6.8-Active-Learner",
    "PORT": 8080,
    "DB_PATH": "titan_v8_recon.db",
    
    # --- RISQUE & CAPITAL ---
    "MAX_OPEN_POSITIONS": 5,           
    "MAX_MACRO_POSITIONS": 1,          
    "RISK_PCT_PER_TRADE": 2.0,       
    "MAX_TOTAL_RISK_PCT": 6.0,         
    "LIVE_THRESHOLD": 76,
    "MACRO_THRESHOLD": 85,             
    "MIN_TRADE_AMOUNT_USD": 150.0,
    
    # --- KILL SWITCH 2-STAGES (V8.6.8) ---
    "MIN_WINRATE_THRESHOLD": 62.0,     
    "PNL_WARN_THRESHOLD": -1.5,        # Déclenche le mode DEGRADED
    "PNL_CRIT_THRESHOLD": -3.0,        # Déclenche le HALT complet
    "WINRATE_LOOKBACK_TRADES": 20,     
    "MARKET_OPEN_BLACKOUT_MIN": 15,    
    
    # --- SCOUT MODE (V8.6.8) ---
    "ALLOW_SCOUT_TRADE": True,         # Autorise 1 trade/jour sans Shadow Promotion
    "SCOUT_RISK_FACTOR": 0.5,          # Risque réduit pour le Scout (1.0%)

    # --- SHADOW PROMOTION ---
    "SHADOW_PROMOTION_MIN_WINS": 2,    
    "SHADOW_PROMOTION_HIGH_CONF": 80,  
    "SHADOW_PROMOTION_WINDOW_HOURS": 24, 

    "MARKET_STRESS_THRESHOLD": 1.8,
    "COOLDOWN_PER_SYMBOL_MIN": 15,
    "ATR_PERIOD": 14,
    "ALLOW_SHORTS": True,
    
    # --- CALIBRATION REGIMES ---
    "REGIME_CONFIG": {
        "TREND": {"TP_MULT": 2.5, "SL_MULT": 1.2, "DESC": "Trend Following", "TRAILING": True},
        "RANGE": {"TP_MULT": 0.8, "SL_MULT": 1.3, "DESC": "Retail Scalp (Hardened)", "TRAILING": False},
        "CHOP":  {"TP_MULT": 0.0, "SL_MULT": 0.0, "DESC": "No trade zone", "TRAILING": False},
        "MACRO": {"TP_MULT": 6.0, "SL_MULT": 4.0, "DESC": "Macro Structural", "TRAILING": False} 
    },
    
    # BE Ajusté dynamiquement dans le code (0.35 pour RANGE, 0.5 sinon)
    "BE_TRIGGER_ATR_STD": 0.5,
    "BE_TRIGGER_ATR_RANGE": 0.35, 
    
    "TRAILING_STEP_ATR": 0.5,
    "ENABLE_ADAPTIVE_GOVERNOR": True,
    "ENABLE_ADAPTIVE_SHADOW": False,
    "VOLATILITY_HIGH_THRESHOLD": 0.005,
    "VOLATILITY_LOW_THRESHOLD": 0.001,
    "MAX_DAILY_DRAWDOWN_PCT": -4.0,
    "MAX_LOSSES_PER_SYMBOL": 2,
    "MAX_SL_UPDATES_PER_TRADE": 20,
    "HEARTBEAT_INTERVAL_MIN": 60,
    "NOTIFY_LEVEL": "INFO",
    "MIN_SL_DISTANCE_USD": 0.05,
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "deepseek/deepseek-v3.2",
    "SCAN_INTERVAL": 60 
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("titan_v8_6_8.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Titan-Learner")

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
        elif priority == "DEGRADED": color = 0x9b59b6 # Purple for degraded mode

        payload = {
            "embeds": [{
                "title": f"[{CONFIG['ENV_MODE']}] {title}",
                "description": message,
                "color": color,
                "footer": {"text": f"Titan v{CONFIG['VERSION']}"},
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

    async def send_trade_exit(self, symbol, result, entry, exit_price, pnl_pct):
        emoji = "💰" if result == "TP" else "🛑"
        msg = f"**Result:** {result}\n**Entry:** ${entry}\n**Exit:** ${exit_price}\n**Move:** {pnl_pct}%"
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

# --- ADAPTIVE MANAGER ---
class AdaptiveManager:
    PROFILES = {
        "STANDARD": {"be_trigger": 1.0, "trailing_step": CONFIG["TRAILING_STEP_ATR"], "trailing_mult_offset": 0.0, "desc": "Standard"},
        "TREND_HV": {"be_trigger": 1.5, "trailing_step": 0.8, "trailing_mult_offset": 0.5, "desc": "Trend High Vol"},
        "TREND_LV": {"be_trigger": 0.8, "trailing_step": 0.3, "trailing_mult_offset": -0.2, "desc": "Trend Low Vol"},
        
        # V8.6.8: Range Scalp with dynamic BE logic handled in lifecycle
        "RANGE_SCALP": {"be_trigger": CONFIG["BE_TRIGGER_ATR_RANGE"], "trailing_step": 0.0, "trailing_mult_offset": 0.0, "desc": "Retail Scalp (Optimized)"},
        
        "MACRO": {"be_trigger": 2.5, "trailing_step": 1.0, "trailing_mult_offset": 1.0, "desc": "Macro Structural"} 
    }
    @staticmethod
    def get_profile(code): return AdaptiveManager.PROFILES.get(code, AdaptiveManager.PROFILES["STANDARD"])
    @staticmethod
    def compute_new_profile_code(current_price, atr, regime):
        if not CONFIG["ENABLE_ADAPTIVE_GOVERNOR"] or atr <= 0: return "STANDARD"
        vol_ratio = atr / current_price
        if regime == "TREND":
            if vol_ratio > CONFIG["VOLATILITY_HIGH_THRESHOLD"]: return "TREND_HV"
            elif vol_ratio < CONFIG["VOLATILITY_LOW_THRESHOLD"]: return "TREND_LV"
            else: return "STANDARD"
        elif regime == "RANGE": return "RANGE_SCALP"
        return "STANDARD"

# --- PERSISTANCE (V8.6.8) ---
class TitanDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # V8.6.8: Added lowest_price for MAE calculation
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
                    side TEXT DEFAULT 'BUY'
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
            conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('system_health', 'HEALTHY')") # HEALTHY, DEGRADED, CRITICAL
            conn.commit()
            
            # Migrations
            try: conn.execute("ALTER TABLE trades ADD COLUMN side TEXT DEFAULT 'BUY'")
            except sqlite3.OperationalError: pass
            try: conn.execute("ALTER TABLE trades ADD COLUMN lowest_price REAL")
            except sqlite3.OperationalError: pass

    def log_trade(self, symbol, qty, price, conf, thesis, mode, tp, sl, dec_id, order_id, atr, regime, side, adaptive_code="INIT"):
        with sqlite3.connect(self.db_path) as conn:
            # Initialize lowest_price with entry price
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status, decision_id, alpaca_order_id, atr_at_entry, market_regime, highest_price, lowest_price, sl_updates_count, adaptive_code, adaptive_reason, side) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,'Initialized',?)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN', dec_id, order_id, atr, regime, price, price, adaptive_code, side))
            conn.commit()

    def update_trade_stats(self, trade_id, highest_price, lowest_price, new_sl, be_active, update_count, adaptive_code=None, adaptive_reason=None):
        with sqlite3.connect(self.db_path) as conn:
            sql = "UPDATE trades SET highest_price=?, lowest_price=?, sl_price=?, is_be_active=?, sl_updates_count=?"
            params = [highest_price, lowest_price, new_sl, 1 if be_active else 0, update_count]
            if adaptive_code:
                sql += ", adaptive_code=?, adaptive_reason=?"
                params.extend([adaptive_code, adaptive_reason])
            sql += " WHERE id=?"
            params.append(trade_id)
            conn.execute(sql, tuple(params))
            conn.commit()

    def close_trade(self, trade_id, exit_price, result):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE trades SET status='CLOSED', exit_price=?, result=?, exit_time=CURRENT_TIMESTAMP 
                WHERE id=?
            """, (exit_price, result, trade_id))
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
    
    # V8.6.8: Count daily live trades for Scout Mode
    def get_daily_live_trade_count(self):
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE mode='LIVE' AND date(entry_time) = date('now')
            """, ()).fetchone()[0]
            return count

    def get_shadow_promotion_stats(self, symbol):
        with sqlite3.connect(self.db_path) as conn:
            threshold = datetime.now() - timedelta(hours=CONFIG["SHADOW_PROMOTION_WINDOW_HOURS"])
            rows = conn.execute("""
                SELECT confidence FROM trades 
                WHERE symbol=? AND mode='SHADOW' AND result='TP' AND exit_time > ?
            """, (symbol, threshold)).fetchall()
            win_count = len(rows)
            max_conf = max([r[0] for r in rows]) if rows else 0
            return win_count, max_conf

    def get_rolling_live_stats(self, limit=20):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            trades = conn.execute("""
                SELECT result, entry_price, exit_price, side FROM trades 
                WHERE mode='LIVE' AND status='CLOSED' 
                ORDER BY exit_time DESC LIMIT ?
            """, (limit,)).fetchall()
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
            shadows = conn.execute("SELECT result FROM trades WHERE mode='SHADOW' AND status='CLOSED' AND date(exit_time) = date('now')").fetchall()
            res['shadow_winrate'] = round((len([r for (r,) in shadows if r == 'TP']) / len(shadows) * 100), 2) if shadows else 0
            res['shadow_open_count'] = conn.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN' AND mode='SHADOW'").fetchone()[0]
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

# --- MOTEUR TITAN ---
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
        self.was_market_open = False
        self.market_open_time = None 
        
        is_halted, reason, health = self.db.get_system_state()
        self.status = {
            "version": CONFIG["VERSION"],
            "state": "HALTED" if is_halted else "INIT",
            "health": health,
            "market": "CLOSED",
            "halt_reason": reason,
            "equity": {"current": 0.0, "pnl_pct": 0.0, "buying_power": 0.0},
            "safety": {"consecutive_sl": 0, "market_stress": False},
            "positions": {"live_broker": 0, "live_titan": 0, "shadow_open": 0},
            "omni": {"decisions_today": 0, "shadow_winrate": 0, "last_action": "N/A"},
            "model": CONFIG["AI_MODEL"]
        }

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _calculate_sma(self, prices, period):
        if len(prices) < period: return None
        return sum(prices[-period:]) / period

    def analyze_market_regime(self, bars):
        if len(bars) < 50: return "UNCERTAIN"
        prices = [b.c for b in bars]
        sma20 = self._calculate_sma(prices, 20)
        sma50 = self._calculate_sma(prices, 50)
        if not sma20 or not sma50: return "UNCERTAIN"
        current_price = prices[-1]
        is_uptrend = current_price > sma20 > sma50
        is_downtrend = current_price < sma20 < sma50
        sma_spread = abs(sma20 - sma50) / sma50 * 100
        
        if sma_spread < 0.5: return "RANGE"
        elif is_uptrend or is_downtrend: return "TREND"
        else: return "CHOP"

    def calculate_atr(self, bars, period=14):
        if len(bars) < period + 1: return 0.0
        tr_list = []
        for i in range(1, len(bars)):
            h, l, prev_c = bars[i].h, bars[i].l, bars[i-1].c
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_list.append(tr)
        if len(tr_list) < period: return 0.0
        return statistics.mean(tr_list[-period:])

    # V8.6.8: 2-Stage Health Check
    async def check_system_health(self):
        winrate, cum_pnl, count = self.db.get_rolling_live_stats(CONFIG["WINRATE_LOOKBACK_TRADES"])
        
        current_health = self.status["health"]
        new_health = "HEALTHY"
        
        if winrate is not None and count >= CONFIG["WINRATE_LOOKBACK_TRADES"]:
            # Check Critical first
            if winrate < CONFIG["MIN_WINRATE_THRESHOLD"] and cum_pnl < CONFIG["PNL_CRIT_THRESHOLD"]:
                reason = f"CRITICAL FAILURE (WR {round(winrate,1)}% / PnL {round(cum_pnl,1)}%)"
                self.db.set_system_state(True, reason, "CRITICAL")
                await self.notifier.send_halt(reason)
                logger.critical(f"🛑 {reason}")
                return False
            # Check Degraded
            elif winrate < CONFIG["MIN_WINRATE_THRESHOLD"] and cum_pnl < CONFIG["PNL_WARN_THRESHOLD"]:
                new_health = "DEGRADED"
                if current_health != "DEGRADED":
                    reason = f"Degraded Mode: WR {round(winrate,1)}% / PnL {round(cum_pnl,1)}%"
                    self.db.set_system_state(False, "", "DEGRADED")
                    await self.notifier.send_degraded(reason)
            else:
                # Recovery to Healthy
                if current_health != "HEALTHY":
                    self.db.set_system_state(False, "", "HEALTHY")
                    await self.notifier.send("✅ System Recovered", "Metrics back to normal.", priority="INFO")

        self.status["health"] = new_health
        return True

    async def sync_data(self):
        try:
            acc = self.alpaca.get_account()
            eq = float(acc.equity)
            bp = float(acc.buying_power)
            start_eq = self.db.get_or_create_daily_stats(eq)
            pnl_pct = ((eq - start_eq) / start_eq * 100) if start_eq > 0 else 0.0
            
            if pnl_pct <= CONFIG["MAX_DAILY_DRAWDOWN_PCT"]:
                is_halted, _, _ = self.db.get_system_state()
                if not is_halted:
                    reason = f"MAX DRAWDOWN HIT ({round(pnl_pct, 2)}%)"
                    self.db.set_system_state(True, reason, "CRITICAL")
                    logger.critical(f"🛑 SYSTEM HALTED: {reason}")
                    await self.notifier.send_halt(reason)

            clock = self.alpaca.get_clock()
            is_open = clock.is_open
            
            if is_open and self.market_open_time is None:
                try:
                    calendar = self.alpaca.get_calendar(start=date.today().isoformat(), end=date.today().isoformat())
                    if calendar:
                        self.market_open_time = calendar[0].open
                        logger.info(f"🕒 Market Open Time Cached: {self.market_open_time}")
                except Exception as e: logger.warning(f"Failed to cache market open: {e}")
            elif not is_open:
                self.market_open_time = None 

            if self.was_market_open and not is_open:
                await self.notifier.send("🏁 Market Closed", f"End of day summary.\n**Equity:** ${round(eq, 2)}\n**Daily PnL:** {round(pnl_pct, 2)}%", priority="INFO")
            self.was_market_open = is_open

            stats = self.db.get_stats()
            self.status.update({
                "market": "OPEN" if is_open else "CLOSED",
                "equity": {"current": round(eq, 2), "pnl_pct": round(pnl_pct, 2), "buying_power": round(bp, 2)},
                "positions": {
                    "live_broker": len(self.alpaca.list_positions()), 
                    "live_titan": stats['live_open_count'],           
                    "shadow_open": stats['shadow_open_count']
                },
                "safety": {"consecutive_sl": stats['consecutive_sl']},
                "omni": stats,
                "model": CONFIG["AI_MODEL"]
            })

            if not await self.check_system_health():
                return

            if datetime.now() - self.last_heartbeat > timedelta(minutes=CONFIG["HEARTBEAT_INTERVAL_MIN"]):
                spy_vol = 0.0 
                await self.notifier.send_heartbeat(f"{self.status['state']} [{self.status['health']}]", round(eq, 2), round(pnl_pct, 2), "N/A", round(bp, 2))
                self.last_heartbeat = datetime.now()

        except Exception as e: logger.error(f"Sync error: {e}")

    async def manage_trade_lifecycle(self, t, current_price, regime_cfg):
        if t['status'] != 'OPEN': return
        current_updates = t['sl_updates_count'] or 0
        if current_updates >= CONFIG["MAX_SL_UPDATES_PER_TRADE"]: return

        trade_side = t['side'] # BUY or SELL
        
        atr = t['atr_at_entry']
        if atr <= 0: return

        # Adaptive Logic
        use_adaptive = CONFIG["ENABLE_ADAPTIVE_GOVERNOR"]
        if t['mode'] == 'SHADOW' and not CONFIG["ENABLE_ADAPTIVE_SHADOW"]: use_adaptive = False
        
        profile_code = "STANDARD"
        save_profile = False
        if use_adaptive:
            if t['adaptive_code'] and t['adaptive_code'] != 'INIT': 
                profile_code = t['adaptive_code']
            else:
                if t['adaptive_code'] == 'MACRO':
                    profile_code = 'MACRO'
                else:
                    regime = t['market_regime'] if 'market_regime' in t.keys() else "TREND"
                    profile_code = AdaptiveManager.compute_new_profile_code(current_price, atr, regime)
                    save_profile = True
        
        profile = AdaptiveManager.get_profile(profile_code)
        
        # V8.6.8: Differentiated BE for RANGE
        if profile_code == "RANGE_SCALP":
            be_trigger = CONFIG["BE_TRIGGER_ATR_RANGE"] # 0.35
        else:
            be_trigger = profile["be_trigger"] # 0.5 (Standard) or others
            
        trailing_step = profile["trailing_step"]
        trailing_offset_mult = profile["trailing_mult_offset"]
        adaptive_desc = profile["desc"]
        
        entry = t['entry_price']
        current_sl = t['sl_price']
        new_sl = current_sl
        update_needed = False
        is_be = bool(t['is_be_active'])
        
        # V8.6.8: Update Highest/Lowest for MFE/MAE
        # Handle None in DB (for old trades)
        highest = t['highest_price'] if t['highest_price'] is not None else entry
        lowest = t['lowest_price'] if t['lowest_price'] is not None else entry
        
        highest = max(highest, current_price)
        lowest = min(lowest, current_price)

        # --- LOGIC FOR BUY (LONG) ---
        if trade_side == "BUY":
            if not is_be and (current_price >= entry + (atr * be_trigger)):
                new_sl = entry 
                is_be = True
                update_needed = True
                await self.notifier.send(f"🛡️ Break-Even (LONG): {t['symbol']}", f"Profile: {profile_code}. SL -> Entry.", priority="TRADE")
            
            if regime_cfg["TRAILING"] and is_be:
                final_sl_mult = regime_cfg["SL_MULT"] + trailing_offset_mult
                if final_sl_mult < 0.5: final_sl_mult = 0.5
                trail_dist = atr * final_sl_mult
                potential_sl = highest - trail_dist
                if potential_sl > (new_sl + (atr * trailing_step)):
                    new_sl = round(potential_sl, 2)
                    update_needed = True
            
            if update_needed and new_sl <= current_sl: update_needed = False

        # --- LOGIC FOR SELL (SHORT) ---
        elif trade_side == "SELL":
            if not is_be and (current_price <= entry - (atr * be_trigger)):
                new_sl = entry
                is_be = True
                update_needed = True
                await self.notifier.send(f"🛡️ Break-Even (SHORT): {t['symbol']}", f"Profile: {profile_code}. SL -> Entry.", priority="TRADE")

            if regime_cfg["TRAILING"] and is_be:
                final_sl_mult = regime_cfg["SL_MULT"] + trailing_offset_mult
                if final_sl_mult < 0.5: final_sl_mult = 0.5
                trail_dist = atr * final_sl_mult
                potential_sl = lowest + trail_dist 
                if potential_sl < (new_sl - (atr * trailing_step)):
                    new_sl = round(potential_sl, 2)
                    update_needed = True
            
            if update_needed and new_sl >= current_sl: update_needed = False

        # Apply Updates (SL or Stats)
        success = False
        if update_needed:
            if t['mode'] == 'LIVE':
                try:
                    orders = self.alpaca.list_orders(status='open', symbols=[t['symbol']])
                    sl_order = next((o for o in orders if o.type == 'stop' and o.parent_id == t['alpaca_order_id']), None)
                    if not sl_order: sl_order = next((o for o in orders if o.type == 'stop'), None)
                    if sl_order:
                        self.alpaca.replace_order(sl_order.id, stop_price=new_sl)
                        success = True
                        logger.info(f"✅ ALPACA ORDER UPDATED {t['symbol']} ({trade_side}) SL -> {new_sl} ({profile_code})")
                    else: logger.error(f"❌ CRITICAL: SL Order NOT FOUND for {t['symbol']}")
                except Exception as e: logger.error(f"❌ API ERROR updating {t['symbol']}: {e}")
            else: success = True
        
        # Always update stats (High/Low/MAE/MFE) even if no SL change
        code_to_save = profile_code if save_profile else None
        reason_to_save = adaptive_desc if save_profile else None
        
        # If no SL update needed, we still want to save stats occasionally? 
        # For simplicity in this logic, we call update_trade_stats every lifecycle tick if price changed bound
        # But to avoid DB spam, we combine it with SL updates OR checking if bounds changed significantly.
        # Here we just update if bounds changed OR sl changed.
        bounds_changed = (highest != t['highest_price']) or (lowest != t['lowest_price'])
        
        if success or bounds_changed:
            self.db.update_trade_stats(t['id'], highest, lowest, new_sl, is_be, current_updates + (1 if success else 0), code_to_save, reason_to_save)

    async def reconcile_trades(self):
        trades = self.db.get_open_trades()
        if not trades: return

        live_positions = {p.symbol: p for p in self.alpaca.list_positions()} if self.alpaca else {}

        for t in trades:
            symbol = t['symbol']
            trade_side = t['side']
            
            if t['mode'] == 'LIVE' and symbol not in live_positions:
                activities = self.alpaca.get_activities(activity_types='FILL')
                symbol_fills = [f for f in activities if f.symbol == symbol]
                
                if t['alpaca_order_id']:
                    try:
                        order = self.alpaca.get_order(t['alpaca_order_id'])
                        if order.status in ['canceled', 'expired', 'rejected']:
                            self.db.close_trade(t['id'], 0, "VOID")
                            continue
                    except: pass

                if symbol_fills:
                    exit_price = float(symbol_fills[0].price)
                    
                    if trade_side == "BUY":
                        res = "TP" if exit_price >= t['entry_price'] else "SL"
                        move_pct = round((exit_price - t['entry_price']) / t['entry_price'] * 100, 2)
                    else: # SELL
                        res = "TP" if exit_price <= t['entry_price'] else "SL"
                        move_pct = round((t['entry_price'] - exit_price) / t['entry_price'] * 100, 2) 

                    self.db.close_trade(t['id'], exit_price, res)
                    await self.notifier.send_trade_exit(symbol, res, t['entry_price'], exit_price, move_pct)
                continue

            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
                if not bars: continue
                current_price = bars[0].c

                if t['mode'] == 'SHADOW':
                    is_closed = False
                    if trade_side == "BUY":
                        if current_price >= t['tp_price']: 
                            self.db.close_trade(t['id'], current_price, "TP")
                            is_closed = True
                        elif current_price <= t['sl_price']: 
                            self.db.close_trade(t['id'], current_price, "SL")
                            is_closed = True
                    else: # SELL
                        if current_price <= t['tp_price']: 
                            self.db.close_trade(t['id'], current_price, "TP")
                            is_closed = True
                        elif current_price >= t['sl_price']: 
                            self.db.close_trade(t['id'], current_price, "SL")
                            is_closed = True

                    if not is_closed:
                        regime = t['market_regime'] if 'market_regime' in t.keys() else "RANGE"
                        # Handle old trades without adaptive code
                        ac = t['adaptive_code'] if t['adaptive_code'] else 'INIT'
                        if ac == 'MACRO': cfg = CONFIG["REGIME_CONFIG"]["MACRO"]
                        else: cfg = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                        await self.manage_trade_lifecycle(t, current_price, cfg)

                elif t['mode'] == 'LIVE':
                    regime = t['market_regime'] if 'market_regime' in t.keys() else "RANGE"
                    ac = t['adaptive_code'] if t['adaptive_code'] else 'INIT'
                    if ac == 'MACRO':
                        cfg = CONFIG["REGIME_CONFIG"]["MACRO"]
                    else:
                        cfg = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                    await self.manage_trade_lifecycle(t, current_price, cfg)
            except Exception as e:
                logger.error(f"Recon error on {symbol}: {e}")

    async def fetch_ai_picks(self):
        s = await self.get_session()
        prompt = (
            "Analyze US market structure. Return ONLY JSON: "
            "{'picks': [{'symbol': 'TICKER', 'side': 'BUY' or 'SELL', 'confidence': 95, 'reason': 'thesis'}]}. "
            "Identify high probability setups."
        )
        try:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": CONFIG["AI_MODEL"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                timeout=30) as r:
                resp_json = await r.json()
                raw_content = resp_json['choices'][0]['message']['content'] if 'choices' in resp_json else str(resp_json)
                parsed = clean_deepseek_json(raw_content)
                if parsed and "picks" in parsed: return {"picks": parsed["picks"], "raw": raw_content, "error": None}
                return {"picks": [], "raw": raw_content, "error": "PARSE_ERROR"}
        except Exception as e:
            return {"picks": [], "raw": str(e), "error": "CONN_ERROR"}

    async def run_logic(self):
        await self.reconcile_trades()
        spy_vol = 0.0
        try:
            spy = self.alpaca.get_bars("SPY", TimeFrame.Minute, limit=20)
            if len(spy) >= 15:
                spy_vol = ((max([b.h for b in spy]) - min([b.l for b in spy])) / min([b.l for b in spy])) * 100
                self.status["safety"]["market_stress"] = spy_vol > CONFIG["MARKET_STRESS_THRESHOLD"]
        except Exception: pass
        
        if self.market_open_time:
             now = datetime.now(timezone.utc)
             if now < (self.market_open_time + timedelta(minutes=CONFIG["MARKET_OPEN_BLACKOUT_MIN"])):
                 self.db.log_decision("SYSTEM", 0, "Market Open Blackout (First 15m)", "IDLE", "")
                 return 

        ai_data = await self.fetch_ai_picks()
        picks, raw_text = ai_data.get("picks", []), ai_data.get("raw", "")

        if not picks:
            self.db.log_decision("SYSTEM", 0, f"Heartbeat (Vol:{round(spy_vol,2)}%).", "IDLE", ai_raw=raw_text)
            return

        intra_loop_committed_cash = 0.0
        intra_loop_committed_risk = 0.0
        
        current_open_trades = self.db.get_open_trades('LIVE')
        current_open_risk = 0.0
        current_macro_count = 0 
        
        for t in current_open_trades:
            trade_risk = abs(t['entry_price'] - t['sl_price']) * t['qty']
            if trade_risk > 0: current_open_risk += trade_risk
            if t['adaptive_code'] == 'MACRO': current_macro_count += 1
        
        # V8.6.8: Scout Trade Check
        daily_live_count = self.db.get_daily_live_trade_count()
        is_scout_mode_eligible = CONFIG["ALLOW_SCOUT_TRADE"] and (daily_live_count == 0) and (len(current_open_trades) == 0)

        for p in picks:
            capped_reason = ""
            symbol = p.get('symbol', '').upper()
            side = p.get('side', 'BUY').upper()
            if side not in ['BUY', 'SELL']: side = 'BUY'
            
            if side == 'SELL' and not CONFIG["ALLOW_SHORTS"]:
                 self.db.log_decision(symbol, 0, p.get('reason',''), "SKIP", "SHORTS_DISABLED", ai_raw=raw_text)
                 continue

            conf, thesis = p.get('confidence', 0), p.get('reason', 'N/A')
            if not symbol: continue
            
            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                self.db.log_decision(symbol, conf, thesis, "SKIP", "COOLDOWN", ai_raw=raw_text)
                continue

            daily_losses = self.db.get_symbol_daily_losses(symbol)
            if daily_losses >= CONFIG["MAX_LOSSES_PER_SYMBOL"]:
                 self.db.log_decision(symbol, conf, thesis, "SKIP", f"TOXIC_SYMBOL_BAN (Losses:{daily_losses})", ai_raw=raw_text)
                 continue

            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=60)
                if not bars or len(bars) < 55:
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "NO_DATA", ai_raw=raw_text)
                    continue

                entry = bars[-1].c
                atr = self.calculate_atr(bars, period=CONFIG["ATR_PERIOD"])
                regime = self.analyze_market_regime(bars)

                if atr == 0: continue
                if regime == "CHOP":
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "REGIME_CHOP", ai_raw=raw_text)
                    continue
                
                is_macro_mode = False
                shadow_wins, shadow_max_conf = self.db.get_shadow_promotion_stats(symbol)
                is_shadow_promoted = (shadow_wins >= CONFIG["SHADOW_PROMOTION_MIN_WINS"]) or \
                                     (shadow_wins >= 1 and shadow_max_conf >= CONFIG["SHADOW_PROMOTION_HIGH_CONF"])
                
                # V8.6.8: Scout Trade Override
                is_scout_trade = False
                if is_scout_mode_eligible and not is_shadow_promoted and not is_macro_mode:
                    if conf > 78: # Good enough for a scout
                         is_scout_trade = True
                         is_scout_mode_eligible = False # One scout per day logic consumed
                
                if conf >= CONFIG["MACRO_THRESHOLD"] or is_shadow_promoted:
                    is_macro_mode = True
                    regime_settings = CONFIG["REGIME_CONFIG"]["MACRO"]
                    mode_tag = "MACRO"
                else:
                    regime_settings = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                    mode_tag = regime
                
                tp_mult = regime_settings["TP_MULT"]
                sl_mult = regime_settings["SL_MULT"]

                sl_dist = atr * sl_mult
                tp_dist = atr * tp_mult

                if side == "BUY":
                    tp = round(entry + tp_dist, 2)
                    sl = round(entry - sl_dist, 2)
                    if tp <= entry: tp = round(entry + 0.01, 2)
                    if sl >= entry: sl = round(entry - 0.01, 2)
                    sl_dist = round(entry - sl, 2)
                else: # SELL
                    tp = round(entry - tp_dist, 2)
                    sl = round(entry + sl_dist, 2)
                    if tp >= entry: tp = round(entry - 0.01, 2)
                    if sl <= entry: sl = round(entry + 0.01, 2)
                    sl_dist = round(sl - entry, 2)
                
                if not is_macro_mode:
                    if sl_dist < CONFIG["MIN_SL_DISTANCE_USD"]:
                        self.db.log_decision(symbol, conf, thesis, "SKIP", "SL_TOO_TIGHT", ai_raw=raw_text)
                        continue
                
                if sl_dist < (entry * 0.001): continue
                
                current_equity = self.status["equity"]["current"]
                if current_equity <= 0: current_equity = 1000.0
                
                # V8.6.8: Risk Adjustment (Scout or Degraded)
                base_risk_pct = CONFIG["RISK_PCT_PER_TRADE"]
                if self.status["health"] == "DEGRADED":
                    base_risk_pct *= 0.5 # Half risk in degraded mode
                    capped_reason += " [DEGRADED_MODE]"
                elif is_scout_trade:
                    base_risk_pct *= CONFIG["SCOUT_RISK_FACTOR"]
                    capped_reason += " [SCOUT_RISK]"
                
                risk_amount_usd = current_equity * (base_risk_pct / 100.0)
                raw_qty = math.floor(risk_amount_usd / sl_dist)
                
                current_bp_snapshot = self.status["equity"]["buying_power"]
                if current_bp_snapshot <= 0: current_bp_snapshot = current_equity
                
                effective_bp = current_bp_snapshot - intra_loop_committed_cash
                max_affordable_qty = math.floor((effective_bp * 0.95) / entry)
                qty = min(raw_qty, max_affordable_qty)
                
                force_shadow = False
                force_reason = ""
                
                if (qty * entry) < CONFIG["MIN_TRADE_AMOUNT_USD"]:
                    if not is_macro_mode and not is_scout_trade:
                        force_shadow = True
                        force_reason = f"DUST_RETAIL (<${CONFIG['MIN_TRADE_AMOUNT_USD']})"
                    else:
                        capped_reason += " [Micro-Pos Allowed]"

                if qty < raw_qty:
                    capped_reason += f" [Cash Drag: Risk ${round(qty*sl_dist, 2)}]"
                
                if qty < 1 and not force_shadow:
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "INSUFFICIENT_FUNDS", ai_raw=raw_text)
                    continue

                new_trade_risk = qty * sl_dist
                potential_total_risk = current_open_risk + intra_loop_committed_risk + new_trade_risk
                max_allowed_risk = current_equity * (CONFIG["MAX_TOTAL_RISK_PCT"] / 100.0)
                
                if potential_total_risk > max_allowed_risk:
                    force_shadow = True
                    force_reason = f"RISK_CAP_BREACH (Total Risk > {CONFIG['MAX_TOTAL_RISK_PCT']}%)"

                if is_macro_mode and current_macro_count >= CONFIG["MAX_MACRO_POSITIONS"]:
                    force_shadow = True
                    force_reason = "MACRO_SLOT_FULL (Sniper Mode)"
                    
                # V8.6.8: Degraded Mode Cap
                if self.status["health"] == "DEGRADED" and (len(current_open_trades) >= 1):
                    force_shadow = True
                    force_reason = "DEGRADED_MODE_CAP (Max 1 Pos)"

                force_live_trigger = (is_shadow_promoted or (conf >= CONFIG["MACRO_THRESHOLD"]) or is_scout_trade)
                
                can_live = (
                    (conf >= CONFIG["LIVE_THRESHOLD"] or force_live_trigger) and 
                    self.status["positions"]["live_titan"] < CONFIG["MAX_OPEN_POSITIONS"] and 
                    not force_shadow
                )
                
                if self.status["safety"]["market_stress"]: can_live = False

                log_msg = f"Side:{side} Mode:{mode_tag} ATR:{round(atr,2)} {capped_reason}"
                if is_shadow_promoted: log_msg += " [SHADOW_PROMOTED]"
                if is_scout_trade: log_msg += " [SCOUT_TRADE]"

                if can_live:
                    dec_id = self.db.log_decision(symbol, conf, thesis, "LIVE", log_msg, ai_raw=raw_text)
                    
                    try:
                        order = self.alpaca.submit_order(
                            symbol=symbol, qty=qty, side=side.lower(), type='market', time_in_force='gtc', 
                            order_class='bracket', 
                            take_profit={'limit_price': tp}, stop_loss={'stop_price': sl}
                        )
                    except APIError as e:
                        self.db.log_decision(symbol, conf, thesis, "LIVE_API_ERROR", str(e), ai_raw=raw_text)
                        await self.notifier.send_rejection(symbol, "API_ERROR", str(e))
                        continue

                    if order.status not in ['new', 'accepted', 'pending_new', 'accepted_for_bidding', 'filled', 'partially_filled']:
                         error_msg = f"Broker Rejected: {order.status}"
                         self.db.log_decision(symbol, conf, thesis, "LIVE_REJECTED", error_msg, ai_raw=raw_text)
                         await self.notifier.send_rejection(symbol, order.status, "Immediate Rejection Post-Submit")
                         continue
                    
                    estimated_cost = qty * entry
                    intra_loop_committed_cash += estimated_cost
                    intra_loop_committed_risk += new_trade_risk
                    self.status["positions"]["live_titan"] += 1 
                    if is_macro_mode: current_macro_count += 1 

                    adaptive_tag = "MACRO" if is_macro_mode else "INIT"
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "LIVE", tp, sl, dec_id, order.id, atr, regime, side, adaptive_tag)
                    
                    logger.info(f"LIVE [{mode_tag}]: {symbol} {side} Qty:{qty} TP:{tp} SL:{sl} {capped_reason}")
                    await self.notifier.send_trade_entry(symbol, side, qty, entry, thesis, mode_tag, scout=is_scout_trade)
                else:
                    rej = "STRESS" if self.status["safety"]["market_stress"] else "CONF/LIMIT"
                    if force_shadow: rej = force_reason 
                    
                    dec_id = self.db.log_decision(symbol, conf, thesis, "SHADOW", f"{rej} ({log_msg})", ai_raw=raw_text)
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "SHADOW", tp, sl, dec_id, None, atr, regime, side, "INIT")
                
                self.last_trade_per_symbol[symbol] = datetime.now()
                
            except Exception as e: 
                logger.error(f"Trade error {symbol}: {e}")

    async def main_loop(self):
        while True:
            try:
                await self.sync_data()
                is_halted, _, _ = self.db.get_system_state()
                if not is_halted:
                    if self.status["market"] == "OPEN":
                        self.status["state"] = "SCANNING"
                        await self.run_logic()
                    else:
                        self.status["state"] = "IDLE"
                        await self.reconcile_trades()
                else: self.status["state"] = "HALTED"
            except Exception as e: logger.error(f"Loop crash: {e}")
            await asyncio.sleep(CONFIG["SCAN_INTERVAL"])

# --- API ENDPOINTS ---
async def api_status(request): return web.json_response(request.app['titan'].status)
async def api_decisions(request):
    limit = int(request.query.get("limit", 20))
    with sqlite3.connect(CONFIG["DB_PATH"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM ai_decision_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return web.json_response([dict(r) for r in rows])
async def api_resume(request):
    request.app['titan'].db.set_system_state(False, "", "HEALTHY")
    await request.app['titan'].notifier.send("▶️ System Resumed", "Manual resume triggered via API. Health reset.", priority="WARNING")
    return web.json_response({"status": "resumed"})

@web.middleware
async def auth_middleware(request, handler):
    if request.method == "OPTIONS": return await handler(request)
    if request.headers.get("Authorization") != f"Bearer {API_TOKEN}":
        return web.json_response({"error": "Auth"}, status=401)
    return await handler(request)

async def main():
    titan = TitanEngine()
    app = web.Application(middlewares=[auth_middleware])
    app['titan'] = titan
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")})
    cors.add(app.router.add_get('/', api_status))
    cors.add(app.router.add_get('/status', api_status))
    cors.add(app.router.add_get('/decisions', api_decisions))
    cors.add(app.router.add_post('/resume', api_resume))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', CONFIG["PORT"]).start()
    logger.info(f"Titan-Active-Learner v8.6.8 Ready. Scout Mode & MAE/MFE Tracking Active.")
    await titan.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
