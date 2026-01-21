import os
import asyncio
import logging
import json
import sqlite3
import aiohttp
import re
import math
import statistics
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from aiohttp import web
import aiohttp_cors

# --- CONFIGURATION V8.4 ---
load_dotenv()

API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN')
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")

CONFIG = {
    "VERSION": "8.4.0-Risk-Governor",
    "PORT": 8080,
    "DB_PATH": "titan_v8_recon.db",
    "MAX_OPEN_POSITIONS": 3,
    "DOLLAR_RISK_PER_TRADE": 50.0,
    "LIVE_THRESHOLD": 82,
    "MARKET_STRESS_THRESHOLD": 1.5,
    "COOLDOWN_PER_SYMBOL_MIN": 15,
    "ATR_PERIOD": 14,
    "REGIME_CONFIG": {
        "TREND": {"TP_MULT": 3.0, "SL_MULT": 1.5, "DESC": "Trend following", "TRAILING": True},
        "RANGE": {"TP_MULT": 1.5, "SL_MULT": 1.0, "DESC": "Mean reversion", "TRAILING": False},
        "CHOP":  {"TP_MULT": 0.0, "SL_MULT": 0.0, "DESC": "No trade zone", "TRAILING": False} 
    },
    "BE_TRIGGER_ATR": 1.0,
    "TRAILING_STEP_ATR": 0.5,
    # --- GOUVERNANCE DES RISQUES (v8.4) ---
    "MAX_DAILY_DRAWDOWN_PCT": -4.0,   # Arrêt d'urgence si perte > 4%
    "MAX_LOSSES_PER_SYMBOL": 2,       # Ban symbole après 2 pertes le même jour
    "MAX_SL_UPDATES_PER_TRADE": 20,   # Sécurité API (héritage 8.3.1)
    # -------------------------------------
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "deepseek/deepseek-v3.2",
    "SCAN_INTERVAL": 60 
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("titan_v8_4.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Titan-Governor")

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

# --- PERSISTANCE (V8.4) ---
class TitanDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()
        self._migrate_v8_4()

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
                    highest_price REAL, is_be_active INTEGER DEFAULT 0,
                    sl_updates_count INTEGER DEFAULT 0
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
            conn.commit()

    def _migrate_v8_4(self):
        """Migration v8.4: S'assurer que les colonnes de robustesse sont là."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN sl_updates_count INTEGER DEFAULT 0")
                logger.info("Migration DB: Colonne 'sl_updates_count' ajoutée.")
            except sqlite3.OperationalError: pass
            try: conn.execute("ALTER TABLE trades ADD COLUMN highest_price REAL") 
            except: pass
            try: conn.execute("ALTER TABLE trades ADD COLUMN is_be_active INTEGER DEFAULT 0") 
            except: pass

    def log_trade(self, symbol, qty, price, conf, thesis, mode, tp, sl, dec_id, order_id=None, atr=0.0, regime="UNKNOWN"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status, decision_id, alpaca_order_id, atr_at_entry, market_regime, highest_price, sl_updates_count) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN', dec_id, order_id, atr, regime, price))
            conn.commit()

    def update_trade_state(self, trade_id, highest_price, new_sl, be_active, update_count):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE trades SET highest_price=?, sl_price=?, is_be_active=?, sl_updates_count=? WHERE id=?
            """, (highest_price, new_sl, 1 if be_active else 0, update_count, trade_id))
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
        """Compte les SL pour un symbole aujourd'hui."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE symbol=? AND result='SL' AND date(exit_time) = date('now')
            """, (symbol,)).fetchone()[0]
            return count

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
            return res

    def get_halt_state(self):
        with sqlite3.connect(self.db_path) as conn:
            is_halted = conn.execute("SELECT value FROM system_state WHERE key = 'is_halted'").fetchone()[0]
            reason = conn.execute("SELECT value FROM system_state WHERE key = 'halt_reason'").fetchone()[0]
            return (is_halted == '1', reason)

    def set_halt_state(self, halted, reason=""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'is_halted'", ('1' if halted else '0',))
            conn.execute("UPDATE system_state SET value = ? WHERE key = 'halt_reason'", (reason,))
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
        self.session = None
        self.last_trade_per_symbol = {}
        
        is_halted, reason = self.db.get_halt_state()
        self.status = {
            "version": CONFIG["VERSION"],
            "state": "HALTED" if is_halted else "INIT",
            "market": "CLOSED",
            "halt_reason": reason,
            "equity": {"current": 0.0, "pnl_pct": 0.0},
            "safety": {"consecutive_sl": 0, "market_stress": False},
            "positions": {"live": 0, "shadow_open": 0},
            "omni": {"decisions_today": 0, "shadow_winrate": 0, "last_action": "N/A"}
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

    async def sync_data(self):
        try:
            acc = self.alpaca.get_account()
            eq = float(acc.equity)
            start_eq = self.db.get_or_create_daily_stats(eq)
            pnl_pct = ((eq - start_eq) / start_eq * 100) if start_eq > 0 else 0.0
            
            # --- GOVERNOR CHECK (v8.4) ---
            if pnl_pct <= CONFIG["MAX_DAILY_DRAWDOWN_PCT"]:
                is_halted, _ = self.db.get_halt_state()
                if not is_halted:
                    reason = f"MAX DRAWDOWN HIT ({round(pnl_pct, 2)}%)"
                    self.db.set_halt_state(True, reason)
                    logger.critical(f"🛑 SYSTEM HALTED: {reason}")
            # -----------------------------

            clock = self.alpaca.get_clock()
            stats = self.db.get_stats()
            self.status.update({
                "market": "OPEN" if clock.is_open else "CLOSED",
                "equity": {"current": round(eq, 2), "pnl_pct": round(pnl_pct, 2)},
                "positions": {"live": len(self.alpaca.list_positions()), "shadow_open": stats['shadow_open_count']},
                "safety": {"consecutive_sl": stats['consecutive_sl']},
                "omni": stats
            })
        except Exception as e: logger.error(f"Sync error: {e}")

    async def manage_trade_lifecycle(self, t, current_price, regime_cfg):
        """Gestionnaire de SL Dynamique (Durci v8.4)"""
        if t['status'] != 'OPEN': return

        # Security: Anti-Spam API
        current_updates = t['sl_updates_count'] or 0
        if current_updates >= CONFIG["MAX_SL_UPDATES_PER_TRADE"]:
            return

        highest = max(t['highest_price'] or t['entry_price'], current_price)
        atr = t['atr_at_entry']
        if atr <= 0: return
        
        entry = t['entry_price']
        current_sl = t['sl_price']
        new_sl = current_sl
        update_needed = False
        is_be = bool(t['is_be_active'])

        # Logique BE
        if not is_be and (current_price >= entry + (atr * CONFIG["BE_TRIGGER_ATR"])):
            new_sl = entry 
            is_be = True
            update_needed = True
            logger.info(f"🛡️ BREAK-EVEN Triggered for {t['symbol']} at {current_price}")

        # Logique Trailing
        if regime_cfg["TRAILING"] and is_be:
            trail_dist = atr * regime_cfg["SL_MULT"]
            potential_sl = highest - trail_dist
            if potential_sl > (new_sl + (atr * CONFIG["TRAILING_STEP_ATR"])):
                new_sl = round(potential_sl, 2)
                update_needed = True
                logger.info(f"📈 TRAILING UP for {t['symbol']}: New SL {new_sl}")

        # Exécution Sécurisée
        if update_needed and new_sl > current_sl:
            success = False
            if t['mode'] == 'LIVE':
                try:
                    orders = self.alpaca.list_orders(status='open', symbols=[t['symbol']])
                    sl_order = next((o for o in orders if o.type == 'stop' and o.parent_id == t['alpaca_order_id']), None)
                    # Fallback robuste
                    if not sl_order:
                        sl_order = next((o for o in orders if o.type == 'stop'), None)

                    if sl_order:
                        self.alpaca.replace_order(sl_order.id, stop_price=new_sl)
                        success = True
                        logger.info(f"✅ ALPACA ORDER UPDATED {t['symbol']} SL -> {new_sl}")
                    else:
                        logger.error(f"❌ CRITICAL: SL Order NOT FOUND for {t['symbol']}")
                except Exception as e:
                    logger.error(f"❌ API ERROR updating {t['symbol']}: {e}")
            else:
                success = True

            if success:
                self.db.update_trade_state(t['id'], highest, new_sl, is_be, current_updates + 1)

    async def reconcile_trades(self):
        trades = self.db.get_open_trades()
        if not trades: return

        live_positions = {p.symbol: p for p in self.alpaca.list_positions()} if self.alpaca else {}

        for t in trades:
            symbol = t['symbol']
            
            # PHASE 1: Vérification Fermeture
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
                    res = "TP" if exit_price >= t['entry_price'] else "SL"
                    self.db.close_trade(t['id'], exit_price, res)
                continue

            # PHASE 2: Gestion Active
            try:
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
                if not bars: continue
                current_price = bars[0].c

                if t['mode'] == 'SHADOW':
                    if current_price >= t['tp_price']: self.db.close_trade(t['id'], current_price, "TP")
                    elif current_price <= t['sl_price']: self.db.close_trade(t['id'], current_price, "SL")
                    else:
                        regime = t['market_regime'] if 'market_regime' in t.keys() else "RANGE"
                        cfg = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                        await self.manage_trade_lifecycle(t, current_price, cfg)

                elif t['mode'] == 'LIVE':
                    regime = t['market_regime'] if 'market_regime' in t.keys() else "RANGE"
                    cfg = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                    await self.manage_trade_lifecycle(t, current_price, cfg)

            except Exception as e:
                logger.error(f"Recon error on {symbol}: {e}")

    async def fetch_ai_picks(self):
        s = await self.get_session()
        prompt = (
            "Analyze US market structure. Return ONLY JSON: "
            "{'picks': [{'symbol': 'TICKER', 'confidence': 95, 'reason': 'short thesis'}]}. "
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
        
        ai_data = await self.fetch_ai_picks()
        picks, raw_text = ai_data.get("picks", []), ai_data.get("raw", "")

        if not picks:
            self.db.log_decision("SYSTEM", 0, f"Heartbeat (Vol:{round(spy_vol,2)}%).", "IDLE", ai_raw=raw_text)
            return

        for p in picks:
            symbol = p.get('symbol', '').upper()
            conf, thesis = p.get('confidence', 0), p.get('reason', 'N/A')
            if not symbol: continue
            
            # --- GOVERNOR CHECKS (v8.4) ---
            
            # 1. Cooldown Temporel
            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                self.db.log_decision(symbol, conf, thesis, "SKIP", "COOLDOWN", ai_raw=raw_text)
                continue

            # 2. Blacklist Journalière (Toxic Symbol)
            daily_losses = self.db.get_symbol_daily_losses(symbol)
            if daily_losses >= CONFIG["MAX_LOSSES_PER_SYMBOL"]:
                 self.db.log_decision(symbol, conf, thesis, "SKIP", f"TOXIC_SYMBOL_BAN (Losses:{daily_losses})", ai_raw=raw_text)
                 continue
            # ------------------------------

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
                
                regime_settings = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                tp_mult = regime_settings["TP_MULT"]
                sl_mult = regime_settings["SL_MULT"]

                sl_dist = atr * sl_mult
                tp_dist = atr * tp_mult
                tp = round(entry + tp_dist, 2)
                sl = round(entry - sl_dist, 2)
                
                if sl_dist < (entry * 0.001): continue

                qty = math.floor(CONFIG["DOLLAR_RISK_PER_TRADE"] / sl_dist)
                if qty < 1: continue

                can_live = (conf >= CONFIG["LIVE_THRESHOLD"] and 
                           self.status["positions"]["live"] < CONFIG["MAX_OPEN_POSITIONS"] and 
                           not self.status["safety"]["market_stress"])
                
                log_msg = f"Regime:{regime} ATR:{round(atr,2)}"

                if can_live:
                    dec_id = self.db.log_decision(symbol, conf, thesis, "LIVE", log_msg, ai_raw=raw_text)
                    order = self.alpaca.submit_order(
                        symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc', 
                        order_class='bracket', 
                        take_profit={'limit_price': tp}, stop_loss={'stop_price': sl}
                    )
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "LIVE", tp, sl, dec_id, order.id, atr, regime)
                    logger.info(f"LIVE [{regime}]: {symbol} Qty:{qty} TP:{tp} SL:{sl}")
                else:
                    rej = "STRESS" if self.status["safety"]["market_stress"] else "CONF/LIMIT"
                    dec_id = self.db.log_decision(symbol, conf, thesis, "SHADOW", f"{rej} ({log_msg})", ai_raw=raw_text)
                    self.db.log_trade(symbol, qty, entry, conf, thesis, "SHADOW", tp, sl, dec_id, None, atr, regime)
                
                self.last_trade_per_symbol[symbol] = datetime.now()
                
            except Exception as e: 
                logger.error(f"Trade error {symbol}: {e}")

    async def main_loop(self):
        while True:
            try:
                await self.sync_data()
                is_halted, _ = self.db.get_halt_state()
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
    request.app['titan'].db.set_halt_state(False)
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
    logger.info(f"Titan-Governor v8.4 Ready. Global Risk Controls Active.")
    await titan.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
