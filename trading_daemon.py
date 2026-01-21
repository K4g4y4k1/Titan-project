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

# --- CONFIGURATION V8.2 ---
load_dotenv()

API_TOKEN = os.getenv('TITAN_DASHBOARD_TOKEN')
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', "")

CONFIG = {
    "VERSION": "8.2.0-Context-Aware",
    "PORT": 8080,
    "DB_PATH": "titan_v8_recon.db",
    "MAX_OPEN_POSITIONS": 3,
    "DOLLAR_RISK_PER_TRADE": 50.0,
    "LIVE_THRESHOLD": 82,
    "MARKET_STRESS_THRESHOLD": 1.5,
    "COOLDOWN_PER_SYMBOL_MIN": 15,
    # --- PARAMETRES ATR & REGIME (v8.2) ---
    "ATR_PERIOD": 14,
    # Les multiplicateurs sont maintenant dynamiques selon le régime
    "REGIME_CONFIG": {
        "TREND": {"TP_MULT": 3.0, "SL_MULT": 1.5, "DESC": "Trend following"},
        "RANGE": {"TP_MULT": 1.5, "SL_MULT": 1.0, "DESC": "Mean reversion"},
        "CHOP":  {"TP_MULT": 0.0, "SL_MULT": 0.0, "DESC": "No trade zone"} 
    },
    # --------------------------------------
    "ENV_MODE": os.getenv('ENV_MODE', 'PAPER'),
    "AI_MODEL": "deepseek/deepseek-v3.2",
    "SCAN_INTERVAL": 60 
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("titan_v8_2.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Titan-Context")

# --- UTILITAIRES DE PARSING ---
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

# --- PERSISTANCE (V8.2) ---
class TitanDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()
        self._migrate_v8_2()

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
                    atr_at_entry REAL
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

    def _migrate_v8_2(self):
        """Migration v8.2: Ajout colonne Regime."""
        with sqlite3.connect(self.db_path) as conn:
            # v8.1 column
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN atr_at_entry REAL")
            except sqlite3.OperationalError: pass
            
            # v8.2 column
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN market_regime TEXT")
                logger.info("Migration DB: Colonne 'market_regime' ajoutée.")
            except sqlite3.OperationalError: pass

    def log_trade(self, symbol, qty, price, conf, thesis, mode, tp, sl, dec_id, order_id=None, atr=0.0, regime="UNKNOWN"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO trades 
                (symbol, qty, entry_price, confidence, thesis, mode, tp_price, sl_price, status, decision_id, alpaca_order_id, atr_at_entry, market_regime) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, qty, price, conf, thesis, mode, tp, sl, 'OPEN', dec_id, order_id, atr, regime))
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
            if mode:
                return conn.execute("SELECT * FROM trades WHERE status='OPEN' AND mode=?", (mode,)).fetchall()
            return conn.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()

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
            last_dec = conn.execute("SELECT symbol, action, reason FROM ai_decision_log ORDER BY id DESC LIMIT 1").fetchone()
            res['last_action'] = f"{last_dec[0]}: {last_dec[1]}" if last_dec else "N/A"
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
        """
        Détermine le régime de marché (TREND, RANGE, CHOP).
        Utilise SMA Alignment et Volatilité relative.
        """
        if len(bars) < 50: return "UNCERTAIN" # Besoin d'historique pour SMA50

        prices = [b.c for b in bars]
        
        sma20 = self._calculate_sma(prices, 20)
        sma50 = self._calculate_sma(prices, 50)
        
        if not sma20 or not sma50: return "UNCERTAIN"

        current_price = prices[-1]
        
        # 1. Trend Filter: Alignement SMA
        # Uptrend: Prix > SMA20 > SMA50
        is_uptrend = current_price > sma20 > sma50
        # Downtrend: Prix < SMA20 < SMA50
        is_downtrend = current_price < sma20 < sma50
        
        # 2. Volatility Context (Range vs Trend strength)
        # On regarde si les SMAs sont proches (Compression/Range) ou écartées (Expansion/Trend)
        sma_spread = abs(sma20 - sma50) / sma50 * 100
        
        if sma_spread < 0.5: 
            # SMAs très proches = Compression = RANGE ou CHOP
            # Si le prix oscille autour des SMAs sans direction claire
            return "RANGE"
        elif is_uptrend or is_downtrend:
            return "TREND"
        else:
            return "CHOP" # Désordre (Prix > SMA50 mais < SMA20 etc.)

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
            
            clock = self.alpaca.get_clock()
            stats = self.db.get_stats()
            
            self.status.update({
                "market": "OPEN" if clock.is_open else "CLOSED",
                "equity": {"current": round(eq, 2), "pnl_pct": round(pnl_pct, 2)},
                "positions": {
                    "live": len(self.alpaca.list_positions()), 
                    "shadow_open": stats['shadow_open_count']
                },
                "safety": {"consecutive_sl": stats['consecutive_sl']},
                "omni": stats
            })
        except Exception as e: logger.error(f"Sync error: {e}")

    async def reconcile_trades(self):
        # 1. SHADOW RECON
        shadow_trades = self.db.get_open_trades(mode='SHADOW')
        for t in shadow_trades:
            try:
                bars = self.alpaca.get_bars(t['symbol'], TimeFrame.Minute, limit=1)
                if not bars: continue
                price = bars[0].c
                if price >= t['tp_price']: self.db.close_trade(t['id'], price, "TP")
                elif price <= t['sl_price']: self.db.close_trade(t['id'], price, "SL")
            except Exception: pass

        # 2. LIVE RECON
        live_trades = self.db.get_open_trades(mode='LIVE')
        if not live_trades: return
        
        positions = {p.symbol: p for p in self.alpaca.list_positions()}
        
        for t in live_trades:
            if t['symbol'] in positions: continue

            # Vérification Fills et Statut Ordre
            activities = self.alpaca.get_activities(activity_types='FILL')
            symbol_fills = [f for f in activities if f.symbol == t['symbol']]
            
            if t['alpaca_order_id']:
                try:
                    order = self.alpaca.get_order(t['alpaca_order_id'])
                    if order.status in ['canceled', 'expired', 'rejected']:
                        self.db.close_trade(t['id'], 0, "VOID")
                        continue
                except Exception: pass

            if symbol_fills:
                exit_price = float(symbol_fills[0].price) 
                res = "TP" if exit_price >= t['entry_price'] else "SL"
                self.db.close_trade(t['id'], exit_price, res)

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
        
        # 1. Market Stress Check
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
            
            if symbol in self.last_trade_per_symbol and (datetime.now() - self.last_trade_per_symbol[symbol]) < timedelta(minutes=CONFIG["COOLDOWN_PER_SYMBOL_MIN"]):
                self.db.log_decision(symbol, conf, thesis, "SKIP", "COOLDOWN", ai_raw=raw_text)
                continue

            try:
                # 2. Context Analysis (v8.2: Besoin de 60 bars pour SMA50 + ATR)
                bars = self.alpaca.get_bars(symbol, TimeFrame.Minute, limit=60)
                if not bars or len(bars) < 55:
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "NO_DATA", ai_raw=raw_text)
                    continue

                entry = bars[-1].c
                atr = self.calculate_atr(bars, period=CONFIG["ATR_PERIOD"])
                regime = self.analyze_market_regime(bars) # Nouveau en v8.2

                if atr == 0: continue

                # 3. Regime-based Parameters
                if regime == "CHOP":
                    self.db.log_decision(symbol, conf, thesis, "SKIP", "REGIME_CHOP", ai_raw=raw_text)
                    continue
                
                # Sélection des multiplicateurs selon le régime
                regime_settings = CONFIG["REGIME_CONFIG"].get(regime, CONFIG["REGIME_CONFIG"]["RANGE"])
                tp_mult = regime_settings["TP_MULT"]
                sl_mult = regime_settings["SL_MULT"]

                sl_dist = atr * sl_mult
                tp_dist = atr * tp_mult
                
                tp = round(entry + tp_dist, 2)
                sl = round(entry - sl_dist, 2)
                
                if sl_dist < (entry * 0.001): continue # SL trop serré

                # Risk Sizing
                qty = math.floor(CONFIG["DOLLAR_RISK_PER_TRADE"] / sl_dist)
                if qty < 1: continue

                # 4. Décision & Execution
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
    logger.info(f"Titan-Context v8.2 Ready. Regime Detection Active.")
    await titan.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
