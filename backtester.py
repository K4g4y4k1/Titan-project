import pandas as pd
import yfinance as yf
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION STRICTE TITAN 5.6.11 ---
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
INITIAL_CAPITAL = 100000
SYMBOLS_TO_TEST = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "META", "NFLX", "AMZN"]
CACHE_DIR = "market_data_cache"

# GOUVERNANCE STRICTE (Issue de ton code source)
GOUVERNANCE = {
    "MAX_SECTOR_EXPOSURE_PCT": 0.25,
    "MAX_POSITION_SIZE_PCT": 0.10,
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "SLIPPAGE_PROTECTION": 0.002,
    "MAX_HOLDING_DAYS": 3,
    "MODES": {
        "EXPLOITATION": { "MIN_SCORE": 85, "MAX_SIGMA": 20, "BASE_RISK": 0.01 },
        "EXPLORATION": { "MIN_SCORE": 72, "MAX_SIGMA": 35, "BASE_RISK": 0.0025 }
    }
}

class DataEngine:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    def get_data(self, symbol, start, end):
        cache_path = os.path.join(self.cache_dir, f"{symbol}_{start}_{end}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty: return df

        print(f"Extraction de {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            time.sleep(1.2)
            df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.to_csv(cache_path)
            return df
        except Exception as e:
            print(f"❌ Erreur {symbol}: {e}")
            return pd.DataFrame()

class SimulatedBrain:
    """Simule Grok-2 tel que défini dans ton TitanEngine."""
    @staticmethod
    def get_ai_score(symbol, date):
        # Simulation déterministe basée sur la date et le symbole
        seed = int(date.timestamp()) + sum(ord(c) for c in symbol)
        np.random.seed(seed % 2**32)
        score = np.random.normal(78, 10) # Moyenne proche de tes seuils
        sigma = np.random.uniform(10, 30)
        return score, sigma

class TitanBacktester:
    def __init__(self, symbols, initial_capital):
        self.symbols = symbols
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.engine = DataEngine(CACHE_DIR)
        self.positions = []
        self.history = []
        self.all_data = {}

    def prepare(self):
        valid_symbols = []
        for sym in self.symbols:
            df = self.engine.get_data(sym, START_DATE, END_DATE)
            if not df.empty:
                self.all_data[sym] = df
                valid_symbols.append(sym)
        self.symbols = valid_symbols

    def run(self):
        if not self.all_data: return
        timeline = self.all_data[self.symbols[0]].index
        
        for current_date in timeline:
            self._update_portfolio(current_date)
            self._scan_opportunities(current_date)

        self._print_final_report()

    def _update_portfolio(self, current_date):
        active_positions = []
        for pos in self.positions:
            sym = pos['symbol']
            if current_date not in self.all_data[sym].index:
                active_positions.append(pos); continue

            day = self.all_data[sym].loc[current_date]
            exit_price = None
            reason = ""

            # Logique de sortie STRICTE (TP/SL Bracket)
            if day['Low'] <= pos['sl']:
                exit_price = pos['sl']
                reason = "STOP_LOSS"
            elif day['High'] >= pos['tp']:
                exit_price = pos['tp']
                reason = "TAKE_PROFIT"
            elif (current_date - pos['entry_date']).days >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                exit_price = day['Close']
                reason = "TIME_EXIT"

            if exit_price:
                pnl = (exit_price - pos['entry_price']) * pos['qty']
                self.capital += (exit_price * pos['qty'])
                self.history.append({'symbol': sym, 'pnl': pnl, 'reason': reason})
            else:
                active_positions.append(pos)
        self.positions = active_positions

    def _scan_opportunities(self, current_date):
        for sym in self.symbols:
            if any(p['symbol'] == sym for p in self.positions): continue
            
            score, sigma = SimulatedBrain.get_ai_score(sym, current_date)
            
            # Logique de sélection de mode STRICTE
            mode = "EXPLOITATION" if score >= 85 and sigma <= 20 else "EXPLORATION" if score >= 72 and sigma <= 35 else None
            
            if mode:
                price = self.all_data[sym].loc[current_date]['Close']
                risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"]
                
                # Calcul de QTY identique à ton code (Audit #1/#2)
                qty = min(int((self.capital * risk) / (price * GOUVERNANCE["BASE_SL_PCT"])), 
                          int((self.capital * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / price))

                if qty > 0 and self.capital >= (qty * price):
                    # Application du slippage de ton code
                    entry_p = price * (1 + GOUVERNANCE["SLIPPAGE_PROTECTION"])
                    self.capital -= (entry_p * qty)
                    self.positions.append({
                        'symbol': sym, 'entry_date': current_date, 'entry_price': entry_p,
                        'qty': qty, 'tp': entry_p * (1 + GOUVERNANCE["BASE_TP_PCT"]),
                        'sl': entry_p * (1 - GOUVERNANCE["BASE_SL_PCT"]), 'mode': mode
                    })

    def _print_final_report(self):
        df_hist = pd.DataFrame(self.history)
        current_val = self.capital + sum(p['qty'] * self.all_data[p['symbol']].iloc[-1]['Close'] for p in self.positions)
        print("\n" + "="*45)
        print(" 🛡️  TITAN 5.6.11 - STRICT LOGIC REPORT")
        print("="*45)
        print(f"Valeur Finale: ${current_val:,.2f}")
        print(f"Profit Total : ${current_val - self.initial_capital:,.2f}")
        if not df_hist.empty:
            print(f"Win Rate     : {(len(df_hist[df_hist['pnl'] > 0]) / len(df_hist)) * 100:.2f}%")
            print("\nRépartition des sorties:")
            print(df_hist['reason'].value_counts())
        print("="*45)

if __name__ == "__main__":
    tester = TitanBacktester(SYMBOLS_TO_TEST, INITIAL_CAPITAL)
    tester.prepare()
    tester.run()
