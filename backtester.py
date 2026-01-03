import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json

# --- CONFIGURATION DU BACKTEST (Alignée sur v5.6.11-LTS) ---
BACKTEST_CONFIG = {
    "CASH_INITIAL": 100000,
    "STRATEGY_VERSION": "5.6.11-LTS (Grok Sentinel)",
    "LOOKBACK_DAYS": 90,           # Fenêtre d'analyse historique
    "HOLDING_DAYS": 3,              # Horizon PEAD 3-jours (Audit Grok)
    "BASE_TP_PCT": 0.06,            # Take Profit 6%
    "BASE_SL_PCT": 0.03,            # Stop Loss 3%
    "MAX_SECTOR_EXPOSURE": 0.25,    # Max 25% par secteur
    "MAX_POS_SIZE": 0.10,           # Max 10% par position
    "SLIPPAGE": 0.001,              # 0.1% de slippage estimé
    "MODES": {
        "EXPLOITATION": {"min_score": 85, "max_sigma": 20},
        "EXPLORATION": {"min_score": 72, "max_sigma": 35}
    }
}

class TitanBacktester:
    def __init__(self, tickers):
        self.tickers = tickers
        self.equity = BACKTEST_CONFIG["CASH_INITIAL"]
        self.cash = BACKTEST_CONFIG["CASH_INITIAL"]
        self.positions = [] # Liste des trades actifs
        self.history = []   # Historique des trades fermés
        self.daily_log = [] # Évolution de l'équité
        
    def mock_grok_scoring(self, symbol, earnings_date, actual_drift):
        """
        Émule le moteur Grok 2. 
        En mode backtest, on simule un score basé sur la réussite du drift 
        avec une composante aléatoire pour refléter l'incertitude réelle.
        """
        # Plus le drift réel est positif, plus le score "aurait été" élevé
        base_score = 75 + (actual_drift * 100)
        noise = np.random.normal(0, 5)
        final_score = np.clip(base_score + noise, 0, 100)
        sigma = np.random.uniform(10, 30)
        
        return round(final_score, 2), round(sigma, 2)

    def run(self):
        print(f"🚀 Démarrage Backtest Titan {BACKTEST_CONFIG['STRATEGY_VERSION']}")
        print(f"Intervalle : {BACKTEST_CONFIG['LOOKBACK_DAYS']} jours | Horizon : 3j Drift")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_CONFIG["LOOKBACK_DAYS"])

        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                # Récupération des données de prix
                df = tk.history(start=start_date, end=end_date)
                if df.empty: continue
                
                # Récupération des dates de résultats (Earnings)
                # Note: yfinance peut être capricieux sur les earnings historiques
                calendar = tk.get_calendar()
                if calendar is None or 'Earnings Date' not in calendar:
                    continue
                
                earnings_dates = calendar['Earnings Date']
                
                for e_date in earnings_dates:
                    if not isinstance(e_date, datetime): continue
                    e_date_str = e_date.strftime('%Y-%m-%d')
                    
                    if e_date_str in df.index.strftime('%Y-%m-%d'):
                        entry_price = df.loc[e_date_str]['Close']
                        
                        # Calcul du drift réel à T+3 pour le simulateur IA
                        future_idx = df.index.get_loc(df.loc[e_date_str].name) + BACKTEST_CONFIG["HOLDING_DAYS"]
                        if future_idx >= len(df): continue
                        
                        exit_price_actual = df.iloc[future_idx]['Close']
                        actual_drift = (exit_price_actual - entry_price) / entry_price
                        
                        # Scoring Grok Sentinel
                        score, sigma = self.mock_grok_scoring(ticker, e_date_str, actual_drift)
                        
                        # Attribution du mode
                        mode = None
                        if score >= BACKTEST_CONFIG["MODES"]["EXPLOITATION"]["min_score"] and sigma <= BACKTEST_CONFIG["MODES"]["EXPLOITATION"]["max_sigma"]:
                            mode = "EXPLOITATION"
                        elif score >= BACKTEST_CONFIG["MODES"]["EXPLORATION"]["min_score"] and sigma <= BACKTEST_CONFIG["MODES"]["EXPLORATION"]["max_sigma"]:
                            mode = "EXPLORATION"
                            
                        if mode:
                            self.execute_trade(ticker, entry_price, exit_price_actual, e_date_str, mode, score)
                            
            except Exception as e:
                print(f"⚠️ Erreur sur {ticker}: {e}")

        self.generate_report()

    def execute_trade(self, symbol, entry, target_exit, date, mode, score):
        # Simulation de la gestion des risques
        risk_per_trade = self.equity * BACKTEST_CONFIG["MAX_POS_SIZE"]
        qty = risk_per_trade / entry
        
        # Vérification TP / SL (Simulation simplifiée sur le prix final)
        pnl_pct = (target_exit - entry) / entry
        
        # Application des barrières de la gouvernance v5.6.11
        final_pnl_pct = pnl_pct
        exit_type = "3-DAY-DRIFT"
        
        if pnl_pct >= BACKTEST_CONFIG["BASE_TP_PCT"]:
            final_pnl_pct = BACKTEST_CONFIG["BASE_TP_PCT"]
            exit_type = "TAKE_PROFIT"
        elif pnl_pct <= -BACKTEST_CONFIG["BASE_SL_PCT"]:
            final_pnl_pct = -BACKTEST_CONFIG["BASE_SL_PCT"]
            exit_type = "STOP_LOSS"
            
        # Application du slippage
        final_pnl_pct -= BACKTEST_CONFIG["SLIPPAGE"]
        pnl_usd = (entry * qty) * final_pnl_pct
        
        self.history.append({
            "date": date,
            "symbol": symbol,
            "mode": mode,
            "score": score,
            "entry": round(entry, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(final_pnl_pct * 100, 2),
            "exit_type": exit_type
        })
        
        self.equity += pnl_usd

    def generate_report(self):
        df_res = pd.DataFrame(self.history)
        if df_res.empty:
            print("❌ Aucun trade exécuté durant la période.")
            return

        total_pnl = df_res['pnl_usd'].sum()
        win_rate = (df_res['pnl_usd'] > 0).mean() * 100
        profit_factor = df_res[df_res['pnl_usd'] > 0]['pnl_usd'].sum() / abs(df_res[df_res['pnl_usd'] < 0]['pnl_usd'].sum()) if any(df_res['pnl_usd'] < 0) else np.inf
        
        print("\n" + "="*45)
        print(f"📊 RAPPORT TITAN v5.6.11-LTS")
        print("="*45)
        print(f"Équité Finale    : ${self.equity:,.2f}")
        print(f"PnL Total        : ${total_pnl:,.2f} ({((self.equity/BACKTEST_CONFIG['CASH_INITIAL'])-1)*100:.2f}%)")
        print(f"Win Rate         : {win_rate:.1f}%")
        print(f"Profit Factor    : {profit_factor:.2f}")
        print(f"Nombre de Trades : {len(df_res)}")
        print("-" * 45)
        print("PERFORMANCE PAR MODE :")
        for m in ["EXPLOITATION", "EXPLORATION"]:
            m_df = df_res[df_res['mode'] == m]
            if not m_df.empty:
                avg_score = m_df['score'].mean()
                exp = m_df['pnl_usd'].mean()
                print(f"[{m}] Trades: {len(m_df)} | Score Avg: {avg_score:.1f} | Expectancy: ${exp:.2f}")
        print("="*45)

if __name__ == "__main__":
    # Univers de test réduit pour l'exemple
    universe = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD", "GOOGL", "META", "NFLX", "JPM", "GS"]
    
    backtester = TitanBacktester(universe)
    backtester.run()
