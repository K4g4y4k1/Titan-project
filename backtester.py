import pandas as pd
import numpy as np
import statistics
import logging
from datetime import datetime, timedelta

# --- CONFIGURATION MIROIR TITAN v4.5 ---
GOUVERNANCE = {
    "BASE_RISK_PER_TRADE_PCT": 0.01,
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "MIN_SL_PCT": 0.015,
    "MAX_SL_PCT": 0.05,
    "MAX_POSITION_SIZE_PCT": 0.10,
    "MAX_CONSECUTIVE_LOSSES": 3,
    "COOLDOWN_HOURS": 4,
    "AI_DISPERSION_THRESHOLD": 25,
    "MAX_TRADES_PER_DAY": 5
}

class TitanBacktesterElite:
    """
    Simulateur Haute Fidélité v4.5.
    Intègre : SL/TP Adaptatif, Cooldown après pertes, et Risk Scaling.
    """
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity_history = [initial_capital]
        self.trades = []
        
        # États de session (Miroir du Daemon)
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.trades_today = 0
        self.current_date = None

    def _get_adaptive_params(self, ai_score):
        """Calcule le SL/TP adaptatif comme le moteur v4.5."""
        conviction_factor = (ai_score - 80) / 20 
        tp_raw = GOUVERNANCE["BASE_TP_PCT"] + (conviction_factor * 0.04)
        sl_raw = GOUVERNANCE["BASE_SL_PCT"] - (conviction_factor * 0.01)
        
        # Clamping de sécurité (Audit v4.4)
        tp_final = min(max(tp_raw, 0.04), 0.12)
        sl_final = min(max(sl_raw, GOUVERNANCE["MIN_SL_PCT"]), GOUVERNANCE["MAX_SL_PCT"])
        return tp_final, sl_final

    def run(self, data_dict):
        """
        Exécute la simulation sur un dictionnaire de DataFrames {symbol: df}.
        Le df doit contenir 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        print(f"🧪 Simulation Titan v4.5 'Sentinel-Elite' sur {len(data_dict)} actifs...")
        
        # On simule un flux temporel unifié (simplifié ici par symbole)
        for symbol, df in data_dict.items():
            self.consecutive_losses = 0 # Reset par symbole pour le test
            
            for i in range(20, len(df) - 5):
                row = df.iloc[i]
                self.current_date = df.index[i]
                
                # 1. Vérification Cooldown & Limites journalières
                if self.cooldown_until and self.current_date < self.cooldown_until:
                    continue
                
                # 2. Simulation Signal PEAD (Triple Beat Proxy)
                # On simule un signal si volume > 1.5x moyenne et hausse > 2%
                vol_avg = df['Volume'].rolling(20).mean().iloc[i]
                if row['Volume'] > vol_avg * 1.5 and row['Close'] > df['Close'].iloc[i-1] * 1.02:
                    
                    # 3. Simulation du Colisée IA (Consensus & Dispersion)
                    # On génère des scores aléatoires centrés sur le signal
                    ai_scores = [np.random.normal(85, 10) for _ in range(3)]
                    avg_score = sum(ai_scores) / 3
                    dispersion = statistics.stdev(ai_scores)
                    
                    # Filtre Disjoncteur IA (v4.3)
                    if dispersion > GOUVERNANCE["AI_DISPERSION_THRESHOLD"] or avg_score < 80:
                        continue

                    # 4. Paramètres Adaptatifs (v4.4)
                    tp_pct, sl_pct = self._get_adaptive_params(avg_score)
                    
                    # 5. Risk Scaling (v4.5)
                    total_dd = (self.capital - self.initial_capital) / self.initial_capital
                    risk_scaling = 0.5 if total_dd < -0.05 else 1.0
                    
                    # 6. Exécution du Trade (Le lendemain à l'Open)
                    entry_price = df.iloc[i+1]['Open']
                    risk_amt = self.capital * GOUVERNANCE["BASE_RISK_PER_TRADE_PCT"] * risk_scaling
                    stop_price = entry_price * (1 - sl_pct)
                    take_profit = entry_price * (1 + tp_pct)
                    
                    qty = int(risk_amt / (entry_price * sl_pct))
                    
                    # Sortie du trade (Simulation Bracket)
                    pnl = 0
                    exit_type = "TIME"
                    for j in range(i+1, len(df)):
                        h, l, c = df.iloc[j]['High'], df.iloc[j]['Low'], df.iloc[j]['Close']
                        if l <= stop_loss:
                            pnl = (stop_loss - entry_price) * qty
                            exit_type = "SL"
                            break
                        if h >= take_profit:
                            pnl = (take_profit - entry_price) * qty
                            exit_type = "TP"
                            break
                        if (j - i) > 10: # Max 10 jours de drift
                            pnl = (c - entry_price) * qty
                            exit_type = "EXPIRED"
                            break
                    
                    # 7. Mise à jour de la résilience
                    self.capital += pnl
                    self.equity_history.append(self.capital)
                    
                    if pnl < 0:
                        self.consecutive_losses += 1
                        if self.consecutive_losses >= GOUVERNANCE["MAX_CONSECUTIVE_LOSSES"]:
                            self.cooldown_until = self.current_date + timedelta(hours=GOUVERNANCE["COOLDOWN_HOURS"])
                    else:
                        self.consecutive_losses = 0
                        
                    self.trades.append({
                        'date': self.current_date,
                        'symbol': symbol,
                        'pnl': pnl,
                        'type': exit_type,
                        'ai_score': avg_score,
                        'dispersion': dispersion,
                        'sl_used': sl_pct,
                        'tp_used': tp_pct,
                        'scaling': risk_scaling
                    })

        self._report()

    def _report(self):
        if not self.trades:
            print("❌ Aucun trade exécuté avec les filtres v4.5.")
            return

        df_res = pd.DataFrame(self.trades)
        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res)
        total_pnl = self.capital - self.initial_capital
        
        print("\n" + "="*40)
        print("📊 RAPPORT AUDIT BACKTEST v4.5")
        print("="*40)
        print(f"Capital Final    : {self.capital:,.2f} $")
        print(f"PnL Total        : {total_pnl:,.2f} $ ({(total_pnl/self.initial_capital):.2%})")
        print(f"Nombre de Trades : {len(df_res)}")
        print(f"Win Rate         : {win_rate:.2%}")
        
        # Calcul Profit Factor
        gains = df_res[df_res['pnl'] > 0]['pnl'].sum()
        pertes = abs(df_res[df_res['pnl'] < 0]['pnl'].sum())
        pf = gains / pertes if pertes > 0 else float('inf')
        print(f"Profit Factor    : {pf:.2f}")
        
        # Calcul Max Drawdown
        equity = np.array(self.equity_history)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        print(f"Max Drawdown     : {drawdowns.min():.2%}")
        print("="*40)

if __name__ == "__main__":
    # Génération de données de test pour vérifier la logique
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    data = {}
    for s in ["TECH_STK", "BIO_STK"]:
        prices = np.random.normal(100, 2, 200).cumsum() + 1000
        data[s] = pd.DataFrame({
            'Open': prices, 'High': prices+5, 'Low': prices-5, 'Close': prices+1, 
            'Volume': np.random.normal(100000, 20000, 200)
        }, index=dates)

    tester = TitanBacktesterElite(initial_capital=100000)
    tester.run(data)
