import pandas as pd
import numpy as np
import logging

class Backtester:
    """
    Simulateur de précision aligné sur Titan-Core v4.1 (Audit R9).
    Intègre les Bracket Orders (TP/SL) et les contraintes de Gouvernance.
    """
    
    def __init__(self, initial_capital=10000):
        # Configuration identique à la Gouvernance v4.1
        self.capital = initial_capital
        self.balance = initial_capital
        self.risk_per_trade_pct = 0.01  # 1%
        self.tp_pct = 0.05              # 5%
        self.sl_pct = 0.03              # 3%
        self.max_pos_size_pct = 0.15    # 15% (Cap institutionnel)
        self.kill_switch_total = -0.10  # -10%
        
        self.trades = []
        self.equity_history = [] # Format compatible v4.1

    def calculate_metrics(self):
        """Analyse de performance post-simulation."""
        if not self.trades:
            return "Aucun trade effectué. Vérifiez les critères de signal."
            
        df = pd.DataFrame(self.trades)
        
        # Calcul des retours
        df['pnl_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
        
        # Max Drawdown (Calcul précis sur courbe d'équité)
        equity_series = pd.Series([h['equity'] for h in self.equity_history])
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()

        # Profit Factor
        wins = df[df['pnl_realized'] > 0]['pnl_realized'].sum()
        losses = abs(df[df['pnl_realized'] <= 0]['pnl_realized'].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')

        return {
            "Capital Final": f"{self.balance:.2f}$",
            "Total Trades": len(df),
            "Win Rate": f"{(df['pnl_realized'] > 0).mean():.2%}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Sharpe Ratio": f"{(df['pnl_pct'].mean() / df['pnl_pct'].std() * np.sqrt(252)):.2f}" if df['pnl_pct'].std() != 0 else "0.00",
            "Max Drawdown": f"{max_dd:.2%}",
            "Total PnL": f"{df['pnl_realized'].sum():.2f}$"
        }

    def run_simulation(self, historical_data):
        """Simulation avec logique Bracket Order & Kill Switch."""
        print("🔍 Simulation Titan v4.1 en cours...")
        
        for symbol, df in historical_data.items():
            if df.empty: continue
            
            # Indicateurs techniques pour signaux proxy
            df['vol_ma'] = df['Volume'].rolling(20).mean()
            
            for i in range(20, len(df) - 10):
                # Vérification Kill Switch Total (R1)
                total_drawdown = (self.balance - self.capital) / self.capital
                if total_drawdown <= self.kill_switch_total:
                    print(f"🚨 KILL SWITCH DÉCLENCHÉ à {df.index[i]}")
                    break

                row = df.iloc[i]
                
                # SIGNAL PROXY PEAD (Volume Spike + Price Action)
                if row['Volume'] > (row['vol_ma'] * 1.5) and row['Close'] > df.iloc[i-1]['Close'] * 1.02:
                    
                    entry_price = row['Close']
                    
                    # 1. CALCUL DU RISK SIZING (R3)
                    risk_amount = self.balance * self.risk_per_trade_pct
                    stop_loss = entry_price * (1 - self.sl_pct)
                    take_profit = entry_price * (1 + self.tp_pct)
                    
                    risk_per_share = entry_price - stop_loss
                    qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                    
                    # 2. CAP DE POSITION (Gouvernance R7/R8)
                    max_val = self.balance * self.max_pos_size_pct
                    max_qty = int(max_val / entry_price)
                    qty = min(qty, max_qty)
                    
                    if qty <= 0: continue

                    # 3. SIMULATION DU BRACKET ORDER (Sortie réelle)
                    exit_price = None
                    exit_date = None
                    
                    # On regarde les jours suivants pour voir quel bracket est touché en premier
                    for j in range(i + 1, len(df)):
                        day_high = df.iloc[j]['High']
                        day_low = df.iloc[j]['Low']
                        
                        if day_low <= stop_loss:
                            exit_price = stop_loss
                            exit_date = df.index[j]
                            break
                        elif day_high >= take_profit:
                            exit_price = take_profit
                            exit_date = df.index[j]
                            break
                        # Time exit après 10 jours si aucun bracket touché
                        elif (j - i) >= 10:
                            exit_price = df.iloc[j]['Close']
                            exit_date = df.index[j]
                            break
                    
                    if exit_price:
                        pnl = (exit_price - entry_price) * qty
                        self.balance += pnl
                        
                        self.trades.append({
                            "symbol": symbol,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "qty": qty,
                            "pnl_realized": pnl,
                            "date": exit_date
                        })
                        
                        self.equity_history.append({
                            "timestamp": exit_date,
                            "equity": self.balance,
                            "drawdown": (self.balance - self.capital) / self.capital
                        })

        return self.calculate_metrics()

if __name__ == "__main__":
    # Génération de données synthétiques pour test de robustesse
    dates = pd.date_range(start="2024-01-01", periods=200)
    mock_data = {
        "TECH_TKR": pd.DataFrame({
            "Close": np.linspace(100, 110, 200) + np.random.normal(0, 1.5, 200),
            "High": np.linspace(102, 112, 200) + 1,
            "Low": np.linspace(98, 108, 200) - 1,
            "Volume": np.random.randint(100, 1000, 200)
        }, index=dates)
    }
    
    # Injection de signaux
    mock_data["TECH_TKR"].iloc[50, 3] = 5000 # Spike volume
    mock_data["TECH_TKR"].iloc[50, 0] = 105  # Spike prix
    
    bt = Backtester(initial_capital=10000)
    results = bt.run_simulation(mock_data)
    
    print("\n--- RÉSULTATS BACKTEST v4.1 (Digital Twin) ---")
    for k, v in results.items():
        print(f"{k:15}: {v}")
