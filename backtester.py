import pandas as pd
import numpy as np
import statistics
import logging
from datetime import datetime, timedelta

# --- CONFIGURATION MIROIR TITAN v4.9.6 ---
# Ces paramètres doivent correspondre à ton trading_daemon.py
GOUVERNANCE = {
    "INITIAL_CAPITAL": 100000,
    "MIN_TRADES_FOR_JUDGEMENT": 10,
    "DEGRADED_THRESHOLD_USD": 0.0,
    "QUARANTINE_THRESHOLD_USD": -15.0,
    "GLOBAL_CAPS": {
        "EXPLOITATION": 0.80, # 80% du risque total max
        "EXPLORATION": 0.20   # 20% du risque total max
    },
    "MODES": {
        "EXPLOITATION": { "BASE_RISK": 0.01 },
        "EXPLORATION": { "BASE_RISK": 0.0025 }
    },
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03
}

class TitanBacktesterV5:
    """
    Simulateur de performance adaptatif v5.0.
    Reproduit la sélection naturelle de la Capital Forge (v4.9.6).
    """
    def __init__(self):
        self.capital = GOUVERNANCE["INITIAL_CAPITAL"]
        self.equity_history = [self.capital]
        self.trade_log = []
        # Mémoire pour le triage de la Forge
        self.forge_memory = {"EXPLOITATION": [], "EXPLORATION": []}
        self.is_promoted = False

    def get_forge_allocation(self, mode):
        """Reproduit le triage à 3 niveaux de la Forge."""
        history = self.forge_memory[mode][-20:] # Fenêtre glissante de 20
        if len(history) < GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
            return 1.0 # Phase d'observation : pleine allocation
        
        expectancy = sum(history) / len(history)
        
        if expectancy <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]:
            return 0.0 # Quarantaine : arrêt total
        if expectancy <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]:
            return 0.5 # Dégradé : risque divisé par 2
        return 1.0     # Actif : allocation nominale

    def run_simulation(self, data_dict):
        """
        Simule le comportement du daemon sur des données historiques.
        data_dict: { 'SYMBOL': DataFrame }
        """
        print(f"🧪 Simulation Titan v5.0 (Miroir Vanguard 4.9.6)")
        print(f"Initialisation : ${self.capital:,.2f}")

        for symbol, df in data_dict.items():
            # On simule un cycle de scan sur l'historique
            for i in range(20, len(df) - 1):
                # 1. Simulation d'un Signal IA (moyenne score 82, sigma 15)
                # En backtest, on tire au sort pour voir si on est en EXPLO ou EXPLORE
                score_ia = np.random.normal(82, 10)
                sigma_ia = np.random.uniform(10, 30)
                
                mode = None
                if score_ia >= 85 and sigma_ia <= 20:
                    mode = "EXPLOITATION"
                elif score_ia >= 72 and sigma_ia <= 35:
                    mode = "EXPLORATION"
                
                if not mode: continue

                # 2. Vérification de la Forge (Triage)
                allocation = self.get_forge_allocation(mode)
                if allocation == 0: continue # Le système refuse le trade (Quarantaine)

                # 3. Risk Sizing v4.9.6 (Caps & Promotion)
                global_cap = GOUVERNANCE["GLOBAL_CAPS"][mode]
                if mode == "EXPLORATION" and self.is_promoted:
                    global_cap = 0.40 # Doublement du cap si promu
                
                base_risk = GOUVERNANCE["MODES"][mode]["BASE_RISK"]
                final_risk_pct = base_risk * allocation * global_cap
                risk_amt = self.capital * final_risk_pct

                # 4. Simulation de l'Exécution (Bracket Order)
                entry_price = df['Open'].iloc[i+1]
                # Sortie simplifiée : Probabilité basée sur l'espérance
                # On simule ici un ratio R:R de 2 (TP 6% / SL 3%)
                win = np.random.random() < 0.52 # Taux de succès moyen simulé
                
                pnl = (risk_amt * (GOUVERNANCE["BASE_TP_PCT"] / GOUVERNANCE["BASE_SL_PCT"])) if win else -risk_amt
                
                # 5. Mise à jour des registres
                self.capital += pnl
                self.equity_history.append(self.capital)
                self.forge_memory[mode].append(pnl)
                
                self.trade_log.append({
                    'timestamp': df.index[i],
                    'symbol': symbol,
                    'mode': mode,
                    'pnl': pnl,
                    'allocation': allocation,
                    'equity': self.capital
                })

                # 6. Mise à jour de la Promotion
                self.update_promotion_status()

        self._generate_report()

    def update_promotion_status(self):
        """Logique de promotion Exploration -> Exploitation."""
        hist_exploit = self.forge_memory["EXPLOITATION"][-10:]
        hist_explore = self.forge_memory["EXPLORATION"][-10:]
        
        if len(hist_explore) >= 10 and len(hist_exploit) >= 10:
            exp_exploit = sum(hist_exploit) / 10
            exp_explore = sum(hist_explore) / 10
            
            if exp_explore > exp_exploit and exp_explore > 0:
                self.is_promoted = True
            else:
                self.is_promoted = False

    def _generate_report(self):
        if not self.trade_log:
            return print("Aucun trade généré.")
        
        df_res = pd.DataFrame(self.trade_log)
        total_trades = len(df_res)
        win_rate = len(df_res[df_res['pnl'] > 0]) / total_trades
        total_pnl = self.capital - GOUVERNANCE["INITIAL_CAPITAL"]
        
        print("\n" + "="*40)
        print(f"📊 RAPPORT D'AUDIT TITAN v5.0")
        print(f"Période de Simulation : {len(df_res)} trades")
        print(f"Capital Final      : ${self.capital:,.2f}")
        print(f"Performance Totale : {(total_pnl/GOUVERNANCE['INITIAL_CAPITAL'])*100:.2f}%")
        print(f"Win Rate Global    : {win_rate:.2%}")
        print(f"Statut Promotion   : {'ACTIF' if self.is_promoted else 'INACTIF'}")
        
        # Stats par mode
        for mode in ["EXPLOITATION", "EXPLORATION"]:
            m_df = df_res[df_res['mode'] == mode]
            if not m_df.empty:
                m_pnl = m_df['pnl'].sum()
                print(f"--- Mode {mode} ---")
                print(f"  PnL : ${m_pnl:,.2f} ({len(m_df)} trades)")
        print("="*40)

if __name__ == "__main__":
    # Génération de données fictives pour test
    dates = pd.date_range(datetime.now() - timedelta(days=100), periods=100)
    mock_data = {
        'AAPL': pd.DataFrame({
            'Open': np.random.normal(150, 2, 100),
            'Volume': np.random.normal(1000000, 100000, 100)
        }, index=dates)
    }
    
    tester = TitanBacktesterV5()
    tester.run_simulation(mock_data)
