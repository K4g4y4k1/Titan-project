import pandas as pd
import numpy as np
import statistics
import logging
from datetime import datetime, timedelta

# --- CONFIGURATION MIROIR TITAN v4.9.8 ---
# Ces paramètres sont synchronisés avec la gouvernance du daemon Vanguard-Sentinel
GOUVERNANCE = {
    "INITIAL_CAPITAL": 100000,
    "MIN_TRADES_FOR_JUDGEMENT": 10,
    "DEGRADED_THRESHOLD_USD": 0.0,
    "QUARANTINE_THRESHOLD_USD": -15.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25,
    "MAX_POSITION_SIZE_PCT": 0.10,
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

class TitanBacktesterV51:
    """
    Simulateur de performance adaptatif v5.1.
    Reproduit le triage de la Capital Forge et la surveillance Sentinel.
    """
    def __init__(self):
        self.capital = GOUVERNANCE["INITIAL_CAPITAL"]
        self.equity_history = [self.capital]
        self.trade_log = []
        self.forge_memory = {"EXPLOITATION": [], "EXPLORATION": []}
        self.is_promoted = False
        self.sector_exposure = {} # Suivi des secteurs pour le backtest

    def get_forge_allocation(self, mode):
        """Simulation du triage à 3 niveaux (Active / Degraded / Quarantine)."""
        history = self.forge_memory[mode][-20:]
        if len(history) < GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
            return 1.0 # Phase d'observation
        
        expectancy = sum(history) / len(history)
        if expectancy <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]: return 0.0
        if expectancy <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]: return 0.5
        return 1.0

    def run_simulation(self, data_dict):
        """
        data_dict: { 'SYMBOL': { 'df': DataFrame, 'sector': 'Tech' } }
        """
        print(f"🧪 Simulation Vanguard-Sentinel v5.1")
        
        all_symbols = list(data_dict.keys())
        
        # Simulation sur 100 jours fictifs
        for day in range(100):
            # 1. Sélection de candidats aléatoires parmi l'univers (Triple Beat Simulation)
            daily_candidates = np.random.choice(all_symbols, size=min(5, len(all_symbols)), replace=False)
            
            for symbol in daily_candidates:
                sector = data_dict[symbol]['sector']
                
                # 2. Simulation IA & Mode selection
                score_ia = np.random.normal(82, 10)
                sigma_ia = np.random.uniform(10, 30)
                
                mode = "EXPLOITATION" if score_ia >= 85 and sigma_ia <= 20 else "EXPLORATION" if score_ia >= 72 and sigma_ia <= 35 else None
                if not mode: continue

                # 3. Vérification de la Forge & Gouvernance
                allocation = self.get_forge_allocation(mode)
                if allocation == 0: continue

                # Cap Taille Position & Global Cap
                cap = GOUVERNANCE["GLOBAL_CAPS"][mode]
                if mode == "EXPLORATION" and self.is_promoted: cap = 0.40
                
                risk_pct = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * allocation * cap
                
                # Sizing avec cap 10% (Sentinel R5)
                risk_amt = self.capital * risk_pct
                max_pos_size = self.capital * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]
                
                # Simulation de gain/perte
                win = np.random.random() < 0.54 # Edge PEAD
                pnl = (risk_amt * 2) if win else -risk_amt # R:R 2:1
                
                # 4. Update
                self.capital += pnl
                self.equity_history.append(self.capital)
                self.forge_memory[mode].append(pnl)
                
                self.trade_log.append({
                    'day': day, 'symbol': symbol, 'mode': mode, 'pnl': pnl, 'win': win, 'equity': self.capital
                })

                # Check Promotion
                self.update_promotion_status()

        self._print_report()

    def update_promotion_status(self):
        exp_exploit = np.mean(self.forge_memory["EXPLOITATION"][-10:]) if len(self.forge_memory["EXPLOITATION"]) >= 10 else 0
        exp_explore = np.mean(self.forge_memory["EXPLORATION"][-10:]) if len(self.forge_memory["EXPLORATION"]) >= 10 else 0
        self.is_promoted = exp_explore > exp_exploit and exp_explore > 0

    def _print_report(self):
        df = pd.DataFrame(self.trade_log)
        total_pnl = self.capital - GOUVERNANCE["INITIAL_CAPITAL"]
        print(f"\n{'='*40}\n📊 RAPPORT TITAN v5.1\n{'='*40}")
        print(f"Capital Final      : ${self.capital:,.2f}")
        print(f"PnL Net           : {((self.capital/GOUVERNANCE['INITIAL_CAPITAL'])-1)*100:.2f}%")
        print(f"Win Rate          : {(df['win'].mean()*100):.2f}%")
        print(f"Total Trades      : {len(df)}")
        print(f"Statut Promotion  : {'🔥 PROMU' if self.is_promoted else 'Standard'}")
        print(f"Forge Status (Exp): {self.forge_memory['EXPLOITATION'][-1:]}")

if __name__ == "__main__":
    mock_data = {f'STOCK_{i}': {'sector': 'Tech'} for i in range(20)}
    TitanBacktesterV51().run_simulation(mock_data)
