import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# --- CONFIGURATION MIROIR TITAN v5.6 ---
# Synchronisation stricte avec la gouvernance du daemon Vanguard-Apex
GOUVERNANCE = {
    "INITIAL_CAPITAL": 100000,
    "MIN_TRADES_FOR_JUDGEMENT": 10,
    "DEGRADED_THRESHOLD_USD": 0.0,
    "QUARANTINE_THRESHOLD_USD": -15.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25,
    "MAX_POSITION_SIZE_PCT": 0.10,
    "MAX_HOLDING_DAYS": 3,              # Aligné v5.6
    "GLOBAL_CAPS": {
        "EXPLOITATION": 0.80,
        "EXPLORATION": 0.20
    },
    "MODES": {
        "EXPLOITATION": { "BASE_RISK": 0.01 },
        "EXPLORATION": { "BASE_RISK": 0.0025 }
    },
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "MAX_DAILY_DRAWDOWN_PCT": 0.02,
    "MAX_TOTAL_DRAWDOWN_PCT": 0.10      # Aligné v5.6
}

class TitanBacktesterV56:
    """
    Simulateur de Portefeuille Temporel v5.6.
    Intègre le Time-Exit J+3 et le Kill-Switch de Drawdown Total.
    """
    def __init__(self):
        self.capital = GOUVERNANCE["INITIAL_CAPITAL"]
        self.initial_equity = GOUVERNANCE["INITIAL_CAPITAL"]
        self.equity_history = [self.capital]
        self.trade_log = []
        self.open_positions = [] # Liste de dictionnaires pour le suivi J+3
        self.forge_memory = {"EXPLOITATION": [], "EXPLORATION": []}
        self.is_halted = False

    def get_forge_allocation(self, mode):
        """Triage Capital Forge (3 niveaux)."""
        history = self.forge_memory[mode][-20:]
        if len(history) < GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
            return 1.0
        
        expectancy = sum(history) / len(history)
        if expectancy <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]: return 0.0
        if expectancy <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]: return 0.5
        return 1.0

    def run_simulation(self, iterations=180):
        """
        Simule 180 jours de trading (environ 6 mois).
        """
        print(f"🧪 Lancement Simulation Apex-Guardian v5.6")
        print(f"Mode: Portfolio Management (J+{GOUVERNANCE['MAX_HOLDING_DAYS']} Exit)")
        
        for day in range(iterations):
            if self.is_halted: break

            # 1. Gestion des positions existantes (Time-Exit & Market Moves)
            self._update_open_positions(day)

            # 2. Vérification des Kill-Switches
            if self._check_kill_switches():
                print(f"🚨 JOUR {day}: KILL-SWITCH DÉCLENCHÉ. Arrêt de la simulation.")
                break

            # 3. Scanning & Entrées (Si marché calme)
            self._scan_and_enter(day)
            
            # 4. Enregistrement équité journalière
            self.equity_history.append(self.capital)

        self._print_report()

    def _update_open_positions(self, current_day):
        """Simule le passage du temps et les sorties TP/SL/Time-Exit."""
        active_positions = []
        for pos in self.open_positions:
            # Simulation du mouvement de prix (Probabiliste)
            # 55% de chance de dériver vers le haut pour PEAD
            move = np.random.choice(['TP', 'SL', 'HOLD'], p=[0.25, 0.15, 0.60])
            
            age = current_day - pos['entry_day']
            
            if move == 'TP':
                pnl = pos['risk_amt'] * (GOUVERNANCE["BASE_TP_PCT"] / GOUVERNANCE["BASE_SL_PCT"])
                self._close_trade(pos, pnl, "TAKE_PROFIT", current_day)
            elif move == 'SL':
                pnl = -pos['risk_amt']
                self._close_trade(pos, pnl, "STOP_LOSS", current_day)
            elif age >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                # Sortie temporelle (Time-Exit) - Aligné v5.6
                pnl = pos['risk_amt'] * np.random.uniform(-0.5, 0.8) # PnL aléatoire réduit
                self._close_trade(pos, pnl, "TIME_EXIT", current_day)
            else:
                active_positions.append(pos)
        
        self.open_positions = active_positions

    def _scan_and_enter(self, current_day):
        """Simule la détection Alpha Vantage + yfinance."""
        # On limite à 2 nouveaux trades max par jour pour rester réaliste
        for _ in range(np.random.randint(0, 3)):
            mode = np.random.choice(["EXPLOITATION", "EXPLORATION"], p=[0.7, 0.3])
            
            # Calcul Allocation Forge
            alloc = self.get_forge_allocation(mode)
            if alloc == 0: continue

            # Sizing v5.6
            risk_pct = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * alloc * GOUVERNANCE["GLOBAL_CAPS"][mode]
            risk_amt = self.capital * risk_pct
            
            # Cap de sécurité 10%
            max_risk_allowed = self.capital * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]
            risk_amt = min(risk_amt, max_risk_allowed)

            if risk_amt > 0:
                self.open_positions.append({
                    'entry_day': current_day,
                    'mode': mode,
                    'risk_amt': risk_amt,
                    'symbol': f"TICKER_{np.random.randint(100, 999)}"
                })

    def _close_trade(self, pos, pnl, reason, day):
        self.capital += pnl
        self.forge_memory[pos['mode']].append(pnl)
        self.trade_log.append({
            'day': day, 'symbol': pos['symbol'], 'mode': pos['mode'], 
            'pnl': pnl, 'reason': reason, 'equity': self.capital
        })

    def _check_kill_switches(self):
        # Total Drawdown check (10%)
        total_dd = (self.capital - self.initial_equity) / self.initial_equity
        if total_dd <= -GOUVERNANCE["MAX_TOTAL_DRAWDOWN_PCT"]:
            self.is_halted = True
            return True
        return False

    def _print_report(self):
        df = pd.DataFrame(self.trade_log)
        if df.empty: return print("Aucun trade généré.")
        
        print(f"\n{'='*45}\n📊 RAPPORT VANGUARD-APEX v5.6\n{'='*45}")
        print(f"Capital Final      : ${self.capital:,.2f}")
        print(f"PnL Net           : {((self.capital/self.initial_equity)-1)*100:.2f}%")
        print(f"Total Trades      : {len(df)}")
        print(f"Exit Temporels    : {len(df[df['reason'] == 'TIME_EXIT'])}")
        print(f"Win Rate          : {(len(df[df['pnl'] > 0]) / len(df) * 100):.2f}%")
        
        # Analyse par mode
        for m in ["EXPLOITATION", "EXPLORATION"]:
            m_df = df[df['mode'] == m]
            if not m_df.empty:
                print(f"--- Mode {m} ---")
                print(f"  Trades : {len(m_df)} | PnL : ${m_df['pnl'].sum():,.2f}")
        print(f"{'='*45}")

if __name__ == "__main__":
    TitanBacktesterV56().run_simulation(180)
