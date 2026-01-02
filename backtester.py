import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# --- CONFIGURATION MIROIR TITAN v5.6.5 "APEX-ULTIMATE" ---
GOUVERNANCE = {
    "INITIAL_CAPITAL": 100000,
    "MIN_TRADES_FOR_JUDGEMENT": 10,
    "DEGRADED_THRESHOLD_USD": 0.0,
    "QUARANTINE_THRESHOLD_USD": -15.0,
    "MAX_SECTOR_EXPOSURE_PCT": 0.25,
    "MAX_POSITION_SIZE_PCT": 0.10,
    "MAX_HOLDING_DAYS": 3,
    "MAX_TRADES_PER_DAY": 12,
    "GLOBAL_CAPS": {
        "EXPLOITATION": 0.80,
        "EXPLORATION": 0.20
    },
    "MODES": {
        "EXPLOITATION": { "MIN_SCORE": 85, "MAX_SIGMA": 20, "BASE_RISK": 0.01 },
        "EXPLORATION": { "MIN_SCORE": 72, "MAX_SIGMA": 35, "BASE_RISK": 0.0025 }
    },
    "BASE_TP_PCT": 0.06,
    "BASE_SL_PCT": 0.03,
    "SLIPPAGE_PROTECTION": 0.002,
    "MAX_TOTAL_DRAWDOWN_PCT": 0.10
}

class TitanBacktesterV565:
    """
    Simulateur de Portefeuille Adaptatif v5.6.5.
    Reproduit fidèlement le comportement de la Capital Forge et du Sentinel Protocol.
    """
    def __init__(self):
        self.capital = GOUVERNANCE["INITIAL_CAPITAL"]
        self.initial_equity = GOUVERNANCE["INITIAL_CAPITAL"]
        self.equity_history = [self.capital]
        self.trade_log = []
        self.open_positions = []
        self.forge_memory = {"EXPLOITATION": [], "EXPLORATION": []}
        self.is_halted = False
        self.current_day = 0

    def get_forge_allocation(self, mode):
        """Calcul de l'allocation dynamique selon l'espérance mathématique (Forge)."""
        history = self.forge_memory[mode][-20:]
        if len(history) < GOUVERNANCE["MIN_TRADES_FOR_JUDGEMENT"]:
            return 1.0 # Phase d'apprentissage
        
        expectancy = sum(history) / len(history)
        if expectancy <= GOUVERNANCE["QUARANTINE_THRESHOLD_USD"]: return 0.0
        if expectancy <= GOUVERNANCE["DEGRADED_THRESHOLD_USD"]: return 0.5
        return 1.0

    def run_simulation(self, days=180):
        """Exécute la simulation sur une période donnée (par défaut 6 mois)."""
        print(f"🔬 Simulation Titan v5.6.5 Apex-Ultimate")
        print(f"Capital Initial: ${self.initial_equity:,} | Mode: {GOUVERNANCE['MAX_HOLDING_DAYS']}d Time-Exit")
        
        for day in range(days):
            self.current_day = day
            if self.is_halted: break

            # 1. Mise à jour des positions (Sorties TP/SL/Time)
            self._update_positions()

            # 2. Vérification du Kill-Switch Total
            total_dd = (self.capital - self.initial_equity) / self.initial_equity
            if total_dd <= -GOUVERNANCE["MAX_TOTAL_DRAWDOWN_PCT"]:
                print(f"🚨 JOUR {day}: KILL-SWITCH TOTAL DÉCLENCHÉ ({total_dd:.2%})")
                self.is_halted = True
                break

            # 3. Scan et Entrées (Simulation de l'Alpha Vantage Scan)
            self._process_daily_scan()
            
            self.equity_history.append(self.capital)

        self._generate_final_report()

    def _update_positions(self):
        """Gère le cycle de vie des positions ouvertes."""
        still_open = []
        for pos in self.open_positions:
            # Simulation probabiliste du mouvement PEAD
            # En Exploitation (Score élevé), on a une probabilité de réussite supérieure
            win_prob = 0.58 if pos['mode'] == "EXPLOITATION" else 0.45
            
            # Détermination de l'issue du trade
            outcome = np.random.choice(['TP', 'SL', 'TIME'], p=[win_prob*0.4, (1-win_prob)*0.4, 0.6])
            
            age = self.current_day - pos['entry_day']
            
            if outcome == 'TP' and age > 0:
                pnl = pos['qty'] * pos['entry_price'] * GOUVERNANCE["BASE_TP_PCT"]
                self._close(pos, pnl, "TAKE_PROFIT")
            elif outcome == 'SL' and age > 0:
                pnl = -pos['qty'] * pos['entry_price'] * GOUVERNANCE["BASE_SL_PCT"]
                self._close(pos, pnl, "STOP_LOSS")
            elif age >= GOUVERNANCE["MAX_HOLDING_DAYS"]:
                # Time-Exit J+3 : On simule un PnL de dérive neutre/faible
                pnl_pct = np.random.uniform(-0.01, 0.02)
                pnl = pos['qty'] * pos['entry_price'] * pnl_pct
                self._close(pos, pnl, "TIME_EXIT")
            else:
                still_open.append(pos)
        
        self.open_positions = still_open

    def _process_daily_scan(self):
        """Simule le scan des opportunités PEAD du jour."""
        # Nombre d'opportunités par jour (variable)
        num_candidates = np.random.randint(0, 5)
        trades_count = 0

        for _ in range(num_candidates):
            if trades_count >= GOUVERNANCE["MAX_TRADES_PER_DAY"]: break
            
            # Simulation du scoring IA
            score = np.random.randint(65, 98)
            sigma = np.random.randint(5, 45)
            
            # Détermination du mode selon les seuils v5.6.5
            mode = None
            if score >= GOUVERNANCE["MODES"]["EXPLOITATION"]["MIN_SCORE"] and sigma <= GOUVERNANCE["MODES"]["EXPLOITATION"]["MAX_SIGMA"]:
                mode = "EXPLOITATION"
            elif score >= GOUVERNANCE["MODES"]["EXPLORATION"]["MIN_SCORE"] and sigma <= GOUVERNANCE["MODES"]["EXPLORATION"]["MAX_SIGMA"]:
                mode = "EXPLORATION"
            
            if not mode: continue

            # Calcul Allocation via la Forge
            alloc_factor = self.get_forge_allocation(mode)
            if alloc_factor == 0: continue # Quarantaine active

            # Sizing Professionnel (Risk-at-Risk)
            risk_pct = GOUVERNANCE["MODES"][mode]["BASE_RISK"] * alloc_factor * GOUVERNANCE["GLOBAL_CAPS"][mode]
            
            # On applique le slippage dès l'entrée
            price = 100.0 # Normalisé
            entry_price = price * (1 + GOUVERNANCE["SLIPPAGE_PROTECTION"])
            
            qty = (self.capital * risk_pct) / (entry_price * GOUVERNANCE["BASE_SL_PCT"])
            
            # Cap de sécurité par ligne (10%)
            max_qty = (self.capital * GOUVERNANCE["MAX_POSITION_SIZE_PCT"]) / entry_price
            qty = min(qty, max_qty)

            if qty > 0:
                self.open_positions.append({
                    'symbol': f"TICKER_{np.random.randint(100,999)}",
                    'entry_day': self.current_day,
                    'entry_price': entry_price,
                    'qty': qty,
                    'mode': mode,
                    'risk_pct': risk_pct
                })
                trades_count += 1

    def _close(self, pos, pnl, reason):
        """Enregistre la clôture d'une position."""
        self.capital += pnl
        self.forge_memory[pos['mode']].append(pnl)
        self.trade_log.append({
            'day': self.current_day,
            'symbol': pos['symbol'],
            'mode': pos['mode'],
            'pnl': pnl,
            'reason': reason,
            'equity': self.capital
        })

    def _generate_final_report(self):
        df = pd.DataFrame(self.trade_log)
        if df.empty:
            print("❌ Aucun trade n'a été exécuté.")
            return

        print(f"\n{'='*50}")
        print(f"📊 RAPPORT FINAL TITAN v5.6.5")
        print(f"{'='*50}")
        print(f"Capital Final       : ${self.capital:,.2f}")
        print(f"Performance Totale : {((self.capital/self.initial_equity)-1)*100:.2f}%")
        print(f"Nombre de Trades    : {len(df)}")
        print(f"Win Rate            : {(len(df[df['pnl'] > 0]) / len(df) * 100):.2f}%")
        
        # Répartition par raison de sortie
        reasons = df['reason'].value_counts()
        print(f"\n--- Répartition des Sorties ---")
        for r, count in reasons.items():
            print(f"  {r:12}: {count} ({count/len(df)*100:.1f}%)")

        # Performance par Mode (Impact de la Forge)
        print(f"\n--- Performance par Mode (Forge Logic) ---")
        for m in ["EXPLOITATION", "EXPLORATION"]:
            m_df = df[df['mode'] == m]
            if not m_df.empty:
                exp = m_df['pnl'].mean()
                print(f"  {m:12}: {len(m_df)} trades | PnL: ${m_df['pnl'].sum():,.2f} | Avg: ${exp:.2f}")

        # Drawdown Max
        rolling_max = pd.Series(self.equity_history).cummax()
        drawdowns = (pd.Series(self.equity_history) - rolling_max) / rolling_max
        print(f"\nMax Drawdown        : {drawdowns.min():.2%}")
        print(f"{'='*50}")

if __name__ == "__main__":
    backtester = TitanBacktesterV565()
    backtester.run_simulation(180) # Simulation de 6 mois
