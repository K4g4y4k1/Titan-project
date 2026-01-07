"""
TITAN CONTROL LAYER v1.0
Système de commandement AU-DESSUS du moteur Titan 5.7.1

Architecture:
    [Titan 5.7.1] --> [DB] --> [Control Layer] --> [Human/Auto Decision]
         |              |           |
      exécution      vérité    commandement

RÈGLE FONDAMENTALE:
    - Le Control Layer LIT uniquement
    - Le Control Layer RECOMMANDE
    - Le Control Layer N'EXÉCUTE PAS directement
    - Titan reste agnostique de cette couche
"""

import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import IntEnum

# --- CONFIGURATION DOCTRINE ---
class KillUrgency(IntEnum):
    """Niveaux d'urgence de sortie"""
    HOLD = 0
    SUGGESTED = 3
    RECOMMENDED = 5
    CRITICAL = 8
    IMMEDIATE = 10

@dataclass
class KillDecision:
    """Décision de sortie structurée"""
    symbol: str
    should_kill: bool
    reason: str
    urgency: KillUrgency
    confidence: float  # 0-100
    metadata: dict

@dataclass
class PositionExtended:
    """Position enrichie avec métadonnées de risque"""
    symbol: str
    qty: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float
    
    # Thesis tracking
    thesis_status: str  # VALID/WEAKENING/INVALID/UNKNOWN
    last_check: Optional[str]
    initial_score: float
    current_confidence: float
    
    # Time tracking
    entry_timestamp: str
    days_held: int
    hours_held: float
    
    # Risk metrics
    blast_radius_score: float
    can_kill_system: bool
    volatility_factor: float
    distance_to_sl_pct: float


class TitanControlLayer:
    """
    Couche de commandement pour Titan 5.7.1
    
    Responsabilités:
        - Analyse des positions (lecture seule)
        - Calcul de risque avancé
        - Recommandations de sortie
        - Audit post-mortem
        
    NON-Responsabilités:
        - Exécution d'ordres (reste dans Titan)
        - Modification de la logique de trading
        - Prise de décision automatique
    """
    
    def __init__(self, db_path: str, equity: float, drawdown_current: float):
        self.db_path = db_path
        self.equity = equity
        self.drawdown_current = drawdown_current
        self.drawdown_limit = 0.02  # 2% circuit breaker
        self.drawdown_remaining = (self.drawdown_limit - drawdown_current) * equity
        
    def _get_position_extended(self, position_data: dict) -> PositionExtended:
        """
        Enrichit une position avec toutes les métadonnées nécessaires
        """
        with sqlite3.connect(self.db_path) as conn:
            # Récupérer les données de trade
            trade = conn.execute("""
                SELECT 
                    entry_price,
                    timestamp,
                    thesis_status,
                    last_thesis_check,
                    consensus as initial_score
                FROM trades 
                WHERE symbol = ? AND status = 'OPEN'
            """, (position_data['symbol'],)).fetchone()
            
            if not trade:
                return None
            
            # Calculs temporels
            entry_time = datetime.fromisoformat(trade[1])
            now = datetime.now()
            days_held = (now - entry_time).days
            hours_held = (now - entry_time).total_seconds() / 3600
            
            # Calculs financiers
            entry_price = trade[0]
            current_price = position_data['current_price']
            unrealized_pnl = position_data['unrealized_pnl']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Calculs de risque
            blast_radius_score = abs(unrealized_pnl) / self.drawdown_remaining if self.drawdown_remaining > 0 else 999
            can_kill = blast_radius_score > 0.5
            
            # Récupérer thesis confidence (si disponible)
            thesis_check = conn.execute("""
                SELECT confidence 
                FROM thesis_checks 
                WHERE client_id = (
                    SELECT client_id FROM trades WHERE symbol = ? AND status = 'OPEN'
                )
                ORDER BY check_timestamp DESC 
                LIMIT 1
            """, (position_data['symbol'],)).fetchone()
            
            current_confidence = thesis_check[0] if thesis_check else 80.0
            
            # Distance au SL (approximation basée sur gouvernance)
            sl_pct = 0.03  # 3% de base
            distance_to_sl = abs(pnl_pct / sl_pct) if pnl_pct < 0 else 999
            
            return PositionExtended(
                symbol=position_data['symbol'],
                qty=position_data['qty'],
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                pnl_pct=pnl_pct,
                thesis_status=trade[2] or 'UNKNOWN',
                last_check=trade[3],
                initial_score=trade[4] or 80.0,
                current_confidence=current_confidence,
                entry_timestamp=trade[1],
                days_held=days_held,
                hours_held=hours_held,
                blast_radius_score=blast_radius_score,
                can_kill_system=can_kill,
                volatility_factor=1.0,  # TODO: calculer via ATR
                distance_to_sl_pct=distance_to_sl
            )
    
    def should_kill_position(self, pos: PositionExtended) -> KillDecision:
        """
        DOCTRINE DE KILL - Version 1.0
        
        Hiérarchie de décision:
            10: IMMEDIATE - Pas de discussion, sortie immédiate
            8:  CRITICAL  - Danger imminent, action requise
            5:  RECOMMENDED - Action fortement suggérée
            3:  SUGGESTED - Signal faible, à surveiller
            0:  HOLD - Position saine
        
        Cette fonction NE DÉCLENCHE RIEN.
        Elle RECOMMANDE uniquement.
        """
        
        # === NIVEAU 10: IMMEDIATE KILL ===
        
        # Thesis invalidée par l'IA
        if pos.thesis_status == "INVALID":
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="AI_THESIS_INVALIDATION",
                urgency=KillUrgency.IMMEDIATE,
                confidence=95.0,
                metadata={
                    "initial_score": pos.initial_score,
                    "current_confidence": pos.current_confidence,
                    "last_check": pos.last_check
                }
            )
        
        # Position peut tuer le système (blast radius > 50%)
        if pos.can_kill_system:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="CIRCUIT_BREAKER_THREAT",
                urgency=KillUrgency.IMMEDIATE,
                confidence=100.0,
                metadata={
                    "blast_radius_score": pos.blast_radius_score,
                    "drawdown_remaining": self.drawdown_remaining,
                    "unrealized_pnl": pos.unrealized_pnl
                }
            )
        
        # === NIVEAU 8: CRITICAL ===
        
        # Temps max atteint
        if pos.days_held >= 5:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="TIME_DECAY_MAX_HOLD",
                urgency=KillUrgency.CRITICAL,
                confidence=90.0,
                metadata={
                    "days_held": pos.days_held,
                    "max_hold": 5,
                    "pnl_pct": pos.pnl_pct
                }
            )
        
        # Proche du SL et en perte
        if pos.distance_to_sl_pct < 0.2 and pos.unrealized_pnl < 0:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="STOP_LOSS_IMMINENT",
                urgency=KillUrgency.CRITICAL,
                confidence=85.0,
                metadata={
                    "distance_to_sl": pos.distance_to_sl_pct,
                    "unrealized_pnl": pos.unrealized_pnl
                }
            )
        
        # === NIVEAU 5: RECOMMENDED ===
        
        # Temps + pas de progrès
        if pos.days_held >= 2 and pos.pnl_pct < 1.0:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="TIME_DECAY_NO_PROGRESS",
                urgency=KillUrgency.RECOMMENDED,
                confidence=75.0,
                metadata={
                    "days_held": pos.days_held,
                    "pnl_pct": pos.pnl_pct,
                    "threshold": 1.0
                }
            )
        
        # Thesis affaiblie + en perte
        if pos.thesis_status == "WEAKENING" and pos.unrealized_pnl < 0:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="THESIS_WEAKENING",
                urgency=KillUrgency.RECOMMENDED,
                confidence=70.0,
                metadata={
                    "thesis_status": pos.thesis_status,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "current_confidence": pos.current_confidence
                }
            )
        
        # === NIVEAU 3: SUGGESTED ===
        
        # Confiance faible
        if pos.current_confidence < 50:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="LOW_CONFIDENCE",
                urgency=KillUrgency.SUGGESTED,
                confidence=60.0,
                metadata={
                    "current_confidence": pos.current_confidence,
                    "initial_score": pos.initial_score
                }
            )
        
        # Blast radius élevé (> 30%)
        if pos.blast_radius_score > 0.3:
            return KillDecision(
                symbol=pos.symbol,
                should_kill=True,
                reason="HIGH_BLAST_RADIUS",
                urgency=KillUrgency.SUGGESTED,
                confidence=65.0,
                metadata={
                    "blast_radius_score": pos.blast_radius_score,
                    "unrealized_pnl": pos.unrealized_pnl
                }
            )
        
        # === NIVEAU 0: HOLD ===
        return KillDecision(
            symbol=pos.symbol,
            should_kill=False,
            reason="POSITION_HEALTHY",
            urgency=KillUrgency.HOLD,
            confidence=80.0,
            metadata={
                "days_held": pos.days_held,
                "pnl_pct": pos.pnl_pct,
                "thesis_status": pos.thesis_status
            }
        )
    
    def analyze_all_positions(self, positions_data: List[dict]) -> dict:
        """
        Analyse toutes les positions et retourne les recommandations
        
        Returns:
            {
                "next_victim": KillDecision (la plus urgente),
                "kill_list": List[KillDecision] (toutes à surveiller),
                "healthy_positions": List[str],
                "summary": dict
            }
        """
        all_decisions = []
        
        for pos_data in positions_data:
            pos_extended = self._get_position_extended(pos_data)
            if pos_extended:
                decision = self.should_kill_position(pos_extended)
                all_decisions.append((pos_extended, decision))
        
        # Trier par urgence décroissante
        all_decisions.sort(key=lambda x: x[1].urgency, reverse=True)
        
        # Séparer kill vs hold
        kill_decisions = [(pos, dec) for pos, dec in all_decisions if dec.should_kill]
        healthy_positions = [pos.symbol for pos, dec in all_decisions if not dec.should_kill]
        
        # Next victim = la plus urgente
        next_victim = kill_decisions[0][1] if kill_decisions else None
        
        return {
            "next_victim": next_victim,
            "kill_list": [dec for _, dec in kill_decisions],
            "healthy_positions": healthy_positions,
            "summary": {
                "total_positions": len(all_decisions),
                "immediate_kills": len([d for _, d in kill_decisions if d.urgency >= 10]),
                "critical_kills": len([d for _, d in kill_decisions if d.urgency >= 8]),
                "recommended_kills": len([d for _, d in kill_decisions if d.urgency >= 5]),
                "suggested_kills": len([d for _, d in kill_decisions if d.urgency >= 3]),
                "healthy_count": len(healthy_positions)
            }
        }
    
    def log_recommendation(self, decision: KillDecision):
        """
        Log les recommandations dans une table séparée
        SANS toucher aux trades de Titan
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS control_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    recommendation TEXT,
                    urgency INTEGER,
                    reason TEXT,
                    confidence REAL,
                    metadata TEXT,
                    was_executed BOOLEAN DEFAULT 0,
                    execution_timestamp DATETIME
                )
            """)
            
            conn.execute("""
                INSERT INTO control_recommendations 
                (symbol, recommendation, urgency, reason, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                decision.symbol,
                'KILL' if decision.should_kill else 'HOLD',
                decision.urgency,
                decision.reason,
                decision.confidence,
                json.dumps(decision.metadata)
            ))


# === EXEMPLE D'UTILISATION ===

def example_usage():
    """
    Exemple d'intégration avec le dashboard
    """
    
    # Données venant de Titan (lecture seule)
    positions_from_alpaca = [
        {
            'symbol': 'FC',
            'qty': 100,
            'current_price': 47.50,
            'unrealized_pnl': -125.00
        },
        {
            'symbol': 'JEF',
            'qty': 50,
            'current_price': 62.30,
            'unrealized_pnl': 85.00
        }
    ]
    
    # Initialiser le Control Layer
    control = TitanControlLayer(
        db_path="titan_prod_v5.db",
        equity=100000.0,
        drawdown_current=0.0085  # 0.85%
    )
    
    # Analyser toutes les positions
    analysis = control.analyze_all_positions(positions_from_alpaca)
    
    # Afficher la recommandation
    if analysis['next_victim']:
        victim = analysis['next_victim']
        print(f"🚨 NEXT VICTIM: {victim.symbol}")
        print(f"   Reason: {victim.reason}")
        print(f"   Urgency: {victim.urgency}/10")
        print(f"   Confidence: {victim.confidence}%")
        
        # Logger la recommandation
        control.log_recommendation(victim)
        
        # Si urgence >= 8, notifier
        if victim.urgency >= KillUrgency.CRITICAL:
            print("⚠️  ACTION REQUIRED - Position critique")
    
    # Résumé
    print(f"\n📊 SUMMARY:")
    print(f"   Total positions: {analysis['summary']['total_positions']}")
    print(f"   Immediate kills: {analysis['summary']['immediate_kills']}")
    print(f"   Critical kills: {analysis['summary']['critical_kills']}")
    print(f"   Healthy: {analysis['summary']['healthy_count']}")


if __name__ == "__main__":
    example_usage()
