"""
TITAN CONTROL SERVER v1.0
API endpoint pour le Control Layer

Architecture:
    [Titan 5.7.1 DB] --> [Control Server] --> [Control Dashboard]
                             (READ ONLY)      (RECOMMENDATIONS)

Ce serveur:
    - LIT les données de Titan (read-only)
    - ANALYSE via le Control Layer
    - EXPOSE les recommandations
    - NE MODIFIE JAMAIS Titan directement
"""

import asyncio
import sqlite3
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import hmac

# Import du Control Layer
from titan_control_layer import TitanControlLayer, KillDecision

# Configuration
DB_PATH = "titan_prod_v5.db"
CONTROL_TOKEN = os.getenv("TITAN_DASHBOARD_TOKEN")  # Même token que Titan
CONTROL_PORT = 8081  # Port séparé du dashboard Titan


class ControlAPIHandler(BaseHTTPRequestHandler):
    """
    Handler HTTP pour le Control Layer
    
    Endpoints:
        GET  /control/analyze    - Analyse toutes les positions
        POST /control/execute    - Log une action manuelle
        GET  /control/audit      - Récupère l'historique des recommandations
        GET  /control/doctrine   - Affiche la doctrine actuelle
    """
    
    def log_message(self, format, *args):
        """Désactive les logs HTTP standards"""
        return
    
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
    
    def _authenticate(self):
        """Vérifie le token d'authentification"""
        auth = self.headers.get('Authorization', "")
        return hmac.compare_digest(auth, f"Bearer {CONTROL_TOKEN}")
    
    def do_OPTIONS(self):
        """CORS preflight"""
        self._set_headers(204)
    
    def do_GET(self):
        """Handler GET pour les endpoints de lecture"""
        if not self._authenticate():
            self._set_headers(401)
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return
        
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == "/control/analyze":
            self._handle_analyze()
        elif path == "/control/audit":
            self._handle_audit()
        elif path == "/control/doctrine":
            self._handle_doctrine()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        """Handler POST pour les actions manuelles"""
        if not self._authenticate():
            self._set_headers(401)
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return
        
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == "/control/execute":
            self._handle_execute()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def _handle_analyze(self):
        """
        Endpoint: GET /control/analyze
        
        Analyse toutes les positions actives et retourne les recommandations
        """
        try:
            # 1. Récupérer l'état actuel de Titan (lecture seule)
            with sqlite3.connect(DB_PATH) as conn:
                # Récupérer equity et drawdown depuis les métriques système
                # (À adapter selon votre structure DB)
                equity = 100000.0  # TODO: récupérer depuis Alpaca
                drawdown_current = 0.01  # TODO: calculer
                
                # Récupérer les positions ouvertes
                positions = conn.execute("""
                    SELECT symbol, qty, entry_price
                    FROM trades
                    WHERE status = 'OPEN'
                """).fetchall()
            
            # 2. Simuler les données de position (à remplacer par vraies données Alpaca)
            positions_data = []
            for symbol, qty, entry_price in positions:
                # TODO: récupérer current_price depuis Alpaca
                current_price = entry_price * 1.02  # Simulation
                unrealized_pnl = (current_price - entry_price) * qty
                
                positions_data.append({
                    'symbol': symbol,
                    'qty': qty,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl
                })
            
            # 3. Initialiser le Control Layer
            control = TitanControlLayer(
                db_path=DB_PATH,
                equity=equity,
                drawdown_current=drawdown_current
            )
            
            # 4. Analyser les positions
            analysis = control.analyze_all_positions(positions_data)
            
            # 5. Logger les recommandations
            if analysis['next_victim']:
                control.log_recommendation(analysis['next_victim'])
            
            # 6. Formater la réponse
            response = {
                "timestamp": datetime.now().isoformat(),
                "next_victim": self._serialize_decision(analysis['next_victim']) if analysis['next_victim'] else None,
                "kill_list": [self._serialize_decision(d) for d in analysis['kill_list']],
                "healthy_positions": analysis['healthy_positions'],
                "summary": analysis['summary']
            }
            
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_execute(self):
        """
        Endpoint: POST /control/execute
        
        Log une action manuelle (human override)
        NE TOUCHE PAS Titan directement
        """
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            action_data = json.loads(post_data.decode('utf-8'))
            
            symbol = action_data.get('symbol')
            action = action_data.get('action')  # KILL ou HOLD
            source = action_data.get('source', 'UNKNOWN')
            
            if not symbol or not action:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Missing required fields"}).encode())
                return
            
            # Logger l'action dans une table séparée
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS control_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        source TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    INSERT INTO control_actions (symbol, action, source, metadata)
                    VALUES (?, ?, ?, ?)
                """, (symbol, action, source, json.dumps(action_data)))
            
            response = {
                "status": "logged",
                "symbol": symbol,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "note": "Action logged. Execute manually in Titan if needed."
            }
            
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_audit(self):
        """
        Endpoint: GET /control/audit
        
        Récupère l'historique des recommandations
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Récupérer les 50 dernières recommandations
                rows = conn.execute("""
                    SELECT 
                        timestamp,
                        symbol,
                        recommendation,
                        urgency,
                        reason,
                        confidence,
                        was_executed
                    FROM control_recommendations
                    ORDER BY timestamp DESC
                    LIMIT 50
                """).fetchall()
                
                recommendations = []
                for row in rows:
                    recommendations.append({
                        "timestamp": row[0],
                        "symbol": row[1],
                        "recommendation": row[2],
                        "urgency": row[3],
                        "reason": row[4],
                        "confidence": row[5],
                        "was_executed": bool(row[6])
                    })
            
            response = {
                "total": len(recommendations),
                "recommendations": recommendations
            }
            
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_doctrine(self):
        """
        Endpoint: GET /control/doctrine
        
        Retourne la doctrine de kill actuelle (pour documentation)
        """
        doctrine = {
            "version": "1.0",
            "description": "Kill decision doctrine for Titan Control Layer",
            "urgency_levels": {
                "10": {
                    "name": "IMMEDIATE",
                    "description": "No discussion, immediate exit required",
                    "triggers": [
                        "AI thesis invalidation",
                        "Circuit breaker threat (blast radius > 50%)"
                    ]
                },
                "8": {
                    "name": "CRITICAL",
                    "description": "Imminent danger, action required soon",
                    "triggers": [
                        "Max hold time reached (5 days)",
                        "Stop loss imminent"
                    ]
                },
                "5": {
                    "name": "RECOMMENDED",
                    "description": "Strong signal to exit",
                    "triggers": [
                        "Time decay with no progress (2 days + <1% gain)",
                        "Thesis weakening + losing position"
                    ]
                },
                "3": {
                    "name": "SUGGESTED",
                    "description": "Weak signal, monitor closely",
                    "triggers": [
                        "Low confidence (<50%)",
                        "High blast radius (>30%)"
                    ]
                },
                "0": {
                    "name": "HOLD",
                    "description": "Position healthy, no action needed",
                    "triggers": []
                }
            },
            "notes": [
                "Control Layer operates in READ-ONLY mode on Titan 5.7.1",
                "All recommendations are logged separately",
                "Titan execution remains autonomous",
                "Human override is always possible"
            ]
        }
        
        self._set_headers(200)
        self.wfile.write(json.dumps(doctrine, indent=2).encode())
    
    def _serialize_decision(self, decision: KillDecision) -> dict:
        """Convertit un KillDecision en dict JSON"""
        if not decision:
            return None
        
        return {
            "symbol": decision.symbol,
            "should_kill": decision.should_kill,
            "reason": decision.reason,
            "urgency": int(decision.urgency),
            "confidence": decision.confidence,
            "metadata": decision.metadata
        }


def run_control_server(port=CONTROL_PORT):
    """
    Lance le serveur Control Layer
    
    Ce serveur tourne en parallèle de Titan mais n'interfère jamais avec lui
    """
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, ControlAPIHandler)
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  TITAN CONTROL LAYER v1.0                                ║
║  Running on port {port}                                    ║
║                                                          ║
║  Endpoints:                                              ║
║    GET  /control/analyze  - Position analysis            ║
║    POST /control/execute  - Log manual action            ║
║    GET  /control/audit    - Recommendations history      ║
║    GET  /control/doctrine - Current doctrine             ║
║                                                          ║
║  Architecture: READ-ONLY on Titan 5.7.1                  ║
║  Mode: Command & Control (C2)                            ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down Control Layer...")
        httpd.shutdown()


# === INTÉGRATION AVEC LE DASHBOARD TITAN EXISTANT ===

class ExtendedTitanHandler(BaseHTTPRequestHandler):
    """
    Extension du handler Titan existant
    Ajoute UNIQUEMENT l'endpoint /control/analyze
    SANS modifier le comportement de Titan
    """
    
    def log_message(self, format, *args):
        return
    
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
    
    def _authenticate(self):
        auth = self.headers.get('Authorization', "")
        return hmac.compare_digest(auth, f"Bearer {CONTROL_TOKEN}")
    
    def do_OPTIONS(self):
        self._set_headers(204)
    
    def do_GET(self):
        if not self._authenticate():
            self._set_headers(401)
            return
        
        parsed_path = urlparse(self.path)
        
        # Route spécifique au Control Layer
        if parsed_path.path == "/control/analyze":
            self._handle_control_analyze()
        else:
            # TODO: Déléguer au handler Titan existant
            self._set_headers(200)
            self.wfile.write(json.dumps({"metrics": "TITAN_METRICS_HERE"}).encode())
    
    def _handle_control_analyze(self):
        """Même logique que le serveur séparé"""
        # Copy du code _handle_analyze ci-dessus
        pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "standalone":
        # Mode standalone: serveur séparé sur port 8081
        run_control_server(CONTROL_PORT)
    else:
        print("""
Usage:
    python titan_control_server.py standalone
        → Run as separate server on port 8081
        
    Or integrate into existing Titan dashboard server
        → Use ExtendedTitanHandler class
        """)
