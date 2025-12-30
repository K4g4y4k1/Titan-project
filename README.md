# 🛡️ Titan-Core v4.5 "Sentinel-Elite"

Titan-Core est un moteur de trading quantitatif de niveau institutionnel conçu pour l'exploitation de la dérive post-annonce de résultats (PEAD).

Cette version v4.5 Sentinel-Elite introduit des barrières de résilience adaptative pour protéger le capital dans toutes les conditions de marché.

## 🚀 Innovations Majeures

- Multi-LLM Consensus : Consultation simultanée de Claude 3.5, GPT-4o et Gemini 1.5 Pro.

- AI Circuit Breaker : Rejet automatique des signaux si l'écart-type des votes IA dépasse le seuil de tolérance.

- Adaptative SL/TP : Le Stop-Loss et le Take-Profit s'ajustent dynamiquement selon la conviction de l'IA.

- Temporal Cooldown : Suspension automatique du trading pendant 4h après une série de 3 pertes.

- Risk Scaling : Réduction automatique de 50% de l'exposition si le drawdown total atteint 5%.

## 📂 Structure du Projet

- trading_daemon.py : Le moteur de production (Daemon).

- backtester.py : Le simulateur de précision (Digital Twin).

- requirements.txt : Liste des dépendances.

- setup_aws.sh : Script d'installation pour serveur Ubuntu.

- titan-core.service : Configuration pour exécution 24/7 sur AWS.

## 🛠️ Déploiement Rapide

- Clonez ce dépôt sur votre serveur.

- Lancez bash setup_aws.sh.

- Configurez vos clés API dans titan-core.service.

- Activez le service : sudo systemctl enable --now titan-core.

## 🛡️ Gouvernance

Le système applique strictement les règles de gestion du risque :

- Risk per Trade : 1% du capital.

- Max Drawdown : 2% jour / 10% total.

- Garde-fous : Heartbeat constant et fichier de verrouillage .halt_trading.

## Avertissement : Ce logiciel est un outil de recherche financière. Le trading comporte des risques réels de perte de capital. Testez toujours en mode PAPER pendant au moins 30 jours avant d'envisager un passage en LIVE.
