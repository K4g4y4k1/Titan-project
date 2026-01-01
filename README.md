# 🛡️ Titan v5.6 "Apex-Guardian"

Titan v5.6 est un moteur de trading algorithmique haute performance spécialisé dans la stratégie PEAD (Post-Earnings Announcement Drift). Cette version "Apex-Guardian" fusionne l'agressivité de la v4.9.8 avec la rigueur institutionnelle de la v5.3.

## 🚀 Architecture Hybride "Full-Free"

Pour garantir une indépendance totale vis-à-vis des abonnements payants (FMP), Titan v5.6 utilise un écosystème de données distribué :

- Signal (Earnings) : Alpha Vantage (Via calendrier CSV optimisé).

- Gouvernance (Secteurs) : yfinance (Avec système de cache SQLite local).

- Exécution (Prix & Ordres) : Alpaca Markets (Temps réel via API Broker).

- Intelligence : OpenRouter (Consensus IA via Gemini 2.0 Flash).

## 🛡️ Disjoncteurs & Gouvernance "Guardian"

Le système est protégé par une triple ceinture de sécurité :

- Kill-Switch de Drawdown :

- Journalier (2%) : Liquidation totale et arrêt si l'équité chute de 2% sur la journée.

- Total (10%) : Verrouillage matériel (fichier .halt_trading) si le capital baisse de 10% par rapport à l'ancre initiale.

- Time-Exit (J+3) : Fermeture automatique des positions stagnantes après 3 jours de détention pour libérer le capital.

- Capital Forge : Système de triage adaptatif qui place les stratégies en Quarantaine ou en mode Dégradé selon leur espérance mathématique réelle.

- Veto Sectoriel : Limitation stricte à 25% d'exposition maximum par secteur d'activité.

## 🛠️ Installation Rapide (AWS EC2)

Préparation du serveur :

- git clone <votre_repo> ~/titan-project
- cd ~/titan-project
- bash setup_aws.sh


Configuration des Secrets : Éditez le fichier de service /etc/systemd/system/titan-core.service avec vos clés :

- ALPACA_API_KEY / ALPACA_SECRET_KEY

- ALPHA_VANTAGE_KEY

- OPENROUTER_API_KEY

Lancement :

- sudo systemctl daemon-reload
- sudo systemctl enable --now titan-core


## 📟 Monitoring & Audit

Le système expose ses métriques en temps réel sur le port 8080.

- Dashboard Live : Accédez à http://<IP_AWS>:8080 (Assurez-vous que le port est ouvert dans votre Security Group AWS).

- Audit des logs : journalctl -u titan-core -f

- Preuve de vie : ls -l .daemon_heartbeat (Le fichier doit être mis à jour toutes les 60 secondes).

## 📊 Structure des Fichiers

- trading_daemon.py : Moteur principal asynchrone.

- backtester.py : Simulateur de portefeuille synchronisé avec la logique v5.6.

- titan_prod_v5.db : Base de données SQLite (Trades, Forge, Cache Sectoriel).

- .halt_trading : Fichier de sécurité (créez-le pour arrêter le bot manuellement).

Note de conformité : Ce logiciel est un outil d'assistance au trading. Le trading comporte des risques importants. Testez toujours en mode PAPER pendant au moins 15 jours avant toute utilisation en capital réel.
