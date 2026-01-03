# 🛡️ TITAN SENTINEL (v5.6.11-LTS)

## Système de Trading Quantitatif PEAD & Architecture de Gouvernance "Grok-Sentinel"
Titan est une infrastructure de trading algorithmique conçue pour exploiter le Post-Earnings Announcement Drift (PEAD) sur les marchés actions US. Le système intègre un pipeline de décision hybride mêlant filtrage déterministe rigoureux et arbitrage cognitif par LLM (Grok-2).

## 🚀 Philosophie : "Capital-First"
- Le système est conçu avec une priorité absolue sur la préservation du capital.
- Zéro Trade en cas de doute : Si les données ou le score IA sont ambigus, le système reste en cash.
- Auto-Quarantaine : Les modes de trading se désactivent automatiquement en cas de performance négative glissante.
- Gouvernance Multi-couches : Limites sectorielles, drawdown journalier et protection contre le slippage intégrées nativement.

## 🏗️ Architecture Technique
- Core : Python asyncio pour une exécution asynchrone non-bloquante.
- Storage : SQLite avec mode WAL pour une persistance rapide et fiable.
- Signal : Scan des résultats via Alpha Vantage & Analyse de sentiment/drift via Grok-2 (xAI).
- Exécution : API Alpaca (Ordres Bracket : Limit + Take Profit + Stop Loss).
- Monitoring : Dashboard temps réel via API sécurisée par HMAC.

## 🛠️ Configuration & Installation
### Pré-requis
Vous aurez besoin des clés API suivantes :
- Alpaca Markets (Trading)
- Alpha Vantage (Données fondamentales)
- OpenRouter (Accès à Grok-2)

### Installation
- git clone [https://github.com/votre-compte/titan-sentinel.git](https://github.com/votre-compte/titan-sentinel.git)
- cd titan-sentinel
- pip install -r requirements.txt

### Variables d'Environnement
Créez un fichier .env ou exportez les variables suivantes :
- export ENV_MODE="PAPER" # ou "LIVE"
- export TITAN_DASHBOARD_TOKEN="votre_token_securise"
- export ALPACA_API_KEY="votre_cle"
- export ALPACA_SECRET_KEY="votre_secret"
- export ALPHA_VANTAGE_KEY="votre_cle"
- export OPENROUTER_API_KEY="votre_cle"

## 📈 Pipeline de Décision
Scanning : 
- Extraction des entreprises publiant leurs résultats le jour J.
- Filtrage : Application des règles de prix ($>5$), de blacklist et d'exposition sectorielle.
- Arbitrage IA : Envoi du contexte à Grok-2 pour évaluation du potentiel de "drift".

### Classification : 
- Exploitation (Score $\ge$ 85, $\sigma \le$ 20)
- Exploration (Score $\ge$ 72, $\sigma \le$ 35)
- Exécution : Placement d'un ordre bracket avec Take Profit (+6%) et Stop Loss (-3%).

## 🛡️ Gouvernance & Risque
Paramètre  /  Limite 
- Max Drawdown Journalier:  2%,
- Max Drawdown Total:        10%,
- Exposition Sectorielle:    25% Max, 
- Taille Position Max:       10% Max,
- Rétention (Holding):       3 Jours Max.

## 📊 Monitoring
Le système expose un endpoint de métriques sécurisé sur le port 8080.
Auth : Bearer Token (HMAC)
Data : Equity, positions ouvertes, ordres en attente, santé de la base de données et performance par mode.

## ⚠️ Avertissement (Disclaimer)
Ce logiciel est fourni à titre éducatif et de recherche. Le trading algorithmique comporte des risques de perte totale du capital. L'utilisateur est seul responsable des configurations et des fonds engagés.

Titan Sentinel - Built for stability, engineered for performance.
