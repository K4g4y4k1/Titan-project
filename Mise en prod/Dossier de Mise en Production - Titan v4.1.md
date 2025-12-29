# 🏦 Dossier de Mise en Production : Titan-Core v4.1

Statut de l'Audit : 🟢 9.1/10 (Production-Ready)

Architecture : Daemon asynchrone avec persistance SQLite (WAL) et Gouvernance R1-R10.

## 1. Architecture de Sécurité (Kill Switches)

Le système opère sous une triple surveillance hiérarchique :

- Niveau 1 (Physique) : Le fichier .halt_trading. Sa détection provoque l'arrêt immédiat de la boucle run_loop.

- Niveau 2 (Quotidien) : MAX_DAILY_DRAWDOWN (2%). Calculé sur l'équité de la veille. Liquidation totale en cas de franchissement.

- Niveau 3 (Structurel) : MAX_TOTAL_DRAWDOWN (10%). Calculé sur le capital initial au boot. Protection du capital de base.

## 2. Protocole de Réconciliation (R4)

La v4.1 utilise le client_order_id comme clé primaire logique :

- Format : titan_{symbol}_{uuid}.

Le moteur synchronise l'état OPEN en base avec les positions réelles chez Alpaca à chaque cycle (60s).

En cas de clôture (Stop/TP touché), le système récupère le filled_avg_price pour un calcul de PnL exact avant archivage.

## 3. Monitoring & Observabilité (OPS)

L'interface de monitoring est accessible via l'endpoint de santé :

- URL : http://<IP_SERVEUR>:8080

Payload JSON :

- status : État opérationnel (ok, market_closed, critical_error).

- equity : Valeur temps réel du portefeuille.

- win_rate / profit_factor : Indicateurs de performance glissants issus de la DB.

- active_positions : Décompte des lignes en cours.

## 4. Checklist de Déploiement (Go-Live)

Environnement : Python 3.9+ sur Linux (Ubuntu conseillé).

- Base de Données : titan_prod_v4_1.db (Auto-générée au premier run).

- Secrets : Injecter via variables d'environnement (ALPACA_API_KEY, ALPACA_SECRET_KEY, FMP_API_KEY).

- Service : Configurer systemd avec Restart=always et un délai de 30s pour permettre la persistance des verrous de sécurité.

## 5. Roadmap v4.2 (Vers le 10/10)

Pour atteindre l'excellence absolue, les modules suivants sont prévus :

- Module de Corrélation : Interdiction d'ouvrir une position si la corrélation historique (30j) avec le portefeuille existant dépasse 0.7.

- Export Prometheus : Intégration de prometheus_client pour un dashboard Grafana professionnel.

- Calcul du Ratio de Sharpe : Intégration native dans le SYSTEM_STATE pour une mesure du risque ajusté en temps réel.

Verdict de l'Auditeur : Le moteur est sain, les barrières sont étanches. Déploiement autorisé en capital réel sous surveillance active pendant les 15 premiers jours.
