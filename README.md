🛡️ Alpha-PEAD Titan v3.7 (Industrial Edition)

Alpha-PEAD Titan est un système de trading quantitatif autonome conçu pour exploiter la dérive post-annonce de résultats (Post-Earnings Announcement Drift). Le système combine l'analyse de données fondamentales en temps réel, le filtrage par Intelligence Artificielle et une gestion du risque de niveau institutionnel.

🎯 Vision & Stratégie

Le bot repose sur la capture d'alpha lors des annonces de résultats d'entreprises US :

Triple Beat Detection : Identification des entreprises dépassant les attentes sur l'EPS, le Revenu et la Guidance.

Filtrage Quantitatif : Analyse de la volatilité (ATR) et du volume relatif (RVOL).

IA Sentinel : Validation qualitative via LLM (GPT-4o/Claude 3.5) pour analyser le contexte sectoriel et éviter la sur-corrélation.

Guardian Mode : Gestion automatisée des sorties (Stop-Loss et Take-Profit dynamiques).

🛠️ Stack Technique

Langage : Python 3.11+

Framework UI : Streamlit (Dashboard temps réel)

Infrastructure : AWS EC2 (ou Hugging Face Spaces pour le monitoring)

Base de données : SQLite (Persistance des trades et logs d'IA)

APIs Tierces :

Alpaca Markets : Exécution des ordres (Paper/Live)

Financial Modeling Prep (FMP) : Données financières fondamentales

OpenRouter : Accès multi-modèles IA

Telegram : Alertes push et monitoring distant

🏗️ Architecture du Système

graph TD
    A[Scanner FMP] -->|Triple Beat| B(IA Sentinel)
    B -->|Score > 80| C{Risk Manager}
    C -->|Calcul Position 1%| D[Alpaca Broker]
    D -->|Ordre Bracket| E{Marché}
    E -->|Notification| F[Telegram Bot]


🚀 Installation & Déploiement

1. Cloner le projet

git clone [https://github.com/votre-username/alpha-pead-titan.git](https://github.com/votre-username/alpha-pead-titan.git)
cd alpha-pead-titan


2. Configuration des Secrets

Créez un fichier .env à la racine ou utilisez les secrets de votre plateforme :

ALPACA_API_KEY=votre_cle
ALPACA_SECRET_KEY=votre_secret
FMP_API_KEY=votre_cle
OPENROUTER_API_KEY=votre_cle
TELEGRAM_BOT_TOKEN=ton_token
TELEGRAM_CHAT_ID=ton_id


3. Installation des dépendances

pip install -r requirements.txt


4. Lancement

# Pour le dashboard
streamlit run app.py


🛡️ Gestion du Risque (Industrial Features)

Fixed Risk Per Trade : Risque limité à 1% du capital total par position.

Daily Kill-Switch : Arrêt automatique si le drawdown journalier dépasse 2%.

Bracket Orders : Chaque achat est accompagné simultanément d'un Stop-Loss et d'un Take-Profit envoyés au serveur du broker.

Sector Capping : Limitation de l'exposition maximale par secteur d'activité (ex: max 25% Tech).

📝 Licence

Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.

Avertissement : Le trading comporte des risques. Ce logiciel est fourni à des fins éducatives. L'auteur n'est pas responsable des pertes financières liées à l'utilisation de ce bot.
