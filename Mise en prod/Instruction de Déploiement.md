# 🚀 Déploiement Industriel : Titan-Core v4.1 (AWS/Linux)

Ce guide remplace les versions antérieures. Il est conçu pour transformer votre instance AWS EC2 en une station de trading institutionnelle résiliente et sécurisée.

## 1. Préparation de l'Infrastructure (AWS EC2)

1. Instance recommandée : Ubuntu 22.04 LTS (Type t3.small minimum pour la gestion des threads async).
2. Security Groups : * Port 22 (SSH) : Restreint à votre IP.Port 8080 (Metrics) : Pour le monitoring de santé Titan.
3. Stockage : SSD (gp3) recommandé pour les écritures rapides de SQLite WAL.

## 2. Installation Automatisée

Utilisez le script setup_aws.sh pour préparer l'environnement :# Sur votre instance AWS
chmod +x deploy/setup_aws.sh
./deploy/setup_aws.sh

## 3. Configuration du Service Systemd (Guardian)

Le bot ne doit plus être lancé manuellement. Il doit être géré par le noyau Linux pour garantir un uptime de 99.9%.

1. Copiez le fichier de service :sudo cp deploy/titan-core.service /etc/systemd/system/
2. Éditez les clés API directement dans le fichier (ou utilisez un fichier .env) :sudo nano /etc/systemd/system/titan-core.service
3. Activez le service :sudo systemctl daemon-reload
sudo systemctl enable titan-core
sudo systemctl start titan-core

## 4. Commandes de Pilotage (Ops)

Action              Commande
Vérifier le statut      sudo systemctl status titan-core
Voir les logs (Live)    journalctl -u titan-core -f
Arrêt d'urgence         touch .halt_trading (Le bot s'arrêtera au prochain cycle)
Redémarrage             sudo systemctl restart titan-core

## 5. Monitoring de Santé (Health Check)

Titan v4.1 expose un serveur HTTP interne sur le port 8080. Vous pouvez vérifier l'état du moteur sans accéder aux logs :
curl http://localhost:8080/health

Réponse attendue : {"status": "ok", "equity": 12540.50, "active_positions": 3}

## 🛡️ Protocole de Sécurité Post-Déploiement

- Rotation des clés : Si vous avez déjà fait un git push sans .gitignore, changez vos clés Alpaca immédiatement.
- Audit SQL : Une fois par semaine, téléchargez titan_prod_v4_1.db pour une analyse approfondie dans votre backtester.py (Digital Twin).