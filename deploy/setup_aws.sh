#!/bin/bash

# --- SCRIPT D'INSTALLATION AUTOMATISÉE TITAN-CORE v4.1 ---
# Cible : Ubuntu 22.04 LTS sur AWS EC2

echo "🚀 Démarrage de l'installation du système Titan-Core..."

# 1. Mise à jour du système
sudo apt update && sudo apt upgrade -y

# 2. Installation des dépendances système (Python, Pip, Venv, SQLite)
sudo apt install -y python3-pip python3-venv git sqlite3 curl

# 3. Création de l'environnement virtuel Python
echo "📦 Configuration de l'environnement Python..."
cd /home/ubuntu/titan-project
python3 -m venv venv
source venv/bin/activate

# 4. Installation des dépendances Python
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt non trouvé, installation des paquets par défaut..."
    pip install alpaca-trade-api pandas numpy aiohttp
fi

# 5. Configuration des permissions pour SQLite et les Logs
echo "🔐 Configuration des permissions..."
sudo chown -R ubuntu:ubuntu /home/ubuntu/titan-project
# Création du dossier de logs système si nécessaire
sudo mkdir -p /var/log/titan-core
sudo chown ubuntu:ubuntu /var/log/titan-core

# 6. Finalisation
echo "✅ Installation terminée avec succès."
echo "👉 Prochaines étapes :"
echo "   1. Configurez vos clés API dans le fichier titan-core.service"
echo "   2. Déplacez le service : sudo cp deploy/titan-core.service /etc/systemd/system/"
echo "   3. Activez le bot : sudo systemctl enable --now titan-core"