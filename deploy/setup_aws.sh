#!/bin/bash
# Script d'installation automatique TITAN-CORE v4.5 sur Ubuntu 22.04 LTS

echo "🚀 Déploiement Titan-Core v4.5 Sentinel-Elite..."

# 1. Mise à jour système
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git sqlite3 curl

# 2. Dossier de projet
mkdir -p ~/Titan-project
cd ~/Titan-project

# 3. Environnement virtuel
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Logs
sudo mkdir -p /var/log/titan-core
sudo chown -R $USER:$USER /var/log/titan-core

echo "✅ Prêt pour configuration des clés API."
