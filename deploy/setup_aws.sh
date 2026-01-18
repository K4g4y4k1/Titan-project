#!/bin/bash
# Script de déploiement Titan v6.6.5 "APEX-ULTIMATE"
# Cible : Ubuntu 22.04+ | Python 3.12

echo "🚀 Initialisation de Titan v6.6.5"

# 1. Mise à jour et dépôts
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt upgrade -y

# 2. Installation de Python 3.12 et outils système
sudo apt install -y python3.12 python3.12-venv python3.12-dev sqlite3 curl git nodejs npm

# 3. Création de la structure propre
mkdir -p ~/titan-project
cd ~/titan-project

# 4. Création de l'environnement virtuel en 3.12
rm -rf venv # On nettoie l'ancien si présent
python3.12 -m venv venv
source venv/bin/activate

# 5. Installation des librairies
pip install --upgrade pip setuptools wheel
pip install alpaca-trade-api pandas numpy aiohttp yfinance python-dotenv

# Installation de Pandas-TA via branche dev pour compatibilité 3.12
echo "⚙️ Installation de Pandas-TA (Branche Development)..."
pip install pandas-ta

echo "✅ Environnement v6.6.5 prêt."
echo "1. Configurez le fichier /etc/systemd/system/titan-core.service"
echo "2. Activez avec : sudo systemctl enable --now titan-core"














