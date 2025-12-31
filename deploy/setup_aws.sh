#!/bin/bash
# Script de déploiement "Bunker" pour Titan v4.9.8
# Cible : Ubuntu 22.04+ | Python 3.12

echo "🚀 Initialisation de l'infrastructure Vanguard-Sentinel..."

# 1. Mise à jour du système et dépôts
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

# 5. Installation des dépendances (Fix Pandas-TA inclus)
pip install --upgrade pip setuptools wheel
pip install pandas==2.2.0 numpy==1.26.0 aiohttp==3.9.1 alpaca-trade-api python-dotenv

# Installation de Pandas-TA via branche dev pour compatibilité 3.12
echo "⚙️ Installation de Pandas-TA (Branche Development)..."
pip install pandas-ta

# 6. Finalisation
echo "✅ Installation de l'environnement terminée avec succès."
echo "👉 Prochaines étapes :"
echo "1. Transférez votre trading_daemon.py (v4.9.8) dans ~/titan-project"
echo "2. Configurez le fichier /etc/systemd/system/titan-core.service"
echo "3. Activez avec : sudo systemctl enable --now titan-core"
