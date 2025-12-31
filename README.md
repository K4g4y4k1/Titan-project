# 🛡️ Titan v4.9.6 "The Final Vanguard"

Ce dépôt contient le code source et les outils de déploiement pour le moteur de trading adaptatif Titan.

## 🚀 Installation Rapide (AWS)

### Clonage & Setup :

- git clone <votre_repo_prive> ~/titan-project
- cd ~/titan-project
- bash setup_aws.sh


### Configuration du Service :
Éditez titan-core.service avec vos clés API réelles, puis :

- sudo cp titan-core.service /etc/systemd/system/
- sudo systemctl daemon-reload
- sudo systemctl enable titan-core
- sudo systemctl start titan-core


### Surveillance :

- Logs : journalctl -u titan-core -f

- Métriques : curl http://localhost:8080 (Ou via Dashboard IP)

## 🛡️ Disjoncteurs Actifs

- Daily DD (2%) : Veto journalier automatique.

- Total DD (10%) : Fermeture de toutes les positions et verrouillage matériel.

- Capital Forge : Triage auto (Active / Degraded / Quarantine) basé sur l'espérance réelle.

- Auto-Promotion : L'Exploration est promue si elle bat l'Exploitation.

- Note : Le fichier .daemon_heartbeat permet de vérifier que la boucle de trading est vivante.
