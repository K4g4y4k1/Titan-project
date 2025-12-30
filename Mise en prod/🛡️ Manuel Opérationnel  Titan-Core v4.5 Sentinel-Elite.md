# **🛡️ Manuel Opérationnel : Titan-Core v4.5 "Sentinel-Elite"**

Ce guide est votre référence ultime pour installer, configurer et exploiter le moteur de trading **Titan v4.5**. Ce système est conçu pour être résilient, auditable et autonome.

## **1\. Architecture du Système**

Titan v4.5 n'est pas un simple script, c'est une infrastructure composée de trois couches :

1. **Le Daemon (Moteur)** : Script Python tournant 24h/24 sur AWS.  
2. **La Sentinelle (Sécurité)** : Multiples disjoncteurs (IA, Drawdown, Cooldown).  
3. **L'Interface (Monitoring)** : Serveur HTTP intégré pour suivre les performances en temps réel.

## **2\. Prérequis (Comptes et Clés)**

Avant de commencer, assurez-vous d'avoir ouvert les comptes suivants :

* **Alpaca Markets** : Courtier pour l'exécution (commencez en mode *Paper Trading*).  
* **Financial Modeling Prep (FMP)** : Source des données fondamentales et prix.  
* **OpenRouter** : Accès aux cerveaux (Claude 3.5, GPT-4o, Gemini 1.5).  
* **AWS (Amazon Web Services)** : Pour héberger le bot (instance EC2 Ubuntu).

## **3\. Étape 1 : Configuration du Serveur AWS**

### **A. Création de l'instance**

1. Lancez une instance **EC2** sous **Ubuntu 22.04 LTS**.  
2. Type d'instance conseillé : t3.small (2 vCPU, 2 Go RAM).

### **B. Sécurité Réseau (Security Groups) \- CRITIQUE**

Dans la console AWS, ouvrez les ports suivants :

* **Port 22 (SSH)** : Source \= My IP (Pour vous connecter au serveur).  
* **Port 8080 (Metrics)** : Source \= My IP (Pour voir votre dashboard depuis votre navigateur).

## **4\. Étape 2 : Installation Technique (Terminal)**

Connectez-vous à votre serveur via SSH et exécutez le script d'installation :

\# Téléchargement du projet (ou création manuelle)  
mkdir \~/Titan-project && cd \~/Titan-project

\# Installation des dépendances via le script automatique  
bash setup\_aws.sh

\# Activation de l'environnement virtuel  
source venv/bin/activate  
pip install \-r requirements.txt

## **5\. Étape 3 : Configuration des Secrets**

Le bot ne contient aucune clé API par défaut. Vous devez les injecter dans le fichier de service Linux.

1. Ouvrez le fichier de service : sudo nano /etc/systemd/system/titan-core.service  
2. Remplissez les variables Environment :  
   * ALPACA\_API\_KEY, ALPACA\_SECRET\_KEY  
   * FMP\_API\_KEY, OPENROUTER\_API\_KEY  
   * ENV\_MODE=PAPER (Laissez en PAPER pour débuter)  
   * LIVE\_AFFIRMATION=False (Verrou de sécurité supplémentaire)

## **6\. Étape 4 : Lancement et Monitoring**

### **Démarrage du Bot**

sudo systemctl daemon-reload  
sudo systemctl enable titan-core  
sudo systemctl start titan-core

### **Surveillance des performances**

Ouvrez votre navigateur et allez sur : http://VOTRE\_IP\_AWS:8080  
Vous verrez un JSON structuré avec :

* equity : Votre capital actuel.  
* win\_rate : Pourcentage de trades gagnants.  
* ai\_dispersion : Si ce chiffre est élevé, les IA ne sont pas d'accord.  
* cooldown\_until : Si le bot a perdu 3 fois de suite, il affichera l'heure de reprise ici.

## **7\. Les Disjoncteurs "Elite" (Comprendre la Sécurité)**

### **Le "Halt" Manuel**

Si vous voulez arrêter le trading immédiatement sans couper le serveur :

touch \~/Titan-project/.halt\_trading

Le bot détectera ce fichier au prochain cycle et se mettra en pause de sécurité. Pour reprendre, supprimez le fichier : rm \~/Titan-project/.halt\_trading.

### **Le Disjoncteur IA (Dispersion)**

Si l'écart entre les notes de Claude, GPT et Gemini est trop grand (ex: l'un dit 90 et l'autre 40), le bot **annule le trade**. Cela vous protège contre les "hallucinations" d'une IA isolée.

### **Le Risk Scaling**

Si votre capital descend de plus de 5% par rapport au capital de départ, le bot **divise automatiquement par deux** la taille de ses prochains trades pour préserver vos fonds.

## **8\. Protocole de Passage en "LIVE"**

**N'activez jamais le mode LIVE avant d'avoir respecté ces points :**

1. **30 jours** de test en mode PAPER sans erreur technique.  
2. **Profit Factor \> 1.2** sur les trades virtuels.  
3. **Vérification de l'IP** : Assurez-vous que votre adresse IP n'a pas changé si vous avez restreint le port 8080\.

Pour passer en LIVE :

1. Éditez le service : ENV\_MODE=LIVE et LIVE\_AFFIRMATION=True.  
2. Redémarrez : sudo systemctl restart titan-core.

**Note de conformité :** Ce système enregistre chaque décision dans titan\_prod\_v4\_5.db. En cas de doute sur un trade, vous pouvez auditer la table ai\_votes pour lire le raisonnement exact de chaque modèle au moment de l'achat.