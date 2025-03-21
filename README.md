Ce projet est une démonstration de l'utilisation de l'algorithme de Deep Q-Learning pour jouer au jeu Pong avec l'API Gym de OpenAI.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé Python 3.11 ou une version ultérieure.

### Dépendances

Les dépendances sont nécessaires pour exécuter le projet. 
Vous pouvez les installer en utilisant le fichier `requirements.txt`.

Les principales bibliothèques utilisées dans ce projet sont :

gymnasium : Une bibliothèque pour l'interface d'environnements de simulation (utilisée pour les jeux comme Pong).
torch : PyTorch, utilisé pour l'apprentissage profond.
stable-baselines3 : Une bibliothèque pour les algorithmes d'apprentissage par renforcement, incluant DQN.
ale-py : Interface pour les jeux Atari avec l'API ALE (Atari Learning Environment).


### INSTALLATION

Clonez le dépôt :

```bash
git clone https://github.com/ton-compte/DQL_Project_Test_w_pong.git
cd DQL_Project_Test_w_pong
```

Créez un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate   # Sur Windows, utilisez .venv\Scripts\activate
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

### Utilisation
Pour démarrer l'entraînement du modèle avec Deep Q-Learning et jouer au jeu Pong, exécutez le fichier main.py

```bash
python main.py
Le modèle commencera à apprendre à jouer au Pong en interagissant avec l'environnement Gym.

