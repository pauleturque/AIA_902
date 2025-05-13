import pickle
import os

#sauvegarde q table
def save_q_table(q_table, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)

# chargement q table
def load_q_table(filename="q_table.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None  # Retourne None si aucune Q-table n'est trouvée
