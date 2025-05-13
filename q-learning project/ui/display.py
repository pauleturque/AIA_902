import matplotlib.pyplot as plt

#affichage du graph
def plot_rewards(episode_rewards):
    plt.plot(episode_rewards)
    plt.xlabel("Épisode")
    plt.ylabel("Score total")
    plt.title("Progression de l'agent au fil des épisodes")
    plt.show()
