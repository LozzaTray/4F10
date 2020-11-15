"""
In order to run the code as is, you will need scipy, pandas and tqdm installed 
(although tqdm is only needed for the progress bar, and pandas is only for the autocorrelation function)
All of these can be installed (on linux) from the command interface using 'pip'
""" 

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from eprank import eprank
import pandas
from cw2 import sorted_barplot


# set seed for reproducibility
np.random.seed(0)
# load data
data = sio.loadmat('tennis_data.mat')
# Array containing the names of each player
W = data['W']
# loop over array to format more nicely
for i, player in enumerate(W):
    W[i] = player[0]
# Array of size num_games x 2. The first entry in each row is the winner of game i, the second is the loser
G = data['G'] - 1
# Number of players
M = W.shape[0]
# Number of Games
N = G.shape[0]


def gibbs_sample_run():
    # number of iterations
    num_iters = 1100
    # perform gibbs sampling, skill samples is an num_players x num_samples array
    skill_samples = gibbs_sample(G, M, num_iters)#, random_nums)

    sorted_barplot(skill_samples[:,-1], W)

    # Code for plotting the autocorrelation function for player p
    p = 5
    autocor = np.zeros(10)
    for i in range(10):
        autocor[i]=pandas.Series.autocorr(pandas.Series(skill_samples[p,:]),lag=i)
    plt.plot(autocor)
    plt.show()


def epranking():
    num_iters = 3
    # run message passing algorithm, returns mean and precision for each player
    mean_player_skills, precision_player_skills = eprank(G, M, num_iters)


if __name__ == "__main__":
    print("---------- 4F13 - CW2 ----------")
    gibbs_sample_run()
    epranking()