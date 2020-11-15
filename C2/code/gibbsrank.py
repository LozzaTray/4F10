import scipy.linalg
import numpy as np
from tqdm import tqdm


def gibbs_sample(G, M, num_iters):
    """
    Gibbs sample player skills.
    
    Accepts:
    G - Game array G[i, 0] is winner of game i G[i, 1] is loser
    M - number of players
    num_iters - number of Gibbs iterations

    Returns:
    skill_samples - skill_samples[i, j] is sample for skill of player i at iteration j
    """

    # number of games
    N = G.shape[0]
    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((M, 1))
    # Array that will contain skill samples
    skill_samples = np.zeros((M, num_iters))
    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(M)

    # number of iterations of Gibbs
    for i in tqdm(range(num_iters)):
        # sample performance given differences in skills and outcomes
        t = np.zeros((N, 1))
        for g in range(N):

            s = w[G[g, 0]] - w[G[g, 1]]  # difference in skills
            t[g] = s + np.random.randn()  # Sample performance
            while t[g] < 0:  # rejection step
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((M, 1))
        for p in range(M):
            # fill in m[p] prediction (natural param conditional)
            wins_array = np.array(G[:, 0] == p).astype(int)
            loss_array = np.array(G[:, 1] == p).astype(int)
            m[p] = np.dot(t[:,0], (wins_array - loss_array))

        iS = np.zeros((M, M))  # Container for sum of precision matrices (likelihood terms)
        for g in range(N):
            # Build the iS matrix
            winner = G[g, 0]
            loser = G[g, 1]

            iS[winner, winner] += 1
            iS[winner, loser] -= 1
            iS[loser, winner] -= 1
            iS[loser, loser] += 1

        # Posterior precision matrix
        iSS = iS + np.diag(1. / pv)

        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(M, 1), check_finite=False)
        skill_samples[:, i] = w[:, 0]

    return skill_samples


