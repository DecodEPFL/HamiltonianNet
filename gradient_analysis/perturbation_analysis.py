"""
Perturbation analysis for counterexample - Appendix III [1].
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python perturbation_analysis.py   --h        [STEP SIZE]       \
                                  --T        [FINAL TIME]      \
                                  --gamma    [GAMMA]           \
                                  --epsilon  [EPSILON]         \
                                  --gpu      [GPU ID]
Flags:
  --h: step size.
  --T: final time T corresponding to the end on the simulation.
  --gamma: perturbation on y_0.
  --epsilon: epsilon coefficient.
  --gradient_info: whether to calculate the gradient norms for each layer at each iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=float, default=0.05)
    parser.add_argument('--T', type=float, default=200)
    parser.add_argument('--gamma', type=float, default=0.005)
    parser.add_argument('--epsilon', type=float, default=1.0)
    args = parser.parse_args()

    # Parameters
    h = args.h
    T = args.T
    gamma = args.gamma
    epsilon = args.epsilon

    # Create J
    nf = 2  # Number of features
    J = np.array([[0, -1], [1, 0]])

    # Randomly initialize y0 and normalize it
    rng = np.random.RandomState(2020)
    y0 = rng.rand(2) - 0.5
    y0 = y0/np.linalg.norm(y0)

    # Define time, y(t), y_per(t) and diff_y(t) vectors, and phi(t) matrix
    N = int(T/h)+1  # Number of datapoints
    t = np.linspace(0, T, N)
    y_t = np.zeros((N, nf))
    y_per_t = np.zeros((N, nf))
    diff_y = np.zeros((N, nf))
    phi_0t = np.zeros((N, nf, nf))

    y_Tt = np.zeros((N, nf))
    y_per_Tt = np.zeros((N, nf))
    diff_y_Tt = np.zeros((N, nf))
    phi_Tt = np.zeros((N, nf, nf))

    y_t_2 = np.zeros((N, nf))
    y_per_t_2 = np.zeros((N, nf))
    diff_y_2 = np.zeros((N, nf))
    phi_TTt_2 = np.zeros((N, nf, nf))

    # # Forward
    print("Starting forward prop...")
    # Calculation of each row of phi_0t [denominator layout]
    for i in range(nf):

        # Initialization
        beta = np.eye(nf)[:, i]
        y_t[0, :] = y0
        y_per_t[0, :] = y0 + gamma * beta
        diff_y[0, :] = (y_per_t[0, :] - y_t[0, :]) / gamma

        # Simulation using symplectic Euler
        for j in range(N-1):
            y_aux = y_t[j, :]
            y_per_aux = y_per_t[j, :]
            for k in range(nf):
                y_next = y_aux + h * epsilon * J.dot(np.tanh(y_aux))
                y_aux[k] = y_next[k]
                y_t[j+1, k] = y_next[k]
                y_next = y_per_aux + h * epsilon * J.dot(np.tanh(y_per_aux))
                y_per_aux[k] = y_next[k]
                y_per_t[j+1, k] = y_next[k]
            diff_y[j+1, :] = (y_per_t[j+1, :] - y_t[j+1, :]) / gamma
        phi_0t[:, i, :] = diff_y

    # # Backward
    print("Starting backward prop...")
    # Calculation of each row of phi_TTt [denominator layout]

    for jj in range(N):
        for i in range(nf):
            # Initialization
            beta = np.eye(nf)[:, i]
            y_t_2[jj, :] = y_t[jj, :]
            y_per_t_2[jj, :] = y_t[jj, :] + gamma * beta

            # Simulation using symplectic Euler
            for j in range(jj, N-1):
                y_aux = y_t_2[j, :]
                y_per_aux = y_per_t_2[j, :]
                for k in range(nf):
                    y_next = y_aux + h * epsilon * J.dot(np.tanh(y_aux))
                    y_aux[k] = y_next[k]
                    y_t_2[j+1, k] = y_next[k]
                    y_next = y_per_aux + h * epsilon * J.dot(np.tanh(y_per_aux))
                    y_per_aux[k] = y_next[k]
                    y_per_t_2[j+1, k] = y_next[k]
            diff_y_2[N-1, :] = (y_per_t_2[N-1, :] - y_t_2[N-1, :]) / gamma
            phi_TTt_2[N-1-jj, i, :] = diff_y_2[N-1, :]

    # Plots
    # fig = plt.figure()
    # ax1, ax2 = fig.subplots(2, 1)
    # ax1.plot(t, y_t[:, 0])
    # ax1.plot(t, y_t[:, 1])
    # ax1.set_title("$y(t)$ trajectory")
    # ax1.set_xlabel("Time")
    # ax1.set_ylabel("Entries of $y(t)$")

    # ax2.plot(t, y_per_t[:, 0])
    # ax2.plot(t, y_per_t[:, 1])
    # ax2.set_title("$y_\gamma(t)$ trajectory")
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("Entries of $y_\gamma(t)$")
    # fig.tight_layout()
    # fig.savefig("y_trajectories.eps", format='eps')

    plt.figure()
    plt.plot(t, phi_0t[:, 0, 0])
    plt.plot(t, phi_0t[:, 0, 1])
    plt.plot(t, phi_0t[:, 1, 0])
    plt.plot(t, phi_0t[:, 1, 1])
    plt.title("$\Phi(t,0)$ trajectory")
    plt.xlabel("Time")
    plt.ylabel("Entries of $\Phi(t)$")
    plt.savefig("phi_trajectory.eps", format='eps')

    fig = plt.figure(figsize=(8, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(y_t[:, 0], y_t[:, 1])
    ax1.scatter(y_t[0, 0], y_t[0, 1], color='red')
    ax2.plot(y_per_t[:, 0], y_per_t[:, 1])
    ax2.scatter(y_per_t[0, 0], y_per_t[0, 1], color='red')
    ax1.set_title("$y(t)$ trajectory")
    ax2.set_title("$y_\gamma(t)$ trajectory")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    fig.tight_layout()
    fig.savefig("y_phase_diagram.eps", format='eps')

    # fig = plt.figure(figsize=(8, 4))
    # ax1, ax2 = fig.subplots(1, 2)
    # ax1.plot(phi_0t[:, 0, 0], phi_0t[:, 0, 1])
    # ax1.scatter(phi_0t[0, 0, 0], phi_0t[0, 0, 1], color='red')
    # ax2.plot(phi_0t[:, 1, 0], phi_0t[:, 1, 1])
    # ax2.scatter(phi_0t[0, 1, 0], phi_0t[0, 1, 1], color='red')
    # ax1.set_title("$\Phi(t)$ trajectory - row 1")
    # ax2.set_title("$\Phi(t)$ trajectory - row 2")
    # ax1.set_xlabel("Feature 1")
    # ax1.set_ylabel("Feature 2")
    # ax2.set_xlabel("Feature 1")
    # ax2.set_ylabel("Feature 2")
    # fig.tight_layout()
    # fig.savefig("phi_phase_diagram.eps", format='eps')

    # Figure 5 - Appendix III:
    plt.figure()
    plt.plot(t, phi_TTt_2[:, 0, 0])
    plt.plot(t, phi_TTt_2[:, 0, 1])
    plt.plot(t, phi_TTt_2[:, 1, 0])
    plt.plot(t, phi_TTt_2[:, 1, 1])
    # plt.title("$\Phi(T,T-t)$ trajectory")
    plt.xlabel("Time $t$")
    # plt.ylabel("Entries of $\Phi(T,T-t)$")
    plt.ylabel("Entries of " + r'$\frac{\partial y(T)}{\partial y(T-t)}$')
    plt.legend(["(1,1)", "(1,2)", "(2,1)", "(2,2)"])
    plt.savefig("phi_inv_trajectory.eps", format='eps')

    # fig = plt.figure(figsize=(8, 4))
    # ax1, ax2 = fig.subplots(1, 2)
    # ax1.plot(phi_TTt_2[:, 0, 0], phi_TTt_2[:, 0, 1])
    # ax1.scatter(phi_TTt_2[0, 0, 0], phi_TTt_2[0, 0, 1], color='red')
    # ax2.plot(phi_TTt_2[:, 1, 0], phi_TTt_2[:, 1, 1])
    # ax2.scatter(phi_TTt_2[0, 1, 0], phi_TTt_2[0, 1, 1], color='red')
    # ax1.set_title("$\Phi(T,T-t)$ trajectory - row 1")
    # ax2.set_title("$\Phi(T,T-t)$ trajectory - row 2")
    # ax1.set_xlabel("Feature 1")
    # ax1.set_ylabel("Feature 2")
    # ax2.set_xlabel("Feature 1")
    # ax2.set_ylabel("Feature 2")
    # fig.tight_layout()
    # fig.savefig("phi_inv_phase_diagram.eps", format='eps')
