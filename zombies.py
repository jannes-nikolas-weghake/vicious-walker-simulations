import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import tqdm as tqdm
from helperfunctions import get_directions



def VWZ(L, occu):
    N = np.floor(occu * L**2)
    grid = np.zeros(L**2, dtype=complex)
    n = 0
    while n < N:
        index = int(np.random.random() * L**2)
        if grid[index] == 0:
            grid[index] = 1
            n += 1

    grid[int(np.random.random() * L**2)] = 1j
    grid = np.reshape(grid, (L, L))

    return grid


def simulate_zombies(occu=0.75, its=20, A=100,alpha=1/33,beta=0.5):
    from tqdm import tqdm
    import matplotlib

    matplotlib.rcParams["text.usetex"] = True
    plt.rc("font", family="serif")
    num_iteration = 400
    temp = []
    for i in tqdm(range(its)):
        walkers = VWZ(int(np.sqrt(A)), occu)
        temp.append(iterations_Z(walkers, num_iteration))
    temp = np.array(temp)
    tempa = np.zeros((2, num_iteration))

    for i in range(num_iteration):
        tempa[0, i] = np.average(temp[:, i, 0])
        tempa[1, i] = np.average(temp[:, i, 1])

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.title(r"\textbf{The tripping Zombie apocalypse}", fontsize=16)
    plt.xlabel(r"time steps", fontsize=16)
    plt.ylabel(r"$\rho$", fontsize=16)

    plt.plot(range(num_iteration), tempa[0] / tempa[1, 0], color="#0C8C00")
    plt.plot(range(num_iteration), tempa[1] / tempa[1, 0], color="#FC797C")
    plt.plot(range(num_iteration), (tempa[1, 0] - tempa[1] - tempa[0]) / tempa[1, 0], color="#754D32")
    plt.legend([r"Zombies", r"Humans", r"Tripped"], prop={"size": 14}, loc="upper right")
    location = "resultsZ/full_zombie" + str(its) + "_" + str(A)
    plt.savefig(location + ".svg", bbox_inches="tight")
    plt.close("all")
    #######
    plt.title(r"\textbf{Zombie apocalypse vs. SIR}", fontsize=16)
    plt.xlabel(r"time steps", fontsize=16)
    plt.ylabel(r"$\rho$", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.plot(range(num_iteration), tempa[0] / tempa[1, 0], color="#0C8C00")
    plt.plot(range(num_iteration), (tempa[1, 0] - tempa[1] - tempa[0]) / tempa[1, 0], color="#754D32")
    plt.legend([r"Zombies alias infected", r"Tripped alias recovered"], prop={"size": 14}, loc="upper right")
    SIR(alpha, beta)
    location = "resultsZ/zombie_vs_SIR" + str(its) + "_" + str(A)
    plt.savefig(location + ".svg", bbox_inches="tight")
    plt.close("all")



def plot_walker2D_Z(walkers):
    walker = np.copy(walkers)
    Non_Zombie = np.real(walker)
    Z = np.imag(walker)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    Z[Z < 1] = 0
    Z[Z >= 1] = 1
    Non_Zombie[Non_Zombie < 1] = 0
    Non_Zombie[Non_Zombie >= 1] = 1

    ax[0].imshow(Non_Zombie, cmap=plt.get_cmap("gray_r"))
    for loc in range(0, np.shape(walkers)[0]):
        ax[0].axvline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)
        ax[0].axhline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)

    ax[1].imshow(Z, cmap=plt.get_cmap("Greens"))
    for loc in range(0, np.shape(walkers)[0]):
        ax[1].axvline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)
        ax[1].axhline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)
    plt.show(block=False)
    plt.pause(17)
    plt.close("all")


def annihilation_Z(walkers):
    walker = walkers.copy().astype(complex)
    zombie = np.imag(walker).astype(complex)
    non_zombie = np.real(walker).astype(complex)
    for i in range(np.shape(zombie)[0]):
        for j in range(np.shape(zombie)[0]):
            if zombie[i, j] > 0:
                non_zombie[i, j] = 0 + non_zombie[i, j] * 1j
    total = 1j * zombie + non_zombie
    return total


def iterations_Z(walkers, num_iteration, tripchance=1 / 33):
    directions = get_directions(dim=2, diag=False)

    temp = np.pad(walkers, 1)
    N_walkers = np.zeros((num_iteration, 2))  # i=0 zombies, i=1 non-zombies

    for i in range(num_iteration):
        Non_Zombie = np.real(temp).astype(complex)
        Zombies = np.imag(temp).astype(complex)
        N_walkers[i] = np.array([np.real(np.sum(Zombies)), np.sum(np.real(Non_Zombie))])

        indicesZ = np.nonzero(Zombies)
        indicesNz = np.nonzero(Non_Zombie)

        numbersZ = (np.floor(len(directions) * np.random.rand(int(N_walkers[i, 0])))).astype(int)
        numbersZ2 = np.random.rand(int(N_walkers[i, 0]))
        numbersNz = (np.floor(len(directions) * np.random.rand(int(N_walkers[i, 1])))).astype(int)

        for i in range(0, np.shape(indicesZ)[1]):  # zombies
            x, y = indicesZ[0][i], indicesZ[1][i]
            tamp = np.copy(Zombies)

            for k in range(0, int(np.real(tamp[x, y]))):
                if numbersZ2[i] < tripchance and (N_walkers[i, 0] > 1 or N_walkers[i, 0] > N_walkers[i, 1]):
                    Zombies[x, y] -= 1 + 0 * 1j
                    numbersZ2[i] = np.random.random()
                else:
                    Zombies[x - 1 : x + 2, y - 1 : y + 2] += directions[numbersZ[i]].astype(complex)
                    numbersZ[i] = int(np.floor(len(directions) * np.random.random()))

        for i in range(0, np.shape(indicesNz)[1]):  # non-zombies
            x, y = indicesNz[0][i], indicesNz[1][i]
            tamp = np.copy(Non_Zombie)
            for i in range(0, int(np.real(tamp[x, y]))):
                Non_Zombie[x - 1 : x + 2, y - 1 : y + 2] += directions[numbersNz[i]].astype(complex)
                numbersNz[i] = int(np.floor(len(directions) * np.random.random()))

        temp = Non_Zombie + 1j * Zombies

        temp[1, :] += temp[-1, :]
        temp[:, 1] += temp[:, -1]
        temp[-2, :] += temp[0, :]
        temp[:, -2] += temp[:, 0]
        temp = np.copy(temp[1:-1, 1:-1])

        temp = annihilation_Z(temp)
        temp = np.pad(temp, 1)

    return N_walkers


def SIR(alpha, beta):
    time_period = 400
    I,R = np.zeros(int(time_period / 0.01)), np.zeros(int(time_period / 0.01))
    I[0] = 1
    S = 1e6 - I - R
    for i in range(0, int(time_period / 0.01) - 1):
        I[i + 1] = I[i] + 0.01 * beta * S[i] * I[i] / 1e6 - 0.01 * alpha * I[i]
        S[i + 1] = S[i] - 0.01 * beta * S[i] * I[i] / 1e6
        R[i + 1] = 1e6 - I[i + 1] - S[i + 1]
    time = np.arange(0.0, time_period, 0.01)
    plt.plot(time, I * 10e-7)
    plt.plot(time, R * 10e-7)
