# to run: mpiexec /np 4 python main.py
# 1 core: 260.3040335178375
# 2 core: 203.06737804412842
# 4 core: 120.15487766265869
# 8 core: 125.28175735473633

import numpy as np
from helperfunctions import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import zombies as z


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))


def simulate_2D(occu=0.75, its=2, L=10):
    num_iteration = 3000
    temp = []
    for i in tqdm(range(its)):
        walkers = VW(L, occu, dim=2)
        temp.append(iterations(walkers, num_iteration, dim=2, diag=False) / (its * occu * L**2))

    temp = np.array(temp)
    get_walker_plots(temp, dim=2, V=L**2, diag=False, chance=1)


def simulate_multicore(dim, V, occu=0.75, its=10, diag=False, chance=1):
    temp = []
    num_iteration = 10000

    if its < size:
        its = size

    if dim == 1:
        L = int(V)
    elif dim == 2:
        L = int(np.sqrt(V))
    elif dim == 3:
        L = int(np.cbrt(V))

    if rank == 0:
        it_per_proc = int(np.floor(its / size))

        for i in range(1, size):
            comm.send(it_per_proc, dest=i)  # seed other processes

    elif rank != 0:
        it_per_proc = comm.recv(source=0)  # get parameter

    walkers = VW(L, occu, dim=dim)
    for i in tqdm(range(it_per_proc)):
        buffer = iterations(walkers, num_iteration, dim=dim, diag=diag, chance=chance)
        temp.append(buffer / (occu * V))  # it_per_proc *
    temp = np.array(temp)

    if rank == 0:
        for i in range(1, size):
            temp = np.append(comm.recv(source=i), temp, axis=0)
        return get_walker_plots(temp, dim=dim, V=V, diag=diag, chance=chance, occu=occu)

    elif rank != 0:
        comm.send(temp, dest=0)  #  finish up processes


def normal(diag):
    Vf = 1e3
    itsf = 64
    fits = []
    fits.append(simulate_multicore(dim=1, V=Vf, occu=0.75, its=itsf, diag=diag, chance=1))
    fits.append(simulate_multicore(dim=2, V=Vf, occu=0.75, its=itsf, diag=diag, chance=1))
    fits.append(simulate_multicore(dim=3, V=Vf, occu=0.75, its=itsf, diag=diag, chance=1))

    if rank == 0:
        print(np.round(np.array(fits), 4))


def zombie():
    z.simulate_zombies(A=13**2, its=64, alpha=1 / 35, beta=0.43)


def chance(dim):
    Vf = 1000
    itsf = 40
    coeffcient_As = []
    coeffcient_Bs = []

    a1 = [1.0e-03, 1.0e-02, 1.0e-01]
    x = np.outer(a1, range(1, 10)).flatten()

    # a1 = [1.0e-01]
    # x = np.outer(a1, range(1, 3)).flatten()

    x = np.round(x, 5)
    for i in x:
        temp = simulate_multicore(dim=dim, V=Vf, its=itsf, occu=0.75, diag=False, chance=i)
        if rank == 0:
            coeffcient_As.append(-1 * temp[0, 0])
            coeffcient_Bs.append(temp[1, 0])

    if rank == 0:
        print(x)
        print(coeffcient_As)
        plt.loglog(x, coeffcient_As)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.xlabel(r"chance $\nu$", fontsize=16)
        plt.ylim([0.1, 1.2])
        plt.ylabel(r"coefficient amplitude", fontsize=16)
        plt.title(r"\textbf{absolute value of $a$ in }" + str(dim) + r"D", fontsize=16)
        plt.savefig("chance/a" + str(dim) + ".svg", bbox_inches="tight")
        plt.close("all")
        plt.figure()
        plt.loglog(x, coeffcient_Bs)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.xlabel(r"chance $\nu$", fontsize=16)
        plt.ylabel(r"coefficient amplitude", fontsize=16)
        plt.title(r"\textbf{absolute value of $b$ in }" + str(dim) + r"D", fontsize=16)
        plt.savefig("chance/b" + str(dim) + ".svg", bbox_inches="tight")
        plt.close("all")


chance(dim=1)
# chance(dim=2)
# chance(dim=3) 
