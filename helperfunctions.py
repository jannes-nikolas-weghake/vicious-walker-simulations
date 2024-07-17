import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


def VW(L, occu, dim):
    if dim == 3:
        if occu > 1:
            return np.ones((L, L, L))

        grid = np.zeros(L**3)
        n = 0
        N = np.floor(occu * L**3)
        while n < N:
            index = int(np.random.random() * L**3)
            if grid[index] == 0:
                grid[index] = 1
                n += 1

        grid = np.reshape(grid, (L, L, L))

    if dim == 2:
        if occu > 1:
            return np.ones((L, L))
        N = np.floor(occu * L**2)
        grid = np.zeros(L**2)
        n = 0
        while n < N:
            index = int(np.random.random() * L**2)
            if grid[index] == 0:
                grid[index] = 1
                n += 1

        grid = np.reshape(grid, (L, L))

    if dim == 1:
        if occu > 1:
            return np.ones(L)
        N = np.floor(occu * L)
        grid = np.zeros(L)
        n = 0
        while n < N:
            index = int(np.random.random() * L)
            if grid[index] == 0:
                grid[index] = 1
                n += 1

    return grid


def plot_walker2D(walkers):
    temp = walkers / np.max(walkers)
    plt.imshow(temp, cmap=plt.get_cmap("gray_r"))
    for loc in range(0, np.shape(walkers)[0]):
        plt.axvline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)
        plt.axhline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)

    plt.show(block=False)
    plt.pause(17)
    plt.close("all")


def save_walker2D(walkers, title):
    plt.imshow(walkers, cmap=plt.get_cmap("gray_r"))
    for loc in range(0, np.shape(walkers)[0]):
        plt.axvline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)
        plt.axhline(loc + 0.5, alpha=0.2, color="#b0b0b0", linestyle="-", linewidth=1)
    plt.savefig("pictures/" + str(title), dpi=300)
    plt.close("all")


def iterations(walkers, num_iteration, dim, diag, chance):
    directions = get_directions(dim=dim, diag=diag)

    temp = np.pad(walkers, 1)
    N_walkers = np.zeros(num_iteration)

    for i in range(num_iteration):
        indices = np.nonzero(temp)

        N_walkers[i] = np.shape(indices)[1]

        numbers = np.floor(len(directions) * np.random.rand(np.shape(indices)[1])).astype(int)

        for i in range(np.shape(indices)[1]):
            if np.shape(indices)[1] < 3:
                pass

            if dim == 1:
                x = indices[0][i]
                for j in range(0, int(temp[x])):
                    temp[x - 1 : x + 2] += directions[numbers[i]]
                    numbers[i] = int(np.floor(len(directions) * np.random.random()))
            elif dim == 2:
                x, y = indices[0][i], indices[1][i]
                for j in range(0, int(temp[x, y])):
                    temp[x - 1 : x + 2, y - 1 : y + 2] += directions[numbers[i]]
                    numbers[i] = int(np.floor(len(directions) * np.random.random()))
            elif dim == 3:
                x, y, z = indices[0][i], indices[1][i], indices[2][i]
                for j in range(0, int(temp[x, y, z])):
                    temp[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2] += directions[numbers[i]]
                    numbers[i] = int(np.floor(len(directions) * np.random.random()))

            #  reset number if multiple walkers are on the same field
        if dim == 1:
            temp[1] += temp[-1]
            temp[-2] += temp[0]
            temp = temp[1:-1]

        elif dim == 2:
            temp[1, :] += temp[-1, :]
            temp[:, 1] += temp[:, -1]
            temp[-2, :] += temp[0, :]
            temp[:, -2] += temp[:, 0]
            temp = temp[1:-1, 1:-1]

        elif dim == 3:
            temp[1, :, :] += temp[-1, :, :]
            temp[:, 1, :] += temp[:, -1, :]
            temp[:, :, 1] += temp[:, :, -1]
            temp[-2, :, :] += temp[0, :, :]
            temp[:, -2, :] += temp[:, 0, :]
            temp[:, :, -2] += temp[:, :, 0]
            temp = temp[1:-1, 1:-1, 1:-1]

        temp = annihilation(temp, chance=chance)

        temp = np.pad(temp, 1)

    return N_walkers


def getline_errors(matrix, dim):
    (its, num_iteration) = np.shape(matrix)
    parameters = np.zeros((its, 2))

    x = range(num_iteration)

    for i in range(its):
        parameters[i] = fit_line(x[11:5000], matrix[i, 11:5000])

    temp = np.average(parameters[:, 0]), np.average(parameters[:, 1])
    temp = np.sum(np.abs(temp[0] - parameters[:, 0])) / its, np.sum(np.abs(temp[1] - parameters[:, 1])) / its

    return temp


def get_walker_plots(matrix, dim, V, diag, chance, occu):
    import matplotlib

    matplotlib.rcParams["text.usetex"] = True
    plt.rc("font", family="serif")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    together = False

    (its, num_iteration) = np.shape(matrix)
    (avg, std) = (np.zeros(num_iteration), np.zeros(num_iteration))

    x = range(num_iteration)
    for i in x:
        avg[i], std[i] = np.average(matrix[:, i]), np.std(matrix[:, i])

    std = 10 * np.convolve(std, np.ones(50) / 50, mode="same") / np.sqrt(its)
    start_point = get_start_point(chance)
    a, b = fit_line(x[start_point:num_iteration], avg[start_point:num_iteration])
    plt.xlabel(r"time steps ", fontsize=16)
    plt.ylabel(r"$\rho$", fontsize=16)

    aspect_ratio = 1 / 2
    axes = plt.gca()
    axes.set_aspect(abs((axes.get_xlim()[0] - axes.get_xlim()[1]) / (axes.get_ylim()[0] - axes.get_ylim()[1])) * aspect_ratio)
    plt.ylim(top=1.2)
    if not together:
        plt.loglog(x, avg, color="#0000a7")
        plt.fill_between(x, avg - std, avg + std, alpha=0.25)
        plt.loglog(x[11:num_iteration], b * x[11:num_iteration] ** a, color="#c1272d")

        a = int(a * 1000) / 1000
        b = int(b * 1000) / 1000
        plt.ylim(bottom=1e-4)
        plt.xlabel(r"time steps ", fontsize=16)
        plt.ylabel(r"$\rho$", fontsize=16)
        text = r"$y=\,$" + str(a) + r"$x\,+\,$" + str(b)
        plt.text(100, avg[15], text, fontsize=14)

        if dim == 1:
            plt.title(r"\textbf{Density decay of vicious Walker in 1D}", fontsize=16)
        elif dim == 2:
            plt.title(r"\textbf{Density decay of vicious Walker in 2D}", fontsize=16)
        elif dim == 3:
            plt.title(r"\textbf{Density decay of vicious Walker in 3D}", fontsize=16)

        infos = r"$\#$ iterations = " + str(its)
        if diag:
            infos += "\n" + r"diagonal moves"
        else:
            infos += "\n" + r"no diagonal moves"
        infos += "\n" + r"volume = " + "{:.0e}".format(V)
        infos += "\n" + r"$\rho(0)$ = " + str(occu) + ","
        infos += r"\;\;$\nu$ = " + str(chance)
        plt.text(1, 1.4 * 1e-4, infos, fontsize=12)

        location = "resultsN/" + str(dim) + "D_" + str(its) + "_" + str(int(V)) + "_" + str(chance)
        if diag:
            location = location + "_diag"

        plt.savefig(location + ".svg", bbox_inches="tight")

        plt.close("all")
        print("plot saved at " + location)
    else:
        plt.loglog(x, avg, label=str(dim) + "D")
        plt.fill_between(x, avg - std, avg + std, alpha=0.25)
        plt.loglog(x[11:num_iteration], b * x[11:num_iteration] ** a)

        if dim == 3:
            plt.title(r"\textbf{Comparison in different dimensions}", fontsize=16)

            plt.legend(prop={"size": 14})
            location = "resultsN/all_" + str(its) + "_" + str(int(V))
            if diag:
                location = location + "_diag"
            plt.savefig(location + ".svg", bbox_inches="tight")

            plt.close("all")
            print("plot saved at " + location)

    timp = getline_errors(matrix, dim)
    return np.array([[a, timp[0]], [b, timp[1]]])


def fit_line(x, y):
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(objective, x, y)
    a, b = popt

    return a, b


def objective(x, a, b):
    return b * x**a


def annihilation(walkers, chance):
    if chance == 1:
        return np.mod(walkers, 2)
    else:
        mask = np.random.choice(a=[0, 1], size=np.shape(walkers), p=[1 - chance, chance])

        modulo = np.mod(walkers, 2)
        walkers = walkers - np.multiply(mask, (walkers - modulo))

        return walkers


def get_directions(dim, diag=False):
    if dim == 1:
        left = np.array([1, -1, 0])
        right = np.array([0, -1, 1])
        return [left, right]

    elif dim == 2 and not diag:
        out = []
        for i in range(0, 9):
            if i != 4:
                temp = np.zeros(9)
                temp[i], temp[4] = 1, -1
                if i % 2 == 1:
                    out.append(np.reshape(temp, (3, 3)))
        return out

    elif dim == 3 and not diag:
        out = []
        for i in range(0, 27):
            if i in [4, 10, 12, 14, 16, 22]:
                temp = np.zeros(27)
                temp[i], temp[13] = 1, -1
                out.append(np.reshape(temp, (3, 3, 3)))
        return out

    elif dim == 2 and diag:
        out = []
        for i in range(0, 9):
            if i != 4:
                temp = np.zeros(9)
                temp[i], temp[4] = 1, -1
                out.append(np.reshape(temp, (3, 3)))
        return out

    elif dim == 3 and diag:
        out = []
        for i in range(0, 27):
            if i != 13:
                temp = np.zeros(27)
                temp[i], temp[13] = 1, -1
                out.append(np.reshape(temp, (3, 3, 3)))
        return out


def get_start_point(chance):
    x = int(13163 / (506 * chance**0.652987 + 1) - 6)
    return x
