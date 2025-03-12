import json
import sys
from matplotlib import pyplot as plt


def plot_route(coordinates, route, c="black", zorder=0, w=1):
    xs = []
    ys = []
    for i in range(len(route)):
        p1 = coordinates[route[i - 1]]
        p2 = coordinates[route[i]]
        xs.append(p1[0])
        xs.append(p2[0])
        ys.append(p1[1])
        ys.append(p2[1])
    plt.plot(xs, ys, c=c, zorder=zorder, linewidth=w)


def plot_locations(coordinates):
    xs = [c[0] for c in coordinates]
    ys = [c[1] for c in coordinates]
    # for [x, y] in coordinates:
    plt.scatter(xs, ys, c="red", zorder=2)


# Expects permutation where locations are indexed from 0
def plot_solution(coordinates, solution, opt_solution=None):
    if opt_solution is not None:
        plot_route(coordinates, opt_solution, c="green", zorder=0, w=5)
    plot_route(coordinates, solution, c="black", zorder=1, w=1)
    plot_locations(coordinates)
    plt.savefig("../viz.png")
    # plt.show()


def read_instance(instance_path):
    with open(instance_path) as f:
        instance = json.load(f)
        coords = instance["Coordinates"]
        best_solution = instance["GlobalBest"]
    return coords, best_solution


def read_solution(solution_path):
    with open(SOLUTION_PATH) as f:
        return json.load(f)


mode = sys.argv[1]
if mode == "INSTANCE_BEST":
    INSTANCE_PATH = sys.argv[2]
    SOLUTION_PATH = None
    coords, best_solution = read_instance(INSTANCE_PATH)
    plot_solution(coords, best_solution)
elif mode == "SINGLE_SOLUTION":
    INSTANCE_PATH = sys.argv[2]
    SOLUTION_PATH = sys.argv[3]
    coords, _ = read_instance(INSTANCE_PATH)
    solution = read_solution(SOLUTION_PATH)
    plot_solution(coords, solution)
elif mode == "MULTI_SOLUTION":
    INSTANCE_PATH = sys.argv[2]
    SOLUTION_PATH = sys.argv[3]
    coords, best_solution = read_instance(INSTANCE_PATH)
    solution = read_solution(SOLUTION_PATH)
    plot_solution(coords, solution, best_solution)
else:
    raise "Unknown mode"
