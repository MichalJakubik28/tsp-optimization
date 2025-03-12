import argparse
import random
import sys
import json
import time


def read_instance_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def write_instance_json(solution, file_path):
    with open(file_path, "w") as f:
        json.dump(solution, f)


def calculate_path_length(path, matrix):
    """Full path computation"""
    return (
        sum([matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)])
        + matrix[path[-1]][path[0]]
    )


def generate_initial_solution(matrix):
    """Initial solution computation using a greedy algorithm."""
    order = [0]

    cur_node = 0
    while len(order) != len(matrix):
        min_neighbor = -1
        closest_neighbor = None
        for i in range(len(matrix)):
            if i not in order and (
                min_neighbor == -1 or matrix[cur_node][i] < min_neighbor
            ):
                min_neighbor = matrix[cur_node][i]
                closest_neighbor = i
        order.append(closest_neighbor)
        cur_node = closest_neighbor

    return order


def perform_2opt(solution, matrix, old_solution_value):
    """Perform 2-opt with incremental path computation."""
    n = len(matrix)
    city1_idx, city2_idx = tuple(sorted(random.sample(solution, 2)))

    # Neighboring cities share only one edge, not a valid 2-opt
    while city2_idx - city1_idx == 1 or (city1_idx == 0 and city2_idx == n - 1):
        city1_idx, city2_idx = tuple(sorted(random.sample(solution, 2)))

    former_distances = (matrix[solution[city1_idx - 1]][solution[city1_idx]]
                        + matrix[solution[city2_idx]][solution[(city2_idx + 1) % n]])
    new_distances = (matrix[solution[city1_idx - 1]][solution[city2_idx]]
                        + matrix[solution[city1_idx]][solution[(city2_idx + 1) % n]])
    
    new_solution = (solution[:city1_idx]
                    + solution[city1_idx:city2_idx + 1][::-1]
                    + (solution[city2_idx + 1:] if city2_idx + 1!= n else []))
    new_solution_value = old_solution_value - former_distances + new_distances

    return new_solution, new_solution_value


def perform_kswap(solution, matrix, old_solution_value, k):
    """Perform k-swap with incremental path computation."""
    n = len(matrix)
    cities = random.sample(solution, k)
    random.shuffle(cities)
    city_indices = [solution.index(city) for city in cities]

    former_distances = 0
    new_distances = 0

    new_solution = solution.copy()
    for i in range(k):
        former_distances += (matrix[cities[i]][new_solution[city_indices[i] - 1]]
                                + matrix[cities[i]][new_solution[(city_indices[i] + 1) % n]])
        new_distances += (matrix[cities[i - 1]][new_solution[city_indices[i] - 1]]
                                + matrix[cities[i - 1]][new_solution[(city_indices[i] + 1) % n]])
        new_solution[city_indices[i]] = cities[i - 1]

    new_solution_value = old_solution_value - former_distances + new_distances
    return new_solution, new_solution_value


def remove_cities_from_solution(solution, remove_count):
    removed_cities = set(random.sample(solution, remove_count))
    new_solution = list(filter(lambda c: c not in removed_cities, solution))

    removed_cities = list(removed_cities)
    random.shuffle(removed_cities)

    return new_solution, removed_cities


def reinsert_cities_to_solution(solution, cities, matrix):
    """Insert cities to solution using a greedy algorithm."""
    for c in cities:
        min_delta = float("inf")
        best_i = None

        # Best position for a city chosen as the one with
        # minimal sum of distances to its neighbors
        for i in range(len(solution)):
            if i == 0:
                delta = (
                    matrix[solution[-1]][c]
                    + matrix[solution[i]][c]
                    - matrix[solution[-1]][solution[i]]
                )
            else:
                delta = (
                    matrix[solution[i - 1]][c]
                    + matrix[solution[i]][c]
                    - matrix[solution[i - 1]][solution[i]]
                )
            if delta < min_delta:
                min_delta = delta
                best_i = i

        solution.insert(best_i, c)

    return solution


def debug_print(message, debug_mode=False):
    if debug_mode:
        print(message)


def find_solution(matrix, time_limit, debug_mode=False):
    n = len(matrix)
    start_time = time.time()

    initial_solution = generate_initial_solution(matrix)
    initial_solution_value = calculate_path_length(initial_solution, matrix)

    current_solution = initial_solution
    current_solution_value = initial_solution_value

    best_solution = current_solution
    best_solution_value = current_solution_value

    if n == 1:
        return best_solution, best_solution_value

    # Destroy-repair metaheuristic until the time runs out
    iteration = 0
    while (time_diff := time.time() - start_time) < time_limit:
        iteration += 1

        # Removing 20% - 40% of cities
        number_to_remove = random.randint(1 * n // 5, 2 * n // 5)
        new_solution, removed_cities = remove_cities_from_solution(
            current_solution, number_to_remove
        )

        new_solution = reinsert_cities_to_solution(new_solution, removed_cities, matrix)
        new_solution_value = calculate_path_length(new_solution, matrix)

        for _ in range(10):
            # Try to improve the repair further by 2-opt and k-swap
            if random.random() < 0.8:
                moved_solution, moved_solution_value = perform_2opt(
                    new_solution, matrix, new_solution_value
                )
            else:
                k = random.randint(2, min(n, 4))
                moved_solution, moved_solution_value = perform_kswap(
                    new_solution, matrix, new_solution_value, k
                )

            if moved_solution_value < new_solution_value:
                new_solution = moved_solution
                new_solution_value = moved_solution_value

        # Solution accepted if better than the current one or sometimes
        # worse to overcome local minima
        if new_solution_value < current_solution_value or (random.random() < 0.1):
            current_solution = new_solution
            current_solution_value = new_solution_value

        if current_solution_value < best_solution_value:
            debug_print(
                f"Improvement during iteration {iteration}, time {time_diff:.4f}, new value: {current_solution_value}",
                debug_mode,
            )
            best_solution = current_solution
            best_solution_value = current_solution_value

    return best_solution, best_solution_value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TSP Solver with debug and timeout options"
    )

    parser.add_argument("instance_path", help="Path to the instance JSON file")
    parser.add_argument("output_path", help="Path to save the solution JSON file")

    parser.add_argument(
        "-D", "--debug", action="store_true", help="Enable debug output"
    )
    parser.add_argument("-T", "--timeout", type=float, help="Custom timeout in seconds")

    return parser.parse_args()


def main():
    args = parse_arguments()

    instance = read_instance_json(args.instance_path)
    matrix = instance["Matrix"]
    time_limit = instance["Timeout"]

    if args.timeout is not None:
        time_limit = args.timeout

    solution, solution_value = find_solution(
        matrix,
        time_limit * 0.95,
        debug_mode=args.debug,
    )
    write_instance_json(solution, args.output_path)

    debug_print("Our best value: " + str(solution_value), args.debug)
    debug_print("Global best value: " + str(instance["GlobalBestVal"]), args.debug)


if __name__ == "__main__":
    main()
