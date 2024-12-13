import random
import math
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from nashpy import Game
from genetic_tree import GeneticTree
from datetime import datetime
from tree.crosser import cross_trees
from perturbation import Perturbation
from perturbation import MixedPerturbation
from genetic_tree import MixedGeneticTree

# Parameters
TREES_POPULATION_SIZE = 200
PERTURBATIONS_POPULATION_SIZE = 2000
TOP_N_TREES = 20

TREES_MUTATION_RATE = 0.8
PERTURBATIONS_MUTATION_RATE = 0.8
MUTATION_REPEATS = 20

TREES_CROSSOVER_RATE = 0.9
PERTURBATIONS_CROSSOVER_RATE = 0.9

GENERATIONS = 1000
GENERATIONS_NO_IMPROVEMENT_LIMIT = 200
GENERATIONS_BETWEEN_SWITCH = 20

OPTIMIZATION_METRIC = 'MAX_REGRET'  # PERTURBED_ACCURACY


def islands_algorithm(data, labels, num_islands=10, generations_per_island=20, k_top=2, migration_topology='ring'):
    # Initialize islands
    islands = []
    for _ in range(num_islands):
        decision_trees = initialize_decision_trees(TREES_POPULATION_SIZE)
        perturbations = initialize_perturbations(PERTURBATIONS_POPULATION_SIZE, data)
        islands.append({
            'decision_trees': decision_trees,
            'perturbations': perturbations,
            'trees_hall_of_fame': [],
            'perturbations_hall_of_fame': []
        })

    # Evolve islands
    no_improvement = 0
    previous_best_fitness = -math.inf
    for generation in range(GENERATIONS):
        for island in islands:
            # Evolve decision trees and perturbations on the island
            for _ in range(generations_per_island):
                island['decision_trees'] = crossover_population(island['decision_trees'])
                island['decision_trees'] = mutate_population(island['decision_trees'])
                island['perturbations'] = crossover_population(island['perturbations'])
                island['perturbations'] = mutate_population(island['perturbations'])

                # Update halls of fame
                island['trees_hall_of_fame'], island['perturbations_hall_of_fame'] = update_halls_of_fame(
                    island['decision_trees'], island['perturbations'],
                    island['trees_hall_of_fame'], island['perturbations_hall_of_fame']
                )

                # Evaluate fitness
                evaluate_fitness_decision_trees(
                    island['decision_trees'], island['perturbations'], island['perturbations_hall_of_fame'], data, labels
                )
                evaluate_fitness_perturbations(
                    island['decision_trees'], island['perturbations'], island['trees_hall_of_fame'], data, labels
                )

                # Selection
                island['decision_trees'] = selection(island['decision_trees'])
                island['perturbations'] = selection(island['perturbations'])

        # Migration (implement based on migration_topology)
        if migration_topology == 'ring':
            migrate_ring(islands, k_top)

        # Check for overall best and stopping condition
        all_trees = [tree for island in islands for tree in island['decision_trees']]
        all_trees.sort(key=lambda x: x.fitness, reverse=True)
        if previous_best_fitness < all_trees[0].fitness:
            previous_best_fitness = all_trees[0].fitness
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= GENERATIONS_NO_IMPROVEMENT_LIMIT:
            break

    best_trees = [max(island['decision_trees'], key=lambda x: x.fitness) for island in islands]
    return best_trees  # Or the combined forest
    # return construct_ensemble(islands)


# Coevolutionary algorithm (CoEvoRDT)
def coevolutionary_algorithm(data, labels):
    # Initialize the populations
    decision_trees = initialize_decision_trees(TREES_POPULATION_SIZE)
    perturbations = initialize_perturbations(PERTURBATIONS_POPULATION_SIZE, data)

    # Initialize the halls of fame
    trees_hall_of_fame = []
    perturbations_hall_of_fame = []

    no_improvement = 0
    previous_best_fitness = -math.inf
    for generation in range(int(GENERATIONS / (2*GENERATIONS_BETWEEN_SWITCH))):

        # decision trees development
        for ig in range(GENERATIONS_BETWEEN_SWITCH):
            # Crossover and mutations
            decision_trees = crossover_population(decision_trees)
            decision_trees = mutate_population(decision_trees)

            trees_hall_of_fame, perturbations_hall_of_fame = update_halls_of_fame(decision_trees, perturbations,
                                                                                  trees_hall_of_fame,
                                                                                  perturbations_hall_of_fame)

            # Evaluate the fitness of decision trees against perturbations
            evaluate_fitness_decision_trees(decision_trees, perturbations, perturbations_hall_of_fame, data, labels)
            evaluate_fitness_perturbations(decision_trees, perturbations, trees_hall_of_fame, data, labels)

            # Selection
            decision_trees = selection(decision_trees)

        # perturbations development
        for ig in range(GENERATIONS_BETWEEN_SWITCH):
            # Crossover and mutations
            perturbations = crossover_population(perturbations)
            perturbations = mutate_population(perturbations)

            trees_hall_of_fame, perturbations_hall_of_fame = update_halls_of_fame(decision_trees, perturbations,
                                                                                  trees_hall_of_fame,
                                                                                  perturbations_hall_of_fame)

            # Evaluate the fitness of decision trees against perturbations
            evaluate_fitness_decision_trees(decision_trees, perturbations, perturbations_hall_of_fame, data, labels)
            evaluate_fitness_perturbations(decision_trees, perturbations, trees_hall_of_fame, data, labels)

            # Selection
            perturbations = selection(perturbations)

            decision_trees.sort(key=lambda x: x.fitness, reverse=True)

            # Check stop condition
            if previous_best_fitness < decision_trees[0].fitness:
                previous_best_fitness = decision_trees[0].fitness
                no_improvement = 0
            else:
                no_improvement += 2*GENERATIONS_BETWEEN_SWITCH

            if no_improvement >= GENERATIONS_NO_IMPROVEMENT_LIMIT:
                break

    return decision_trees[0]


# Initialize the decision trees population
def initialize_decision_trees(population_size):
    decision_trees = []
    for _ in range(population_size):
        decision_tree = GeneticTree()
        decision_trees.append(decision_tree)
    return decision_trees


# Initialize the perturbations population
def initialize_perturbations(population_size, data):
    perturbations = []
    for _ in range(population_size):
        perturbation = Perturbation()
        perturbation.initialize_perturbation(data.shape[0], data.shape[1], EPS) # Generate a random perturbation vector
        perturbations.append(perturbation)
    return perturbations


# Evaluate the fitness of decision trees against perturbations
def evaluate_fitness_decision_trees(decision_trees, perturbations, perturbations_hof, data, labels):
    for tree in decision_trees:
        tree.fitness = -math.inf
        for perturbation in perturbations:
            perturbed_data = perturbation.apply(data)
            perturbed_predictions = tree.predict(perturbed_data)

            if OPTIMIZATION_METRIC == 'PERTURBED_ACCURACY':
                perturbed_accuracy = accuracy_score(labels, perturbed_predictions)
                tree.fitness = min(perturbed_accuracy, tree.fitness)
            if OPTIMIZATION_METRIC == 'MAX_REGRET':
                if perturbation.cart_result is None:
                    cart = DecisionTreeClassifier()
                    cart.fit(perturbed_data, labels)
                    perturbation.cart_result = accuracy_score(labels, cart.predict(perturbed_data))
                max_regret = accuracy_score(labels, perturbed_predictions) - perturbation.cart_result
                tree.fitness = min(max_regret, tree.fitness)

        for mixed_perturbation in perturbations_hof:
            tree.fitness = min(mixed_perturbation.compute_fitness(tree, data, labels, OPTIMIZATION_METRIC), tree.fitness)


# Evaluate the fitness of perturbations against decision trees
def evaluate_fitness_perturbations(decision_trees, perturbations, decision_trees_hof, data, labels):
    for perturbation in perturbations:
        perturbed_data = perturbation.apply(data)
        perturbation.fitness = -math.inf
        decision_trees.sort(key=lambda x: x.fitness, reverse=True)

        for i in range(0, TOP_N_TREES):
            perturbed_predictions = decision_trees[i].predict(perturbed_data)
            perturbed_accuracy = accuracy_score(labels, perturbed_predictions)
            perturbation.fitness = min(-perturbed_accuracy, perturbation.fitness)

        for mixed_tree in decision_trees_hof:
            perturbation.fitness = min(mixed_tree.compute_fitness(perturbation, data, labels), perturbation.fitness)


# Binary tournament selection
def selection(population):
    selected_population = []
    for _ in range(len(population)):
        idx1 = random.randint(0, len(population) - 1)
        idx2 = random.randint(0, len(population) - 1)
        if population[idx1].fitness > population[idx2].fitness:
            selected_population.append(population[idx1])
        else:
            selected_population.append(population[idx2])
    return selected_population


def crossover_perturbations(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    offspring = parent1[:crossover_point] + parent2[crossover_point:]
    return offspring


def mutate_perturbation(perturbation):
    for m in range(0, MUTATION_REPEATS):
        rows, cols = perturbation.p.shape
        random_row = np.random.randint(0, rows)
        random_col = np.random.randint(0, cols)

        perturbation.p[random_row, random_col] = np.random.uniform(-EPS, EPS)
    return perturbation


def mutate_tree(tree):
    return tree.mutator.mutate()


def crossover_trees(parent1, parent2):
    return cross_trees(parent1, parent2)


def mutate_population(population):
    mutated_population = population
    for individual in population:
        if isinstance(individual, GeneticTree):
            if random.random() < TREES_MUTATION_RATE:
                mutated_population.append(mutate_tree(individual))
        else:
            if random.random() < PERTURBATIONS_MUTATION_RATE:
                mutated_population.append(mutate_perturbation(individual))

    return mutated_population


def crossover_population(population):
    crossed_population = population
    for individual in population:
        if isinstance(individual, GeneticTree):
            if random.random() < TREES_CROSSOVER_RATE:
                crossed_population.append(crossover_trees(individual, population[np.random.randint(0, len(population))]))
        else:
            if random.random() < PERTURBATIONS_CROSSOVER_RATE:
                crossed_population.append(crossover_perturbations(individual, population[np.random.randint(0, len(population))]))

    return crossed_population


def compute_mixed_nash_equilibrium(trees, perturbations):
    trees_fitness = np.array([t.fitness for t in trees])
    perturbations_fitness = np.array([p.fitness for p in perturbations])
    nash_game = Game(trees_fitness, perturbations_fitness)
    equilibrium = nash_game.lemke_howson()
    print("Decision Trees Strategy:", equilibrium[0])
    print("Perturbations Strategy:", equilibrium[1])
    return MixedGeneticTree(trees, equilibrium[0]), MixedPerturbation(perturbations, equilibrium[1]),


def update_trees_hall_of_fame(hall_of_fame):
    # Select the top individuals from the hall of fame
    top_individuals = sorted(hall_of_fame, key=lambda x: x.fitness, reverse=True)[:TREES_POPULATION_SIZE]
    return top_individuals


def update_perturbations_hall_of_fame(hall_of_fame):
    # Select the top individuals from the hall of fame
    top_individuals = sorted(hall_of_fame, key=lambda x: x.fitness, reverse=True)[:PERTURBATIONS_POPULATION_SIZE]
    return top_individuals


def update_halls_of_fame(trees, perturbations, trees_hall_of_fame, perturbations_hall_of_fame):
    mixed_tree, mixed_perturbation = compute_mixed_nash_equilibrium(trees, perturbations)
    trees_hall_of_fame = update_trees_hall_of_fame(trees_hall_of_fame.append(mixed_tree))
    perturbations_hall_of_fame = update_perturbations_hall_of_fame(perturbations_hall_of_fame.append(mixed_tree))

    return trees_hall_of_fame, perturbations_hall_of_fame

def migrate_ring(islands, k_top):
    num_islands = len(islands)
    for i in range(num_islands):
        # Migrate from the right neighbor
        right_neighbor = (i + 1) % num_islands
        islands[i]['decision_trees'] += islands[right_neighbor]['decision_trees'][:k_top]
        islands[i]['perturbations'] += islands[right_neighbor]['perturbations'][:k_top]

        # Migrate from the left neighbor
        left_neighbor = (i - 1) % num_islands
        islands[i]['decision_trees'] += islands[left_neighbor]['decision_trees'][:k_top]
        islands[i]['perturbations'] += islands[left_neighbor]['perturbations'][:k_top]


def construct_ensemble(islands):
    # Construct the final forest using Nash-based voting
    best_trees = [max(island['decision_trees'], key=lambda x: x.fitness) for island in islands]
    all_perturbations = [perturbation for island in islands for perturbation in island['perturbations']]

    # Compute Mixed Nash Equilibrium
    mixed_tree, mixed_perturbation = compute_mixed_nash_equilibrium(best_trees, all_perturbations)

    # The probabilities from the mixed equilibrium DT strategy are used for voting weights
    voting_weights = mixed_tree.probabilities

    # Make predictions using the ensemble with Nash-based voting
    def predict(X):
        predictions = np.array([tree.predict(X) for tree in best_trees])
        weighted_predictions = np.average(predictions, axis=0, weights=voting_weights)
        return np.argmax(weighted_predictions, axis=1)

    # The 'predict' function now uses the ensemble of best trees and their voting weights
    return predict  # Return the predict function that uses the ensemble


if __name__ == "__main__":

    EPS = 0.1  # Maximum perturbation value

    #data, labels = load_wine()
    #data, labels = fetch_openml(name="diabetes", as_frame=False, return_X_y=True)
    #data, labels = fetch_openml(name="wine", as_frame=False, return_X_y=True)

    dataset_name = "Fashion-MNIST"

    benchmark_data, benchmark_labels = fetch_openml(dataset_name, version=1, return_X_y=True)

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(benchmark_data, benchmark_labels, test_size=0.2, random_state=42)

    # Normalize the data to range [0, 1]
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    file =  open(dataset_name + '_' + datetime.today().strftime('%Y-%m-%d') + '.txt', 'a')

    for i in range(20):
        print("-- ITERATION " + str(i) + " -- ")
        # Run the coevolutionary algorithm
        start = time.time()
        best_decision_tree = islands_algorithm(train_data, train_labels)
        end = time.time()

        print("Best decision tree train value: ", best_decision_tree.fitness)

        test_perturbations = initialize_perturbations(100000, test_data)

        evaluate_fitness_decision_trees([best_decision_tree], test_perturbations, [], test_data, test_labels)

        print("Best decision tree test value: ", best_decision_tree.fitness)

        file.write(str(best_decision_tree.fitness) + '\t' + str(print(end - start)) + '\n');

    file.close()
    print("-- FINISHED --")