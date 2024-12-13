import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class Perturbation:
    def __init__(self):
        self.p = None
        self.size = None
        self.fitness = None
        self.cart_result = None

    def apply(self, data):
        perturbed_data = data + self.p
        return perturbed_data

    def set_fitness(self, fitness):
        self.fitness = fitness

    def initialize_perturbation(self, width, height, eps):
        self.p = np.random.uniform(low=-eps, high=eps, size=(width, height))


class MixedPerturbation:
    def __init__(self, perturbations, probabilities):
        self.perturbations = perturbations
        self.probabilities = probabilities

    def compute_fitness(self, tree, data, labels):
        result = 0
        for i in range(0, len(self.perturbations)):
            perturbation = self.perturbations[i]
            perturbed_data = perturbation.apply(data)
            perturbed_predictions = tree.predict(perturbed_data)
            perturbed_accuracy = accuracy_score(labels, perturbed_predictions)
            result = result + self.probabilities[i] * perturbed_accuracy