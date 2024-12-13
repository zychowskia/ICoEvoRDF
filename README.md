# Cultivating Archipelago of Forests: Evolving Robust Decision Trees through Island Coevolution

## Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

## AAAI 2025

## Abstract

Decision trees are widely used in machine learning due to their simplicity and interpretability, but they often lack robustness to adversarial attacks and data perturbations. The paper proposes a novel island-based coevolutionary algorithm (ICoEvoRDF) for constructing robust decision tree ensembles. The algorithm operates on multiple islands, each containing populations of decision trees and adversarial perturbations. The populations on each island evolve independently, with periodic migration of top-performing decision trees between islands. This approach fosters diversity and enhances the exploration of the solution space, leading to more robust and accurate decision tree ensembles. ICoEvoRDF utilizes a popular game theory concept of mixed Nash equilibrium for ensemble weighting, which further leads to improvement in results. ICoEvoRDF is evaluated on 20 benchmark datasets, demonstrating its superior performance compared to state-of-the-art methods in optimizing both adversarial accuracy and minimax regret. The flexibility of ICoEvoRDF allows for the integration of decision trees from various existing methods, providing a unified framework for combining diverse solutions. Our approach offers a promising direction for developing robust and interpretable machine learning models.

## Dependencies

The code requires the following Python libraries:

- numpy
- scikit-learn
- nashpy
- datetime
- genetic_tree


## Note
The script uses the Fashion-MNIST dataset by default. You can modify the dataset by uncommenting and using one of the other available datasets.

## Parameters

The algorithm includes several tunable parameters:

`TREES_POPULATION_SIZE`: Number of decision trees per island.

`PERTURBATIONS_POPULATION_SIZE`: Number of perturbations per island.

`TREES_MUTATION_RATE` and `TREES_CROSSOVER_RATE`: Rates for evolving decision trees.

`PERTURBATIONS_MUTATION_RATE` and `PERTURBATIONS_CROSSOVER_RATE`: Rates for evolving perturbations.

`GENERATIONS`: Total number of generations for the algorithm.

`GENERATIONS_NO_IMPROVEMENT_LIMIT`: Stop criterion for lack of improvement.

`GENERATIONS_BETWEEN_SWITCH`: Interval for switching strategies.

Optimization Metric: Supports `MAX_REGRET` or `PERTURBED_ACCURACY` as the target metric.
