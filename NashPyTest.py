import nashpy as nash
import numpy as np

if __name__ == "__main__":
    A = np.random.random((100,100))
    matching_pennies = nash.Game(A)
    matching_pennies.lemke_howson(initial_dropped_label=0)
    equilibria = matching_pennies.lemke_howson_enumeration()
    for eq in equilibria:
        print(eq)