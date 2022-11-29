"""This script performs the simulation that validates the found necessary ks.
Generates multiple tree topologies and many mutation datasets per tree topology,
and reports the proportion of mutation datasets that satisfy the reconstruction
criterion for each tree."""

import argparse
from functools import partial

import utilities

# Set the number of mutation datasets per topology
num_mut_datasets = 100
# Set the number of topologies generated in the 'exponential_plus_c' simulation
num_trees_exponential_sim = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lamb", type=float, help="The mutation rate simulation parameter.")
    parser.add_argument("q", type=float, help="The collision parameter of the mutation state distribution")
    parser.add_argument("k", type=float, help="The k to validate.")
    parser.add_argument("depth", type=float, help="The depth of the tree, 2**(depth - 1) is the number of leaves.")
    parser.add_argument("tree_sim", type=str, help="The tree simulation function.")
    parser.add_argument("alg", type=str, help="The reconstruction algorithm function.")
    parser.add_argument("score_method", type=str, help="The reconstruction criterion function.")
    parser.add_argument("out_file", type=str, help="The filepath to write the results.")
    
    # Read in the args from parser and assign the correct data types
    args = parser.parse_args()
    lamb = args.lamb
    lamb = float(lamb)
    q = args.q
    depth = args.depth
    depth = int(depth)
    k = args.k
    k = int(k)
    tree_sim = args.tree_sim
    alg = args.alg
    score_method = args.score_method
    out_path = args.out_file

    # Load the function that simulates the tree from the string specifying it
    tree_sim_function = None
    if tree_sim == "complete_binary":
        tree_sim_function = utilities_review.complete_binary_topology_sim
    elif tree_sim == "exponential_plus_c":
        tree_sim_function = utilities_review.exponential_plus_c_topology_sim
    
    # Load the solver function from the string specifying it
    alg_function = None
    if alg == "percolation":
        alg_function = utilities.percolation_solve
    elif alg == "shared_mutation":
        alg_function = utilities.shared_mutation_solve
    
    # Load the reconstruction criterion function from the string specifying it
    score_method_function = None    
    if score_method == "triplets":
        score_method_function = utilities.triplets_score
    elif score_method == "rf":
        score_method_function = utilities.robinson_foulds_score
    elif score_method == "d_triplets":
        score_method_function = partial(utilities.depth_isomorphism, 0.5)
    elif score_method == "d_triplets_20":
        score_method_function = partial(utilities.depth_isomorphism, 0.2)

    # Open the output file
    f = open(out_path, "a")

    # Generate a uniform state distribution with probability q for each state,
    # with the the number of possible mutation states being 1/q
    num_states = int(1/q)
    q_dist = dict(zip(range(1, num_states + 1), [q] * num_states))

    # Set the number of trees to be used
    if tree_sim == "exponential_plus_c":
        num_trees = num_trees_exponential_sim
    elif tree_sim == "complete_binary":
        num_trees = 1

    # The main loop. For each tree topology, generate a number of mutation
    # datasets, and score the final proportion of datasets for each tree that
    # pass the reconstruction criterion
    for tree_ind in range(num_trees):
        successes = 0
        topology = tree_sim_function(depth)

        for _ in range(num_mut_datasets):
            topology_copy = topology.copy()
            # Simulate the tree using the specified parameters, reconstruct the
            # tree using the specified function, and score the reconstruction
            tree = utilities_review.overlay_mut_data(topology, num_chars, q_dist, lamb)
            recon_tree = alg_function(tree)
            if score_method_function(tree, recon_tree):
                successes += 1

        # For each topology, write out the proportion of mutation datasets that
        # succeed
        f.write(str(q) + "\t" + str(lamb) + "\t" + str(k) + "\t" + str(depth) + "\t" + str(tree_ind) + "\t" + str(successes/num_mut_datasets) + "\n")

    f.close()

if __name__=="__main__":
    main()