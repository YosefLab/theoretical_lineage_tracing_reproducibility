"""This script performs the simulation that finds the empirical minimum 
necessary k. Takes in simulation parameters and performs a binary search."""

import argparse
from functools import partial

import utilities

# Set the max for the range of ks to be searched over 
max_k = 4096
# Set the number of trees in the binary search
num_trees = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lamb", type=float, help="The mutation rate simulation parameter.")
    parser.add_argument("num_states", type=float, help="The number of states in the state distribution.")
    parser.add_argument("depth", type=float, help="The depth of the tree, 2**(depth - 1) is the number of leaves.")
    parser.add_argument("tree_sim", type=str, help="The tree simulation function.")
    parser.add_argument("alg", type=str, help="The reconstruction algorithm function.")
    parser.add_argument("score_method", type=str, help="The reconstruction criterion function.")
    parser.add_argument("out_file", type=str, help="The filepath to write the results.")

    # Read in the args from parser and assign the correct data types
    args = parser.parse_args()
    lamb = args.lamb
    lamb = float(lamb)
    num_states = args.num_states
    num_states = int(num_states)
    depth = args.depth
    depth = int(depth)
    tree_sim = args.tree_sim
    alg = args.alg
    score_method = args.score_method
    out_path = args.out_file

    # Load the function that simulates the tree from the string specifying it
    tree_sim_function = None
    if tree_sim == "complete_binary":
        tree_sim_function = utilities.complete_binary_tree_sim
    elif tree_sim == "exponential_plus_c":
        tree_sim_function = utilities.exponential_plus_c_tree_sim
    elif tree_sim == "missing":
        tree_sim_function = utilities.complete_binary_missing_tree_sim

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

    # Generate a uniform state distribution from the number of possible 
    # mutation states, with q = 1/(unumber of states)
    q_dist = dict(zip(range(1, num_states + 1), [1/num_states] * num_states))

    # Initialize the candidate k at 2 before performing binary search
    k_cand = 2

    # Initial loop that iterates by powers of 2 until 90% of trees pass the
    # reconstruction criterion to find an upper bound on k. Terminates when the
    # max k is reached
    success = False
    while not success and k_cand <= max_k:
        num_chars = k_cand
        rf_successes = []
        for _ in range(num_trees):
            # Simulate the tree using the specified parameters, reconstruct the
            # tree using the specified function, and score the reconstruction
            tree = tree_sim_function(num_chars, q_dist, lamb, depth)
            recon_tree = alg_function(tree)
            if score_method_function(tree, recon_tree):
                rf_successes.append(1)
            else:
                rf_successes.append(0)

        if sum(rf_successes)/num_trees >= 0.9:
            success = True

        # For each k, write out the proportion of trees that succeed
        f.write(str(float(num_states)) + "\t" + str(lamb) + "\t" + str(k_cand) + "\t" + str(depth) + "\t" + str(sum(rf_successes)/len(rf_successes)) + "\n")

        if not success:
            k_cand = int(k_cand * 2)

    # Loop to search within the bin [upper bound / 2, upper bound] for the 
    # upper bound found in the previous loop to find the lowest k that 
    # passes the reconstruction criterion. Skips if no k passes the 
    # reconstruction criterion in the previous loop
    if success:
        success = False

        lower_k_bin = lower_k_bin_orig = k_cand // 2
        upper_k_bin = upper_k_bin_orig = k_cand
    
        while lower_k_bin < upper_k_bin and upper_k_bin - lower_k_bin > 1:
            curr_k = (lower_k_bin + upper_k_bin)//2
            if curr_k == upper_k_bin_orig:
                break
            if curr_k == lower_k_bin_orig:
                break
            success = False
            num_chars = curr_k
            rf_successes = []

            for num in range(num_trees):
                # Simulate the tree using the specified parameters, reconstruct the
                # tree using the specified function, and score the reconstruction
                tree = tree_sim_function(num_chars, q_dist, lamb, depth)
                recon_tree = alg_function(tree)
                if score_method_function(tree, recon_tree):
                    rf_successes.append(1)
                else:
                    rf_successes.append(0)
            
            if sum(rf_successes)/num_trees >= 0.9:
                success = True
            
            # For each k, write out the proportion of trees that succeed
            f.write(str(float(num_states)) + "\t" + str(lamb) + "\t" + str(curr_k) + "\t" + str(depth) + "\t" + str(sum(rf_successes)/len(rf_successes)) + "\n")
            
            if success:
                upper_k_bin = curr_k
            else:
                lower_k_bin = curr_k

    f.close()

if __name__=="__main__":
    main()