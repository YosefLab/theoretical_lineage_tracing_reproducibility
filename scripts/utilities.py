"""This file stores the utilities used in the simulations of the Threshold and 
Bottom-Up Algorithms. Requires the Cassiopeia software package: 
https://github.com/YosefLab/Cassiopeia."""

from typing import Dict, Tuple

import networkx as nx
import pandas as pd
import numpy as np
import pickle as pic
import copy
import random
import itertools

import cassiopeia.critique as critique
import cassiopeia.data as data
import cassiopeia.simulator as simulator
from cassiopeia.simulator.TreeSimulator import TreeSimulatorError
from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver
from cassiopeia.solver.PercolationSolver import PercolationSolver
from cassiopeia.solver.SharedMutationJoiningSolver import SharedMutationJoiningSolver
from cassiopeia.solver import dissimilarity_functions

def complete_binary_tree_sim(k_cand: int, q_dist: Dict[int, float], lamb: float, depth: int) -> data.CassiopeiaTree:
    """Simulates a lineage tracing experiment with a complete binary topology.

    Args:
        k_cand: The number of characters
        q_dist: The state distribution
        lamb: The value of lambda
        depth: The depth of the tree (including the unary implicit root edge)

    Returns:
        A CassiopeiaTree containing the lineage tracing experiment
    """

    # Creating the binary networkx tree
    tree = nx.balanced_tree(2, (depth - 1), create_using = nx.DiGraph)
    tree = nx.relabel_nodes(tree, dict(zip(list(tree.nodes), [i + 1 for i in tree.nodes])))

    # Appending the implicit root edge
    tree.add_edge(0, 1)

    # Storing the networkx tree in a CassiopeiaTree container object
    complete_binary = data.CassiopeiaTree(tree = tree)

    # Relabling the leaves
    leaves = complete_binary.leaves
    complete_binary.relabel_nodes(dict(zip(leaves, ['c' + str(i) for i in leaves])))
    
    # Normalizing the times and edge lengths by the total time of the tree
    time_dict = {}
    for i in complete_binary.nodes:
        time_dict[i] = complete_binary.get_time(i)/depth
    complete_binary.set_times(time_dict)
    
    # Generating the lineage tracing data with the supplied parameters and
    # with no missing data
    lt_sim = simulator.Cas9LineageTracingDataSimulator(
        number_of_cassettes = k_cand,
        size_of_cassette = 1,
        mutation_rate = lamb,
        state_priors = q_dist,
        heritable_silencing_rate = 0,
        stochastic_silencing_rate = 0,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
    )
    lt_sim.overlay_data(complete_binary)

    return complete_binary

def exponential_plus_c_tree_sim(k_cand: int, q_dist: Dict[int, float], lamb: float, depth: int) -> data.CassiopeiaTree:
    """Simulates a lineage tracing experiment with an asynchronous topology.

    Args:
        k_cand: The number of characters
        q_dist: The state distribution
        lamb: The value of lambda
        depth: The depth of the tree (including the unary implicit root edge)

    Returns:
        A CassiopeiaTree containing the lineage tracing experiment
    """
    # We define the number of leaves wanted as 2**(depth - 1)
    num_cells = 2**(depth - 1)
    size = 0
    c = 0.05

    # The birth rate and death rate are hard-coded such that they produce
    # trees with an average of 256 leaves
    birth_rate = 23.697339506322916 
    death_rate = 2.3697339506322916

    # The waiting distributions for our Bellman-Harris process with death
    initial_birth_scale = 1/birth_rate
    birth_waiting_distribution = lambda scale: np.random.exponential(scale) + c
    death_waiting_distribution = lambda: np.random.exponential(1/death_rate + c)

    # We generate trees until the number of leaves is between a factor of 0.8
    # and 1.2 of our wanted number of leaves
    while size < num_cells * 0.8 or size > num_cells * 1.2:
        try:
            bd_sim = simulator.BirthDeathFitnessSimulator(
                birth_waiting_distribution = birth_waiting_distribution,
                initial_birth_scale = initial_birth_scale,
                death_waiting_distribution = death_waiting_distribution,
                experiment_time = 1,
            )
            topology = bd_sim.simulate_tree()
            size = topology.n_cell
        except TreeSimulatorError:
            size = 0

    # Relabeling the leaves
    topology.relabel_nodes(dict(zip(topology.leaves, ["c" + i for i in topology.leaves])))

    # Generating the lineage tracing data with the supplied parameters and
    # with no missing data
    lt_sim = simulator.Cas9LineageTracingDataSimulator(
        number_of_cassettes = k_cand,
        size_of_cassette = 1,
        mutation_rate = lamb,
        state_priors = q_dist,
        heritable_silencing_rate = 0,
        stochastic_silencing_rate = 0,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
    )
    lt_sim.overlay_data(topology)

    return topology

def complete_binary_missing_tree_sim(k_cand: int, q_dist: Dict[int, float], lamb: float, depth: int) -> data.CassiopeiaTree:
    """Simulates a lineage tracing experiment with a complete binary topology
    and missing data.

    Args:
        k_cand: The number of characters
        q_dist: The state distribution
        lamb: The value of lambda
        depth: The depth of the tree (including the unary implicit root edge)

    Returns:
        A CassiopeiaTree containing the lineage tracing experiment
    """

    # Creating the binary networkx tree
    tree = nx.balanced_tree(2, (depth - 1), create_using = nx.DiGraph)
    tree = nx.relabel_nodes(tree, dict(zip(list(tree.nodes), [i + 1 for i in tree.nodes])))

    # Appending the implicit root edge
    tree.add_edge(0, 1)

    # Storing the networkx tree in a CassiopeiaTree container object
    complete_binary = data.CassiopeiaTree(tree = tree)

    # Relabling the leaves
    leaves = complete_binary.leaves
    complete_binary.relabel_nodes(dict(zip(leaves, ['c' + str(i) for i in leaves])))

    # Normalizing the times and edge lengths by the total time of the tree
    time_dict = {}
    for i in complete_binary.nodes:
        time_dict[i] = complete_binary.get_time(i)/depth
    complete_binary.set_times(time_dict)
    
    # Generating the lineage tracing data with the supplied parameters and
    # with 10% stochastic missing data
    lt_sim = simulator.Cas9LineageTracingDataSimulator(
        number_of_cassettes = k_cand,
        size_of_cassette = 1,
        mutation_rate = lamb,
        state_priors = q_dist,
        heritable_silencing_rate = 0,
        stochastic_silencing_rate = 0.1,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
    )
    lt_sim.overlay_data(complete_binary)

    return complete_binary

def complete_binary_missing_tree_sim(k_cand: int, q_dist: Dict[int, float], lamb: float, depth: int) -> data.CassiopeiaTree:
    """Simulates a lineage tracing experiment with a complete binary topology
    and missing data.

    Args:
        k_cand: The number of characters
        q_dist: The state distribution
        lamb: The value of lambda
        depth: The depth of the tree (including the unary implicit root edge)

    Returns:
        A CassiopeiaTree containing the lineage tracing experiment
    """

    # Creating the binary networkx tree
    tree = nx.balanced_tree(2, (depth - 1), create_using = nx.DiGraph)
    tree = nx.relabel_nodes(tree, dict(zip(list(tree.nodes), [i + 1 for i in tree.nodes])))

    # Appending the implicit root edge
    tree.add_edge(0, 1)

    # Storing the networkx tree in a CassiopeiaTree container object
    complete_binary = data.CassiopeiaTree(tree = tree)

    # Relabling the leaves
    leaves = complete_binary.leaves
    complete_binary.relabel_nodes(dict(zip(leaves, ['c' + str(i) for i in leaves])))

    # Normalizing the times and edge lengths by the total time of the tree
    time_dict = {}
    for i in complete_binary.nodes:
        time_dict[i] = complete_binary.get_time(i)/depth
    complete_binary.set_times(time_dict)
    
    # Generating the lineage tracing data with the supplied parameters and
    # with 10% stochastic missing data
    lt_sim = simulator.Cas9LineageTracingDataSimulator(
        number_of_cassettes = k_cand,
        size_of_cassette = 1,
        mutation_rate = lamb,
        state_priors = q_dist,
        heritable_silencing_rate = 0,
        stochastic_silencing_rate = 0.1,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
    )
    lt_sim.overlay_data(complete_binary)

    return complete_binary

def percolation_solve(tree: data.CassiopeiaTree) -> data.CassiopeiaTree:
    """Solves a tree using the Threshold Algorithm from a character matrix.

    Args:
        tree: A CassiopeiaTree object without the `tree` instance populated

    Returns:
        A CassiopeiaTree object containing the solved tree topology
    """

    # Extract the character matrix from the CassiopeiaTree container object
    cm = tree.character_matrix
    cm = cm.astype(int)

    # Instantiate a new CassiopeiaTree with no tree initialized
    recon_tree = data.CassiopeiaTree(character_matrix = cm, missing_state_indicator = -1)

    # Instantiate a Threshold Algorithm solver object and solve
    psolver = PercolationSolver(joining_solver = VanillaGreedySolver())
    psolver.solve(recon_tree)

    return recon_tree

def shared_mutation_solve(tree: data.CassiopeiaTree) -> data.CassiopeiaTree:
    """Solves a tree using the Threshold Algorithm from a character matrix.

    Args:
        tree: A CassiopeiaTree object without the `tree` instance populated

    Returns:
        A CassiopeiaTree object containing the solved tree topology
    """
    # Extract the character matrix from the CassiopeiaTree container object
    cm = tree.character_matrix
    cm = cm.astype(int)

    # Instantiate a new CassiopeiaTree with no tree initialized
    recon_tree = data.CassiopeiaTree(character_matrix = cm, missing_state_indicator = -1)

    # Instantiate a Bottom-Up Algorithm solver object and solve
    smjsolver = SharedMutationJoiningSolver(similarity_function = dissimilarity_functions.hamming_similarity_without_missing)
    smjsolver.solve(recon_tree)

    return recon_tree

def robinson_foulds_score(ground_tree: data.CassiopeiaTree, recon_tree: data.CassiopeiaTree) -> bool:
    """Determines if the Robinson-Foulds Distance of two tree topologies is 0.

    Args:
        ground_tree: The ground truth tree topology
        recon_tree: The reconstructed tree topology

    Returns:
        If the two topologies have a Robinson-Foulds Distance of 0
    """
    rf, rf_max = critique.compare.robinson_foulds(ground_tree, recon_tree)

    return rf == 0

def find_triplet_structure(triplet: Tuple[str, str, str], tree: nx.DiGraph) -> str:
    """Finds the ingroup of a triplet on a networkx tree topology.

    Args:
        triplet: A tuple containing the three leaf nodes forming the triplet
        tree: The tree topology, stored as a networkx object

    Returns:
        A string determining the ingroup of the triplet
    """
    # Extract the nodes from the tuple
    a, b, c = triplet[0], triplet[1], triplet[2]

    # Find the list of ancestors on the tree for each node
    a_ancestors = [node for node in nx.ancestors(tree, a)]
    b_ancestors = [node for node in nx.ancestors(tree, b)]
    c_ancestors = [node for node in nx.ancestors(tree, c)]

    # Calculate the number of ancestors shared between each leaf node
    ab_common = len(set(a_ancestors) & set(b_ancestors))
    ac_common = len(set(a_ancestors) & set(c_ancestors))
    bc_common = len(set(b_ancestors) & set(c_ancestors))

    # The nodes that share the most number of ancestors are the ingroup,
    # if it is a tie, there is no ingroup
    structure = "-"
    if ab_common > bc_common and ab_common > ac_common:
        structure = "ab"
    elif ac_common > bc_common and ac_common > ab_common:
        structure = "ac"
    elif bc_common > ab_common and bc_common > ac_common:
        structure = "bc"

    return structure

def calculate_triplets_correct(ground_tree: data.CassiopeiaTree, recon_tree: data.CassiopeiaTree, sample: int = 1000) -> Tuple[int, int]:
    """Samples triplets and returns the number that are correctly resolved.

    Args:
        ground_tree: The ground truth tree topology
        recon_tree: The reconstructed tree topology
        sample: The number of triplets to sample

    Returns:
        A tuple containing the number of correct triplets and the number sampled
    """

    # Track the number of correct triplets
    num_correct = 0

    # Extract tree topology as networkx objects
    recon = recon_tree.get_tree_topology()
    ground = ground_tree.get_tree_topology()

    leaves = ground.leaves

    for _ in range(sample):
        # Sample a triplet from the leaves of the ground truth
        sampled_triplet = np.random.choice(leaves, 3, replace = False)
        # Find the ingroup for both tree topologies
        recon_triplet = find_triplet_structure(sampled_triplet, recon)
        ground_triplet = find_triplet_structure(sampled_triplet, ground)
        # If the ingroup is the same, add one to the number of correct triplets
        num_correct += int(recon_triplet == ground_triplet)

    return num_correct, sample

def triplets_score(ground_tree: data.CassiopeiaTree, recon_tree: data.CassiopeiaTree) -> bool:
    """Determines if the proportion of triplets correctly resolved is >= 95%.

    Args:
        ground_tree: The ground truth tree topology
        recon_tree: The reconstructed tree topology

    Returns:
        If the topologies have the same ingroup on >= 95% of sampled triplets
    """
    correct_triplets, total_triplets = calculate_triplets_correct(ground_tree, recon_tree)

    return correct_triplets/total_triplets >= 0.95
    
def depth_isomorphism(d: float, ground_tree: data.CassiopeiaTree, recon_tree: data.CassiopeiaTree) -> bool:
    """Determines if two trees have the same splits up to depth d.

    Determining if two trees have the same splits up to depth d (are depth 
    isomorphic at d) is the same as saying that all triplets whose LCAs
    occur before depth d will be resolved correctly. The reconstruction 
    criterion is the same as setting \ell^* equal to the minimum edge length
    and d^* to d, as are done in the simulations varying d^*.

    Args:
        d: The depth to which to check if the splits are the same for two trees
        ground_tree: The ground truth tree topology
        recon_tree: The reconstructed tree topology

    Returns:
        If the topologies have the same splits up to depth d
    """

    tree_g = copy.copy(ground_tree)
    tree_r = copy.copy(recon_tree)

    tree_g.collapse_unifurcations()
    tree_r.collapse_unifurcations()
    
    def check_split(node_g, tree_g, node_r, tree_r, d, flag):
        groups_g = {}
        for i in tree_g.children(node_g):
            groups_g[i] = set(tree_g.leaves_in_subtree(i))
        groups_r = {}
        for i in tree_r.children(node_r):
            groups_r[i] = set(tree_r.leaves_in_subtree(i))
        node_pairs = []
        for i in groups_g:
            for j in groups_r:
                if groups_g[i] == groups_r[j]:
                    node_pairs.append((i, j))
        if len(node_pairs) == 0:
            flag[0] = False
        for i in node_pairs:
            if tree_g.get_attribute(i[0], "time") <= d and flag[0] and len(tree_g.leaves_in_subtree(i[0])) > 1:
                check_split(i[0], tree_g, i[1], tree_r, d, flag)
    
    correct_flag = [True]            
    check_split(tree_g.root, tree_g, tree_r.root, tree_r, d, correct_flag)

    return correct_flag[0]

def complete_binary_topology_sim(depth: int) -> data.CassiopeiaTree:
    """Generates a complete binary topology.

    Used in the validation of k experiments where multiple mutation datasets
    are generated for each topology.
    
    Args:
        depth: The depth of the tree (including the unary implicit root edge)

    Returns:
        A complete binary topology
    """

    # Creating the binary networkx tree   
    topology = nx.balanced_tree(2, (depth - 1), create_using = nx.DiGraph)
    topology = nx.relabel_nodes(topology, dict(zip(list(topology.nodes), [i + 1 for i in topology.nodes])))

    # Appending the implicit root edge
    topology.add_edge(0, 1)

    # Storing the networkx tree in a CassiopeiaTree container object
    complete_binary = data.CassiopeiaTree(tree = topology)

    # Relabling the leaves
    leaves = complete_binary.leaves
    complete_binary.relabel_nodes(dict(zip(leaves, ['c' + str(i) for i in leaves])))

    # Normalizing the times and edge lengths by the total time of the tree
    time_dict = {}
    for i in complete_binary.nodes:
        time_dict[i] = complete_binary.get_time(i)/depth
    complete_binary.set_times(time_dict)

    return complete_binary

def exponential_plus_c_topology_sim(depth: int) -> data.CassiopeiaTree:
    """Generates an asynchronous topology.

    Used in the validation of k experiments where multiple mutation datasets
    are generated for each topology.

    Args:
        depth: The depth of the tree (including the unary implicit root edge)

    Returns:
        An asynchronous topology
    
    """
    # We define the number of leaves wanted as 2**(depth - 1)
    num_cells = 2**(depth - 1)
    size = 0
    c = 0.05

    # The birth rate and death rate are hard-coded such that they produce
    # trees with an average of 256 leaves
    birth_rate = 23.697339506322916 
    death_rate = 2.3697339506322916

    # The waiting distributions for our Bellman-Harris process with death
    initial_birth_scale = 1/birth_rate
    birth_waiting_distribution = lambda scale: np.random.exponential(scale) + c
    death_waiting_distribution = lambda: np.random.exponential(1/death_rate + c)

    # We generate trees until the number of leaves is between a factor of 0.8
    # and 1.2 of our wanted number of leaves
    while size < num_cells * 0.8 or size > num_cells * 1.2:
        try:
            bd_sim = simulator.BirthDeathFitnessSimulator(
                birth_waiting_distribution = birth_waiting_distribution,
                initial_birth_scale = initial_birth_scale,
                death_waiting_distribution = death_waiting_distribution,
                experiment_time = 1,
            )
            topology = bd_sim.simulate_tree()
            size = topology.n_cell
        except TreeSimulatorError:
            size = 0

    # Relabeling the leaves
    topology.relabel_nodes(dict(zip(topology.leaves, ["c" + i for i in topology.leaves])))
    
    return topology

def overlay_mut_data(topology: data.CassiopeiaTree, k_cand: int, q_dist: Dict[int, float], lamb: float) -> data.CassiopeiaTree:
    """Overlays lineage tracing mutations over a provided topology.

    Used in the validation of k experiments where multiple mutation datasets
    are generated for each topology.

    Args:
        topology: The CassiopeiaTree object containing the tree topology
        k_cand: The number of characters
        q_dist: The state distribution
        lamb: The value of lambda

    Returns:
        A CassiopeiaTree object containing the topology as well as the
        lineage tracing mutations
    """

    # Generating the lineage tracing data with the supplied parameters and
    # with no missing data
    lt_sim = simulator.Cas9LineageTracingDataSimulator(
        number_of_cassettes = k_cand,
        size_of_cassette = 1,
        mutation_rate = lamb,
        state_priors = q_dist,
        heritable_silencing_rate = 0,
        stochastic_silencing_rate = 0,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
    )
    lt_sim.overlay_data(topology)

    return topology