import numpy as np


def evaluate(df, sol):
    """Calculate the sum of feedforward weights for a given solution

    Args:
        df (pd.DataFrame): connectivity data
        sol (pd.Series): solution with node ID and order

    Returns:
        sum_ff (np.int64): sum of feedforward weights (score)
    """
    source_order = df["Source Node ID"].map(sol)
    target_order = df["Target Node ID"].map(sol)
    sum_ff = df["Edge Weight"][(target_order > source_order)].sum().item()
    return sum_ff


def perturb_sol(df, sol, perturbation):
    """Perturb the solution by randomly interchanging nodes

    Args:
        df (pd.DataFrame): connectivity data
        sol (pd.Series): solution with node ID and order
        perturbation (float): percentage to perturb the solution

    Returns:
        sol_new (pd.Series): new solution with perturbed order
    """
    sol_new = sol.copy()
    diff_node = 0

    while diff_node < perturbation * sol.shape[0]:
        rand_edge = df.iloc[np.random.choice(df.shape[0])]
        source_id = rand_edge["Source Node ID"]
        target_id = rand_edge["Target Node ID"]

        # interchange the order
        sol_new[source_id], sol_new[target_id] = sol_new[target_id], sol_new[source_id]
        diff_node = sum(sol != sol_new)

    # sorted by the order
    sol_new.sort_values(inplace=True)
    return sol_new


def extract_subgraph(source_node_ids, target_node_ids, node_id):
    """Extract a subgraph that contains all the nodes connected to a given node ID

    Args:
        source_node_ids (np.array): array of source node IDs
        target_node_ids (np.array): array of target node IDs
        node_id (int): node ID for which to extract the subgraph

    Returns:
        nodes_subset (np.array): array of node IDs in the subgraph
    """
    mask = (source_node_ids == node_id) | (target_node_ids == node_id)
    subgraph_source = source_node_ids[mask]
    subgraph_target = target_node_ids[mask]

    nodes_subset = np.unique((subgraph_source, subgraph_target))

    return nodes_subset


def delta_swap(pi, i, d_weights):
    """Calculate the change in score after swapping two nodes

    Args:
        pi (np.array): current order for all the nodes in the subgraph
        i (int): index of the node to be swapped
        d_weights (dict): dictionary of edge weights

    Returns:
        delta (int): change in the score
    """
    assert 0 <= i and i < len(pi) - 1, "i should be in the range of [0, len(pi) - 1)"

    delta = d_weights.get((pi[i + 1].item(), pi[i].item()), 0) - d_weights.get(
        (pi[i].item(), pi[i + 1].item()), 0
    )
    return delta


def swap(pi, i):
    """Swap the positions of two adjacent nodes in the current order

    Args:
        pi (np.array): current order for all the nodes in the subgraph
        i (int): index of the node to be swapped

    Returns:
        pi_ new (np.array): new order of nodes after the swap
    """
    assert 0 <= i and i < len(pi) - 1, "i should be in the range of [0, len(pi) - 1)"
    pi_new = pi.copy()
    pi_new[i], pi_new[i + 1] = pi_new[i + 1], pi_new[i]
    return pi_new


def swap_left(pi, i, d_weights):
    """Calculate the difference in the score for each insert location via sequential swap towards the left

    Args:
        pi (np.array): current order for all the nodes in the subgraph
        i (int): current position of the node id
        d_weights (dict): edge weights

    Returns:
        d_diff (dict): node ID (key) and the difference in the score at that location (value)
    """
    d_diff = dict()
    cumsum = 0

    for j in range(i - 1, -1, -1):
        swap_partner = pi[j]  # node ID of the swap partner
        cumsum += delta_swap(pi, j, d_weights)
        pi = swap(pi, j)
        d_diff[swap_partner] = cumsum

    return d_diff


def swap_right(pi, i, d_weights):
    """Calculate the difference in the score for each insert location via sequential swap towards the right

    Args:
        pi (np.array): current order for all the nodes in the subgraph
        i (int): current position of the node id
        d_weights (dict): edge weights

    Returns:
        d_diff (dict): node ID (key) and the difference in the score at that location (value)
    """
    d_diff = dict()
    cumsum = 0

    for j in range(i, len(pi) - 1, 1):
        swap_partner = pi[j + 1]  # node ID of the swap partner
        cumsum += delta_swap(pi, j, d_weights)
        pi = swap(pi, j)
        d_diff[swap_partner] = cumsum

    return d_diff


def create_delta(source_node_ids, target_node_ids, sol_array, d_weights, node_id):
    """Use the sequential swaps to evalute insertions (Schiavinotto & Stutzle, 2004)

    Args:
        source_node_ids (np.array): source node IDs
        target_node_ids (np.array): target node IDs
        sol_array (np.array): current solution, where the indices represent the order of nodes in the solution
        d_weights (dict): edge weights
        node_id (int): node ID to be inserted

    Returns:
        sol_subset_keep (np.array): subset of the solution that includes the node_id
        sol_subset_indices (np.array): indices of sol_subset_keep
        d_diff (dict): dictionary of score differences
    """
    nodes_subset = extract_subgraph(source_node_ids, target_node_ids, node_id)

    # identify the indices of the nodes in the solution
    subset_indices = np.nonzero(np.isin(sol_array, nodes_subset))[0]

    # get the ordered subset of nodes from the current solution
    sol_subset = sol_array[subset_indices]
    sol_subset_keep = sol_subset.copy()

    i = np.where(sol_subset == node_id)[0][0]
    d_left = swap_left(sol_subset, i, d_weights)
    d_right = swap_right(sol_subset, i, d_weights)
    d_diff = {**d_left, **d_right, node_id: 0}
    return sol_subset_keep, subset_indices, d_diff


def insert_location(d_diff, sol_subset_keep, sol_subset_indices):
    """Determine the optimal insertion location that maximizes score based on the provided differences

    Args:
        d_diff (dict): dictionary of differences in score
        sol_subset_keep (np.array): subset of the solution to consider
        sol_subset_indices (np.array): indices of the solution to that corresponds to the subset

    Returns:
        max_loc (int): the optimal insertion location for the node_id
        diff (int): the difference in score resulting from the insertion
    """
    insert_partner = max(d_diff, key=d_diff.get)  # node ID for insertion partner
    diff = d_diff[insert_partner]
    max_loc = sol_subset_indices[np.where(sol_subset_keep == insert_partner)[0][0]]
    return max_loc, diff


def insert(sol_array, i, j):
    """Perform the insertion (see p.5 of Schiavinotto & Stutzle (2004) for the definition of swap)

    Args:
        sol (np.array): current solution
        i (int): current location of the node to be moved
        j (int): new index for the node to be inserted
                 If i < j, the node is inserted after j.
                 If i > j, the node is inserted before j.

    Returns:
        sol_array_new (np.array): updated solution
    """
    assert 0 <= i and i < len(sol_array), "i should be in the range of [0, len(sol))"
    assert 0 <= j and j < len(sol_array), "j should be in the range of [0, len(sol))"
    order_min = min(i, j)
    order_max = max(i, j)

    unchanged_1_indices = np.arange(order_min)
    unchanged_2_indices = np.arange(start=(order_max + 1), stop=sol_array.shape[0])
    middle_indices = np.arange(order_min, (order_max + 1))

    roll_dir = np.sign(j - i)
    changed = np.roll(middle_indices, roll_dir)

    sol_new_indices = np.concatenate(
        [unchanged_1_indices, changed, unchanged_2_indices]
    )

    # sort the solution based on the new order
    sorted_indices = np.argsort(sol_new_indices)
    sol_array_new = sol_array[sorted_indices]
    return sol_array_new