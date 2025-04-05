import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import utils

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)


def run_LS_insertion(sol_array, source_node_ids, target_node_ids, 
                     d_weights, n_epochs, score, output_dir):
    """Run the main local search loop for a specified number of epochs

    Args:
        sol_array (np.array): current solution as an array of node IDs
        source_node_ids (np.array): source node IDs
        target_node_ids (np.array): target node IDs
        d_weights (dict): dictionary of edge weights
        n_epochs (int): number of epochs
        score (float): initial score
        output_dir (str): directory to save the output

    Returns:
        None
    """

    for epoch in range(n_epochs):
        sum_epoch = 0

        # shuffle node order before each epoch
        node_order = np.random.permutation(sol_array)

        for node_id in tqdm(node_order):

            # calculate the difference in the score for each insert location for a given node_id
            sol_subset_keep, sol_subset_indices, d_diff = utils.create_delta(
                source_node_ids, target_node_ids, sol_array, d_weights, node_id
            )
            max_loc, diff = utils.insert_location(
                d_diff, sol_subset_keep, sol_subset_indices
            )

            # perform the insertion when there is an increase in the score
            if diff > 0:
                sol_array = utils.insert(
                    sol_array,
                    i=sol_subset_indices[np.where(sol_subset_keep == node_id)[0][0]],
                    j=max_loc,
                )
                score += diff
                sum_epoch += diff

        # exit the loop when there is no improvement
        if sum_epoch == 0:
            break

        # save the result
        sol = pd.Series(data=np.arange(len(sol_array)), index=sol_array, name='Order')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sol.to_csv(os.path.join(output_dir, str(score) + ".csv"), header=True,
                   index_label="Node ID")
    return


def main(args):

    np.random.seed(args.seed)
    print(f"random seed: {args.seed}")
    print(f"random perturbation percentage: {args.perturbation}")

    df = pd.read_csv(os.path.join(args.data_dir, "connectome_graph.csv"))

    if args.init_sol == "benchmark":
        sol_df = pd.read_csv(os.path.join(args.data_dir, "benchmark.csv"))
    else:
        sol_df = pd.read_csv(os.path.join(args.data_dir, args.init_sol + ".csv"))

    sol = sol_df.set_index("Node ID")["Order"]  # Convert to pd.Series
    init_score = utils.evaluate(df, sol)
    print(f"initial score: {init_score}")

    # create a dictionary of edge weights for faster lookup
    d_weights = dict(
        zip(zip(df["Source Node ID"], df["Target Node ID"]), df["Edge Weight"])
    )

    # perturb the initial solution
    sol_new = utils.perturb_sol(df, sol, args.perturbation)
    score = utils.evaluate(df, sol_new)
    sol = sol_new
    print(f"perturbed score: {score}")

    # convert DataFrame columns to NumPy arrays
    source_node_ids = df["Source Node ID"].to_numpy()
    target_node_ids = df["Target Node ID"].to_numpy()
    sol_array = sol.index.to_numpy()

    # main optimization
    run_LS_insertion(sol_array, source_node_ids, target_node_ids, d_weights, args.n_epochs, score, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, 
                        default=os.path.join('..', 'data'), 
                        help='path to the data directory')
    parser.add_argument('--init_sol', '-i', type=str, default='benchmark', 
                        help='name of the initial solution file (without extension)')
    parser.add_argument('--n_epochs', '-n', type=int, default=10, 
                        help="number of epochs")
    parser.add_argument('--output_dir', '-o', type=str, 
                        default=os.path.join('..', 'output'), 
                        help='path to the output directory')
    parser.add_argument('--perturbation', '-p', type=float,default=0.01, 
                        help='percentage to perturb the solution')
    parser.add_argument('--seed', '-s', type=int,default=42, 
                        help='random seed')
    args  = parser.parse_args()
    main(args)