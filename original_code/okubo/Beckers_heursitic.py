import numpy as np
import pandas as pd

# Read the connectivity data
df = pd.read_csv("../data/connectome_graph.csv")
df.columns = ["Source Node ID", "Target Node ID", "Edge Weight"]
df.head()

def calculate_scores(df):
    """
    Calculates the scores for each node in the graph.

    Args:
        df (pd.DataFrame): adjacency list of the graph. 
    
    Returns:
        scores (dict): scores for each node.
    """

    # Create a dict to store the scores for each node
    scores = {}

    # Calculate the weights of ingoing and outgoing for each node
    in_weights = df.groupby('Target Node ID')['Edge Weight'].sum().to_dict()
    out_weights = df.groupby('Source Node ID')['Edge Weight'].sum().to_dict()

    # Calculate the score for each node
    for node in set(df['Source Node ID']).union(set(df['Target Node ID'])):
        if node in out_weights and node in in_weights:
            scores[node] = out_weights[node] / in_weights[node]
        elif node in out_weights:
            scores[node] = out_weights[node]
        else:
            scores[node] = 0    # If the node is not in the graph, set the score to 0

    return scores

scores = calculate_scores(df)

for node in list(scores.keys())[:5]:
    print(f"{node}: {scores[node]:.2f}")

def run_Beckers(df):
    """
    Run Becker's heuristic to sort the nodes based on their scores.

    Parameters:
        df (pd.DataFrame): The dataframe adjacency matrix.

    Returns:
        df_result (pd.DataFrame): The dataframe of the result with the columns "Node ID" and "Order".
    """

    scores = calculate_scores(df)

    # Sort the nodes based on their scores
    sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Organize the result into a DataFrame
    df_result = pd.DataFrame(sorted_nodes, columns=['Node ID'])
    df_result["Order"] = range(0, len(df_result))
    
    return df_result

sol = run_Beckers(df)
sol.head()

'''
Node ID	Order
0	720575940623295415	0
1	720575940619358885	1
2	720575940609713710	2
3	720575940617415803	3
4	720575940621106977	4
'''

def evaluate(df, sol):
    """Evaluate the sum of feedforward weights and fraction of feedforward weights given a solution.

    Args:
        df (pd.DataFrame): adjacency list of the graph
        sol (pd.DataFrame): solution with Node ID and Order

    Returns:
        sum_ff (np.int64): sum of feedforward weights
        ratio (np.float): proportion of feedforward weights over sum of weights

    """
    source_order = df['Source Node ID'].map(sol.set_index('Node ID')['Order'])
    target_order = df['Target Node ID'].map(sol.set_index('Node ID')['Order'])
    sum_ff = df['Edge Weight'][(target_order > source_order)].sum()
    ratio = sum_ff / np.sum(df["Edge Weight"])

    return sum_ff.item(), ratio.item()


sum_ff, ratio = evaluate(df, sol)
print(f'sum of feedforward weights: {sum_ff}, ratio: {ratio:.2f}') #sum of feedforward weights: 30136267, ratio: 0.72


# Compare with the benchmark solution
sol_benchmark = pd.read_csv("../data/benchmark.csv")
sum_ff, ratio = evaluate(df, sol_benchmark)
print(f'sum of feedforward weights: {sum_ff}, ratio: {ratio:.2f}') #sum of feedforward weights: 29023882, ratio: 0.69