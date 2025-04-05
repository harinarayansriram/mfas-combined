import jax
import jax.numpy as jnp
from jax import random, jit
import optax
import polars as pl
import numpy as np
import functions

# Enable 64-bit precision and debug options
jax.config.update("jax_enable_x64", True)

import pandas as pd

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <run_idx>")
        sys.exit(1)

    run_idx = sys.argv[1]
    print("STARTING RUN", run_idx)

    # Load the data
    df = pl.read_csv("./connectome_graph.csv")
    # Extract arrays
    source_nodes = df[df.columns[0]].to_numpy().astype(np.int64)
    target_nodes = df[df.columns[1]].to_numpy().astype(np.int64)
    edge_weights = df[df.columns[2]].to_numpy().astype(np.int64)

    # Get unique node IDs and map to indices
    unique_nodes = np.unique(np.concatenate((source_nodes, target_nodes)))
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}
    key = random.PRNGKey(17)
    num_nodes = len(unique_nodes)
    base_positions = random.uniform(key, shape=(num_nodes,))
    # Map node IDs to indices in edge lists
    source_indices = np.array([node_id_to_index[node_id] for node_id in source_nodes])
    target_indices = np.array([node_id_to_index[node_id] for node_id in target_nodes])

    # Convert to JAX arrays
    source_indices = jnp.array(source_indices)
    target_indices = jnp.array(target_indices)
    edge_weights = jnp.array(edge_weights)

    sorted_indices = jnp.argsort(base_positions)

    # Create a mapping from node index to order in the sequence
    node_order = jnp.zeros(num_nodes, dtype=int)
    node_order = node_order.at[sorted_indices].set(jnp.arange(num_nodes))

    node_order, best_metric = functions.monte_carlo_node_ordering(
        source_indices, target_indices, node_order, edge_weights, num_iterations=1e9
    )

    source_order = node_order[source_indices]
    target_order = node_order[target_indices]
    metric = functions.calculate_node_forward(source_order, target_order, edge_weights)
    ordered_node_ids = [index_to_node_id[int(idx)] for idx in node_order]

    ordered_nodes_df = pd.DataFrame(
        {"Node ID": ordered_node_ids, "Order": jnp.arange(node_order.shape[0])}
    )
    ordered_nodes_df.to_csv(f"./ordered_nodes_{metric}_brute.csv", index=False)