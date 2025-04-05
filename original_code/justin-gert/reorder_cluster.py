import jax
import jax.numpy as jnp
from jax import random, jit
import optax
import polars as pl
import numpy as np
import functions

# Enable 64-bit precision and debug options
jax.config.update("jax_enable_x64", True)

def run(run_idx):
    # Load the data
    df = pl.read_csv("./connectome_graph.csv")

    # Extract arrays with appropriate data types
    source_nodes = df[df.columns[0]].to_numpy().astype(np.int64)
    target_nodes = df[df.columns[1]].to_numpy().astype(np.int64)
    edge_weights = df[df.columns[2]].to_numpy().astype(np.float64)

    # Get unique node IDs and map to indices
    unique_nodes = np.unique(np.concatenate((source_nodes, target_nodes)))
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}
    source_indices = jnp.array([node_id_to_index[node_id] for node_id in source_nodes])
    target_indices = jnp.array([node_id_to_index[node_id] for node_id in target_nodes])
    edge_weights = jnp.array(edge_weights)
    max_edge_weight = jnp.max(edge_weights)

    # Normalize edge weights
    edge_weights = edge_weights / max_edge_weight

    num_nodes = len(unique_nodes)
    key = random.PRNGKey(int(run_idx)+220)
    # How to initialize positions - uniform or normal distributions seem reasonable
    positions = random.normal(key, shape=(num_nodes,))

    # Hyper parameter
    num_epochs = 20000

    exponential_decay_scheduler = optax.exponential_decay(init_value=.05, transition_steps=num_epochs,
                                                        decay_rate=0.1, transition_begin=int(num_epochs*0.5),
                                                        staircase=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=exponential_decay_scheduler)
    )
    best_positions = positions
    n_refresh = 1
    best_metric = 0

    @jax.jit
    def optimization_step(relu_weight, positions,beta, opt_state,edge_weights):
        params = (positions)#(positions, w)

        loss, grads = jax.value_and_grad(functions.objective_function, argnums=(1))(
            relu_weight, positions, beta, source_indices, target_indices, edge_weights
        )

        # Update positions and w
        updates, opt_state = optimizer.update(grads, opt_state, params)
        positions= optax.apply_updates(params, updates)
        return positions, opt_state, loss
    for itr in range(n_refresh):
    # Initialize optimizer state
        positions = best_positions
        opt_state = optimizer.init(positions)
        n_cycles = 5

        betas = (jnp.cos(np.linspace(0, 2 * n_cycles * np.pi, num_epochs)) + 1.1) / 2
        # Create a wrapper for the objective function to fix additional arguments

        for epoch in range(num_epochs):
            key, subkey = random.split(key)
            noise = jax.random.normal(subkey, (positions.shape[0], ))
            positions, opt_state, loss = optimization_step(
                0, positions, betas[epoch], opt_state,edge_weights
            )

            metric = functions.calculate_metric(
                positions, num_nodes, source_indices, target_indices, edge_weights
            )
            if metric > best_metric:
                best_metric = metric
                best_positions = positions
                print(f"New best metric: {best_metric:.2f}")
            if epoch % 2000 == 0:
                print(itr, f"Epoch {epoch}, Loss: {loss}, Best metric: {best_metric:.2f}, Beta: {betas[epoch]:.2f}")
    # Map back to original node IDs and save the ordering
    sorted_indices = jnp.argsort(best_positions)
    ordered_node_ids = [index_to_node_id[int(idx)] for idx in sorted_indices]

    # Save the ordering to a CSV file
    import pandas as pd
    ordered_nodes_df = pd.DataFrame({"Node ID": ordered_node_ids, "Order": jnp.arange(num_nodes)})
    ordered_nodes_df.to_csv(f"./checkpoints/ordered_nodes_{best_metric}_{run_idx}.csv", index=False)
    # Save weights
    #jnp.save(f'./checkpoints/weights_{best_metric}_{run_idx}.npy', w)
    jnp.save(f'./checkpoints/positions_{best_metric}_{run_idx}.npy', best_positions)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <run_idx>")
        sys.exit(1)

    run_idx = sys.argv[1]
    print("STARTING RUN", run_idx)
    run(run_idx)