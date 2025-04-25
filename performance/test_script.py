import os
import time
import cffi
import sys

from connectomics_c import ffi, lib

print('Minimum Feedback Arc Set Combined Codebase Test Script')

# --- Configuration ---
# Adjust this path to where your graph.csv is located
DEFAULT_GRAPH_FILE = "./graph.csv"
# Use an environment variable or provide a default
GRAPH_FILE_PATH = os.environ.get("CONNECTOMICS_GRAPH_PATH", DEFAULT_GRAPH_FILE)

# Algorithm Parameters (Adjust as needed for testing)
OKUBO_EPOCHS = 3
OKUBO_LOG_INTERVAL = 10000 # Log approx every 10k nodes processed

BADER_THREADS = 4
BADER_ITERS_PER_THREAD = 1000000 # Keep relatively low for a quick test
BADER_UPDATES_PER_THREAD = 5
BADER_TMIN = 0.001
BADER_TMAX = 0.1
BADER_GOBACK_WINDOW = 500000
BADER_TOPOSHUFFLE_FREQ = 250000
BADER_VERBOSITY = 1 # 0=silent, 1=basic, 2=detailed, 10=debug

HASHORVA_INITIAL_TEMP = 10.0 # Lower temp for faster testing
HASHORVA_COOLING_RATE = 1e-7 # Faster cooling for testing
HASHORVA_MAX_ITERATIONS = 2000000 # Keep low for testing
HASHORVA_LOG_INTERVAL = 500000

LOG_PROGRESS = True
SEED = 42
# --- End Configuration ---

def main():
    """Runs the test script."""

    print("--- Connectomics C Library Test Script ---")
    print(f"Using graph file: {GRAPH_FILE_PATH}")
    if not os.path.exists(GRAPH_FILE_PATH):
        print(f"Error: Graph file not found at {GRAPH_FILE_PATH}")
        print("Please set the CONNECTOMICS_GRAPH_PATH environment variable or place graph.csv.")
        return

    connectome = ffi.NULL
    instance = ffi.NULL
    bader_best_storage = ffi.NULL

    try:
        # 1. Initialization
        print(f"\n[1] Initializing and Seeding RNG (Seed: {SEED})...")
        lib.seed_rng(SEED)
        print("RNG Seeded.")

        # 2. Load Connectome
        print(f"\n[2] Loading Connectome...")
        start_time = time.time()
        # Pass filename as bytes
        connectome = lib.load_connectome(GRAPH_FILE_PATH.encode('utf-8'))
        load_time = time.time() - start_time
        if connectome == ffi.NULL:
            print("Error: Failed to load connectome.")
            return
        print(f"Connectome loaded successfully in {load_time:.2f} seconds.")
        num_nodes = lib.get_connectome_num_nodes(connectome)
        print(f"    Unique Nodes: {num_nodes}")
        # Accessing struct fields directly (example)
        print(f"    Total Connections: {connectome.num_connections}")
        print(f"    Total Weight: {connectome.total_weight}")


        # 3. Create Initial Solution
        print("\n[3] Creating Random Initial Solution...")
        start_time = time.time()
        instance = lib.create_random_solution_instance(connectome)
        create_time = time.time() - start_time
        if instance == ffi.NULL:
            print("Error: Failed to create solution instance.")
            return
        print(f"Random solution instance created in {create_time:.2f} seconds.")
        initial_score = lib.get_solution_score(instance)
        print(f"    Initial Score: {initial_score}")
        if initial_score == 0:
            print("Warning: Initial score is 0. This might indicate an issue or a very sparse graph.")

        # 4. Prepare Bader's Best Solution Storage
        print("\n[4] Preparing Best Solution Storage for Bader...")
        bader_best_storage = lib.create_best_solution_storage()
        if bader_best_storage == ffi.NULL:
            print("Error: Failed to create best solution storage.")
            return
        lib.init_best_solution_storage(bader_best_storage, instance)
        print(f"    Initialized Bader Best Storage with score: {lib.get_best_solution_score(bader_best_storage)}")

        # --- Run Algorithms Sequentially ---
        # Note: Each algorithm starts from the state left by the previous one.
        # For independent tests, create a new instance before each run.

        # 5. Run Okubo Local Search
        print(f"\n[5] Running Okubo Local Search ({OKUBO_EPOCHS} epochs)...")
        start_time = time.time()
        lib.run_okubo_local_search(
            instance,
            connectome,
            OKUBO_EPOCHS,
            LOG_PROGRESS,
            OKUBO_LOG_INTERVAL
        )
        okubo_time = time.time() - start_time
        score_after_okubo = lib.get_solution_score(instance)
        print(f"Okubo LS finished in {okubo_time:.2f} seconds.")
        print(f"    Score after Okubo LS: {score_after_okubo}")

        # 6. Run Bader Parallel Annealing
        print(f"\n[6] Running Bader Parallel Annealing ({BADER_THREADS} threads)...")
        start_time = time.time()
        lib.run_simanneal_parallel_with_toposhuffle(
            instance,
            connectome,
            BADER_THREADS,
            BADER_ITERS_PER_THREAD,
            BADER_UPDATES_PER_THREAD,
            BADER_TMIN,
            BADER_TMAX,
            BADER_GOBACK_WINDOW,
            BADER_TOPOSHUFFLE_FREQ,
            BADER_VERBOSITY,
            bader_best_storage # Pass the storage object
        )
        bader_time = time.time() - start_time
        score_after_bader = lib.get_solution_score(instance)
        bader_best_score = lib.get_best_solution_score(bader_best_storage)
        print(f"Bader Annealing finished in {bader_time:.2f} seconds.")
        print(f"    Instance Score after Bader: {score_after_bader}")
        print(f"    Bader Best Storage Score:   {bader_best_score}")
        if score_after_bader != bader_best_score:
            print("    Note: Instance score and best storage score differ. Instance holds one thread's result, best storage holds overall best.")


        # 7. Run Hashorva Simulated Annealing
        print(f"\n[7] Running Hashorva Simulated Annealing...")
        start_time = time.time()
        lib.run_simanneal_parallel(
            instance,
            connectome,
            HASHORVA_INITIAL_TEMP,
            HASHORVA_COOLING_RATE,
            HASHORVA_MAX_ITERATIONS,
            HASHORVA_LOG_INTERVAL,
            LOG_PROGRESS
        )
        hashorva_time = time.time() - start_time
        score_after_hashorva = lib.get_solution_score(instance)
        print(f"Hashorva Annealing finished in {hashorva_time:.2f} seconds.")
        print(f"    Score after Hashorva: {score_after_hashorva}")
        print("    Note: Hashorva's internal best is not directly captured here.")


        # 8. Retrieve Final Solution (Example)
        print("\n[8] Retrieving Final Solution State (Example)...")
        final_score = lib.get_solution_score(instance)
        final_size = lib.get_solution_size(instance)
        solution_ptr = lib.get_solution_array_ptr(instance)

        if final_size > 0 and solution_ptr != ffi.NULL:
            # Convert C array pointer to Python list
            final_solution_dense = ffi.unpack(solution_ptr, final_size)
            print(f"    Final Score: {final_score}")
            print(f"    Solution Size: {final_size}")
            print(f"    First 10 dense indices: {final_solution_dense[:10]}")
            print(f"    Last 10 dense indices: {final_solution_dense[-10:]}")

            # Example: Get original IDs for the first 10
            id_map_ptr = lib.get_dense_idx_to_node_id_array_ptr(connectome)
            if id_map_ptr != ffi.NULL:
                 id_map = ffi.unpack(id_map_ptr, num_nodes) # num_nodes should == final_size
                 first_10_ids = [id_map[dense_idx] for dense_idx in final_solution_dense[:10]]
                 print(f"    First 10 original node IDs: {first_10_ids}")
            else:
                 print("    Could not retrieve original node ID map.")

        else:
            print("    Could not retrieve final solution array.")

        # Retrieve Bader's Best Solution
        print("\n    Retrieving Bader's Best Recorded Solution...")
        bader_best_score = lib.get_best_solution_score(bader_best_storage)
        bader_sol_size = bader_best_storage.solution_size # Access struct field directly
        bader_sol_ptr = lib.get_best_solution_array_ptr(bader_best_storage)
        if bader_sol_size > 0 and bader_sol_ptr != ffi.NULL:
             bader_best_dense = ffi.unpack(bader_sol_ptr, bader_sol_size)
             print(f"    Bader Best Score: {bader_best_score}")
             print(f"    Bader Best Size: {bader_sol_size}")
             print(f"    Bader Best First 10 dense: {bader_best_dense[:10]}")
        else:
             print("    Could not retrieve Bader's best solution array.")


    except Exception as e:
        print(f"\n--- An error occurred: {e} ---")
        import traceback
        traceback.print_exc()

    finally:
        # 9. Cleanup
        print("\n[9] Cleaning up C memory...")
        if instance != ffi.NULL:
            lib.free_solution_instance(instance)
            print("    Solution instance freed.")
        if bader_best_storage != ffi.NULL:
             lib.free_best_solution_storage(bader_best_storage)
             print("    Bader best storage freed.")
        if connectome != ffi.NULL:
            lib.free_connectome(connectome)
            print("    Connectome freed.")
        print("Cleanup complete.")

    print("\n--- Test Script Finished ---")

# if __name__ == "__main__":
#     main()