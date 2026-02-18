import os
import time
import cffi

from connectomics_c import ffi, lib

DEFAULT_GRAPH_FILE = "../data/connectome_graph.csv"
GRAPH_FILE_PATH = os.environ.get("CONNECTOMICS_GRAPH_PATH", DEFAULT_GRAPH_FILE)

# Algorithm Parameters
OKUBO_EPOCHS = 3
OKUBO_LOG_INTERVAL = 10000 

BADER_THREADS = 4
BADER_ITERS_PER_THREAD = 5000000 
BADER_UPDATES_PER_THREAD = 5
BADER_TMIN = 0.001
BADER_TMAX = 0.1
BADER_GOBACK_WINDOW = 500000
BADER_TOPOSHUFFLE_FREQ = 250000
BADER_VERBOSITY = 1 # 0=silent, 1=basic, 2=detailed, 10=debug

HASHORVA_INITIAL_TEMP = 10.0 
HASHORVA_COOLING_RATE = 1e-7 
HASHORVA_MAX_ITERATIONS = 20000000 
HASHORVA_LOG_INTERVAL = 500000

LOG_PROGRESS = True
SEED = 42

def main():
    print("Performance version of MFAS connectomics algorithms test")

    print(f"Using graph file: {GRAPH_FILE_PATH}")
    if not os.path.exists(GRAPH_FILE_PATH):
        print(f"Error: Graph file not found at {GRAPH_FILE_PATH}")
        print("Please set the CONNECTOMICS_GRAPH_PATH environment variable or place graph.csv")
        return

    connectome = ffi.NULL
    instance = ffi.NULL
    bader_best_storage = ffi.NULL

    try:
        print(f"\nInitializing and Seeding RNG (Seed: {SEED})")
        lib.seed_rng(SEED)
        print("RNG Seeded")

        print("Loading Connectome")
        start_time = time.perf_counter()
        connectome = lib.load_connectome(GRAPH_FILE_PATH.encode('utf-8'))
        load_time = time.perf_counter() - start_time
        if connectome == ffi.NULL:
            print("Error: Failed to load connectome")
            return
        print(f"Connectome loaded successfully in {load_time:.2f} seconds")
        num_nodes = lib.get_connectome_num_nodes(connectome)
        print(f"\tUnique Nodes: {num_nodes}")

        #Create Initial Solution
        print("\nCreating Random Initial Solution")
        start_time = time.perf_counter()
        instance = lib.create_random_solution_instance(connectome)
        create_time = time.perf_counter() - start_time
        if instance == ffi.NULL:
            print("Error: Failed to create solution instance")
            return
        print(f"Random solution instance created in {create_time:.2f} seconds")
        initial_score = lib.get_solution_score(instance)
        print(f"\tInitial Score: {initial_score}")
        if initial_score == 0:
            print("Warning: Initial score is 0. This might indicate an issue or a very sparse graph")

        #Prepare Bader's Best Solution Storage
        print("\nPreparing Best Solution Storage for Bader")
        bader_best_storage = lib.create_best_solution_storage()
        if bader_best_storage == ffi.NULL:
            print("Error: Failed to create best solution storage")
            return
        lib.init_best_solution_storage(bader_best_storage, instance)
        print(f"\tInitialized Bader Best Storage with score: {lib.get_best_solution_score(bader_best_storage)}")

        # Run Algorithms Sequentially
        # Note: Each algorithm starts from the state left by the previous one.
        # For independent tests, create a new instance before each run.

        #Run Okubo Local Search
        print(f"\nRunning Okubo Local Search ({OKUBO_EPOCHS} epochs)")
        start_time = time.perf_counter()
        lib.run_okubo_local_search(
            instance,
            connectome,
            OKUBO_EPOCHS,
            LOG_PROGRESS,
            OKUBO_LOG_INTERVAL
        )
        okubo_time = time.perf_counter() - start_time
        score_after_okubo = lib.get_solution_score(instance)
        print(f"Okubo LS finished in {okubo_time:.2f} seconds")
        print(f"\tScore after Okubo LS: {score_after_okubo}")
        print(f"Actual score is: {lib.calculate_forward_score(instance, connectome)}")

        #Run Bader Parallel Annealing
        print(f"\nRunning Bader Parallel Annealing ({BADER_THREADS} threads)")
        start_time = time.perf_counter()
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
        bader_time = time.perf_counter() - start_time
        score_after_bader = lib.get_solution_score(instance)
        bader_best_score = lib.get_best_solution_score(bader_best_storage)
        print(f"Bader Annealing finished in {bader_time:.2f} seconds")
        print(f"\tInstance Score after Bader: {score_after_bader}")
        print(f"Actual score is: {lib.calculate_forward_score(instance, connectome)}")

        print(f"\tBader Best Storage Score:   {bader_best_score}")
        if score_after_bader != bader_best_score:
            print("\tNote: Instance score and best storage score differ. Instance holds one thread's result, best storage holds overall best")


        #Run Hashorva Simulated Annealing
        print("\nRunning Hashorva Simulated Annealing")
        start_time = time.perf_counter()
        lib.run_simanneal_parallel(
            instance,
            connectome,
            HASHORVA_INITIAL_TEMP,
            HASHORVA_COOLING_RATE,
            HASHORVA_MAX_ITERATIONS,
            HASHORVA_LOG_INTERVAL,
            LOG_PROGRESS
        )
        hashorva_time = time.perf_counter() - start_time
        score_after_hashorva = lib.get_solution_score(instance)
        print(f"Hashorva Annealing finished in {hashorva_time:.2f} seconds")
        print(f"\tScore after Hashorva: {score_after_hashorva}")
        print("\tNote: Hashorva's internal best is not directly captured here")


        #Retrieve Final Solution (Example)
        print("\nRetrieving Final Solution State (Example)")
        final_score = lib.get_solution_score(instance)
        final_size = lib.get_solution_size(instance)
        solution_ptr = lib.get_solution_array_ptr(instance)

        if final_size > 0 and solution_ptr != ffi.NULL:
            # Convert C array pointer to Python list
            final_solution_dense = ffi.unpack(solution_ptr, final_size)
            print(f"\tFinal Score: {final_score}")
            print(f"Actual score is: {lib.calculate_forward_score(instance, connectome)}")

            print(f"\tSolution Size: {final_size}")
            print(f"\tFirst 10 dense indices: {final_solution_dense[:10]}")
            print(f"\tLast 10 dense indices: {final_solution_dense[-10:]}")

            # Example: Get original IDs for the first 10
            id_map_ptr = lib.get_dense_idx_to_node_id_array_ptr(connectome)
            if id_map_ptr != ffi.NULL:
                id_map = ffi.unpack(id_map_ptr, num_nodes) # num_nodes should == final_size
                first_10_ids = [id_map[dense_idx] for dense_idx in final_solution_dense[:10]]
                print(f"\tFirst 10 original node IDs: {first_10_ids}")
            else:
                print("\tCould not retrieve original node ID map")

        else:
            print("\tCould not retrieve final solution array")

        # Retrieve Bader's Best Solution
        print("\n\tRetrieving Bader's Best Recorded Solution")
        bader_best_score = lib.get_best_solution_score(bader_best_storage)
        bader_sol_size = bader_best_storage.solution_size # Access struct field directly
        bader_sol_ptr = lib.get_best_solution_array_ptr(bader_best_storage)
        if bader_sol_size > 0 and bader_sol_ptr != ffi.NULL:
            bader_best_dense = ffi.unpack(bader_sol_ptr, bader_sol_size)
            print(f"\tBader Best Score: {bader_best_score}")
            print(f"\tBader Best Size: {bader_sol_size}")
            print(f"\tBader Best First 10 dense: {bader_best_dense[:10]}")
        else:
            print("\tCould not retrieve Bader's best solution array")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        #Cleanup
        print("\nCleaning up C memory")
        if instance != ffi.NULL:
            lib.free_solution_instance(instance)
            print("\tSolution instance freed")
        if bader_best_storage != ffi.NULL:
            lib.free_best_solution_storage(bader_best_storage)
            print("\tBader best storage freed")
        if connectome != ffi.NULL:
            lib.free_connectome(connectome)
            print("\tConnectome freed")
        print("Cleanup complete")

if __name__ == "__main__":
    main()
