import os
import time
import cffi

from connectomics_c_productivity import ffi, lib

DEFAULT_GRAPH_FILE = "../data/connectome_graph.csv"
GRAPH_FILE_PATH = os.environ.get("CONNECTOMICS_GRAPH_PATH", DEFAULT_GRAPH_FILE)

# Algorithm Parameters
OKUBO_EPOCHS = 3
OKUBO_LOG_INTERVAL = 10000

BADER_TOTAL_ITERATIONS = 5000000
BADER_UPDATES_FREQUENCY = 5
BADER_TMIN = 0.001
BADER_TMAX = 0.1
BADER_GOBACK_WINDOW = 500000
BADER_TOPOSHUFFLE_FREQ = 250000
BADER_VERBOSITY = 1

HASHORVA_INITIAL_TEMP = 10.0
HASHORVA_COOLING_RATE = 1e-7
HASHORVA_MAX_ITERATIONS = 20000000
HASHORVA_LOG_INTERVAL = 500000

LOG_PROGRESS = True
SEED = 42

def main():
    print("Productivity version of MFAS connectomics algorithms test")
    print(f"Using graph file: {GRAPH_FILE_PATH}")
    if not os.path.exists(GRAPH_FILE_PATH):
        print(f"Error: Graph file not found at {GRAPH_FILE_PATH}")
        print("Please set the CONNECTOMICS_GRAPH_PATH environment variable or place graph.csv")
        return

    connectome = ffi.NULL
    instance = ffi.NULL
    finetune_best_storage = ffi.NULL

    try:
        print(f"\nInitializing and Seeding RNG (Seed: {SEED})")
        lib.seed_rng(SEED)
        print("RNG Seeded")

        print("\nLoading Connectome")
        start_time = time.perf_counter()
        connectome = lib.load_connectome(GRAPH_FILE_PATH.encode('utf-8'))
        load_time = time.perf_counter() - start_time
        if connectome == ffi.NULL:
            print("Error: Failed to load connectome")
            return
        print(f"Connectome loaded successfully in {load_time:.2f} seconds")
        num_nodes = lib.get_connectome_num_nodes(connectome)
        print(f"\tUnique Nodes: {num_nodes}")

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

        print("\nPreparing Best Solution Storage")
        finetune_best_storage = lib.create_best_solution_storage()
        if finetune_best_storage == ffi.NULL:
            print("Error: Failed to create best solution storage")
            return
        lib.init_best_solution_storage(finetune_best_storage, instance)
        print(f"\tInitialized Best Storage with score: {lib.get_best_solution_score(finetune_best_storage)}")

        # Okubo Local Search
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
        print(f"    Score after Okubo LS: {score_after_okubo}")

        lib.update_best_solution_storage_if_better(finetune_best_storage, instance)


        # Fine-tune annealing with toposhuffle (Bader)
        print("\nRunning Fine-tuning Annealing (with Toposhuffle)")
        start_time = time.perf_counter()
        # Note: The instance state from Okubo is passed here
        # The best storage is also passed to potentially capture a better state found during annealing
        lib.run_simanneal_with_toposhuffle( 
            instance,
            connectome,
            BADER_TOTAL_ITERATIONS, 
            BADER_UPDATES_FREQUENCY, 
            BADER_TMIN,
            BADER_TMAX,
            BADER_GOBACK_WINDOW,
            BADER_TOPOSHUFFLE_FREQ,
            BADER_VERBOSITY,
            finetune_best_storage, 
        )
        finetune_time = time.perf_counter() - start_time
        score_after_finetune = lib.get_solution_score(instance)
        finetune_best_score = lib.get_best_solution_score(finetune_best_storage)
        print(f"Fine-tuning Annealing finished in {finetune_time:.2f} seconds")
        print(f"\tInstance Score after Fine-tune: {score_after_finetune}")
        print(f"\tBest Storage Score after Fine-tune:   {finetune_best_score}")
        # The instance holds the result of the single annealing run
        # The best storage holds the best score seen across Okubo and this run


        print("\nRunning Simulated Annealing")
        start_time = time.perf_counter()
        # Note: Instance state from the previous step is used
        lib.run_simanneal( 
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
        print(f"Simulated Annealing finished in {hashorva_time:.2f} seconds")
        print(f"\tScore after Simulated Annealing: {score_after_hashorva}")
        # Update best storage if this annealing run found a better solution
        lib.update_best_solution_storage_if_better(finetune_best_storage, instance)


        print("\nFinal solution:")
        final_instance_score = lib.get_solution_score(instance)
        final_best_storage_score = lib.get_best_solution_score(finetune_best_storage)
        final_size = lib.get_solution_size(instance)
        solution_ptr = lib.get_solution_array_ptr(instance)

        print(f"\tFinal Instance Score: {final_instance_score}")
        print(f"\tFinal Best Storage Score: {final_best_storage_score}") # Best overall

        if final_size > 0 and solution_ptr != ffi.NULL:
            final_solution_dense = ffi.unpack(solution_ptr, final_size)
            print(f"\tInstance Solution Size: {final_size}")
            print(f"\tInstance First 10 dense indices: {final_solution_dense[:10]}")

            id_map_ptr = lib.get_dense_idx_to_node_id_array_ptr(connectome)
            if id_map_ptr != ffi.NULL and num_nodes > 0:
                id_map = ffi.unpack(id_map_ptr, num_nodes)
                if len(id_map) == num_nodes: # sanity check
                    first_10_ids = [id_map[dense_idx] for dense_idx in final_solution_dense[:10] if dense_idx < num_nodes]
                    print(f"\tInstance First 10 original node IDs: {first_10_ids}")
                else:
                    print("\tWarning: ID map size mismatch")
            else:
                print("\tCould not retrieve original node ID map")
        else:
            print("\tCould not retrieve final instance solution array")

        print("\n\tRetrieving Overall Best Recorded Solution")
        best_sol_size = finetune_best_storage.solution_size
        best_sol_ptr = lib.get_best_solution_array_ptr(finetune_best_storage)
        if best_sol_size > 0 and best_sol_ptr != ffi.NULL:
            best_dense = ffi.unpack(best_sol_ptr, best_sol_size)
            print(f"\tOverall Best Score: {final_best_storage_score}")
            print(f"\tOverall Best Size: {best_sol_size}")
            print(f"\tOverall Best First 10 dense: {best_dense[:10]}")
        else:
            print("\tCould not retrieve overall best solution array")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # cleanup
        print("\nCleaning up C memory")
        if instance != ffi.NULL:
            lib.free_solution_instance(instance)
            print("\tSolution instance freed")
        if finetune_best_storage != ffi.NULL:
            lib.free_best_solution_storage(finetune_best_storage)
            print("\tBest storage freed")
        if connectome != ffi.NULL:
            lib.free_connectome(connectome)
            print("\tConnectome freed")
        print("Cleanup complete")

if __name__ == "__main__":
    main()