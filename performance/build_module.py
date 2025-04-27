from cffi import FFI

ffi = FFI()

ffi.cdef("""
    // Represents a directed connection to a neighbor node
    typedef struct {
        // long long neighbor_id; // The ID of the node connected to (either to_id or from_id)
        int neighbor_dense_idx; // The DENSE index of the node connected to
        int weight;      // The weight of the connection
    } ConnectionNeighbor;

    // Represents the entire connectome graph
    typedef struct {
        // long long max_node_id;            // Highest node ID encountered + 1 (size needed for arrays)
        long num_nodes;             // Count of unique node IDs encountered
        long num_connections;       // Total number of directed edges
        long long total_weight;     // Sum of all connection weights
        uint64_t* dense_idx_to_node_id; // Map: dense_idx -> original uint64_t node ID. Size = num_nodes

        ConnectionNeighbor** outgoing; // Array of arrays: outgoing[from_id] -> sorted list of {to_id, weight}
        int* out_degree;            // out_degree[from_id] = number of outgoing connections

        ConnectionNeighbor** incoming; // Array of arrays: incoming[to_id] -> sorted list of {from_id, weight}
        int* in_degree;             // in_degree[to_id] = number of incoming connections

        // Optional: Could add node ID mapping if needed, but using direct IDs simplifies things if feasible
        // int* node_id_map;        // Map from original node ID to a dense index (0 to num_nodes-1)
        // int* dense_idx_to_node_id; // Inverse map
        // NodeIdMapping* node_id_map_hash; // Pointer to the hash table head (uthash)

    } Connectome;

    // From solution_instance.h
    typedef struct {
        int* solution;         // Array of node IDs in the current order. Size = connectome->max_node_id
                            // Note: Nodes without connections might exist here, placed arbitrarily.
                            // The score calculation implicitly ignores them.
        int* node_to_position; // Map: node_to_position[node_id] = index in the solution array. Size = connectome->max_node_id
        long long forward_score;   // Current forward score (sum of weights of forward edges)
        int solution_size;     // Number of elements in solution array (connectome->max_node_id)
        // int instance_id;    // Can keep if needed for tracking multiple instances
    } SolutionInstance;

    // Structure to hold the best solution found so far
    typedef struct {
        int *best_solution_array; // Copy of the best solution permutation found
        long long best_score;       // The score of the best solution
        int solution_size;        // Size of the best_solution_array
        // int instance_id;       // ID of the instance that found this best solution
    } BestSolutionStorage;

    // Connectome Management
    Connectome* load_connectome(const char* graph_filename);
    void free_connectome(Connectome* connectome);
    int get_connection_weight(const Connectome* connectome, int from_dense_idx, int to_dense_idx);
    long get_connectome_num_nodes(const Connectome* connectome);
    uint64_t* get_dense_idx_to_node_id_array_ptr(Connectome* connectome); // Allow Python to read the map

    // Solution Instance Management
    SolutionInstance* create_solution_instance(const Connectome* connectome, int* initial_solution_dense_array, bool calculate_initial_score);
    SolutionInstance* create_random_solution_instance(const Connectome* connectome);
    void free_solution_instance(SolutionInstance* instance);
    long long get_solution_score(const SolutionInstance* instance); // Simple accessor
    long get_solution_size(const SolutionInstance* instance);
    int* get_solution_array_ptr(SolutionInstance* instance); // To read solution back in Python
    
    long long calculate_forward_score(const SolutionInstance* instance, const Connectome* connectome);

    // Hashorva Algorithm Entry Point
    void run_simanneal_parallel(
        SolutionInstance* instance,
        const Connectome* connectome,
        double initial_temperature,
        double cooling_rate,
        long long max_iterations,
        long long iterations_per_log,
        bool log_progress
    );

    // Bader Algorithm Entry Point
    void run_simanneal_parallel_with_toposhuffle(
        SolutionInstance* instance,
        const Connectome* connectome,
        int num_threads,
        long long iterations_per_thread,
        int updates_per_thread,
        double tmin,
        double tmax,
        int go_back_to_best_window,
        int toposhuffle_frequency,
        int verbosity,
        BestSolutionStorage* global_best // Python will need to manage this struct
    );
         
    // Okubo Local Search Entry Point
    void run_okubo_local_search(
        SolutionInstance* instance,
        const Connectome* connectome,
        int n_epochs,
        bool log_progress,
        int log_interval
    );

    // Best Solution Management
    BestSolutionStorage* create_best_solution_storage();
    void free_best_solution_storage(BestSolutionStorage* storage);
    void init_best_solution_storage(BestSolutionStorage* storage, const SolutionInstance* instance);
    long long get_best_solution_score(const BestSolutionStorage* storage);
    int* get_best_solution_array_ptr(BestSolutionStorage* storage); // To read best solution

    // Utility
    void seed_rng(unsigned int seed); // Function to seed RNG from Python
""")

source_files = [
    "connectome.c",
    "hashorva/solution_instance.c",
    "hashorva/program.c",
    "simanneal_finetune_parallel.c",
    "okubo_ls.c"
]

c_source = """
#include "exposed_elements.h"
"""

# Configure the CFFI builder
ffi.set_source(
    "connectomics_c",  #Python module name
    c_source,
    sources=source_files,
    include_dirs=["hashorva"], 
    # libraries=["m"],  # Link against math library for exp, log
    extra_compile_args=["-O2", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
    print("\nCFFI module compilation finished.")
    print("Generated module: connectomics_c")
