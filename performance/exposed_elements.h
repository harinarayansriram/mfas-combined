#include "connectome.h" // Include base types
#include "solution_instance.h" // Include solution types
#include "simanneal_finetune_parallel.h" // Include bader function prototypes
#include "okubo_ls.h"

// Connectome Management
Connectome* load_connectome(const char* graph_filename);
void free_connectome(Connectome* connectome);
int get_connection_weight(const Connectome* connectome, int from_dense_idx, int to_dense_idx);
long get_connectome_num_nodes(const Connectome* connectome);
uint64_t* get_dense_idx_to_node_id_array_ptr(const Connectome* connectome); // Allow Python to read the map

// Solution Instance Management
SolutionInstance* create_solution_instance(const Connectome* connectome, int* initial_solution_dense_array, bool calculate_initial_score);
SolutionInstance* create_random_solution_instance(const Connectome* connectome);
void free_solution_instance(SolutionInstance* instance);
long long get_solution_score(const SolutionInstance* instance); // Simple accessor
long get_solution_size(const SolutionInstance* instance);
int* get_solution_array_ptr(SolutionInstance* instance); // To read solution back in Python

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

// Best Solution Management (Python needs to create and manage this)
BestSolutionStorage* create_best_solution_storage();
void free_best_solution_storage(BestSolutionStorage* storage);
void init_best_solution_storage(BestSolutionStorage* storage, const SolutionInstance* instance);
long long get_best_solution_score(const BestSolutionStorage* storage);
int* get_best_solution_array_ptr(BestSolutionStorage* storage); // To read best solution

// Utility
void seed_rng(unsigned int seed); // Function to seed RNG from Python
