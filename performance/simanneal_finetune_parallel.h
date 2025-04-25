#pragma once

#include "connectome.h"
#include "solution_instance.h" // Use the same solution instance structure

// Function prototype for Bader's parallel annealing approach
void run_simanneal_parallel_with_toposhuffle(
    SolutionInstance* instance,      // Starting solution (will be modified in place by the best thread)
    const Connectome* connectome,
    int num_threads,
    long long iterations_per_thread,
    int updates_per_thread,        // How often each thread prints progress (if verbosity allows)
    double tmin,
    double tmax,
    int go_back_to_best_window,   // Interval to reset thread state to its best known state
    int toposhuffle_frequency,    // How often to perform a random topological sort
    int verbosity,                // Logging level (0=silent, 1=basic, 2=detailed, 10=debug)
    BestSolutionStorage* global_best // Pointer to track the overall best across all threads
);

// Helper for topological sort (might need adaptation)
void random_toposort(SolutionInstance* instance, const Connectome* connectome, int verbosity);