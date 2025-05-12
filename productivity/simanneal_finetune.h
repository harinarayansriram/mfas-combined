#pragma once

#include "connectome.h"
#include "solution_instance.h" 

// Function prototype for the serial annealing approach
void run_simanneal_with_toposhuffle( 
    SolutionInstance* instance,      // Starting solution (modified in place)
    const Connectome* connectome,
    long long total_iterations,   
    int updates_frequency,         
    double tmin,
    double tmax,
    int go_back_to_best_window,   // Interval to reset state to its best known state (within the run)
    int toposhuffle_frequency,    // How often to perform a toposhuffle
    int verbosity,                // Logging level (0=silent, 1=basic, 2=detailed, 10=debug)
    BestSolutionStorage* global_best // Pointer to track the overall best across potentially multiple calls
);

// Helper for toposhuffle
void random_toposort(SolutionInstance* instance, const Connectome* connectome, int verbosity);
