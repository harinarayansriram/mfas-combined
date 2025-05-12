#pragma once

#include <stdbool.h>
#include "connectome.h"
#include "solution_instance.h" 

// Main function for Okubo's Iterated Local Search based on insertion neighborhood
void run_okubo_local_search(
    SolutionInstance* instance,      // Solution instance to modify
    const Connectome* connectome,
    int n_epochs,                  // Number of outer loop iterations
    bool log_progress,             // Flag for basic console logging
    int log_interval               // How often to log progress (e.g., every N nodes)
);
