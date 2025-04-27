#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../connectome.h"
#include "solution_instance.h"

// --- Hashorva Algorithm Entry Point ---
void run_simanneal_parallel(
    SolutionInstance* instance,      // Input instance, WILL BE MODIFIED to hold the best result
    const Connectome* connectome,
    double initial_temperature,
    double cooling_rate,             // Rate applied per step
    long long max_iterations,        // Total iterations across all threads (approx)
    long long iterations_per_log,    // Logging frequency per thread
    bool log_progress)
{
    if (!instance || !connectome) {
        fprintf(stderr, "Error (run_simanneal_parallel): NULL instance or connectome.\n");
        return;
    }

    long n = connectome->num_nodes;
    if (n <= 1) {
        printf("Warning: Graph has 0 or 1 nodes. No annealing possible.\n");
        return;
    }
    if (initial_temperature <= 0) {
         fprintf(stderr, "Error: Initial temperature must be positive.\n");
         return;
    }
     if (cooling_rate < 0 || cooling_rate >= 1) {
         fprintf(stderr, "Error: Cooling rate must be between 0 and 1 (exclusive of 1).\n");
         return;
     }

    int num_threads = omp_get_max_threads(); // Use max available threads

    // We need a way to store the globally best score found so far across threads
    // We will use the input instance itself, protected by a lock/critical section.
    // Initialize its score if it wasn't calculated before.
    if (instance->forward_score < 0) {
         instance->forward_score = calculate_forward_score(instance, connectome);
    }
    long long global_best_score = instance->forward_score; // Initial best is the input score

    // Calculate iterations per thread. Simple division.
    long long iters_per_thread = (max_iterations + num_threads - 1) / num_threads; // Ceiling division


    if (log_progress) {
        printf("Starting Hashorva Simulated Annealing (Parallel)...\n");
        printf("  Threads: %d\n", num_threads);
        printf("  Target Iterations: %lld (approx %lld per thread)\n", max_iterations, iters_per_thread);
        printf("  Initial Temp: %f\n", initial_temperature);
        printf("  Cooling Rate (per step): %e\n", cooling_rate);
        printf("  Log Interval (per thread): %lld\n", iterations_per_log);
        printf("  Initial Score: %lld\n", instance->forward_score);
        printf("------------------------------------------\n");
    }


    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int thread_seed = time(NULL) ^ (unsigned int)thread_id ^ (unsigned int)random_u64(); // Per-thread RNG seed

        // --- Thread-Local State Initialization ---
        // Create a *copy* of the initial instance for this thread to work on.
        SolutionInstance* thread_instance = create_solution_instance(connectome, instance->solution, false); // Copy solution, don't recalc score yet
        if (!thread_instance) {
             fprintf(stderr, "Error: Thread %d failed to allocate SolutionInstance.\n", thread_id);
             #pragma omp cancellation point parallel // If possible
        }
        thread_instance->forward_score = instance->forward_score; // Copy initial score

        double current_temperature = initial_temperature;
        long long thread_accepted_moves = 0;


        // --- Annealing Loop (Thread-Local) ---
        for (long long iter = 0; iter < iters_per_thread; ++iter) {

            // Check for cancellation (optional)
            #pragma omp cancellation point parallel

            // --- Select Move: Swap two random positions ---
            int pos1 = rand_r(&thread_seed) % n;
            int pos2 = rand_r(&thread_seed) % (n - 1);
            if (pos2 >= pos1) pos2++; // Ensure distinct positions

            // --- Attempt Swap ---
            // Apply swap modifies thread_instance *in place* if accepted
            bool accepted = apply_swap(thread_instance, connectome, pos1, pos2, current_temperature, true, &thread_seed); // `true` = always accept better (matches C# n_f >= o_f)

            if (accepted) {
                thread_accepted_moves++;
            }

            // --- Update Temperature (per step, like C#) ---
            current_temperature *= (1.0 - cooling_rate);
             // Prevent temperature from becoming zero or negative if cooling rate is high
             if (current_temperature < 1e-9) {
                 current_temperature = 1e-9; // Set a floor to avoid issues with exp()
             }

            // --- Logging ---
            if (log_progress && iterations_per_log > 0 && (iter + 1) % iterations_per_log == 0) {
                #pragma omp critical (log_output) // Synchronize console output if needed
                {
                    printf("Thr %d: Iter %lld | Temp %.4e | Score %lld | Accepted %lld\n",
                           thread_id, iter + 1, current_temperature, thread_instance->forward_score, thread_accepted_moves);
                }
                thread_accepted_moves = 0; // Reset counter for next interval
            }

             // Check if we should break (e.g., temperature too low)? C# loops until T < 0.
             // This C loop runs for a fixed number of iterations.
        } // End of annealing loop for this thread


        // --- Update Global Best (using the input instance) ---
        // Safely compare this thread's result with the current best stored in the input 'instance'
        #pragma omp critical (update_global_best)
        {
            // Re-read the potentially updated global best score before comparing
            // Note: instance->forward_score could have been updated by another thread.
            if (thread_instance->forward_score > instance->forward_score) {
                 if (log_progress) {
                      printf("Thr %d: Found new best score: %lld (was %lld). Updating global.\n",
                             thread_id, thread_instance->forward_score, instance->forward_score);
                 }
                 // Copy the better solution into the shared input instance
                 copy_solution(instance->solution, thread_instance->solution, n);
                 // Update the node_to_position map for the shared instance
                 for(int i=0; i<n; ++i) {
                     instance->node_to_position[instance->solution[i]] = i;
                 }
                 // Update the score of the shared instance
                 instance->forward_score = thread_instance->forward_score;

                 // Update the variable tracking the best score seen so far
                 // (This isn't strictly needed if we always read instance->forward_score inside critical,
                 // but can be helpful for clarity or if used elsewhere)
                 // global_best_score = thread_instance->forward_score;
            }
        }

        // --- Thread Cleanup ---
        free_solution_instance(thread_instance);

    } // End of OpenMP parallel region


    // --- Post-Processing ---
    // The best solution found by any thread is now stored in the input 'instance'.
    // Final recalculation just in case
    long long final_recalculated_score = calculate_forward_score(instance, connectome);
    if (final_recalculated_score != instance->forward_score) {
        fprintf(stderr, "Warning: Final incremental score (%lld) differs from recalculation (%lld)!\n", instance->forward_score, final_recalculated_score);
        instance->forward_score = final_recalculated_score; // Optionally correct it
    }


    if (log_progress) {
        printf("------------------------------------------\n");
        printf("Hashorva Simulated Annealing finished.\n");
        printf("Final Best Score (in input instance): %lld\n", instance->forward_score);
        printf("------------------------------------------\n");
    }
}

