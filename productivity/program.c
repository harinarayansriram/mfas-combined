#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "connectome.h"
#include "solution_instance.h"

// --- Hashorva Algorithm Entry Point (Serial) ---
void run_simanneal( // Renamed function
    SolutionInstance* instance,      // Input instance, WILL BE MODIFIED
    const Connectome* connectome,
    double initial_temperature,
    double cooling_rate,             // Rate applied per step
    long long max_iterations,        // Total iterations
    long long iterations_per_log,    // Logging frequency
    bool log_progress)
{
    if (!instance || !connectome) {
        fprintf(stderr, "Error (run_simanneal): NULL instance or connectome.\n");
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

    // Initialize score if it wasn't calculated before.
    if (instance->forward_score < 0) {
         instance->forward_score = calculate_forward_score(instance, connectome);
    }
    // Keep track of the best score found during this run, as the instance itself fluctuates
    long long best_score_so_far = instance->forward_score;
    // We could also store a copy of the best solution array if needed,
    // but the current logic seems to just keep the final state.
    // Let's refine this: keep track of the best state found.
    int* best_solution_buffer = (int*)malloc(n * sizeof(int));
    if (!best_solution_buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for best solution buffer.\n");
        return;
    }
    copy_solution(best_solution_buffer, instance->solution, n); // Store initial as best

    // Seed the random number generator (once for the serial execution)
    // Using the provided random_u64 for seeding standard rand if available,
    // otherwise fall back to time(NULL).
    srand(time(NULL) ^ (unsigned int)random_u64());


    if (log_progress) {
        printf("Starting Hashorva Simulated Annealing (Serial)...\n");
        // printf("  Threads: %d\n", num_threads); // Removed threads
        printf("  Iterations: %lld\n", max_iterations);
        printf("  Initial Temp: %f\n", initial_temperature);
        printf("  Cooling Rate (per step): %e\n", cooling_rate);
        printf("  Log Interval: %lld\n", iterations_per_log);
        printf("  Initial Score: %lld\n", instance->forward_score);
        printf("------------------------------------------\n");
    }

    double current_temperature = initial_temperature;
    long long accepted_moves = 0;
    unsigned int rng_seed = (unsigned int)rand(); // Seed for rand_r

    // --- Annealing Loop (Serial) ---
    for (long long iter = 0; iter < max_iterations; ++iter) {

        // --- Select Move: Swap two random positions ---
        // Use rand_r for potentially better random numbers if needed,
        // or just rand() if sufficient.
        // int pos1 = rand_r(&rng_seed) % n;
        int pos1 = random_u32() % n;
        // int pos2 = rand_r(&rng_seed) % (n - 1);
        int pos2 = random_u32() % (n - 1);
        if (pos2 >= pos1) pos2++; // Ensure distinct positions

        // --- Attempt Swap ---
        // apply_swap modifies the instance *in place* if accepted
        // Need to pass the rng_seed to apply_swap as well
        bool accepted = apply_swap(instance, connectome, pos1, pos2, current_temperature, true, &rng_seed); // `true` = always accept better

        if (accepted) {
            accepted_moves++;
            // If the move was accepted and resulted in a better score, update the best found state
            if (instance->forward_score > best_score_so_far) {
                 best_score_so_far = instance->forward_score;
                 copy_solution(best_solution_buffer, instance->solution, n);
            }
        }

        // --- Update Temperature (per step) ---
        current_temperature *= (1.0 - cooling_rate);
         // Prevent temperature from becoming zero or negative
         if (current_temperature < 1e-9) {
             current_temperature = 1e-9;
         }

        // --- Logging ---
        if (log_progress && iterations_per_log > 0 && (iter + 1) % iterations_per_log == 0) {
            printf("Iter %lld | Temp %.4e | Score %lld (Best %lld) | Accepted %lld\n",
                   iter + 1, current_temperature, instance->forward_score, best_score_so_far, accepted_moves);
            accepted_moves = 0; // Reset counter for next interval
        }
    } // End of annealing loop

    // --- Post-Processing ---
    // Restore the best solution found during the run into the instance
    copy_solution(instance->solution, best_solution_buffer, n);
    // Recalculate score and update maps just to be sure the instance is consistent
    // with the best solution found.
    instance->forward_score = calculate_forward_score(instance, connectome);
    // Need to update node_to_position map as well
    for(int i=0; i<n; ++i) {
        instance->node_to_position[instance->solution[i]] = i;
    }

    // Verify the final score matches the tracked best score
    if (instance->forward_score != best_score_so_far) {
         fprintf(stderr, "Warning: Final score (%lld) after restoring best differs from tracked best score (%lld)!\n", instance->forward_score, best_score_so_far);
         instance->forward_score = calculate_forward_score(instance, connectome);
    }

    // Cleanup
    free(best_solution_buffer);

    if (log_progress) {
        printf("------------------------------------------\n");
        printf("Hashorva Simulated Annealing finished.\n");
        printf("Final Best Score: %lld\n", instance->forward_score);
        printf("------------------------------------------\n");
    }
}