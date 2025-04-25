#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h> // Keep if using OpenMP in Hashorva's original approach
#include <stdbool.h>

#include "../connectome.h"
#include "solution_instance.h"

// --- Configuration Constants (formerly macros/globals) ---
// const int THREAD_NUM = 4; // Example, maybe configure externally
const long long ITERATIONS_PER_TEMP_CHECK = 1000000; // How often to check temp/log


// --- Global for Best Solution ---
// It's often better to pass this around, but a single global might be okay
// if only one annealing process runs at a time.
BestSolutionStorage overall_best_solution = {NULL, -1, 0}; // Initialize score to invalid

// --- Forward Declarations (if needed, e.g., for thread_run if using) ---
// void thread_run(...); // If using Hashorva's threading model

// Simple SimAnneal implementation (can be adapted for threading later)
void run_simanneal_parallel(
    SolutionInstance* instance,      // Starting (and evolving) solution
    const Connectome* connectome,
    double initial_temperature,
    double cooling_rate,
    long long max_iterations,        // Total iterations limit
    long long iterations_per_log,    // How often to print status
    bool log_progress              // Flag to enable/disable console logging
) {
    if (!instance || !connectome) return;

    double temperature = initial_temperature;
    long long iteration = 0;
    time_t last_log_time = time(NULL);

    // Initial check against the best solution found so far
    update_best_solution(&overall_best_solution, instance);

    printf("Starting Simulated Annealing...\n");
    printf("  Initial Temp: %.4f, Cooling Rate: %.10f, Max Iterations: %lld\n",
           initial_temperature, cooling_rate, max_iterations);
     printf("  Initial Score: %lld (Best known: %lld)\n", instance->forward_score, overall_best_solution.best_score);

    while (temperature > 1e-9 && (max_iterations <= 0 || iteration < max_iterations)) {
        // --- Perform a batch of swaps ---
        // Original Hashorva used threads here. Simplified sequential version:
        for (long long i = 0; i < ITERATIONS_PER_TEMP_CHECK && (max_iterations <= 0 || iteration < max_iterations); ++i) {
            // Select two distinct random positions
            int pos1 = random_u64() % instance->solution_size;
            int pos2 = random_u64() % instance->solution_size;
            if (pos1 == pos2) {
                pos2 = (pos1 + 1) % instance->solution_size; // Ensure different
            }

            // Attempt the swap
            bool accepted = apply_swap(instance, connectome, pos1, pos2, temperature, false);

             // Check if the current state is the new best overall
            if (accepted) { // Only need to check if score changed
                 update_best_solution(&overall_best_solution, instance);
            }

            iteration++;
        } // End batch of swaps

        // --- Update temperature ---
        temperature *= (1.0 - cooling_rate); // Geometric cooling

        // --- Logging ---
        if (log_progress && (iteration % iterations_per_log == 0 || (time(NULL) - last_log_time >= 10))) {
             time_t now = time(NULL);
             char timestamp[64];
             strftime(timestamp, sizeof(timestamp), "%H:%M:%S", localtime(&now));

             double progress_percent = (max_iterations > 0) ? (double)iteration * 100.0 / max_iterations : 0.0;
             double ratio = (connectome->total_weight > 0) ? (double)overall_best_solution.best_score / connectome->total_weight : 0.0;

             printf("%s [%lld / %lld, %.1f%%] Temp: %.4g, Current Score: %lld, Best Score: %lld (Ratio: %.6f)\n",
                    timestamp, iteration, max_iterations > 0 ? max_iterations : -1, progress_percent,
                    temperature, instance->forward_score, overall_best_solution.best_score, ratio);
             last_log_time = now;

             // Optional: Log to file (better handled by Python wrapper)
             // FILE* log_file = fopen("simanneal_log.txt", "a");
             // if (log_file) {
             //     fprintf(log_file, "%ld\t%lld\t%.6g\t%lld\t%.6f\n", now, iteration, temperature, overall_best_solution.best_score, ratio);
             //     fclose(log_file);
             // }
        }

    } // End while temperature > threshold

    printf("Simulated Annealing finished.\n");
    printf("  Final Temperature: %.4g\n", temperature);
    printf("  Total Iterations: %lld\n", iteration);
    printf("  Best Score Achieved: %lld\n", overall_best_solution.best_score);

    // Optional: Copy the best solution back into the instance if desired
     if (overall_best_solution.best_solution_array && overall_best_solution.best_score > instance->forward_score) {
         printf("  (Restoring best found solution into the instance)\n");
         copy_solution(instance->solution, overall_best_solution.best_solution_array, instance->solution_size);
         // Recalculate node_to_position map and score for the instance
         for(int i=0; i < instance->solution_size; ++i) instance->node_to_position[instance->solution[i]] = i;
         instance->forward_score = overall_best_solution.best_score; // Should match calculate_forward_score
     }
}


// Example Main function (will be replaced by Python calls)
int main_hashorva_example(int argc, char *argv[]) {
    printf("Hashorva Feed Forward Refactored Example\n");
    srand(time(NULL)); // Seed random number generator ONCE

    // --- Configuration ---
    const char* graph_filename = "./graph.csv"; // Input graph
    double initial_temp = 50.0;
    double cool_rate = 1e-9; // Very slow cooling
    long long total_iterations = 50000000; // Example limit
    long long log_interval = 5000000; // Log every 5M iterations

    // --- Load Connectome ---
    Connectome* connectome = load_connectome(graph_filename);
    if (!connectome) {
        return 1;
    }

    // --- Create Initial Solution ---
    // SolutionInstance* current_solution = read_solution_from_file(...) // If reading start point
    SolutionInstance* current_solution = create_random_solution_instance(connectome);
    if (!current_solution) {
        free_connectome(connectome);
        return 1;
    }

    // Initialize best solution storage before starting
    overall_best_solution.best_score = -1; // Reset best score

    // --- Run Annealing ---
    run_simanneal_parallel(
        current_solution,
        connectome,
        initial_temp,
        cool_rate,
        total_iterations,
        log_interval,
        true // Enable logging
    );

    // --- Results ---
    printf("\nFinal Best Score: %lld\n", overall_best_solution.best_score);

    // Optional: Save the best solution (better handled by Python wrapper)
    if (overall_best_solution.best_solution_array) {
        printf("Saving best solution to best_solution_hashorva.txt\n");
        FILE* outfile = fopen("best_solution_hashorva.txt", "w");
        if (outfile) {
             for (int i = 0; i < overall_best_solution.solution_size; ++i) {
                 // Only write nodes that actually existed (e.g., based on degree)
                 // Or just write all indices if the array represents the full range 0..max_id
                 fprintf(outfile, "%d\n", overall_best_solution.best_solution_array[i]);
             }
             fclose(outfile);
        } else {
            perror("Failed to write best solution file");
        }
    }

    // --- Cleanup ---
    free(overall_best_solution.best_solution_array);
    free_solution_instance(current_solution);
    free_connectome(connectome);

    printf("Cleanup complete. Exiting.\n");
    return 0;
}

// Note: The main function above is just for testing.
// The actual entry points for CFFI would be functions like load_connectome,
// create_random_solution_instance, run_simanneal_parallel, free_solution_instance, etc.