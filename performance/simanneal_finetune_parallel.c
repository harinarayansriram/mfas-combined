#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h> // For parallel execution

#include "simanneal_finetune_parallel.h"

// --- Toposhuffle ---
// Needs careful adaptation to use the Connectome and SolutionInstance structs
void random_toposort(SolutionInstance* instance, const Connectome* connectome, int verbosity) {
    long n = instance->solution_size;
    if (n <= 0) return;

    // Data structures for Kahn's algorithm
    int* indegree = (int*)calloc(n, sizeof(int)); // In-degree based on *current* forward edges
    int* queue = (int*)malloc(n * sizeof(int));
    int queue_head = 0, queue_tail = 0;
    int* new_ordering = (int*)malloc(n * sizeof(int)); // Stores the dense indices in topo order
    int ordering_idx = 0;
    long long current_forward_weight = 0; // For verification/logging

    if (!indegree || !queue || !new_ordering) {
        perror("Failed to allocate memory for toposort");
        free(indegree); free(queue); free(new_ordering);
        return;
    }

    // Calculate in-degrees based *only* on forward edges in the current solution
    for (int i = 0; i < n; ++i) {
        int u_dense_idx = instance->solution[i]; // Dense index at position i
        if (u_dense_idx < 0 || u_dense_idx >= connectome->num_nodes || !connectome->outgoing[u_dense_idx]) continue;
    
        for (int j = 0; j < connectome->out_degree[u_dense_idx]; ++j) {
            int v_dense_idx = connectome->outgoing[u_dense_idx][j].neighbor_dense_idx;
            if (v_dense_idx >= n) continue; // Skip nodes outside range
            // int v_pos = instance->node_to_position[v];
            int v_pos = instance->node_to_position[v_dense_idx];

            if (v_pos > i) { // If u -> v is a forward edge
                indegree[v_dense_idx]++; // Increment in-degree of the target node v
                current_forward_weight += connectome->outgoing[u_dense_idx][j].weight;
            }
        }
    }
     if (verbosity >= 10) printf("Toposshuffle: Initial forward weight calculated: %lld\n", current_forward_weight);


    // Initialize queue with nodes having zero in-degree (in the forward graph)
    for (int node_id = 0; node_id < n; ++node_id) {
         // Only consider nodes that might actually be part of the graph
        if (node_id >= 0 && node_id < connectome->num_nodes && (connectome->out_degree[node_id] > 0 || connectome->in_degree[node_id] > 0)) {
            if (indegree[node_id] == 0) {
                queue[queue_tail++] = node_id;
            }
         } else {
            // Ensure nodes without connections but within the dense index range are added
            if (node_id >= 0 && node_id < connectome->num_nodes && indegree[node_id] == 0) {
                 queue[queue_tail++] = node_id;
             }
         }
    }

    // Process the queue (Kahn's algorithm)
    while (queue_head < queue_tail) {
        // Randomly pick from the current queue elements (nodes ready to be placed)
        int rand_idx_in_queue = queue_head + (random_u64() % (queue_tail - queue_head));
        int u_dense_idx = queue[rand_idx_in_queue];
        // Swap with head to 'remove' it easily
        queue[rand_idx_in_queue] = queue[queue_head];
        queue_head++;

        new_ordering[ordering_idx++] = u_dense_idx; // Add node to the new topological order

        // Process neighbors only considering forward edges
        int u_pos = instance->node_to_position[u_dense_idx]; // Original position
        if (u_dense_idx < 0 || u_dense_idx >= connectome->num_nodes || !connectome->outgoing[u_dense_idx]) continue;
        for (int j = 0; j < connectome->out_degree[u_dense_idx]; ++j) {
            int v_dense_idx = connectome->outgoing[u_dense_idx][j].neighbor_dense_idx;
            if (v_dense_idx < 0 || v_dense_idx >= n) continue;
            int v_pos = instance->node_to_position[v_dense_idx];

            if (v_pos > u_pos) { // If u -> v was a forward edge
                indegree[v_dense_idx]--;
                if (indegree[v_dense_idx] == 0) {
                    queue[queue_tail++] = v_dense_idx; // Add neighbor to queue if ready
                }
            }
        }
    }

    // Check for cycles and handle unconnected nodes
    if (ordering_idx < n) {
        fprintf(stderr, "Warning: Cycle detected during topological sort or unconnected nodes missed? Processed %d / %d nodes.\n", ordering_idx, n);
        // Fill remaining spots in new_ordering with unprocessed nodes if any
        // This part needs careful handling depending on how cycles should be treated.
        // For now, we might end up with an incomplete or invalid ordering.
        // A simple fallback: Keep the original ordering if a cycle is detected.
         if (verbosity >= 1) printf("Toposort failed (cycle?), keeping original order.\n");
         free(indegree); free(queue); free(new_ordering);
         return; // Do not modify the instance state
    }

     if (verbosity >= 2) printf("Toposort successful. Updating instance state.\n");

    // Update the SolutionInstance with the new topological order
    copy_solution(instance->solution, new_ordering, n);
    for (int i = 0; i < n; ++i) {
        instance->node_to_position[instance->solution[i]] = i;
    }
    // Recalculate the forward score for consistency
    instance->forward_score = calculate_forward_score(instance, connectome);

    if (verbosity >= 10) printf("Toposort: New forward score: %lld\n", instance->forward_score);


    free(indegree);
    free(queue);
    free(new_ordering);
}


// --- Bader's Annealing Logic (Adapted for Parallel Execution) ---
void run_simanneal_parallel_with_toposhuffle(
    SolutionInstance* instance, // Starting instance (will be updated with best result)
    const Connectome* connectome,
    int num_threads,
    long long iterations_per_thread,
    int updates_per_thread,
    double tmin,
    double tmax,
    int go_back_to_best_window,
    int toposhuffle_frequency,
    int verbosity,
    BestSolutionStorage* global_best // Shared tracker for the overall best solution
) {
    if (!instance || !connectome || num_threads <= 0 || iterations_per_thread <= 0) {
        fprintf(stderr, "Invalid arguments for parallel annealing.\n");
        return;
    }
    if (tmin <= 0 || tmax <= 0 || tmin > tmax) {
        fprintf(stderr, "Invalid temperature range (tmin=%.4g, tmax=%.4g).\n", tmin, tmax);
        return;
    }

    printf("Starting Bader Parallel Annealing...\n");
    printf("  Threads: %d, Iter/Thread: %lld, T: [%.4g, %.4g]\n", num_threads, iterations_per_thread, tmin, tmax);
    printf("  GoBackWindow: %d, TopoShuffleFreq: %d\n", go_back_to_best_window, toposhuffle_frequency);

    // Array to hold thread-local best scores found during this run
    long long* thread_best_scores = (long long*)malloc(num_threads * sizeof(long long));
    int best_thread_index = -1; // Index of the thread that ended with the best score

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        // Ensure each thread has its own random seed (important!)
        unsigned int seed = (unsigned int) ( (time(NULL) ^ (thread_id << 16) ^ random_u64()) & UINT_MAX);
        srand(seed); // Seed thread-local random state if rand() is used directly, or use thread-safe RNG

        // Create a thread-local copy of the solution instance
        SolutionInstance* local_instance = create_solution_instance(connectome, instance->solution, false); // Don't recalc score yet
        if (!local_instance) {
             fprintf(stderr, "Thread %d: Failed to create local instance!\n", thread_id);
             #pragma omp atomic write
             thread_best_scores[thread_id] = -1; // Indicate error
        } else {
            local_instance->forward_score = instance->forward_score; // Copy initial score

            // Thread-local best tracking
            BestSolutionStorage thread_best = {NULL, local_instance->forward_score, local_instance->solution_size};
            thread_best.best_solution_array = (int*)malloc(thread_best.solution_size * sizeof(int));
            if(thread_best.best_solution_array) {
                copy_solution(thread_best.best_solution_array, local_instance->solution, thread_best.solution_size);
            } else {
                perror("Thread failed to allocate local best storage");
                // Continue without local best tracking? Risky.
            }


            double current_temp = tmax;
            double tfactor = (tmax > tmin) ? -log(tmax / tmin) : 0.0; // Avoid log(1)
            long long trials = 0, accepts = 0, improves = 0;
            long long update_interval = (updates_per_thread > 0) ? (iterations_per_thread / updates_per_thread) : -1;

            if (verbosity >= 2) printf("Thread %d: Starting annealing loop. Initial score: %lld\n", thread_id, local_instance->forward_score);

            for (long long step = 0; step < iterations_per_thread; ++step) {
                // Update temperature (linear interpolation in log space)
                 if (tfactor != 0.0) {
                     current_temp = tmax * exp(tfactor * (double)step / (double)iterations_per_thread);
                 } else {
                     current_temp = tmax; // Constant temperature if tmin == tmax
                 }
                 current_temp = fmax(current_temp, tmin); // Ensure temp doesn't go below tmin


                // Periodically go back to thread's best known state
                if (go_back_to_best_window > 0 && step > 0 && step % go_back_to_best_window == 0) {
                    if (thread_best.best_solution_array && thread_best.best_score > local_instance->forward_score) {
                         if (verbosity >= 10) printf("Thread %d Step %lld: Going back to best (Score %lld)\n", thread_id, step, thread_best.best_score);
                         copy_solution(local_instance->solution, thread_best.best_solution_array, local_instance->solution_size);
                         // Update pos map and score
                         for(int i=0; i < local_instance->solution_size; ++i) local_instance->node_to_position[local_instance->solution[i]] = i;
                         local_instance->forward_score = thread_best.best_score;
                    }
                }

                 // Periodically perform topological shuffle
                 if (toposhuffle_frequency > 0 && step > 0 && step % toposhuffle_frequency == 0) {
                     if (verbosity >= 2) printf("Thread %d Step %lld: Performing toposort shuffle\n", thread_id, step);
                     random_toposort(local_instance, connectome, verbosity); // Modifies local_instance in place
                     // Update thread's best if toposort improved it
                      if (thread_best.best_solution_array) {
                          update_best_solution(&thread_best, local_instance);
                      }
                     // Also check against global best
                     #pragma omp critical (GlobalBestUpdate)
                     {
                         update_best_solution(global_best, local_instance);
                     }
                 }

                // --- Select move
                long pos1 = random_u64() % local_instance->solution_size;
                long pos2 = random_u64() % local_instance->solution_size;
                if (pos1 == pos2) pos2 = (pos1 + 1) % local_instance->solution_size;

                trials++;
                long long dE = calculate_score_delta_on_swap(local_instance, connectome, pos1, pos2);

                // --- Acceptance Criteria ---
                bool accepted = false;
                 if (dE > 0) { // Improvement
                     accepted = true;
                     improves++;
                 } else if (current_temp > 1e-9) { // Avoid issues with zero temp
                     double prob = exp((double)dE / current_temp);
                     if (prob > random_double()) {
                         accepted = true;
                     }
                 }

                 if (accepted) {
                     accepts++;
                     // Apply swap modifies local_instance score and arrays
                      apply_swap(local_instance, connectome, pos1, pos2, current_temp, true); // Use helper, always accept 0 delta here

                     // Update thread-local best
                      if (thread_best.best_solution_array) {
                         update_best_solution(&thread_best, local_instance);
                      }

                     // Update global best (needs synchronization)
                     #pragma omp critical (GlobalBestUpdate)
                     {
                         update_best_solution(global_best, local_instance);
                     }
                 }

                 // --- Logging within thread ---
                 if (verbosity == 10 && update_interval > 0 && step > 0 && step % update_interval == 0) {
                     float accept_percent = (trials > 0) ? (float)accepts * 100.0f / trials : 0.0f;
                     float improve_percent = (trials > 0) ? (float)improves * 100.0f / trials : 0.0f;
                     printf("  T%d [%lld/%lld] T:%.3g E:%lld BestE:%lld Acc:%.2f%% Imp:%.2f%%\n",
                            thread_id, step, iterations_per_thread, current_temp,
                            local_instance->forward_score, thread_best.best_score, accept_percent, improve_percent);
                     trials = 0; accepts = 0; improves = 0; // Reset counters for next interval
                 }

            } // End annealing loop for thread

            // Store the final score achieved by this thread
            #pragma omp atomic write
            thread_best_scores[thread_id] = thread_best.best_score;

            // Copy the best solution found by THIS thread back to its local_instance
            if (thread_best.best_solution_array && thread_best.best_score > local_instance->forward_score) {
                 copy_solution(local_instance->solution, thread_best.best_solution_array, local_instance->solution_size);
                 for(int i=0; i < local_instance->solution_size; ++i) local_instance->node_to_position[local_instance->solution[i]] = i;
                 local_instance->forward_score = thread_best.best_score;
            }

             if (verbosity >= 1) printf("Thread %d finished. Final score: %lld (Best found by thread: %lld)\n", thread_id, local_instance->forward_score, thread_best.best_score);

             // --- Critical section to determine which thread's final state is best ---
             // This is slightly complex: we want the *state* of the thread that achieved the overall best score *at the end*.
             // It's simpler to just rely on the global_best structure updated during the run.
             // Let's find which thread ended with the highest score among all threads.
             #pragma omp critical(FindBestThread)
             {
                if (best_thread_index == -1 || thread_best_scores[thread_id] > thread_best_scores[best_thread_index]) {
                    best_thread_index = thread_id;
                }
             }

             // Cleanup thread-local resources
             free(thread_best.best_solution_array);
             // Keep local_instance alive temporarily until the best one is chosen below
             //#pragma omp barrier // Ensure all threads reach here before proceeding
             // If we decide to copy the state from the best thread:
             // This needs careful synchronization or post-processing outside the parallel region.


        } // End check if local_instance was created
    } // End parallel region

    // --- Post-processing ---
    // At this point, global_best holds the best solution found across all threads during the run.
    // Update the original input 'instance' with this global best.
    if (global_best->best_solution_array && global_best->best_score > instance->forward_score) {
         printf("Updating input instance with overall best score: %lld (was %lld)\n", global_best->best_score, instance->forward_score);
         copy_solution(instance->solution, global_best->best_solution_array, instance->solution_size);
         // Update position map and score
         for(int i=0; i < instance->solution_size; ++i) instance->node_to_position[instance->solution[i]] = i;
         instance->forward_score = global_best->best_score;
    } else {
        printf("Parallel run did not improve upon the initial global best score (%lld).\n", global_best->best_score);
    }


    // Cleanup
    free(thread_best_scores);

    printf("Bader Parallel Annealing finished.\n");
}


// Example Main function (will be replaced by Python calls)
int main_example(int argc, char *argv[]) {
     printf("Bader Parallel Annealing Refactored Example\n");
     srand(time(NULL));

     // --- Configuration ---
     const char* graph_filename = "./graph.csv"; // Input graph
     int num_threads = 4; // Example
     long long iters_per_thread = 25000000; // Example
     double tmin = 0.001;
     double tmax = 0.1;
     int go_back_window = 10000000; // Large window
     int toposhuffle_freq = 5000000; // Relatively frequent
     int verbosity = 2; // Detailed logging
     int updates_per_thread = 10; // Log 10 times per thread run

     // --- Load Connectome ---
     Connectome* connectome = load_connectome(graph_filename);
     if (!connectome) return 1;

     // --- Create Initial Solution ---
     SolutionInstance* current_solution = create_random_solution_instance(connectome);
     if (!current_solution) {
         free_connectome(connectome);
         return 1;
     }

     // --- Initialize Global Best Storage ---
     // Important: Initialize before the parallel run
     BestSolutionStorage global_best = {NULL, -1, 0};
     global_best.solution_size = current_solution->solution_size;
     global_best.best_score = current_solution->forward_score; // Start with initial random score
     global_best.best_solution_array = (int*)malloc(global_best.solution_size * sizeof(int));
     if (global_best.best_solution_array) {
         copy_solution(global_best.best_solution_array, current_solution->solution, global_best.solution_size);
     } else {
         perror("Failed to allocate global best storage");
         free_solution_instance(current_solution);
         free_connectome(connectome);
         return 1;
     }


     // --- Run Bader's Annealing ---
     run_simanneal_parallel_with_toposhuffle(
         current_solution, // This instance will be updated with the best result
         connectome,
         num_threads,
         iters_per_thread,
         updates_per_thread,
         tmin,
         tmax,
         go_back_window,
         toposhuffle_freq,
         verbosity,
         &global_best // Pass the shared global best tracker
     );


     // --- Results ---
     printf("\nFinal Best Score (Bader): %lld\n", global_best.best_score);
     printf("Solution instance score after run: %lld\n", current_solution->forward_score); // Should match global best

    // Optional: Save the best solution (better handled by Python wrapper)
    if (global_best.best_solution_array) {
        printf("Saving best solution to best_solution_bader.txt\n");
        FILE* outfile = fopen("best_solution_bader.txt", "w");
        if (outfile) {
             for (int i = 0; i < global_best.solution_size; ++i) {
                 fprintf(outfile, "%d\n", global_best.best_solution_array[i]);
             }
             fclose(outfile);
        } else {
            perror("Failed to write best solution file");
        }
    }

     // --- Cleanup ---
     free(global_best.best_solution_array);
     free_solution_instance(current_solution);
     free_connectome(connectome);

     printf("Cleanup complete. Exiting.\n");
     return 0;
}