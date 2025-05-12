#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#include "connectome.h"
#include "solution_instance.h"
#include "simanneal_finetune_parallel.h"

// Forward declaration for helper functions internal to this file
long long calculate_delta_score_on_swap_positions(
    const SolutionInstance* instance,
    const Connectome* connectome,
    int dense_idx_a,
    int dense_idx_b);

// --- Random Topological Sort (Toposhuffle) ---
// This function performs a randomized topological sort *based on the current solution's forward edges*.
// It modifies the instance in place and recalculates its score.
void random_toposort(SolutionInstance* instance, const Connectome* connectome, int verbosity) {
    long n = connectome->num_nodes;
    if (n == 0) return; // Nothing to sort

    if(verbosity >= 2) printf("Performing random toposort...\n");

    // --- 1. Build temporary graph based on forward edges and calculate in-degrees ---
    int* current_indegree = (int*)calloc(n, sizeof(int));
    // We don't strictly need to store the forward edges explicitly,
    // we can re-check the position condition when processing neighbors.
    // However, calculating in-degrees requires iterating through all edges once.

    // Use a temporary structure to store forward neighbors efficiently during the sort
    // This avoids repeatedly checking positions inside the main toposort loop.
    ConnectionNeighbor** forward_adj = (ConnectionNeighbor**)calloc(n, sizeof(ConnectionNeighbor*));
    int* forward_degree = (int*)calloc(n, sizeof(int)); // Store count of forward outgoing edges

    if (!current_indegree || !forward_adj || !forward_degree) {
        fprintf(stderr, "Error: Failed to allocate memory for toposort structures.\n");
        if (current_indegree) free(current_indegree);
        if (forward_adj) free(forward_adj); // Note: inner pointers not allocated yet
        if (forward_degree) free(forward_degree);
        // Consider a more robust error handling strategy if needed
        return;
    }

    // First pass: Count forward degrees to allocate memory for forward_adj lists
    for (int i_dense = 0; i_dense < n; ++i_dense) {
        int i_pos = instance->node_to_position[i_dense];
        for (int k = 0; k < connectome->out_degree[i_dense]; ++k) {
            int neighbor_dense_idx = connectome->outgoing[i_dense][k].neighbor_dense_idx;
            int neighbor_pos = instance->node_to_position[neighbor_dense_idx];
            if (i_pos < neighbor_pos) { // Is it a forward edge in the current solution?
                forward_degree[i_dense]++;
            }
        }
        if (forward_degree[i_dense] > 0) {
            forward_adj[i_dense] = (ConnectionNeighbor*)malloc(forward_degree[i_dense] * sizeof(ConnectionNeighbor));
            if (!forward_adj[i_dense]) {
                 fprintf(stderr, "Error: Failed to allocate memory for forward adjacency list.\n");
                 // Need cleanup for previously allocated parts
                 free(current_indegree);
                 free(forward_degree);
                 for(int j=0; j<i_dense; ++j) free(forward_adj[j]);
                 free(forward_adj);
                 return;
            }
        }
    }

    // Second pass: Populate forward_adj and calculate initial in-degrees
    memset(forward_degree, 0, n * sizeof(int)); // Reset counts to use as insertion indices
    for (int i_dense = 0; i_dense < n; ++i_dense) {
        int i_pos = instance->node_to_position[i_dense];
        for (int k = 0; k < connectome->out_degree[i_dense]; ++k) {
            int neighbor_dense_idx = connectome->outgoing[i_dense][k].neighbor_dense_idx;
            int weight = connectome->outgoing[i_dense][k].weight;
            int neighbor_pos = instance->node_to_position[neighbor_dense_idx];

            if (i_pos < neighbor_pos) { // Forward edge
                current_indegree[neighbor_dense_idx]++;
                // Store this forward edge
                forward_adj[i_dense][forward_degree[i_dense]].neighbor_dense_idx = neighbor_dense_idx;
                forward_adj[i_dense][forward_degree[i_dense]].weight = weight; // Store weight if needed later? (Not strictly needed for sort)
                forward_degree[i_dense]++; // Increment index for next insertion
            }
        }
    }


    // --- 2. Initialize queue with nodes having in-degree 0 ---
    int* queue = (int*)malloc(n * sizeof(int));
    int queue_head = 0; // conceptually, where we add next
    int queue_tail = 0; // conceptually, where we remove from (used for random choice)
    for (int i = 0; i < n; ++i) {
        if (current_indegree[i] == 0) {
            queue[queue_head++] = i;
        }
    }

    // --- 3. Perform the randomized topological sort ---
    int* new_ordering_dense_idx = (int*)malloc(n * sizeof(int));
    int ordering_count = 0;
    unsigned int seed = time(NULL) ^ (unsigned int)omp_get_thread_num() ^ (unsigned int)(random_u64() & UINT_MAX); // Thread-specific seed

    while (queue_tail < queue_head) {
        int available_count = queue_head - queue_tail;
        if (available_count == 0) break; // Should not happen in DAG derived from valid order

        // Pick a random node from the available part of the queue
        int rand_idx_in_queue = queue_tail + (rand_r(&seed) % available_count);
        int node_dense_idx = queue[rand_idx_in_queue];

        // Swap with the element at queue_tail to remove it easily
        queue[rand_idx_in_queue] = queue[queue_tail];
        queue[queue_tail] = node_dense_idx; // Put the chosen one at the tail
        queue_tail++; // Effectively remove it

        // Add to the new ordering
        new_ordering_dense_idx[ordering_count++] = node_dense_idx;

        // Process its forward neighbors
        for (int k = 0; k < forward_degree[node_dense_idx]; ++k) {
             int neighbor_dense_idx = forward_adj[node_dense_idx][k].neighbor_dense_idx;
             current_indegree[neighbor_dense_idx]--;
             if (current_indegree[neighbor_dense_idx] == 0) {
                 queue[queue_head++] = neighbor_dense_idx; // Add to queue
             }
        }
    }

    if (ordering_count != n) {
         fprintf(stderr, "Warning: Toposort did not include all nodes (%d/%ld). Original graph might have cycles or issue with forward edge detection.\n", ordering_count, n);
         // Handle error - maybe revert to original state? For now, proceed but score will be wrong.
    }

    // --- 4. Update the SolutionInstance ---
    // Update solution array (dense indices in the new order)
    memcpy(instance->solution, new_ordering_dense_idx, n * sizeof(int));

    // Update node_to_position map
    for (int i = 0; i < n; ++i) {
        instance->node_to_position[instance->solution[i]] = i;
    }

    // --- 5. Recalculate score ---
    instance->forward_score = calculate_forward_score(instance, connectome);
    if (verbosity >= 10) {
         printf("    Thread %d: Toposort complete. New Score: %lld\n", omp_get_thread_num(), instance->forward_score);
    }


    // --- 6. Cleanup ---
    free(current_indegree);
    free(queue);
    free(new_ordering_dense_idx);
    for(int i=0; i<n; ++i) {
        if(forward_adj[i]) free(forward_adj[i]);
    }
    free(forward_adj);
    free(forward_degree);
}


// --- Calculate Delta Score ---
// Calculates the change in *forward score* if nodes at dense indices
// dense_idx_a and dense_idx_b were swapped.
// NOTE: This calculates the change assuming the swap *happens*.
// The returned value `delta` means: new_score = old_score + delta.
// A positive delta means the forward score *increases* (worse for minimization).
long long calculate_delta_score_on_swap_positions(
    const SolutionInstance* instance,
    const Connectome* connectome,
    int dense_idx_a,
    int dense_idx_b)
{
    long long delta = 0;
    int pos_a = instance->node_to_position[dense_idx_a];
    int pos_b = instance->node_to_position[dense_idx_b];

    // --- Process Neighbors of A (excluding B itself for now) ---
    // Outgoing from A: a -> x
    for (int i = 0; i < connectome->out_degree[dense_idx_a]; ++i) {
        int dense_idx_x = connectome->outgoing[dense_idx_a][i].neighbor_dense_idx;
        int weight = connectome->outgoing[dense_idx_a][i].weight;

        if (dense_idx_x == dense_idx_b) continue; // Handle a<->b separately

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was a -> x forward?
        bool was_forward = pos_a < pos_x;
        // Will a -> x be forward? (a moves to pos_b)
        bool is_forward = pos_b < pos_x;

        if (is_forward && !was_forward) {
            delta += weight;
        } else if (!is_forward && was_forward) {
            delta -= weight;
        }
    }
    // Incoming to A: x -> a
    for (int i = 0; i < connectome->in_degree[dense_idx_a]; ++i) {
        int dense_idx_x = connectome->incoming[dense_idx_a][i].neighbor_dense_idx;
        int weight = connectome->incoming[dense_idx_a][i].weight;

        if (dense_idx_x == dense_idx_b) continue; // Handle a<->b separately

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was x -> a forward?
        bool was_forward = pos_x < pos_a;
        // Will x -> a be forward? (a moves to pos_b)
        bool is_forward = pos_x < pos_b;

        if (is_forward && !was_forward) {
            delta += weight;
        } else if (!is_forward && was_forward) {
            delta -= weight;
        }
    }

    // --- Process Neighbors of B (excluding A itself) ---
    // Outgoing from B: b -> x
    for (int i = 0; i < connectome->out_degree[dense_idx_b]; ++i) {
        int dense_idx_x = connectome->outgoing[dense_idx_b][i].neighbor_dense_idx;
        int weight = connectome->outgoing[dense_idx_b][i].weight;

        if (dense_idx_x == dense_idx_a) continue; // Already covered a's perspective, or will be handled below

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was b -> x forward?
        bool was_forward = pos_b < pos_x;
        // Will b -> x be forward? (b moves to pos_a)
        bool is_forward = pos_a < pos_x;

        if (is_forward && !was_forward) {
            delta += weight;
        } else if (!is_forward && was_forward) {
            delta -= weight;
        }
    }
    // Incoming to B: x -> b
    for (int i = 0; i < connectome->in_degree[dense_idx_b]; ++i) {
        int dense_idx_x = connectome->incoming[dense_idx_b][i].neighbor_dense_idx;
        int weight = connectome->incoming[dense_idx_b][i].weight;

         if (dense_idx_x == dense_idx_a) continue; 

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was x -> b forward?
        bool was_forward = pos_x < pos_b;
        // Will x -> b be forward? (b moves to pos_a)
        bool is_forward = pos_x < pos_a;

        if (is_forward && !was_forward) {
            delta += weight;
        } else if (!is_forward && was_forward) {
            delta -= weight;
        }
    }

    // --- Handle the direct a <-> b interaction explicitly ---
    // Note: get_connection_weight uses binary search on sorted outgoing lists
    int weight_ab = get_connection_weight(connectome, dense_idx_a, dense_idx_b);
    int weight_ba = get_connection_weight(connectome, dense_idx_b, dense_idx_a);

    // Contribution of a -> b
    if (weight_ab > 0) {
        bool was_forward_ab = pos_a < pos_b;
        bool is_forward_ab = pos_b < pos_a; // New positions
        if (is_forward_ab && !was_forward_ab) delta += weight_ab;
        else if (!is_forward_ab && was_forward_ab) delta -= weight_ab;
    }

    // Contribution of b -> a
     if (weight_ba > 0) {
        bool was_forward_ba = pos_b < pos_a;
        bool is_forward_ba = pos_a < pos_b; // New positions
        if (is_forward_ba && !was_forward_ba) delta += weight_ba;
        else if (!is_forward_ba && was_forward_ba) delta -= weight_ba;
    }

    return delta;
}

// --- Main Parallel Simulated Annealing Function ---
void run_simanneal_parallel_with_toposhuffle(
    SolutionInstance* instance,      // Starting solution (will be modified in place by the best thread)
    const Connectome* connectome,
    int num_threads,
    long long iterations_per_thread,
    int updates_per_thread,        // How often each thread prints progress
    double tmin,
    double tmax,
    int go_back_to_best_window,   // Interval to reset thread state to its best
    int toposhuffle_frequency,    // How often to perform random toposort
    int verbosity,                // Logging level
    BestSolutionStorage* global_best // Tracks overall best across threads
) {
    long n = connectome->num_nodes;
    if (n <= 1) {
        printf("Warning: Graph has 0 or 1 nodes. No annealing possible.\n");
        // Ensure global_best is initialized if needed, maybe copy initial state
        if (global_best && global_best->best_solution_array == NULL) {
             init_best_solution_storage(global_best, instance);
        }
        return;
    }
    if (tmin <= 0.0) {
        fprintf(stderr, "Error: tmin must be positive.\n");
        return;
    }
    if (tmax < tmin) {
         fprintf(stderr, "Error: tmax must be >= tmin.\n");
         return;
    }


    if (verbosity >= 1) {
        printf("Starting parallel simulated annealing with toposhuffle...\n");
        printf("  Threads: %d\n", num_threads);
        printf("  Iterations/Thread: %lld\n", iterations_per_thread);
        printf("  TMax: %f, TMin: %f\n", tmax, tmin);
        printf("  Go Back Window: %d\n", go_back_to_best_window);
        printf("  Toposhuffle Freq: %d\n", toposhuffle_frequency);
        printf("  Initial Score: %lld\n", instance->forward_score);
    }

    // Ensure global_best is initialized with the starting state
    if (global_best->best_solution_array == NULL || global_best->best_score > instance->forward_score) {
         init_best_solution_storage(global_best, instance);
         if (verbosity >= 2) {
              printf("Initialized global best score to: %lld\n", global_best->best_score);
         }
    }


    // omp_set_num_threads(num_threads);

    

        long long thread_current_score = instance->forward_score;
        long long thread_best_score = thread_current_score;

        int* thread_best_solution = (int*)malloc(n * sizeof(int));
        // if (!thread_best_solution) {
        //     fprintf(stderr, "Error: Thread %d failed to allocate memory for best solution state.\n", thread_id);
        //     free_solution_instance(thread_instance); // Clean up instance
        //     #pragma omp cancellation point parallel
        // }
        memcpy(thread_best_solution, instance->solution, n * sizeof(int));

        long long trials = 0, accepts = 0, improves = 0;
        long long update_wavelength = (updates_per_thread > 0) ? (iterations_per_thread / updates_per_thread) : -1;
        if (update_wavelength == 0) update_wavelength = 1; // Avoid division by zero if iters < updates

        double tfactor = (tmax > tmin) ? -log(tmax / tmin) : 0.0; // Avoid log(1)=0 or log(<1) issues if tmax=tmin


        // if (verbosity >= 2) {
        //     printf("Thread %d starting. Initial score: %lld\n", thread_id, thread_current_score);
        // }

        if (verbosity >= 2) {
            printf("Starting with initial score: %lld\n", thread_current_score);
        }

        // --- Annealing Loop ---
        for (long long iter = 1; iter <= iterations_per_thread; ++iter) {

            // Calculate temperature for this step
            double t;
            if (tmax == tmin) {
                t = tmax;
            } else {
                t = tmax * exp(tfactor * (double)iter / (double)iterations_per_thread);
                if (t < tmin) t = tmin; // Ensure t doesn't go below tmin
            }


            // --- Select Move: Swap two positions ---
            // Using the "moveIfImpactful" logic from Go (prefer neighbors)
            int pos_a, pos_b;
            int dense_idx_a, dense_idx_b;

            pos_a = (int)(random_u32() % n);

            dense_idx_a = instance->solution[pos_a];

            int out_deg = connectome->out_degree[dense_idx_a];
            int in_deg = connectome->in_degree[dense_idx_a];
            int total_deg = out_deg + in_deg;

            if (total_deg == 0) {
                 // Node has no connections, pick a random *other* node
                 pos_b = random_u32() % (n - 1);
                 if (pos_b >= pos_a) pos_b++; // Ensure pos_b is different from pos_a
            } else {
                 // Pick a random neighbor
                 int neighbor_choice_idx = random_u32() % total_deg;
                 if (neighbor_choice_idx < out_deg) {
                     // Pick from outgoing neighbors
                     dense_idx_b = connectome->outgoing[dense_idx_a][neighbor_choice_idx].neighbor_dense_idx;
                 } else {
                     // Pick from incoming neighbors
                     dense_idx_b = connectome->incoming[dense_idx_a][neighbor_choice_idx - out_deg].neighbor_dense_idx;
                 }
                 pos_b = instance->node_to_position[dense_idx_b];
            }
            dense_idx_b = instance->solution[pos_b]; // Get b's dense index

            // Ensure a != b (can happen if neighbor logic picks itself, though unlikely)
            if (pos_a == pos_b) {
                // Fallback to purely random distinct positions
                // pos_a = rand_r(&thread_seed) % n;
                pos_a = (int)(random_u32() % n);
                // pos_b = rand_r(&thread_seed) % (n - 1);
                pos_b = (int)(random_u32() % (n - 1));
                if (pos_b >= pos_a) pos_b++;
                dense_idx_a = instance->solution[pos_a];
                dense_idx_b = instance->solution[pos_b];
            }


            // --- Calculate Score Change ---
            long long delta_score = calculate_delta_score_on_swap_positions(instance, connectome, dense_idx_a, dense_idx_b);
            trials++;

            // --- Acceptance Check ---
            // Remember: objective is MAXIMIZING forward_score.
            // delta_score > 0 is an improvement.
            // We accept if delta_score > 0 or with probability exp(delta_score / t)
            bool accepted = false;
            if (delta_score > 0) { // Improvement
                accepted = true;
                improves++;
            } else if (t > 1e-9) { // Avoid division by zero/NaN if t is tiny
                 // Accept worsening move with probability exp(delta_score / t)
                 // Note: delta_score is <= 0 here.
                 if (random_double() < exp((double)delta_score / t)) {
                     accepted = true;
                 }
            } // else: delta_score <= 0 and t is effectively 0, reject worsening move.


            // --- Apply or Reject Move ---
            if (accepted) {
                accepts++;

                // Apply the swap to the thread's instance
                instance->solution[pos_a] = dense_idx_b;
                instance->solution[pos_b] = dense_idx_a;
                instance->node_to_position[dense_idx_a] = pos_b;
                instance->node_to_position[dense_idx_b] = pos_a;

                // Update the score
                thread_current_score += delta_score;
                instance->forward_score = thread_current_score; // Keep instance score consistent

                // Check if this is the thread's best score
                if (thread_current_score > thread_best_score) {
                    thread_best_score = thread_current_score;
                    memcpy(thread_best_solution, instance->solution, n * sizeof(int));
                    // No need to update global best here yet, do it at the end or periodically with critical section
                }
            } // else: Rejected, state and score remain unchanged


            // --- Go Back to Best State Periodically ---
            if (go_back_to_best_window > 0 && iter % go_back_to_best_window == 0) {
                 if (verbosity >= 10) {
                     printf("    Iter %lld, reverting to best state (Score: %lld)\n", iter, thread_best_score);
                 }
                 memcpy(instance->solution, thread_best_solution, n * sizeof(int));
                 // Rebuild node_to_position map
                 for (int i = 0; i < n; ++i) {
                     instance->node_to_position[instance->solution[i]] = i;
                 }
                 thread_current_score = thread_best_score;
                 instance->forward_score = thread_best_score;
            }

            // --- Perform Random Toposort Periodically ---
             if (toposhuffle_frequency > 0 && iter % toposhuffle_frequency == 0) {
                 random_toposort(instance, connectome, verbosity);
                 // Toposort updates instance score, so update current score
                 thread_current_score = instance->forward_score;

                 // Check if the toposorted state is the new best for the thread
                 if (thread_current_score > thread_best_score) {
                     thread_best_score = thread_current_score;
                     memcpy(thread_best_solution, instance->solution, n * sizeof(int));
                 }
             }


             // --- Logging ---
             if (verbosity == 10 && update_wavelength > 0 && iter % update_wavelength == 0) {
                  double accept_rate = (trials > 0) ? (double)accepts / trials * 100.0 : 0.0;
                  double improve_rate = (trials > 0) ? (double)improves / trials * 100.0 : 0.0;
                  printf("Step %lld/%lld | T: %.6f | Score: %lld | Best: %lld | Accept: %.2f%% | Improve: %.2f%%\n",
                         iter, iterations_per_thread, t, thread_current_score, thread_best_score, accept_rate, improve_rate);
                  trials = accepts = improves = 0; // Reset counters for next interval
             }
             // Add omp cancellation point if needed/supported
             // #pragma omp cancellation point parallel

        } // End of annealing loop for this thread

        // --- Update Global Best ---
        // Use a critical section to safely compare and update the shared global_best
        // #pragma omp critical
        // {
        //      if (thread_best_score > global_best->best_score) {
        //          if (verbosity >= 1) {
        //              printf("Found new global best score: %lld (was %lld)\n",
        //                     thread_best_score, global_best->best_score);
        //          }
        //          global_best->best_score = thread_best_score;
        //          memcpy(global_best->best_solution_array, thread_best_solution, n * sizeof(int));
        //          // global_best->instance_id = thread_id; // Optional: track which thread found it
        //      }
        // }

        // --- Thread Cleanup ---
        // free_solution_instance(instance);
        // free(thread_best_solution);

        // if (verbosity >= 2) {
        //      printf("Thread %d finished. Best score found by thread: %lld\n", thread_id, thread_best_score);
        // }

    // } // End of OpenMP parallel region


    // --- Post-Processing ---
    // The best solution is now in global_best.
    // Copy it back to the original instance provided by the caller.
     global_best->best_score = instance->forward_score;
     memcpy(global_best->best_solution_array, instance->solution, n * sizeof(int));

    // if (global_best->best_solution_array != NULL) {
    //     memcpy(instance->solution, global_best->best_solution_array, n * sizeof(int));
    //     // Rebuild node_to_position for the final best solution
    //     for (int i = 0; i < n; ++i) {
    //         instance->node_to_position[instance->solution[i]] = i;
    //     }
    //     instance->forward_score = global_best->best_score;
    // } else {
    //     // This shouldn't happen if initialization worked, but handle defensively
    //     fprintf(stderr, "Warning: global_best solution array is NULL after parallel execution.\n");
    //     // instance remains unchanged in this case
    // }


    if (verbosity >= 1) {
        printf("Parallel annealing finished. Final best score: %lld\n", instance->forward_score);
        printf("------------------------------------------\n");
    }
}


