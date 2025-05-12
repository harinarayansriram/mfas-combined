#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#include "connectome.h"
#include "solution_instance.h"
#include "simanneal_finetune.h"

// Forward declaration for helper functions internal to this file
long long calculate_delta_score_on_swap_positions(
    const SolutionInstance* instance,
    const Connectome* connectome,
    int dense_idx_a,
    int dense_idx_b);

// --- Random Topological Sort (Toposhuffle) ---
void random_toposort(SolutionInstance* instance, const Connectome* connectome, int verbosity) {
    long n = connectome->num_nodes;
    if (n == 0) return;

    if (verbosity >= 2) {
        printf("Performing random toposort...\n");
    }

    int* current_indegree = (int*)calloc(n, sizeof(int));
    ConnectionNeighbor** forward_adj = (ConnectionNeighbor**)calloc(n, sizeof(ConnectionNeighbor*));
    int* forward_degree = (int*)calloc(n, sizeof(int));

    if (!current_indegree || !forward_adj || !forward_degree) {
        fprintf(stderr, "Error: Failed to allocate memory for toposort structures.\n");
        if (current_indegree) free(current_indegree);
        if (forward_adj) free(forward_adj);
        if (forward_degree) free(forward_degree);
        return;
    }

    // First pass: Count forward degrees
    for (int i_dense = 0; i_dense < n; ++i_dense) {
        int i_pos = instance->node_to_position[i_dense];
        for (int k = 0; k < connectome->out_degree[i_dense]; ++k) {
            int neighbor_dense_idx = connectome->outgoing[i_dense][k].neighbor_dense_idx;
            int neighbor_pos = instance->node_to_position[neighbor_dense_idx];
            if (i_pos < neighbor_pos) {
                forward_degree[i_dense]++;
            }
        }
        if (forward_degree[i_dense] > 0) {
            forward_adj[i_dense] = (ConnectionNeighbor*)malloc(forward_degree[i_dense] * sizeof(ConnectionNeighbor));
            if (!forward_adj[i_dense]) {
                 fprintf(stderr, "Error: Failed to allocate memory for forward adjacency list.\n");
                 free(current_indegree);
                 free(forward_degree);
                 for(int j=0; j<i_dense; ++j) if(forward_adj[j]) free(forward_adj[j]);
                 free(forward_adj);
                 return;
            }
        }
    }

    // Second pass: Populate forward_adj and calculate initial in-degrees
    memset(forward_degree, 0, n * sizeof(int)); // Reset counts
    for (int i_dense = 0; i_dense < n; ++i_dense) {
        int i_pos = instance->node_to_position[i_dense];
        for (int k = 0; k < connectome->out_degree[i_dense]; ++k) {
            int neighbor_dense_idx = connectome->outgoing[i_dense][k].neighbor_dense_idx;
            int weight = connectome->outgoing[i_dense][k].weight;
            int neighbor_pos = instance->node_to_position[neighbor_dense_idx];

            if (i_pos < neighbor_pos) { // Forward edge
                current_indegree[neighbor_dense_idx]++;
                forward_adj[i_dense][forward_degree[i_dense]].neighbor_dense_idx = neighbor_dense_idx;
                forward_adj[i_dense][forward_degree[i_dense]].weight = weight;
                forward_degree[i_dense]++;
            }
        }
    }

    // --- 2. Initialize queue ---
    int* queue = (int*)malloc(n * sizeof(int));
    int queue_head = 0;
    int queue_tail = 0;
    for (int i = 0; i < n; ++i) {
        if (current_indegree[i] == 0) {
            queue[queue_head++] = i;
        }
    }

    // --- 3. Perform the randomized topological sort ---
    int* new_ordering_dense_idx = (int*)malloc(n * sizeof(int));
    int ordering_count = 0;

    while (queue_tail < queue_head) {
        int available_count = queue_head - queue_tail;
        if (available_count == 0) break;

        int rand_idx_in_queue = queue_tail + (random_u32() % available_count); // Use random_u32
        int node_dense_idx = queue[rand_idx_in_queue];

        queue[rand_idx_in_queue] = queue[queue_tail];
        queue[queue_tail] = node_dense_idx;
        queue_tail++;

        new_ordering_dense_idx[ordering_count++] = node_dense_idx;

        for (int k = 0; k < forward_degree[node_dense_idx]; ++k) {
             int neighbor_dense_idx = forward_adj[node_dense_idx][k].neighbor_dense_idx;
             current_indegree[neighbor_dense_idx]--;
             if (current_indegree[neighbor_dense_idx] == 0) {
                 queue[queue_head++] = neighbor_dense_idx;
             }
        }
    }

    if (ordering_count != n) {
         fprintf(stderr, "Warning: Toposort did not include all nodes (%d/%ld).\n", ordering_count, n);
    }

    // --- 4. Update the SolutionInstance ---
    memcpy(instance->solution, new_ordering_dense_idx, n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        instance->node_to_position[instance->solution[i]] = i;
    }

    // --- 5. Recalculate score ---
    instance->forward_score = calculate_forward_score(instance, connectome);
    if (verbosity >= 10) {
         printf("    Toposort complete. New Score: %lld\n", instance->forward_score);
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
long long calculate_delta_score_on_swap_positions(
    const SolutionInstance* instance,
    const Connectome* connectome,
    int dense_idx_a,
    int dense_idx_b)
{
    long long delta = 0;
    int pos_a = instance->node_to_position[dense_idx_a];
    int pos_b = instance->node_to_position[dense_idx_b];

    // Process Neighbors of A
    for (int i = 0; i < connectome->out_degree[dense_idx_a]; ++i) {
        int dense_idx_x = connectome->outgoing[dense_idx_a][i].neighbor_dense_idx;
        int weight = connectome->outgoing[dense_idx_a][i].weight;
        if (dense_idx_x == dense_idx_b) continue;
        int pos_x = instance->node_to_position[dense_idx_x];
        bool was_forward = pos_a < pos_x;
        bool is_forward = pos_b < pos_x;
        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }
    for (int i = 0; i < connectome->in_degree[dense_idx_a]; ++i) {
        int dense_idx_x = connectome->incoming[dense_idx_a][i].neighbor_dense_idx;
        int weight = connectome->incoming[dense_idx_a][i].weight;
        if (dense_idx_x == dense_idx_b) continue;
        int pos_x = instance->node_to_position[dense_idx_x];
        bool was_forward = pos_x < pos_a;
        bool is_forward = pos_x < pos_b;
        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }

    // Process Neighbors of B
    for (int i = 0; i < connectome->out_degree[dense_idx_b]; ++i) {
        int dense_idx_x = connectome->outgoing[dense_idx_b][i].neighbor_dense_idx;
        int weight = connectome->outgoing[dense_idx_b][i].weight;
        if (dense_idx_x == dense_idx_a) continue;
        int pos_x = instance->node_to_position[dense_idx_x];
        bool was_forward = pos_b < pos_x;
        bool is_forward = pos_a < pos_x;
        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }
    for (int i = 0; i < connectome->in_degree[dense_idx_b]; ++i) {
        int dense_idx_x = connectome->incoming[dense_idx_b][i].neighbor_dense_idx;
        int weight = connectome->incoming[dense_idx_b][i].weight;
         if (dense_idx_x == dense_idx_a) continue;
        int pos_x = instance->node_to_position[dense_idx_x];
        bool was_forward = pos_x < pos_b;
        bool is_forward = pos_x < pos_a;
        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }

    // Handle direct a <-> b interaction
    int weight_ab = get_connection_weight(connectome, dense_idx_a, dense_idx_b);
    int weight_ba = get_connection_weight(connectome, dense_idx_b, dense_idx_a);

    if (weight_ab > 0) {
        bool was_forward_ab = pos_a < pos_b;
        bool is_forward_ab = pos_b < pos_a;
        if (is_forward_ab && !was_forward_ab) delta += weight_ab;
        else if (!is_forward_ab && was_forward_ab) delta -= weight_ab;
    }
     if (weight_ba > 0) {
        bool was_forward_ba = pos_b < pos_a;
        bool is_forward_ba = pos_a < pos_b;
        if (is_forward_ba && !was_forward_ba) delta += weight_ba;
        else if (!is_forward_ba && was_forward_ba) delta -= weight_ba;
    }

    return delta;
}

// --- Main Simulated Annealing Function (Serial) ---
void run_simanneal_with_toposhuffle( 
    SolutionInstance* instance,      // Starting solution (modified in place)
    const Connectome* connectome,
    long long total_iterations,    
    int updates_frequency,         
    double tmin,
    double tmax,
    int go_back_to_best_window,   // Interval to reset state to its best
    int toposhuffle_frequency,    // How often to perform a toposhuffle
    int verbosity,                // Logging level
    BestSolutionStorage* global_best // Tracks overall best (can be used across multiple calls)
) {
    long n = connectome->num_nodes;
    if (n <= 1) {
        printf("Warning: Graph has 0 or 1 nodes. No annealing possible.\n");
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
        printf("Starting simulated annealing with toposhuffle...\n");
        printf("  Iterations: %lld\n", total_iterations);
        printf("  TMax: %f, TMin: %f\n", tmax, tmin);
        printf("  Go Back Window: %d\n", go_back_to_best_window);
        printf("  Toposhuffle Freq: %d\n", toposhuffle_frequency);
        printf("  Initial Score: %lld\n", instance->forward_score);
    }

    // Initialize global_best if it's worse than the starting instance
    // or if it's uninitialized.
    if (global_best) {
        if (global_best->best_solution_array == NULL || global_best->best_score > instance->forward_score) {
            init_best_solution_storage(global_best, instance);
            if (verbosity >= 2) {
                printf("Initialized global best score to: %lld\n", global_best->best_score);
            }
        }
    } else {
        fprintf(stderr, "Warning: global_best is NULL. Cannot track best solution across calls.\n");
    }

    // Local tracking for "go back to best" logic within this run
    long long current_score = instance->forward_score;
    long long best_score_this_run = current_score;
    int* best_solution_this_run = (int*)malloc(n * sizeof(int));
    if (!best_solution_this_run) {
        fprintf(stderr, "Error: Failed to allocate memory for best solution state.\n");
        return; // Exit if memory allocation fails
    }
    memcpy(best_solution_this_run, instance->solution, n * sizeof(int));

    long long trials = 0, accepts = 0, improves = 0;
    long long update_wavelength = (updates_frequency > 0) ? (total_iterations / updates_frequency) : -1;
    if (update_wavelength == 0) update_wavelength = 1;

    double tfactor = (tmax > tmin) ? -log(tmax / tmin) : 0.0;

    if (verbosity >= 2) {
        printf("Starting annealing. Initial score: %lld\n", current_score);
    }

    // --- Annealing Loop ---
    for (long long iter = 1; iter <= total_iterations; ++iter) {

        // Calculate temperature
        double t;
        if (tmax == tmin) {
            t = tmax;
        } else {
            t = tmax * exp(tfactor * (double)iter / (double)total_iterations);
            if (t < tmin) t = tmin;
        }

        // --- Select Move: Swap two positions ---
        int pos_a, pos_b;
        int dense_idx_a, dense_idx_b;

        pos_a = (int)(random_u32() % n);
        dense_idx_a = instance->solution[pos_a];

        int out_deg = connectome->out_degree[dense_idx_a];
        int in_deg = connectome->in_degree[dense_idx_a];
        int total_deg = out_deg + in_deg;

        if (total_deg == 0 || n <= 1) { // Added n<=1 check for safety
             if (n <= 1) {
                // Cannot select a different node, skip iteration? Or handle earlier.
                // This case should ideally be caught by the initial n<=1 check.
                continue;
             }
             // Node has no connections, pick a random *other* node
             pos_b = (int)(random_u32() % (n - 1));
             if (pos_b >= pos_a) pos_b++;
        } else {
             // Pick a random neighbor
             // int neighbor_choice_idx = rand_r(&thread_seed) % total_deg;
             int neighbor_choice_idx = random_u32() % total_deg;
             if (neighbor_choice_idx < out_deg) {
                 dense_idx_b = connectome->outgoing[dense_idx_a][neighbor_choice_idx].neighbor_dense_idx;
             } else {
                 dense_idx_b = connectome->incoming[dense_idx_a][neighbor_choice_idx - out_deg].neighbor_dense_idx;
             }
             pos_b = instance->node_to_position[dense_idx_b];
        }
        dense_idx_b = instance->solution[pos_b];

        if (pos_a == pos_b) {
             if (n <= 1) continue; // Cannot swap if only one node
            // Fallback to purely random distinct positions
            pos_a = (int)(random_u32() % n);
            pos_b = (int)(random_u32() % (n - 1));
            if (pos_b >= pos_a) pos_b++;
            dense_idx_a = instance->solution[pos_a];
            dense_idx_b = instance->solution[pos_b];
        }


        // --- Calculate Score Change ---
        // Use the current state of 'instance'
        long long delta_score = calculate_delta_score_on_swap_positions(instance, connectome, dense_idx_a, dense_idx_b);
        trials++;

        // --- Acceptance Check ---
        bool accepted = false;
        if (delta_score > 0) { // Improvement
            accepted = true;
            improves++;
        } else if (t > 1e-9) {
             if (random_double() < exp((double)delta_score / t)) {
                 accepted = true;
             }
        }

        // --- Apply or Reject Move ---
        if (accepted) {
            accepts++;

            // Apply the swap directly to the input 'instance'
            instance->solution[pos_a] = dense_idx_b;
            instance->solution[pos_b] = dense_idx_a;
            instance->node_to_position[dense_idx_a] = pos_b;
            instance->node_to_position[dense_idx_b] = pos_a;

            // Update the score
            current_score += delta_score;
            instance->forward_score = current_score; // Keep instance score consistent

            // Check if this is the best score found in *this run*
            if (current_score > best_score_this_run) {
                best_score_this_run = current_score;
                memcpy(best_solution_this_run, instance->solution, n * sizeof(int));

                // Optionally update global_best immediately if desired,
                // but typically done at the end.
                // if (global_best && best_score_this_run > global_best->best_score) { ... }
            }
        } // else: Rejected, instance state and current_score remain unchanged


        // --- Go Back to Best State Periodically ---
        if (go_back_to_best_window > 0 && iter % go_back_to_best_window == 0) {
             if (verbosity >= 10) {
                 printf("    Iter %lld, reverting to best state (Score: %lld)\n", iter, best_score_this_run);
             }
             // Restore instance state from the best state found *in this run*
             memcpy(instance->solution, best_solution_this_run, n * sizeof(int));
             for (int i = 0; i < n; ++i) {
                 instance->node_to_position[instance->solution[i]] = i;
             }
             current_score = best_score_this_run;
             instance->forward_score = best_score_this_run; // Ensure instance score matches
        }

        // --- Perform Random Toposort Periodically ---
         if (toposhuffle_frequency > 0 && iter % toposhuffle_frequency == 0) {
             // Pass the current instance directly
             random_toposort(instance, connectome, verbosity);
             // Toposort updates instance score, so update current score tracker
             current_score = instance->forward_score;

             // Check if the toposorted state is the new best for *this run*
             if (current_score > best_score_this_run) {
                 best_score_this_run = current_score;
                 memcpy(best_solution_this_run, instance->solution, n * sizeof(int));
             }
         }


         // --- Logging ---
         if (verbosity >= 1 && update_wavelength > 0 && iter % update_wavelength == 0) {
              double accept_rate = (trials > 0) ? (double)accepts / trials * 100.0 : 0.0;
              double improve_rate = (trials > 0) ? (double)improves / trials * 100.0 : 0.0;
              
              printf("Step %lld/%lld | T: %.6f | Score: %lld | Best: %lld | Accept: %.2f%% | Improve: %.2f%%\n",
                     iter, total_iterations, t, current_score, best_score_this_run, accept_rate, improve_rate);
              trials = accepts = improves = 0; // Reset counters for next interval
         }

    } // End of annealing loop

    // --- Post-Processing ---
    // Update the input instance to reflect the best state found *during this run*.
    memcpy(instance->solution, best_solution_this_run, n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        instance->node_to_position[instance->solution[i]] = i;
    }
    instance->forward_score = best_score_this_run;

    // Update the global_best structure if the result of this run is better.
    if (global_best) {
        if (best_score_this_run > global_best->best_score) {
             if (verbosity >= 1) {
                 printf("Found new global best score: %lld (was %lld)\n",
                        best_score_this_run, global_best->best_score);
             }
             global_best->best_score = best_score_this_run;
             // Ensure global_best has allocated memory
             if (global_best->best_solution_array == NULL) {
                 global_best->best_solution_array = (int*)malloc(n * sizeof(int));
                 global_best->solution_size = n; // Assuming size matches n
             }
             // Check allocation success before copying
             if (global_best->best_solution_array) {
                 memcpy(global_best->best_solution_array, best_solution_this_run, n * sizeof(int));
             } else {
                 fprintf(stderr, "Error: Failed to allocate memory for global_best->best_solution_array.\n");
             }
        }
    }

    // --- Cleanup for this run ---
    free(best_solution_this_run);

    if (verbosity >= 1) {
        printf("Annealing finished. Final score for this run: %lld\n", instance->forward_score);
        if (global_best) {
             printf("Current overall best score: %lld\n", global_best->best_score);
        }
        printf("------------------------------------------\n");
    }
}