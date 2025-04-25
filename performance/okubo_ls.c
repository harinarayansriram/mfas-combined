#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#include "okubo_ls.h"
#include "connectome.h" // Includes random_u64
#include "solution_instance.h"

// --- Helper Functions (Ported/Adapted from utils.py) ---

// Helper to calculate delta score for swapping two adjacent nodes in a sequence
// Equivalent to delta_swap, but operates on dense indices directly.
static inline long long delta_swap_c(int dense_idx_a, int dense_idx_b, const Connectome* connectome) {
    long long weight_ba = get_connection_weight(connectome, dense_idx_b, dense_idx_a);
    long long weight_ab = get_connection_weight(connectome, dense_idx_a, dense_idx_b);
    return weight_ba - weight_ab;
}

// Structure to hold results from calculating insertion deltas
typedef struct {
    long long* delta_values;       // Score change for inserting at the position BEFORE the partner
    int* partner_dense_idx;        // The dense index of the node that would be AFTER the inserted node
    int subset_size;               // Size of the subset considered
    int original_node_pos_in_subset; // Index of the node_to_move within ordered_subset_dense_indices
} InsertionDeltaResult;

// Calculate insertion score differences using sequential swaps.
// Equivalent to create_delta combined with swap_left/swap_right logic.
// Returns results in the pre-allocated delta_result structure.
static bool calculate_insertion_deltas(
    int node_to_move_dense_idx,      // The node we are considering moving
    const SolutionInstance* instance,
    const Connectome* connectome,
    int* ordered_subset_dense_indices, // Pre-filled array: node_to_move and neighbors, ordered by current solution position
    int subset_size,                 // Size of the subset array
    InsertionDeltaResult* delta_result // Structure to store results
) {
    if (subset_size <= 1) {
        delta_result->subset_size = subset_size;
        return true; // Nothing to calculate
    }

    // Find the position of node_to_move within the ordered subset
    int current_pos_in_subset = -1;
    for (int i = 0; i < subset_size; ++i) {
        if (ordered_subset_dense_indices[i] == node_to_move_dense_idx) {
            current_pos_in_subset = i;
            break;
        }
    }
    if (current_pos_in_subset == -1) {
        fprintf(stderr, "Error: node_to_move not found in its own subset!\n");
        return false; // Should not happen
    }

    // Allocate or ensure delta_result arrays are large enough
    if (delta_result->delta_values == NULL || delta_result->subset_size < subset_size) {
         free(delta_result->delta_values);
         free(delta_result->partner_dense_idx);
         delta_result->delta_values = (long long*)malloc(subset_size * sizeof(long long));
         delta_result->partner_dense_idx = (int*)malloc(subset_size * sizeof(int));
         if (!delta_result->delta_values || !delta_result->partner_dense_idx) {
             perror("Failed to allocate memory for delta results");
             delta_result->subset_size = 0;
             return false;
         }
    }
    delta_result->subset_size = subset_size;
    delta_result->original_node_pos_in_subset = current_pos_in_subset;

    // Initialize: The delta for inserting at its original position is 0
    delta_result->delta_values[current_pos_in_subset] = 0;
    delta_result->partner_dense_idx[current_pos_in_subset] = node_to_move_dense_idx; // Self-partner

    // --- Simulate swaps to the left ---
    // Create a temporary copy of the subset to simulate swaps on
    int* temp_subset = (int*)malloc(subset_size * sizeof(int));
    if (!temp_subset) { perror("Failed to allocate temp subset"); return false; }
    memcpy(temp_subset, ordered_subset_dense_indices, subset_size * sizeof(int));

    long long cumsum_left = 0;
    for (int j = current_pos_in_subset - 1; j >= 0; --j) {
        // Calculate delta for swapping temp_subset[j] and temp_subset[j+1] (which is the node being moved)
        cumsum_left += delta_swap_c(temp_subset[j], temp_subset[j + 1], connectome);

        // Simulate the swap in the temporary array
        int swap_temp = temp_subset[j];
        temp_subset[j] = temp_subset[j + 1];
        temp_subset[j + 1] = swap_temp;

        // Store the result: cumsum_left is the delta if we insert node_to_move *before* original partner at index j
        delta_result->delta_values[j] = cumsum_left;
        delta_result->partner_dense_idx[j] = ordered_subset_dense_indices[j]; // Original node at this position
    }

    // --- Simulate swaps to the right ---
    // Restore the temporary subset to the original order before swapping right
    memcpy(temp_subset, ordered_subset_dense_indices, subset_size * sizeof(int));
    long long cumsum_right = 0;
    for (int j = current_pos_in_subset; j < subset_size - 1; ++j) {
         // Calculate delta for swapping temp_subset[j] (the node being moved) and temp_subset[j+1]
        cumsum_right += delta_swap_c(temp_subset[j], temp_subset[j + 1], connectome);

         // Simulate the swap
        int swap_temp = temp_subset[j];
        temp_subset[j] = temp_subset[j + 1];
        temp_subset[j + 1] = swap_temp;

        // Store the result: cumsum_right is the delta if we insert node_to_move *after* the original partner at index j+1
        delta_result->delta_values[j + 1] = cumsum_right;
        delta_result->partner_dense_idx[j + 1] = ordered_subset_dense_indices[j + 1]; // Original node at this position
    }

    free(temp_subset);
    return true;
}


// Finds the best insertion location based on calculated deltas.
// Returns the position in the *original* solution array where the node should be inserted.
static int find_best_insertion_location(
    const InsertionDeltaResult* delta_result,
    const int* subset_positions_in_solution, // Original solution positions corresponding to the subset
    long long* best_delta_out             // Output: the score difference for the best move
) {
    long long max_diff = 0; // We only move if diff > 0
    int best_subset_idx = delta_result->original_node_pos_in_subset; // Default to original position (no move)

    for (int k = 0; k < delta_result->subset_size; ++k) {
        if (delta_result->delta_values[k] > max_diff) {
            max_diff = delta_result->delta_values[k];
            best_subset_idx = k;
        }
    }

    *best_delta_out = max_diff;

    // Return the position in the *full* solution array corresponding to the best partner found
    return subset_positions_in_solution[best_subset_idx];
}

// Performs the insertion efficiently in the solution array and updates the position map.
// Equivalent to insert function.
static void perform_insertion_c(
    SolutionInstance* instance,
    int current_pos,    // Current position of the node to move
    int target_pos      // Target position for the node to move
) {
    if (current_pos == target_pos) return; // No move needed

    int dense_idx_to_move = instance->solution[current_pos];
    long n = instance->solution_size;

    if (target_pos > current_pos) {
        // Move node forward: Shift elements [current_pos+1, target_pos] one step left
        // Use memmove for overlapping regions
        memmove(instance->solution + current_pos,           // Destination
                instance->solution + current_pos + 1,       // Source
                (target_pos - current_pos) * sizeof(int)); // Number of bytes
        instance->solution[target_pos] = dense_idx_to_move;

        // Update position map for shifted elements + the moved element
        for (int i = current_pos; i < target_pos; ++i) {
            instance->node_to_position[instance->solution[i]] = i;
        }
        instance->node_to_position[dense_idx_to_move] = target_pos;

    } else { // target_pos < current_pos
        // Move node backward: Shift elements [target_pos, current_pos-1] one step right
        memmove(instance->solution + target_pos + 1,     // Destination
                instance->solution + target_pos,         // Source
                (current_pos - target_pos) * sizeof(int)); // Number of bytes
        instance->solution[target_pos] = dense_idx_to_move;

        // Update position map for shifted elements + the moved element
        instance->node_to_position[dense_idx_to_move] = target_pos;
        for (int i = target_pos + 1; i <= current_pos; ++i) {
            instance->node_to_position[instance->solution[i]] = i;
        }
    }
}

// Let's use a temporary struct for sorting.
typedef struct { int dense_idx; int position; } SortItem;

// Comparison function for qsort
int compareSortItems(const void* a, const void* b) {
    return ((SortItem*)a)->position - ((SortItem*)b)->position;
}

// --- Main Local Search Function ---

void run_okubo_local_search(
    SolutionInstance* instance,
    const Connectome* connectome,
    int n_epochs,
    bool log_progress,
    int log_interval // Log approx every N node considerations
) {
    if (!instance || !connectome || n_epochs <= 0) {
        fprintf(stderr, "Invalid arguments for Okubo local search.\n");
        return;
    }

    long n_nodes = instance->solution_size;
    if (n_nodes == 0) return;

    // --- Pre-allocate temporary structures to reuse ---
    // Max possible subset size is max degree + 1, but use n_nodes as safe upper bound
    int* nodes_subset_dense_indices = (int*)malloc(n_nodes * sizeof(int));
    int* subset_positions_in_solution = (int*)malloc(n_nodes * sizeof(int));
    bool* is_in_subset = (bool*)calloc(n_nodes, sizeof(bool)); // For quick neighbor lookup
    InsertionDeltaResult delta_result = {NULL, NULL, 0, -1}; // Will be allocated inside helper

    if (!nodes_subset_dense_indices || !subset_positions_in_solution || !is_in_subset) {
        perror("Failed to allocate temporary structures for Okubo LS");
        // Free any potentially allocated parts
        free(nodes_subset_dense_indices);
        free(subset_positions_in_solution);
        free(is_in_subset);
        free(delta_result.delta_values);
        free(delta_result.partner_dense_idx);
        return;
    }


    printf("Starting Okubo Local Search (Insertion Neighborhood)...\n");
    printf("  Epochs: %d, Nodes: %ld\n", n_epochs, n_nodes);

    // --- Main Epoch Loop ---
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        long long sum_epoch_improvement = 0;
        int nodes_processed_log_interval = 0;

        // Create a random permutation of node DENSE indices to process
        int* node_process_order = (int*)malloc(n_nodes * sizeof(int));
        if (!node_process_order) { perror("Failed to allocate node order"); break; }
        for (long i = 0; i < n_nodes; ++i) {
            node_process_order[i] = i; // Fill with 0, 1, ..., n_nodes-1
        }
        // Fisher-Yates shuffle (ensure srand was called externally)
        for (long i = n_nodes - 1; i > 0; --i) {
            long j = random_u64() % (i + 1);
            int temp = node_process_order[i];
            node_process_order[i] = node_process_order[j];
            node_process_order[j] = temp;
        }

        // --- Inner Loop: Iterate through nodes in random order ---
        for (long node_idx = 0; node_idx < n_nodes; ++node_idx) {
            int node_to_move_dense_idx = node_process_order[node_idx];

            // 1. Extract relevant subgraph (neighbors + node itself)
            int current_subset_size = 0;
            nodes_subset_dense_indices[current_subset_size++] = node_to_move_dense_idx;
            is_in_subset[node_to_move_dense_idx] = true; // Mark node itself

            // Add outgoing neighbors
            if (connectome->outgoing[node_to_move_dense_idx]) {
                for (int i = 0; i < connectome->out_degree[node_to_move_dense_idx]; ++i) {
                    int neighbor_dense_idx = connectome->outgoing[node_to_move_dense_idx][i].neighbor_dense_idx;
                    if (!is_in_subset[neighbor_dense_idx]) {
                         nodes_subset_dense_indices[current_subset_size++] = neighbor_dense_idx;
                         is_in_subset[neighbor_dense_idx] = true;
                    }
                }
            }
            // Add incoming neighbors
             if (connectome->incoming[node_to_move_dense_idx]) {
                for (int i = 0; i < connectome->in_degree[node_to_move_dense_idx]; ++i) {
                    int neighbor_dense_idx = connectome->incoming[node_to_move_dense_idx][i].neighbor_dense_idx;
                     if (!is_in_subset[neighbor_dense_idx]) {
                         nodes_subset_dense_indices[current_subset_size++] = neighbor_dense_idx;
                         is_in_subset[neighbor_dense_idx] = true;
                    }
                }
            }

            // 2. Get current positions and sort the subset by position
            for (int i = 0; i < current_subset_size; ++i) {
                subset_positions_in_solution[i] = instance->node_to_position[nodes_subset_dense_indices[i]];
            }

            // We need the subset dense indices sorted by their current position in the solution
            // Simple bubble sort or insertion sort is fine for small subsets, or qsort.
            SortItem* sort_array = (SortItem*)malloc(current_subset_size * sizeof(SortItem));
             if (!sort_array) { perror("Failed to allocate sort array"); continue; } // Skip node
            for(int i=0; i<current_subset_size; ++i) {
                sort_array[i].dense_idx = nodes_subset_dense_indices[i];
                sort_array[i].position = subset_positions_in_solution[i];
            }
            
            
            qsort(sort_array, current_subset_size, sizeof(SortItem), compareSortItems);

            // Extract the sorted dense indices and their original positions
            int* ordered_subset_dense_indices_sorted = (int*)malloc(current_subset_size * sizeof(int));
            int* original_positions_sorted = (int*)malloc(current_subset_size * sizeof(int));
             if (!ordered_subset_dense_indices_sorted || !original_positions_sorted) {
                 perror("Failed to allocate sorted subset arrays");
                 free(sort_array); free(ordered_subset_dense_indices_sorted); free(original_positions_sorted);
                 continue; // Skip node
             }
            for(int i=0; i<current_subset_size; ++i) {
                ordered_subset_dense_indices_sorted[i] = sort_array[i].dense_idx;
                original_positions_sorted[i] = sort_array[i].position;
            }
            free(sort_array);


            // 3. Calculate deltas for inserting node_to_move within this ordered subset
            if (!calculate_insertion_deltas(node_to_move_dense_idx, instance, connectome,
                                          ordered_subset_dense_indices_sorted, current_subset_size,
                                          &delta_result)) {
                fprintf(stderr, "Error calculating deltas for node %d\n", node_to_move_dense_idx);
                 free(ordered_subset_dense_indices_sorted); free(original_positions_sorted);
                continue; // Skip node
            }


            // 4. Find the best insertion location
            long long best_delta = 0;
            int best_target_pos_in_solution = find_best_insertion_location(&delta_result,
                                                                      original_positions_sorted,
                                                                      &best_delta);

            // 5. Perform insertion if improvement is found (best_delta > 0)
            if (best_delta > 0) {
                int current_pos_in_solution = instance->node_to_position[node_to_move_dense_idx];

                // Determine the insertion index 'j' based on Schiavinotto & Stutzle definition
                // If inserting before target: j = target_pos
                // If inserting after target: j = target_pos
                // Our helper needs the final index where the node *will be*.
                int final_insertion_idx = -1;
                if (best_target_pos_in_solution > current_pos_in_solution) {
                     final_insertion_idx = best_target_pos_in_solution; // Insert will place it AT target_pos (shifting others left)
                } else { // best_target_pos_in_solution <= current_pos_in_solution
                    final_insertion_idx = best_target_pos_in_solution; // Insert will place it AT target_pos (shifting others right)
                }

                 // Adjust if inserting relative to self (no move)
                 if (final_insertion_idx == current_pos_in_solution && best_delta > 0) {
                     // This case implies the best "partner" was itself, but delta > 0? Should not happen if delta for self is 0.
                     // Or maybe it means insert just before/after self? Let's stick to the calculated best_target_pos_in_solution.
                 }

                perform_insertion_c(instance, current_pos_in_solution, final_insertion_idx);

                // Update score (important!)
                instance->forward_score += best_delta;
                sum_epoch_improvement += best_delta;

                // // Verification (optional, disable for performance)
                // long long check_score = calculate_forward_score(instance, connectome);
                // if (check_score != instance->forward_score) {
                //     fprintf(stderr, "Score mismatch after Okubo insertion! Node: %d, Delta: %lld, Expected: %lld, Actual: %lld\n",
                //             node_to_move_dense_idx, best_delta, instance->forward_score, check_score);
                //      instance->forward_score = check_score; // Correct it
                //     // exit(1);
                // }
            }

            // Cleanup for next node
            free(ordered_subset_dense_indices_sorted);
            free(original_positions_sorted);
            // Reset the boolean marker array for the next node
            for (int i = 0; i < current_subset_size; ++i) {
                is_in_subset[nodes_subset_dense_indices[i]] = false;
            }

            // --- Logging ---
            nodes_processed_log_interval++;
            if (log_progress && log_interval > 0 && nodes_processed_log_interval >= log_interval) {
                 printf("  Epoch %d, Node %ld/%ld: Current Score = %lld (Epoch Δ = %lld)\n",
                       epoch + 1, node_idx + 1, n_nodes, instance->forward_score, sum_epoch_improvement);
                 nodes_processed_log_interval = 0;
            }


        } // End inner loop (processing nodes)

        free(node_process_order);

        if (log_progress) {
             printf("Epoch %d completed. Total Improvement this Epoch: %lld. Final Score: %lld\n",
                    epoch + 1, sum_epoch_improvement, instance->forward_score);
        }

        // Exit loop if no improvement was made in the entire epoch
        if (sum_epoch_improvement == 0) {
            if (log_progress) printf("No improvement in epoch %d, stopping local search.\n", epoch + 1);
            break;
        }

    } // End outer loop (epochs)

    // --- Cleanup ---
    free(nodes_subset_dense_indices);
    free(subset_positions_in_solution);
    free(is_in_subset);
    free(delta_result.delta_values);
    free(delta_result.partner_dense_idx);

    printf("Okubo Local Search finished. Final score: %lld\n", instance->forward_score);
}