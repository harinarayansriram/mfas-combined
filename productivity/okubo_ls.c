#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#include "okubo_ls.h"
#include "connectome.h" 
#include "solution_instance.h"

static inline long long delta_swap_c(int dense_idx_a, int dense_idx_b, const Connectome* connectome) {
    long long weight_ba = get_connection_weight(connectome, dense_idx_b, dense_idx_a);
    long long weight_ab = get_connection_weight(connectome, dense_idx_a, dense_idx_b);
    return weight_ba - weight_ab;
}

typedef struct {
    long long* delta_values;
    int* partner_dense_idx;
    int subset_size;
    int original_node_pos_in_subset;
} InsertionDeltaResult;

static bool calculate_insertion_deltas(
    int node_to_move_dense_idx,
    const Connectome* connectome,
    int* ordered_subset_dense_indices,
    int subset_size,
    InsertionDeltaResult* delta_result
) {
    if (subset_size <= 1) {
        delta_result->subset_size = subset_size;
        delta_result->original_node_pos_in_subset = (subset_size == 1) ? 0 : -1;
        return true;
    }

    int current_pos_in_subset = -1;
    for (int i = 0; i < subset_size; ++i) {
        if (ordered_subset_dense_indices[i] == node_to_move_dense_idx) {
            current_pos_in_subset = i;
            break;
        }
    }
    if (current_pos_in_subset == -1) {
        fprintf(stderr, "Error: node_to_move not found in its own subset!\n");
        return false;
    }

    // Ensure delta_result arrays are allocated and appropriately sized
    // This logic assumes delta_result might be reused, checking capacity
    if (delta_result->delta_values == NULL || delta_result->subset_size < subset_size) {
         free(delta_result->delta_values); // free(NULL) is safe
         free(delta_result->partner_dense_idx);
         delta_result->delta_values = (long long*)malloc(subset_size * sizeof(long long));
         delta_result->partner_dense_idx = (int*)malloc(subset_size * sizeof(int));
         if (!delta_result->delta_values || !delta_result->partner_dense_idx) {
            perror("Failed to allocate memory for delta results");
            free(delta_result->delta_values); // Clean up partially allocated memory
            free(delta_result->partner_dense_idx);
            delta_result->delta_values = NULL;
            delta_result->partner_dense_idx = NULL;
            delta_result->subset_size = 0;
            return false;
         }
    }
    delta_result->subset_size = subset_size;
    delta_result->original_node_pos_in_subset = current_pos_in_subset;

    delta_result->delta_values[current_pos_in_subset] = 0;
    delta_result->partner_dense_idx[current_pos_in_subset] = node_to_move_dense_idx;

    int* temp_subset = (int*)malloc(subset_size * sizeof(int));
    if (!temp_subset) { perror("Failed to allocate temp subset"); return false; }

    // Simulate swaps to the left
    memcpy(temp_subset, ordered_subset_dense_indices, subset_size * sizeof(int));
    long long cumsum_left = 0;
    for (int j = current_pos_in_subset - 1; j >= 0; --j) {
        cumsum_left += delta_swap_c(temp_subset[j], temp_subset[j + 1], connectome);
        int swap_temp = temp_subset[j];
        temp_subset[j] = temp_subset[j + 1];
        temp_subset[j + 1] = swap_temp;
        delta_result->delta_values[j] = cumsum_left;
        delta_result->partner_dense_idx[j] = ordered_subset_dense_indices[j];
    }

    // Simulate swaps to the right
    memcpy(temp_subset, ordered_subset_dense_indices, subset_size * sizeof(int));
    long long cumsum_right = 0;
    for (int j = current_pos_in_subset; j < subset_size - 1; ++j) {
        cumsum_right += delta_swap_c(temp_subset[j], temp_subset[j + 1], connectome);
        int swap_temp = temp_subset[j];
        temp_subset[j] = temp_subset[j + 1];
        temp_subset[j + 1] = swap_temp;
        delta_result->delta_values[j + 1] = cumsum_right;
        delta_result->partner_dense_idx[j + 1] = ordered_subset_dense_indices[j + 1];
    }

    free(temp_subset);
    return true;
}

static int find_best_insertion_location(
    const InsertionDeltaResult* delta_result,
    const int* subset_positions_in_solution,
    long long* best_delta_out
) {
    long long max_diff = 0; // Improvement must be strictly positive
    int best_subset_idx = delta_result->original_node_pos_in_subset;

    for (int k = 0; k < delta_result->subset_size; ++k) {
        // We are maximizing the delta (improvement)
        if (delta_result->delta_values[k] > max_diff) {
            max_diff = delta_result->delta_values[k];
            best_subset_idx = k;
        }
    }

    *best_delta_out = max_diff;
    // Return the position in the *full* solution array corresponding to the best partner
    return subset_positions_in_solution[best_subset_idx];
}

static void perform_insertion_c(
    SolutionInstance* instance,
    int current_pos,
    int target_pos
) {
    if (current_pos == target_pos) return;

    int dense_idx_to_move = instance->solution[current_pos];
    long n = instance->solution_size;

    if (target_pos > current_pos) {
        // Move forward
        memmove(instance->solution + current_pos,
                instance->solution + current_pos + 1,
                (target_pos - current_pos) * sizeof(int));
        instance->solution[target_pos] = dense_idx_to_move;
        // Update position map
        for (int i = current_pos; i < target_pos; ++i) {
            instance->node_to_position[instance->solution[i]] = i;
        }
        instance->node_to_position[dense_idx_to_move] = target_pos;
    } else {
        // Move backward
        memmove(instance->solution + target_pos + 1,
                instance->solution + target_pos,
                (current_pos - target_pos) * sizeof(int));
        instance->solution[target_pos] = dense_idx_to_move;
        // Update position map
        instance->node_to_position[dense_idx_to_move] = target_pos;
        for (int i = target_pos + 1; i <= current_pos; ++i) {
            instance->node_to_position[instance->solution[i]] = i;
        }
    }
}

typedef struct { int dense_idx; int position; } SortItem;

int compareSortItems(const void* a, const void* b) {
    return ((SortItem*)a)->position - ((SortItem*)b)->position;
}

void run_okubo_local_search(
    SolutionInstance* instance,
    const Connectome* connectome,
    int n_epochs,
    bool log_progress,
    int log_interval
) {
    if (!instance || !connectome || n_epochs <= 0) {
        fprintf(stderr, "Invalid arguments for Okubo local search.\n");
        return;
    }

    long n_nodes = instance->solution_size;
    if (n_nodes <= 1) return; // No moves possible

    int* nodes_subset_dense_indices = (int*)malloc(n_nodes * sizeof(int));
    int* subset_positions_in_solution = (int*)malloc(n_nodes * sizeof(int));
    bool* is_in_subset = (bool*)calloc(n_nodes, sizeof(bool));
    InsertionDeltaResult delta_result = {NULL, NULL, 0, -1};
    int* node_process_order = (int*)malloc(n_nodes * sizeof(int));
    SortItem* sort_array = (SortItem*)malloc(n_nodes * sizeof(SortItem)); // Allocate max needed size once
    int* ordered_subset_dense_indices_sorted = (int*)malloc(n_nodes * sizeof(int));
    int* original_positions_sorted = (int*)malloc(n_nodes * sizeof(int));


    if (!nodes_subset_dense_indices || !subset_positions_in_solution || !is_in_subset || !node_process_order || !sort_array || !ordered_subset_dense_indices_sorted || !original_positions_sorted) {
        perror("Failed to allocate temporary structures for Okubo LS");
        free(nodes_subset_dense_indices);
        free(subset_positions_in_solution);
        free(is_in_subset);
        free(delta_result.delta_values);
        free(delta_result.partner_dense_idx);
        free(node_process_order);
        free(sort_array);
        free(ordered_subset_dense_indices_sorted);
        free(original_positions_sorted);
        return;
    }

    if (log_progress) {
        printf("Starting Okubo Local Search (Insertion Neighborhood)...\n");
        printf("  Epochs: %d, Nodes: %ld, Initial Score: %lld\n", n_epochs, n_nodes, instance->forward_score);
    }

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        long long sum_epoch_improvement = 0;
        int nodes_processed_log_interval = 0;

        for (long i = 0; i < n_nodes; ++i) node_process_order[i] = i;
        for (long i = n_nodes - 1; i > 0; --i) {
            long j = random_u64() % (i + 1);
            int temp = node_process_order[i];
            node_process_order[i] = node_process_order[j];
            node_process_order[j] = temp;
        }

        for (long node_idx = 0; node_idx < n_nodes; ++node_idx) {
            int node_to_move_dense_idx = node_process_order[node_idx];

            int current_subset_size = 0;
            nodes_subset_dense_indices[current_subset_size++] = node_to_move_dense_idx;
            is_in_subset[node_to_move_dense_idx] = true;

            for (int i = 0; i < connectome->out_degree[node_to_move_dense_idx]; ++i) {
                int neighbor_dense_idx = connectome->outgoing[node_to_move_dense_idx][i].neighbor_dense_idx;
                if (!is_in_subset[neighbor_dense_idx]) {
                     nodes_subset_dense_indices[current_subset_size++] = neighbor_dense_idx;
                     is_in_subset[neighbor_dense_idx] = true;
                }
            }
            for (int i = 0; i < connectome->in_degree[node_to_move_dense_idx]; ++i) {
                int neighbor_dense_idx = connectome->incoming[node_to_move_dense_idx][i].neighbor_dense_idx;
                 if (!is_in_subset[neighbor_dense_idx]) {
                     nodes_subset_dense_indices[current_subset_size++] = neighbor_dense_idx;
                     is_in_subset[neighbor_dense_idx] = true;
                }
            }

            for (int i = 0; i < current_subset_size; ++i) {
                 int dense_idx = nodes_subset_dense_indices[i];
                 sort_array[i].dense_idx = dense_idx;
                 sort_array[i].position = instance->node_to_position[dense_idx];
                 subset_positions_in_solution[i] = sort_array[i].position; // Keep original unsorted positions
            }

            qsort(sort_array, current_subset_size, sizeof(SortItem), compareSortItems);

            for(int i=0; i<current_subset_size; ++i) {
                ordered_subset_dense_indices_sorted[i] = sort_array[i].dense_idx;
                original_positions_sorted[i] = sort_array[i].position; // Store sorted original positions
            }


            if (!calculate_insertion_deltas(node_to_move_dense_idx, connectome,
                                          ordered_subset_dense_indices_sorted, current_subset_size,
                                          &delta_result)) {
                fprintf(stderr, "Error calculating deltas for node %d\n", node_to_move_dense_idx);
                for (int i = 0; i < current_subset_size; ++i) is_in_subset[nodes_subset_dense_indices[i]] = false; // Reset subset markers
                continue;
            }

            long long best_delta = 0;
            // Use original_positions_sorted which corresponds to the sorted subset
            int best_target_pos_in_solution = find_best_insertion_location(&delta_result, original_positions_sorted, &best_delta);

            if (best_delta > 0) {
                int current_pos_in_solution = instance->node_to_position[node_to_move_dense_idx];
                perform_insertion_c(instance, current_pos_in_solution, best_target_pos_in_solution);
                instance->forward_score += best_delta;
                sum_epoch_improvement += best_delta;
            }

            // Reset subset markers
            for (int i = 0; i < current_subset_size; ++i) {
                is_in_subset[nodes_subset_dense_indices[i]] = false;
            }

            nodes_processed_log_interval++;
            if (log_progress && log_interval > 0 && (node_idx + 1) % log_interval == 0) {
                 printf("  Epoch %d, Node %ld/%ld: Current Score = %lld (Epoch Δ = %lld)\n",
                       epoch + 1, node_idx + 1, n_nodes, instance->forward_score, sum_epoch_improvement);
                 nodes_processed_log_interval = 0;
            }

        } 

        if (log_progress) {
             printf("Epoch %d completed. Total Improvement: %lld. Current Score: %lld\n",
                    epoch + 1, sum_epoch_improvement, instance->forward_score);
        }

        if (sum_epoch_improvement == 0) {
            if (log_progress) printf("No improvement in epoch %d, stopping local search.\n", epoch + 1);
            break;
        }
    }

    // Cleanup
    free(nodes_subset_dense_indices);
    free(subset_positions_in_solution);
    free(is_in_subset);
    free(delta_result.delta_values);
    free(delta_result.partner_dense_idx);
    free(node_process_order);
    free(sort_array);
    free(ordered_subset_dense_indices_sorted);
    free(original_positions_sorted);


    if (log_progress) printf("Okubo Local Search finished. Final score: %lld\n", instance->forward_score);
}
