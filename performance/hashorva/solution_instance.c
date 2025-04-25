#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include "solution_instance.h" // Includes connectome.h

// Helper to copy solution array
void copy_solution(int* dest, const int* src, long size) {
    memcpy(dest, src, size * sizeof(int));
}

// Create a SolutionInstance
SolutionInstance* create_solution_instance(const Connectome* connectome, int* initial_solution_dense_array, bool calculate_initial_score) {
    if (!connectome) return NULL;

    SolutionInstance* instance = (SolutionInstance*)malloc(sizeof(SolutionInstance));
    if (!instance) {
        perror("Failed to allocate SolutionInstance");
        return NULL;
    }

    instance->solution_size = connectome->num_nodes; // Use the full size for indexing
    instance->solution = (int*)malloc(instance->solution_size * sizeof(int));
    instance->node_to_position = (int*)malloc(instance->solution_size * sizeof(int));

    if (!instance->solution || !instance->node_to_position) {
        perror("Failed to allocate solution arrays");
        free(instance->solution);
        free(instance->node_to_position);
        free(instance);
        return NULL;
    }

    if (initial_solution_dense_array) {
        copy_solution(instance->solution, initial_solution_dense_array, instance->solution_size);
    } else {
        // Default: Initialize with identity permutation of dense indices (0, 1, 2, ...)
        for (int i = 0; i < instance->solution_size; i++) {
            instance->solution[i] = i;
        }
    }

    // Initialize node_to_position lookup
    for (int i = 0; i < instance->solution_size; i++) {
        instance->node_to_position[instance->solution[i]] = i;
    }

    // Calculate initial score if requested
    if (calculate_initial_score) {
        instance->forward_score = calculate_forward_score(instance, connectome);
    } else {
        instance->forward_score = 0; // Or some sentinel value like -1
    }

    // instance->instance_id = 0; // Set if needed

    return instance;
}

// Create a SolutionInstance with a random permutation
SolutionInstance* create_random_solution_instance(const Connectome* connectome) {
    if (!connectome) return NULL;

    long size = connectome->num_nodes;
    int* initial_solution = (int*)malloc(size * sizeof(int));
    if (!initial_solution) {
        perror("Failed to allocate temporary solution for randomization");
        return NULL;
    }

    // Initialize with identity permutation
    for (int i = 0; i < size; i++) {
        initial_solution[i] = i;
    }

    // Fisher-Yates shuffle
    // Ensure srand() was called before this point
    for (int i = size - 1; i > 0; i--) {
        int j = random_u64() % (i + 1);
        int temp = initial_solution[i];
        initial_solution[i] = initial_solution[j];
        initial_solution[j] = temp;
    }

    SolutionInstance* instance = create_solution_instance(connectome, initial_solution, true); // Calculate score

    free(initial_solution);
    return instance;
}

// Free a SolutionInstance
void free_solution_instance(SolutionInstance* instance) {
    if (instance) {
        free(instance->solution);
        free(instance->node_to_position);
        free(instance);
    }
}

// Calculate the total forward score from scratch
long long calculate_forward_score(const SolutionInstance* instance, const Connectome* connectome) {
    long long score = 0;
    for (long i = 0; i < instance->solution_size; i++) { // Iterate through positions
        int from_dense_idx = instance->solution[i]; // Get dense index at position i
        
        if (from_dense_idx < 0 || from_dense_idx >= connectome->num_nodes || !connectome->outgoing[from_dense_idx]) {
            continue; // Skip nodes outside range or with no outgoing edges
        }

        ConnectionNeighbor* neighbors = connectome->outgoing[from_dense_idx];
        int degree = connectome->out_degree[from_dense_idx];

        for (int j = 0; j < degree; j++) {
            int to_dense_idx = neighbors[j].neighbor_dense_idx;
            int weight = neighbors[j].weight;

            // Check if the 'to_node' exists in the solution's position mapping
            if (to_dense_idx >= 0 && to_dense_idx < instance->solution_size) {
                int to_pos = instance->node_to_position[to_dense_idx];
                if (to_pos > i) { // Check if the connection is forward
                    score += weight;
                }
             } else {
                  // This case should ideally not happen if solution_size == max_node_id
                 fprintf(stderr, "Warning: Encountered to_dense_idx %d >= solution_size %d\n", to_dense_idx, instance->solution_size);
             }
        }
    }
    return score;
}


// Calculate the CHANGE in forward score if nodes at pos1 and pos2 were swapped.
// More efficient than recalculating the whole score.
long long calculate_score_delta_on_swap(const SolutionInstance* instance, const Connectome* connectome, int pos1, int pos2) {
    if (pos1 == pos2) return 0;

    // Ensure pos1 < pos2
    if (pos2 < pos1) {
        int temp = pos1;
        pos1 = pos2;
        pos2 = temp;
    }

    int dense_idx1  = instance->solution[pos1];
    int dense_idx2  = instance->solution[pos2];

    long long delta = 0;

    // 1. Consider connection between node1 and node2
    int w12 = get_connection_weight(connectome, dense_idx1, dense_idx2 ); // Weight node1 -> node2
    int w21 = get_connection_weight(connectome, dense_idx2, dense_idx1 ); // Weight node2 -> node1

    // Original state: node1 at pos1, node2 at pos2 (pos1 < pos2)
    // Contribution: w12 (if exists)
    // New state: node2 at pos1, node1 at pos2 (pos1 < pos2)
    // Contribution: w21 (if exists)
    delta += (w21 - w12);

    // 2. Consider connections involving node1 and OTHER nodes (k != node2)
    // Outgoing from node1: (node1 -> k)
    if (dense_idx1 >= 0 && dense_idx1 < connectome->num_nodes && connectome->outgoing[dense_idx1]) {
        for (int i = 0; i < connectome->out_degree[dense_idx1]; ++i) {
            int k_dense_idx = connectome->outgoing[dense_idx1][i].neighbor_dense_idx;
            if (k_dense_idx == dense_idx2) continue; // Handled above
            int weight = connectome->outgoing[dense_idx1][i].weight;
            int k_pos = instance->node_to_position[k_dense_idx];

            // // Was it forward? (k_pos > pos1) -> Loses weight if k_pos < pos2
            // if (k_pos > pos1 && k_pos < pos2) delta -= weight;
            // // Was it backward? (k_pos < pos1) -> Gains weight if k_pos < pos2
            // if (k_pos < pos1 && k_pos < pos2) delta += weight; // Mistake in logic? Let's rethink.

            // Correct logic: Check original vs new forward status
            bool originally_forward = (k_pos > pos1);
            bool new_forward = (k_pos > pos2); // dense_idx1 moves to pos2

            if (originally_forward && !new_forward) delta -= weight; // Loses forward score
            if (!originally_forward && new_forward) delta += weight; // Gains forward score
        }
    }
     // Incoming to node1: (k -> node1)
     if (dense_idx1 >= 0 && dense_idx1 < connectome->num_nodes && connectome->incoming[dense_idx1]) {
        for (int i = 0; i < connectome->in_degree[dense_idx1]; ++i) {
           int k_dense_idx = connectome->incoming[dense_idx1][i].neighbor_dense_idx;
           if (k_dense_idx == dense_idx2) continue; // Handled above (as w21)
           int weight = connectome->incoming[dense_idx1][i].weight;
           int k_pos = instance->node_to_position[k_dense_idx];

            // Was it forward? (k_pos < pos1) -> Loses weight if k_pos > pos2
            if (k_pos < pos1 && k_pos > pos2) delta -= weight;
            // Was it backward? (k_pos > pos1) -> Gains weight if k_pos > pos2
             if (k_pos > pos1 && k_pos > pos2) delta += weight; // Again, let's verify logic.

            // Correct logic: Check original vs new forward status
            bool originally_forward = (k_pos < pos1);
            bool new_forward = (k_pos < pos2); // node1 moves to pos2

            if (originally_forward && !new_forward) delta -= weight; // Loses forward score
            if (!originally_forward && new_forward) delta += weight; // Gains forward score
        }
    }

    // 3. Consider connections involving node2 and OTHER nodes (k != node1)
    // Outgoing from node2: (node2 -> k)
    if (dense_idx2 >= 0 && dense_idx2 < connectome->num_nodes && connectome->outgoing[dense_idx2]) {
        for (int i = 0; i < connectome->out_degree[dense_idx2]; ++i) {
            int k_dense_idx = connectome->outgoing[dense_idx2][i].neighbor_dense_idx;
            if (k_dense_idx == dense_idx1) continue; // Handled above
            int weight = connectome->outgoing[dense_idx2][i].weight;
            int k_pos = instance->node_to_position[k_dense_idx];

            bool originally_forward = (k_pos > pos2);
            bool new_forward = (k_pos > pos1); // dense_idx2 moves to pos1

            if (originally_forward && !new_forward) delta -= weight;
            if (!originally_forward && new_forward) delta += weight;
        }
    }
     // Incoming to node2: (k -> node2)
     if (dense_idx2 >= 0 && dense_idx2 < connectome->num_nodes && connectome->incoming[dense_idx2]) {
        for (int i = 0; i < connectome->in_degree[dense_idx2]; ++i) {
           int k_dense_idx = connectome->incoming[dense_idx2][i].neighbor_dense_idx;
           if (k_dense_idx == dense_idx1) continue; // Handled above
           int weight = connectome->incoming[dense_idx2][i].weight;
           int k_pos = instance->node_to_position[k_dense_idx];

            bool originally_forward = (k_pos < pos2);
            bool new_forward = (k_pos < pos1); // node2 moves to pos1

            if (originally_forward && !new_forward) delta -= weight;
            if (!originally_forward && new_forward) delta += weight;
        }
     }

    return delta;
}


// Attempts a swap and applies it based on simulated annealing criteria.
// Returns true if the swap was accepted, false otherwise.
bool apply_swap(SolutionInstance* instance, const Connectome* connectome, int pos1, int pos2, double temperature, bool always_accept_better) {
    if (pos1 == pos2 || pos1 < 0 || pos2 < 0 || pos1 >= instance->solution_size || pos2 >= instance->solution_size) {
        return false; // Invalid positions
    }

    long long delta_score = calculate_score_delta_on_swap(instance, connectome, pos1, pos2);

    bool accept = false;
    if (delta_score > 0) { // Improvement
        accept = true;
    } else if (always_accept_better && delta_score == 0) { // Accept neutral moves if flag is set
         accept = true;
    } else if (temperature > 1e-9) { // Avoid division by zero or tiny temps; use annealing probability
        double acceptance_probability = exp((double)delta_score / temperature);
        if (acceptance_probability > random_double()) {
            accept = true;
        }
    } // else: delta_score < 0 and T is too low or zero, so reject

    if (accept) {
        // Apply the swap
        int dense_idx1 = instance->solution[pos1];
        int dense_idx2 = instance->solution[pos2];

        instance->solution[pos1] = dense_idx2;
        instance->solution[pos2] = dense_idx1;

        instance->node_to_position[dense_idx1] = pos2;
        instance->node_to_position[dense_idx2] = pos1;

        instance->forward_score += delta_score;

        // Verification (optional, disable for performance)
        // long long recalculated_score = calculate_forward_score(instance, connectome);
        // if (instance->forward_score != recalculated_score) {
        //     fprintf(stderr, "Score mismatch after swap! Delta: %lld, Incremental: %lld, Recalculated: %lld\n",
        //             delta_score, instance->forward_score, recalculated_score);
        //     // exit(1); // Or handle error
        //     instance->forward_score = recalculated_score; // Correct it
        // }

        return true;
    }

    return false;
}

// Check if the current instance is better than the best found so far and update if necessary.
bool update_best_solution(BestSolutionStorage* best_storage, const SolutionInstance* current_instance) {
    if (current_instance->forward_score > best_storage->best_score) {
        // Allocate or reallocate best_solution_array if needed
        if (!best_storage->best_solution_array || best_storage->solution_size != current_instance->solution_size) {
            free(best_storage->best_solution_array); // Free old one if size differs
            best_storage->solution_size = current_instance->solution_size;
            best_storage->best_solution_array = (int*)malloc(best_storage->solution_size * sizeof(int));
            if (!best_storage->best_solution_array) {
                perror("Failed to allocate memory for best solution storage");
                best_storage->best_score = -1; // Indicate error state
                return false;
            }
        }

        // Update best score and copy the solution
        best_storage->best_score = current_instance->forward_score;
        copy_solution(best_storage->best_solution_array, current_instance->solution, best_storage->solution_size);
        // best_storage->instance_id = current_instance->instance_id; // If tracking origin

        return true; // Improvement found
    }
    return false; // No improvement
}

long long get_solution_score(const SolutionInstance* instance) {
    return instance->forward_score;
}

long get_solution_size(const SolutionInstance* instance) {
    return instance->solution_size;
}

int* get_solution_array_ptr(SolutionInstance* instance) {
    return instance->solution;
}

BestSolutionStorage* create_best_solution_storage() {
    BestSolutionStorage* storage = (BestSolutionStorage*)malloc(sizeof(BestSolutionStorage));
    storage->solution_size = 0;
    storage->best_score = -1;
    storage->best_solution_array = NULL;
    return storage;
}

void free_best_solution_storage(BestSolutionStorage* storage) {
    free(storage->best_solution_array);
}

void init_best_solution_storage(BestSolutionStorage* storage, const SolutionInstance* instance) {
    storage->solution_size = instance->solution_size;
    storage->best_score = instance->forward_score;
    storage->best_solution_array = (int*)malloc(instance->solution_size * sizeof(int));
    copy_solution(storage->best_solution_array, instance->solution, instance->solution_size);    
}

long long get_best_solution_score(const BestSolutionStorage* storage) {
    return storage->best_score;
}

int* get_best_solution_array_ptr(BestSolutionStorage* storage) {
    return storage->best_solution_array;
}
