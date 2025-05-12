#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#include "solution_instance.h"
#include "connectome.h"

// Creates a SolutionInstance. If initial_solution_array is NULL, creates a random one.
SolutionInstance* create_solution_instance(const Connectome* connectome, int* initial_solution_dense_array, bool calculate_initial_score) {
    if (!connectome) {
        fprintf(stderr, "Error (create_solution_instance): Connectome pointer is NULL.\n");
        return NULL;
    }
    long n = connectome->num_nodes;
    if (n <= 0) {
         fprintf(stderr, "Error (create_solution_instance): Connectome has no nodes.\n");
         return NULL;
    }

    SolutionInstance* instance = (SolutionInstance*)malloc(sizeof(SolutionInstance));
    if (!instance) {
        perror("Failed to allocate memory for SolutionInstance");
        return NULL;
    }

    instance->solution_size = n;
    instance->solution = (int*)malloc(n * sizeof(int));
    instance->node_to_position = (int*)malloc(n * sizeof(int));
    instance->forward_score = 0; // Initialize score

    if (!instance->solution || !instance->node_to_position) {
        perror("Failed to allocate memory for solution arrays");
        free(instance->solution); // Free partially allocated memory
        free(instance);
        return NULL;
    }

    // Initialize solution array
    if (initial_solution_dense_array) {
        memcpy(instance->solution, initial_solution_dense_array, n * sizeof(int));
    } else {
	printf("Creating random initial selection\n");
        // Create identity permutation (0, 1, 2, ..., n-1) first
        for (int i = 0; i < n; ++i) {
            instance->solution[i] = i;
        }
        // Shuffle it randomly (Fisher-Yates)
        unsigned int seed = time(NULL) ^ (unsigned int)random_u64(); 
        for (int i = n - 1; i > 0; --i) {
            int j = rand_r(&seed) % (i + 1);
            // Swap
            int temp = instance->solution[i];
            instance->solution[i] = instance->solution[j];
            instance->solution[j] = temp;
        }
    }

    // Initialize node_to_position map
    for (int i = 0; i < n; ++i) {
        instance->node_to_position[instance->solution[i]] = i;
    }

    // Calculate initial score if requested
    if (calculate_initial_score) {
        instance->forward_score = calculate_forward_score(instance, connectome);
    } else {
         instance->forward_score = -1; // Indicate score is not calculated yet
    }

    return instance;
}

// Creates a SolutionInstance with a random permutation.
SolutionInstance* create_random_solution_instance(const Connectome* connectome) {
    // Pass NULL for initial solution to trigger random creation, calculate score
    return create_solution_instance(connectome, NULL, true);
}

// Frees memory associated with a SolutionInstance.
void free_solution_instance(SolutionInstance* instance) {
    if (!instance) return;
    free(instance->solution);
    free(instance->node_to_position);
    free(instance);
}

// Calculates the total forward score from scratch.
long long calculate_forward_score(const SolutionInstance* instance, const Connectome* connectome) {
    if (!instance || !connectome) return 0;

    long long current_score = 0;
    long n = instance->solution_size;

    // Iterate through each node in its current position
    for (int i = 0; i < n; ++i) {
        int from_dense_idx = instance->solution[i]; // Get node at position i
        int from_pos = i;

        // Iterate through its outgoing neighbors
        if (connectome->outgoing[from_dense_idx]) { // Check if the list exists
            for (int k = 0; k < connectome->out_degree[from_dense_idx]; ++k) {
                int to_dense_idx = connectome->outgoing[from_dense_idx][k].neighbor_dense_idx;
                int weight = connectome->outgoing[from_dense_idx][k].weight;
                int to_pos = instance->node_to_position[to_dense_idx];

                // If neighbor position is after current node position, it's a forward edge
                if (from_pos < to_pos) {
                    current_score += weight;
                }
            }
        }
    }
    return current_score;
}


// Calculates the change in forward score IF nodes at pos1 and pos2 were swapped.
// Returns delta such that new_score = old_score + delta.
long long calculate_score_delta_on_swap(
    const SolutionInstance* instance,
    const Connectome* connectome,
    int pos1,
    int pos2)
{
    long long delta = 0;
    int dense_idx_a = instance->solution[pos1]; // Node currently at pos1
    int dense_idx_b = instance->solution[pos2]; // Node currently at pos2

    // Ensure pos1 < pos2 for consistency, though logic should handle either way
    // int pos_a = pos1;
    // int pos_b = pos2;


    // --- Process Neighbors of A (node at pos1) ---
    // Outgoing from A: a -> x
    for (int i = 0; i < connectome->out_degree[dense_idx_a]; ++i) {
        int dense_idx_x = connectome->outgoing[dense_idx_a][i].neighbor_dense_idx;
        int weight = connectome->outgoing[dense_idx_a][i].weight;

        if (dense_idx_x == dense_idx_b) continue; // Handle a<->b separately

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was a -> x forward? (a is at pos1)
        bool was_forward = pos1 < pos_x;
        // Will a -> x be forward? (a moves to pos2)
        bool is_forward = pos2 < pos_x;

        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }
    // Incoming to A: x -> a
    for (int i = 0; i < connectome->in_degree[dense_idx_a]; ++i) {
        int dense_idx_x = connectome->incoming[dense_idx_a][i].neighbor_dense_idx;
        int weight = connectome->incoming[dense_idx_a][i].weight;

        if (dense_idx_x == dense_idx_b) continue; // Handle a<->b separately

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was x -> a forward? (a is at pos1)
        bool was_forward = pos_x < pos1;
        // Will x -> a be forward? (a moves to pos2)
        bool is_forward = pos_x < pos2;

        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }

    // --- Process Neighbors of B (node at pos2) ---
    // Outgoing from B: b -> x
    for (int i = 0; i < connectome->out_degree[dense_idx_b]; ++i) {
        int dense_idx_x = connectome->outgoing[dense_idx_b][i].neighbor_dense_idx;
        int weight = connectome->outgoing[dense_idx_b][i].weight;

        if (dense_idx_x == dense_idx_a) continue; // Already handled

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was b -> x forward? (b is at pos2)
        bool was_forward = pos2 < pos_x;
        // Will b -> x be forward? (b moves to pos1)
        bool is_forward = pos1 < pos_x;

        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }
    // Incoming to B: x -> b
    for (int i = 0; i < connectome->in_degree[dense_idx_b]; ++i) {
        int dense_idx_x = connectome->incoming[dense_idx_b][i].neighbor_dense_idx;
        int weight = connectome->incoming[dense_idx_b][i].weight;

         if (dense_idx_x == dense_idx_a) continue; // Already handled

        int pos_x = instance->node_to_position[dense_idx_x];

        // Was x -> b forward? (b is at pos2)
        bool was_forward = pos_x < pos2;
        // Will x -> b be forward? (b moves to pos1)
        bool is_forward = pos_x < pos1;

        if (is_forward && !was_forward) delta += weight;
        else if (!is_forward && was_forward) delta -= weight;
    }

    // --- Handle the direct a <-> b interaction explicitly ---
    int weight_ab = get_connection_weight(connectome, dense_idx_a, dense_idx_b);
    int weight_ba = get_connection_weight(connectome, dense_idx_b, dense_idx_a);

    // Contribution of a -> b
    if (weight_ab > 0) {
        bool was_forward_ab = pos1 < pos2;
        bool is_forward_ab = pos2 < pos1; // New positions
        if (is_forward_ab && !was_forward_ab) delta += weight_ab;
        else if (!is_forward_ab && was_forward_ab) delta -= weight_ab;
    }

    // Contribution of b -> a
     if (weight_ba > 0) {
        bool was_forward_ba = pos2 < pos1;
        bool is_forward_ba = pos1 < pos2; // New positions
        if (is_forward_ba && !was_forward_ba) delta += weight_ba;
        else if (!is_forward_ba && was_forward_ba) delta -= weight_ba;
    }

    return delta;
}


// Attempts to swap nodes at pos1 and pos2 based on SA criteria.
// Modifies the instance IN PLACE if the swap is accepted.
// Returns true if swap was accepted, false otherwise.
// NOTE: Requires external RNG state (e.g., rand_r or thread-local)
bool apply_swap(SolutionInstance* instance, const Connectome* connectome, int pos1, int pos2, double temperature, bool always_accept_better, unsigned int *rng_seed) {
    if (!instance || !connectome || pos1 == pos2 || pos1 < 0 || pos2 < 0 || pos1 >= instance->solution_size || pos2 >= instance->solution_size) {
        return false; // Invalid input
    }

    // 1. Calculate Score Delta
    long long delta_score = calculate_score_delta_on_swap(instance, connectome, pos1, pos2);

    // 2. Determine Acceptance
    bool accept = false;
    if (delta_score > 0) { // Improvement (higher score is better)
        accept = true;
    } else if (always_accept_better && delta_score == 0) { // Accept neutral moves
         accept = true;
    } else if (temperature > 1e-9) { // Avoid issues with T=0
        if (random_double() < exp((double)delta_score / temperature)) {
             accept = true;
        }
    }
    // else: delta_score < 0 and T=0, reject worsening move

    // 3. Apply or Reject
    if (accept) {
        // Perform the swap
        int dense_idx1 = instance->solution[pos1];
        int dense_idx2 = instance->solution[pos2];

        instance->solution[pos1] = dense_idx2;
        instance->solution[pos2] = dense_idx1;

        instance->node_to_position[dense_idx1] = pos2;
        instance->node_to_position[dense_idx2] = pos1;

        // Update the score incrementally
        instance->forward_score += delta_score;
    }
    // else: do nothing, instance remains unchanged

    return accept;
}


// Simple accessor for the score
long long get_solution_score(const SolutionInstance* instance) {
    return instance ? instance->forward_score : -1;
}

// Simple accessor for the size
long get_solution_size(const SolutionInstance* instance) {
     return instance ? instance->solution_size : 0;
}

// Returns a pointer to the solution array (use with caution - read only ideally)
int* get_solution_array_ptr(SolutionInstance* instance) {
     return instance ? instance->solution : NULL;
}

// Helper to copy solution array
void copy_solution(int* dest, const int* src, long size) {
    if (dest && src && size > 0) {
        memcpy(dest, src, size * sizeof(int));
    }
}

// --- Best Solution Storage Management ---

BestSolutionStorage* create_best_solution_storage() {
    BestSolutionStorage* storage = (BestSolutionStorage*)malloc(sizeof(BestSolutionStorage));
    if (!storage) {
        perror("Failed to allocate BestSolutionStorage");
        return NULL;
    }
    storage->best_solution_array = NULL; // Not allocated until initialized
    storage->best_score = -1; // Or some indicator of uninitialized
    storage->solution_size = 0;
    return storage;
}

void free_best_solution_storage(BestSolutionStorage* storage) {
    if (!storage) return;
    free(storage->best_solution_array); // Free the array first
    free(storage); // Then free the struct
}

// Initializes the storage with the state of a given instance.
void init_best_solution_storage(BestSolutionStorage* storage, const SolutionInstance* instance) {
    if (!storage || !instance || instance->solution_size <= 0) return;

    // Allocate or reallocate if size differs (or if uninitialized)
    if (!storage->best_solution_array || storage->solution_size != instance->solution_size) {
        free(storage->best_solution_array); // Free old one if necessary
        storage->solution_size = instance->solution_size;
        storage->best_solution_array = (int*)malloc(storage->solution_size * sizeof(int));
        if (!storage->best_solution_array) {
            perror("Failed to allocate memory for best solution array in storage");
            storage->solution_size = 0; // Mark as invalid
            return;
        }
    }

    // Copy the solution and score
    memcpy(storage->best_solution_array, instance->solution, storage->solution_size * sizeof(int));
    storage->best_score = instance->forward_score;
}

// Updates the best storage if the current instance is better. Returns true if updated.
bool update_best_solution(BestSolutionStorage* best_storage, const SolutionInstance* current_instance) {
    if (!best_storage || !current_instance) return false;

    // Check if current is better (higher score) or if storage is uninitialized
    if (current_instance->forward_score > best_storage->best_score || best_storage->best_solution_array == NULL) {
        init_best_solution_storage(best_storage, current_instance); // Use init to handle allocation/copying
        return true;
    }
    return false;
}

long long get_best_solution_score(const BestSolutionStorage* storage) {
    return storage ? storage->best_score : -1;
}

int* get_best_solution_array_ptr(BestSolutionStorage* storage) {
    return storage ? storage->best_solution_array : NULL;
}

void update_best_solution_storage_if_better(BestSolutionStorage* storage, const SolutionInstance* instance) {
    if (!storage || !instance) {
        fprintf(stderr, "Warning: NULL pointer passed to update_best_solution_storage_if_better\n");
        return;
    }

    // Check if the instance score is strictly better than the stored best score
    if (instance->forward_score > storage->best_score) {

        // Ensure storage is initialized or has matching size
        if (storage->best_solution_array == NULL || storage->solution_size != instance->solution_size) {
            free(storage->best_solution_array); // Free old buffer if size mismatch or NULL
            storage->best_solution_array = (int*)malloc(instance->solution_size * sizeof(int));
            if (!storage->best_solution_array) {
                fprintf(stderr, "Error: Failed to allocate memory in update_best_solution_storage_if_better\n");
                storage->solution_size = 0;
                storage->best_score = -LLONG_MAX; // Indicate error state
                return;
            }
            storage->solution_size = instance->solution_size; // Update size only after successful allocation
        }

        // Copy the better solution and score
        memcpy(storage->best_solution_array, instance->solution, instance->solution_size * sizeof(int));
        storage->best_score = instance->forward_score;
    }
}

