#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "connectome.h"

// Represents a potential solution (ordering of nodes)
typedef struct {
    int* solution;         // Array of node IDs in the current order. Size = connectome->max_node_id
                           // Note: Nodes without connections might exist here, placed arbitrarily.
                           // The score calculation implicitly ignores them.
    int* node_to_position; // Map: node_to_position[node_id] = index in the solution array. Size = connectome->max_node_id
    long long forward_score;   // Current forward score (sum of weights of forward edges)
    int solution_size;     // Number of elements in solution array (connectome->max_node_id)
    // int instance_id;    // Can keep if needed for tracking multiple instances
} SolutionInstance;

// Structure to hold the best solution found so far
typedef struct {
    int *best_solution_array; // Copy of the best solution permutation found
    long long best_score;       // The score of the best solution
    int solution_size;        // Size of the best_solution_array
    // int instance_id;       // ID of the instance that found this best solution
} BestSolutionStorage;


// Function Prototypes for solution instances
SolutionInstance* create_solution_instance(const Connectome* connectome, int* initial_solution_array, bool calculate_initial_score);
SolutionInstance* create_random_solution_instance(const Connectome* connectome);
void free_solution_instance(SolutionInstance* instance);
long long calculate_forward_score(const SolutionInstance* instance, const Connectome* connectome);
long long calculate_score_delta_on_swap(const SolutionInstance* instance, const Connectome* connectome, int pos1, int pos2);
bool apply_swap(SolutionInstance* instance, const Connectome* connectome, int pos1, int pos2, double temperature, bool always_accept_better, unsigned int *rng_seed);
bool update_best_solution(BestSolutionStorage* best_storage, const SolutionInstance* current_instance);
void copy_solution(int* dest, const int* src, long size);

long long get_solution_score(const SolutionInstance* instance);
long get_solution_size(const SolutionInstance* instance);
int* get_solution_array_ptr(SolutionInstance* instance);

void init_best_solution_storage(BestSolutionStorage* storage, const SolutionInstance* instance);
long long get_best_solution_score(const BestSolutionStorage* storage);
int* get_best_solution_array_ptr(BestSolutionStorage* storage);

void update_best_solution_storage_if_better(BestSolutionStorage* storage, const SolutionInstance* instance);

BestSolutionStorage* create_best_solution_storage();
void free_best_solution_storage(BestSolutionStorage* storage);
