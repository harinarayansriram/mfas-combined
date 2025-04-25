#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "uthash.h"

// Represents a directed connection to a neighbor node
typedef struct {
    // long long neighbor_id; // The ID of the node connected to (either to_id or from_id)
    int neighbor_dense_idx; // The DENSE index of the node connected to
    int weight;      // The weight of the connection
} ConnectionNeighbor;

// Structure for the hash map (uint64_t ID -> dense int index)
typedef struct {
    uint64_t id;             // key (original node ID)
    int dense_idx;           // value (internal dense index)
    UT_hash_handle hh;       // makes this structure hashable
} NodeIdMapping;

// Represents the entire connectome graph
typedef struct {
    // long long max_node_id;            // Highest node ID encountered + 1 (size needed for arrays)
    long num_nodes;             // Count of unique node IDs encountered
    long num_connections;       // Total number of directed edges
    long long total_weight;     // Sum of all connection weights
    uint64_t* dense_idx_to_node_id; // Map: dense_idx -> original uint64_t node ID. Size = num_nodes

    ConnectionNeighbor** outgoing; // Array of arrays: outgoing[from_id] -> sorted list of {to_id, weight}
    int* out_degree;            // out_degree[from_id] = number of outgoing connections

    ConnectionNeighbor** incoming; // Array of arrays: incoming[to_id] -> sorted list of {from_id, weight}
    int* in_degree;             // in_degree[to_id] = number of incoming connections

    NodeIdMapping* node_id_map_hash; // Pointer to the hash table head (uthash)

} Connectome;

// Function Prototypes for connectome handling
Connectome* load_connectome(const char* graph_filename);
void free_connectome(Connectome* connectome);
int get_connection_weight(const Connectome* connectome, int from_dense_idx, int to_dense_idx);
// Utility for qsort
int compare_neighbors(const void* a, const void* b);
// Utility for random number generation (consistent across files)
uint64_t random_u64();
double random_double();
void seed_rng(unsigned int seed);

long get_connectome_num_nodes(const Connectome* connectome);
uint64_t* get_dense_idx_to_node_id_array_ptr(const Connectome* connectome);