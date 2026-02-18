#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>
#include "connectome.h"

// In case your RAND_MAX is 2**15 for whatever reason
uint64_t random_u64(){
    return ((uint64_t) rand() << 0)  ^ ((uint64_t) rand() << 15) ^
           ((uint64_t) rand() << 30) ^ ((uint64_t) rand() << 45) ^
           (((uint64_t) rand() & 0xf) << 60);
}

double random_double() {
    return (double)random_u64() / (double)UINT64_MAX;
}

void seed_rng(unsigned int seed) {
    srand(seed);
}

long get_connectome_num_nodes(const Connectome* connectome) {
    return connectome->num_nodes;
}

uint64_t* get_dense_idx_to_node_id_array_ptr(const Connectome* connectome) {
    return connectome->dense_idx_to_node_id;
}

// Comparison function for qsort
int compare_neighbors(const void* a, const void* b) {
    ConnectionNeighbor* neighborA = (ConnectionNeighbor*)a;
    ConnectionNeighbor* neighborB = (ConnectionNeighbor*)b;
    return (neighborA->neighbor_dense_idx - neighborB->neighbor_dense_idx);
}

 // Helper to find or add node ID to hash map and return dense index
int get_or_assign_dense_idx(NodeIdMapping** map_hash_head, uint64_t node_id, long* current_node_count) {
    NodeIdMapping* found_entry;
    HASH_FIND(hh, *map_hash_head, &node_id, sizeof(uint64_t), found_entry);
   
    if (found_entry) return found_entry->dense_idx;

    found_entry = (NodeIdMapping*)malloc(sizeof(NodeIdMapping));
    if (!found_entry) {
            perror("Failed to allocate NodeIdMapping entry");
            // Should probably more robust error handling, maybe return -1 and check outside
            exit(EXIT_FAILURE); // Simple exit for now
    }
    found_entry->id = node_id;
    found_entry->dense_idx = (*current_node_count)++;
    HASH_ADD(hh, *map_hash_head, id, sizeof(uint64_t), found_entry);
    return found_entry->dense_idx;
    
}

// Load connectome from CSV file
Connectome* load_connectome(const char* graph_filename) {
    FILE* file = fopen(graph_filename, "r");
    if (!file) {
        perror("Failed to open graph file");
        return NULL;
    }
    printf("Loading connectome from %s...\n", graph_filename);

    Connectome* connectome = (Connectome*)calloc(1, sizeof(Connectome));
    if (!connectome) {
        fclose(file);
        perror("Failed to allocate memory for Connectome struct");
        return NULL;
    }

    connectome->node_id_map_hash = NULL;

    char line[1024];
    long long current_total_weight = 0;
    long current_num_connections = 0;
    // int current_max_id = -1;
    long current_num_nodes = 0;

    // Pass 1
    if (!fgets(line, sizeof(line), file)) { // Skip header
         fprintf(stderr, "Error: Could not read header line or file is empty.\n");
         fclose(file); free(connectome); return NULL;
    }

    long degree_array_capacity = 200000; // Initial estimate
    int* temp_out_degree = (int*)calloc(degree_array_capacity, sizeof(int));
    int* temp_in_degree = (int*)calloc(degree_array_capacity, sizeof(int));

    if (!temp_out_degree || !temp_in_degree) {
        perror("Failed to allocate temporary degree count arrays");
        fclose(file); free(connectome); return NULL;
    }

    while (fgets(line, sizeof(line), file)) {
        // int from_id, to_id, weight;
        // if (sscanf(line, "%d,%d,%d", &from_id, &to_id, &weight) != 3) {
        uint64_t from_id_u64, to_id_u64;
        int weight;
        if (sscanf(line, "%llu,%llu,%d", &from_id_u64, &to_id_u64, &weight) != 3) {
            fprintf(stderr, "Warning: Skipping malformed line: %s", line);
            continue;
        }

        // Get dense indices, potentially resizing degree arrays if needed
        int from_dense_idx = get_or_assign_dense_idx(&connectome->node_id_map_hash, from_id_u64, &current_num_nodes);
        int to_dense_idx = get_or_assign_dense_idx(&connectome->node_id_map_hash, to_id_u64, &current_num_nodes);

        // Check if dense index exceeds capacity and realloc if necessary
        if (from_dense_idx >= degree_array_capacity || to_dense_idx >= degree_array_capacity) {
            long old_capacity = degree_array_capacity;
            degree_array_capacity = (current_num_nodes > degree_array_capacity * 1.5) ? current_num_nodes + 1000 : degree_array_capacity * 2; // Grow capacity
            printf("    Reallocating temp degree arrays from %ld to %ld\n", old_capacity, degree_array_capacity);
            int* new_out_degree = (int*)realloc(temp_out_degree, degree_array_capacity * sizeof(int));
            int* new_in_degree = (int*)realloc(temp_in_degree, degree_array_capacity * sizeof(int));
            if (!new_out_degree || !new_in_degree) {
                 perror("Failed to realloc temporary degree arrays"); fclose(file); /* cleanup */ exit(EXIT_FAILURE);
            }
            // Zero out the newly allocated part
            memset(new_out_degree + old_capacity, 0, (degree_array_capacity - old_capacity) * sizeof(int));
            memset(new_in_degree + old_capacity, 0, (degree_array_capacity - old_capacity) * sizeof(int));
            temp_out_degree = new_out_degree;
            temp_in_degree = new_in_degree;
        }

        temp_out_degree[from_dense_idx]++;
        temp_in_degree[to_dense_idx]++;
        current_num_connections++;
        current_total_weight += weight;
    }

    connectome->num_nodes = current_num_nodes;
    connectome->num_connections = current_num_connections;
    connectome->total_weight = current_total_weight;

    printf("  - Unique Nodes: %ld\n", connectome->num_nodes);
    printf("  - Connections: %ld\n", connectome->num_connections);
    printf("  - Total Weight: %lld\n", connectome->total_weight);

    connectome->dense_idx_to_node_id = (uint64_t*)malloc(connectome->num_nodes * sizeof(uint64_t));
    connectome->out_degree = (int*)calloc(connectome->num_nodes, sizeof(int));
    connectome->in_degree = (int*)calloc(connectome->num_nodes, sizeof(int));
    connectome->outgoing = (ConnectionNeighbor**)calloc(connectome->num_nodes, sizeof(ConnectionNeighbor*));
    connectome->incoming = (ConnectionNeighbor**)calloc(connectome->num_nodes, sizeof(ConnectionNeighbor*));

    if (!connectome->dense_idx_to_node_id || !connectome->out_degree || !connectome->in_degree || !connectome->outgoing || !connectome->incoming) {
        perror("Failed to allocate connectome arrays");
        // Need more robust cleanup here
        fclose(file); free(temp_out_degree); free(temp_in_degree); free_connectome(connectome); return NULL;
    }

    // Populate the dense_idx_to_node_id map from the hash table
    NodeIdMapping *current_mapping, *tmp;
    HASH_ITER(hh, connectome->node_id_map_hash, current_mapping, tmp) {
        if(current_mapping->dense_idx >= connectome->num_nodes) {
            fprintf(stderr, "Error: Dense index %d out of bounds (num_nodes=%ld)\n", current_mapping->dense_idx, connectome->num_nodes);
             /* cleanup */ exit(EXIT_FAILURE);
        }
         connectome->dense_idx_to_node_id[current_mapping->dense_idx] = current_mapping->id;
    }


    // Allocate space for each node's neighbor list based on Pass 1 counts
    for (int i = 0; i < connectome->num_nodes; i++) {
        if (temp_out_degree[i] > 0) {
            connectome->outgoing[i] = (ConnectionNeighbor*)malloc(temp_out_degree[i] * sizeof(ConnectionNeighbor));
            if (!connectome->outgoing[i]) { perror("malloc outgoing[i]"); exit(EXIT_FAILURE); }
        }
        if (temp_in_degree[i] > 0) {
            connectome->incoming[i] = (ConnectionNeighbor*)malloc(temp_in_degree[i] * sizeof(ConnectionNeighbor));
             if (!connectome->incoming[i]) { perror("malloc incoming[i]"); exit(EXIT_FAILURE); }
        }
    }
    free(temp_out_degree); // Done with temporary counts
    free(temp_in_degree);

    // Pass 2
    rewind(file);
    if (!fgets(line, sizeof(line), file)) { /* Handle error reading header */ }

    // Use out/in_degree arrays as insertion indices
    while (fgets(line, sizeof(line), file)) {
        uint64_t from_id_u64, to_id_u64;
        int weight;

        if(sscanf(line, "%llu,%llu,%d", &from_id_u64, &to_id_u64, &weight) != 3) continue; // Already warned in pass 1

        // Lookup dense indices (should exist now)
        NodeIdMapping* from_entry;
        NodeIdMapping* to_entry;
        HASH_FIND(hh, connectome->node_id_map_hash, &from_id_u64, sizeof(uint64_t), from_entry);
        HASH_FIND(hh, connectome->node_id_map_hash, &to_id_u64, sizeof(uint64_t), to_entry);

        if (!from_entry || !to_entry) {
             fprintf(stderr, "Error: Node ID mapping not found in pass 2 (%" PRIu64 " or %" PRIu64 "). Should not happen.\n", from_id_u64, to_id_u64);
             /* cleanup */ exit(EXIT_FAILURE);
        }
        int from_dense_idx = from_entry->dense_idx;
        int to_dense_idx = to_entry->dense_idx;

        int out_idx = connectome->out_degree[from_dense_idx]++;
        connectome->outgoing[from_dense_idx][out_idx].neighbor_dense_idx = to_dense_idx;
        connectome->outgoing[from_dense_idx][out_idx].weight = weight;

        int in_idx = connectome->in_degree[to_dense_idx]++;
        connectome->incoming[to_dense_idx][in_idx].neighbor_dense_idx = from_dense_idx;
        connectome->incoming[to_dense_idx][in_idx].weight = weight;
    }
    fclose(file);

    // Free the hash map now that it's transferred to dense_idx_to_node_id
    HASH_CLEAR(hh, connectome->node_id_map_hash); // Free all items
    free(connectome->node_id_map_hash); // Not strictly needed after HASH_CLEAR frees items
    connectome->node_id_map_hash = NULL;

    //  Pass 3
    printf("  - Sorting neighbor lists...\n");
    for (int i = 0; i < connectome->num_nodes; i++) {
        if (connectome->out_degree[i] > 0) {
            qsort(connectome->outgoing[i], connectome->out_degree[i], sizeof(ConnectionNeighbor), compare_neighbors);
        }
        if (connectome->in_degree[i] > 0) {
             qsort(connectome->incoming[i], connectome->in_degree[i], sizeof(ConnectionNeighbor), compare_neighbors);
        }
    }

    return connectome;
}

// Free memory allocated for the Connectome
void free_connectome(Connectome* connectome) {
    if (!connectome) return;

    // Free hash map if it wasn't freed during loading (e.g., on error)
    if (connectome->node_id_map_hash) {
        NodeIdMapping *current_mapping, *tmp;
        HASH_ITER(hh, connectome->node_id_map_hash, current_mapping, tmp) {
            HASH_DEL(connectome->node_id_map_hash, current_mapping);
            free(current_mapping);
        }
        connectome->node_id_map_hash = NULL;
    }
    
    free(connectome->dense_idx_to_node_id);

    if (connectome->outgoing) {
        for (int i = 0; i < connectome->num_nodes; i++) {
            if (connectome->outgoing[i]) free(connectome->outgoing[i]);
        }
        free(connectome->outgoing);
    }
     if (connectome->incoming) {
        for (int i = 0; i < connectome->num_nodes; i++) {
            if (connectome->incoming[i]) free(connectome->incoming[i]);
        }
        free(connectome->incoming);
    }
    free(connectome->out_degree);
    free(connectome->in_degree);
    free(connectome);
}

// Get connection weight using binary search on sorted outgoing list
int get_connection_weight(const Connectome* connectome, int from_dense_idx, int to_dense_idx) {
    if (!connectome || from_dense_idx < 0 || from_dense_idx >= connectome->num_nodes || !connectome->outgoing[from_dense_idx]) {
        return 0; // Node doesn't exist or has no outgoing connections
    }

    ConnectionNeighbor* list = connectome->outgoing[from_dense_idx];
    int count = connectome->out_degree[from_dense_idx];
    int low = 0, high = count - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (list[mid].neighbor_dense_idx == to_dense_idx) {
            return list[mid].weight;
        } else if (list[mid].neighbor_dense_idx < to_dense_idx) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return 0; // Connection not found
}