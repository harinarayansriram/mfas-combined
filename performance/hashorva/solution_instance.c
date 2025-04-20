#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "solution_instance.h"

// In case your RAND_MAX is 2**15 for whatever reason
uint64_t random(){
    return ((uint64_t) rand() << 0) ^ ((uint64_t) rand() << 15) ^ ((uint64_t) rand() << 30) ^ ((uint64_t) rand() << 45) ^ (((uint64_t) rand() & 0xf) << 60);
}

SolutionInstance* create_solution_instance(int *solution, int solution_size, int forward_score) {
    SolutionInstance *instance = (SolutionInstance*)malloc(sizeof(SolutionInstance));
    
    // Copy solution
    instance->solution = (int*)malloc(solution_size * sizeof(int));
    memcpy(instance->solution, solution, solution_size * sizeof(int));
    
    // Initialize ix_lookup
    instance->ix_lookup = (int*)malloc((MAX_CELL_ID + 1) * sizeof(int));
    
    if (forward_score == 0) {
        // Calculate score
        Score score = calculate_score(instance);
        instance->forward_score = score.forward;
        instance->backward_score = score.backward;
    } else {
        // Use provided score and update ix_lookup
        for (int i = 0; i < solution_size; i++) {
            instance->ix_lookup[instance->solution[i]] = i;
        }
        instance->forward_score = forward_score;
    }
    
    instance->instance_id = 0;
    
    return instance;
}

SolutionInstance* create_random_solution_instance() {
    // Generate random solution
    int *solution = (int*)malloc(MAX_CELL_ID * sizeof(int));
    
    for (int i = 0; i < MAX_CELL_ID; i++) {
        solution[i] = i + 1;
    }
    
    // Randomize solution
    srand(time(NULL));
    for (int i = 0; i < MAX_CELL_ID; i++) {
        int j = random() % MAX_CELL_ID;
        int temp = solution[i];
        solution[i] = solution[j];
        solution[j] = temp;
    }
    
    SolutionInstance *instance = create_solution_instance(solution, MAX_CELL_ID, 0);
    
    free(solution);
    return instance;
}

void free_solution_instance(SolutionInstance *instance) {
    if (instance) {
        if (instance->solution) free(instance->solution);
        if (instance->ix_lookup) free(instance->ix_lookup);
        free(instance);
    }
}

Score calculate_score(SolutionInstance *instance) {
    // Update ix_lookup
    for (int i = 0; i < MAX_CELL_ID; i++) {
        instance->ix_lookup[instance->solution[i]] = i;
    }
    
    Score score = {0, 0, 0.0};
    int *visited = (int*)calloc(MAX_CELL_ID + 1, sizeof(int));
    
    for (int i = 0; i < MAX_CELL_ID; i++) {
        int from = instance->solution[i];
        
        if (visited[from]) {
            fprintf(stderr, "Collection corrupted!\n");
            exit(1);
        } else {
            visited[from] = 1;
        }
        
        // Check if there are outgoing connections
        if (outgoing_connections_size[from] == 0) continue;
        
        for (int j = 0; j < outgoing_connections_size[from]; j++) {
            int to = outgoing_connections[from][j];
            
            // Find the connection
            Connection *conn = NULL;
            for (long k = 0; k < connection_dict_count; k++) {
                if (connection_by_cells_id_dict[k].from_id == from && 
                    connection_by_cells_id_dict[k].to_id == to) {
                    conn = &connection_by_cells_id_dict[k];
                    break;
                }
            }
            
            if (!conn) continue;
            
            int to_ix = instance->ix_lookup[to];
            
            if (to_ix > i) {
                score.forward += conn->weight;
            } else {
                score.backward += conn->weight;
            }
        }
    }
    
    // Calculate ratio
    if (score.forward + score.backward != 0) {
        score.ratio = (double)score.forward / (score.forward + score.backward);
    } else {
        score.ratio = 0.0;
    }
    
    free(visited);
    return score;
}

void swap(SolutionInstance *instance, int i1, int i2, double temperature) {
    if (i1 == i2) return;
    
    // Ensure i1 < i2
    if (i2 < i1) {
        int temp = i1;
        i1 = i2;
        i2 = temp;
    }
    
    int c1 = instance->solution[i1];
    int c2 = instance->solution[i2];
    
    int o_f = 0; // Original forward connections
    
    // Calculate original forward connections
    // For incoming to c1
    if (incoming_connections_size[c1] > 0) {
        for (int i = 0; i < incoming_connections_size[c1]; i++) {
            int c = incoming_connections[c1][i];
            int cix = instance->ix_lookup[c];
            
            if (cix < i1) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c && 
                        connection_by_cells_id_dict[j].to_id == c1) {
                        o_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // For incoming to c2
    if (incoming_connections_size[c2] > 0) {
        for (int i = 0; i < incoming_connections_size[c2]; i++) {
            int c = incoming_connections[c2][i];
            int cix = instance->ix_lookup[c];
            
            if (cix < i2) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c && 
                        connection_by_cells_id_dict[j].to_id == c2) {
                        o_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // For outgoing from c1
    if (outgoing_connections_size[c1] > 0) {
        for (int i = 0; i < outgoing_connections_size[c1]; i++) {
            int c = outgoing_connections[c1][i];
            int cix = instance->ix_lookup[c];
            
            if (cix == i2) continue;
            if (cix > i1) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c1 && 
                        connection_by_cells_id_dict[j].to_id == c) {
                        o_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // For outgoing from c2
    if (outgoing_connections_size[c2] > 0) {
        for (int i = 0; i < outgoing_connections_size[c2]; i++) {
            int c = outgoing_connections[c2][i];
            int cix = instance->ix_lookup[c];
            
            if (cix == i1) continue;
            if (cix > i2) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c2 && 
                        connection_by_cells_id_dict[j].to_id == c) {
                        o_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // Swap the elements
    instance->ix_lookup[instance->solution[i1]] = i2;
    instance->ix_lookup[instance->solution[i2]] = i1;
    
    int temp = instance->solution[i1];
    instance->solution[i1] = instance->solution[i2];
    instance->solution[i2] = temp;
    
    // Swap c1 and c2 for calculations
    temp = c1;
    c1 = c2;
    c2 = temp;
    
    int n_f = 0; // New forward connections
    
    // Calculate new forward connections - follows same pattern as above
    // For incoming to c1
    if (incoming_connections_size[c1] > 0) {
        for (int i = 0; i < incoming_connections_size[c1]; i++) {
            int c = incoming_connections[c1][i];
            int cix = instance->ix_lookup[c];
            
            if (cix < i1) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c && 
                        connection_by_cells_id_dict[j].to_id == c1) {
                        n_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // For incoming to c2
    if (incoming_connections_size[c2] > 0) {
        for (int i = 0; i < incoming_connections_size[c2]; i++) {
            int c = incoming_connections[c2][i];
            int cix = instance->ix_lookup[c];
            
            if (cix < i2) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c && 
                        connection_by_cells_id_dict[j].to_id == c2) {
                        n_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // For outgoing from c1
    if (outgoing_connections_size[c1] > 0) {
        for (int i = 0; i < outgoing_connections_size[c1]; i++) {
            int c = outgoing_connections[c1][i];
            int cix = instance->ix_lookup[c];
            
            if (cix == i2) continue;
            if (cix > i1) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c1 && 
                        connection_by_cells_id_dict[j].to_id == c) {
                        n_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // For outgoing from c2
    if (outgoing_connections_size[c2] > 0) {
        for (int i = 0; i < outgoing_connections_size[c2]; i++) {
            int c = outgoing_connections[c2][i];
            int cix = instance->ix_lookup[c];
            
            if (cix > i2) {
                // Find connection weight
                for (long j = 0; j < connection_dict_count; j++) {
                    if (connection_by_cells_id_dict[j].from_id == c2 && 
                        connection_by_cells_id_dict[j].to_id == c) {
                        n_f += connection_by_cells_id_dict[j].weight;
                        break;
                    }
                }
            }
        }
    }
    
    // Simulated annealing acceptance probability
    double acceptance_probability = exp((n_f - o_f) / temperature);
    
    // Decide whether to accept the swap
    if (n_f >= o_f || acceptance_probability > ((double)random() / UINT64_MAX)) {
        instance->forward_score += n_f - o_f;
    } else {
        // Swap back if not accepted
        instance->ix_lookup[instance->solution[i1]] = i2;
        instance->ix_lookup[instance->solution[i2]] = i1;
        
        temp = instance->solution[i1];
        instance->solution[i1] = instance->solution[i2];
        instance->solution[i2] = temp;
    }
}

bool check_best_solution(BestSolution *best, SolutionInstance *instance) {
    if (instance->forward_score > best->score) {
        if (best->best) free(best->best);
        
        best->score = instance->forward_score;
        best->best = (int*)malloc(MAX_CELL_ID * sizeof(int));
        memcpy(best->best, instance->solution, MAX_CELL_ID * sizeof(int));
        best->instance_id = instance->instance_id;
        
        return true;
    }
    
    return false;
}

SolutionInstance* read_solution_from_file(const char *filename, bool randomize) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open solution file");
        return NULL;
    }
    
    int *solution = (int*)malloc(MAX_CELL_ID * sizeof(int));
    int count = 0;
    char line[1024];
    
    while (fgets(line, sizeof(line), file) && count < MAX_CELL_ID) {
        int value;
        if (sscanf(line, "%d", &value) != 1) continue;
        
        solution[count++] = value;
    }
    
    fclose(file);
    
    if (count != MAX_CELL_ID) {
        fprintf(stderr, "Corrupted solution file! Expected %d elements, got %d\n", MAX_CELL_ID, count);
        free(solution);
        return NULL;
    }
    
    if (randomize) {
        // Randomize the solution
        srand(time(NULL));
        for (int i = 0; i < MAX_CELL_ID; i++) {
            int j = random() % MAX_CELL_ID;
            int temp = solution[i];
            solution[i] = solution[j];
            solution[j] = temp;
        }
    }
    
    SolutionInstance *instance = create_solution_instance(solution, MAX_CELL_ID, 0);
    
    free(solution);
    return instance;
}

long get_connection_hash(long from_id, long to_id) {
    return (from_id << 18) | to_id;
}