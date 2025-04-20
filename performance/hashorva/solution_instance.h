#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    int from_id;
    int to_id;
    int weight;
} Connection;

typedef struct {
    int forward;
    int backward;
    double ratio;  // calculated: forward / (forward + backward)
} Score;

typedef struct {
    int *best;
    int score;
    int instance_id;
} BestSolution;

typedef struct {
    int *solution;
    int *ix_lookup;
    int forward_score;
    int backward_score;
    int instance_id;
} SolutionInstance;

// These would be defined in the main program, referenced here
int MAX_CELL_ID;
Connection *connection_by_cells_id_dict;
long connection_dict_count;
int **outgoing_connections;
int *outgoing_connections_size;
int **incoming_connections;
int *incoming_connections_size;

// Function prototypes
SolutionInstance* create_solution_instance(int *solution, int solution_size, int forward_score);
SolutionInstance* create_random_solution_instance();
SolutionInstance* read_solution_from_file(const char *filename, bool randomize);
void free_solution_instance(SolutionInstance *instance);
Score calculate_score(SolutionInstance *instance);
void swap(SolutionInstance *instance, int i1, int i2, double temperature);
bool check_best_solution(BestSolution *best, SolutionInstance *instance);
uint64_t random();
long get_connection_hash(long from_id, long to_id);