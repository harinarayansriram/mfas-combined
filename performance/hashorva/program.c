#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include "solution_instance.h"

#define MAX_CELL_ID 136648
#define TOTAL_CONNECTIONS 41912141

// Global variables
char **cell_names;
int *cell_names_r_keys;
int *cell_names_r_values;
int cell_names_count;

// Connection *connection_by_cells_id_dict;
long connection_dict_count;

int **outgoing_connections;
int *outgoing_connections_size;
int **incoming_connections;
int *incoming_connections_size;

long cnt;
long last_change_step;
long last_step;
double last_temperature;

BestSolution best_solution;

void randomize_ints(int *list, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        int j = (int)(random() % size);
        int temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}

void randomize_ints_range(int *list, int min_ix, int max_ix) {
    srand(time(NULL));
    int range = max_ix - min_ix + 1;
    
    for (int i = min_ix; i <= max_ix; i++) {
        int j = min_ix + (int)(random() % range);
        int temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}

void read_connections(char* filename) {
    printf("Reading connections...\n");
    
    // Allocate memory for connections
    connection_by_cells_id_dict = (Connection*)malloc(sizeof(Connection) * (1 << 24));
    
    // Allocate memory for outgoing and incoming connections
    outgoing_connections = (int**)calloc(MAX_CELL_ID + 1, sizeof(int*));
    outgoing_connections_size = (int*)calloc(MAX_CELL_ID + 1, sizeof(int));
    incoming_connections = (int**)calloc(MAX_CELL_ID + 1, sizeof(int*));
    incoming_connections_size = (int*)calloc(MAX_CELL_ID + 1, sizeof(int));
    
    // Open file
    // char filename[256];
    // sprintf(filename, "%s/graph.csv", WORK_DIR);
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open graph file");
        exit(1);
    }
    
    // Read the header line
    char line[1024];
    fgets(line, sizeof(line), file);
    
    // First pass: count connections for each cell
    while (fgets(line, sizeof(line), file)) {
        int from_id, to_id, weight;
        if (sscanf(line, "%d,%d,%d", &from_id, &to_id, &weight) != 3) continue;
        
        outgoing_connections_size[from_id]++;
        incoming_connections_size[to_id]++;
    }
    
    // Allocate memory for each cell's connections
    for (int i = 0; i <= MAX_CELL_ID; i++) {
        if (outgoing_connections_size[i] > 0) {
            outgoing_connections[i] = (int*)malloc(sizeof(int) * outgoing_connections_size[i]);
        }
        if (incoming_connections_size[i] > 0) {
            incoming_connections[i] = (int*)malloc(sizeof(int) * incoming_connections_size[i]);
        }
    }
    
    // Reset counts to use as indices
    memset(outgoing_connections_size, 0, (MAX_CELL_ID + 1) * sizeof(int));
    memset(incoming_connections_size, 0, (MAX_CELL_ID + 1) * sizeof(int));
    
    // Second pass: populate connections
    rewind(file);
    fgets(line, sizeof(line), file); // skip header
    
    while (fgets(line, sizeof(line), file)) {
        int from_id, to_id, weight;
        if (sscanf(line, "%d,%d,%d", &from_id, &to_id, &weight) != 3) continue;
        
        long key = get_connection_hash(from_id, to_id);
        connection_by_cells_id_dict[connection_dict_count].from_id = from_id;
        connection_by_cells_id_dict[connection_dict_count].to_id = to_id;
        connection_by_cells_id_dict[connection_dict_count].weight = weight;
        connection_dict_count++;
        
        outgoing_connections[from_id][outgoing_connections_size[from_id]++] = to_id;
        incoming_connections[to_id][incoming_connections_size[to_id]++] = from_id;
    }
    
    fclose(file);
    printf("Connections loaded: %ld\n", connection_dict_count);
}

void read_cell_names(char* filename) {
    printf("Reading cell names...\n");
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open cells.txt");
        exit(1);
    }
    
    // Count lines
    char line[1024];
    int line_count = 0;
    while (fgets(line, sizeof(line), file)) {
        line_count++;
    }
    
    cell_names = (char**)malloc(sizeof(char*) * (MAX_CELL_ID + 1));
    cell_names_r_keys = (int*)malloc(sizeof(int) * line_count);
    cell_names_r_values = (int*)malloc(sizeof(int) * line_count);
    
    // Read cell names
    rewind(file);
    fgets(line, sizeof(line), file); // skip header
    
    while (fgets(line, sizeof(line), file)) {
        int id;
        char name[256];
        if (sscanf(line, "%d,%255[^\n]", &id, name) != 2) continue;
        
        cell_names[id] = strdup(name);
        cell_names_r_keys[cell_names_count] = id;
        cell_names_r_values[cell_names_count] = id;
        cell_names_count++;
    }
    
    fclose(file);
    printf("Cell names loaded: %d\n", cell_names_count);
}

void convert_to_cell_names(char* source_filename, char* dest_filename) {
    FILE *source_file = fopen(source_filename, "r");
    if (!source_file) {
        perror("Failed to open source file");
        return;
    }
    
    FILE *dest_file = fopen(dest_filename, "w");
    if (!dest_file) {
        perror("Failed to open destination file");
        fclose(source_file);
        return;
    }
    
    fprintf(dest_file, "Node ID,Order\n");
    
    // Process each line
    char line[1024];
    int i = 0;
    while (fgets(line, sizeof(line), source_file)) {
        if (strlen(line) == 0) continue;
        
        int id;
        if (sscanf(line, "%d", &id) != 1) continue;
        
        fprintf(dest_file, "%s,%d\n", cell_names[id], i);
        i++;
    }
    
    fclose(source_file);
    fclose(dest_file);
}

void thread_run(double temperature, double cooling_rate, SolutionInstance* the_solution) {
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        #pragma omp for
        for (int i = 0; i < 1000000; i++) {
            int i1 = (int)(random() % MAX_CELL_ID);
            int i2 = (int)(random() % (MAX_CELL_ID-1));

            if(i1 == i2) i2++;
                        
            swap(the_solution, i1, i2, temperature);
            
            #pragma omp critical
            {
                check_best_solution(&best_solution, the_solution);
                cnt++;
            }
            
            temperature *= (1 - cooling_rate);
        }
    }
    
    last_temperature = temperature;
}

char* float_to_string(double value) {
    static char buffer[64];
    sprintf(buffer, "%.17g", value);
    return buffer;
}

void write_solution_to_file(BestSolution *solution, double temperature) {
    char filename[256];
    sprintf(filename, "./state/%08d_%011ld_%s.txt", solution->score, cnt, 
            float_to_string(temperature));
    
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file for writing solution");
        return;
    }
    
    for (int i = 0; i < MAX_CELL_ID; i++) {
        fprintf(file, "%d\n", solution->best[i]);
    }
    
    fclose(file);
}

void perform_simanneal(double temperature, double cooling_rate, SolutionInstance* the_solution, bool log_to_file, bool write_updated_solution) {
    // Initialize random seed
    srand(time(NULL));
    
    // Simulated annealing loop
    while (temperature >= 0) {
        thread_run(temperature, cooling_rate, the_solution);
        
        temperature = last_temperature;
        
        if (best_solution.best != NULL) {
            
            if(log_to_file){
                char timestamp[64];
                time_t now = time(NULL);
                strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
                printf("%s\t%.17g\t%d\t%f\n", timestamp, temperature, best_solution.score, 
                    (double)best_solution.score / TOTAL_CONNECTIONS);
                
                // Log to file
                char* log_filename = "./log.txt";
                FILE *log_file = fopen(log_filename, "a");
                if (log_file) {
                    fprintf(log_file, "%s\t%.17g\t%d\t%f\n", timestamp, temperature, best_solution.score, 
                            (double)best_solution.score / TOTAL_CONNECTIONS);
                    fclose(log_file);
                }
            }
            
            if(write_updated_solution){
                // Write solution to file
                write_solution_to_file(&best_solution, temperature);
            }
        }
    }
    if (best_solution.best != NULL) {
        write_solution_to_file(&best_solution, temperature);
    }
}

void cleanup(){
    for (int i = 0; i <= MAX_CELL_ID; i++) {
        if (outgoing_connections[i]) free(outgoing_connections[i]);
        if (incoming_connections[i]) free(incoming_connections[i]);
        if (cell_names[i]) free(cell_names[i]);
    }
    
    free(outgoing_connections);
    free(outgoing_connections_size);
    free(incoming_connections);
    free(incoming_connections_size);
    free(cell_names);
    free(cell_names_r_keys);
    free(cell_names_r_values);
    free(connection_by_cells_id_dict);
    
    if (best_solution.best) free(best_solution.best);
    
}

int main(int argc, char *argv[]) {
    printf("Feed Forward algorithm starting...\n");
    
    char* filename = "./graph.csv";
    read_connections(filename);

    char* filename2 = "./cells.txt";
    read_cell_names(filename2);
    
    SolutionInstance *the_solution = create_random_solution_instance();
    the_solution->instance_id = 1;
    
    double temperature = 50.0;
    double cooling_rate = 0.000000001;
    cnt = 0;
    
    perform_simanneal(temperature, cooling_rate, the_solution, true, true);
    
    free_solution_instance(the_solution);
    cleanup();
    
    return 0;
}