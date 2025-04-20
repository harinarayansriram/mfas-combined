#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>
#include <omp.h>

#define DEFAULT_GO_BACK_TO_BEST_WINDOW 100000000
#define TOPOSHUFFLE_FREQUENCY 50000000
#define DEFAULT_UPDATES 2500
#define DEFAULT_TMIN 0.001
#define DEFAULT_TMAX 0.1
#define DEFAULT_VERBOSITY 10
#define DEFAULT_ITERATIONS_TO_SAVE 15000000

typedef struct {
    int node;
    int weight;
} Edge;

typedef struct {
    time_t start_time;
    int iterations;
    int cumulative_iterations;
    int energy;
} LogEntry;

typedef struct {
    int score;
    int id;
} Result;

typedef struct {
    char* in_file;
    char* out_file;
    char* log_file;
    char* graph_file;
    int max_iters;
    bool infinite_iters;
    int verbosity;
    int updates;
    int iters_per_thread;
    int threads;
    int go_back_to_best_window;
    double tmin;
    double tmax;
} CLIArgs;

Edge** out_adj;
Edge** in_adj;
int* out_adj_size;
int* in_adj_size;

long long** graph;
int graph_size;
int* node_to_sorted_idx;
long long* sorted_idx_to_node;
int node_count;

int* state;
LogEntry* logs;
int log_count;

int threads, iters_per_thread, max_iters, updates, verbosity, go_back_to_best_window;
double tmin, tmax;
char *in_path, *out_path, *graph_path, *log_path;

volatile sig_atomic_t interrupted = 0;

// In case your RAND_MAX is 2**15 for whatever reason
uint64_t random(){
    return ((uint64_t) rand() << 0) ^ ((uint64_t) rand() << 15) ^ ((uint64_t) rand() << 30) ^ ((uint64_t) rand() << 45) ^ (((uint64_t) rand() & 0xf) << 60);
}

// Handle interruption signals
void handle_signal(int sig) {
    interrupted = 1;
}

// Handle errors
void handle_error(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(EXIT_FAILURE);
}

// Parse command line arguments
void parse_cli_args(int argc, char** argv, CLIArgs* args) {
    // Set default values
    args->verbosity = DEFAULT_VERBOSITY;
    args->updates = DEFAULT_UPDATES;
    args->go_back_to_best_window = DEFAULT_GO_BACK_TO_BEST_WINDOW;
    args->tmin = DEFAULT_TMIN;
    args->tmax = DEFAULT_TMAX;
    args->infinite_iters = false;
    args->max_iters = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) {
            args->in_file = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            args->out_file = argv[++i];
        } else if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
            args->graph_file = argv[++i];
        } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            args->log_file = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            args->threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters-per-thread") == 0 && i + 1 < argc) {
            args->iters_per_thread = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--infinite-iters") == 0) {
            args->infinite_iters = true;
        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            args->max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbosity") == 0 && i + 1 < argc) {
            args->verbosity = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--updates") == 0 && i + 1 < argc) {
            args->updates = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--go-back-to-best-window") == 0 && i + 1 < argc) {
            args->go_back_to_best_window = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tmin") == 0 && i + 1 < argc) {
            args->tmin = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tmax") == 0 && i + 1 < argc) {
            args->tmax = atof(argv[++i]);
        }
    }

    // Validate required arguments
    if (!args->in_file || !args->out_file || !args->graph_file || args->threads == 0 || args->iters_per_thread == 0) {
        printf("One or more required flags not provided\n");
        exit(EXIT_FAILURE);
    }

    // Check for mutually exclusive flags
    if (!args->infinite_iters && args->max_iters == 0) {
        printf("Error: --infinite-iters or --max-iters must be specified\n");
        exit(EXIT_FAILURE);
    }

    if (args->infinite_iters && args->max_iters != 0) {
        printf("Error: --infinite-iters and --max-iters are mutually exclusive\n");
        exit(EXIT_FAILURE);
    }

    // Validate temperature values
    if (args->tmin == 0.0) {
        printf("tmin must be non-zero\n");
        exit(EXIT_FAILURE);
    }

    if (args->tmin > args->tmax) {
        printf("tmin must be less than tmax\n");
        exit(EXIT_FAILURE);
    }

    // Validate verbosity
    if (args->verbosity < 0 || args->verbosity > 10) {
        printf("verbosity must be between 0 and 10\n");
        exit(EXIT_FAILURE);
    }

    if (args->verbosity > 2 && args->verbosity != 10) {
        printf("verbosity must be 0, 1, 2, or 10; high verbosity is 10\n");
        exit(EXIT_FAILURE);
    }
}

// Function to read CSV files
char*** read_csv(const char* file, int* rows, int* cols) {
    FILE* fp = fopen(file, "r");
    if (!fp) {
        handle_error("Failed to open file");
    }

    // Count rows and estimate columns
    char buffer[4096];
    *rows = 0;
    *cols = 0;
    
    if (fgets(buffer, sizeof(buffer), fp)) {
        (*rows)++;
        char* token = strtok(buffer, ",");
        while (token) {
            (*cols)++;
            token = strtok(NULL, ",");
        }
    }
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        (*rows)++;
    }
    
    // Allocate memory for data
    char*** data = (char***)malloc(*rows * sizeof(char**));
    for (int i = 0; i < *rows; i++) {
        data[i] = (char**)malloc(*cols * sizeof(char*));
        for (int j = 0; j < *cols; j++) {
            data[i][j] = (char*)malloc(256 * sizeof(char));
        }
    }
    
    // Read data
    rewind(fp);
    for (int i = 0; i < *rows; i++) {
        if (!fgets(buffer, sizeof(buffer), fp)) {
            break;
        }
        buffer[strcspn(buffer, "\r\n")] = 0; // Remove newline
        
        char* token = strtok(buffer, ",");
        int j = 0;
        while (token && j < *cols) {
            strcpy(data[i][j], token);
            token = strtok(NULL, ",");
            j++;
        }
    }
    
    fclose(fp);
    return data;
}

// Function to write results to CSV
void write_to_csv(const char* file, int* state, long long* sorted_idx_to_node, int size) {
    FILE* fp = fopen(file, "w");
    if (!fp) {
        handle_error("Failed to open output file");
    }
    
    // Write header
    fprintf(fp, "Node ID,Order\n");
    
    // Write data
    for (int i = 0; i < size; i++) {
        fprintf(fp, "%ld,%d\n", sorted_idx_to_node[i], state[i]);
    }
    
    fclose(fp);
}

// Function to write logs to CSV
void write_logs(const char* file, LogEntry* logs, int log_count) {
    FILE* fp = fopen(file, "w");
    if (!fp) {
        handle_error("Failed to open log file");
    }
    
    // Write header
    fprintf(fp, "Start Time,Iterations,Cumulative Iterations,Energy\n");
    
    // Write data
    for (int i = 0; i < log_count; i++) {
        fprintf(fp, "%ld,%d,%d,%d\n", logs[i].start_time, logs[i].iterations, logs[i].cumulative_iterations, logs[i].energy);
    }
    
    fclose(fp);
}

// Function to calculate energy
int energy(int* state) {
    long long total = 0;
    for (int i = 0; i < graph_size; i++) {
        long long source = graph[i][0];
        long long target = graph[i][1];
        long long weight = graph[i][2];
        
        if (state[node_to_sorted_idx[source]] < state[node_to_sorted_idx[target]]) {
            total += weight;
        }
    }
    return (int)(-total);
}

// Function to compute change in energy
int compute_change(int* nodes, int* prev, int* cur, int* state) {
    int delta = 0;
    int a = nodes[0];
    int b = nodes[1];
    
    for (int i = 0; i < 2; i++) {
        int node = nodes[i];
        int prev_pos = prev[i];
        int cur_pos = cur[i];
        
        for (int j = 0; j < out_adj_size[node]; j++) {
            int next_node = out_adj[node][j].node;
            int weight = out_adj[node][j].weight;
            int next_pos = state[next_node];
            
            if (next_node == a) {
                next_pos = cur[0];
            }
            
            if (next_node == b) {
                next_pos = cur[1];
            }
            
            if (prev_pos < state[next_node]) {
                delta -= weight;
            }
            if (cur_pos < next_pos) {
                delta += weight;
            }
        }
        
        for (int j = 0; j < in_adj_size[node]; j++) {
            int prev_node = in_adj[node][j].node;
            int weight = in_adj[node][j].weight;
            
            if (prev_node == a || prev_node == b) {
                continue;
            }
            
            if (state[prev_node] < prev_pos) {
                delta -= weight;
            }
            if (state[prev_node] < cur_pos) {
                delta += weight;
            }
        }
    }
    
    return delta;
}

// Function to make an impactful move
int move_if_impactful(int* state, int* a, int* b) {
    int n = node_count;
    
    // Select a random node
    *a = (int)(random() % n);
    
    // Find a node that shares an edge with a
    while (out_adj_size[*a] == 0 && in_adj_size[*a] == 0) {
        *a = (int)(random() % n);
    }
    
    int b_i = (int)(random() % (out_adj_size[*a] + in_adj_size[*a]));
    
    if (b_i < out_adj_size[*a]) {
        *b = out_adj[*a][b_i].node;
    } else {
        b_i -= out_adj_size[*a];
        *b = in_adj[*a][b_i].node;
    }
    
    // Capture current and proposed orders
    int current_order[2] = {state[*a], state[*b]};
    int proposed_order[2] = {current_order[1], current_order[0]};
    
    // Compute energy change
    int indices[2] = {*a, *b};
    int energy_delta = compute_change(indices, current_order, proposed_order, state);
    
    // Apply the swap
    int temp = state[*a];
    state[*a] = state[*b];
    state[*b] = temp;
    
    return -energy_delta;
}

// Function to perform random topological sort
void random_toposort(int* state) {
    int n = node_count;
    int* ordering = (int*)malloc(n * sizeof(int));
    Edge*** cur_out_adj = (Edge***)malloc(n * sizeof(Edge**));
    int* cur_out_adj_size = (int*)calloc(n, sizeof(int));
    int* queue = (int*)malloc(n * sizeof(int));
    int queue_size = 0;
    int ordering_size = 0;
    int* indeg = (int*)calloc(n, sizeof(int));
    
    // Initialize current adjacency lists
    for (int i = 0; i < n; i++) {
        cur_out_adj[i] = (Edge**)malloc(out_adj_size[i] * sizeof(Edge*));
        cur_out_adj_size[i] = 0;
    }
    
    // Build current graph based on current state
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_adj_size[i]; j++) {
            int neighbor = out_adj[i][j].node;
            if (state[i] < state[neighbor]) {
                cur_out_adj[i][cur_out_adj_size[i]] = &out_adj[i][j];
                cur_out_adj_size[i]++;
                indeg[neighbor]++;
            }
        }
    }
    
    // Find nodes with no incoming edges
    for (int i = 0; i < n; i++) {
        if (indeg[i] == 0) {
            queue[queue_size++] = i;
        }
    }
    
    int tot_weight = 0;
    
    // Random topological sort
    while (ordering_size < n) {
        if (queue_size == 0) {
            printf("Error: Graph has a cycle\n");
            break;
        }
        
        // Pick a random node from the queue
        int q_idx = (int)(random() % queue_size);
        int node = queue[q_idx];
        
        // Swap with last element and pop
        queue[q_idx] = queue[queue_size - 1];
        queue_size--;
        
        // Add to ordering
        ordering[ordering_size++] = node;
        
        // Process outgoing edges
        for (int i = 0; i < cur_out_adj_size[node]; i++) {
            int neighbor = cur_out_adj[node][i]->node;
            tot_weight += cur_out_adj[node][i]->weight;
            indeg[neighbor]--;
            if (indeg[neighbor] == 0) {
                queue[queue_size++] = neighbor;
            }
        }
    }
    
    if (verbosity >= 2) {
        printf("Weight of topsort %d\n", tot_weight);
    }
    
    // Update state based on new ordering
    for (int i = 0; i < n; i++) {
        state[ordering[i]] = i;
    }
    
    // Free allocated memory
    free(ordering);
    for (int i = 0; i < n; i++) {
        free(cur_out_adj[i]);
    }
    free(cur_out_adj);
    free(cur_out_adj_size);
    free(queue);
    free(indeg);
}

// Function to perform parallel annealing
void parallel_anneal(int id, double tmin, double tmax, int steps, int* state, Result* results) {
    int step = 0;
    double t = tmax;
    int e = energy(state);
    
    if (tmin <= 0.0) {
        printf("Tmin needs to be nonzero\n");
        results[id].score = e;
        results[id].id = id;
        return;
    }
    
    double tfactor = -log(tmax / tmin);
    
    // Create copies of state
    int* prev_state = (int*)malloc(node_count * sizeof(int));
    memcpy(prev_state, state, node_count * sizeof(int));
    int prev_energy = e;
    
    int* best_state = (int*)malloc(node_count * sizeof(int));
    memcpy(best_state, state, node_count * sizeof(int));
    int best_energy = e;
    
    int trials = 0, accepts = 0, improves = 0;
    
    int update_wavelength;
    if (updates > 1) {
        update_wavelength = steps / updates;
    } else {
        update_wavelength = -1;
    }
    
    if (verbosity == 10) {
        printf("Steps: %d Temperature: %f Energy: %d\n", step, t, e);
    }
    
    for (step = 0; step < steps && !interrupted; step++) {
        // Update temperature
        if (tmax - tmin == 0.0) {
            t = tmax;
        } else {
            t = tmax * exp(tfactor * (double)step / (double)steps);
        }
        
        // Make a move
        int a, b;
        int dE = move_if_impactful(state, &a, &b);
        e += dE;
        trials++;
        
        // Periodically go back to best state
        if (step % go_back_to_best_window == 0) {
            memcpy(state, best_state, node_count * sizeof(int));
            e = best_energy;
        }

        // Acceptance criterion
        if (dE > 0 && exp(-dE / t) < (double)random() / UINT64_MAX && step % go_back_to_best_window != 0) {
            // Restore previous state (swap back)
            int temp = state[a];
            state[a] = state[b];
            state[b] = temp;
            e = prev_energy;
        } else {
            accepts++;
            if (dE < 0) {
                improves++;
            }
            
            if (go_back_to_best_window != 0) {
                int temp = prev_state[a];
                prev_state[a] = prev_state[b];
                prev_state[b] = temp;
            } else {
                memcpy(prev_state, state, node_count * sizeof(int));
            }
            
            prev_energy = e;
            
            if (e < best_energy) {
                best_energy = e;
                memcpy(best_state, state, node_count * sizeof(int));
            }
        }
        
        // Periodic toposhuffle
        if (TOPOSHUFFLE_FREQUENCY > 0 && step % TOPOSHUFFLE_FREQUENCY == 0) {
            if (verbosity >= 2) {
                printf("Toposorting during annealing\n");
            }
            random_toposort(state);
        }
        
        // Print updates
        if (updates > 1 && update_wavelength > 0 && step % update_wavelength == 0) {
            float accept_percent = (float)accepts / trials * 100;
            float improve_percent = (float)improves / trials * 100;
            printf("Steps: %d\tTemperature: %f\tEnergy: %d\tAccept: %.5f%%\tImprove: %.5f%%\n", 
                   step, t, e, accept_percent, improve_percent);
            trials = 0;
            accepts = 0;
            improves = 0;
        }
    }
    
    // Use best state
    memcpy(state, best_state, node_count * sizeof(int));
    
    if (verbosity >= 1) {
        printf("\nBest energy: %d\n", best_energy);
    }
    
    results[id].score = best_energy;
    results[id].id = id;
    
    // Free allocated memory
    free(prev_state);
    free(best_state);
}

int main(int argc, char** argv) {
    // Parse command line arguments
    CLIArgs args = {0};
    parse_cli_args(argc, argv, &args);
    
    // Set global variables from args
    threads = args.threads;
    iters_per_thread = args.iters_per_thread;
    if (args.infinite_iters) {
        max_iters = -1;
    } else {
        max_iters = args.max_iters;
    }
    go_back_to_best_window = args.go_back_to_best_window;
    tmin = args.tmin;
    tmax = args.tmax;
    verbosity = args.verbosity;
    
    if (verbosity == 10) {
        updates = args.updates;
    } else {
        updates = 0;
    }
    
    in_path = args.in_file;
    out_path = args.out_file;
    graph_path = args.graph_file;
    
    if (args.log_file) {
        log_path = args.log_file;
    } else {
        log_path = (char*)malloc(strlen(in_path) + 9);
        sprintf(log_path, "%s_log.csv", in_path);
    }
    
    // Initialize random seed
    srand(time(NULL));
    
    // Read graph file
    int rows, cols;
    char*** graph_records = read_csv(graph_path, &rows, &cols);
    
    // Initialize node set
    int max_node_id = 0;
    for (int i = 1; i < rows; i++) {
        long long source_id = atoll(graph_records[i][0]);
        long long target_id = atoll(graph_records[i][1]);
        
        if (source_id > max_node_id) max_node_id = source_id;
        if (target_id > max_node_id) max_node_id = target_id;
    }
    
    // Allocate memory for graph
    graph_size = rows - 1;  // Excluding header row
    graph = (long long**)malloc(graph_size * sizeof(long long*));
    for (int i = 0; i < graph_size; i++) {
        graph[i] = (long long*)malloc(3 * sizeof(long long));
    }
    
    // Create temporary hash set for nodes
    int* node_exists = (int*)calloc(max_node_id + 1, sizeof(int));
    
    // Process graph records
    for (int i = 1; i < rows; i++) {
        long long source_id = atoll(graph_records[i][0]);
        long long target_id = atoll(graph_records[i][1]);
        long long edge_weight = atoi(graph_records[i][2]);
        
        graph[i - 1][0] = source_id;
        graph[i - 1][1] = target_id;
        graph[i - 1][2] = edge_weight;
        
        node_exists[source_id] = 1;
        node_exists[target_id] = 1;
    }
    
    // Count unique nodes
    node_count = 0;
    for (int i = 0; i <= max_node_id; i++) {
        if (node_exists[i]) {
            node_count++;
        }
    }
    
    // Create sorted node list
    sorted_idx_to_node = (long long*)malloc(node_count * sizeof(long long));
    int idx = 0;
    for (int i = 0; i <= max_node_id; i++) {
        if (node_exists[i]) {
            sorted_idx_to_node[idx++] = i;
        }
    }
    
    // Create node to index mapping
    node_to_sorted_idx = (int*)malloc((max_node_id + 1) * sizeof(int));
    for (int i = 0; i < node_count; i++) {
        node_to_sorted_idx[sorted_idx_to_node[i]] = i;
    }
    
    // Initialize adjacency lists
    out_adj = (Edge**)malloc(node_count * sizeof(Edge*));
    in_adj = (Edge**)malloc(node_count * sizeof(Edge*));
    out_adj_size = (int*)calloc(node_count, sizeof(int));
    in_adj_size = (int*)calloc(node_count, sizeof(int));
    
    // Count edges for each node
    for (int i = 0; i < graph_size; i++) {
        long long source_id = graph[i][0];
        long long target_id = graph[i][1];
        
        int source_idx = node_to_sorted_idx[source_id];
        int target_idx = node_to_sorted_idx[target_id];
        
        out_adj_size[source_idx]++;
        in_adj_size[target_idx]++;
    }
    
    // Allocate memory for adjacency lists
    for (int i = 0; i < node_count; i++) {
        out_adj[i] = (Edge*)malloc(out_adj_size[i] * sizeof(Edge));
        in_adj[i] = (Edge*)malloc(in_adj_size[i] * sizeof(Edge));
    }
    
    // Reset counts for filling
    int* out_count = (int*)calloc(node_count, sizeof(int));
    int* in_count = (int*)calloc(node_count, sizeof(int));
    
    // Fill adjacency lists
    for (int i = 0; i < graph_size; i++) {
        long long source_id = graph[i][0];
        long long target_id = graph[i][1];
        long long weight = graph[i][2];
        
        int source_idx = node_to_sorted_idx[source_id];
        int target_idx = node_to_sorted_idx[target_id];
        
        out_adj[source_idx][out_count[source_idx]].node = target_idx;
        out_adj[source_idx][out_count[source_idx]].weight = weight;
        out_count[source_idx]++;
        
        in_adj[target_idx][in_count[target_idx]].node = source_idx;
        in_adj[target_idx][in_count[target_idx]].weight = weight;
        in_count[target_idx]++;
    }
    
    // Initialize state
    state = (int*)malloc(node_count * sizeof(int));
    
    // Read initial state from input file
    int state_rows, state_cols;
    char*** state_records = read_csv(in_path, &state_rows, &state_cols);
    
    for (int i = 1; i < state_rows; i++) {
        long long node_id = atoll(state_records[i][0]);
        int order = atoi(state_records[i][1]);
        state[node_to_sorted_idx[node_id]] = order;
    }
    
    // Set up signal handlers
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    
    // Initialize logs
    logs = (LogEntry*)malloc(sizeof(LogEntry) * 1000);  // Initial capacity
    log_count = 0;
    
    // Add initial log entry
    logs[log_count].start_time = time(NULL);
    logs[log_count].iterations = 0;
    logs[log_count].cumulative_iterations = 0;
    logs[log_count].energy = energy(state);
    log_count++;
    
    // Main optimization loop
    int outer_loop_count = 0;
    
    while (1) {
        outer_loop_count++;
        if (max_iters > 0 && outer_loop_count > max_iters / iters_per_thread) {
            break;
        }
        
        if (interrupted) {
            if (verbosity >= 1) {
                printf("Interrupt received, saving results and exiting...\n");
                printf("Iterations: %d\n", outer_loop_count * iters_per_thread);
            }
            write_to_csv(out_path, state, sorted_idx_to_node, node_count);
            write_logs(log_path, logs, log_count);
            break;
        }
        
        // Allocate memory for thread states and results
        int** states_arr = (int**)malloc(threads * sizeof(int*));
        for (int i = 0; i < threads; i++) {
            states_arr[i] = (int*)malloc(node_count * sizeof(int));
            memcpy(states_arr[i], state, node_count * sizeof(int));
        }
        
        Result* results = (Result*)malloc(threads * sizeof(Result));
        
        // Run parallel annealing using OpenMP
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < threads; i++) {
            if (verbosity >= 2) {
                printf("Thread %d started\n", i);
            }
            parallel_anneal(i, tmin, tmax, iters_per_thread, states_arr[i], results);
            if (verbosity >= 2) {
                printf("Thread %d finished\n", i);
            }
        }
        
        // Collect results from threads
        int bestEnergy = results[0].score;
        int bestIndex = 0;
        for (int i = 1; i < threads; i++) {
            if (results[i].score < bestEnergy) {
                bestEnergy = results[i].score;
                bestIndex = i;
            }
        }
        
        // Update logs with cumulative iterations
        int cumulative_iters = outer_loop_count * iters_per_thread;
        logs[log_count].start_time = time(NULL);
        logs[log_count].iterations = iters_per_thread;
        logs[log_count].cumulative_iterations = cumulative_iters;
        logs[log_count].energy = bestEnergy;
        log_count++;
        
        // Update global state with best result from this outer loop
        memcpy(state, states_arr[bestIndex], node_count * sizeof(int));
        
        if (verbosity >= 1) {
            printf("Final energy: %d\n", bestEnergy);
            printf("Iterations: %d\n", cumulative_iters);
            printf("------------------------------------------\n");
        }
        
        // Optionally save intermediate results
        if (iters_per_thread >= DEFAULT_ITERATIONS_TO_SAVE ||
           (iters_per_thread < DEFAULT_ITERATIONS_TO_SAVE && outer_loop_count % (DEFAULT_ITERATIONS_TO_SAVE / iters_per_thread) == 0)) {
            if (verbosity >= 2) {
                printf("Saving results...\n");
            }
            write_to_csv(out_path, state, sorted_idx_to_node, node_count);
            write_logs(log_path, logs, log_count);
        }
        
        // Free allocated memory for thread states and results
        for (int i = 0; i < threads; i++) {
            free(states_arr[i]);
        }
        free(states_arr);
        free(results);
    }
    
    // After finishing the outer loop, write final results and logs
    write_to_csv(out_path, state, sorted_idx_to_node, node_count);
    write_logs(log_path, logs, log_count);
    
    return 0;
}
