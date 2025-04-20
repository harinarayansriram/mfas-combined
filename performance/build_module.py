from cffi import FFI

ffi = FFI()

ffi.cdef("""
typedef struct {
    int from_id;
    int to_id;
    int weight;
} Connection;

typedef struct {
    int forward;
    int backward;
    double ratio; 
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

extern int MAX_CELL_ID;
extern Connection *connection_by_cells_id_dict;
extern long connection_dict_count;
extern int **outgoing_connections;
extern int *outgoing_connections_size;
extern int **incoming_connections;
extern int *incoming_connections_size;

extern char **cell_names;
extern int *cell_names_r_keys;
extern int *cell_names_r_values;
extern int cell_names_count;

extern long cnt;
extern long last_change_step;
extern long last_step;
extern double last_temperature;

extern BestSolution best_solution;

SolutionInstance* create_solution_instance(int *solution, int solution_size, int forward_score);
SolutionInstance* create_random_solution_instance();
SolutionInstance* read_solution_from_file(const char *filename, bool randomize);
Score calculate_score(SolutionInstance *instance);
void swap(SolutionInstance *instance, int i1, int i2, double temperature);
bool check_best_solution(BestSolution *best, SolutionInstance *instance);
uint64_t random();

void read_connections(char* filename);
void read_cell_names(char *filename);
void perform_simanneal(double temperature, double cooling_rate, SolutionInstance *the_solution, bool log_to_file, bool write_updated_solution);
void free_solution_instance(SolutionInstance *instance);
void cleanup();
""")

ffi.set_source(
    "simanneal_hash",
    """
    #include "exposed_elements.h"
    """,
    sources=["hashorva/solution_instance.c", "hashorva/program.c"],
    extra_compile_args=["-O2"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)