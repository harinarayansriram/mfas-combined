# Connectome_C library 
A C library for connectome manipulation (currently just for finding the Minimum Feedback Arc Set from the top solutions from the Princeton Neuroscience Institute MFAS competition) with CFFI bindings for use with Python

## Connectome_C vs Connectome_C_Productivity
These are essentially the exact same, just that the latter is serial whereas the former is parallelized. Function calling convention is exactly the same, just omitting a "_parallel" in the function calls where applicable. For simplicity, the documentation will work with connectome_c but as noted they are practically the same.

## Installation
To build either the performance or productivity version, cd into the appropriate directory and install the requisite packages:
```bash
pip install setuptools cffi
```
Then run:
```bash
python3 build_module.py
```

to build the library.

Due to the use of rand_r threadsafe RNG in one of the algorithms, it's **unlikely that this will compile on Windows** but this should be fixed in the near future.

To use within a Python, create a file in the same directory and add the following import:
```py
from connectomics_c import ffi, lib
```

or
```py
from connectomics_c_productivity import ffi, lib
```
for the productivity version.

Then methods within the library (e.g. load_connectome) can be executed as such:
```py
connectome = lib.load_connectome("graph.csv")
```

Currently, these bindings are very one-to-one with the C code, so you will likely need to call the relevant functions to free the memory you allocate at the end of your program. 

There is also a test_script.py for both versions that uses all the major methods and data structures for additional examples.

## Data structures
### ConnectionNeighbor
An edge struct essentially

Properties:
- neighbor_dense_idx: int - a index mapping into an array of dense IDs (full node IDs)
- weight: int - the weight of the edge

### Connectome
A wrapper for all the parsed connectome data; it isn't needed to this extent in the actual code but it's more extensible for potential future usecases

Properties:
- num_nodes: long - the number of nodes in the connectome
- num_connections: long - number of arcs in the connectome
- total_weight: long long - sum of all the connection edges
- dense_idx_to_node_id: uint64_t[] - mapping between indices and the actual node IDs they represent

- outgoing: ConnectionNeighbor[][] - map of outgoing[from_id] -> edges sorted by to_id, weight
- incoming: ConnectionNeighbor[][] - map of incoming[to_id] -> edges sorted by from_id, weight

### SolutionInstance
Container for the solution that also has some other basic info about it

Properties:
- solution: int[] - ordered array of node IDs 
- node_to_position: int[] - mapping between node IDs and indices
- forward_score: long long - sum of weights of forward arcs
- solution_size: int - len(solution)

### BestSolutionStorage
Container for the best solution so far, almost identical to the SolutionInstance

Properties:
- best_solution_array: int[] - ordered array of node IDs of the best solution
- best_score: long long - sum of weights of forward arcs of the best solution
- solution_size: int - len(best_solution_array)


## Methods
### Connectome Management

#### load_connectome
Loads in the Connectome object from the file argument

Signature: 
```c
    Connectome* load_connectome(const char* graph_filename);
```
- Inputs: path to the connectome graph CSV file (string)
- Returns: Connectome object with relevant parsed data


#### free_connectome
Frees the allocated memory of the Connectome object returned by load_connectome. Should be called at the end of the program to free the memory. 

Signature:
```c
    void free_connectome(Connectome* connectome);
```
- Inputs: Connectome object (reference to Connectome)
- Returns: None

#### get_connection_weight
Finds the edge specified by the from and to node indices and - Returns its weight.

Signature:
```c
    int get_connection_weight(const Connectome* connectome, int from_dense_idx, int to_dense_idx);
```
- Inputs: Connectome object (reference to Connectome), index of the "from" node (int), index of the "to" node (int)
- Returns: weight of the edge (int)

#### get_connectome_num_nodes
A getter for the number of nodes in the connectome.

Signature:
```c
    long get_connectome_num_nodes(const Connectome* connectome);
```
- Inputs: Connectome object (reference to Connectome)
- Returns: number of nodes in the connectome (long)

#### get_dense_idx_to_node_id_array_ptr
Provides the mapping from indices to the actual Node IDs

Signature:
```c
    uint64_t* get_dense_idx_to_node_id_array_ptr(Connectome* connectome); 
```

- Inputs: Connectome object (reference to Connectome)
- Returns: uint64_t[] of indices --> Node IDs

### Solution Instance Management
#### create_solution_instance
Initializes a solution_instance object given the connectome, the solution array, and optionally calculating the initial score

Signature:
```c
    SolutionInstance* create_solution_instance(const Connectome* connectome, int* initial_solution_dense_array, bool calculate_initial_score);
```

- Inputs: Connectome object (reference to Connectome), initial solution array of indices (int[]), whether or not to calcluate the initial score to update inside the solution_instance object (bool)
- Returns: SolutionInstance object

#### create_random_solution_instance
Initializes a random solution for the given connectome

Signature:
```c
    SolutionInstance* create_random_solution_instance(const Connectome* connectome);
```

- Inputs: Connectome object (reference to Connectome)
- Returns: SolutionInstance object

#### free_solution_instance
Frees the memory allocated for a particular SolutionInstance object. This should be called at the end of the program to free the memory.

Signature:
```c
    void free_solution_instance(SolutionInstance* instance);
```

- Inputs: SolutionInstance object (reference to SolutionInstance)
- Returns: None

#### get_solution_score
Getter for the solution_score within a SolutionInstance object.

Signature:
```c
    long long get_solution_score(const SolutionInstance* instance);
```

- Inputs: SolutionInstance object (reference to SolutionInstance)
- Returns: the score of that solution as given within the object (long long)

#### get_solution_size
Getter for the solution_size within a SolutionInstance object.

Signature:
```c
    long get_solution_size(const SolutionInstance* instance);
```

- Inputs: SolutionInstance object (reference to SolutionInstance)
- Returns: the number of nodes within that solution as given within the object (long)

#### get_solution_array_ptr
Getter for the solution_array within a SolutionInstance object.

Signature:
```c
    int* get_solution_array_ptr(SolutionInstance* instance);
```

- Inputs: SolutionInstance object (reference to SolutionInstance)
- Returns: the solution array as given within the object (int[])

#### calculate_forward_score
Calculates the sum of the forward arcs for the Connectome given a particular solution.

Signature:
```c
    long long calculate_forward_score(const SolutionInstance* instance, const Connectome* connectome);
```
- Inputs: SolutionInstance object (reference to SolutionInstance), Connectome object (reference to Connectome)
- Returns: sum of forward arcs of the connectome given the solution

### Main Algorithms

#### run_simanneal_parallel
This is the Hashorva algorithm which runs standard Simulated Annealing over pair order-swaps with added parallelization.

Signature:
```c
void run_simanneal_parallel(
        SolutionInstance* instance,
        const Connectome* connectome,
        double initial_temperature,
        double cooling_rate,
        long long max_iterations,
        long long iterations_per_log,
        bool log_progress
    );
```

- Inputs: SolutionInstance object, Connectome object, initial temperature for the simulated annealing (double), rate of cooling for the schedule (double), the maximum number of iterations to execute (long long), the number of iterations per log output (long long), whether to log progress (bool)
- Outputs: None, but mutates the SolutionInstance object


#### run_simanneal_parallel_with_toposhuffle
This is the Bader algorithm which runs parallelized Simulated Annealing while interleaving runs of an algorithm called Toposhuffle that essentially performs a randomized topological sort on the forward edges and looks for new forward edges in the resultant ordering.

Signature:
```c
void run_simanneal_parallel_with_toposhuffle(
        SolutionInstance* instance,
        const Connectome* connectome,
        int num_threads,
        long long iterations_per_thread,
        int updates_per_thread,
        double tmin,
        double tmax,
        int go_back_to_best_window,
        int toposhuffle_frequency,
        int verbosity,
        BestSolutionStorage* global_best
    );
```

- Inputs: SolutionInstance object, Connectome object, the number of threads on which the execute the algorithm (int), the number of iterations to execute per thread (long long), the number of logging updates desired per thread (int), the start temperature (double), the final temperature (double), the maxmimum number of moves until the state is reverted back to the previous best (int), the frequency at which to execute the toposhuffle algorithm (int), the verbosity of the logging (0,1,2, or 10) (int), and a BestSolutionStorage object
- Returns: None, but mutates the SolutionInstance and BestSolutionStorage

#### run_okubo_local_search
This is the Okubo local search algorithm.

Signature:
```c
    void run_okubo_local_search(
        SolutionInstance* instance,
        const Connectome* connectome,
        int n_epochs,
        bool log_progress,
        int log_interval
    );
```

- Inputs: SolutionInstance object, Connectome object, number of epochs to run the local search (int), whether to log progress (bool), and how often to log progress (int)
- Returns: None, but mutates the SolutionInstance

### Best Solution Management

#### create_best_solution_storage
A constructor that allocates an empty BestSolutionStorage object

Signature:
```c
    BestSolutionStorage* create_best_solution_storage();
```

- Inputs: None
- Returns: BestSolutionStorage object

#### free_best_solution_storage
Frees the memory allocated for the BestSolutionStorage object. This should be run at the end of the program to free the allocated memory.

Signature:
```c
    void free_best_solution_storage(BestSolutionStorage* storage);
```

- Inputs: BestSolutionStorage object
- Returns: None

#### init_best_solution_storage
Initializes the BestSolutionStorage object with the information from a particular SolutionInstance.

Signature:
```c
    void init_best_solution_storage(BestSolutionStorage* storage, const SolutionInstance* instance);
```

- Inputs: BestSolutionStorage object, SolutionInstance object
- Returns: None, but mutates the BestSolutionStorage

#### get_best_solution_score
Getter for the best_solution_score of the BestSolutionStorage object

Signature:
```c
    long long get_best_solution_score(const BestSolutionStorage* storage);
```

- Inputs: BestSolutionStorage object
- Returns: best solution score as given in the object (long long)

#### get_best_solution_array_ptr
Getter for the solution array from the BestSolutionStorage object

Signature:
```c
    int* get_best_solution_array_ptr(BestSolutionStorage* storage);
```

- Inputs: BestSolutionStorage object
- Returns: Array of indices of the best solution given in the object (int[])

### Utility

#### seed_rng
Seed the RNG for the library to get consistent results

Signature:
```c
    void seed_rng(unsigned int seed);
```

- Inputs: seed (unsigned int)
- Returns: None