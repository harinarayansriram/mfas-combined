# Solution for the Minimum Feedback Challenge
This repo contains the Team CIBR's second place solution for the Minimum Feedback Challenge (members: Xin Zheng, Runxuan Tang, and [Tatsuo Okubo](https://cibr.ac.cn/science/team/detail/975?language=en)). The [competition](https://codex.flywire.ai/app/mfas_challenge) was organized by the FlyWire team at Princeton University, and we thank the organizers for this opportunity.

- `/data`: contains the connectome graph data and the benchmark solution downloaded from competition website 
- `results.csv`: our final solution (total sum of feedforward weights: 35,374,656, 84%) 
- `environment.yml`: We have included this file that could be used to recreate the conda environment, but our code only uses basic functions of Numpy and Pandas, so it should work on most versions.

# Construction methods
We tested two methods for generating a solution from scratch. Both of these methods ran very quickly (several seconds).

## Heuristic based on Becker (1967)
This is a simple heuristic that tries to put the nodes with many outgoing edges towards the left and many incoming edges towards the right by sorting neurons based on the score calculated as (out_degree / in_degree), from highest to lowest. `Beckers_heuristic.ipynb` demonstrates this algorithm on the FlyWire data, which leads to the score 30,136,267 (71.9%). 

## Graph reduction + divide and conquer
This is a cutting-edge method described in [Xiong...Khoussainov (Computers & Operations Research, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0305054824001965). The C++ code is available from [the author's GitHub repo](https://github.com/forgottencsc/FindingSmallFeedbackArcSetOnLargeGraphs). Using RASstar algorithm on the FlyWire leads to a high score of 33,037,112 (78.8%) and the result (33037112.csv) is included for reference.

# Improvement methods
We used **iterated local search (ILS)** to improve on the initial solution created by construction methods. While the two construction methods mentioned above gave us higher scores compared to the baseline, we used the benchmark solution as our initial solution. Our local search algorithm is based on [Schiavinotto & Stützle (J. of Mathematical Modelling and Algorithms, 2004)](https://link.springer.com/article/10.1023/B:JMMA.0000049426.06305.d8), which uses sequential application of swapping neighboring nodes to explore insertion neighborhood.

- `LS_insertion.py`: main code implementing local search insertion algorithm. 
- `utils.py`: utility functions supporting the local search implementation. 
- `LS_insertion_step_by_step.ipynb`: a Jupyter notebook demonstrating the local search insertion algorithm step by step.

## Instructions to run the code

To execute the code, you can use the following command in your terminal:

```bash
python LS_insertion.py --data_dir <path_to_data_directory> --init_sol <initial_solution> --n_epochs <number_of_epochs> --output_dir <path_to_output_directory> --perturbation <perturbation_percentage> --seed <random_seed>
```

**Arguments**

- `-d`, `--data_dir`: path to the data directory (default: ../data)
- `-i`, `--init_sol`: name of the initial solution file without extension (default: benchmark)
- `-n`, `--n_epochs`: number of epochs for the local search (default: 10)
- `-o`, `--output_dir`: path to the output directory (default: ../output)
- `-p`, `--perturbation`: percentage to perturb the solution (default: 0.01)
- `-s`, `--seed`: random seed for reproducibility (default: 42)

**Example command**

```bash
python LS_insertion.py --data_dir ../data --init_score benchmark --n_epochs 10 --output_dir ../output --perturbation 0.01 --seed 42
```