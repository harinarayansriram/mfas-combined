import subprocess
import datetime
import json

n_parallel = 10
n_series = 1
n_cpu = 4
now = datetime.datetime.now()
timestr = now.strftime("%Y-%m-%d-%H:%M:%S")
commands = []
for i in range(n_parallel):
    command = f'bsub -J "flyorder_{i}" -n {n_cpu} -gpu "num=1" -q gpu_a100 -o run_results/reo/out_{i}.log python3 -u reorder_cluster.py {i}'
    commands.append(command)

for i, cmd in enumerate(commands):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Check if command was successful
    if process.returncode != 0:
        print(f"Command '{cmd}' failed with error:\n{stderr.decode()}")
    else:
        print(f"Command '{cmd}' succeeded with output:\n{stdout.decode()}")