# a1-QIXIN-ACT
## Clocking CPU parallelism
### (a)
+ Python file to run original code: [run_simulation.py](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1a/run_simulation.py)
+ Python file to compile original code: [compile_health_index.py](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1a/compile_health_index.py)
+ Python file to run compiled code: [run_compiled_silumation.py](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1a/run_compiled_silumation.py)
+ Sbatch file to run codes on Midway cluster: [1a_submit.sbatch](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1a/1a_submit.sbatch) 
+ Finding: Based on the [output file](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1a/1a_simulation_runs_19326770.out), elapsed time with AOT-compiled code is 0.202 seconds while elapsed time without AOT-compiled code is 3.015 seconds. The speedup is obvious - the pre-compiled code is about 15 times faster than original code. 
### (b)
+ Python file to compile original code: [compile_health_index.py](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1b/compile_health_index.py)
+ Python file to use multiple cores to run the compiled code: [mpi_health_simulation.py](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1b/mpi_health_simulation.py)
+ Sbatch file to run codes using multiple cores on Midway cluster: [1b_submit.sbatch](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1b/1b_submit.sbatch)
+ Jupter notebook to produce plot based on output: [plotting.ipynb](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1b/plotting.ipynb)
+ Plot for q1b: ![1b_plot](https://github.com/macs30123-s24/a1-QIXIN-ACT/blob/main/q1b/1b_plot.png)
### (c)
Firstly, parallelization overhead plays a pivotal role. Each additional core introduces its own overhead for communication. This includes the time taken to set up inter-process communication, distribute the workload, synchronize the execution across cores, and gather the results. As the number of cores increases, so does the overhead, which in turn diminishes the returns on speedup.

Secondly, Amdahl's Law articulates a limitation in parallel computing. It suggests that the speedup of a program using multiple processors is restricted by the sequential fraction of the program. If there's any part that must run sequentially, it creates a bottleneck that limits the overall speedup.

Another consideration is resource contention. Multiple cores often vie for the same resources, like memory or input/output capabilities. Such contention can cause delays, as each core waits for access, thus impeding the speedup.

Cache effects also come into play in multi-core processors, which have intricate cache hierarchies. More cores competing for limited cache can lead to cache contention and increased cache misses, resulting in time-consuming memory accesses.

Furthermore, communication bandwidth can also be a bottleneck. The system has finite bandwidth for data transfer, and as the number of cores increases, so does the volume of data that needs to be moved, potentially saturating the available bandwidth.

Startup costs associated with initializing parallel processes can't be ignored either. These costs become more evident when the amount of computation required per core is relatively low.

Lastly, non-computational delays, such as those stemming from job scheduling systems, can affect the observed computation time. These delays won't benefit from additional cores.

That's why while adding more cores initially decreases computation time significantly, the marginal time saved decreases with each additional core.
