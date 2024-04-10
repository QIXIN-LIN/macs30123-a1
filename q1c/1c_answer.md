Firstly, parallelization overhead plays a pivotal role. Each additional core introduces its own overhead for communication. This includes the time taken to set up inter-process communication, distribute the workload, synchronize the execution across cores, and gather the results. As the number of cores increases, so does the overhead, which in turn diminishes the returns on speedup.

Secondly, Amdahl's Law articulates a limitation in parallel computing. It suggests that the speedup of a program using multiple processors is restricted by the sequential fraction of the program. If there's any part that must run sequentially, it creates a bottleneck that limits the overall speedup.

Another consideration is resource contention. Multiple cores often vie for the same resources, like memory or input/output capabilities. Such contention can cause delays, as each core waits for access, thus impeding the speedup.

Cache effects also come into play in multi-core processors, which have intricate cache hierarchies. More cores competing for limited cache can lead to cache contention and increased cache misses, resulting in time-consuming memory accesses.

Furthermore, communication bandwidth can also be a bottleneck. The system has finite bandwidth for data transfer, and as the number of cores increases, so does the volume of data that needs to be moved, potentially saturating the available bandwidth.

Startup costs associated with initializing parallel processes can't be ignored either. These costs become more evident when the amount of computation required per core is relatively low.

Lastly, non-computational delays, such as those stemming from job scheduling systems, can affect the observed computation time. These delays won't benefit from additional cores.

That's why while adding more cores initially decreases computation time significantly, the marginal time saved decreases with each additional core.






