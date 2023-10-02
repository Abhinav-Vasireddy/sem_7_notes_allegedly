#### Introduction
- Moore's Law - "Computers will become fast very fast" ~ "Speed will double every 18 months"
- "Density" of these chips going up every year
- "Why Parallel?"
	- Do more per clock
	- Even if processor performs more operations per second (2x), the DRAM (1.1x) will be a bottleneck
	- increased caching
	- fastest growing applications in parallel computing utilize not their raw computational speed but rather how fast they can pump data to memory and disk
- Two main Archs - Shared Memory (OpenMP) vs Message passing (MPI) ![[Pasted image 20230901180638.png]]
- "Embarrassingly" Parallel - When you can simply parallelize stuff without having to worry about what comes after what
- Automatic vs Manual Parallelization
	- Mp is very hard, slow and bug prone
	- Ap compilers can analyze the source code to identify parallelism
		- loops are a common target
		- but we might get wrong results, performance may degrade and produce bugs (as much as mp).
		- sometimes because of the optimization framework (**has knowledge of the underlying architecture**), which decides to parallelize a section of code (**which is parallelizable**) depending on the overhead produced, might get its calculations wrong and therefore not parallelize a section which could have been parallelized 
##### Shared Memory
- All processors see a global address space
- Two kinds - **U**niform **M**emory **A**ccess , **N**on **UMA**
- UMA
	- Typically Symmetric Multiprocessors (**SMP**)
	- All processors see the full memory
	- Equal access and access times to memory between processors
- NUMA
	- There still is a global address space, but there may be memory local to one processor, i.e. one processor will have faster access to that memory.
	- Can also be that one common global space, and a small range in that is local to you
- Pros - easier to program, typically fast mem access
- Cons - Hard to scale, adding CPUs increases traffic
#### Distributed
- Communication network, typically between processors
- Processor local memory
- Access to another processor's data through well defined communication protocol
	- 'implicit synchronization semantics'
- Inter process synchronization by programmer
#### Parallel Programming Models
- Shared Memory
	- tasks share common address space that they access async
	- locks / semaphores used to control access to the shared mem
	- data may be cached on the processor that works on it
	- compiler translates user variables into global memory addresses
- Message Passing
	- a set of tasks that use their own local memory during computation
	- Data transfer usually requires co-operation, i.e. a send has to be followed by a receive
- Threads
	- Multiple threads, each has local data, but also share common data
	- May communicate through global mem
	- commonly associated with shared mem architectures
- Data Parallel
	- focus on parallel operations on a set(array) of data items
	- tasks perform the same operation on different parts of the data structure
#### Metrics
- Speedup S$_p$ = Exec time using 1 processor system (T$_1$) / Exec time using p processor system (T$_p$)   
- Efficiency = S$_p$ / p
- Cost C$_p$  = p * T$_p$  
- Amdahl's Law
![[Pasted image 20230901190736.png]]
---
#### Components
- Main components -
	- Processors
	- Memory
		- shared
		- distributed
	- Communication
		- hierarchical, crossbar, bus, memory
		- synchronization
	- Control
		- central
		- distributed
- Flynn's Taxonomy -![[Pasted image 20230904174333.png]]
- idk ![[Pasted image 20230904174747.png]]
- (Problem?) With SIMD, lets say you have 
```
if(b1<b2)
	l=1;
else
	l=2;
```
And there are some n number of processors. For x of them, b1 < b2 and for y of them the other way around. Lets say it takes 10 clock cycles to assign a value to l. Since it's single instruction, first all the x processors will execute the same instruction while the rest stay idle, then the remaining y will execute the different instruction while the x stay idle. So the total clock cycles for this will be 20.
- Interconnect
	- Direct, Indirect (has some switch in between)
	- Topologies
		- Bus : cost scales well, performance does not
		- Crossbar : MxN grid connects M inputs to N outputs, performance scales well, cost does not
		- Complete graph
		- multi stage networks
- cache coherence must be done in hardware
- Modern pcs are multi proc. Multi cpu and Multi core. generally uses bus.
- Torus - last proc is connected to the first proc, in a grid.
-  diameter of a hypercube - longest diagonal distance
---
- Fat tree - as you go up the tree, the bandwidth of the wires go up
- butterfly - kind of like bits. simple logic, less contention than hypercube
![[Pasted image 20230925184435.png]]
- One way to measure performance - flops - floating point operations per second
---
#### Open MP
```
#pragma omp parallel
//the following region should be run in parallel
{
	//thread i of n
	switch(omp_get_thread_num()){
		case 0 :
		case 1 :
		...
	}
}
//normal code
```
- fork-join model
- not necessary that all threads run at the same speed
- Encountering thread(pragma) creates a team : itself, n-1 additional threads
- Each thread executes the {} separately.
- There is an implicit barrier at the end of the block, beyond which only the master continues
- May be nested
#### Memory Model
- Notion of temp view
	- allows local caching
	- need to relax consistency model
- supports thread private memory
- variables declared before parallel construct
		- shared by default
		- n-1 copies created
		- may not be initialized by the system
- Variable Sharing
	- Shared
		- Whatever is allocated on the heap
		- static data members
		- const variable
	- Private
		- Variables declared in a scope inside the (parallel) construct
		- Loop variable in for construct 
	- Others are shared unless declared private. Can change the default behavior 
	- Arguments passed by reference inherit from original
--- 
#### Relaxed consistency
- (Byte is atomic ??. Atom size differs)
-  flush makes everything equal. Can flush multiple variables at the same time.
-  If the intersection of the flush-sets of two flushes performed by two different threads is non empty, then the two flushes must be completed as if in some sequential order, as seen by all threads. If not, order doesn't matter
- If the intersection of the flush-sets of two flushes performed by one thread is nonempty, then the two flushes  must appear to be completed in that thread's program order
- Somethings are automatically flushed, like all variables at the end of a parallel construct
- Sometimes compiler reordering of shared variables can cause deadlocks
---
