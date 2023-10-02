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
#### Thread control
- Environment variables
	- OMP_NUM_THREADS  = Number of threads
		- omp_set_num_threads
		- omp_get_max_threads
		- Initial value = implementation defined
	- OMP_DYNAMIC = how work among the threads is allocated
		- opm_set_dynamic
		- opm_get_dynamic
		- implementation defined
	- OMP_NESTED = can you nest
		- opm_set_nested
		- opm_get_nested
		- false
	- OMP_SCHEDULE = similar to dynamic
		- implementation defined  
- Example
	```
	#pragma omp parallel \
	if(boolean) \
	private(var1,...) \
	firstprivate(var1,...) \
	default(private | shared | none)
	shared(var1,...), \
	copyin(var1,...), \
	reduction(operator:list) \
	num_threads(n)
	```
	- If(boolean) if you want to make the execution of the parallel clause conditional
	- private() to define which variables to be private
	- firstprivate() to initialize the variables with the values of the main thread
	- default() to define default behavior of variables.
	- shared() to define which variables to be shared
	- copyin() is similar to firstprivate but for threadprivate variables(global?), and the values can come from any thread
	- reduction() to define what to do with the n results. find sum, max, etc.
	- num_threads to define number of threads.
- Thread_ID usually has to be private
- ```#pragma omp for```  for loops
```
/*
this is expected to be inside a parallel construct
so parallel -> for -> for loop
if the middle for is missing, then each thread from the first parallel will run the loop n times.
if the middle for is present, then the loop is shared by each thread
*/
#pragma omp for \
private(var1,...) \
firstprivate(var1,...) \
lastprivate(var1,...) \
reduction(operator:list) \
ordered, \
schedule(kind[,chunk_size]) \
nowait
Canonical For loop ( no break )
```
- 
	- compiler put a barrier at the end(??)
	- lastprivate() to decide values in the end for the original 
	- ordered says there is an ordered construct inside the body. Kind of like saying two different threads cannot be there at the same time. (Critical section)
	- Schedule divides iterations into contiguous sets, chunks
		- static : chunks are assigned in a round robin fashion. cs ~ load/num_threads
		- dynamic : chunks are assigned to threads as requested. default is 1
		- guided : dynamic, with cs proportional to number of unassigned iterations / num_threads. default is 1
		- runtime : taken from environment variable
		- 
	- kind is what type of distribution to do, and chunk size is in how many chunks(i.e for thread 3 do i = 10 to 13)
---
 - reduction ops = +, \*, & , |, ^, &&, ||
#### Single construct
- ```#pragma omp single```
- only one of the threads executes this.
- other threads wait, unless nowait is specified
- sometimes threads may not hit single
- also has copyprivate, which writes back to the master
#### Sections construct
```
#pragma omp sections
{
	#pragma omp section
	{
	
	}
	#pragma omp section
	{
	
	}
	...
}
```
- Basically like switch case for threads
#### Other directives
- ```#pragma omp master```
	- only master executes
	- no implied barrier
- ```#pragma omp critical(bankbalance)```
	- bankbalance is a section
	- single thread at a time 
	- applies to all threads
	- name is optional, if anon, then global critical region
- ```#pragma omp barrier```
	- standalone(?)
	- all threads must execute
- ```#pragma omp ordered```
	- the block is executed in sequential order
	- loop must declare the ordered clause
	- each thread must encounter only one ordered region
- ```#pragma omp flush(var1,var2)```
	- standalone
	- only directly affects encountering thread
	- ensure that any compiler reordering moves all flushes together(?)
- ```#pragma omp atomic```
	- guess
	- only for - x++, ++x, x--, --x
#### Helper functions
- General
	- void omp_set_dynamic(int);
		- int omp_get_dynamic();
	- void omp_set_nested(int);
		- int omp_get_nested();
	- int omp_get_num_procs(); 
	- int omp_get_num_threads();
	- int omp_get_thread_num();
	- int omp_get_ancestor_thread_num();
	- double omp_get_wtime();
- Mutex
	- void omp_init_lock(omp_lock_t \*);
		- void omp_destroy_lock(omp_lock_t \*);
	- void omp_set_lock(omp_lock_t \*);
	- void omp_unset_lock(omp_lock_t \*);
	- int omp_test_lock(omp_lock_t \*);
#### Nesting restrictions
- A critical region may not be nested ever inside a critical region with the same name
	- not sufficient to prevent deadlock
- cannot directly nest the following (work sharing = for) - 
	- inside work-sharing, critical, ordered, or master
		- work-sharing
		- barrier
	- inside work-sharing
		- master
	- inside a critical region
		- ordered region