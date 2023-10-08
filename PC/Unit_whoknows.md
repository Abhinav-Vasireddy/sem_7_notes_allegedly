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
---
 - bunch of examples, watch vid
 - cant keep ordered right (inside/next) another ordered
 #### Matrix Multiplication
 - normal code
 ```
 for(int i = 0;i<n;++i)
 {
	 for(int j = 0;j<n;++j)
	 {
		 c[i][j] = 0.0;
		 for(int k = 0;k<n;++k)
		 {
			 c[i][j] += a[i][k] * b[k][j];
		 }
	 }
 }
 ```
 - with omp
```
#pragma omp parallel for
 for(int i = 0;i<n;++i)
 {
	 #pragma omp parallel for
	 for(int j = 0;j<n;++j)
	 {
		 int sum = 0;
		 #pragma omp parallel for firstprivate(sum) reduction(+:sum)
		 for(int k = 0;k<n;++k)
		 {
			 sum = a[i][k] * b[k][j];
			 c[i][j] = sum;  
		 }
	 }
 } 
```
---
#### Modes of Parallel Computation
- abstract(verb) stuff
- should track performance
- general classes
	- shared mem vs distributed mem
	- sync(pram) vs async
- PRAM model
	- synch, shared mem
	- arbitrary number of processors, each
		- has local mem
		- arbitrary number of cells(mem loc)
		- know its ID
		- can access a shared memory location in constant time
	-  At each time step, each P$_i$ can 
		- read some mem cell
		- perform local computation step
		- write a mem cell
		- co - access may be restricted
	- Thus a pair of processors can communicate in two steps - constant time
	- In/Outputs are placed at designated addresses
	- each instruction take O(1) time. basic instructions
	- Processors are synch. Although Async PRAM models exist
	- Cost analysis
		- time taken by longest running processor
		- total number of memory cells accessed
		- maximum number of active processors
- Shared mem Access models
	- EREW (Exclusive read Exclusive write)
	- CREW (Concurrent read Exclusive write)
	- ERCW - least used
	- CRCW - 'most powerful' 
	- For concurrent writes -
		- Priority CW (normally lower index)
		- Common CW - succeeds only if all writes have the same value
		- Arbitrary/Random CW - guess
	- In order of power - 
		- EREW <= CREW <= Common <= Arbitrary <= Priority
		- higher can simulate lower
---
 - pardo = parallel do
 - For addition 
	 - proc = n/2
	 - speed up = n/log(n)
	 - efficiency = speedup / \#proc = 1/log(n)
	 - work = \#proc * steps =n * log(n)
 - Membership problem
	 - n integers in n mem cells
	 - does 'x' exist in input
	 - steps
		 1) if p$_0$ broadcast x
		 2) proc p$_i$ searches in i$^{th}$ (n/p) block and sets a flag
		 3) p$_0$ checks if any flag is 1 and prints answer
	 - For EREW
		 1) 1 proc will read x, then write to another location. Then two procs will read the two spots with x, then write to two more locations and so on. So total time for all procs to know x will be log(p)
		 2) n/p
		 3) log(p)
	 - For CREW
		 1) 1
		 2) n/p
		 3) log(p)
	- For CRCW
		 1) 1
		 2) n/p
		 3) 1
- Work Time Scheduling Principle
	- t(n) steps
	- W(n) work
	- Time <= sum(ceil(W$_i$(n)/p)) <= floor(W(n)/p) + t(n)
- watch vid to simulate lesser proc/mem
- Performance evaluation 
	- Generally use W(n) :  (work is O(n) and time is O(n) ) is better than (work is O(nlogn) and time is O(nlogn) )
	- If W(n) is similar, use t(n)
---
- Brents theorem
	- time taken by p processors = t$_p$(n) = O(W(n)/p + t(n))
	- cost = p * t$_p$(n) = O(W(n) + p * t(n))
	- Work = cost if : W(n) + p * t(n) = O(W(n))
	- p = O(W(n)/t(n))
- Optimal Summation
	- Using n proc
		- Work = O(n)
		- Cost = O(n logn)
	- Using n/log(n) proc
		- local sum takes logn time
		- parallel sum takes log(n/logn) = O(logn)
		- Total steps = (2?)logn
		- Cost = proc * steps/time = n
	- Often useful to have large sequential sections
-  Notions of optimality : sim to brent
- Why PRAM
	- easy to design and specify algos
	- easy to analyze
#### Bulk Synchronous Parallel Model
- set of pairs (proc,mem)
- point to point interconnect
- barrier sync
- Super step
	- local compu
	- comms
	- barrier sync
![[Pasted image 20231004171154.png]]
- Steps are machine wide, similar to PRAM
- simple to program and analyze algos
- "bridges" well to physical arch
- locality of proc or computation ignored : leaves optimizations out of the equation
- Interconnect
	- atmost h messages in a super step 
	- Cost = gh = th + s(overhead)
	- ignores message count, distribution
- Super step cost accounting
	- barrier  = I
	- one step = w(local comp) + gh + I
	- Total S steps = sigma(w) + g * sigma(h) + SI
#### LogP Model
- pair wise sync
- per message overhead : accounts for latency
- g is the minimal permissible gap between message sends from a single process
#### Complexity class NC
- using poly procs, you can solve the problem is polylog time
- time = log$^{k1}$ n , procs = n$^{k2}$
---
- For bin search
	- naive approach of p procs
		- split n into n/p, each proc does bin searc
		- time = log(n/p), work = p * log(n/p) 
		- problem is a proc will know quickly that it doesnt have it, and one proc will keep working while others are idle
	- second approach
		- each checks if val is in its range with two comparisons, either checks ends or check the result of proc to the left. Then problem size will be n -> n/p -> n/p$^2$ and so on
		- time = log(n)/log(p)
- NP -> P, P -> NC (idk)
#### Cache coherence
- Snoopy
	- all transactions to shared mem visible to all processors
	- caches contain info on which addresses they store
	- cache controller snoops all transactions on the bus
	- ensures coherence if a relevant transaction
- Memory consistency issues
	- basically compiler might break things
	- sometimes even network might break things
- "Strict"
	- global clock
	- no two things can happen at the same time
	- if two make requests at a similar time, the one who did it earlier should start and finish their work before second one starts. Everything is atomic
	- someone wants to read, everyone else stops until this fellow reads
- Practical Mem model
	- Sequential consistency
		- A multiprocessor is sequentially consistent if the result of any execution is the same as if the operations of all the processors were executed in some sequential order, and the operations of each individual processor appear in this sequence in the order specified by its program
		- hard to implement, performance not great
		```
		r = 0;d = 0
		t1                                           t2
		d = 1;                                      while(!r);
		r = 1;                                      p = d;
		```
		- If t2 sees r is 1, then d must be 1. Don't care about anything else
---
- Seq consistency is hard to implement
	- poor performance
	- no re ordering
	- new mem op cant be issued until previous one is complete
	- more popular to simply switch to weaker models
	- if you are writing to an L1 cache (write through) and that write is in queue, and you get a read for something else, you cannot read before that write is over, even though it doesnt affect anything, as the other procs will see a different order. this is a waste of cache
- Causal Consistency
	- write is causally ordered after all earlier reads/writes 
	- read is causally ordered after write to the same variable on a processor
- Processor Consistency
	- All processes see memory writes for on process in the order they were issued from the process
	- writes from different processes may be seen in a different order on different processors
	- only the write order needs to be consistent
	- all instances of write(x) are seen everywhere in the same order  
- Weak Consistency
	- sync accesses are sequentially consistent 
	- this is open mp, sync access = flush. (technically open mp is weaker than weak)
![[Pasted image 20231004192051.png]]
- True sharing
	- frequent writes to a var can create a bottleneck
	- sol : make copies of value, one per processor
- False
	- two distinct vars on the same cache block
	- sol : allocate contiguous blocks
#### Other performance considerations
- keep critical region short
- limit fork join
- invert loops(?)
- use enough threads  
---
-  Converting serial to parallel
	- Analyze which parts invite parallelization
		- data parallel
		- natural decomp : f(x) = h(g1(x),g2(x))
		- recursion
		- extract direct dependencies
	- Identify serial program's hotspots
	- If there is a slow stage in the pipeline, try to reconstruct it
- Investigate the direct parallel algorithm
- When you decompose a task into smaller tasks, try to minimize dependencies (data/task)
- Need to find good size for a task
	- many small tasks cause alot of dependencies
	- few large tasks cause unnecessary idle time  
- Communication (even indirect) always requires some level of synchronization
- Load balancing
- Main issues
	- Granularity
	- communication
	- synchronization
	- load balancing
#### Communication Issues
- Inter task comm_n has overhead
	- sync and network congestion
- Latency / Bandwidth tradeoff
	- try hiding latency 
		- by having many tasks
		- by prefetching
			- ex: for matrix mul, have another variable to read ahead the values, and only start computing the current values when you have the next set ready, and so on
- sync vs async comm_n
	- sync/blocking comm_ns require handshaking between tasks
	- async/nonblocking allow tasks to transfer data independently from one another
		- interleave computation with communication, hide latency
- Scope of comm_n syncs
	- point to point : sender receiver, producer customer
	- collective : group/multicast, reduce, scatter/gather
#### Synchronization
- Barrier
	- expensive as all threads must participate 
	- variant called memory barrier also exists : all operations to section x of mem have to be done before section y. Bit more lightweight
- Lock / semaphore
	- can be blocking or non blocking
- **"Synchronization is the largest source of errors in parallel computing"**
	- first get syncs right before performance
- Stick with locks
	- reasonably efficient
	- fits all kinds of situations
#### Load Balancing
- Keep tasks busy
- keep every processor occupied all the time
- Equally partition work, but not necessary all the time for the tasks to be similar in size 
- Look out for hetero machines
- dynamic work assignment 
- use a scheduler, have alot of tasks ready
#### Granularity
- Typical cycle : compute then communicate/synchronize
- Fine grain
	- small amounts of computation between communication events
	- easier to balance load
- Coarse
	- large amounts
	- harder
- Best one? Balance it 
	- start with big tasks, keep reducing until you have enough
#### Task decomp
Categories :
- Domain : partition data, task processes a section of data
- Functional : task performs a share of the work. Can maintain work pool(?)
Techniques :
- Pipeline
- Recursive : divide and conquer. Not necessary to be implemented "recursively"
- Exploratory : search problems, optimization problems, game playing
- Speculative : new tasks are generated.  speculatively execute dependent tasks. if the results are needed at a later stage keep them, otherwise discard 
- Hybrid 
- Static / dynamic tasks(new tasks are generated)
#### Task Mapping : Reduce Idle Cycles
- If you have p processors, generate 20 to 30 \* p tasks
- Heuristics :
	- Map independent tasks to different processes
	- Assign tasks on critical path as soon as possible
	- Map tasks with dense interactions together
- Easier if number and sizes of tasks can be predicted
	- as well as data size per task
- Inter task interaction pattern
	- static vs dynamic
	- regular vs irregular
	- read only vs read write
	- one way vs two way
- Static
	- Knap sack
	- Data partitioning
		- array distribution
			- block distribution
			- cyclic
			- block cyclic
			- randomized block distribution
		- graph partitioning
		- allocate sub graphs to each processor
	- Task partitioning
		- task interaction graph
		- graph cut
- Dynamic
	- task allocation vs work queues
		- central or distributed
	- Work stealing (im done with mine, if you have left i will take a part of it)
		- how are sensing and receiving processes paired together
		- who initiates work transfer
		- how much work is transferred
		- when a transfer is triggered
#### GP TIPS
- correct before fast
- know your target architecture
- know your application
	- look for data and task parallelism
	- try to keep threads independent
	- low sync and comm_n
	- start fine grained, then combine
	- make sure "hot spots" are parallelized
	- use thread safe libraries
	- never assume the state of a variable or another thread. always enforce it
---
#### Shared Memory and Memory passing
![[Pasted image 20231008163804.png]]
- t1 says values a,c,b should be written to x before 0 is written to y, but that is inconsistent with t3 
#### MPI overview
- MPI by itself is a library specification
- you need a library that implements it, like openMPI
- performs message passing and more
	- high level constructs
		- broadcast, reduce, scatter, gather message
	- packaging, buffering etc automatically handled
	- Also
		- staring and ending tasks remotely 
		- task identification
- Portable
#### Running MPI
- compile : (mpic++/mpicc) -O -o exec code.cpp
- run : -mpirun -host host1,host2 exec args
#### Remote Execution
- Allow remote shell command execution
	- using ssh
		- without password
- set up public private key pair blah blah blah
#### Process organization
- Context
	- comm_n universe
	- message across context have no interference
- Groups
	- collection of processes
	- creates hierarchy
- Communicator (kinda abstract of channel)
	- Groups of processes that share a context
	- Notion of inter-communicator
	- Default : MPI_COMM_WORLD
- Rank
	- In the group associated with a communicator
- Starting and ending 
	- MPI_INIT(&argc, &argv) and MPI_Finalize() (required)
- Send / recieve
	- int MPI_Send(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
	- receive : source -> dest
	- message contents : block of mem
	- count : number of items in message
	- message type : MPI_type of each item
	- destination : rank of recipient
	- tag : integer "message type"
	- communicator
---
- Example
```c++
#include <stdio.h>
#include <string.h>
#include "mpi.h"

#define MAXSIZE 100

int main(int argc,char* argv[])
{
	MPI_INIT(&argc,&argv);                             //start

	MPI_Comm_size(MPI_COMM_WORLD, &numProc);           //group size
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);            //get my rank
	doProcessing(myRank,numProc);
	
	MPI_Finalize();                                    //stop
}
```
```c++
if (myRank != 0) {
	//create message
	sprintf(mesg,"Hello form %d", myRank);
	dest = 0;
	MPI_send(mesg,strlen(mesg)+1,MPI_CHAR,dest,tag,MPI_COMM_WORLD);
}
else {
	for(source = 0;source<numProc;++source){
		if(MPI_Recv(mesg,MAXSIZE<MPI_CHAR,source,tag,MPI_COMM_WORLD,&status) == MPI_SUCCESS)
		printf("Received form %d : %s\n",source,mesg);
		else
		printf("Received form %d and failed\n",source);
	}
}
```
- use buffer instead of sync
#### MPI Send and Receive
- It is blocking
	- R blocks until message is received
	- s may be sync or buffered
- Standard (default) | MPI_Send
	- "chef's choice"
	- changes from call to call, chooses the best option
	- implementation dependent
	- buffering improves performance, but requires sufficient resources
- Buffered | MPI_Bsend
	- you provide a buffer (+size)
	- If no receive posted, system must buffer
- Synchronous | MPI_Ssend
	- will complete only if receive on the other end has been accepted
	- send can be started based on whether a matching receive was posted or not
- Ready | MPI_Rsend
	- Send may start only if receive has been started (doesnt matter if it completes, only started) on the other end
	- buffer may be reused
	- like standard but better performance
- Only one MPI_Recv mode
#### Message semantics
- need to be careful about order (no global order)
- for a matching send/recv pair, at least one of these two operations will complete
- fairness not guaranteed
	- a send or recv may starve because all matches are satisfied by others (kinda like race condition)
- Resource limitations can lead to deadlocks
- Synchronous sends rely on the least resources
	- may be used as a debugging tool 
-  Async (I for async?)
	- MPI_Isend() / MPI_Irecv()
		- non blocking
		- blocking and non blocking send/recv can match
		- still lower send overhead if recv has been posted 
	- functions have extra parameter : MPI_Request \*request. 
		- kinda like id per send
		- use this to check if this request has succeeded or not  (for both send and recv)
	-  MPI_Wait(&request,&status)
		- status returns status similar to recv
		- blocks for send until safe to reuse buffer
			- means message was copied out or recv started
		- blocks for receive until message is in the buffer
			- call to send may not have returned yet
		- request is de allocated
		- more efficient if you have a lot of processes
	- MPI_Test (&request, &flag, &status)
		- does not block
		- flag indicated whether the op is complete
		- poll
	- MPI_Request_get_status(&request, &flag, &status)
		- doesnt de allocate
	- MPI_Request_free(&request)
		- frees request 
	- MPI_Waitany(count, requestarray, &whichReady, status)
		- for multiple
	- MPI_Waitsome
- MPI_(I?)Probe(source,tag,comm,&flag,&status)
	- checks info about incoming messages without actually receiving them
	- ex - useful to know message size
	- next (matching) recv will receive it  
- MPI_Cancel(&request)
	- request cancellation
	- non blocking