# Lec 21

#### Architecture
- tesla consists of
	- Processors grouped into groups
	- groups share scratch mem
	- all processors share cache
	- and lastly main mem for the gpu
- When you read things from disk, it goes to system mem first then gpu mem
- Fermi has bigger scratch, more procs, L2 is bigger and gets used. Rest is same
- CPU big cache small computation vs GPU small cache big computation
- Each multiprocessor consists of several SIMD processors
#### CUDAAAAAAAA
- Compute unified device architecture
- user runs batches of threads on the gpu
- cuda is c like
- serial or modestly parallel parts in host C code, highly parallel parts in device kernel C code
- A compute device
	- is a coprocessors
	- has its own DRAM
	- typically a gpu
- Data parallel portions of a program are expressed as **device kernels** which run on many threads
- GPU threads are very lightweight, require little creation overhead. But you need 1000x threads for full efficiency, as opposed to a CPU which only needs a few.
- Extended C
	- piece of mem (variables, func) can be
		- ```__x__```
		- global - can write to, basically main mem
		- device
		- shared - cache / scratch
		- local
		- const
	- syncthreads is basically a barrier (scope is block)
	- cudamalloc to get mem from gpu
#### Software stack
- nvcc is basically the cuda compiler
	- separates the host code from the device code
	- creates ptx code which gets converted later to the specific cuda device.
	- ptx is like java bytecode
- SASS converts ptx to machine code
- Driver at the bottoms then sends the SASS generated code to the gpu
- App talks to CUDA libraries, which talks to CUDA Runtime, which talks to CUDA Driver, which talks to the device
#### Working?
- a CUDA kernel can beexecuted many times
	- by a block of threads
	- once per thread, each running same kernel (SPMD)
- a CUDA kernel can be executed many times
	- by multiple blocks of threads (grid)
	- blocks are concurrent 
	- threads in diff blocks only share global mem
		- shared mem's life and scope is the block
	- threads in a block have shared mem, global mem, barrier etc

# Lec 22
- 32 threads -> Warp
- not sure about this ->
	- within a warp, everything is synced, 
	- within a block it is not synced, thats why we use syncthreads
#### Main CUDA construct
- run k instances of function f
	- this f is called a kernel
	- declared with ```__global__```
	```
	 __global__ void f (float* A)
	{
		int id = threadIdx.x;
		...
	}
	int main()
	{
		//calling kernel
		f<<<1,N>>>(A); // <<<A,B>>> A blocks, each of size B
	}
	```
	- all threads within a block are scheduled on the same SIMD SM
	- 'thread local mem' live in global mem
- all threads of a warp start together, but may branch out. These paths are serialized until they converge
- if 3d, then thread ID = B.x + B.y * (x dimension) + B.z \* (x dimension) \* (y dimension)
- Memory model overview
	- Global
		- Main host-device data comm path
		- visible to all threads
		- long latency
	- Shared
		- Fast
		- use as scratch
		- shared across a block
	-  Constant and texture
		- cached and read only
		- (technically can be written but yeah)

# Lec 23
- App must explicitly allocate/de-allocate device memory
- cudaMalloc()
	- allocates in global memory
	- params
		- address of a pointer
		- size 
- cudaFree()
	- frees object from global mem
- All of these are called on the host
- cudaMemcpy(dest,host,size,cudaMemcpyHostToDevice /cudaMemcpyDeviceToHost)
	- device to device also possible
	- host to host is unnecessary 
- cudaMallocPitch for 2d, cudaMalloc3D for 3d
- cudaMemcpyToSymbol(dest var, source var, size)
- There is also page locked host mem
	- getting data is faster, as you can directly map page locked host mem to device address space
	- cudaHostAlloc() gets host pointer
	- cudaHostGetDevicePointer()
- Can directly access host mem from kernel, but will have to worry about RAW WAR WAW
- global is called from host, run on device
- device is called from device, run on device
- 1.x doesnt have recursion, or a 'real' cache
- device functions are inlined (can make something non inlined, or force inlined, upto compiler in the end)
- async exists, can return before the task is complete
	- can do device to device mem copies
	- host to device as well
	- can disable async for debugging.
		- CUDA_LAUNCH_BUGGING = 1
# Lec 24
#### Matrix mult
- one thread for one product
```c
int val = 0;
for(int k = 0; k< width;++k)
{
	M = Md[threadidx.y*width + k];
	N = Nd[k*width + threadidx.x];
	val += M*N;
}
Pd[threadidx.y*width+threadidx.x] = val;
```
- If matrix is too big, break it down
- whatever this means
	- 2D thread block computes a (TILE_WIDTH)$^2$ sub matrix of pd - (WIDTH/TILE_WIDTH)$^2$ blocks
#### Synchronization 
- use underscore x 2 before the function name
- syncthreads()
	- block barrier
	- ensure all global/shared memory accesses by all threads are visible in a block
- syncthreads(int predicate)
	- returns the count where predicate != 0
- \_and( int predicate )
	- iff predicate!=0 for all
- \_or (int predicate )
	- iff!=0 for any
- all, and and any are also there for intra warp sync
	- also have ballot, which returns int with nth bit set if predicate!=0
#### Atomic ops
- only for the context of the gpu (not host)
- CAS
	```c //add
	int old = *address, assumed;
	do {
		assumed = old;
		old = someshitidk;
	}while(assumed!=old);
	return old;
	```

# Lec 25
#### Bitonic sort
- goes up then down or down then up
- swap (if needed) at 2$^x$ bit distance ( watch vid for explanation )
- need to sync after every step
#### Memory fences
- \_\_threadfence_block
	- wait until all global/shared memory accesses by the caller are visible to block
- \_\_threadfence
	- wait until all memory accesses by the caller are visible to
		- all threads in the block for shared mem access
		- all threads in the device for global mem access
- \_\_threadfence\_system (only in 2.x)
	- wait until all memory accesses by the caller are visible to
		- all threads in the thread block for shared mem acc
		- all threads in the device for global mem access
		- Host threads for page-locked host mem access
- 2.x has weaker consistency compared to 1.x
- Reduce
```c
float partialsum = calcpsum(array,N);
if(threadIdx.x==0)
{
	result[blockIdx.x] = partialsum;
	__threadfence();
	int val = atomicInc(&count,gridDim.x); //returns old val, gridDim.x = number                                             //of blocks
	isLastBlockDone = (val==gridDim.x-1);
}
```
- everyone checks if they are done
```c
__syncthreads(); //makes everyone wait
if(isLastBlockDone)
{
	float tsum = calctsum(result);
	if(threadIdx.x==0)
	{
		result[0] = tsum;
		count = 0; //reset for next kernel call (why?)
	}
}
```
#### Streams
- Basically a higher level thread
- functions in a stream execute in order
- different streams may interleave
- when you dont explicitly mention a stream, its 0- stream
- no stream = 0-stream
	- serial stream
	- begins only after all preceding operations are done
	- subsequent operations may not begin until it is done
- Stream sync
	- cudaThreadSynchronize()
		- waits until all preceding commands in all streams have completed
	- cudaStreamSynchronize
		- waits until all preceding commands in a given stream have completed
	- cudaStreamWaitEvent()
		- makes all the commands added to the given stream after this call to wait until the given event
	- cudaStreamQuery()
		- check if all preceding commands in stream have completed
# Lec 26
#### Performance tips
- Maximize parallelism
	- many threads
	- minimize inter-block communication and sync
	- use streams to maximize concurrency if required
- overlap mem access with computations
	- many arithmetic ops per memory op (40-50x)
	- space out SFU functions (SFU?)
- minimize low bandwidth mem transfer
- minimize low throughput instructions
#### Instruction Issue Throughput
- 1.x
	- Fast (4 cycles)
		- float add,mult,madd
		- int add,bitwise,compare,min/max
		- type conversion
	- Slower(16 cycles)
		- reciprocal, reciprocal square root
		- 32 bit int mult
	- Really slow
		- int division, modulo
- Scheduling
	- 1.x
		- Hardware
			- 8 CUDA cores
			- 1 dp fp unit
			- 2 SFU
			- 1 warp scheduler
		- Scheduling
			- 4 cc for int or float arith
			- 32 for dp fp arith
			- 16 cc for sp fp 'transcendental' (??) instruction
	- 2.0
		- 32 CUDA cores
		- 4 special function units for sp fp
		- 2 warp scheduler, each one issues one instruction
	- 2.1
		- 48
		- 8
		- 2 warp scheduler, each one issues two instructions
- Try to minimize the number of divergent warps
- branch predictions
- whenever you can, use shared mem, try not to use syncthreads.
#### Global mem organization
- read data in bulks
- 256 byte boundary
- access can be of 32, 64, 12 bit words
- single load instruction if size is 4,8,16
- must be aligned to size bytes
```c
struct __align__(16) {
	float a,b,c,d,e;
}x,y;
//results in two 128-loads, not 5 32-bit loads
//idk
```
#### Global mem coalescing
- mem transactions must be 32, 64 or 128 bytes
	- must be aligned to as many bytes
- upto 16 accesses by a half-warp coalesced into a single transaction if addresses are within a segment of 
	- 32 bytes if all threads accesses 8 bit words
	- 64 bytes if all threads accesses 16 bit words
	- 128 bytes if all threads accesses 32/64 bit words
	- two transactions for 128 bit words
- if 2d, array width should be rounded upto 16x, as mem coalescing iff width of thread block is 16x and WIDTH is 16x
	- use cudaMallocPitch for this
- If you are getting every nth value of an array, no bank conflict for odd values of step
# Lec 27
#### Block sizing
- have more blocks in the grid then \#SMs
- have enough threads in a block to fill one SM
- block size should be multiple of warp size
- generally 192 or 256, never below 64. also dont make them too big
#### Memory transfers
- try to do it in batches, i.e. all stuff to device at once, all stuff to host at once
- if possible, use page locked mem
#### Instruction Issue and Latency
- one warp instruction issue takes 4 cc
- two warps take 2 cc
	- two instructions for two warps in 2 cc
- if you need something from previous instruction
	- must wait for that to get over
	- 400-800 cc latency on dram access
	- ~22 cc for many instructions
		- so thats why you need alot of other warps to fill up this latency
#### Algorithms :(
- First 1 problem
	- input : n bit vector
	- output: minimum index of 1 bit
	- Algo
		- divide into root(n) blocks of root(n) bits each
		- for block i, B\[i] = 1, if there is a 1 in the block
			- arbitrary CRCW, O(n) work, O(1) time
		- find the first j such that B\[j] = 1
			- O(n) work, O(1) time
		- find first 1 in the jth block 
			- O(n) work, O(1) time
	- what?
- All nearest smaller value
	- for each element A\[i], find largest j where A\[j] < A\[i] (and j<i?)
	- O(n$^2$) work, O(1) time 
- Prefix minima
	- leftmost element which doesnt have ansv is the prefix minima
	- O(n$^2$) work, O(1) time
# Lec 28
#### Prefix minima again
1. partition into root(n) blocks of root(n) each \[O(1) time, O(n) work]
2. find prefix min for each block recusively
3. consider B = m1...mroot(n)
	- where mi = minima of block i \[O(1) time, O(n) work]
	- find prefix min of B in constant time
4. minimum of A\[1] .. A\[j] is the minimum of 
	- m1 .. mi-1 and its block prefix minima(from step 2) \[O(1) time, O(n) work]
- Overall - O(nloglogn) work, O(loglogn) time
#### Find Root of forest
- O(log(height)) time
- basically pointer jumping 
#### List ranking
- pointer jumping again
```c
//set all values = 1, end val = 0
parallel for all i
	 if next[i] != null
		 d[i]  = d[i] + d[next[i]]
		 next[i] = next[next[i]]
until next[i]==null for all i
```
- O(nlogn) work, O(logn) time
#### Symmetry breaking
- cycle coloring
- colors = numbers
	- initially, c\[i] = i
	- iterate, reducing colors each time
	```python
	parallel for all i do
		k = least significant bit in which c[i] and c[next[i]] differ
		c[i] = 2k+ bit k of c[i] 
	```
	- something something proof by contradiction
	- O(log\*n) time, O(nlog\*n) work (what is \*?)
	- but it stops at 6 colors
	- one more step to remove 3 extra colors
#### Bitonic merge sort
- watch some other vid for explanation
- sort each pair, alternatively in increasing and decreasing
- every sequence of length 4 is bitonic
- sort recursively
	- again alternate inc and dec
	- length 8 now
	- so on
- log$^2$n time and nlog$^2$n (allegedly)
# Lec 29
#### Batcher's odd even merge
- kinda like bubble sort
	- compare 01,23.. so on
	- next stage compare 12,34..
	- repeat n times
- n time and n$^2$ work
- can be improved to logn time (unsure)
#### Fast sort
- basically like finding rank
- idk
#### Quicksort
- divide into n/p blocks
- idk
#### Bucket and Sample sorting
- Parallel bucket sort
	- each processor is responsible for a certain range
	- each proc reads through its local list and sends the element to the correct proc
	- using single all to all comm
	- each proc sorts the elements it receives
- using some scheme, suitable splitter selection, we can guarantee that the number of elements ending up in each bucket is less than 2n/m. (m = number of blocks)
#### Radix sort
- idk
# Lec 30
- some yapping about parallel bucket sort
- O(n/plogn/p + plogp)
- bitonic if n>p
	- O(n/plogn/p + n/plog$^2$p) time \[sorting, comp splits]
- odd even if n>p
	- O(n/plogn/p) local sort + O(n) p comp split
- optimal merge sort
	- merge at each level of recursion using the optimal parallel merge
		- O(loglogn) time
		- O(n) work
	- logn merge tree levels
		- O(logn loglogn) time
		- O(nlogn) work
- optimal multiway merge N=P$^2$
	- divide both lists into P sublists each
	- ith sublist contains i, P+i, 2P+i
	- P$_i$ merges the ith sublist from L1 and L2
		- results go back to the positions originally occupied by the two sublists
	- Each element is atmost P+1 off from its final position
	- Divide each list in blocks of P
	- P$_i$ merges pairs of blocks
		- 2i and 2i+1, results are put in place
	- P$_i$ merges pairs of blocks
		- 2i+1 and 2i+2, results are put in place 
# Lec 31
#### Some proof for multiway merge (A)
- take L1i, L2i as ith sublists of L1 and L2
- A, B are the mth,m+1th in L1i, X is the nth in L2i
	- A is i+mP, B is i+(m+1)P
- We know val of X is in between val of A and val of B
- so atleast (i + mP + 1)+(i + nP) elements to the left of X
	- rearrange = i + P(m+n+1) + (i-P+1)
- so atmost (i + (m+1)P)+(i + nP) elements to the left of X
	- rearrange = i+P(m+n+1)
- (somehow) position of X after megre is (m+n+1)P + i 
#### proof B 
- idk
#### proof C
- idk
# Lec 32
- if N > P$^2$, it is O(N/P log N/P)
- using c cover, can find rank of A in B and B in A in O(1) time and O(n) work
- optimal O(logn) time merge sort
	- runs in stages of a binary tree
	- normally youd wait for you children to be sorted before you start merging them
	- here you dont have to wait
	- you become active when stage counter is your stage, and you remain active until the counter is 3\*your stage
	- so at each stage you dont merge all the elements from left and right, only some of them (sample)
		- first every 4th, then every 2nd, then all
	- node becomes full (sorted) at stage 3\*height(node)
	- something about covers and merge time at each stage being O(1) 
# Lec 33
- Optimal merge sort
	- will be done in 3logn, so O(logn)
	- work is O(nlogn)
- Minimum finding
	- lower bound - log log n
		- p <= n
	- some graph theory 
		- a graph with n nodes and m edges has an independent set of at least n$^2$ / (2m+n) 
	- min finding theory
		- at the end of the (i+1)st step, the input may lie in the set C$_{i+1}$ as large as c$_{i+1}$ such that
			- no two elements of C$_{i+1}$ have been compared before
			- c$_{i+1}$ >= c$^2$$_{i}$ / (2p+c$_{i}$)
	- basically a graph of comparisions
	- proof?
		- c$_{i+1}$ >= c$^2$$_{i}$ / (2p+c$_{i}$)
			- in the beginning, c$_{i}$ is n, and p is n
		- c$_{i+1}$ >= c$^2$$_{i}$ / 3n
			- solving this recurrence relationship
		- c$_{i+1}$ >= n/(3$^{2^i-1}$) c$^2$$_{i}$/3n
		- which basically means c$_i$ <= 1 after omega(loglogn) steps
- Similarly, other lower bounds (all theta)
	- merging : n/p + loglogn, if p = O(nlog$^r$n)
	- sorting : logn, if p<=n
	- count 0s in a monotonic binary sequence : log(n/p)
	- parity of n variables : logn / loglogn
#### Synchronization
- Mutex
	- blocking
- Lock free
	- at least one op is going to complete in a finite amount of time
- wait free
	- every op will finish in a finite amount of steps
- lock free, wait free require hardware supported atomic operations 
	- like CAS
```c
old = *address;
new = (old==compare?val:old);
*address = new;
return old;
```
#### Dynamic load balancing
- static task list
- while (next = worklist.front() != END)
	- perform work
- Find a busy proc, share its load
- Repeat
	- for a proc p
	- lock it, share its remaining load
	- unlock
- 