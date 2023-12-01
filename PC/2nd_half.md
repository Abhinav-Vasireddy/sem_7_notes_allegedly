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