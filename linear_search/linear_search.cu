#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda_profiler_api.h>
// Kernel function to add the elements of two arrays
__global__ void gpu_search(int *isFound, float *x, int number, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

}

__global__ void gpu_search_2(int *isFound, float *x, int number, int N)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int first = index * N;
  int last = first + N;
  

  while (first <= last) {


      first++;
    }

}

void cpu_search(int *isFound, float *x, int number, int size) {
  for (size_t i = 0; i < size; i++) {
      // if (x[i] == number)
  }
}

int main(int argc, char **argv)
{
  int N = 100000000;
  float *x;
  int *isFound;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&isFound, sizeof(int));
  srand(time(NULL));
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = rand() % 10000;
  }
  cudaProfilerStart();


  double time = clock();
  gpu_search<<<(N + 1024)/1024, 1024>>>(isFound, x, atoi(argv[1]), N);

  // Wait for GPU to finish before accessing on host

  printf("GPU time taking is %f\n", ((float) clock() - time)/CLOCKS_PER_SEC);
  cudaDeviceSynchronize();

  cudaProfilerStop();

  int threadsPerBlock = 512;


  int blocks = ((N + threadsPerBlock)/threadsPerBlock) / atoi(argv[2]); 

  printf("%d\n", blocks);


  int totalThreads = blocks * threadsPerBlock;

  time = clock();

  cudaProfilerStart();

  gpu_search_2<<<blocks, threadsPerBlock>>>(isFound, x, atoi(argv[1]), N/totalThreads);

  cudaDeviceSynchronize();
  printf("gpu_search_2 time taking is %f\n", ((float) clock() - time)/CLOCKS_PER_SEC);


  cudaProfilerStop();


  if (*isFound == 1) printf("found");



  time = clock();

  cpu_search(isFound, x, atoi(argv[1]), N);

  printf("CPU time taking is %f\n", ((float) clock() - time)/CLOCKS_PER_SEC);
  



  // Free memory
  cudaFree(x);

  
  return 0;
}