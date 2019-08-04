#include<stdio.h>
#include <stdlib.h> 
#include<time.h> 
#define intswap(A,B) {int temp=A;A=B;B=temp;}


__global__ void kernel(int *array,bool *flag,int size,int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < N; i++){
        if(i % 2 == 0){
            if(array[index + 1] > array[index]){
                intswap(array[index + 1], array[index]);
            }
        }else {
            if(array[index + 2] > array[index + 1]){
            intswap(array[index + 2], array[index + 1]);

            }  
            
    }
    __syncthreads(); 


}

int main()
{
    int BlockNumber;
    int ThreadNumber;
    int N ; 
    int *array;
	bool *flag ;
	
	srand(time(0));

    printf("Enter The array size: ");
    scanf("%d", &N);
    printf("Enter Block number: ");
    scanf("%d", &BlockNumber);
    printf("Enter Thread number: ");
    scanf("%d", &ThreadNumber);
    
    
    
    cudaMallocManaged(&array, N*sizeof(int));
    cudaMallocManaged(&flag, sizeof(bool));
    for(int i = 0; i < N; i++){
        array[i] = rand() % N + 1 ;
    }
    

	for(int i = 0; i < N; i++){
        printf("%d\n",array[i] );
	}
	
    

    kernel<<<BlockNumber, ThreadNumber>>>(array, flag,N/(BlockNumber*ThreadNumber),N);
    cudaDeviceSynchronize();

    if(*flag == true){
        printf("goal is found \n");
    }else printf("goal not found\n");


}
