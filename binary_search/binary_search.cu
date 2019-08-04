#include<stdio.h>

__global__ void kernel(int *array,int goal,bool *flag,int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int first = index * size ;
    int last = first + size;
    int middle = (first+last)/2;

    while (first <= last) {
        if (array[middle] < goal)
           first = middle + 1;    
        else if (array[middle] == goal) {
            // printf("number is found in bolackid=%d threadid=%d\n",blockIdx.x,threadIdx.x);
            *flag = true; 
            // assert(0);
           break;
        }
        else
           last = middle - 1;
   
        middle = (first + last)/2;
     }

    if(array[threadIdx.x] == goal){
        *flag = true; 
    }
}

int main()
{
    int BlockNumber;
    int ThreadNumber;
    int Goal;
    int N ; 
    int *array;
    bool *flag ;

    printf("Enter The array size: ");
    scanf("%d", &N);
    printf("Enter Block number: ");
    scanf("%d", &BlockNumber);
    printf("Enter Thread number: ");
    scanf("%d", &ThreadNumber);
    printf("Enter the number to find: ");
    scanf("%d", &Goal);
    
    
    
    cudaMallocManaged(&array, N*sizeof(int));
    cudaMallocManaged(&flag, sizeof(bool));
    for(int i = 0; i < N; i++){
        array[i] = i ;
    }
    

    kernel<<<BlockNumber, ThreadNumber>>>(array, Goal, flag,N/(BlockNumber*ThreadNumber));
    cudaDeviceSynchronize();

    if(*flag == true){
        printf("goal is found \n");
    }else printf("goal not found\n");


}
