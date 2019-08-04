#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

double time_diff(struct timeval x , struct timeval y)
{
    double x_ms , y_ms , diff;
      
    x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
    y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
      
    diff = (double)y_ms - (double)x_ms;
      
    return diff;
}
void swap(int *,int *);
void oddeven_sort(int *, int);
 
int main()
{
   int n, search, *array;
   struct timeval start, end;
   srand((unsigned)time(NULL));
 
   printf("Enter number of elements\n");
   scanf("%d",&n);

   array = (int*) malloc(n * sizeof(int));

   for (int i = 0; i < n; i++)
      array[i] = rand()% n ;

    gettimeofday(&start, NULL);
    oddeven_sort(array,n);
    gettimeofday(&end, NULL);

    printf("Total time elapsed : %.0lf us\n\n", time_diff(start , end));

    for(int i= n - 1000;i<n;i++)
   {
    printf("%d ", array[i]);
   }

    printf("\n");

}
 
/* swaps the elements */
void swap(int * x, int * y)
{
    int temp;
 
    temp = *x;
    *x = *y;
    *y = temp; 
}
 
/* sorts the array using oddeven algorithm */
void oddeven_sort(int * x, int n)
{
    int sort = 0, i;
 
    while (!sort)
    {
        sort = 1;
        for (i = 1;i < n;i += 2)
        {
            if (x[i] > x[i+1])
            {
                swap(&x[i], &x[i+1]);
                sort = 0;
            }
        }
        for (i = 0;i < n - 1;i += 2)
        {
            if (x[i] > x[i + 1])
            {
                swap(&x[i], &x[i + 1]);
                sort = 0;
            }
        }
    }
}