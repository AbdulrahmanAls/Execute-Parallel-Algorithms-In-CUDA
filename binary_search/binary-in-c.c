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
 
int main()
{
   int n, search, *array;
   struct timeval start, end;
 
   printf("Enter number of elements\n");
   scanf("%d",&n);

   array = (int*) malloc(n * sizeof(int));
   for (int i = 0; i < n; i++)
      array[i] = i; 
 
   printf("Enter value to find\n");
   scanf("%d", &search);
 
    gettimeofday(&start, NULL);
   int first = 0;
   int last = n - 1;
   int middle = (first+last)/2;
    
 
   while (first <= last) {
      if (array[middle] < search)
         first = middle + 1;    
      else if (array[middle] == search) {
         printf("%d found at location %d.\n", search, middle+1);
         break;
      }
      else
         last = middle - 1;
 
      middle = (first + last)/2;
   }
   gettimeofday(&end, NULL);
   if (first > last)
      printf("Not found! %d isn't present in the list.\n", search);

      
      printf("Total time elapsed : %.0lf us\n\n", time_diff(start , end));
    
 
 
   return 0;  
}

