#include <stdio.h>

int add(int x, int y);
void swap(int *x, int*y);
 
int main(){

    int a = 10;
    int b = 26;

    int result = add(a,b);

     swap(&a , &b);

    printf("Sum is %d\n", result);
    //printf("After swapping a = %d and b = %d", a,b);

    int arr[4] = {3,1,8,24};
    printf("%p\n", arr);
    printf("%p\n", (arr+1));

    
    return 0;
 }

 int add(int x, int y){
    int sum = x + y;
    return sum;
 }

 void swap(int* x, int* y){
    int temp = *x;
    *x = *y;
    *y = temp;

 }