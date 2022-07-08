#include <thrust/sort.h>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <chrono>


using std::cout; using std::cin;
using std::endl; using std::vector;
using std::copy;

/* experiment with N */
#define nElem (8192)
#define THREADS_PER_BLOCK 8192
#define SIZE_OF_ARRAY(array) (sizeof(array)/sizeof(array[0]))

void initialData(int *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (int)(rand() & 0xFF);
    }
    
}

__global__ void predicate(int* d_in, int* d_out, int* d_numOfOne, int bitNo)
{
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	if((d_in[idx] & bitNo) == bitNo) {d_out[idx] = 0;}
	else {d_out[idx] = 1;atomicAdd(d_numOfOne, 1);}
}

__global__ void scan(int *d_in, int *d_out, int blocksize){
    extern __shared__ int temp[nElem];

    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    temp[thid] = d_in[thid];
    //upsweep
    int i;
    for (i = 2; i <= blocksize; i <<= 1)  // log(threadIdx.x)
    {   
        if((thid+1) % i == 0){
            int offset = i >> 1;
            temp[thid] += temp[thid - offset];
         } 
         
    }
    __syncthreads();
    if(thid == 0) temp[blocksize-1] = 0;
    //downsweep
    for (int j = i>>1; j >= 2 ; j >>= 1)
    {
        int offset = j >> 1;
        if ((thid+1) % j == 0)
        {
            int t = temp[thid];
            temp[thid] += temp[thid - offset];
            temp[thid - offset] = t;
        }
        
    }
    
    __syncthreads();

    d_out[thid] = temp[thid];

}


__global__ void radixSort(int* d_in, int* d_out, int* d_predict, int* d_scan, int* d_numOfOne){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int new_idx;
    if(d_predict[idx] == 0){
        new_idx = idx - d_scan[idx] + *d_numOfOne;
        d_out[new_idx] = d_in[idx];
    }
    else{
        new_idx = d_scan[idx];
        d_out[new_idx] = d_in[idx];
    }
    
    __syncthreads();
    // printf("in = %d, d_perdict = %d, d_scan = %d, d_numOfOne = %d, new index = %d, out = %d\n", d_in[idx], d_predict[idx], d_scan[idx], *d_numOfOne, new_idx, d_out[new_idx]);
}




int main(){
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    LARGE_INTEGER start, end;
    size_t size = nElem * sizeof(int);
    int *a,*b, *h_predicate_result, *h_numOfOne, *h_scan, *h_result;
    int *d_a, *d_predicate_result, *d_numOfOne, *d_scan, *d_result;
    int grid_size = (nElem + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK;
    int block_size = THREADS_PER_BLOCK;

    /* allocate space for device copies of a */
	cudaMalloc( (void **) &d_a, size );
    cudaMalloc( (void **) &d_predicate_result, size );
    cudaMalloc( (void **) &d_numOfOne, sizeof(int) );
    cudaMalloc( (void **) &d_scan, size );
    cudaMalloc( (void **) &d_result, size );

    /* allocate space for host copies of a and setup input values */
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    h_predicate_result = (int *)malloc(size);
    h_numOfOne = (int *)malloc(sizeof(int));
    h_scan = (int *)malloc(size);
    h_result = (int *)malloc(size);

    initialData(a, nElem);
    // printf("initial data :\n");
    // for (int i = 0; i < nElem; i++) printf("%d ", a[i]);
    // printf("\n\n");

    /* copy inputs to device */ 
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
    int bit = 1;
    QueryPerformanceCounter(&start);
    for (size_t i = 0; i < 8; i++)
    {
        cudaMemset(d_numOfOne, 0,sizeof(int));
        //predict
        predicate<<< grid_size, block_size >>>( d_a , d_predicate_result, d_numOfOne, bit);
        scan<<< grid_size, block_size >>>(d_predicate_result, d_scan, block_size);
        
        /* Confirmation of value
        //copy predict result and number of ones to host
        cudaMemcpy(h_numOfOne, d_numOfOne, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_predicate_result, d_predicate_result, size, cudaMemcpyDeviceToHost);
        // print the predicate array and number of ones
        printf("predicate :\n");
        for(int i=0;i<nElem;i++)printf("%d ",h_predicate_result[i]);
        printf("\n");
        printf("num of ones : %d\n",*h_numOfOne);
        
        cudaMemcpy(h_scan, d_scan, size, cudaMemcpyDeviceToHost );
        printf("\nscan data :\n");
        for (int i = 0; i < nElem; i++) printf("%d ", h_scan[i]);
        printf("\n\n");
        */
        
        radixSort<<< grid_size, block_size >>>( d_a , d_result, d_predicate_result, d_scan, d_numOfOne);
        
        cudaMemcpy(d_a, d_result, size, cudaMemcpyDeviceToDevice);
        bit <<= 1;
    }
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&end);
    double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    // printf("sorted array with radix sort :\n");
    // for(int i=0;i<nElem;i++)printf("%d ",h_result[i]);
    // printf("\n");
    printf("radix sort: time %lf[ms]\n\n", time);
    printf("\n");


    memcpy(b, a, sizeof(a));
    QueryPerformanceCounter(&start);
    thrust::sort(a, a + nElem);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&end);
    time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    // printf("\nsort data with cuda library :\n");
    // for (int i = 0; i < nElem; i++) cout << a[i] << " ";
    // printf("\n");
    printf("cada liblary: time %lf[ms]\n\n", time);

    QueryPerformanceCounter(&start);
    std::sort(b, b + SIZE_OF_ARRAY(b));
    QueryPerformanceCounter(&end);
    time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    printf("c++ liblary: time %lf[ms]\n\n", time);
    

    free(a);
    free(h_numOfOne);
    free(h_predicate_result);
    free(h_result);
    free(h_scan);
    cudaFree(d_a);
    cudaFree(d_numOfOne);
    cudaFree(d_predicate_result);
    cudaFree(d_result);
    cudaFree(d_scan);
    return 0;
}
