#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#define TILE_WIDTH 16

__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins)
{


        // Original Code From slides. Using shared memory
/*	__shared__ int temp[1024];
	temp[threadIdx.x] = 0;
	__syncthreads();
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	int size = HISTO_HEIGHT * HISTO_WIDTH;
	while (i < size) 
        {
		atomicAdd(&temp[input[i]], 1);
		i += offset;
	} 
	__syncthreads();
	atomicAdd(&(bins[threadIdx.x]), temp[threadIdx.x]);
*/


        // Optimization Step 1 (Sometime works, but slower)
        // Each thread processes input in a strided pattern       
/*	__shared__ int Buffer[1024];	
	Buffer[threadIdx.x] = 0;
	Buffer[threadIdx.x + 512] = 0;
	int i = threadIdx.x + blockIdx.x * blockDim.x;

        //All threads process consecutive elements
	for (int Stride = 0; Stride < (width / TILE_WIDTH); ++Stride)
	{
            atomicAdd(Buffer + input[((i / TILE_WIDTH) * height) + ((i % TILE_WIDTH) * (width / TILE_WIDTH)) +  Stride], 1);
	}
	__syncthreads();

	atomicAdd(bins + threadIdx.x, Buffer[threadIdx.x]);
	atomicAdd(bins + threadIdx.x + 512, Buffer[threadIdx.x + 512]);
*/

       
        // Optimization Step 2 (Works)
        // Using 2D to process data
/*	int Col = blockDim.x * blockIdx.x + threadIdx.x;
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	if (Row == 0 && Col < 1024) 
        {
	   bins[Col] = 0;
	}
	__syncthreads();
	if (Row < height && Col < width) 
        {
	   atomicAdd(&bins[input[Col + Row * ((INPUT_WIDTH + 128) & 0xFFFFFF80)]], 1);
	}
*/

        // Optimization Step 3 (Works)
        // Using 2D to process data, and setting boundary 
        int Col = blockDim.x * blockIdx.x + threadIdx.x;
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Mask = (INPUT_WIDTH + 128) & 0xFFFFFF80;

	if (Row == 0 && Col < 1024) 
        {
	   bins[Col] = 0;
	}

	int i;
	__syncthreads();

	if (Row < height && Col < width) 
        {
	   i = input[Col + Row * Mask];
	   if (bins[i] < 255)
              atomicAdd(&bins[i], 1);
	}	

}

__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;	
	output[Index] = (uint8_t)((input[Index] < UINT8_MAX) * input[Index]) + (input[Index] >= UINT8_MAX) * UINT8_MAX;
	__syncthreads();
}

void opt_2dhisto(uint32_t* D_input, size_t height, size_t width, uint8_t* D_bins, uint32_t* G_bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */


    //Setting dimension
    int Grid_idx = ((INPUT_WIDTH + 128) & 0xFFFFFF80) / TILE_WIDTH;
    int Grid_idy = INPUT_HEIGHT / TILE_WIDTH;

    //Launch kernel
    dim3 DimGrid(Grid_idx, Grid_idy, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    opt_2dhistoKernel<<<DimGrid, DimBlock>>>(D_input, height, width, G_bins);
    cudaThreadSynchronize();      

    // Convert 32 bit to 8 bit
    opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH/512, 512>>>(G_bins, D_bins, 1024);
    cudaThreadSynchronize();    

}

/* Include below the implementation of any other functions you need */
void* AllocDev(size_t size)
{
	void* Device;
	cudaMalloc(&Device, size);
	return Device;
}


void CpyToDev(void* Device, void* Host, size_t size)
{
	cudaMemcpy(Device, Host, size, cudaMemcpyHostToDevice);
}


void CpyFromDev(void* Host, void* Device, size_t size)
{
	cudaMemcpy(Host, Device, size, cudaMemcpyDeviceToHost);
}


void FreeDev(void* Device)
{
	cudaFree(Device);
}

