#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

// MP5.2 - You can use any other block size you wish.
#define BLOCK_SIZE 256

//#define CONFLICT_FREE_OFFSET(numElements) \
//        ((numElements) >> NUM_BANKS + (numElements) >> (2 * LOG_NUM_BANKS)) 

// MP5.2 - Host Helper Functions (allocate your own data structure...)

// MP5.2 - Kernel Functions

__global__ void prescanArray_kernel(float *outArray, float *inArray, float *Res, int numElements)
{
   __shared__ float scan_array[2*BLOCK_SIZE + 512];

   unsigned int t = threadIdx.x;
   unsigned int start = 2*BLOCK_SIZE*blockIdx.x;

   int bankOffsetA = CONFLICT_FREE_OFFSET(t);
   int bankOffsetB = CONFLICT_FREE_OFFSET(t+BLOCK_SIZE);

   if (start + t < numElements)
      scan_array[t + bankOffsetA] = inArray[start + t];
   else
      scan_array[t + bankOffsetA] = 0;
   if (start + BLOCK_SIZE + t < numElements)
      scan_array[BLOCK_SIZE + t + bankOffsetB] = inArray[start + BLOCK_SIZE + t];
   else
      scan_array[BLOCK_SIZE + t + bankOffsetB] = 0;
   __syncthreads();

   // Reduction 
   int stride_R = 1;
   while (stride_R <= BLOCK_SIZE)
   {
      int index_A = (t + 1)*stride_R*2 - 1;
      int index_B = index_A - stride_R;
      index_A += CONFLICT_FREE_OFFSET(index_A);
      index_B += CONFLICT_FREE_OFFSET(index_B);
      if (index_A < 2*BLOCK_SIZE + 512)
         scan_array[index_A] += scan_array[index_B];
      stride_R = stride_R*2;
      __syncthreads();
   }

   int index_C = 2*BLOCK_SIZE - 1;
   index_C += CONFLICT_FREE_OFFSET(index_C);
   if (Res != NULL &&t == 0)
      Res[blockIdx.x] = scan_array[index_C];
    
   if (t == 0)
   {
      scan_array[index_C] = 0;
   }

/*   for (int stride_R = 1; stride_R <= BLOCK_SIZE; stride_R = stride_R*2)
   {
      int index_A = (t + 1)*stride_R*2 - 1;
      int index_B = index_A - stride_R;
      index_A += CONFLICT_FREE_OFFSET(index_A);
      index_B += CONFLICT_FREE_OFFSET(index_B);

      if (index_A < 2*BLOCK_SIZE + 512)
         scan_array[index_A] += scan_array[index_B];
      __syncthreads();
   }
*/ 

   // Post scan step
   int stride_P = BLOCK_SIZE;
   while (stride_P > 0)
   {
      int index_D = (t + 1)*stride_P*2 - 1;
      int index_E = index_D - stride_P;
      index_D += CONFLICT_FREE_OFFSET(index_D);
      index_E += CONFLICT_FREE_OFFSET(index_E);      
 
      if (index_D < 2*BLOCK_SIZE + 512)
      {
         float temp = scan_array[index_D];
         scan_array[index_D] += scan_array[index_E];
         scan_array[index_E] = temp;
      }
      stride_P = stride_P / 2; 
      __syncthreads();
   }

/*   for (stride_P = BLOCK_SIZE/2; stride_P; stride_P = stride_P/2)
   {
      int index = (t + 1)*stride_P*2 - 1;
      if (index + stride_P < 2*BLOCK_SIZE)
         scan_array[index + stride_P] += scan_array[index];
      __syncthreads();
   }
*/

   if (start + t < numElements)
      outArray[start + t] = scan_array[t + bankOffsetA];
   if (start + BLOCK_SIZE + t < numElements)
      outArray[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t + bankOffsetB];

}

// MP5.2 - Device Functions
__global__ void sumUp(float *outArray, float *Res, int numElements) 
{
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < numElements)
       outArray[start + t] += Res[blockIdx.x];
    if (start + BLOCK_SIZE + t < numElements)
       outArray[start + BLOCK_SIZE + t] += Res[blockIdx.x];
}


void prescan(float *outArray, float *inArray, int numElements, int Level, float ** Blk_level)
{
    int numBlocks = ceil((float)numElements/(BLOCK_SIZE<<1));
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    
    if (numBlocks > 1)
    {
       prescanArray_kernel<<<dimGrid, dimBlock>>>(outArray, inArray, Blk_level[Level], numElements);
       prescan(Blk_level[Level], Blk_level[Level], numBlocks, Level+1, Blk_level);
       sumUp<<<dimGrid, dimBlock>>>(outArray, Blk_level[Level], numElements);
    }
    else
    {
       prescanArray_kernel<<<dimGrid, dimBlock>>>(outArray, inArray, NULL, numElements);
    }
}


// **===-------- MP5.2 - Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
float** Blk_level;
unsigned int Level = 0;
void preallocBlockSums(int numElements)
{
   int Num_ele = numElements;
   int Level = 0;
   do
   {
      int Num = ceil((float)Num_ele/(2.0*BLOCK_SIZE));
      if (Num > 1)
         Level++;
      Num_ele = Num;
   }while(Num_ele > 1);
   Blk_level = (float**) malloc(Level * sizeof(float));

   Num_ele = numElements;   
   for (int i = 0; i < Level; i++)
   {
      int Num = ceil((float)Num_ele/(2.0*BLOCK_SIZE));
      cudaMalloc((void**)&Blk_level[i], Num*sizeof(float));
      Num_ele = Num;
   }
}

void deallocBlockSums()
{
   for (int i=0; i < Level; i++)
   {
      cudaFree(Blk_level[i]);       
   }
   free((void **)Blk_level);
   Level = 0;
}

void prescanArray(float *outArray, float *inArray, int numElements)
{
   float ** Blk_level;
   int Num_ele = numElements;
   int Level = 0;
   do
   {
      int Num = ceil((float)Num_ele/(2.0*BLOCK_SIZE));
      if (Num > 1)
         Level++;
      Num_ele = Num;
   }while(Num_ele > 1);
   Blk_level = (float**) malloc(Level * sizeof(float));

   Num_ele = numElements;   
   for (int i = 0; i < Level; i++)
   {
      int Num = ceil((float)Num_ele/(2.0*BLOCK_SIZE));
      cudaMalloc((void**)&Blk_level[i], Num*sizeof(float));
      Num_ele = Num;
   }

   prescan(outArray, inArray, numElements, 0, Blk_level);

   for (int i=0; i < Level; i++)
   {
      cudaFree(Blk_level[i]);       
   }
   free((void **)Blk_level);

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
