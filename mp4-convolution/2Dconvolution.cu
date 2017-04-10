/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix convolution.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <2Dconvolution_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
        cudaEvent_t GPU_alltimestart, CPU_timestart, GPU_alltimestop, CPU_timestop;
        cudaEventCreate(&GPU_alltimestart);
       	cudaEventCreate(&CPU_timestart);
        cudaEventCreate(&GPU_alltimestop);
        cudaEventCreate(&CPU_timestop);

           
	srand(2012);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
		N  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		P  = AllocateMatrix(N.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = NULL; 
		unsigned int data_read = 0;
		cutReadFilei(argv[1], &params, &data_read, true);
		if(data_read != 2){
			printf("Error reading parameter file\n");
			cutFree(params);
			return 1;
		}

		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		N  = AllocateMatrix(params[0], params[1], 0);		
		P  = AllocateMatrix(params[0], params[1], 0);
		cutFree(params);
		(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[3]);
	}

    //Measure GPU overall time, Start.
    //unsigned int GPU_alltime;
    //cutCreateTimer(&GPU_alltime);
    //cutStartTimer(GPU_alltime);
    cudaEventRecord(GPU_alltimestart,0);    

    // M * N on the device
    ConvolutionOnDevice(M, N, P);

    //Measure GPU overall time, Stop.
    //cutStopTimer(GPU_alltime);
    //GPU_headtime = cutGetTimerValue(GPU_alltime)/1000 - cutGetTimerValue(GPU_time)/1000;
    cudaEventRecord(GPU_alltimestop,0);
    cudaEventSynchronize(GPU_alltimestart);
    cudaEventSynchronize(GPU_alltimestop);
    float GPU_alltime;
    cudaEventElapsedTime(&GPU_alltime, GPU_alltimestart, GPU_alltimestop);
    printf("GPU overall time is: %f ms\n", GPU_alltime);
    //cutDeleteTimer(GPU_alltime);

    //Measure CPU time, Start.
    //unsigned int CPU_time;
    //cutCreateTimer(&CPU_time);
    //cutStartTimer(CPU_time);
    cudaEventRecord(CPU_timestart,0);

    // compute the matrix convolution on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);
    
    //Measure CPU time,	Stop.
    //cutStopTimer(CPU_time);
    cudaEventRecord(CPU_timestop,0);
    cudaEventSynchronize(CPU_timestart);
    cudaEventSynchronize(CPU_timestop);
    float CPU_time;
    cudaEventElapsedTime(&CPU_time, CPU_timestart, CPU_timestop);
    printf("CPU process time is: %f ms\n", CPU_time);
    //cutDeleteTimer(CPU_time);

    
    // in this case check if the result is equivalent to the expected soluion
    CUTBoolean res = cutComparefe(reference.elements, P.elements, P.width * P.height, 0.001f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Set timer
    cudaEvent_t GPU_timestart, GPU_timestop;
    cudaEventCreate(&GPU_timestart);
    cudaEventCreate(&GPU_timestop);

    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory
    
    // Setup the execution configuration
    cudaMemcpyToSymbol(M_c, M.elements, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
    dim3 DimGrid((N.width-1)/TILE_SIZE+1, (N.height-1)/TILE_SIZE+1, 1);
    dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    //Measure GPU time, Start.
    cudaEventRecord(GPU_timestart,0);
    //unsigned int GPU_time;
    //cutCreateTimer(&GPU_time);
    //cutStartTimer(GPU_time);

    // Launch the device computation threads!
    ConvolutionKernel<<<DimGrid,DimBlock>>>(Md, Nd, Pd);
    cudaDeviceSynchronize();

    //Measure GPU time, Stop.
    cudaEventRecord(GPU_timestop,0);
    cudaEventSynchronize(GPU_timestart);
    cudaEventSynchronize(GPU_timestop);
    float GPU_time;
    cudaEventElapsedTime(&GPU_time, GPU_timestart, GPU_timestop);
    printf("GPU process time is: %f ms\n", GPU_time);
    //cutStopTimer(GPU_time);
    //printf("GPU process time is: %f ms\n", cutGetTimerValue(GPU_time));
    //cutDeleteTimer(GPU_time);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height * M->width;
	cutReadFilef(file_name, &(M->elements), &data_read, true);
	return data_read;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    cutWriteFilef(file_name, M.elements, M.width*M.height,
                       0.0001f);
}
