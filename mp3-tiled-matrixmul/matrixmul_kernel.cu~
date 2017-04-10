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

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#define TILE_WIDTH 16
//#define max(a,b,c) (a>b?(a>c?a:c):(b>c?b:c));

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    __shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];
  
    int bx = blockIdx.x;
    int	by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Calculate row index and column index for P
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
   
    //Find the longest value of height and width of M and N
    //int Max_width = max(M.height, M.width, N.width); 

    //Multiply the two matrices
    float Pvalue = 0;
    for (int m = 0; m < (M.width-1)/TILE_WIDTH+1; ++m)
    {
        if (Row < M.height && m*TILE_WIDTH+tx < M.width)       
           M_s[ty][tx] = M.elements[Row*M.width + m*TILE_WIDTH + tx];
        else
           M_s[ty][tx] = 0;
        if (Col < N.width && m*TILE_WIDTH+ty < M.width)
           N_s[ty][tx] = N.elements[(m*TILE_WIDTH + ty)*N.width + Col];         
        else
           N_s[ty][tx] = 0;

           __syncthreads();  

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
             Pvalue += M_s[ty][k]*N_s[k][tx];
            // __syncthreads(); 
        }
             __syncthreads();
    }

    if (Row < M.height && Col < N.width)
    P.elements[Row * N.width + Col] = Pvalue;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
