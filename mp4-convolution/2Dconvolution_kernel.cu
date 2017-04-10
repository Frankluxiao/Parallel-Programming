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

#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix convolution kernel specification
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{

    __shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int TILE_SIZE = BLOCK_SIZE - KERNEL_SIZE + 1;

    //Calculate row index and column index for P
    int Row_o = by * TILE_SIZE + ty;
    int Col_o = bx * TILE_SIZE + tx;
    int n = KERNEL_SIZE/2;
    int Row_i = Row_o - n;
    int Col_i = Col_o - n;

    //Multiply the two matrices
    float Pvalue = 0;  
    if (Row_i >= 0 && Row_i < N.height && Col_i >= 0 && Col_i < N.width)
       N_s[ty][tx] = N.elements[Row_i*N.width + Col_i];
    else
       N_s[ty][tx] = 0;
       __syncthreads();


    if (ty < TILE_SIZE && tx < TILE_SIZE)
    {
        for (int i = 0; i < KERNEL_SIZE; i++)
        {
            for(int j = 0; j < KERNEL_SIZE; j++)
            {
                Pvalue += M_c[i][j]*N_s[i+ty][j+tx];
            }
            __syncthreads();
        }

        if (Row_o < P.height && Col_o < P.width)
           P.elements[Row_o * P.width + Col_o] = Pvalue;
    }
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
