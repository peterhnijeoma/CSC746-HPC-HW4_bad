#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   int ii, kk, jj;     // block row and col indexes
 
  // declare and dynamically allocate 2D arrays
  double **AA, **BB, **CC;
  double **AAA, **BBB, **CCC;

  AA = new double *[n];
  BB = new double *[n];
  CC = new double *[n];
  for (int i = 0; i < n; i++)
  {
     AA[i] = new double [n];
     BB[i] = new double [n];
     CC[i] = new double [n];
  }

  // copy column major vector A, B, and C into 2D arrays AA, BB, and CC respectively 
  for (int k = 0, i = 0; i < n*n; k++, i+=n)
  {
     for (int j = 0; j < n; j++)
     {
        AA[j][k] = A[i+j];
        BB[j][k] = B[i+j];
        CC[j][k] = C[i+j];
     }
  }
  
  //std::cout << "start mm \n";
  // block matrix multiplication logic
  for (ii = 0; ii < n; ii += block_size)  // partition rows by block size; iterate for n/block_size blocks
  {
    for (jj = 0; jj < n; jj += block_size) // partition columns by block size; iterate for n/block_size blocks
    {
      copy_matrix_block(CC, CCC, ii*block_size, jj*block_size, block_size);
      for (kk = 0; kk < n; kk += block_size)  // for each row and column of blocks
      {
        copy_matrix_block(AA, AAA, ii*block_size, kk*block_size, block_size);
        copy_matrix_block(BB, BBB, kk*block_size, jj*block_size, block_size);
        // basic matrix multiple applied to matrix blocks
        matrix_multiply(AAA, BBB, CCC, block_size, block_size);
      //   for (int arow = ii; arow < ii + block_size; arow++)
      //   {
      //     for (int bcol = jj; bcol < jj + block_size; bcol++)
      //     {
      //        for (int k = kk; k < kk + block_size; k++)
      //        {
      //          //  std::cout << "arow is: " << arow << "; bcol is: " << bcol << "; k is: " << k << '\n';
      //          //  std::cout << "CC[" << arow << "][" << bcol << "] before is: " << CC[arow][bcol] << '\n';
      //           CC[arow][bcol] += AA[arow][k] * BB[k][bcol];
      //          //  std::cout << "CC[" << arow << "][" << bcol << "] after is: " << CC[arow][bcol] << '\n';
      //        }
      //     }
      //   }
      }
      // copy block product to produc matrix
      copy_block_to_matrix(CCC, CC, ii*block_size, jj*block_size, block_size);
    }
  }
  
  // copy 2d array CC to column major vector C
  for (int i = 0; i < n; i++)
  {
     for (int j = 0; j < n; j++)
     {
        C[i*n+j] = CC[j][i];
     }
  }

  // release allocated memory
  for (int i = 0; i < n; i++)
  {
     delete [] AA[i];
     delete [] BB[i];
     delete [] CC[i];
  }
  delete [] AA;
  delete [] BB;
  delete [] CC;
}

void copy_matrix_block(double **S, double **D, int brl, int bcl, int bs)
{
  for (int row = brl; row < bs; row++)
  {
     for (int col = bcl; col < bs; col++)
     {
        D[row][col] = S[row][col];
     }
  }
}

void matrix_multiply(double **AA, double **BB, double **PROD, int num_rows, int num_cols)
{
   for (int row = 0; row < num_rows; row++)
   {
      for (int col = 0; col < num_cols; col++)
      {
         for (int k = 0; k < num_cols; k++)
         {
            //  std::cout << "arow is: " << arow << "; bcol is: " << bcol << "; k is: " << k << '\n';
            //  std::cout << "CC[" << arow << "][" << bcol << "] before is: " << CC[arow][bcol] << '\n';
            PROD[row][col] += AA[row][k] * BB[k][col];
            //  std::cout << "CC[" << arow << "][" << bcol << "] after is: " << CC[arow][bcol] << '\n';
         }
      }
   }
}

void copy_block_to_matrix(double **S, double **D, int brl, int bcl, int bs)
{
  for (int row = 0; row < bs; row++)
  {
     for (int col = 0; col < bs; col++)
     {
        D[brl+row][bcl+col] = S[row][col];
     }
  }
}
