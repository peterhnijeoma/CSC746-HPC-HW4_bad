/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */

#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

void square_dgemm(int n, double* A, double* B, double* C) 
{
   // insert your code here: implementation of basic matrix multiply with OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside
   // the block of parallel code, but before your matrix multiply code, and
   // then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   double cvalue = 0.0;     // accumulator fo matrix rowXcolumn product

   #pragma omp parallel for collapse(2)
   for (int arow = 0; arow < n; arow++)
   {
      for (int bcol = 0; bcol < n; bcol++)
      {
         cvalue = 0.0; // initialize accumulator for another product value
         #pragma omp parallel for reduction (+:cvalue)
         for (int k = 0; k < n; k++)
         {
            // accumulated matrix product
            // note that the indexes k and bcol are multiplied by n (the matrix size)
            // this is done to navigate the 1D vector properly 
            cvalue += A[arow+k*n] * B[bcol*n+k];
         }
         //std::cout << "cvalue = " << cvalue << '\n';   //debug - show mul in action
         //std::cout << "C index is " << bcol*n+arow << '\n'; //debug - show product index - flat memory
         
         C[bcol*n+arow] += cvalue; // write to product matrix - column major
      }
   }
}
