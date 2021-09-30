/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */

#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

#define BASIC_MARKER_REGION "Basic_Likwid_mark"

const char *dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

void square_dgemm(int n, double *A, double *B, double *C)
{
   // insert your code here: implementation of basic matrix multiply with OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside
   // the block of parallel code, but before your matrix multiply code, and
   // then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   #pragma omp parallel
   {
      #pragma omp for
      for (int arow = 0; arow < n; arow++)
      {
         LIKWID_MARKER_START(BASIC_MARKER_REGION);
         for (int bcol = 0; bcol < n; bcol++)
         {
            //LIKWID_MARKER_START(BASIC_MARKER_REGION);
            for (int k = 0; k < n; k++)
            {
               //LIKWID_MARKER_START(BASIC_MARKER_REGION);
               C[bcol * n + arow] += A[arow + k * n] * B[bcol * n + k];
               //LIKWID_MARKER_STOP(BASIC_MARKER_REGION);
            }
            //LIKWID_MARKER_STOP(BASIC_MARKER_REGION);
         }
         LIKWID_MARKER_STOP(BASIC_MARKER_REGION);
      } // end #pragma omp for
   } // end #pragma omp parallel
}
