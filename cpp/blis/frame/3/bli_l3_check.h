/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


//
// Prototype object-based check functions.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t*  alpha, \
       const obj_t*  a, \
       const obj_t*  b, \
       const obj_t*  beta, \
       const obj_t*  c, \
       const cntx_t* cntx  \
    );

GENPROT( gemm )
GENPROT( gemmt )
GENPROT( her2k )
GENPROT( syr2k )
GENPROT( shr2k )
GENPROT( skr2k )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
             side_t  side, \
       const obj_t*  alpha, \
       const obj_t*  a, \
       const obj_t*  b, \
       const obj_t*  beta, \
       const obj_t*  c, \
       const cntx_t* cntx  \
    );

GENPROT( hemm )
GENPROT( symm )
GENPROT( shmm )
GENPROT( skmm )
GENPROT( trmm3 )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t*  alpha, \
       const obj_t*  a, \
       const obj_t*  beta, \
       const obj_t*  c, \
       const cntx_t* cntx  \
    );

GENPROT( herk )
GENPROT( syrk )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
             side_t  side, \
       const obj_t*  alpha, \
       const obj_t*  a, \
       const obj_t*  b, \
       const cntx_t* cntx  \
    );

GENPROT( trmm )
GENPROT( trsm )


// -----------------------------------------------------------------------------

void bli_gemm_basic_check
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );

void bli_gemmt_basic_check
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );

void bli_hemm_basic_check
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );

void bli_trsm_basic_check
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );

void bli_herk_basic_check
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  ah,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );

void bli_her2k_basic_check
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  bh,
       const obj_t*  b,
       const obj_t*  ah,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );

void bli_l3_basic_check
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx
     );
