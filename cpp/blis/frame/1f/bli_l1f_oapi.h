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
// Prototype object-based interfaces.
//

#undef  GENTPROT
#define GENTPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alphax, \
       const obj_t* alphay, \
       const obj_t* x, \
       const obj_t* y, \
       const obj_t* z  \
       BLIS_OAPI_EX_PARAMS  \
     );

GENTPROT( axpy2v )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* x, \
       const obj_t* y  \
       BLIS_OAPI_EX_PARAMS  \
     );

GENTPROT( axpyf )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* xt, \
       const obj_t* x, \
       const obj_t* y, \
       const obj_t* rho, \
       const obj_t* z  \
       BLIS_OAPI_EX_PARAMS  \
     );

GENTPROT( dotaxpyv )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alphaw, \
       const obj_t* alphax, \
       const obj_t* at, \
       const obj_t* a, \
       const obj_t* w, \
       const obj_t* x, \
       const obj_t* beta, \
       const obj_t* y, \
       const obj_t* z  \
       BLIS_OAPI_EX_PARAMS  \
     );

GENTPROT( dotxaxpyf )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* x, \
       const obj_t* beta, \
       const obj_t* y  \
       BLIS_OAPI_EX_PARAMS  \
     );

GENTPROT( dotxf )

