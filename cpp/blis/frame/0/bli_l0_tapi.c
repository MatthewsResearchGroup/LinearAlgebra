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

#include "blis.h"

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername ) \
\
void PASTEMAC(ch,opname) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
             ctype* psi  \
     ) \
{ \
	bli_init_once(); \
\
	ctype chi_conj; \
\
	PASTEMAC(ch,copycjs)( conjchi, *chi, chi_conj ); \
	PASTEMAC(ch,kername)( chi_conj, *psi ); \
}

INSERT_GENTFUNC_BASIC( addsc, adds )
INSERT_GENTFUNC_BASIC( divsc, invscals )
INSERT_GENTFUNC_BASIC( subsc, subs )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername ) \
\
void PASTEMAC(ch,opname) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
             ctype* psi  \
     ) \
{ \
	bli_init_once(); \
\
	ctype chi_conj; \
\
	PASTEMAC(ch,copycjs)( conjchi, *chi, chi_conj ); \
	PASTEMAC(ch,kername)( chi_conj ); \
	PASTEMAC(ch,copys)( chi_conj, *psi ); \
}

INSERT_GENTFUNC_BASIC( invertsc, inverts )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, kername ) \
\
void PASTEMAC(ch,opname) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
             ctype* psi  \
     ) \
{ \
	bli_init_once(); \
\
	ctype chi_conj; \
	ctype_r chi_conj_r, chi_conj_i; \
\
	PASTEMAC(ch,copycjs)( conjchi, *chi, chi_conj ); \
	PASTEMAC(ch,gets)( chi_conj, chi_conj_r, chi_conj_i ); \
	PASTEMAC(ch,sets)( -chi_conj_r, -chi_conj_i, *psi ); \
}

INSERT_GENTFUNCR_BASIC( negsc, inverts )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername ) \
\
void PASTEMAC(ch,opname) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
             ctype* psi  \
     ) \
{ \
	bli_init_once(); \
\
	if ( PASTEMAC(ch,eq0)( *chi ) ) \
	{ \
		/* Overwrite potential Infs and NaNs. */ \
		PASTEMAC(ch,set0s)( *psi ); \
	} \
	else \
	{ \
		ctype chi_conj; \
\
		PASTEMAC(ch,copycjs)( conjchi, *chi, chi_conj ); \
		PASTEMAC(ch,kername)( chi_conj, *psi ); \
	} \
}

INSERT_GENTFUNC_BASIC( mulsc, scals )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype*   chi, \
             ctype_r* absq  \
     ) \
{ \
	bli_init_once(); \
\
	ctype_r chi_r; \
	ctype_r chi_i; \
	ctype_r absq_i; \
\
	( void )absq_i; \
\
	PASTEMAC(ch,chr,gets)( *chi, chi_r, chi_i ); \
\
	/* absq   = chi_r * chi_r + chi_i * chi_i; \
	   absq_r = 0.0; (thrown away) */ \
	PASTEMAC(ch,absq2ris)( chi_r, chi_i, *absq, absq_i ); \
\
	( void )chi_i; \
}

INSERT_GENTFUNCR_BASIC( absqsc )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype*   chi, \
             ctype_r* norm  \
     ) \
{ \
	bli_init_once(); \
\
	/* norm = sqrt( chi_r * chi_r + chi_i * chi_i ); */ \
	PASTEMAC(ch,chr,abval2s)( *chi, *norm ); \
}

INSERT_GENTFUNCR_BASIC( normfsc )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype* chi, \
             ctype* psi  \
     ) \
{ \
	bli_init_once(); \
\
	/* NOTE: sqrtsc/sqrt2s differs from normfsc/abval2s in the complex domain. */ \
	PASTEMAC(ch,sqrt2s)( *chi, *psi ); \
}

INSERT_GENTFUNC_BASIC( sqrtsc )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype* chi, \
             ctype* psi  \
     ) \
{ \
	bli_init_once(); \
\
	const ctype_r chi_r = PASTEMAC(ch,real)( *chi ); \
\
	PASTEMAC(chr,ch,sqrt2s)( chi_r, *psi ); \
}

INSERT_GENTFUNCR_BASIC( sqrtrsc )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype*  chi, \
             double* zeta_r, \
             double* zeta_i  \
     ) \
{ \
	bli_init_once(); \
\
	PASTEMAC(ch,d,gets)( *chi, *zeta_r, *zeta_i ); \
}

INSERT_GENTFUNC_BASIC( getsc )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       double zeta_r, \
       double zeta_i, \
       ctype* chi  \
     ) \
{ \
	bli_init_once(); \
\
	PASTEMAC(d,ch,sets)( zeta_r, zeta_i, *chi ); \
}

INSERT_GENTFUNC_BASIC( setsc )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype*   chi, \
             ctype_r* zeta_r, \
             ctype_r* zeta_i  \
     ) \
{ \
	bli_init_once(); \
\
	PASTEMAC(ch,chr,gets)( *chi, *zeta_r, *zeta_i ); \
}

INSERT_GENTFUNCR_BASIC( unzipsc )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype_r* zeta_r, \
       const ctype_r* zeta_i, \
             ctype*   chi  \
     ) \
{ \
	bli_init_once(); \
\
	PASTEMAC(chr,ch,sets)( *zeta_r, *zeta_i, *chi ); \
}

INSERT_GENTFUNCR_BASIC( zipsc )

// -----------------------------------------------------------------------------

void bli_igetsc
     (
       const dim_t*  chi,
             double* zeta_r,
             double* zeta_i
     )
{
	bli_init_once();

	PASTEMAC(i,d,gets)( *chi, *zeta_r, *zeta_i );
}

void bli_isetsc
     (
       double zeta_r,
       double zeta_i,
       dim_t* chi
     )
{
	bli_init_once();

	PASTEMAC(d,i,sets)( zeta_r, zeta_i, *chi );
}

