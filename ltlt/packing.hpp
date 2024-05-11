#ifndef _PACKING_HPP_
#define _PACKING_HPP_

#include "ltlt.hpp"
#include "blis.h"

struct skparams
{
    const void* t;
    //const double* t;
    inc_t inct;
    dim_t n;
};

void packing
     (
      struc_t strucc, \
     diag_t  diagc, \
     uplo_t  uploc, \
     conj_t  conjc, \
     pack_t  schema, \
     bool    invdiag, \
     dim_t   panel_dim, \
     dim_t   panel_len, \
     dim_t   panel_dim_max, \
     dim_t   panel_len_max, \
     dim_t   panel_dim_off, \
     dim_t   panel_len_off, \
     dim_t   panel_bcast, \
       const double*   kappa, \
       const double*   c, inc_t incc, inc_t ldc, \
             double*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx \

     );

#endif
