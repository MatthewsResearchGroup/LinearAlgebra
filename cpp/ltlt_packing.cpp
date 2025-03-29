#include "ltlt.hpp"

#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"

void packing
     ( \
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
     ) \
{
    skparams* skp = (skparams*) params;
    const dim_t n = skp->n;
    const inc_t inct = skp->inct;
    const double* t = (double*)skp->t + panel_len_off*inct;

    double kappa_local = *kappa;
    int last_column = n - panel_len_off - 1;

    if ((panel_len_off) == 0 and (n > panel_len))
    {
        for (int i = 0; i < panel_dim; i++)
            p[i] =  kappa_local *t[0] * c[i*incc+(1)*ldc];

        for (int j = 1; j < panel_len; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
    }
    /*
     * The panel is the first and the last one (which means the length of matrix A is less or equal to the panel_dim)
     * We need to deal some edge cases:
     * 1. The first column uses the equation p[i] = kappa * t[0*inct] * c[i*incc+(1)*ldc]
     * 2. c is row major and p is column major
     *
     */
    else if ((panel_len_off) == 0 and (n <= panel_len))
    {
        for (int i = 0; i < panel_dim; i++)
            p[i] = kappa_local *t[0] * c[i*incc+(1)*ldc];

        for (int j = 1; j < n - 1; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];

        for (int i = 0; i < panel_dim; i++)
            p[i+(last_column)*ldp] = - kappa_local * t[(n-2)*inct] * c[i*incc+(n-2)*ldc];
    }
    else if (n - panel_len_off == panel_len)
    {
        for (int j = 0; j < n - panel_len_off; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];

        for (int i = 0; i < panel_dim; i++)
            p[i+(last_column)*ldp] = - kappa_local * t[(last_column-1)*inct]*c[i*incc+(last_column-1)*ldc];
    }
    else
    {
        for (int j = 0; j < panel_len; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
    }

    PASTEMAC(d,set0s_edge) \
    ( \
      panel_dim*panel_bcast, panel_dim_max*panel_bcast, \
      panel_len, panel_len_max, \
      p, ldp  \
    ); \

}

