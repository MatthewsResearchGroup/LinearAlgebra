#include "packing.hpp"

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
    const double* t = (double*)skp->t;
    const dim_t  n = skp->n;
    const inc_t  inct = skp->inct;
    t += panel_len_off * inct;

    double kappa_local = *kappa;
    
    if (panel_len_off == 0)
    {
        for (int i = 0; i < panel_dim; i++)
                p[i] = kappa_local *t[0] * c[i*incc+(1)*ldc];
        for (int j = 1; j < panel_len; j++)
        {
            for (int i = 0; i < panel_dim; i++)
            {
                p[i+j*ldp] = kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
            }
        }
    }
    else if (n - panel_len_off == panel_len)
    {
        for (int j = 0; j < n - panel_len_off; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] = kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
        
        int last_column = n - panel_len_off;
        for (int i = 0; i < panel_dim; i++)
            p[i+(last_column)*ldp] = - kappa_local * t[(last_column-1)*inct]*c[i*incc+(last_column-1)*ldc];
    }
    else
    {
        for (int j = 0; j < panel_len; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] = kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];

    }
    // The first block is the last block. 




    // for (int j = 0; j < panel_len; j++)
    // {
    //     if ((panel_len_off == 0) and (j == 0))
    //     {
    //         for (int i = 0; i < panel_dim; i++)
    //             p[i+j*ldp] = kappa_local *t[j*inct] * c[i*incc+(j+1)*ldc];
    //     }
    //     else if (j == n - panel_len_off)
    //     {
    //         for (int i = 0; i < panel_dim; i++)
    //             p[i+j*ldp] = - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
    //     }
    //     else
    //     {
    //         for (int i = 0; i < panel_dim; i++)
    //         {
    //             p[i+j*ldp] = kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
    //         }
    //     }
    // }

    PASTEMAC(d,set0s_edge) \
    ( \
      panel_dim*panel_bcast, panel_dim_max*panel_bcast, \
      panel_len, panel_len_max, \
      p, ldp  \
    ); \
    
}

