#include "packing.hpp"

struct skparams
{
    void* t;
    inc_t inct;
    dim_t n;
}



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
       ctype*  kappa, \
       ctype*  c, inc_t incc, inc_t ldc, \
       ctype*  p,             inc_t ldp, \
                  inc_t is_p, \
       void*   params, \
       cntx_t* cntx  \
     ) \
{
    skparams* skp = (skparams*) params;
    const ctype* t = skp->t;
    const dim_t  n = skp->n;
    t += panel_len_off *inct;

    ctype kappa_local = *kappa;
    
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
        
        int last_column = n - panel_len_off
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

    PASTEMAC(ch,set0s_edge) \
    ( \
      paneld_dim*dfac, panel_dim_max*dfac, \
      panel_len, panel_len_max, \
      p, ldp  \
    ); \
    
}
