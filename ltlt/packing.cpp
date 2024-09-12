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

    //printf("Print t\n");
    //for (auto i : range(n-1))
    //    printf("%f, ", t[i*inct]);
    //printf("\n\n");
    //printf("incc, ldc, ldp, inct = %d, %d, %d, %d\n", incc, ldc, ldp, inct);
    double kappa_local = *kappa;
    //printf("kappa_local = %f\n", kappa_local);
    
    if ((panel_len_off) == 0 and (n > panel_len))
    {
        for (int i = 0; i < panel_dim; i++)
                p[i] =  kappa_local *t[0] * c[i*incc+(1)*ldc];
        for (int j = 1; j < panel_len; j++)
        {
            for (int i = 0; i < panel_dim; i++)
            {
                p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
            }
        }
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
        {
            p[i] = kappa_local *t[0] * c[i*incc+(1)*ldc];
            // printf("p[%d] = kappa_local * t[0] * c[%d]\n", i, i*incc+(1)*ldc);
            // printf("%f = %f * %f * %f \n", p[i], kappa_local , t[0] , c[i*incc+(1)*ldc]);
        }    
        for (int i = 0; i < panel_dim; i++)
        {
            // printf("Hello here ldc %d, incc: %d\n", ldc, incc);
            p[i+(n-1)*ldp] = - kappa_local * t[(n-2)*inct] * c[i*incc+(n-2)*ldc];
            // printf("p[%d] = - kappa_local * t[%d] * c[%d]\n", i+(n-1)*ldp, (n-2)*inct, i*incc+(n-2)*ldc);
            // printf("%f = - %f * %f * %f \n", p[i+(n-1)*ldp] , kappa_local ,  t[(n-2)*inct] ,c[i*incc+(n-2)*ldc]);
        }
        for (int j = 1; j < n - 1; j++)
        {
            for (int i = 0; i < panel_dim; i++)
            {
                //p[i+j*ldp] = kappa_local * t[j]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)]*c[i*incc+(j-1)*ldc];
                p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
                // printf("p[%d] = kappa_local * t[%d] * c[%d] - kappa_local * t[%d] * c[%d]\n", i+j*ldp, j*inct, i*incc+(j+1)*ldc, (j-1)*inct, i*incc+(j-1)*ldc);
                // printf("%f = %f * %f * %f - %f * %f * %f\n", p[i+j*ldp],  kappa_local,  t[j*inct], c[i*incc+(j+1)*ldc],  kappa_local, t[(j-1)*inct], c[i*incc+(j-1)*ldc]);
            }
        }
    }
    else if (n - panel_len_off == panel_len)
    {
        // printf("Chao 3\n");
        for (int j = 0; j < n - panel_len_off; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];
        
        int last_column = n - panel_len_off;
        for (int i = 0; i < panel_dim; i++)
            p[i+(last_column)*ldp] = - kappa_local * t[(last_column-1)*inct]*c[i*incc+(last_column-1)*ldc];
    }
    else
    {
        // printf("Chao 4\n");
        for (int j = 0; j < panel_len; j++)
        for (int i = 0; i < panel_dim; i++)
            p[i+j*ldp] =  kappa_local * t[j*inct]*c[i*incc+(j+1)*ldc] - kappa_local * t[(j-1)*inct]*c[i*incc+(j-1)*ldc];

    }
    // The first block is the last block. 

    // printf("\n\nPrint P\n");

    // for (auto i = 0; i < ldp; i++)
    // {
    //     for (auto j= 0; j < n; j++)
    //     {
    //         printf("%f ", p[i+j*ldp]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");


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

