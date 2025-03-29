#include "ltlt.hpp"

#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"

/*
 * x <- alpha T x
 */
 static void sktrmv(double alpha, const row_view<const double>& T, const row_view<double>& x)
 {
     PROFILE_FUNCTION
     auto n = x.length();
     MARRAY_ASSERT(T.length(0) == n-1);

     if (n == 0)
         return;

     if (n == 1)
     {
         x[0] = 0.0;
         return;
     }

     auto ximinus1 = x[0];
     x[0] = alpha * (-T[0] * x[1]);

     for (auto i : range(1,n-1))
     {
         auto xi = alpha * (T[i-1] * ximinus1 - T[i] * x[i+1]);
         ximinus1 = x[i];
         x[i] = xi;
     }

     x[n-1] = alpha * (T[n-2] * ximinus1);

     PROFILE_FLOPS(4*n-4);
 }

 /*
  * A <- alpha T A
  */
 static void sktrmm(double alpha, const row_view<const double>& T, const matrix_view<double>& A)
 {
     PROFILE_FUNCTION
     for (auto i : columns(A))
         sktrmv(alpha, T, A[all][i]);
 }

template <int Options>
void gemm_sktri
     (
       double  alpha,
       const matrix_view<const double>&  a,
       const row_view<const double>& d,
       const matrix_view<const double>&  b,
       double  beta,
       const matrix_view<double>&  c
     )
{
    PROFILE_FUNCTION

    if (!(Options & FUSED_L3))
    {
        matrix<double> tempB = b;
        sktrmm(1, d, tempB);
        PROFILE_SECTION("blas::gemm")
        blas::gemm(alpha, a, tempB, beta, c);
        PROFILE_FLOPS(2*a.length(0)*a.length(1)*tempB.length(1));
        PROFILE_STOP
        return;
    }

    PROFILE_FLOPS(2*a.length(0)*a.length(1)*b.length(1));
    if (a.length(1) <= 1) return;

    bli_init();

    obj_t alpha_local, beta_local, a_local, b_local, c_local;
    bli_obj_create_1x1_with_attached_buffer(BLIS_DOUBLE, &alpha, &alpha_local);
    bli_obj_create_1x1_with_attached_buffer(BLIS_DOUBLE, &beta, &beta_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, a.length(0), a.length(1), (void*)a.data(), a.stride(0), a.stride(1), &a_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, b.length(0), b.length(1), (void*)b.data(), b.stride(0), b.stride(1), &b_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, c.length(0), c.length(1), c.data(), c.stride(0), c.stride(1), &c_local);

    // Default to using native execution.
    num_t dt = bli_obj_dt( &c_local );
    ind_t im = BLIS_NAT;

    // If each matrix operand has a complex storage datatype, try to get an
    // induced method (if one is available and enabled). NOTE: Allowing
    // precisions to vary while using 1m, which is what we do here, is unique
    // to gemm; other level-3 operations use 1m only if all storage datatypes
    // are equal (and they ignore the computation precision).
    if ( bli_obj_is_complex( &c_local ) &&
         bli_obj_is_complex( &a_local ) &&
         bli_obj_is_complex( &b_local ) )
    {
        // Find the highest priority induced method that is both enabled and
        // available for the current operation. (If an induced method is
        // available but not enabled, or simply unavailable, BLIS_NAT will
        // be returned here.)
        im = bli_ind_oper_find_avail( BLIS_GEMM, dt );
    }

    // If necessary, obtain a valid context from the gks using the induced
    // method id determined above
    auto cntx = bli_gks_query_cntx();

    auto pack_side = 1;
    auto rsc = bli_obj_row_stride( &c_local );
    auto csc = bli_obj_col_stride( &c_local );

    gemm_cntl_t cntl;
    bli_gemm_cntl_init
    (
      im,
      BLIS_GEMM,
      &alpha_local,
      &a_local,
      &b_local,
      &beta_local,
      &c_local,
      cntx,
      &cntl
    );

    if (rsc != bli_obj_row_stride( &c_local ) ||
        csc != bli_obj_col_stride( &c_local ))
        bli_negsc( &alpha_local, &alpha_local );

    func_t pack;
    bli_func_set_dt((void*)&packing, BLIS_DOUBLE, &pack);
    if (pack_side == 0)
        bli_gemm_cntl_set_packa_ukr_simple(&pack, &cntl);
    else
        bli_gemm_cntl_set_packb_ukr_simple(&pack, &cntl);

    skparams params;
    params.t = static_cast<const void*>(d.data());
    params.inct = d.stride();
    params.n = a.length(pack_side);
    if (pack_side == 0)
        bli_gemm_cntl_set_packa_params(&params, &cntl);
    else
        bli_gemm_cntl_set_packb_params(&params, &cntl);

    // Invoke the internal back-end via the thread handler.
    bli_l3_thread_decorator
    (
      &a_local,
      &b_local,
      &c_local,
      cntx,
      ( cntl_t* )&cntl,
      nullptr
    );
}

template <int Options>
void gemmt_sktri
     (
       char  uploc,
       double  alpha,
       const matrix_view<const double>&  a,
       const row_view<const double>& d,
       const matrix_view<const double>&  b,
       double  beta,
       const matrix_view<double>&  c
     )
{
    PROFILE_FUNCTION

    if (!(Options & FUSED_L3))
    {
        matrix<double> tempB = a.T();
        sktrmm(1, d, tempB);
        PROFILE_SECTION("blas::gemmt")
        blas::gemmt(uploc, alpha, a, tempB, beta, c);
        PROFILE_FLOPS(a.length(0)*a.length(1)*tempB.length(1));
        PROFILE_STOP
        return;
    }

    PROFILE_FLOPS(a.length(0)*a.length(1)*b.length(1));
    if (a.length(1) == 1) return;

    bli_init();

    uplo_t uploc_local;

    obj_t alpha_local, beta_local, a_local, b_local, c_local;
    bli_obj_create_1x1_with_attached_buffer(BLIS_DOUBLE, &alpha, &alpha_local);
    bli_obj_create_1x1_with_attached_buffer(BLIS_DOUBLE, &beta, &beta_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, a.length(0), a.length(1), (void*)a.data(), a.stride(0), a.stride(1), &a_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, b.length(0), b.length(1), (void*)b.data(), b.stride(0), b.stride(1), &b_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, c.length(0), c.length(1), c.data(), c.stride(0), c.stride(1), &c_local);

    if (uploc == 'L')
        uploc_local = BLIS_LOWER;
    else if (uploc == 'R')
        uploc_local = BLIS_UPPER;
    else
        uploc_local = BLIS_DENSE;

    // Default to using native execution.
    bli_obj_set_uplo( uploc_local, &c_local );
    bli_obj_set_struc( BLIS_TRIANGULAR, &c_local);
    num_t dt = bli_obj_dt( &c_local );
    ind_t im = BLIS_NAT;

    // If all matrix operands are complex and of the same storage datatype, try
    // to get an induced method (if one is available and enabled).
    if ( bli_obj_dt( &a_local ) == bli_obj_dt( &c_local ) &&
         bli_obj_dt( &b_local ) == bli_obj_dt( &c_local ) &&
         bli_obj_is_complex( &c_local ) )
    {
        // Find the highest priority induced method that is both enabled and
        // available for the current operation. (If an induced method is
        // available but not enabled, or simply unavailable, BLIS_NAT will
        // be returned here.)
        // im = bli_gemmtind_find_avail( dt );
        im = bli_ind_oper_find_avail( BLIS_GEMMT, dt );
    }

    // If necessary, obtain a valid context from the gks using the induced
    // method id determined above.
    auto cntx = bli_gks_query_cntx();

    auto rsc = bli_obj_row_stride( &c_local );
    auto csc = bli_obj_col_stride( &c_local );

    gemm_cntl_t cntl;
    bli_gemm_cntl_init
    (
      im,
      BLIS_GEMMT,
      &alpha_local,
      &a_local,
      &b_local,
      &beta_local,
      &c_local,
      cntx,
      &cntl
    );

    if (rsc != bli_obj_row_stride( &c_local ) ||
        csc != bli_obj_col_stride( &c_local ))
        bli_negsc( &alpha_local, &alpha_local );

    func_t pack;
    bli_func_set_dt((void*)&packing, BLIS_DOUBLE, &pack);
    bli_gemm_cntl_set_packb_ukr_simple(&pack, &cntl);

    skparams params;
    params.t = static_cast<const void*>(d.data());
    params.inct = d.stride();
    params.n = a.length(1);
    bli_gemm_cntl_set_packb_params(&params, &cntl);

    // Invoke the internal back-end via the thread handler.
    bli_l3_thread_decorator
    (
      &a_local,
      &b_local,
      &c_local,
      cntx,
      ( cntl_t* )&cntl,
      nullptr
    );
}

template <int Options>
void gemv_sktri(double alpha,         const matrix_view<const double>& A,
                                      const row_view   <const double>& T,
                                      const row_view   <const double>& x,
                        double beta,  const row_view   <      double>& y)
{
    PROFILE_FUNCTION

    if (!(Options & FUSED_L2))
    {
        row<double> tempx = x;
        sktrmv(1.0, T, tempx);
        PROFILE_SECTION("blas::gemv")
        blas::gemv(alpha, A, tempx, beta, y);
        PROFILE_FLOPS(2*A.length(0)*A.length(1));
        PROFILE_STOP
        return;
    }

    auto cntx = bli_gks_query_cntx();
    axpyf_ker_ft kfp_af = (axpyf_ker_ft)bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYF_KER, cntx );

    //constexpr int BS = 8;
    auto BS = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_AF, cntx);

    auto n = A.length(1);
    auto m = A.length(0);

    MARRAY_ASSERT( T.length() == n - 1 );
    MARRAY_ASSERT( y.length() == m );

    if ( n == 0)
        return;

    if (n == 1)
    {
        y *= beta;
        return;
    }

    const double* restrict Ap =  A.data();
    auto rsa = A.stride(0);
    auto csa = A.stride(1);
    const double* restrict xp = x.data();
          double* restrict yp = y.data();
    const double* restrict Tp = T.data();
    auto incx = x.stride();
    auto incy = y.stride();
    auto inct = T.stride();

    auto Tx = [&](int j, int n, int incx)
    {
        if (j == 0)
        {
            return - Tp[j*inct] * xp[(j+1)*incx];
        }
        else if (j == n-1)
        {
            return Tp[(j-1)*inct] * xp[(j-1)*incx];
        }
        else
        {
            return Tp[(j-1)*inct] * xp[(j-1)*incx] - Tp[j*inct] * xp[(j+1)*incx];
        }
    };

    //
    // A as the column major, rsa = 1 and incy = 1
    //
    //
    if ((rsa == 1) && (incy == 1))
    {
        //printf("COLUMN major\n");
        #pragma omp parallel if (Options & PARALLEL_L2)
        {
            int start, end;
            std::tie(start, end) = partition(m, BS, omp_get_num_threads(), omp_get_thread_num());

            if (start != end)
            {
                double Txj[16];
                //
                for (auto i = start; i < end; i++)
                {
                    yp[i] *= beta;
                }

                int begin = 0;
                auto body = [&](int& begin, int BS)
                {
                    int j0 = begin;
                    for (; j0 + BS <= n; j0+=BS)
                    {
                        for (auto j = j0; j < j0+BS; j++)
                        {
                            Txj[j-j0] = Tx(j, n, incx);
                        }

                        kfp_af(BLIS_NO_CONJUGATE,
                               BLIS_NO_CONJUGATE,
                               end-start,
                               BS,
                               &alpha,
                               &Ap[start + j0*csa], 1, csa,
                               Txj, 1,
                               &y[start], 1,
                               cntx);

                        //for (auto i = start; i < end; i++)
                        //for (auto j = j0; j < j0 + BS; j++)
                        //        yp[i] += Ap[i + j * csa] * Txj[j-j0];
                    }

                    begin = j0;

                };

                body(begin, BS);
                body(begin, 1);
            }
        }
    }
    else if ((csa == 1) && (incx == 1))
    {
        std::vector<double> Txj(n);
        for (auto j = 0;j < n;j++)
            Txj[j] = Tx(j, n, incx);

        #pragma omp parallel if (Options & PARALLEL_L2)
        {
            int start, end;
            std::tie(start, end) = partition(m, BS, omp_get_num_threads(), omp_get_thread_num());

            double yi[16];

            auto body = [&](int& start, int BS)
            {
                auto i0 = start;
                for (;i0+BS <= end;i0 += BS)
                {
                    memset(&yi, 0, BS*sizeof(double));
                    for (auto i = i0;i < i0+BS;i++)
                    {
                       //yi[i-i0] += Ap[i*rsa + 0] * Tx(0, n, 1);
                       yi[i-i0] += Ap[i*rsa + 0] * Txj[0];
                    }
                    for (auto j = 1;j < n-1;j++)
                    for (auto i = i0;i < i0+BS;i++)
                    {
                        //yi[i-i0] += Ap[i*rsa + j] * Tx(j, n, 1);
                        yi[i-i0] += Ap[i*rsa + j] * Txj[j];
                    }
                    for (auto i = i0;i < i0+BS;i++)
                    {
                        //yi[i-i0] += Ap[i*rsa + n-1] * Tx(n-1, n, 1);
                        yi[i-i0] += Ap[i*rsa + n-1] * Txj[n-1];
                    }

                    if (beta != 0.0)
                    {
                        for (auto i = i0;i < i0+BS;i++)
                            yp[i*incy] = alpha*yi[i-i0] + beta*yp[i*incy];
                    }
                    else
                    {
                        for (auto i = i0;i < i0+BS;i++)
                            yp[i*incy] = alpha*yi[i-i0];
                    }
                }

                start = i0;
            };

            body(start, BS);
            body(start, 1);
        }
    }
    else
    {
        #pragma omp parallel for if (Options & PARALLEL_L2)
        for (auto i : range(m))
        {
            auto temp = 0.0;
            for (auto j : range(n))
            {
                if (j == 0)
                {
                    temp += (Ap[i*rsa+1*csa] * T[0]) * xp[j*incx];
                }
                else if (j == n-1)
                {
                    temp += (-Ap[i*rsa+(n-2)*csa] * T[n-2]) * xp[j*incx];
                }
                else
                {
                    temp+= (- Ap[i*rsa+(j-1)*csa] * T[j-1] + Ap[i*rsa+(j+1)*csa] * T[j]) * xp[j*incx] ;
                }
            }
            yp[i*incy] = alpha * temp + beta * yp[i*incy];
        }
    }

    PROFILE_FLOPS(2*A.length(0)*A.length(1));
}

template <int Options>
void skr2(char uplo,
          double alpha, const row_view<const double>& a,
                        const row_view<const double>& b,
          double beta,  const matrix_view<   double>& C)
{
    PROFILE_FUNCTION

    if (!(Options & FUSED_L2))
    {
        PROFILE_SECTION("blas::skr2")
        blas::skr2(uplo, alpha, a, b, beta, C);
        PROFILE_FLOPS(2*C.length(0)*C.length(1));
        PROFILE_STOP
        return;
    }

    auto cntx = bli_gks_query_cntx();

    axpy2v_ker_ft kfp_2v = (axpy2v_ker_ft)bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPY2V_KER, cntx );

    constexpr int BS = 2; // DO NOT CHANGE!

    auto m = C.length(0);
    auto n = C.length(1);

    MARRAY_ASSERT(m == n);

    PROFILE_FLOPS(2*m*n);

    const double* restrict ap = a.data();
    const double* restrict bp = b.data();
          double* restrict Cp = C.data();

    auto inca = a.stride();
    auto incb = a.stride();
    auto rsc = C.stride(0);
    auto csc = C.stride(1);

    if (n == 0)
        return ;
    if (n == 1)
    {
        Cp[0] *= beta;
        return ;
    }

    if (csc == 1)
    {
        std::swap(ap, bp);
        std::swap(inca, incb);
        std::swap(rsc, csc);
        uplo = uplo == 'L' ? 'U' : 'L';
    }

    assert(rsc == 1);

    if (uplo == 'L')
    {
        #pragma omp parallel if (Options & PARALLEL_L2)
        {
            auto tid = omp_get_thread_num();
            auto nt = omp_get_num_threads();

            auto work = (n*n)/4;
            auto start = (work*tid)/nt;
            auto end = (work*(tid+1))/nt;

            if (start != end)
            {
                auto jstart = 0;
                for (auto pos = 0; jstart <= n-BS; jstart += BS)
                {
                    if (start < pos+n-jstart-1)
                    {
                        start -= pos;
                        break;
                    }
                    pos += n-jstart-1;
                }

                auto jend = 0;
                for (auto pos = 0; jend <= n-BS; jend += BS)
                {
                    if (end <= pos+n-jend-1)
                    {
                        end -= pos;
                        break;
                    }
                    pos += n-jend-1;
                }

                auto body = [&](auto j0, auto start, auto end, auto inca, auto incb)
                {
                    // 2x2 diagonal
                    if (start == 0)
                    {
                        auto i = j0+1;
                        auto j = j0;
                        Cp[i+j*csc] = alpha * (ap[i*inca] * bp[j*incb] - ap[j*inca] * bp[i*incb]) + beta * Cp[i+j*csc];
                        start++;
                    }

                    double alpha_aj[BS];
                    double alpha_bj[BS];

                    for (auto j = j0; j < j0+BS ; j++)
                    {
                        alpha_aj[j-j0] = -alpha * ap[j*inca];
                        alpha_bj[j-j0] =  alpha * bp[j*incb];
                    }

                    for (auto pos = start; pos < end; pos++)
                    {
                        auto i = j0+BS+pos-1;
                        for (auto j = j0; j < j0+BS ; j++)
                            Cp[i+j*csc] = ap[i*inca] * alpha_bj[j-j0] + alpha_aj[j-j0] * bp[i*incb] + beta * Cp[i+j*csc];
                    }
                };

                if (inca == 1 && incb == 1)
                {
                    if (jstart == jend)
                    {
                        body(jstart, start, end, 1, 1);
                    }
                    else
                    {
                        body(jstart, start, n-jstart-1, 1, 1);
                        for (auto j0 = jstart+BS; j0 < jend; j0 += BS)
                            body(j0, 0, n-j0-1, 1, 1);
                        body(jend, 0, end, 1, 1);
                    }
                }
                else
                {
                    if (jstart == jend)
                    {
                        body(jstart, start, end, inca, incb);
                    }
                    else
                    {
                        body(jstart, start, n-jstart-1, inca, incb);
                        for (auto j0 = jstart+BS; j0 < jend; j0 += BS)
                            body(j0, 0, n-j0-1, inca, incb);
                        body(jend, 0, end, inca, incb);
                    }
                }
            }
        }
    }
    else
    {
        #pragma omp parallel if (Options & PARALLEL_L2)
        {
            auto tid = omp_get_thread_num();
            auto nt = omp_get_num_threads();

            auto work = (n*n)/4;
            auto start = (work*tid)/nt;
            auto end = (work*(tid+1))/nt;

            if (start != end)
            {
                auto jstart = n&1;
                for (auto pos = 0; jstart <= n-BS; jstart += BS)
                {
                    if (start < pos+jstart+1)
                    {
                        start -= pos;
                        break;
                    }
                    pos += jstart+1;
                }

                auto jend = n&1;
                for (auto pos = 0; jend <= n-BS; jend += BS)
                {
                    if (end <= pos+jend+1)
                    {
                        end -= pos;
                        break;
                    }
                    pos += jend+1;
                }

                auto body = [&](auto j0, auto start, auto end, auto inca, auto incb)
                {
                    // 2x2 diagonal
                    if (end == j0+1)
                    {
                        auto i = j0;
                        auto j = j0+1;
                        Cp[i+j*csc] = alpha * (ap[i*inca] * bp[j*incb] - ap[j*inca] * bp[i*incb]) + beta * Cp[i+j*csc];
                        end--;
                    }

                    for (auto pos = start; pos < end; pos++)
                    {
                        auto i = pos;
                        auto alpha_ai = alpha * ap[i*inca];
                        auto alpha_bi = alpha * bp[i*incb];
                        for (auto j = j0; j < j0+BS ; j++)
                            Cp[i+j*csc] = alpha_ai * bp[j*incb] - ap[j*inca] * alpha_bi + beta * Cp[i+j*csc];
                    }
                };

                if (inca == 1 && incb == 1)
                {
                    if (jstart == jend)
                    {
                        body(jstart, start, end, 1, 1);
                    }
                    else
                    {
                        body(jstart, start, jstart+1, 1, 1);
                        for (auto j0 = jstart+BS; j0 < jend; j0 += BS)
                            body(j0, 0, j0+1, 1, 1);
                        body(jend, 0, end, 1, 1);
                    }
                }
                else
                {
                    if (jstart == jend)
                    {
                        body(jstart, start, end, inca, incb);
                    }
                    else
                    {
                        body(jstart, start, jstart+1, inca, incb);
                        for (auto j0 = jstart+BS; j0 < jend; j0 += BS)
                            body(j0, 0, j0+1, inca, incb);
                        body(jend, 0, end, inca, incb);
                    }
                }
            }
        }
    }
}

template <int Options>
void ger2(double alpha, const row_view<const double> a,
                        const row_view<const double> b,
          double beta,  const row_view<const double> c,
                        const row_view<const double> d,
          double gamma, const matrix_view<   double> E)
{
    PROFILE_FUNCTION

    if (!(Options & FUSED_L2))
    {
        PROFILE_SECTION("blas::ger")
        blas::ger(alpha, a, b, gamma, E);
        blas::ger(beta, c, d, 1.0, E);
        PROFILE_FLOPS(4*E.length(0)*E.length(1));
        PROFILE_STOP
        return;
    }

    constexpr int BS = 5;

    auto m = E.length(0);
    auto n = E.length(1);

    auto la = a.length();
    auto lb = b.length();
    auto lc = c.length();
    auto ld = d.length();

    MARRAY_ASSERT(la == m);
    MARRAY_ASSERT(lb == n);
    MARRAY_ASSERT(lc == m);
    MARRAY_ASSERT(ld == n);

    const double* restrict ap = a.data();
    const double* restrict bp = b.data();
    const double* restrict cp = c.data();
    const double* restrict dp = d.data();
          double* restrict Ep = E.data();

    int rse = E.stride(0);
    int cse = E.stride(1);
    int inca = a.stride();
    int incb = b.stride();
    int incc = c.stride();
    int incd = d.stride();

    if (rse == 1)
    {
        if ( inca == 1 && incb == 1 && incc == 1 && incd == 1 ) // a, b, c and d are unit stride.
        {
            #pragma omp parallel if (Options & PARALLEL_L2)
            {
                auto tid = omp_get_thread_num();
                auto nt = omp_get_num_threads();

                int start, end;
                std::tie(start, end) = partition(n, BS, nt, tid);
                auto body = [&](int& start, int BS)
                {
                    auto j0 = start;
                    for (; j0 + BS <= end; j0+=BS)
                    {
                        for (auto i = 0; i < m; i++)
                        {
                            auto tmpa = ap[i];
                            auto tmpc = cp[i];
                            for( auto j = j0; j < j0 + BS; j++)
                            {
                                Ep[i+j*cse] = alpha * tmpa * bp[j] + beta * tmpc * dp[j] + gamma * Ep[i+j*cse];
                            }
                        }
                    }
                    start = j0;
                };
                body(start, BS);
                body(start, 1);
            }
        }
        else
        {
            #pragma omp parallel if (Options & PARALLEL_L2)
            {
                auto tid = omp_get_thread_num();
                auto nt = omp_get_num_threads();

                int start, end;
                std::tie(start, end) = partition(n, BS, nt, tid);
                auto body = [&](int& start, int BS)
                {
                    auto j0 = start;
                    for (; j0 + BS <= end; j0+=BS)
                    {
                        for (auto i = 0; i < m; i++)
                        {
                            auto tmpa = ap[i*inca];
                            auto tmpc = cp[i*incc];
                            for( auto j = j0; j < j0 + BS; j++)
                            {
                                Ep[i+j*cse] = alpha * tmpa * bp[j*incb] + beta * tmpc * dp[j*incd] + gamma * Ep[i+j*cse];
                            }
                        }
                    }
                    start = j0;
                };
                body(start, BS);
                body(start, 1);
            }
        }
    }
    else if (cse == 1)
    {
        if ( inca == 1 && incb == 1 && incc == 1 && incd == 1 ) // a, b , c and d are unit stride
        {
            #pragma omp parallel if (Options & PARALLEL_L2)
            {
                auto tid = omp_get_thread_num();
                auto nt = omp_get_num_threads();

                int start, end;
                std::tie(start, end) = partition(m, BS, nt, tid);
                auto body = [&](int& start, int BS)
                {
                    auto i0 = start;
                    for(; i0 + BS <= end; i0+=BS)
                    {
                        for(auto j = 0; j < n; j++)
                        {
                            auto tmpb = bp[j];
                            auto tmpd = dp[j];
                            for(auto i = i0; i < i0 + BS; i++)
                            {
                                Ep[i*rse+j] = alpha * ap[i] * tmpb + beta * cp[i] * tmpd + gamma * Ep[i*rse+j];
                            }
                        }
                    }
                    start = i0;
                };

                body(start, BS);
                body(start, 1);
            }
        }
        else
        {
            #pragma omp parallel if (Options & PARALLEL_L2)
            {
                auto tid = omp_get_thread_num();
                auto nt = omp_get_num_threads();

                int start, end;
                std::tie(start, end) = partition(m, BS, nt, tid);
                auto body = [&](int& start, int BS)
                {
                    auto i0 = start;
                    for(; i0 + BS <= end; i0+=BS)
                    {
                        for(auto j = 0; j < n; j++)
                        {
                            auto tmpb = bp[j*incb];
                            auto tmpd = dp[j*incd];
                            for(auto i = i0; i < i0 + BS; i++)
                            {
                                Ep[i*rse+j] = alpha * ap[i*inca] * tmpb + beta * cp[i*incc] * tmpd + gamma * Ep[i*rse+j];
                            }
                        }
                    }
                    start = i0;
                };

                body(start, BS);
                body(start, 1);
            }
        }
    }
    else
    {
        printf("General function hasn't been impletemented!\n");
        exit(1);
    }

    PROFILE_FLOPS(4*m*n);
}

#define INSTANTIATIONS(N) \
\
template void gemm_sktri<STEP_##N> \
     ( \
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d, \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c \
     ); \
\
template void gemmt_sktri<STEP_##N> \
     ( \
       char  uploc, \
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d, \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c \
     ); \
\
template void gemv_sktri<STEP_##N>(double alpha,         const matrix_view<const double>& A, \
                                      const row_view   <const double>& T, \
                                      const row_view   <const double>& x, \
                        double beta,  const row_view   <      double>& y); \
\
template void skr2<STEP_##N>(char uplo, \
          double alpha, const row_view<const double>& a, \
                        const row_view<const double>& b, \
          double beta,  const matrix_view<   double>& C); \
\
template void ger2<STEP_##N>(double alpha, const row_view<const double> a, \
                        const row_view<const double> b, \
          double beta,  const row_view<const double> c, \
                        const row_view<const double> d, \
          double gamma, const matrix_view<   double> E);

INSTANTIATIONS(0)
INSTANTIATIONS(1)
INSTANTIATIONS(2)
INSTANTIATIONS(3)
INSTANTIATIONS(4)
INSTANTIATIONS(5)
