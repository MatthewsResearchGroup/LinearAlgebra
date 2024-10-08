#include "ltlt.hpp"
#include "packing.hpp"

template <int Options>
void gemm_sktri
     (
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d,  \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c \
     )
{
    if (!(Options & FUSED_L3))
    {
        blas::skew_tridiag_gemm(alpha, a, d, b, beta, c);
        return;
    }

    //std::cout << "matrix a" << std::endl;
    //matrixprint(a);
    //std::cout << "matrix b" << std::endl;
    //matrixprint(b);
    //std::cout << "matrix c" << std::endl;
    //matrixprint(c);
    PROFILE_FUNCTION
    PROFILE_FLOPS(2*a.length(0)*a.length(1)*b.length(1));
    if (a.length(1) == 1)
        return;

	bli_init();

    //printf("a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1) = %d, %d, %d, %d, %d, %d\n", a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1));
    obj_t alpha_local, beta_local, a_local, b_local, c_local;
    bli_obj_create_1x1_with_attached_buffer(BLIS_DOUBLE, &alpha, &alpha_local);
    bli_obj_create_1x1_with_attached_buffer(BLIS_DOUBLE, &beta, &beta_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, a.length(0), a.length(1), (void*)a.data(), a.stride(0), a.stride(1), &a_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, b.length(0), b.length(1), (void*)b.data(), b.stride(0), b.stride(1), &b_local);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, c.length(0), c.length(1), c.data(), c.stride(0), c.stride(1), &c_local);

	// // Check the operands.
	// if ( bli_error_checking_is_enabled() )
	// 	bli_gemm_check( &alpha_local, &a_local, &b_local, &beta_local, &c_local, cntx );

	// // Check for zero dimensions, alpha == 0, or other conditions which
	// // mean that we don't actually have to perform a full l3 operation.
	// if ( bli_l3_return_early_if_trivial( &alpha_local, &a_local, &b_local, &beta_local, &c_local ) == BLIS_SUCCESS )
	// 	return;

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

    if (c.stride(pack_side) == 1) // C with the ROW-MAJOR
        bli_negsc( &alpha_local, &alpha_local );
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

    //bli_negsc( &alpha_local, &alpha_local );
    //std::cout << "transpose"  << transpose << std::endl;
    //bli_printm("alpha_local", &alpha_local, "%5.2f", "");
    //if (!transpose)
    //    bli_negsc( &alpha_local, &alpha_local);
    //bli_printm("alpha_local", &alpha_local, "%5.2f", "");

    func_t pack;
    bli_func_set_dt((void*)&packing, BLIS_DOUBLE, &pack);
    if (pack_side == 0)
        bli_gemm_cntl_set_packa_ukr_simple(&pack, &cntl); // packa
    else
        bli_gemm_cntl_set_packb_ukr_simple(&pack, &cntl); // packa
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
       char  uploc, \
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d,  \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c \
     )
{
    if (!(Options & FUSED_L3))
    {
        blas::skew_tridiag_rankk(uploc, alpha, a, d, beta, c);
        return;
    }

    PROFILE_FUNCTION
    PROFILE_FLOPS(a.length(0)*a.length(1)*b.length(1));
    if (a.length(1) == 1)
        return;
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
	// // Check the operands.
	// if ( bli_error_checking_is_enabled() )
	// 	bli_gemmt_check( alpha, a, b, beta, c, cntx );

	// // Check for zero dimensions, alpha == 0, or other conditions which
	// // mean that we don't actually have to perform a full l3 operation.
	// if ( bli_l3_return_early_if_trivial( alpha, a, b, beta, c ) == BLIS_SUCCESS )
	// 	return;

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

    if (c.stride(1) == 1) // C with the ROW-MAJOR
        bli_negsc( &alpha_local, &alpha_local );
	// Alias A, B, and C in case we need to apply transformations.
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
    if (!(Options & FUSED_L2))
    {
        blas::skewtrigemv(alpha, A, T, x, beta, y);
        return;
    }

    constexpr int BS = 5;

    PROFILE_FUNCTION
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

    // to get the normal base (0), we create a new matrix A_temp with base as 0,
    // otherwise we will use A element wrong when we call it by index.
    // Question, I don't know how native gemv impletation solves this issue.
    auto restrict Ap =  A.data();
    auto rsa = A.stride(0);
    auto csa = A.stride(1);
    auto restrict xp = x.data();
    auto restrict yp = y.data();
    auto restrict Tp = T.data();
    auto incx = x.stride();
    auto incy = y.stride();
    auto inct = T.stride();

    // printf("rsa, csa, incx, incy, inct = %d, %d, %d, %d, %d\n", rsa, csa, incx, incy, inct);

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
            double Txj[BS];
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

                    for (auto i = start; i < end; i++)
                    for (auto j = j0; j < j0 + BS; j++)
                            yp[i] += alpha * Ap[i + j * csa] * Txj[j-j0];
                }

                begin = j0;

            };

            body(begin, BS);
            body(begin, 1);
        }
    }
    else if ((csa == 1) && (incx == 1))
    {
        //printf("row major\n");
        //std::vector<double> Txj(n);
        //for (auto j = 0;j < n;j++)
        //    Txj[j] = Tx(j, n, incx);

        #pragma omp parallel if (Options & PARALLEL_L2)
        {
            int start, end;
            std::tie(start, end) = partition(m, BS, omp_get_num_threads(), omp_get_thread_num());
            //printf("start, end, OMM_NUM_THREAD, THREAD_ID = %d, %d, %d, %d\n", start, end, omp_get_num_threads(), omp_get_thread_num());

            double yi[BS];

            auto body = [&](int& start, int BS)
            {
                auto i0 = start;
                for (;i0+BS <= end;i0 += BS)
                {
                    memset(&yi, 0, BS*sizeof(double));
                    for (auto i = i0;i < i0+BS;i++)
                    {
                       yi[i-i0] += Ap[i*rsa + 0] * Tx(0, n, 1);
                       //yi[i-i0] += Ap[i*rsa + 0] * Txj[0];
                    }
                    for (auto j = 1;j < n-1;j++)
                    for (auto i = i0;i < i0+BS;i++)
                    {
                        yi[i-i0] += Ap[i*rsa + j] * Tx(j, n, 1);
                        //yi[i-i0] += Ap[i*rsa + j] * Txj[j];
                    }
                    for (auto i = i0;i < i0+BS;i++)
                    {
                        yi[i-i0] += Ap[i*rsa + n-1] * Tx(n-1, n, 1);
                        //yi[i-i0] += Ap[i*rsa + n-1] * Txj[n-1];
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
        printf("gernel format\n");
        #pragma omp parallel for if (Options & PARALLEL_L2)
        for (auto i : range(m))
        {
            // auto id = omp_get_thread_num();
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
void skr2(char uplo, \
          double alpha, const row_view<const double>& a,
                        const row_view<const double>& b,
          double beta,  const matrix_view<   double>& C)
{
    if (!(Options & FUSED_L2))
    {
        blas::skr2(uplo, alpha, a, b, beta, C);
        return;
    }

    constexpr int BS = 2; // DO NOT CHANGE!

    auto m = C.length(0);
    auto n = C.length(1);

    MARRAY_ASSERT(m == n);

    PROFILE_FUNCTION
    PROFILE_FLOPS(2*m*n);

    auto restrict ap = a.data();
    auto restrict bp = b.data();
    auto restrict Cp = C.data();

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
            auto start0 = start;
            auto end0 = end;

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

                //printf("[%d/%d]: %d:%d of %d, %d,%d -> %d,%d\n", tid, nt, start0, end0, work, jstart, start, jend, end);

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

                    for (auto pos = start; pos < end; pos++)
                    {
                        auto i = j0+BS+pos-1;
                        for (auto j = j0; j < j0+BS ; j++)
                            Cp[i+j*csc] = alpha * (ap[i*inca] * bp[j*incb] - ap[j*inca] * bp[i*incb]) + beta * Cp[i+j*csc];
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
            auto start0 = start;
            auto end0 = end;

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

                //printf("[%d/%d]: %d:%d of %d, %d,%d -> %d,%d\n", tid, nt, start0, end0, work, jstart, start, jend, end);

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
                        for (auto j = j0; j < j0+BS ; j++)
                            Cp[i+j*csc] = alpha * (ap[i*inca] * bp[j*incb] - ap[j*inca] * bp[i*incb]) + beta * Cp[i+j*csc];
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
    if (!(Options & FUSED_L2))
    {
        blas::ger(alpha, a, b, gamma, E);
        blas::ger(beta, c, d, 1.0, E);
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

    PROFILE_FUNCTION
    auto restrict ap = a.data();
    auto restrict bp = b.data();
    auto restrict cp = c.data();
    auto restrict dp = d.data();
    auto restrict Ep = E.data();

    int rse = E.stride(0);
    int cse = E.stride(1);
    int inca = a.stride();
    int incb = b.stride();
    int incc = c.stride();
    int incd = d.stride();

    //if (rse == 1 && inca == 1 && incb == 1 && incc == 1 && incd == 1) // COLUMN MAJOR
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

                // auto j0 = tid*BS;
                // for (; j0 <= n -BS; j0+=nt*BS)
                // {
                //     for (auto i = 0; i < m; i++)
                //     for (auto j = j0; j < j0+BS; j++)
                //         Ep[i+j*cse] = alpha * ap[i] * bp[j] + beta * cp[i] * dp[j] + gamma * Ep[i+j*cse];

                // }
                // for (auto j = j0; j < n; j++)
                // {
                //     for (auto i = 0; i < m; i++)
                //         Ep[i+j*cse] = alpha * ap[i] * bp[j] + beta * cp[i] * dp[j] + gamma * Ep[i+j*cse];
                // }

            }
        }
        else
        {
            //printf("nonuint stride GER2 COLUMN MAJOR\n");
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
    //else if (cse == 1 && inca == 1 && incb == 1 && incc == 1 && incd == 1) // ROW MAJOR
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
            //printf("nonuint stride GER2 ROW MAJOR\n");
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
    }

    PROFILE_FLOPS(4*m*n);
}

#define INSTANTIATIONS(N) \
\
template void gemm_sktri<STEP_##N> \
     ( \
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d,  \
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
       const row_view<const double>& d,  \
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
