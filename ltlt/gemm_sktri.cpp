#include "ltlt.hpp"
#include "packing.hpp"


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
    PROFILE_FUNCTION
    PROFILE_FLOPS(2*a.length(0)*a.length(1)*b.length(1));
    if (a.length(0) == 1)
        return;

	bli_init();

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

    func_t pack;
    bli_func_set_dt((void*)&packing, BLIS_DOUBLE, &pack);
    bli_gemm_cntl_set_packb_ukr_simple(&pack, &cntl);
    skparams params;
    //params.t = static_cast<const void*>(d.data());   
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
    PROFILE_FUNCTION
    PROFILE_FLOPS(a.length(0)*a.length(1)*b.length(1));
    if (a.length(0) == 1)
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


void gemv_sktri(double alpha,         const matrix_view<const double>& A,
                                      const row_view   <const double>& T,
                                      const row_view   <const double>& x,
                        double beta,  const row_view   <      double>& y)
{
    PROFILE_FUNCTION
    PROFILE_FLOPS(2*A.length(0)*A.length(1));
    auto n = A.length(1);
    auto m = A.length(0);

    MARRAY_ASSERT(T.length() == n - 1);

    if ( n == 0)
        return;

    if (n == 1)
    {
        y[0] *= beta;
        return;
    }
    // to get the normal base (0), we create a new matrix A_temp with base as 0,
    // otherwise we will use A element wrong when we call it by index. 
    // Question, I don't know how native gemv impletation solves this issue.
    // matrix<double> A_temp = A; 
    auto restrict Ap =  A.data();
    auto rsa = A.stride(0);
    auto csa = A.stride(1);
    auto restrict xp = x.data();
    auto incx = x.stride(); 
    auto incy = y.stride(); 

    // printf("rsa, csa, xp, incx, incy = %d, %d, %d, %d, %d\n", rsa, csa, xp, incx, incy);

    #pragma omp paralell for private(i) shared(Ap, xp, T, y) reduction(+:temp)
    for (auto i : range(m))
    {
        auto temp = 0.0;
        for (auto k : range(n))
        {
            if (k == 0)
            {
                temp += (Ap[i*rsa+1*csa] * T[0]) * xp[k*incx];
            }
            else if (k == n-1)
            {
                temp += (-Ap[i*rsa+(n-2)*csa] * T[n-2]) * xp[k*incx];
            }
            else
            {
                temp+= (- Ap[i*rsa+(k-1)*csa] * T[k-1] + Ap[i*rsa+(k+1)*csa] * T[k]) * xp[k*incx] ;
            }
        }
        y[i*incy] = alpha * temp + beta * y[i*incy]; 
    }


    // row<double> tempx = x; 

    // MARRAY_ASSERT(T.length() == n - 1);

    // if ( n == 0)
    //     return;

    // if (n == 1)
    // {
    //     y[0] *= beta;
    //     return;
    // }
    // 
    // auto ximinus1 = tempx[0];
    // tempx[0] = (-T[0] * tempx[1]);
    // for (auto i : range(1,n-1))
    // {
    //     auto xi = (T[i-1] * ximinus1 - T[i] * tempx[i+1]);
    //     ximinus1 = tempx[i];
    //     tempx[i] = xi;
    // }

    // tempx[n-1] =  (T[n-2] * ximinus1);

    // for (auto i : range(m))
    // {
    //     double temp  = 0.0;
    //     for (auto j : range(n))
    //     {
    //         temp += alpha * Ap[i*rsa+j*csa] * tempx[j*incx];
    //         // printf("%f = %f * %f * %f\n", temp, alpha, A[i][j], tempx[j]);
    //     }
    //     y[i] = temp + beta * y[i];

    // }

}
