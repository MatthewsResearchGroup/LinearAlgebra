[1mdiff --git a/ltlt/ltlt.hpp b/ltlt/ltlt.hpp[m
[1mindex 6b64a38..57931e1 100644[m
[1m--- a/ltlt/ltlt.hpp[m
[1m+++ b/ltlt/ltlt.hpp[m
[36m@@ -11,17 +11,20 @@[m
 //must come first[m
 #define MARRAY_USE_BLIS[m
 #define BLIS_ENABLE_STD_COMPLEX[m
[32m+[m[32m#define BLIS_DISABLE_BLAS_DEFS[m
 #include "blis.h"[m
 [m
[32m+[m[32mtemplate <typename T>[m
[32m+[m[32mbool foo() { static_assert(std::is_same_v<T,std::complex<double>>, ""); return true; }[m
[32m+[m
[32m+[m[32mstatic auto check = foo<dcomplex>();[m
[32m+[m
 #include "marray_view.hpp"[m
 #include "expression.hpp"[m
 #include "blas.h"[m
 #include "flame.hpp"[m
 #include "bli_clock.h"[m
 [m
[31m-template <typename T>                                                                                                                                    [m
[31m-bool foo() { static_assert(std::is_same_v<T,int>, ""); return true; }[m
[31m-[m
 using namespace MArray;[m
 using MArray::slice::all;[m
 using std::tie;[m
[36m@@ -102,7 +105,6 @@[m [minline void skew_tridiag_gemm(double alpha, const matrix_view<const double>& A,[m
     gemm(alpha, A, tempB, beta, C);[m
 }[m
 [m
[31m-[m
 /**[m
  * Perform the upper or lower triangular portion of the skew tridiag matrix multiplication \f$ C = \alpha ABA^T + \beta C \f$.[m
  *[m
[36m@@ -119,13 +121,13 @@[m [minline void skew_tridiag_gemm(double alpha, const matrix_view<const double>& A,[m
  *[m
  * @param C     A `m`x`m` skew matrix or matrix view.[m
  */[m
[31m-[m
 inline void skew_tridiag_rankk(char uplo,[m
                                double alpha, const matrix_view<const double>& A,[m
                                              const row_view   <const double>& T,[m
                                double beta,  const matrix_view<      double>& C)[m
 {[m
     // B <- T B[m
[32m+[m
     // copy of B[m
     matrix<double> tempB = A.T();[m
     sktrmm(1, T, tempB);[m
[36m@@ -137,27 +139,14 @@[m [minline void skew_tridiag_rankk(char uplo,[m
 } //namespace blas[m
 } //namespace MArray[m
 [m
[31m-template <typename T> inline range_t<T> not_first(const range_t<T>& x)[m
[32m+[m[32mtemplate <typename T>[m[41m [m
[32m+[m[32minline range_t<T> not_first(const range_t<T>& x)[m
 {[m
     return range(x.from()+1, x.to());[m
 }[m
 [m
[31m-// template <typename T> range_t<T> R3_trunc(const range_t<T>& R0, const range_t<T>& R3, len_type k)[m
[31m-// {[m
[31m-//     if ( R0.from() + k < R3.from())[m
[31m-//     {[m
[31m-//         return range(R3.from(), -1);[m
[31m-//     }[m
[31m-//     else[m
[31m-//     {[m
[31m-//         return range(R3.from(), R0.from() + k);[m
[31m-//     }[m
[31m-// }[m
[31m-[m
[31m-[m
 [m
[31m-[m
[31m-template <typename T> [m
[32m+[m[32mtemplate <typename T>[m
 inline auto R3_trunc(const range_t<T>& R0, const range_t<T>& R3, len_type k)[m
 {[m
     if ( R0.from() + k < R3.from())[m
[36m@@ -170,6 +159,7 @@[m [minline auto R3_trunc(const range_t<T>& R0, const range_t<T>& R3, len_type k)[m
     }[m
 }[m
 [m
[32m+[m
 inline matrix<double> make_L(const matrix_view<const double>& X)[m
 {[m
     auto n = X.length(0);[m
