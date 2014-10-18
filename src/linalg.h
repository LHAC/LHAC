//
//  linalg.h
//  pepper
//
//  Created by Xiaocheng Tang on 3/21/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef pepper_linalg_h
#define pepper_linalg_h

#define MAX_SY_PAIRS 100

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
	AtlasConj=114};

//#ifdef __APPLE__
#ifdef USE_CBLAS
//#include <Accelerate/Accelerate.h>
#define INTT int
extern "C" {
    double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
    void cblas_dgemv(const enum CBLAS_ORDER Order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *X, const int incX, const double beta,
                     double *Y, const int incY);
    void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const double alpha, const double *A,
                     const int lda, const double *B, const int ldb,
                     const double beta, double *C, const int ldc);
}
#else
#include <stddef.h>
//#include "lapack.h"
//#include "blas.h"
#if defined(_WIN32) || defined(__hpux)
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif
#define INTT ptrdiff_t
#define dgemm FORTRAN_WRAPPER(dgemm)
#define dgemv FORTRAN_WRAPPER(dgemv)
#define ddot FORTRAN_WRAPPER(ddot)
extern "C" {
    double ddot(INTT *n, double *dx, INTT *incx, double *dy, INTT *incy);
    void dgemm(char   *transa, char   *transb, INTT *m, INTT *n, INTT *k,
               double *alpha, double *a, INTT *lda, double *b, INTT *ldb,
               double *beta, double *c, INTT *ldc);
    void dgemv(char   *trans, INTT *m, INTT *n, double *alpha, double *a,
               INTT *lda, double *x, INTT *incx, double *beta, double *y,
               INTT *incy);
}
static char CBLAS_TRANSPOSE_CHAR[] = {'N', 'T', 'C'};
inline char *blas_transpose(CBLAS_TRANSPOSE TransA)
{
	switch(TransA)
	{
		case CblasNoTrans:      return &CBLAS_TRANSPOSE_CHAR[0];
		case CblasTrans:        return &CBLAS_TRANSPOSE_CHAR[1];
		case CblasConjTrans:	return &CBLAS_TRANSPOSE_CHAR[2];
        case AtlasConj:         return NULL;
	}
	return NULL;
}

#endif

extern "C" {
    int dpotrf_(char *uplo, INTT *n, double *a, INTT *lda, INTT *info);
    int dpotri_(char *uplo, INTT *n, double *a, INTT *lda, INTT *info);
    int dgetrf_(INTT *m, INTT *n, double *a, INTT *lda, INTT *ipiv, INTT *info);
    int dgetri_(INTT *n, double *a, INTT *lda, INTT *ipiv, double *work, INTT *lwork, INTT *info);
}


inline void lcdpotrf_(double* w, const unsigned long n, int* _info) {
    INTT info = 0;
    INTT p0 = (INTT) n;
    dpotrf_((char*) "U", &p0, w, &p0, &info);
    
    *_info = (int) info;
}

inline void lcdpotri_(double* w, const unsigned long n, int* _info) {
    INTT info = 0;
    INTT p0 = (INTT) n;
    dpotri_((char*) "U", &p0, w, &p0, &info);
    
    *_info = (int) info;
}


/* w square matrix */
inline int inverse(double*w, const int _n) {
    INTT info = 0;
    INTT n = (INTT) _n;
    static INTT ipiv[MAX_SY_PAIRS+1];
    static INTT lwork = MAX_SY_PAIRS*MAX_SY_PAIRS;
    static double work[MAX_SY_PAIRS*MAX_SY_PAIRS];
    dgetrf_(&n, &n, w, &n, ipiv, &info);
    dgetri_(&n, w, &n, ipiv, work, &lwork, &info);
    
    return (int) info;
}

inline double lcddot(const int n, double* dx, const int incx,
                     double* dy, const int incy) {
#ifdef USE_CBLAS
    return cblas_ddot(n, dx, incx, dy, incy);
#else
    INTT _n = (INTT) n;
    INTT _incx = (INTT) incx;
    INTT _incy = (INTT) incy;
    return ddot(&_n, dx, &_incx, dy, &_incy);
#endif
}


inline void lcdgemv(const enum CBLAS_ORDER Order,
                    const enum CBLAS_TRANSPOSE TransA,
                    double* A, double* b, double* c,
                    const int m, const int n, const int lda)
{
#ifdef USE_CBLAS
    cblas_dgemv(Order, TransA, m, n, 1.0, A, lda, b, 1, 0.0, c, 1);
#else
    static double one = 1.0;
    static double zero = 0.0;
    INTT one_int = 1;
    INTT blas_m = (INTT) m;
    INTT blas_n = (INTT) n;
    INTT blas_lda = (INTT) lda;
    dgemv(blas_transpose(TransA), &blas_m, &blas_n, &one, A, &blas_lda, b, &one_int, &zero, c, &one_int);
#endif
}

inline void lcgdgemm(const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB,
                     const int M, const int N,
                     const int K, const double alpha, double *A,
                     const int lda, double *B, const int ldb,
                     const double beta, double *C, const int ldc) {
#ifdef USE_CBLAS
    cblas_dgemm(CblasColMajor, TransA, TransB, M, N, K,
                alpha, A, lda, B, ldb, beta, C, ldc);
#else
    INTT _M = (INTT) M;
    INTT _N = (INTT) N;
    INTT _K = (INTT) K;
    INTT _lda = (INTT) lda;
    INTT _ldb = (INTT) ldb;
    INTT _ldc = (INTT) ldc;
    double _alpha = alpha;
    double _beta = beta;
    dgemm(blas_transpose(TransA), blas_transpose(TransB),
          &_M, &_N,&_K, &_alpha, A, &_lda, B, &_ldb, &_beta, C, &_ldc);
#endif
}

inline void lcdgemm(double* A, double* B, double* C,
                    const int mA, const int nB) {
#ifdef USE_CBLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mA, nB, mA, 1.0, A, mA, B, mA, 0.0, C, mA);
#else
    INTT _mA = (INTT) mA;
    INTT _nB = (INTT) nB;
    static double one = 1.0;
    static double zero = 0.0;
    dgemm((char*) "N", (char*) "N", &_mA, &_nB, &_mA, &one, A, &_mA, B, &_mA, &zero, C, &_mA);
#endif
}


#endif
