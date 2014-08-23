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

//#ifdef __APPLE__
#ifdef  USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#define INTT int
#else
#include <stddef.h>
#include "lapack.h"
#include "blas.h"
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
	AtlasConj=114};
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
#define INTT ptrdiff_t
#endif


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
#ifdef USE_ACCELERATE
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
#ifdef USE_ACCELERATE
    cblas_dgemv(Order, TransA, m, n, 1.0, A, lda, b, 1, 0.0, c, 1);
#else
    double one = 1.0;
    double zero = 0.0;
    INTT one_int = 1;
    INTT blas_m = (INTT) m;
    INTT blas_n = (INTT) n;
    INTT blas_lda = (INTT) lda;
    dgemv_(blas_transpose(TransA), &blas_m, &blas_n, &one, A, &blas_lda, b, &one_int, &zero, c, &one_int);
#endif
}

inline void lcdgemm(double* A, double* B, double* C,
                    const int mA, const int nB) {
#ifdef USE_ACCELERATE
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mA, nB, mA, 1.0, A, mA, B, mA, 0.0, C, mA);
#else
    INTT _mA = (INTT) mA;
    INTT _nB = (INTT) nB;
    double one = 1.0;
    double zero = 0.0;
    dgemm_((char*) "N", (char*) "N", &_mA, &_nB, &_mA, &one, A, &_mA, B, &_mA, &zero, C, &_mA);
#endif
}


#endif
