//
//  liblapack.h
//  pepper
//
//  Created by Xiaocheng Tang on 3/21/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef pepper_liblapack_h
#define pepper_liblapack_h

#define MAX_SY_PAIRS 100



#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

inline void lcdpotrf_(double* w, unsigned long n, int* _info) {
    __CLPK_integer info = 0;
    __CLPK_integer p0 = (__CLPK_integer) n;
    dpotrf_((char*) "U", &p0, w, &p0, &info);
    
    *_info = info;
}

inline void lcdpotri_(double* w, unsigned long n, int* _info) {
    __CLPK_integer info = 0;
    __CLPK_integer p0 = (__CLPK_integer) n;
    dpotri_((char*) "U", &p0, w, &p0, &info);
    
    *_info = info;
}


/* w square matrix */
inline int inverse(double*w, int _n) {
    __CLPK_integer info = 0;
    __CLPK_integer n = (__CLPK_integer) _n;
    static __CLPK_integer ipiv[MAX_SY_PAIRS+1];
    static __CLPK_integer lwork = MAX_SY_PAIRS*MAX_SY_PAIRS;
    static double work[MAX_SY_PAIRS*MAX_SY_PAIRS];
    dgetrf_(&n, &n, w, &n, ipiv, &info);
    dgetri_(&n, w, &n, ipiv, work, &lwork, &info);
    
    return info;
}


inline double lcddot(int n, double* dx, int incx, double* dy, int incy) {
    return cblas_ddot(n, dx, incx, dy, incy);

}


inline void lcdgemv(const enum CBLAS_ORDER Order,
                    const enum CBLAS_TRANSPOSE TransA,
                    double* A,
                    double* b, double* c,
                    int m, int n)
{
    cblas_dgemv(Order, TransA, m, n, 1.0, A, n, b, 1, 0.0, c, 1);
}

inline void lcdgemm(double* A, double* B, double* C, int mA, int nB) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mA, nB, mA, 1.0, A, mA, B, mA, 0.0, C, mA);
}







#else
#include "lapack.h"
#include "blas.h"

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
	AtlasConj=114};


inline void lcdpotrf_(double* w, unsigned long n, int* _info) {
    ptrdiff_t info = 0;
    ptrdiff_t p0 = n;
    dpotrf_((char*) "U", &p0, w, &p0, &info);
    
    *_info = info;
}

inline void lcdpotri_(double* w, unsigned long n, int* _info) {
    ptrdiff_t info = 0;
    ptrdiff_t p0 = n;
    dpotri_((char*) "U", &p0, w, &p0, &info);
    
    *_info = info;
}

/* w square matrix */
inline int inverse(double*w, int _n) {
    ptrdiff_t info = 0;
    ptrdiff_t n = (ptrdiff_t) _n;
    static ptrdiff_t ipiv[MAX_SY_PAIRS+1];
    static ptrdiff_t lwork = MAX_SY_PAIRS*MAX_SY_PAIRS;
    static double work[MAX_SY_PAIRS*MAX_SY_PAIRS];
    dgetrf_(&n, &n, w, &n, ipiv, &info);
    dgetri_(&n, w, &n, ipiv, work, &lwork, &info);
    
    return (int) info;
}

inline double lcddot(int n, double* dx, int incx, double* dy, int incy) {
    ptrdiff_t _n = (ptrdiff_t) n;
    ptrdiff_t _incx = (ptrdiff_t) incx;
    ptrdiff_t _incy = (ptrdiff_t) incy;
    return ddot(&_n, dx, &_incx, dy, &_incy);
}

inline void lcdgemv(const enum CBLAS_ORDER Order,
                    const enum CBLAS_TRANSPOSE TransA,
                    double* A,
                    double* b, double* c,
                    int m, int n)
{
    ptrdiff_t blas_m;
    ptrdiff_t blas_n;
    double one = 1.0;
    double zero = 0.0;
    ptrdiff_t one_int = 1;
    switch (Order) {
        case CblasRowMajor:
            switch (TransA) {
                case CblasNoTrans:
                    blas_m = (ptrdiff_t) n;
                    blas_n = (ptrdiff_t) m;
                    dgemv_((char*) "T", &blas_m, &blas_n, &one, A, &blas_m, b, &one_int, &zero, c, &one_int);
                    break;
                    
                default:
                    break;
            }
            break;
            
        case CblasColMajor:
            switch (TransA) {
                case CblasNoTrans:
                    blas_m = (ptrdiff_t) m;
                    blas_n = (ptrdiff_t) n;
                    dgemv_((char*) "N", &blas_m, &blas_n, &one, A, &blas_m, b, &one_int, &zero, c, &one_int);
                    break;
                    
                default:
                    break;
            }
            break;
            
        default:
            break;
    }
}

inline void lcdgemm(double* A, double* B, double* C, int mA, int nB) {
    ptrdiff_t _mA = (ptrdiff_t) mA;
    ptrdiff_t _nB = (ptrdiff_t) nB;
    double one = 1.0;
    double zero = 0.0;
    dgemm_((char*) "N", (char*) "N", &_mA, &_nB, &_mA, &one, A, &_mA, B, &_mA, &zero, C, &_mA);
}



#endif

#endif
