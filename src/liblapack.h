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
//    __CLPK_integer ipiv[n+1];
    static __CLPK_integer lwork = MAX_SY_PAIRS*MAX_SY_PAIRS;
//    __CLPK_integer lwork = n*n;
    static double work[MAX_SY_PAIRS*MAX_SY_PAIRS];
//    double work[n*n];
    dgetrf_(&n, &n, w, &n, ipiv, &info);
    dgetri_(&n, w, &n, ipiv, work, &lwork, &info);
    
    return info;
}


inline double lcddot(int n, double* dx, int incx, double* dy, int incy) {
    return cblas_ddot(n, dx, incx, dy, incy);
}




#else
#include "lapack.h"
#include "blas.h"

#define cblas_dgemv dgemv_
#define cblas_dgemm dgemm_

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



#endif

#endif
