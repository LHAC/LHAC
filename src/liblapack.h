//
//  liblapack.h
//  pepper
//
//  Created by Xiaocheng Tang on 3/21/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef pepper_liblapack_h
#define pepper_liblapack_h

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

inline void lcdpotrf_(double* w, unsigned long n, int* _info) {
    __CLPK_integer info = 0;
    __CLPK_integer p0 = n;
    dpotrf_((char*) "U", &p0, w, &p0, &info);
    
    *_info = info;
}

inline void lcdpotri_(double* w, unsigned long n, int* _info) {
    __CLPK_integer info = 0;
    __CLPK_integer p0 = n;
    dpotri_((char*) "U", &p0, w, &p0, &info);
    
    *_info = info;
}




#else
#include "lapack.h"
#include "blas.h"

#define cblas_ddot ddot_
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



#endif

#endif
