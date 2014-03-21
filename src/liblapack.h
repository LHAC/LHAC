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
#else
#include "lapack.h"
#include "blas.h"

#define cblas_ddot ddot_
#define cblas_dgemv dgemv_
#define cblas_dgemm dgemm_
#endif

#endif
