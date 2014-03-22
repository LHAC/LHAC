//
//  myUtilities.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/10/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__myUtilities__
#define __LHAC_v1__myUtilities__

#include "lhac.h"
#include "Lbfgs.h"


enum { ROW_MAJOR, COL_MAJOR };

void printout(const char* desc, double* x, unsigned long s1);

void printout(const char* desc, double* x, unsigned long s1, int mode);

void printout(const char* desc, double* x, unsigned long s1, unsigned long s2);

void printout(const char* desc, double* x, unsigned long s1, unsigned long s2, int mode);

void printout(const char* desc, work_set_struct* work_set);

void printout(const char* desc, LMatrix* x);

void printout(const char* desc, LMatrix* x, int mode);

void printout(const char* desc, solution* sols, param* param);


double norm(double* x, unsigned long s1);




void readTextFile(const char* filename, double* &S, unsigned long* _p);

#ifdef __MATLAB_API__
void readMatFiles(const char* fileName, double* &S, unsigned long* _p);

void write2mat(const char* fileName, const char* name,
               double* x, unsigned long s1, unsigned long s2);

void write2mat(const char* fileName, const char* name, LMatrix* x);

#endif





#endif /* defined(__LHAC_v1__myUtilities__) */
