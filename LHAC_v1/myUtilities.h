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

void printout(const char* desc, training_set_sp* Dset, int mode);

void printout(const char* desc, solution* sols, l1log_param* param);


double norm(double* x, unsigned long s1, int p);

void generateRandomProb(training_set* Dset,
                        unsigned long p, unsigned long N, double nnz_percent);

void generateRandomProb(training_set_sp* Dset_sp, training_set* Dset,
                        unsigned long p, unsigned long N, double nnz_percent);

void releaseProb(training_set* Dset);

void releaseProb(training_set_sp* Dset);

void writeToFile(training_set* Dset);

void readLibsvm(const char *filename, training_set_sp* Dset_col);

void releaseSolution(solution* sols);

void transformToSparseFormat(training_set* Dset, training_set_sp* Dset_sp);

void transformToDenseFormat(training_set* Dset, training_set_sp* Dset_sp);

int cmp_by_vlt(const void *a, const void *b);

#ifdef __MATLAB_API__
void readMatFiles(const char* fileName, double* &S, unsigned long* _p);

void write2mat(const char* fileName, const char* name,
               double* x, unsigned long s1, unsigned long s2);

void write2mat(const char* fileName, const char* name, LMatrix* x);

void write2mat(const char* fileName, const char* name,
               work_set_struct* work_set);

void write2mat(const char* fileName, const char* name,
               unsigned long* x, unsigned long s1, unsigned long s2);

void readMatFiles(const char* fileName, training_set* Dset);
#endif





#endif /* defined(__LHAC_v1__myUtilities__) */
