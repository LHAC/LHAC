//
//  LogReg.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/30/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#include "LogReg.h"
#include "linalg.h"
#ifdef _OPENMP
#include <omp.h>
#endif



LogReg::LogReg(const Parameter* param)
{
    
    // libsvm (sparse) format
    _Dset_sp_col = new training_set_sp;
    _Dset_sp_row = new training_set_sp;
    // in libsvm format
    read_problem(param->fileName, _Dset_sp_row);
    transpose(_Dset_sp_row, _Dset_sp_col);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", _Dset_sp_col->p, _Dset_sp_col->N, _Dset_sp_col->nnz);
    
    _p = _Dset_sp_col->p;
    _N = _Dset_sp_col->N;
    
    if (param->dense > 0) {
        _format = DENSE;
        _Dset = new training_set;
        transformToDenseFormat(_Dset, _Dset_sp_col);
        _X = _Dset->X;
        _y = _Dset->y;
    }
    else {
        _format = SPARSE;
        _xcols = _Dset_sp_col->X;
        _xrows = _Dset_sp_row->X;
        _y = _Dset_sp_col->y;
    }


    _posweight = param->posweight;
    _e_ywx = new double[_N]; // N
    _B = new double[_N]; // N
    
    switch (_format) {
        case DENSE:
            delete _Dset_sp_row;
            delete _Dset_sp_col;
            _Dset_sp_row = NULL;
            _Dset_sp_col = NULL;
            break;
        
        case SPARSE:
            _Dset = NULL;
            break;
            
        default:
            break;
    }
}

LogReg::LogReg(const Parameter* param, double* X, double* y,
      unsigned long N, unsigned long p)
{
    _format = DENSE;
    _p = p;
    _N = N;
    _X = X;
    _y = y;
    _e_ywx = new double[_N]; // N
    _B = new double[_N]; // N
    
    _Dset_sp_row = NULL;
    _Dset_sp_col = NULL;
    _Dset = NULL;
    _posweight = param->posweight;
}

void LogReg::sparseVectorProduct(const SPARSE_TRANSPOSE trans,
                                 const double* b, double* c)
{
    switch (trans) {
        case SparseNoTrans:
#pragma omp parallel for
            for (unsigned long i = 0; i < _N; i++) {
                feature_node* xnode = _xrows[i];
                c[i] = 0;
                while (xnode->index != -1) {
                    int ind = xnode->index-1;
                    c[i] += b[ind]*(xnode->value);
                    xnode++;
                }
            }
            break;
            
        
        case SparseTrans:
#pragma omp parallel for
            for (unsigned long i = 0; i < _p; i++) {
                feature_node* xnode = _xcols[i];
                c[i] = 0;
                while (xnode->index != -1) {
                    int ind = xnode->index-1;
                    c[i] += b[ind]*(xnode->value);
                    xnode++;
                }
            }
            break;
    }
}

double LogReg::computeObject(double* wnew)
{
    double fval = 0;
    
    switch (_format) {
        case DENSE:
            lcdgemv(CblasColMajor, CblasNoTrans, _X, wnew, _e_ywx, (int)_N, (int)_p, (int)_N);
            break;
            
        case SPARSE:
            sparseVectorProduct(SparseNoTrans, wnew, _e_ywx);
            break;
    }
#pragma omp parallel for reduction(+:fval)
    for (unsigned long i = 0; i < _N; i++) {
        double nc1;
        double nc2;
        double weight = (_y[i]>0)?_posweight:1;
        nc1 = _e_ywx[i]*_y[i];
        _e_ywx[i] = exp(nc1);
        if (nc1 <= 0) {
            nc2 = _e_ywx[i];
            fval += weight*(log((1+nc2))-nc1);
        }
        else {
            nc2 = exp(-nc1);
            fval += weight*log((1+nc2));
        }
        
    }
    
    return fval / _N;
}

// always computed after computeObject
void LogReg::computeGradient(const double* wnew, double* df)
{
#pragma omp parallel for
    for (unsigned long i = 0; i < _N; i++) {
        double weight = (_y[i]>0)?_posweight:1;
        _B[i] = -weight*_y[i]/(1+_e_ywx[i])/_N;
    }
    switch (_format) {
        case DENSE:
            lcdgemv(CblasColMajor, CblasTrans, _X, _B, df, (int)_N, (int)_p, (int)_N);
            break;
            
        case SPARSE:
            sparseVectorProduct(SparseTrans, _B, df);
            break;
    }
    return;
}

LogReg::~LogReg()
{
    delete [] _e_ywx;
    delete [] _B;
    delete _Dset;
    delete _Dset_sp_row;
    delete _Dset_sp_col;
}





















