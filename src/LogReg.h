//
//  LogReg.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/30/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__LogReg__
#define __LHAC_v1__LogReg__

#include "Objective.h"
#include "Parameter.h"
#include "utils.h"

enum {SPARSE, DENSE};

enum SPARSE_TRANSPOSE {SparseNoTrans=12, SparseTrans=13};

/* implements l1log reg objective */
class LogReg : public Objective<LogReg>
{
public:
    unsigned long getSize() const { return _N; };
    
    unsigned long getDims() const { return _p; };
    
    double computeObject(double* wnew);
    
    void computeGradient(const double* wnew, double* const df);
    
    /* data input file name */
    LogReg(const Parameter* param);
    
    LogReg(const Parameter* param, double* X, double* y,
          unsigned long N, unsigned long p);
    
    ~LogReg();
    
private:
    unsigned long _p;
    unsigned long _N;
    
    double _posweight=1;
    int _format=DENSE;
    
    training_set_sp* _Dset_sp_row;
    training_set_sp* _Dset_sp_col;
    training_set* _Dset;
    feature_node** _xcols;
    feature_node** _xrows;
    
    double* _X;
    double* _y;
    double* _e_ywx; // N
    double* _B; // N
    
    void sparseVectorProduct(const SPARSE_TRANSPOSE trans,
                             const double* b, double* c);
    
};

#endif /* defined(__LHAC_v1__LogReg__) */
