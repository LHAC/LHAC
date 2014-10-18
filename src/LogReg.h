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

/* implements l1log reg objective */
class LogReg : public Objective<LogReg>
{
public:
    unsigned long getDims() const;
    
    double computeObject(double* wnew);
    
    void computeGradient(const double* wnew, double* const df);
    
    /* data input file name */
    LogReg(Parameter* param);
    
    LogReg(Parameter* param, double* X, double* y,
          unsigned long N, unsigned long p);
    
    ~LogReg();
    
private:
    unsigned long _p;
    unsigned long _N;
    
    double* _X;
    double* _y;
    double* _e_ywx; // N
    double* _B; // N
    
};

#endif /* defined(__LHAC_v1__LogReg__) */
