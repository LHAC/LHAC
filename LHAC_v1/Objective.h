//
//  Objective.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/29/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Objective__
#define __LHAC_v1__Objective__

#include "lhac.h"

/* implements l1log reg objective */
class Objective
{
public:
    unsigned long getDims();
    
    double computeObject(double* wnew);
    
    void computeGradient(double* wnew, double* df);
    
    Objective(Parameter* param);
    
    ~Objective();
    
private:
    unsigned long p;
    unsigned long N;
    
    double lmd;
    double* X;
    double* y;
    double* e_ywx; // N
    double* B; // N
    double* Xd; // N
    
};

#endif /* defined(__LHAC_v1__Objective__) */
