//
//  Lasso.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 10/18/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Lasso__
#define __LHAC_v1__Lasso__

#include "Objective.h"

class Lasso : public Objective<Lasso>
{
public:
    unsigned long getDims() const;
    
    double computeObject(double* wnew);
    
    void computeGradient(const double* wnew, double* const df);
    
    /* data input file name */
    /* by default compute a^a */
    Lasso(const char *filename, bool isCached=true);
    
    ~Lasso();
    
private:
    bool _isCached; // if aTa is computed
    
    unsigned long _p;
    unsigned long _N;
    double _bTb;
    
    double* _A; // N x p; allocated when _isCached is false
    double* _Ax; // N x 1; allocated when _isCached is false
    double* _aTa; // p x p; allocated when _isCached is true
    double* _aTax; // p x 1; allocated when _isCached is true
    double* _aTb; // p x 1
    
};
#endif /* defined(__LHAC_v1__Lasso__) */
