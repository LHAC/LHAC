//
//  Objective.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/29/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Objective__
#define __LHAC_v1__Objective__


class Objective
{
public:
    virtual unsigned long getDims() =0;
    
    virtual double computeObject(double* wnew) =0;
    
    virtual void computeGradient(double* wnew, double* df) =0;
    
    virtual ~Objective() {};
};

#endif /* defined(__LHAC_v1__Objective__) */
