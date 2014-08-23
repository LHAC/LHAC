//
//  Objective.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/29/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Objective__
#define __LHAC_v1__Objective__

template <typename Derived>
class Objective
{
public:
    inline unsigned long getDims() {
        return static_cast<Derived*>(this)->getDims();
    };
    
    inline double computeObject(double* wnew) {
        return static_cast<Derived*>(this)->computeObject(wnew);
    };
    
    inline void computeGradient(const double* wnew, double* df) {
        static_cast<Derived*>(this)->computeGradient(wnew, df);
    };
    
};

#endif /* defined(__LHAC_v1__Objective__) */
