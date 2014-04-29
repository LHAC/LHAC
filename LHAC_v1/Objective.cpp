//
//  Objective.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/29/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#include "Objective.h"


Objective::Objective()
{
    
}

unsigned long Objective::getDims()
{
    return p;
}

double Objective::computeObject(double* wnew)
{
    double fval = 0;
    
    return fval;
}

void Objective::computeGradient(double* wnew, double* df)
{
    return;
}