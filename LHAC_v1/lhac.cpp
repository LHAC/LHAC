//
//  lhac.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/5/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "lhac.h"


solution* lhac(l1log* mdl)
{
    double timeBegin = mdl->timeBegin;
    
    double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
    
    l1log_param* param = mdl->param;
    int sd_flag = param->sd_flag;
    unsigned long l = param->l;
    double opt_outer_tol = param->opt_outer_tol;
    unsigned short max_iter = param->max_iter;
    
    solution* sols = new solution;
    sols->fval = new double[max_iter];
    sols->normgs = new double[max_iter];
    sols->t = new double[max_iter];
    sols->niter = new int[max_iter];
    sols->numActive = new unsigned long[max_iter];
    sols->cdTime = 0;
    sols->lbfgsTime1 = 0;
    sols->lbfgsTime2 = 0;
    sols->lsTime = 0;
    sols->ngval = 0;
    sols->nfval = 0;
    sols->gvalTime = 0.0;
    sols->fvalTime = 0.0;
    sols->nls = 0;
    
    unsigned long p = mdl->p;
    
    LBFGS* lR = new LBFGS(p, l, param->shrink);
    
    lR->initData(mdl->w, mdl->w_prev, mdl->L_grad, mdl->L_grad_prev);
    
    // active set
    work_set_struct* work_set = new work_set_struct;
    work_set->idxs = new ushort_pair_t[p];
    work_set->permut = new unsigned long[p];
    
    double normsg = mdl->computeSubgradient();
    unsigned short newton_iter;
    sols->size = 0;
    for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
        mdl->iter = newton_iter;
        
        mdl->computeWorkSet(work_set);
        
        
        double lbfgs1 = CFAbsoluteTimeGetCurrent();
        lR->computeLowRankApprox_v2(work_set);
        sols->lbfgsTime1 += CFAbsoluteTimeGetCurrent() - lbfgs1;
        
        
        if (sd_flag == 0) {
            /* old sufficient decrease condition */
            mdl->coordinateDsecent(lR, work_set);
            double eTime = CFAbsoluteTimeGetCurrent();
            mdl->lineSearch();
            eTime = CFAbsoluteTimeGetCurrent() - eTime;
            sols->lsTime += eTime;
            
        }
        else {
            mdl->suffcientDecrease(lR, work_set, 1.0, sols);
        }
        
        double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
        
        normsg = mdl->computeSubgradient();
        
        if (newton_iter == 1 || newton_iter % 30 == 0 ) {
            sols->fval[sols->size] = mdl->f_current;
            sols->normgs[sols->size] = normsg;
            sols->t[sols->size] = elapsedTime;
            sols->niter[sols->size] = newton_iter;
            sols->numActive[sols->size] = work_set->numActive;
            (sols->size)++;
        }
        if (mdl->MSG >= LHAC_MSG_NEWTON) {
            printf("%.4e  iter %2d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                   elapsedTime, newton_iter, mdl->f_current, normsg, work_set->numActive);
        }
        
        
        memcpy(mdl->L_grad_prev, mdl->L_grad, p*sizeof(double));
        
        double gradientTime = CFAbsoluteTimeGetCurrent();
        mdl->computeGradient();
        sols->gvalTime += CFAbsoluteTimeGetCurrent() - gradientTime;
        
        /* update LBFGS */
        double lbfgs2 = CFAbsoluteTimeGetCurrent();
        lR->updateLBFGS(mdl->w, mdl->w_prev, mdl->L_grad, mdl->L_grad_prev);
        sols->lbfgsTime2 += CFAbsoluteTimeGetCurrent() - lbfgs2;
        
        

        if (normsg <= opt_outer_tol*mdl->normsg0) {
            //            printf("# of line searches = %d.\n", lineiter);
            break;
        }
        
    }
    
//    double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
//    sols->fval[sols->size] = mdl->f_current;
//    sols->normgs[sols->size] = normsg;
//    sols->t[sols->size] = elapsedTime;
//    sols->niter[sols->size] = newton_iter;
//    (sols->size)++;
    
    
    delete mdl;
    delete lR;
    delete [] work_set->idxs;
    delete work_set;
    return sols;
}