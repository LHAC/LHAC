//
//  lhac.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/5/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "lhac.h"

#include <math.h>

/* may generalize to other regularizations beyond l1 */
static inline double computeReg(double* w, unsigned long p, Parameter* param)
{
    double gval = 0.0;
    
    double lmd = param->lmd;
    
    for (unsigned long i = 0; i < p; i++) {
        gval += lmd*abs(w[i]);
    }
    
    return gval;
}

static inline double computeSubgradient(double lmd, double* L_grad, double* w, unsigned long p)
{
    
    double subgrad = 0.0;
    
    for (unsigned long i = 0; i < p; i++) {
        double g = L_grad[i];
        if (w[i] != 0.0 || (fabs(g) > lmd)) {
            if (w[i] > 0)
                g += lmd;
            else if (w[i] < 0)
                g -= lmd;
            else
                g = fabs(g) - lmd;
            subgrad += fabs(g);
        }
    }
    
    return subgrad;
}

static inline void computeWorkSet( work_set_struct* &work_set, double lmd,
                                  double* L_grad, double* w, unsigned long p)
{
    ushort_pair_t* &idxs = work_set->idxs;
    unsigned long numActive = 0;
    
    
    /*** select rule 2 ***/
    for (unsigned long j = 0; j < p; j++) {
        double g = L_grad[j];
        if (w[j] != 0.0 || (fabs(g) > lmd)) {
            idxs[numActive].i = (unsigned short) j;
            idxs[numActive].j = (unsigned short) j;
            numActive++;
        }
    }
    
    work_set->numActive = numActive;
    
    
    /* reset permutation */
    for (unsigned long j = 0; j < work_set->numActive; j++) {
        work_set->permut[j] = j;
    }
    return;
}

static inline void suffcientDecrease(LBFGS* lR, work_set_struct* work_set, solution* sols,
                                     Objective* mdl, Parameter* param, unsigned short iter, double* w,
                                     double* w_prev, double* D, double* d_bar, double* H_diag,
                                     double* L_grad, unsigned long p, double* f_current)
{
    int max_sd_iters = 20;
    double mu = 1.0;
    double rho = param->rho;
    int msgFlag = param->verbose;
    
    double z = 0.0;
    double Hd_j;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    double Qd_bar;
    
    double f_trial;
    double f_mdl;
    double rho_trial;
    
    memcpy(w_prev, w, p*sizeof(double));
    
    double lmd = param->lmd;
    unsigned long l = param->l;
    
    double* Q = lR->Q;
    double* Q_bar = lR->Q_bar;
    const unsigned short m = lR->m;
    const double gama = lR->gama;
    
    memset(D, 0, p*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    

    for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
        H_diag[i] = gama;
        for (unsigned long j = 0; j < m; j++)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[k+j];
    }
    
    
    /* mdl value change */
    double dQ = 0;
    
    //    unsigned long max_cd_pass = std::min(1 + iter/3, param->max_inner_iter);
    unsigned long max_cd_pass = 1 + iter / param->cd_rate;
    //    unsigned long max_cd_pass = param->max_inner_iter;
    unsigned long* permut = work_set->permut;
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long cd_pass;
    int sd_iters;
    
    for (sd_iters = 0; sd_iters < max_sd_iters; sd_iters++) {
        
        
        double gama_scale = mu*gama;
        double dH_diag = gama_scale-gama;
        
        for (cd_pass = 1; cd_pass <= max_cd_pass; cd_pass++) {
            double diffd = 0;
            double normd = 0;
            
            for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
                unsigned long rii = ii;
                unsigned long idx = idxs[rii].j;
                unsigned long idx_Q = permut[rii];
                unsigned long Q_idx_m = idx_Q*m;
                
                Qd_bar = cblas_ddot(m, &Q[Q_idx_m], 1, d_bar, 1);
                Hd_j = gama_scale*D[idx] - Qd_bar;
                
                Hii = H_diag[idx_Q] + dH_diag;
                G = Hd_j + L_grad[idx];
                Gp = G + lmd;
                Gn = G - lmd;
                wpd = w_prev[idx] + D[idx];
                Hwd = Hii * wpd;
                
                z = -wpd;
                if (Gp <= Hwd)
                    z = -Gp/Hii;
                if (Gn >= Hwd)
                    z = -Gn/Hii;
                
                D[idx] = D[idx] + z;
                
                for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                    d_bar[j] = d_bar[j] + z*Q_bar[k+j];
                
                diffd += fabs(z);
                normd += fabs(D[idx]);
                
            }
            
            if (msgFlag >= LHAC_MSG_CD) {
                printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e   Change in Q = %+.4e\n",
                       cd_pass, diffd, normd, dQ);
            }
            
            //            shuffle( work_set );
        }
        
        for (unsigned long i = 0; i < p; i++) {
            w[i] = w_prev[i] + D[i];
        }
        
//        f_trial = (mdl->computeObject(w) + computeReg(w, p, param));
        f_trial = mdl->computeObject(w);
        f_trial += computeReg(w, p, param);
        double order1 = cblas_ddot((int)p, D, 1, L_grad, 1);
        double order2 = 0;
        double l1norm = 0;
        
        double* buffer = lR->buff;
        
        int cblas_M = (int) work_set->numActive;
        int cblas_N = (int) m;
        
        cblas_dgemv(CblasRowMajor, CblasNoTrans, cblas_M, cblas_N, 1.0, Q, cblas_N, d_bar, 1, 0.0, buffer, 1);
        
        double vp = 0;
        for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
            unsigned long idx = idxs[ii].j;
            unsigned long idx_Q = permut[ii];
            vp += D[idx]*buffer[idx_Q];
        }
        
        order2 = mu*gama*cblas_ddot((int)p, D, 1, D, 1)-vp;
        order2 = order2*0.5;
        
        for (unsigned long i = 0; i < p; i++) {
            l1norm += lmd*(fabs(w[i]) - fabs(w_prev[i]));
        }
        
        f_mdl = *f_current + order1 + order2 + l1norm;
        
        rho_trial = (f_trial-*f_current)/(f_mdl-*f_current);
        
        printf("\t \t \t # of line searches = %3d; model quality: %+.3f\n", sd_iters, rho_trial);
        
        
        if (rho_trial > rho) {
            *f_current = f_trial;
            break;
        }
        mu = 2*mu;
        
    }
    
    return;
    
}


solution* lhac(Objective* mdl, Parameter* param)
{
    
    double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
    
    unsigned long l = param->l;
    double opt_outer_tol = param->opt_outer_tol;
    unsigned short max_iter = param->max_iter;
    double lmd = param->lmd;
    int msgFlag = param->verbose;
    
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
    
    unsigned long p = mdl->getDims();
    
    double* D;
    double normsg0;
    double normsg;
    double f_current;
    double f_trial;
    double* w_prev;
    double* w;
    double* L_grad_prev;
    double* L_grad;
    double* H_diag; // p
    double* d_bar; // 2*l
    
    w_prev = new double[p];
    w = new double[p];
    L_grad_prev = new double[p];
    L_grad = new double[p];
    D = new double[p];
    H_diag = new double[p]; // p
    d_bar = new double[2*param->l]; // 2*l
    
    /* initiate */
    memset(w, 0, p*sizeof(double));
    memset(w_prev, 0, p*sizeof(double));
    memset(D, 0, p*sizeof(double));
    
    f_current = (mdl->computeObject(w) + computeReg(w, p, param));
    mdl->computeGradient(w, L_grad);
    normsg0 = computeSubgradient(lmd, L_grad, w, p);
    
    // ista step
    for (unsigned long idx = 0; idx < p; idx++) {
        double G = L_grad[idx];
        double Gp = G + lmd;
        double Gn = G - lmd;
        double Hwd = 0.0;
        
        double z = 0.0;
        if (Gp <= Hwd)
            z = -Gp;
        if (Gn >= Hwd)
            z = -Gn;
        
        D[idx] = D[idx] + z;
    }
    
    double a = 1;
    
    double l1_current = 0.0;
    double l1_next = 0.0;
    double delta = 0.0;
    
    unsigned long max_linesearch_iter = param->max_linesearch_iter;
    double bbeta = param->bbeta;
    double ssigma = param->ssigma;
    
    for (unsigned long i = 0; i < p; i++) {
        l1_current += lmd*fabs(w[i]);
        w_prev[i] = w[i];
        w[i] += D[i];
        l1_next += lmd*fabs(w[i]);
        delta += L_grad[i]*D[i];
    }
    
    delta += l1_next - l1_current;
    
    // line search
    for (unsigned long lineiter = 0; lineiter < max_linesearch_iter; lineiter++) {
        f_trial = (mdl->computeObject(w) + computeReg(w, p, param));
        if (f_trial < f_current + a*ssigma*delta) {
            f_current = f_trial;
            break;
        }
        
        a = bbeta*a;
        
        for (unsigned long i = 0; i < p; i++) {
            w[i] = w_prev[i] + a*D[i];
        }
    }

    memcpy(L_grad_prev, L_grad, p*sizeof(double));
    mdl->computeGradient(w, L_grad);
    
    // active set
    work_set_struct* work_set = new work_set_struct;
    work_set->idxs = new ushort_pair_t[p];
    work_set->permut = new unsigned long[p];
    

    unsigned short newton_iter;
    sols->size = 0;
    
    LBFGS* lR = new LBFGS(p, l, param->shrink);
    lR->initData(w, w_prev, L_grad, L_grad_prev);
    for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
        
        computeWorkSet( work_set, lmd,
                       L_grad, w, p);
        
        
        lR->computeLowRankApprox_v2(work_set);
        suffcientDecrease(lR, work_set, sols, mdl, param,
                          newton_iter, w, w_prev, D, d_bar, H_diag,
                          L_grad, p, &f_current);
        
        
        double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
        
        normsg = computeSubgradient(lmd, L_grad, w, p);
        
        if (newton_iter == 1 || newton_iter % 30 == 0 ) {
            sols->fval[sols->size] = f_current;
            sols->normgs[sols->size] = normsg;
            sols->t[sols->size] = elapsedTime;
            sols->niter[sols->size] = newton_iter;
            sols->numActive[sols->size] = work_set->numActive;
            (sols->size)++;
        }
        if (msgFlag >= LHAC_MSG_NEWTON) {
            printf("%.4e  iter %2d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld   |w| = %.4e\n",
                   elapsedTime, newton_iter, f_current, normsg, work_set->numActive, computeReg(w, p, param));
        }
        
        
        memcpy(L_grad_prev, L_grad, p*sizeof(double));
        
        mdl->computeGradient(w, L_grad);
        
        /* update LBFGS */
        lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        

        if (normsg <= opt_outer_tol*normsg0) {
            break;
        }
        
    }
    
    
    
    delete mdl;
    delete lR;
    delete [] work_set->idxs;
    delete work_set;
    
    return sols;
}



//
//solution* lhac(l1log* mdl)
//{
//    double timeBegin = mdl->timeBegin;
//    
//    double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
//    
//    l1log_param* param = mdl->param;
//    int sd_flag = param->sd_flag;
//    unsigned long l = param->l;
//    double opt_outer_tol = param->opt_outer_tol;
//    unsigned short max_iter = param->max_iter;
//    
//    solution* sols = new solution;
//    sols->fval = new double[max_iter];
//    sols->normgs = new double[max_iter];
//    sols->t = new double[max_iter];
//    sols->niter = new int[max_iter];
//    sols->numActive = new unsigned long[max_iter];
//    sols->cdTime = 0;
//    sols->lbfgsTime1 = 0;
//    sols->lbfgsTime2 = 0;
//    sols->lsTime = 0;
//    sols->ngval = 0;
//    sols->nfval = 0;
//    sols->gvalTime = 0.0;
//    sols->fvalTime = 0.0;
//    sols->nls = 0;
//    
//    unsigned long p = mdl->p;
//    
//    LBFGS* lR = new LBFGS(p, l, param->shrink);
//    
//    lR->initData(mdl->w, mdl->w_prev, mdl->L_grad, mdl->L_grad_prev);
//    
//    // active set
//    work_set_struct* work_set = new work_set_struct;
//    work_set->idxs = new ushort_pair_t[p];
//    work_set->permut = new unsigned long[p];
//    
//    double normsg = mdl->computeSubgradient();
//    unsigned short newton_iter;
//    sols->size = 0;
//    for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
//        mdl->iter = newton_iter;
//        
//        mdl->computeWorkSet(work_set);
//        
//        
//        double lbfgs1 = CFAbsoluteTimeGetCurrent();
//        lR->computeLowRankApprox_v2(work_set);
//        sols->lbfgsTime1 += CFAbsoluteTimeGetCurrent() - lbfgs1;
//        
//        
//        if (sd_flag == 0) {
//            /* old sufficient decrease condition */
//            mdl->coordinateDsecent(lR, work_set);
//            double eTime = CFAbsoluteTimeGetCurrent();
//            mdl->lineSearch();
//            eTime = CFAbsoluteTimeGetCurrent() - eTime;
//            sols->lsTime += eTime;
//            
//        }
//        else {
//            mdl->suffcientDecrease(lR, work_set, 1.0, sols);
//        }
//        
//        double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
//        
//        normsg = mdl->computeSubgradient();
//        
//        if (newton_iter == 1 || newton_iter % 30 == 0 ) {
//            sols->fval[sols->size] = mdl->f_current;
//            sols->normgs[sols->size] = normsg;
//            sols->t[sols->size] = elapsedTime;
//            sols->niter[sols->size] = newton_iter;
//            sols->numActive[sols->size] = work_set->numActive;
//            (sols->size)++;
//        }
//        if (mdl->MSG >= LHAC_MSG_NEWTON) {
//            printf("%.4e  iter %2d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
//                   elapsedTime, newton_iter, mdl->f_current, normsg, work_set->numActive);
//        }
//        
//        
//        memcpy(mdl->L_grad_prev, mdl->L_grad, p*sizeof(double));
//        
//        double gradientTime = CFAbsoluteTimeGetCurrent();
//        mdl->computeGradient();
//        sols->gvalTime += CFAbsoluteTimeGetCurrent() - gradientTime;
//        
//        /* update LBFGS */
//        double lbfgs2 = CFAbsoluteTimeGetCurrent();
//        lR->updateLBFGS(mdl->w, mdl->w_prev, mdl->L_grad, mdl->L_grad_prev);
//        sols->lbfgsTime2 += CFAbsoluteTimeGetCurrent() - lbfgs2;
//        
//        
//
//        if (normsg <= opt_outer_tol*mdl->normsg0) {
//            //            printf("# of line searches = %d.\n", lineiter);
//            break;
//        }
//        
//    }
//    
//    //    double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
//    //    sols->fval[sols->size] = mdl->f_current;
//    //    sols->normgs[sols->size] = normsg;
//    //    sols->t[sols->size] = elapsedTime;
//    //    sols->niter[sols->size] = newton_iter;
//    //    (sols->size)++;
//    
//    
//    delete mdl;
//    delete lR;
//    delete [] work_set->idxs;
//    delete work_set;
//    return sols;
//}
