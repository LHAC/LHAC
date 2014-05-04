//
//  lhac.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/31/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__lhac__
#define __LHAC_v1__lhac__

#include "Lbfgs.h"
#include "Objective.h"
#include <math.h>
#include <Accelerate/Accelerate.h>


#define MAX_LENS 1024

#define __MATLAB_API__


enum { LHAC_MSG_NO=0, LHAC_MSG_NEWTON, LHAC_MSG_SD, LHAC_MSG_CD, LHAC_MSG_MAX };


enum{  GREEDY= 1, STD };

typedef struct {
    double f;
    double g;
    double val; // f + g
    
    inline void add(double _f, double _g) {
        f = _f;
        g = _g;
        val = f + g;
    };
} Func;

typedef struct solution_struct {
    double* t;
    double* fval;
    double* normgs;
    int* niter;
    unsigned long* numActive;
    double cdTime;
    double lsTime;
    double lbfgsTime1;
    double lbfgsTime2;
    unsigned long size; // max_newton_iter
    
    unsigned long ngval;
    unsigned long nfval;
    unsigned long nls; // # of line searches
    double gvalTime;
    double fvalTime;
    
    /* result */
    int p_sics; //dimension of w
    
    ~solution_struct() {
        delete [] fval;
        delete [] normgs;
        delete [] t;
        delete [] niter;
        
        return;
    };
} Solution;

typedef struct {
    char* fileName;
    unsigned long work_size;
    unsigned short max_iter;
    unsigned long max_inner_iter;
    double lmd;
    double opt_inner_tol;
    double opt_outer_tol;
    /**** line search ****/
    double bbeta;
    double ssigma;
    unsigned long max_linesearch_iter;
    
    unsigned long l; // lbfgs sy pair number <= MAX_SY_PAIRS
    int verbose;
    
    /* line search */
    int sd_flag; // 1 = sufficient decrease; 0 = line search
    
    /* gama in lbfgs */
    double shrink; // gama = gama/shrink
    
    double rho;
    
    unsigned long cd_rate;
    
    // active set stragety
    unsigned long active_set;
    
} Parameter;


template <typename Derived>
class LHAC
{
public:
    
    LHAC(Objective<Derived>* _mdl, Parameter* _param) : mdl(_mdl), param(_param) {
        p = mdl->getDims();
        obj = new Func;
        
        l = param->l;
        opt_outer_tol = param->opt_outer_tol;
        max_iter = param->max_iter;
        lmd = param->lmd;
        msgFlag = param->verbose;
        
        sols = new Solution;
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
        sols->size = 0;
        
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
        
        // active set
        work_set = new work_set_struct;
        work_set->idxs = new ushort_pair_t[p];
        work_set->permut = new unsigned long[p];
        
        lR = new LBFGS(p, l, param->shrink);
    
    };
    
    Solution* solve()
    {
        
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        
        obj->add(mdl->computeObject(w), computeReg(w));
        mdl->computeGradient(w, L_grad);
        normsg0 = computeSubgradient();
        
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
            double f_trial = mdl->computeObject(w);
            double g_trial = computeReg(w);
            double obj_trial = f_trial + g_trial;
            if (obj_trial < obj->val + a*ssigma*delta) {
                obj->add(f_trial, g_trial);
                break;
            }
            
            a = bbeta*a;
            
            for (unsigned long i = 0; i < p; i++) {
                w[i] = w_prev[i] + a*D[i];
            }
        }
        
        memcpy(L_grad_prev, L_grad, p*sizeof(double));
        mdl->computeGradient(w, L_grad);
        

        lR->initData(w, w_prev, L_grad, L_grad_prev);
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            suffcientDecrease();
            double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (newton_iter == 1 || newton_iter % 30 == 0 ) {
                sols->fval[sols->size] = obj->val;
                sols->normgs[sols->size] = normsg;
                sols->t[sols->size] = elapsedTime;
                sols->niter[sols->size] = newton_iter;
                sols->numActive[sols->size] = work_set->numActive;
                (sols->size)++;
            }
            if (msgFlag >= LHAC_MSG_NEWTON) {
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            }
            memcpy(L_grad_prev, L_grad, p*sizeof(double));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
        }
        
        delete lR;
        delete [] work_set->idxs;
        delete work_set;
        return sols;
    };
    
private:
    Objective<Derived>* mdl;
    Parameter* param;
    Solution* sols;
    work_set_struct* work_set;
    
    unsigned long l;
    double opt_outer_tol;
    unsigned short max_iter;
    double lmd;
    int msgFlag;
    
    unsigned long p;
    
    double* D;
    double normsg0;
    double normsg;
    double* w_prev;
    double* w;
    double* L_grad_prev;
    double* L_grad;
    double* H_diag; // p
    double* d_bar; // 2*l
    
    Func* obj;
    LBFGS* lR;
    
    unsigned short newton_iter;
    
    


    /* may generalize to other regularizations beyond l1 */
    double computeReg(double* wnew)
    {
        double gval = 0.0;
        
        for (unsigned long i = 0; i < p; i++) {
            gval += lmd*fabs(wnew[i]);
        }
        
        return gval;
    }

    double computeSubgradient()
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

    void computeWorkSet()
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


    void suffcientDecrease()
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
        
        //    unsigned long max_cd_pass = std::min(1 + iter/3, param->max_inner_iter);
        unsigned long max_cd_pass = 1 + newton_iter / param->cd_rate;
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
                    printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                           cd_pass, diffd, normd);
                }
                
                //            shuffle( work_set );
            }
            
            for (unsigned long i = 0; i < p; i++) {
                w[i] = w_prev[i] + D[i];
            }
            
            double f_trial = mdl->computeObject(w);
            double g_trial = computeReg(w);
            double obj_trial = f_trial + g_trial;
            double order1 = cblas_ddot((int)p, D, 1, L_grad, 1);
            double order2 = 0;
            
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
            
            f_mdl = obj->f + order1 + order2 + g_trial;
            
            rho_trial = (obj_trial-obj->val)/(f_mdl-obj->val);
            
            if (msgFlag >= LHAC_MSG_SD) {
                printf("\t \t \t # of line searches = %3d; model quality: %+.3f\n", sd_iters, rho_trial);
            }
            
            if (rho_trial > rho) {
                obj->add(f_trial, g_trial);
                break;
            }
            mu = 2*mu;
            
        }
        
        return;
        
    }

    
};





#endif /* defined(__LHAC_v1__lhac__) */
