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
#include "linalg.h"
#include "timing.h"
#include "Parameter.h"
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define max(a,b) (a>b?a:b)

#define MAX_LENS 1024

#define __MATLAB_API__


enum { LHAC_MSG_NO=0, LHAC_MSG_NEWTON, LHAC_MSG_SD, LHAC_MSG_CD, LHAC_MSG_MAX };


enum{  GREEDY= 1, STD, GREEDY_CUTZERO, GREEDY_CUTGRAD, GREEDY_ADDZERO, STD_CUTGRAD, STD_CUTGRAD_AGGRESSIVE };



struct Func {
    double f;
    double g;
    double val; // f + g
    
    inline void add(const double _f, const double _g) {
        f = _f;
        g = _g;
        val = f + g;
    };
};

struct Solution {
    double* t;
    double* fval;
    double* normgs;
    int* niter;
    unsigned long* numActive;
    
    double* w;
    unsigned long p; //dimension of w
    
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
    
    inline void addEntry(double objval, double normsg, double elapsedTime,
                         int iter, unsigned long _numActive) {
        fval[size] = objval;
        normgs[size] = normsg;
        t[size] = elapsedTime;
        niter[size] = iter;
        numActive[size] = _numActive;
        (size)++;
    };
    
    inline void finalReport(const int error, double* wfinal) {
        memcpy(w, wfinal, p*sizeof(double));
        unsigned long last = size - 1;
        printf(
               "=========================== final report ========================\n"
               );
        if (error)
            printf("Terminated!\n");
        else
            printf("Optimal!\n");
        printf(
               "Best objective value found %+.6e\n"
               "In %3d iterations (%.4e seconds)\n"
               "With a precision of: %+.4e\n"
               "=================================================================\n",
               fval[last], niter[last], t[last], normgs[last] / normgs[0]
               );
    };
    
    Solution(unsigned long max_iter, unsigned long _p) {
        fval = new double[max_iter];
        normgs = new double[max_iter];
        t = new double[max_iter];
        niter = new int[max_iter];
        numActive = new unsigned long[max_iter];
        cdTime = 0;
        lbfgsTime1 = 0;
        lbfgsTime2 = 0;
        lsTime = 0;
        ngval = 0;
        nfval = 0;
        gvalTime = 0.0;
        fvalTime = 0.0;
        nls = 0;
        size = 0;
        p = _p;
        w = new double[p];
    };
    
    ~Solution() {
        delete [] w;
        delete [] fval;
        delete [] normgs;
        delete [] t;
        delete [] niter;
        
        return;
    };
};

inline void fistaUpdate(const unsigned long p, double* const t,
                        double* const x, double* const w) {
    double t_ = *t;
    *t = (1 + sqrt(1+4*t_*t_))*0.5;
    double c = (t_ - 1) / *t;
    for (unsigned long i = 0; i < p; i++) {
        double yi = w[i] + c*(w[i] - x[i]); // x is x_{k-1}
        x[i] = w[i];
        w[i] = yi;
    }
};


template <typename InnerSolver>
class Subproblem
{
public:
    inline void build(LBFGS* lR, double* grad,
                      work_set_struct* work_set) {
        return static_cast<InnerSolver*>(this)->build(lR, grad, work_set);
    };
    
    inline double objective_value(const double gama) {
        return static_cast<InnerSolver*>(this)->objective_value(gama);
    };
    
    inline void solve(double* w,
                      const double* const w_prev,
                      const unsigned short k,
                      const double gama) {
        static_cast<InnerSolver*>(this)->solve(w, w_prev, k, gama);
    };
    
    virtual ~Subproblem() {};
    
};

class CoordinateDescent: public Subproblem<CoordinateDescent>
{
public:
    CoordinateDescent(const Parameter* const _param, unsigned long _p)
                      : p(_p), lmd(_param->lmd), l(_param->l),
                        cd_rate(_param->cd_rate),
                        msgFlag(_param->verbose)
    {
        D = new double[p];
        H_diag = new double[p]; // p
        d_bar = new double[2*l]; // 2*l
    };
    
    ~CoordinateDescent() {
        delete [] D;
        delete [] H_diag;
        delete [] d_bar;
    };
    
    
    void build(LBFGS* lR, double* grad,
               work_set_struct* work_set) {
        Q = lR->Q;
        Q_bar = lR->Q_bar;
        m = lR->m;
        L_grad = grad;
        permut = work_set->permut;
        idxs = work_set->idxs;
        numActive = work_set->numActive;
        gama0 = lR->gama;
        buffer = lR->buff;
        memset(D, 0, p*sizeof(double));
        memset(d_bar, 0, 2*l*sizeof(double));
        for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
            H_diag[i] = gama0;
            for (unsigned long j = 0; j < m; j++)
                H_diag[i] -= Q_bar[k+j]*Q[k+j];
        }
    };
    
    double objective_value(const double gama) const {
        double order1 = lcddot((int)p, D, 1, L_grad, 1);
        double order2 = 0;
        int cblas_M = (int) numActive;
        int cblas_N = (int) m;
        lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, buffer, cblas_N, cblas_M, cblas_N);
        double vp = 0;
        for (unsigned long ii = 0; ii < numActive; ii++) {
            unsigned long idx = idxs[ii].j;
            unsigned long idx_Q = permut[ii];
            vp += D[idx]*buffer[idx_Q]; //vp += D[idx]*lcddot(m, &Q[idx_Q*m], 1, d_bar, 1);
        }
        order2 = gama*lcddot((int)p, D, 1, D, 1)-vp;
        order2 = order2*0.5;
        return order1 + order2;
    }
    
    void solve(double* w, const double* const w_prev,
               const unsigned short k, const double gama) {
        double z = 0.0;
        double Hd_j, Hii, G, Gp, Gn;
        double wpd, Hwd, Qd_bar;
        const double dH_diag = gama-gama0;
        const unsigned long max_cd_pass = 1 + k / cd_rate;
        for (unsigned long cd_pass = 1; cd_pass <= max_cd_pass; cd_pass++) {
            double diffd = 0;
            double normd = 0;
            for (unsigned long ii = 0; ii < numActive; ii++) {
                unsigned long rii = ii;
                unsigned long idx = idxs[rii].j;
                unsigned long idx_Q = permut[rii];
                unsigned long Q_idx_m = idx_Q*m;
                Qd_bar = lcddot(m, &Q[Q_idx_m], 1, d_bar, 1);
                Hd_j = gama*D[idx] - Qd_bar;
                Hii = H_diag[idx_Q] + dH_diag;
                G = Hd_j + L_grad[idx];
                Gp = G + lmd;
                Gn = G - lmd;
                wpd = w_prev[idx] + D[idx];
                Hwd = Hii * wpd;
                z = -wpd;
                if (Gp <= Hwd) z = -Gp/Hii;
                if (Gn >= Hwd) z = -Gn/Hii;
                D[idx] = D[idx] + z;
                for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                    d_bar[j] += z*Q_bar[k+j];
                diffd += fabs(z);
                normd += fabs(D[idx]);
            }
            if (msgFlag >= LHAC_MSG_CD) {
                printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                       cd_pass, diffd, normd);
            }
        }
        for (unsigned long i = 0; i < p; i++) {
            w[i] = w_prev[i] + D[i];
        }
    };
    
private:
    /* own */
    double *D, *d_bar, *H_diag;
    
    double *Q, *Q_bar, *L_grad, *buffer;
    const unsigned long cd_rate, l, p;
    unsigned long* permut;
    ushort_pair_t* idxs;
    const double lmd;
    double gama0;
    unsigned long numActive;
    const int msgFlag;
    unsigned short m;
    
};


class ProximalGradient: public Subproblem<ProximalGradient>
{
public:
    ProximalGradient(const Parameter* const _param, unsigned long _p)
    : p(_p), lmd(_param->lmd), l(_param->l),
    cd_rate(_param->cd_rate),
    msgFlag(_param->verbose)
    {
        w_ = new double[p];
        x = new double[p];
        d_bar = new double[2*l]; // 2*l
    };
    
    ~ProximalGradient() {
        delete [] w_;
        delete [] x;
        delete [] d_bar;
    };
    
    
    void build(LBFGS* lR, double* grad,
               work_set_struct* work_set) {
        Q = lR->Q;
        Q_bar = lR->Q_bar;
        m = lR->m;
        L_grad = grad;
        permut = work_set->permut;
        idxs = work_set->idxs;
        numActive = work_set->numActive;
        gama0 = lR->gama;
        Qd_bar = lR->buff;
        f_curr = 0.0;
        f_trial = 0.0;
        ista_size = 1/gama0;
        memset(Qd_bar, 0, p*sizeof(double));
    };
    
    double objective_value(const double gama) const {
        return f_curr;
    }
    
    void solve(double* w, const double* const w_prev,
               const unsigned short k, const double gama) {
        double z = 0.0;
        double Q_grad_j, Hd_j, Qd_bar_j;
        const unsigned long max_sub_iter = 1 + k / cd_rate;
        unsigned long nb = 0;
        /* gama modified */
        if (f_curr != 0) {
            f_curr = 0.0;
            for (unsigned long ii = 0; ii < numActive; ii++) {
                unsigned long idx = idxs[ii].j;
                unsigned long idx_Q = permut[ii];
                double Dj = w[idx] - w_prev[idx];
                f_curr += Dj*L_grad[idx] + 0.5*(gama*Dj*Dj - Dj*Qd_bar[idx_Q]);
            }
        }
        double tf = 1.0;
        memcpy(x, w, p*sizeof(double));
        for (unsigned long sub_iter = 1; sub_iter <= max_sub_iter; sub_iter++) {
            memcpy(w_, w, p*sizeof(double));
            const double* w_curr = w_;
            double diffd;
            for (int backtrack=0; backtrack<100; backtrack++) {
                diffd = 0.0;
                double t = ista_size*lmd;
                memset(d_bar, 0, 2*l*sizeof(double));
                for (unsigned long ii = 0; ii < numActive; ii++) {
                    unsigned long idx = idxs[ii].j;
                    unsigned long idx_Q = permut[ii];
                    unsigned long Q_idx_m = idx_Q*m;
                    Qd_bar_j = Qd_bar[idx_Q];
                    z = w_curr[idx] - w_prev[idx]; // w[idx] == w_[idx]
                    Hd_j = gama*z - Qd_bar_j;
                    Q_grad_j = Hd_j + L_grad[idx];
                    double uj = w_curr[idx] - ista_size*Q_grad_j;
                    if (uj > t)
                        w[idx] = uj - t;
                    else if (uj < -t)
                        w[idx] = uj + t;
                    else
                        w[idx] = 0.0;
                    z = w[idx] - w_prev[idx];
                    for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                        d_bar[j] += z*Q_bar[k+j];
                    double wd = w[idx] - w_curr[idx];
                    diffd += fabs(wd);
                    /*******************************************************************************
                     double reduction = 
                     wd*Q_grad_j + (0.5/ista_size)*wd*wd
                     + lmd*(fabs(w_curr[idx]+wd) - fabs(w_curr[idx]));
                     *******************************************************************************/
                }
                double order1 = 0.0;
                double order2 = 0.0;
                for (unsigned long ii = 0; ii < numActive; ii++) {
                    unsigned long idx = idxs[ii].j;
                    unsigned long idx_Q = permut[ii];
                    double dj = w[idx] - w_curr[idx];
                    double Dj_prev = w_curr[idx] - w_prev[idx];
                    order2 += dj*dj;
                    order1 += dj*(gama*Dj_prev - Qd_bar[idx_Q] + L_grad[idx]);
                }
                int cblas_M = (int) numActive;
                int cblas_N = (int) m;
                lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, Qd_bar, cblas_N, cblas_M, cblas_N);
                f_trial = 0.0;
                for (unsigned long ii = 0; ii < numActive; ii++) {
                    unsigned long idx = idxs[ii].j;
                    unsigned long idx_Q = permut[ii];
                    double Dj = w[idx] - w_prev[idx];
                    f_trial += Dj*L_grad[idx] + 0.5*(gama*Dj*Dj - Dj*Qd_bar[idx_Q]);
                }
                if (diffd <= 1e-15)
                    break;
                /*******************************************************************************
                 double q_trial = f_trial;
                 double q_delta = order1 + (0.5/ista_size)*order2;
                 for (unsigned long ii = 0; ii < numActive; ii++) {
                     unsigned long idx = idxs[ii].j;
                     q_trial += lmd*fabs(w[idx]);
                     q_delta += lmd*fabs(w[idx]) - lmd*fabs(w_curr[idx]);
                 }
                 *******************************************************************************/
                double f_approx = f_curr + order1 + (0.5/ista_size)*order2;
                if (msgFlag >= LHAC_MSG_CD) {
                    printf("\t\t Proximal Gradient Iter %3ld: backtrack %3d:   f_trial = %+.4e, f_approx = %+.4e\n",
                           sub_iter, backtrack, f_trial, f_approx);
                }
                if (f_trial > f_approx) {
                    ista_size = ista_size * 0.5;
                    continue;
                }
                nb += backtrack;
                break;
            }
            fistaUpdate(p, &tf, x, w); // w becomes y_{k+1} from x_k, x becomes x_k
            memset(d_bar, 0, 2*l*sizeof(double));
            for (unsigned long ii = 0; ii < numActive; ii++) {
                unsigned long idx = idxs[ii].j;
                unsigned long idx_Q = permut[ii];
                unsigned long Q_idx_m = idx_Q*m;
                z = w[idx] - w_prev[idx];
                for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                    d_bar[j] += z*Q_bar[k+j];
            }
            int cblas_M = (int) numActive;
            int cblas_N = (int) m;
            lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, Qd_bar, cblas_N, cblas_M, cblas_N);
            f_curr = 0.0;
            for (unsigned long ii = 0; ii < numActive; ii++) {
                unsigned long idx = idxs[ii].j;
                unsigned long idx_Q = permut[ii];
                double Dj = w[idx] - w_prev[idx];
                f_curr += Dj*L_grad[idx] + 0.5*(gama*Dj*Dj - Dj*Qd_bar[idx_Q]);
            }
//            f_curr = f_trial;
            if (diffd <= 1e-15)
                break;
        }
//        printf("backtrack: %ld\n", nb);
        // now store the w - w_prev
//        memset(w_, 0, p*sizeof(double));
//        for (unsigned long ii = 0; ii < numActive; ii++) {
//            unsigned long idx = idxs[ii].j;
//            w_[idx] = w[idx] - w_prev[idx];
//        }
    };
    
private:
    /* own */
    double *d_bar, *w_, *x;
    
    double f_curr, f_trial;
    double ista_size, max_H_diag;
    double *Q, *Q_bar, *L_grad, *Qd_bar;
    const unsigned long cd_rate, l, p;
    unsigned long* permut;
    ushort_pair_t* idxs;
    const double lmd;
    double gama0;
    unsigned long numActive;
    const int msgFlag;
    unsigned short m;
    
};




template <typename Derived>
class LHAC
{
public:
    
    LHAC(Objective<Derived>* _mdl, const Parameter* const _param)
    : mdl(_mdl), param(_param) {
        p = mdl->getDims();
        obj = new Func;
        
        l = param->l;
        opt_outer_tol = param->opt_outer_tol;
        max_iter = param->max_iter;
        lmd = param->lmd;
        msgFlag = param->verbose;
        
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
        
        sols = new Solution(max_iter, p);
        work_set = new work_set_struct(p);
        lR = new LBFGS(p, l, param->shrink);
        
        ista_size = 1;
    
    };
    
    ~LHAC() {
        delete [] w_prev;
        delete [] w;
        delete [] L_grad;
        delete [] L_grad_prev;
        delete [] D;
        delete [] H_diag;
        delete [] d_bar;
        delete lR;
        delete work_set;
        
    }
    
    int ista() {
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        int error = 0;
        normsg = normsg0;
        for (ista_iter = 1; ista_iter <= max_iter; ista_iter++) {
            error = istaStep();
            if (error) {
                break;
            }
            double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            if (ista_iter == 1 || ista_iter % 30 == 0 )
                sols->addEntry(obj->val, normsg, elapsedTime, ista_iter, work_set->numActive);
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e\n",
                       elapsedTime, ista_iter, obj->f, normsg);
            normsg = computeSubgradient();
            mdl->computeGradient(w, L_grad);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
        }
        return error;
    }
    
    // proximal inexact quasi-newton
    int piqn() {
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        initialStep();
        int error = 0;
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            sols->addEntry(obj->val, normsg, elapsedTime, newton_iter, work_set->numActive);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
            error = suffcientDecrease();
            if (error) {
                break;
            }
            memcpy(L_grad_prev, L_grad, p*sizeof(double));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        }
        return error;
    }
    
    template <typename InnerSolver>
    int piqnGeneral(Subproblem<InnerSolver>* subprob) {
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        initialStep();
        int error = 0;
        const unsigned short max_inner_iter = 200;
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            sols->addEntry(obj->val, normsg, elapsedTime, newton_iter, work_set->numActive);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
            
            /* inner solver starts*/
            subprob->build(lR, L_grad, work_set);
            double gama = lR->gama;
            double rho_trial = 0.0;
            memcpy(w_prev, w, p*sizeof(double));
            unsigned short inner_iter;
            for (inner_iter = 0; inner_iter < max_inner_iter; inner_iter++) {
                subprob->solve(w, w_prev, newton_iter, gama);
                const bool good_d = sufficientDecreaseCheck(w, subprob, gama, &rho_trial);
                if (good_d) {
                    if (msgFlag >= LHAC_MSG_SD)
                        printf("\t \t \t # of line searches = %3d; model quality: %+.3f\n", inner_iter, rho_trial);
                    break;
                }
                else
                    gama *= 2.0;
                
            }
            /* inner solver ends */
            if (inner_iter >= max_inner_iter) {
                error = 1;
                break;
            }

            memcpy(L_grad_prev, L_grad, p*sizeof(double));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        }
        return error;
    }
    
    /* fast proximal inexact quasi-newton */
    int fpiqn() {
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        initialStep();
        double t = 1.0;
        int error = 0;
        double* x = new double[p];
        memcpy(x, w, p*sizeof(double)); // w_1 (y_1) == x_0
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            sols->addEntry(obj->val, normsg, elapsedTime, newton_iter, work_set->numActive);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
            error = suffcientDecrease();
            if (error) {
                break;
            }
            fistaUpdate(p, &t, x, w);
            obj->add(mdl->computeObject(w), computeReg(w));
            memcpy(L_grad_prev, L_grad, p*sizeof(double));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        }
        return error;
    }
    
    Solution* solve()
    {
        obj->add(mdl->computeObject(w), computeReg(w));
        mdl->computeGradient(w, L_grad);
        normsg0 = computeSubgradient();
        int error = 0;
        
        switch (param->method_flag) {
            case 1:
                error = ista();
                break;
                
            case 2:
                error = piqn();
                break;
                
            case 3:
                error = fpiqn();
                break;
                
            case 41:
                error = piqnGeneral(new ProximalGradient(param, p));
                break;
            
            case 42:
                error = piqnGeneral(new CoordinateDescent(param, p));
                break;
                
            default:
                error = 1;
                fprintf(stderr, "ValueError: flag q only accept value 1 (ISTA), 2 (lhac) or 3 (f-lhac).\n");
                break;
        }
        
        sols->finalReport(error, w);
        return sols;
    };
    
private:
    Objective<Derived>* mdl;
    const Parameter* param;
    Solution* sols;
    work_set_struct* work_set;
    Func* obj;
    LBFGS* lR;
    
    unsigned long l;
    double opt_outer_tol;
    unsigned short max_iter;
    double lmd;
    int msgFlag;
    
    unsigned long p;
    unsigned short newton_iter;
    unsigned short ista_iter;
    double ista_size;
    
    double* D;
    double normsg0;
    double normsg;
    double* w_prev;
    double* w;
    double* L_grad_prev;
    double* L_grad;
    double* H_diag; // p
    double* d_bar; // 2*l
    
    
    void initialStep() {
        // initial step (only for l1)
        for (unsigned long idx = 0; idx < p; idx++) {
            double G = L_grad[idx];
            double Gp = G + lmd;
            double Gn = G - lmd;
            double Hwd = 0.0;
            if (Gp <= Hwd)
                D[idx] = -Gp;
            else if (Gn >= Hwd)
                D[idx] = -Gn;
            else
                D[idx] = 0.0;
        }
        double a = 1.0;
        double l1_next = 0.0;
        double delta = 0.0;
        for (unsigned long i = 0; i < p; i++) {
            w[i] += D[i];
            l1_next += lmd*fabs(w[i]);
            delta += L_grad[i]*D[i];
        }
        delta += l1_next - obj->g;
        // line search
        for (unsigned long lineiter = 0; lineiter < 1000; lineiter++) {
            double f_trial = mdl->computeObject(w);
            double obj_trial = f_trial + l1_next;
            if (obj_trial < obj->val + a*0.001*delta) {
                obj->add(f_trial, l1_next);
                break;
            }
            a = 0.5*a;
            l1_next = 0;
            for (unsigned long i = 0; i < p; i++) {
                w[i] = w_prev[i] + a*D[i];
                l1_next += lmd*fabs(w[i]);
            }
        }
        memcpy(L_grad_prev, L_grad, p*sizeof(double));
        mdl->computeGradient(w, L_grad);
        lR->initData(w, w_prev, L_grad, L_grad_prev);
    }
    
    int istaStep() {
        memcpy(w_prev, w, p*sizeof(double));
        for (int backtrack=0; backtrack<200; backtrack++) {
            double t = ista_size*lmd;
            unsigned long i;
#pragma omp parallel for private(i)
            for (i = 0; i < p; i++) {
                double ui = w_prev[i] - ista_size*L_grad[i];
                if (ui > t)
                    w[i] = ui - t;
                else if (ui < -t)
                    w[i] = ui + t;
                else
                    w[i] = 0.0;
                D[i] = w[i] - w_prev[i];
            }
            double order1 = lcddot((int)p, D, 1, L_grad, 1);
            double order2 = lcddot((int)p, D, 1, D, 1);
            double f_trial = mdl->computeObject(w);
            if (f_trial > obj->f + order1 + (0.5/ista_size)*order2) {
                ista_size = ista_size * 0.5;
                continue;
            }
            obj->add(f_trial, 0);
            return 0;
        }
        return 1;
    }
    


    /* may generalize to other regularizations beyond l1 */
    double computeReg(const double* const wnew) {
        double gval = 0.0;
        for (unsigned long i = 0; i < p; i++)
            gval += lmd*fabs(wnew[i]);
        return gval;
    }

    double computeSubgradient() {
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
    
    static int _cmp_by_vlt(const void *a, const void *b)
    {
        const ushort_pair_t *ia = (ushort_pair_t *)a;
        const ushort_pair_t *ib = (ushort_pair_t *)b;
        
        if (ib->vlt - ia->vlt > 0) {
            return 1;
        }
        else if (ib->vlt - ia->vlt < 0){
            return -1;
        }
        else
            return 0;
    }
    
    static int _cmp_by_vlt_reverse(const void *a, const void *b)
    {
        const ushort_pair_t *ia = (ushort_pair_t *)a;
        const ushort_pair_t *ib = (ushort_pair_t *)b;
        if (ib->vlt - ia->vlt > 0) {
            return -1;
        }
        else if (ib->vlt - ia->vlt < 0){
            return 1;
        }
        else
            return 0;
    }
    
    void computeWorkSet()
    {
        switch (param->active_set) {
            case GREEDY:
                greedySelector();
                break;
                
            case STD:
                stdSelector();
                break;
                
            case STD_CUTGRAD:
                stdSelector_cutgrad();
                break;
                
            case STD_CUTGRAD_AGGRESSIVE:
                stdSelector_cutgrad_aggressive();
                break;
            
            case GREEDY_CUTGRAD:
                greedySelector();
                break;
                
            case GREEDY_CUTZERO:
                greedySelector_cutzero();
                break;
                
            case GREEDY_ADDZERO:
                greedySelector_addzero();
                break;
                
            default:
                stdSelector();
                break;
        }
        /* reset permutation */
        for (unsigned long j = 0; j < work_set->numActive; j++) {
            work_set->permut[j] = j;
        }
        return;
    }
    
    void stdSelector()
    {
        
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        /*** select rule 2 ***/
        for (unsigned long j = 0; j < p; j++) {
            double g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd)) {
                idxs[numActive].i = j;
                idxs[numActive].j = j;
                numActive++;
            }
        }
        work_set->numActive = numActive;
        return;
    }
    
    void stdSelector_cutgrad()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        /*** select rule 2 ***/
        for (unsigned long j = 0; j < p; j++) {
            double g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd + 0.01)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                numActive++;
            }
        }
        work_set->numActive = numActive;
        return;
    }
    
    void stdSelector_cutgrad_aggressive()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        /*** select rule 2 ***/
        for (unsigned long j = 0; j < p; j++) {
            double g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd + 0.5)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                numActive++;
            }
        }
        work_set->numActive = numActive;
        return;
    }
    
    void greedySelector()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long zeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            double g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                g = fabs(g) - lmd;
                idxs[numActive].vlt = fabs(g);
                numActive++;
                if (w[j] == 0.0) zeroActive++;
            }
        }
        qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), _cmp_by_vlt);
        work_set->numActive = numActive;
    }
    
    
    void greedySelector_cutgrad()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long zeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            double g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd + 0.01)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                g = fabs(g) - lmd;
                idxs[numActive].vlt = fabs(g);
                numActive++;
                if (w[j] == 0.0) zeroActive++;
            }
        }
        qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), _cmp_by_vlt);
        work_set->numActive = numActive;
    }
    
    void greedySelector_cutzero()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long zeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            double g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd)) {
                idxs[numActive].i = j;
                idxs[numActive].j = j;
                g = fabs(g) - lmd;
                idxs[numActive].vlt = fabs(g);
                numActive++;
                if (w[j] == 0.0) zeroActive++;
            }
        }
        qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), _cmp_by_vlt);
        // zerosActive small means found the nonzeros subspace
        numActive = (zeroActive<100)?numActive:(numActive-zeroActive);
        work_set->numActive = numActive;
    }
    
    void _insert(unsigned long idx, double vlt, unsigned long n)
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long end = p-1-n;
        unsigned long j;
        for (j = p-1; j > end; j--) {
            if (idxs[j].vlt >= vlt) continue;
            else {
                for (unsigned long i = j+1, k = j; i > end; i--, k--) {
                    // swap
                    unsigned long tmpj = idxs[k].j;
                    double tmpv = idxs[k].vlt;
                    idxs[k].j = idx;
                    idxs[k].vlt = vlt;
                    vlt = tmpv;
                    idx = tmpj;
                }
                break;
            }
        }
        if (j == end) {
            idxs[end].j = idx;
            idxs[end].vlt = vlt;
        }
    }
    
    double _vlt(unsigned long j)
    {
        double g = L_grad[j];
        if (w[j] > 0) g += lmd;
        else if (w[j] < 0) g -= lmd;
        else g = fabs(g) - lmd;
        return g;
    }
    
    
    /* not converging on a9a */
    void greedySelector_addzero_no()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long work_size = param->work_size;
        unsigned long zeroActive = 0;
        unsigned long nzeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            double g = fabs(L_grad[j]) - lmd;
            if (g > 0) {
                unsigned long end = p-1-zeroActive;
                idxs[end].j = j;
                idxs[end].vlt = g;
                zeroActive++;
            }
            else if (w[j] != 0.0) {
                idxs[nzeroActive].j = j;
                nzeroActive++;
            }
        }
        if (zeroActive>2*nzeroActive) {
            unsigned long pos = p - zeroActive;
            qsort((void *)(idxs+pos), (size_t) zeroActive, sizeof(ushort_pair_t), _cmp_by_vlt_reverse);
            work_size = (nzeroActive<10)?zeroActive/3:nzeroActive;
        }
        else work_size = zeroActive;
        numActive = nzeroActive;
        unsigned long end = p-work_size;
        for (unsigned long j = p-1; j >= end; j--) {
            idxs[numActive].j = idxs[j].j;
            numActive++;
        }
        work_set->numActive = numActive;
    }
    
    void greedySelector_addzero()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long work_size = param->work_size;
        unsigned long zeroActive = 0;
        unsigned long nzeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            double g = fabs(L_grad[j]) - lmd;
            if (g > 0) {
                _insert(j, g, zeroActive);
                zeroActive++;
            }
            else if (w[j] != 0.0) {
                idxs[nzeroActive].j = j;
                nzeroActive++;
            }
        }
        work_size = (nzeroActive<10)?zeroActive:nzeroActive;
        work_size = (zeroActive>2*nzeroActive)?work_size:zeroActive;
        numActive = nzeroActive;
        unsigned long end = p-work_size;
        for (unsigned long k = p, j = p-1; k > end; k--, j--) {
            idxs[numActive].j = idxs[j].j;
            numActive++;
        }
        work_set->numActive = numActive;
    }
    
    static inline void shuffle( work_set_struct* work_set )
    {
        unsigned long lens = work_set->numActive;
        ushort_pair_t* idxs = work_set->idxs;
        unsigned long* permut = work_set->permut;

        for (unsigned long i = 0; i < lens; i++) {
            unsigned long j = i + rand()%(lens - i);
            unsigned short k1 = idxs[i].i;
            unsigned short k2 = idxs[i].j;
            double vlt = idxs[i].vlt;
            idxs[i].i = idxs[j].i;
            idxs[i].j = idxs[j].j;
            idxs[i].vlt = idxs[j].vlt;
            idxs[j].i = k1;
            idxs[j].j = k2;
            idxs[j].vlt = vlt;

            /* update permutation */
            unsigned long tmp = permut[i];
            permut[i] = permut[j];
            permut[j] = tmp;
        }
        
        return;
    }


    int suffcientDecrease() {
        int max_sd_iters = 200;
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
        const double lmd = param->lmd;
        const unsigned long l = param->l;
        double* Q = lR->Q;
        const double* Q_bar = lR->Q_bar;
        const unsigned short m = lR->m;
        const double gama = lR->gama;
        memset(D, 0, p*sizeof(double));
        memset(d_bar, 0, 2*l*sizeof(double));
        for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
            H_diag[i] = gama;
            for (unsigned long j = 0; j < m; j++) H_diag[i] -= Q_bar[k+j]*Q[k+j];
        }
        unsigned long max_cd_pass = 1 + newton_iter / param->cd_rate;
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
                    Qd_bar = lcddot(m, &Q[Q_idx_m], 1, d_bar, 1);
                    Hd_j = gama_scale*D[idx] - Qd_bar;
                    Hii = H_diag[idx_Q] + dH_diag;
                    G = Hd_j + L_grad[idx];
                    Gp = G + lmd;
                    Gn = G - lmd;
                    wpd = w_prev[idx] + D[idx];
                    Hwd = Hii * wpd;
                    z = -wpd;
                    if (Gp <= Hwd) z = -Gp/Hii;
                    if (Gn >= Hwd) z = -Gn/Hii;
                    D[idx] = D[idx] + z;
                    for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                        d_bar[j] += z*Q_bar[k+j];
                    diffd += fabs(z);
                    normd += fabs(D[idx]);
                }
                if (msgFlag >= LHAC_MSG_CD) {
                    printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                           cd_pass, diffd, normd);
                }
            }
            for (unsigned long i = 0; i < p; i++) {
                w[i] = w_prev[i] + D[i];
            }
            double f_trial = mdl->computeObject(w);
            double g_trial = computeReg(w);
            double obj_trial = f_trial + g_trial;
            double order1 = lcddot((int)p, D, 1, L_grad, 1);
            double order2 = 0;
            double* buffer = lR->buff;
            int cblas_M = (int) work_set->numActive;
            int cblas_N = (int) m;
            lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, buffer, cblas_N, cblas_M, cblas_N);
            double vp = 0;
            for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
                unsigned long idx = idxs[ii].j;
                unsigned long idx_Q = permut[ii];
                vp += D[idx]*buffer[idx_Q];
            }
            order2 = mu*gama*lcddot((int)p, D, 1, D, 1)-vp;
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
        if (sd_iters == max_sd_iters) {
            fprintf(stderr, "failed to satisfy sufficient decrease condition.\n");
            return -1;
        }
        return 0;
    }
    
    template <typename InnerSolver>
    const bool sufficientDecreaseCheck(double* w, Subproblem<InnerSolver>* const subprob,
                                       const double gama, double* rho_trial) {

        double f_trial = mdl->computeObject(w);
        double g_trial = computeReg(w);
        double obj_trial = f_trial + g_trial;
        double f_mdl = obj->f + subprob->objective_value(gama) + g_trial;
        *rho_trial = (obj_trial-obj->val)/(f_mdl-obj->val);
//        printf("obj_trial: %.4e, f_mdl: %.4e\n", obj_trial, f_mdl);
        if (*rho_trial > param->rho) {
            obj->add(f_trial, g_trial);
            return true;
        }
        return false;

    }
};






#endif /* defined(__LHAC_v1__lhac__) */
