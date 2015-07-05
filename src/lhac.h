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
#ifdef _OPENMP
#include <omp.h>
#endif


#define MAX_LENS 1024

#define __MATLAB_API__


enum { LHAC_MSG_NO=0, LHAC_MSG_NEWTON, LHAC_MSG_SD, LHAC_MSG_CD, LHAC_MSG_MAX };


enum{  GREEDY= 1, STD, GREEDY_CUTZERO, GREEDY_CUTGRAD, GREEDY_ADDZERO, STD_CUTGRAD, STD_CUTGRAD_AGGRESSIVE };



struct Func {
    double f;
    double g;
    double val; // f + g
    
    inline void add(double _f, double _g) {
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



template <typename Derived>
class LHAC
{
public:
    
    LHAC(Objective<Derived>* _mdl, const Parameter* _param)
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
    
    Solution* ista() {
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
        sols->finalReport(error, w);
//        memcpy(sols->w, w, p*sizeof(double));
        return sols;
    }
    
    Solution* solve()
    {
        obj->add(mdl->computeObject(w), computeReg(w));
        mdl->computeGradient(w, L_grad);
        normsg0 = computeSubgradient();
        
        if (param->method_flag == 1) {
            return ista();
        }
        
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        
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
    
    int istaStep() {
        memcpy(w_prev, w, p*sizeof(double));
        for (int backtrack=0; backtrack<200; backtrack++) {
            double t = ista_size*lmd;
            double order1 = 0;
            double order2 = 0;
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
                order1 += D[i]*L_grad[i];
                order2 += D[i]*D[i];
            }
            double f_trial = mdl->computeObject(w);
            if (f_trial > obj->f + order1 + (0.5/ista_size)*order2) {
                ista_size = ista_size * 0.5;
                continue;
            }
            obj->add(f_trial, computeReg(w));
            return 0;
        }
        return 1;
    }

    /* may generalize to other regularizations beyond l1 */
    double computeReg(const double* wnew) {
        double gval = 0.0;
        for (unsigned long i = 0; i < p; i++) gval += lmd*fabs(wnew[i]);
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
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
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
};





#endif /* defined(__LHAC_v1__lhac__) */
