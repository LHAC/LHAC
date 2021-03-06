//
//  sics_lhac.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/21/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>


#include "linalg.h"
#include "timing.h"
#include "Lbfgs.h"
#include "sics_lhac.h"


unsigned long work_size;
unsigned short max_iter;
unsigned long max_inner_iter;
double* lmd;
unsigned long cd_rate;

double opt_inner_tol;
double opt_outer_tol;
/**** line search ****/
double bbeta; // 0.5
double ssigma; // 0.001
unsigned long max_linesearch_iter;

unsigned long p_sics;
unsigned long l;
unsigned long p_2;

double f_current;
double l1normW;
double logdetW;
double trSW;

int msgFlag;

/* choose suffcient decease or line search */
int sd_flag;
/* rho in sd condition */
double rho;
/* initial mu for H + muI */
double mu0=1.0;

solution* sols;



//
//static inline void shuffle( work_set_struct* work_set )
//{
//    unsigned long lens = work_set->numActive;
//    ushort_pair_t* idxs = work_set->idxs;
//    unsigned long* permut = work_set->permut;
//    
//    for (unsigned long i = 0; i < lens; i++) {
//        unsigned long j = i + rand()%(lens - i);
//        unsigned short k1 = idxs[i].i;
//        unsigned short k2 = idxs[i].j;
//        double vlt = idxs[i].vlt;
//        idxs[i].i = idxs[j].i;
//        idxs[i].j = idxs[j].j;
//        idxs[i].vlt = idxs[j].vlt;
//        idxs[j].i = k1;
//        idxs[j].j = k2;
//        idxs[j].vlt = vlt;
//        
//        /* update permutation */
//        unsigned long tmp = permut[i];
//        permut[i] = permut[j];
//        permut[j] = tmp;
//    }
//    
//    return;
//}

//
//static inline void greedySelector( double* w, double* L_grad, work_set_struct* work_set, double* normsg )
//{
//    ushort_pair_t* idxs = work_set->idxs;
//    unsigned long numActive = 0;
//    
//    double _normsg = 0.0;
//    
//    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += p_sics) {
//        for (unsigned long j = 0; j <= i; j++) {
//            double g = L_grad[k+j];
//            if (w[k+j] == 0.0 && (fabs(g) > lmd[k+j])) {
//                idxs[numActive].i = (unsigned short) i;
//                idxs[numActive].j = (unsigned short) j;
//                g = fabs(g) - lmd[k+j];
//                idxs[numActive].vlt = fabs(g);
//                numActive++;
//                _normsg += fabs(g);
//            }
//
//            if (w[k+j] != 0.0) {
//                if (w[k+j] > 0)
//                    g += lmd[k+j];
//                else
//                    g -= lmd[k+j];
//                _normsg += fabs(g);
//            }
//            
//        }
//    }
//    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
//
//    numActive = (numActive<work_size)?numActive:work_size;
//    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += p_sics) {
//        for (unsigned long j = 0; j <= i; j++) {
//            if (w[k+j] != 0) {
//                idxs[numActive].i = (unsigned short) i;
//                idxs[numActive].j = (unsigned short) j;
//                numActive++;
//            }
//        }
//    }
//    
//    *normsg = _normsg;
//    
//    work_set->numActive = numActive;
//    
//    return;
//}

static inline void stdSelector( double* w, double* L_grad, work_set_struct* work_set, double* normsg )
{
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long numActive = 0;
    
    double _normsg = 0.0;
    
    
    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += p_sics) {
        for (unsigned long j = 0; j <= i; j++) {
            double g = L_grad[k+j];
            if (w[k+j] != 0.0 || (fabs(g) > lmd[k+j])) {
                idxs[numActive].i = (unsigned short) i;
                idxs[numActive].j = (unsigned short) j;
                if (w[k+j] > 0)
                    g += lmd[k+j];
                else if (w[k+j] < 0)
                    g -= lmd[k+j];
                else
                    g = fabs(g) - lmd[k+j];
                _normsg += fabs(g);
                idxs[numActive].vlt = fabs(g);
                numActive++;
            }
        }
    }
    
    *normsg = _normsg;
    
    work_set->numActive = numActive;
    
    return;
}

static inline void computeWorkSet( double* w, double* L_grad, work_set_struct* work_set, double* normsg )
{
    stdSelector(w, L_grad, work_set, normsg);
    
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* idxs_vec_l = work_set->idxs_vec_l;
    unsigned long* idxs_vec_u = work_set->idxs_vec_u;
    unsigned long* permut = work_set->permut;
    
    for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
        unsigned long idx = idxs[ii].i;
        unsigned long jdx = idxs[ii].j;
        idxs_vec_l[ii] = jdx*p_sics + idx;
        idxs_vec_u[ii] = idx*p_sics + jdx;
        permut[ii] = ii;
    }
    
    return;
}


// Return the objective value.
static inline double DiagNewton(const double* S, const double* Lambda, const double* X,
                                const double* W, double* D)
{
    for (unsigned long ip = 0, i = 0; i < p_sics; i++, ip += p_sics) {
        for (unsigned long jp = 0, j = 0; j < i; j++, jp += p_sics) {
            unsigned long ij = ip + j;
            double a = W[ip + i]*W[jp + j];
            double ainv = 1.0/a;  // multiplication is cheaper than division
            double b = S[ij];
            double l = Lambda[ij]*ainv;
            double f = b*ainv;
            double mu;
            double x = -b*ainv;
            if (0 > f) {
                mu = -f - l;
                x -= l;
                if (mu < 0.0) {
                    mu = 0.0;
                    D[ij] = -X[ij];
                } else {
                    D[ij] += mu;
                }
            } else {
                mu = -f + l;
                if (mu > 0.0) {
                    mu = 0.0;
                    D[ij] = -X[ij];
                } else {
                    D[ij] += mu;
                }
            }
        }
    }
    logdetW = 0.0;
    l1normW = 0.0;
    trSW = 0.0;
    for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
        logdetW += log(X[k]);
        l1normW += fabs(X[k])*Lambda[k];
        trSW += X[k]*S[k];
        double a = W[k]*W[k];
        double ainv = 1.0/a;  // multiplication is cheaper than division
        double b = S[k] - W[k];
        double l = Lambda[k]*ainv;
        double c = X[k];
        double f = b*ainv;
        double mu;
        if (c > f) {
            mu = -f - l;
            if (c + mu < 0.0) {
                D[k] = -X[k];
                continue;
            }
        } else {
            mu = -f + l;
            if (c + mu > 0.0) {
                D[k] = -X[k];
                continue;
            }
        }
        D[k] += mu;
    }
    double fX = -logdetW + trSW + l1normW;
    return fX;
}

static inline void lineSearch(const double* S, double* D, double* L_grad, double* w,
                              double* L_grad_prev, double* w_prev)
{
    memcpy(w_prev, w, p_2*sizeof(double));
    memcpy(L_grad_prev, L_grad, p_2*sizeof(double));
    
    double gradD = 0.0;
    for (unsigned long i = 0, k = 0; i < p_sics ; i++, k += p_sics) {
        for (unsigned long j = 0; j < i; j++) {
            unsigned long ij = k + j;
            gradD += L_grad[ij]*D[ij];
        }
    }
    gradD *= 2.0;
    for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
        gradD += L_grad[k]*D[k];
    }
    
    double a = 1;
    double Delta = 0.0;
    double f_trial;
    unsigned long lineiter;
    for (lineiter = 1; lineiter < max_linesearch_iter; lineiter++) {
        double l1normW1 = 0.0;
        double trSW1 = 0.0;
        for (unsigned long i = 0, k = 0; i < p_sics ; i++, k += p_sics) {
            for (unsigned long j = 0; j < i; j++) {
                unsigned long ij = k + j;
                w[ij] = w_prev[ij] + a*D[ij];
                l1normW1 += fabs(w[ij])*lmd[ij];
                trSW1 += w[ij]*S[ij];
            }
        }
        l1normW1 *= 2.0;
        trSW1 *= 2.0;
        for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
            w[k] = w_prev[k] + a*D[k];
            l1normW1 += fabs(w[k])*lmd[k];
            trSW1 += w[k]*S[k];
        }
        if (a == 1) {
            Delta = gradD + l1normW1 - l1normW;
//            MSG("\t \t Delta = %+.3e\n", Delta);
        }
        
        int info = 0;
        lcdpotrf_(w, p_sics, &info);
        
        (sols->record1)++;
        if (info != 0) {
            a *= bbeta;
//            MSG(" a = %f\n", a);
            continue;
        }
        double logdetW1 = 0.0;
        for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
            logdetW1 += log(w[k]);
        }
        logdetW1 *= 2.0;
        f_trial = (trSW1 + l1normW1) - logdetW1;
        if (f_trial <= f_current + a*ssigma*Delta) {
            f_current = f_trial;
            l1normW = l1normW1;
            logdetW = logdetW1;
            trSW = trSW1;
            break;
        }
        a *= bbeta;
    }
    

    
    int info = 0;
    lcdpotri_(w, p_sics, &info);
    
    // fill in the lower triangle
    for (unsigned long i = 0; i < p_sics; i++) {
        for (unsigned long j = 0; j <= i; j++) {
            double tmp = w[i*p_sics+j];
            w[j*p_sics+i] = tmp;
        }
    }
    
    
    // w stored X-1
    // now we wrote it back to X
    for (unsigned long i = 0; i < p_sics; i++) {
        for (unsigned long j = 0; j <= i; j++) {
            unsigned long ij = i*p_sics + j;
            unsigned long ji = j*p_sics + i;
            L_grad[ij] = S[ij] - w[ij];
            L_grad[ji] = L_grad[ij];
            w[ij] = w_prev[ij] + a*D[ij];
            w[ji] = w[ij];
        }
    }
    
    return;
}

/*******************************************************************************
 compute model function value
 H = mu*H
 *******************************************************************************/
static inline double computeModelValue(double* D, double* L_grad, double* d_bar, double* w_prev,
                                       LBFGS* lR, work_set_struct* work_set, double mu)
{
    double fval = 0;
    double order1;
    double order2 = 0;
    double l1norm = 0;
    
    double* Q = lR->Q;
    const unsigned short m = lR->m; // # of cols in Q
    const double gama = lR->gama;
    double* buffer = lR->buff;
    

    /* Q in row major */
    int cblas_M = (int) work_set->numActive + (int) p_sics;
    int cblas_N = (int) m;
//    lcdgemv(CblasRowMajor, CblasNoTrans, Q, d_bar, buffer, cblas_M, cblas_N);
    lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, buffer, cblas_N, cblas_M, cblas_N);
    
    
    double vp1 = 0;
    double vp2 = 0;
    double vp = 0;
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* permut = work_set->permut;
    unsigned long* idxs_vec_u = work_set->idxs_vec_u;
    for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
        unsigned long idx = idxs[ii].i;
        unsigned long jdx = idxs[ii].j;
        unsigned long ij = idxs_vec_u[permut[ii]];
        unsigned long idx_Q = p_sics + permut[ii];
        if (idx == jdx) {
            vp1 += D[ij]*buffer[idx_Q];
        }
        else
            vp2 += D[ij]*buffer[idx_Q];
            
    }
    /* symmetric */
    vp2 *= 2.0;
    vp = vp1 + vp2;
    
    order2 = mu*gama*lcddot((int)p_2, D, 1, D, 1)-vp;
    order2 = order2*0.5;
    
    l1norm = 0.0;
    for (unsigned long i = 0, k = 0; i < p_sics ; i++, k += p_sics) {
        for (unsigned long j = 0; j < i; j++) {
            unsigned long ij = k + j;
            double wnew = w_prev[ij] + D[ij];
            l1norm += fabs(wnew)*lmd[ij];
        }
    }
    l1norm *= 2.0;
    for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
        double wnew = w_prev[k] + D[k];
        l1norm += fabs(wnew)*lmd[k];
    }
    
    order1 = lcddot((int)p_2, D, 1, L_grad, 1);
    
    fval = trSW - logdetW + order1 + order2 + l1norm;
    
    return fval;
}

/*******************************************************************************
 suffcient decrease
 *******************************************************************************/
static inline double suffcientDecrease(double* S, double* w, unsigned long iter, LBFGS* lR,
                                       double* L_grad, work_set_struct* work_set,
                                       double* d_bar, double* H_diag,
                                       double* H_diag_2, double* w_prev, double* D)
{
    int max_sd_iters = 30;
    double mu = mu0;
    double f_trial;
    double f_mdl;
    double rho_trial;
    
    
    /* timing performance */
    double cdtime = 0.0;
    double fvaltime = 0.0;
    double gvaltime = 0.0;
    double eTime = 0;
    
    memcpy(w_prev, w, p_2*sizeof(double));
    
    memset(D, 0, p_2*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    
    
    double* Q = lR->Q;
    double* Q_bar = lR->Q_bar;
    unsigned short m = lR->m;
    const double gama = lR->gama;
    
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* idxs_vec_l = work_set->idxs_vec_l;
    unsigned long* idxs_vec_u = work_set->idxs_vec_u;
    unsigned long* permut = work_set->permut;
    
    
    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += m) {
        H_diag[i] = mu0*gama;
        for (unsigned long j = 0; j < m; j++)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[k+j];
    }
    
    
    double z = 0.0;
    double Hd_j;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    unsigned long idx;
    unsigned long jdx;
    unsigned long ij;
    unsigned long ji;
    unsigned long Q_idx;
    
    for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
        idx = idxs[ii].i;
        jdx = idxs[ii].j;
        H_diag_2[ii] =  lcddot(m, &Q[idx*m], 1, &Q_bar[jdx*m], 1);
    }
    
    unsigned long max_cd_pass = 1 + iter/cd_rate;
    unsigned long cd_pass;
    int sd_iters;
    
    for (sd_iters = 0; sd_iters < max_sd_iters; sd_iters++) {
        
        double gama_scale = mu*gama;
        double dH_diag = gama_scale-mu0*gama;
        double dH_diag2 = 2*dH_diag;
        
        cdtime = CFAbsoluteTimeGetCurrent();
        for (cd_pass = 1; cd_pass <= max_cd_pass; cd_pass++) {
            double diffd = 0;
            double normd = 0;
            
            for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
                unsigned long rii = ii;
                idx = idxs[rii].i;
                jdx = idxs[rii].j;
                ij = idxs_vec_u[permut[rii]];
                ji = idxs_vec_l[permut[rii]];
                Q_idx = p_sics + permut[rii];
                unsigned long Q_idx_m = Q_idx*m;
                
                Hd_j = gama_scale*D[ij] - lcddot(m, &Q[Q_idx_m], 1, d_bar, 1);
                if (idx == jdx) {
                    G = Hd_j + L_grad[ij];
                    Gp = G + lmd[ij];
                    Gn = G - lmd[ij];
                    Hii = H_diag[idx] + dH_diag;
                }
                else {
                    G = 2*Hd_j + 2*L_grad[ij];
                    Gp = G + 2*lmd[ij];
                    Gn = G - 2*lmd[ij];
                    Hii = H_diag_2[permut[ii]];
                    Hii = Hii*(-2);
                    Hii += H_diag[idx] + H_diag[jdx] + dH_diag2;
                }
                
                wpd = w_prev[ij] + D[ij];
                Hwd = Hii * wpd;
                z = -wpd;
                if (Gp <= Hwd)
                    z = -Gp/Hii;
                if (Gn >= Hwd)
                    z = -Gn/Hii;
                
                if (idx == jdx) {
                    D[ij] = D[ij] + z;
                    for (unsigned long k = Q_idx_m, j = 0; j < m; j++, k++)
                        d_bar[j] = d_bar[j] + z*Q_bar[k];
                    normd += fabs(D[ij]);
                }
                else {
                    D[ij] = D[ij] + z;
                    D[ji] = D[ji] + z;
                    for (unsigned long k1 = Q_idx_m, j = 0; j < m; j++, k1++) {
                        d_bar[j] = d_bar[j] + 2*z*Q_bar[k1];
                    }
                    normd += 2*fabs(D[ij]);
                }
                
                diffd += fabs(z);
                
            }
            
            
            if (msgFlag >= LHAC_MSG_CD) {
                MSG("\t\t Coordinate descent pass %3ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                       cd_pass, diffd, normd);
            }

        }
        sols->cdTime += (CFAbsoluteTimeGetCurrent() - cdtime);
        
        fvaltime = CFAbsoluteTimeGetCurrent();
        
        double l1normW1 = 0.0;
        double trSW1 = 0.0;
        for (unsigned long i = 0, k = 0; i < p_sics ; i++, k += p_sics) {
            for (unsigned long j = 0; j < i; j++) {
                unsigned long ij = k + j;
                w[ij] = w_prev[ij] + D[ij];
                l1normW1 += fabs(w[ij])*lmd[ij];
                trSW1 += w[ij]*S[ij];
            }
        }
        l1normW1 *= 2.0;
        trSW1 *= 2.0;
        for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
            w[k] = w_prev[k] + D[k];
            l1normW1 += fabs(w[k])*lmd[k];
            trSW1 += w[k]*S[k];
        }
        
        if (sd_iters == 0) {
            eTime = CFAbsoluteTimeGetCurrent();
        }
        
        gvaltime = CFAbsoluteTimeGetCurrent();

        int info;
        lcdpotrf_(w, p_sics, &info);
        (sols->record1)++;
        sols->gvalTime += CFAbsoluteTimeGetCurrent() - gvaltime;
        
        sols->fvalTime += CFAbsoluteTimeGetCurrent() - fvaltime;
        
        if (info != 0) {
            mu = 2*mu;
            if (msgFlag >= LHAC_MSG_CD) {
                MSG("\t \t \t # of line searches = %3d; no PSD!; gama_scale = %f\n", sd_iters, gama_scale);
            }
            
            continue;
        }
        
        
        fvaltime = CFAbsoluteTimeGetCurrent();
        
        double logdetW1 = 0.0;
        for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
            logdetW1 += log(w[k]);
        }
        logdetW1 *= 2.0;
        
        f_trial = ((trSW1 + l1normW1) - logdetW1);
        
        f_mdl = computeModelValue(D, L_grad, d_bar, w_prev, lR, work_set, mu);
        
        sols->fvalTime += CFAbsoluteTimeGetCurrent() - fvaltime;
        
        rho_trial = (f_trial-f_current)/(f_mdl-f_current);
        
        if (msgFlag >= LHAC_MSG_NEWTON) {
            MSG("\t \t \t # of line searches = %3d; model quality: %+.3f; Delta = %+.3e\n", sd_iters, rho_trial, rho*(f_mdl - f_current));
        }
        
        if (rho_trial > rho) {
            
            if (msgFlag >= LHAC_MSG_NEWTON) {
                MSG("\t \t \t function decrease = %+.5e; mu0 = %f\n", f_current - f_trial, mu0);
            }
            
            f_current = f_trial;
            l1normW = l1normW1;
            logdetW = logdetW1;
            trSW = trSW1;
            
            gvaltime = CFAbsoluteTimeGetCurrent();

            int info;
            lcdpotri_(w, p_sics, &info);
            
            sols->gvalTime += CFAbsoluteTimeGetCurrent() - gvaltime;
            (sols->record1)++;
            
            // fill in the lower triangle
            for (unsigned long i = 0; i < p_sics; i++) {
                for (unsigned long j = 0; j <= i; j++) {
                    double tmp = w[i*p_sics+j];
                    w[j*p_sics+i] = tmp;
                }
            }
            
            // w stored X-1
            // now write it back to X
            for (unsigned long i = 0; i < p_sics; i++) {
                for (unsigned long j = 0; j <= i; j++) {
                    unsigned long ij = i*p_sics + j;
                    unsigned long ji = j*p_sics + i;
                    L_grad[ij] = S[ij] - w[ij];
                    L_grad[ji] = L_grad[ij];
                    w[ij] = w_prev[ij] + D[ij];
                    w[ji] = w[ij];
                }
            }

            break;
        }
        
        mu = 2*mu;
        
    }
    
    sols->nls += sd_iters;
    
    eTime = CFAbsoluteTimeGetCurrent() - eTime;
    
    return eTime;
    
}



static inline void coordinateDescent(double* w, unsigned long iter, LBFGS* lR,  double* L_grad,
                                     work_set_struct* work_set, double* d_bar, double* H_diag, double* H_diag_2, double* D)
{
    
    memset(D, 0, p_2*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    
    double* Q = lR->Q;
    double* Q_bar = lR->Q_bar;
    unsigned short m = lR->m;
    const double gama = lR->gama;
    
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* idxs_vec_l = work_set->idxs_vec_l;
    unsigned long* idxs_vec_u = work_set->idxs_vec_u;
    unsigned long* permut = work_set->permut;

    
    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += m) {
        H_diag[i] = gama;
        for (unsigned long j = 0; j < m; j++)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[k+j];
    }
    
    
    double z = 0.0;
    double Hd_j;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    unsigned long idx;
    unsigned long jdx;
    unsigned long ij;
    unsigned long ji;
    unsigned long Q_idx;
    
    for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
        idx = idxs[ii].i;
        jdx = idxs[ii].j;
        H_diag_2[ii] =  lcddot(m, &Q[idx*m], 1, &Q_bar[jdx*m], 1);
    }
    
    unsigned long max_inneriter;
    max_inneriter = std::min(1 + iter/3, max_inner_iter);

    for (unsigned long inneriter = 1; inneriter <= max_inneriter; inneriter++) {
        double diffd = 0;
        double normd = 0;

        for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
            idx = idxs[ii].i;
            jdx = idxs[ii].j;
            ij = idxs_vec_u[permut[ii]];
            ji = idxs_vec_l[permut[ii]];
            Q_idx = p_sics + permut[ii];
            unsigned long Q_idx_m = Q_idx*m;
            
            Hd_j = gama*D[ij] - lcddot(m, &Q[Q_idx_m], 1, d_bar, 1);
            if (idx == jdx) {
                G = Hd_j + L_grad[ij];
                Gp = G + lmd[ij];
                Gn = G - lmd[ij];
                Hii = H_diag[idx];
            }
            else {
                G = 2*Hd_j + L_grad[ij] + L_grad[ji];
                Gp = G + 2*lmd[ij];
                Gn = G - 2*lmd[ij];
                Hii = H_diag_2[permut[ii]];
                Hii = Hii*(-2);
                Hii += H_diag[idx] + H_diag[jdx];
            }
            
            
            wpd = w[ij] + D[ij];
            Hwd = Hii * wpd;
            z = -wpd;
            if (Gp <= Hwd)
                z = -Gp/Hii;
            if (Gn >= Hwd)
                z = -Gn/Hii;
            
            if (idx == jdx) {
                D[ij] = D[ij] + z;
                for (unsigned long k = Q_idx_m, j = 0; j < m; j++, k++)
                    d_bar[j] = d_bar[j] + z*Q_bar[k];
                normd += fabs(D[ij]);
            }
            else {
                D[ij] = D[ij] + z;
                D[ji] = D[ji] + z;
                for (unsigned long k1 = Q_idx_m, j = 0; j < m; j++, k1++) {
                    d_bar[j] = d_bar[j] + 2*z*Q_bar[k1];
                }
                normd += 2*fabs(D[ij]);
            }
            
            diffd += fabs(z);
        }
        
        if (msgFlag >= LHAC_MSG_CD) {
            MSG("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                   inneriter, diffd, normd);
        }
        
        if (diffd < opt_inner_tol*normd) {
            break;
        }
        
//        shuffle( work_set );
    }
    
    return;
}

solution* sics_lhac(double* S, unsigned long _p, param* prm)
{
    l = prm->l;
    opt_outer_tol = prm->opt_outer_tol;
    max_iter = prm->max_iter;
    lmd = prm->lmd;
    bbeta = prm->bbeta;
    ssigma = prm->ssigma;
    msgFlag = prm->verbose;
    work_size = prm->work_size;
    max_linesearch_iter = prm->max_linesearch_iter;
    max_inner_iter = prm->max_inner_iter;
    sd_flag = prm->sd_flag;
    rho = prm->rho;
    cd_rate = prm->cd_rate;
    
    p_sics = _p;
    p_2 = p_sics*p_sics;
    
    
    sols = new solution;
    sols->fval = new double[max_iter];
    sols->normgs = new double[max_iter];
    sols->t = new double[max_iter];
    sols->niter = new int[max_iter];
    sols->numActive = new unsigned long[max_iter];
    sols->cdTime = 0;
    sols->lbfgsTime1 = 0;
    sols->lbfgsTime2 = 0;
    sols->lsTime = 0;
    sols->size = 0;
    sols->record1 = 0;
    sols->ngval = 0;
    sols->nfval = 0;
    sols->gvalTime = 0.0;
    sols->fvalTime = 0.0;
    sols->nls = 0;
    
    
    double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
    
    double* w = new double[p_2];
    double* w_prev = new double[p_2];
    double* D = new double[p_2];
    double* L_grad = new double[p_2]; // also store the inverse of w
    double* L_grad_prev = new double[p_2];
    double* d_bar = new double[2*l];
    double* H_diag = new double[p_sics];
    double* H_diag_2 = new double[(p_sics+1)*p_sics/2]; // for computing Hii when i~=j
    
    
    // initialize to identity
    memset(w, 0, p_2*sizeof(double));
    memset(w_prev, 0, p_2*sizeof(double));
    memset(L_grad, 0, p_2*sizeof(double));
    for (unsigned long i = 0, k = 0; i < p_sics; i++, k += p_sics) {
        w[k+i] = 1;
        w_prev[k+i] = 1;
        L_grad[k+i] = 1;
    }
    
    memset(D, 0, p_2*sizeof(double));
    f_current = DiagNewton(S, lmd, w, L_grad, D);
    
    // compute gradient at identity
    memcpy(L_grad, S, p_2*sizeof(double));
    for (unsigned long i = 0, k = 0; i < p_sics; i++, k += (p_sics+1)) {
        L_grad[k] -= 1.0;
    }
    
    // compute norm1 of subgradient at identity
    double normsg0 = 0.0;
    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += p_sics) {
        for (unsigned long j = 0; j <= i; j++) {
            double g = L_grad[k+j];
            if (w[k+j] != 0.0 || (fabs(g) > lmd[k+j])) {
                if (w[k+j] > 0)
                    g += lmd[k+j];
                else if (w[k+j] < 0)
                    g -= lmd[k+j];
                else
                    g = fabs(g) - lmd[k+j];
                normsg0 += fabs(g);
            }
        }
    }
    
    lineSearch(S, D, L_grad, w, L_grad_prev, w_prev);
    
    LBFGS* lR = new LBFGS(p_2, l, prm->shrink);
    lR->initData(w, w_prev, L_grad, L_grad_prev);
    
    // active set
    work_set_struct* work_set = new work_set_struct;
    work_set->idxs = new ushort_pair_t[(p_sics+1)*p_sics/2];
    work_set->idxs_vec_l = new unsigned long[(p_sics+1)*p_sics/2];
    work_set->idxs_vec_u = new unsigned long[(p_sics+1)*p_sics/2];
    work_set->permut = new unsigned long[(p_sics+1)*p_sics/2];
    work_set->_p_sics_ = p_sics;
    
    unsigned short newton_iter;
    double normsg;
    for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
        computeWorkSet(w, L_grad, work_set, &normsg);
        

        double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
        if (msgFlag >= LHAC_MSG_MIN) {
            MSG("%.4e  iter %2d:   obj.f = %+.5e    obj.normsg = %+.4e   |work_set| = %ld\n",
                   elapsedTime, newton_iter, f_current, normsg, work_set->numActive);
        }
        
        double lbfgs1 = CFAbsoluteTimeGetCurrent();
        lR->computeLowRankApprox_v2(work_set);
        sols->lbfgsTime1 += CFAbsoluteTimeGetCurrent() - lbfgs1;
        
        if (sd_flag == 0) {
            coordinateDescent(w, newton_iter, lR, L_grad, work_set, d_bar, H_diag, H_diag_2, D);
            double eTime = CFAbsoluteTimeGetCurrent();
            lineSearch(S, D, L_grad, w, L_grad_prev, w_prev);
            eTime = CFAbsoluteTimeGetCurrent() - eTime;
            sols->lsTime += eTime;
        }
        else {
            memcpy(L_grad_prev, L_grad, p_2*sizeof(double));
            suffcientDecrease(S, w, newton_iter, lR,  L_grad, work_set, d_bar, H_diag,
                              H_diag_2, w_prev, D);
        }
        
        
        /* elapsed time */
        elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
        sols->fval[sols->size] = f_current;
        sols->normgs[sols->size] = normsg;
        sols->t[sols->size] = elapsedTime;
        sols->niter[sols->size] = newton_iter;
        sols->numActive[sols->size] = work_set->numActive;
        (sols->size)++;
        
        double lbfgs2 = CFAbsoluteTimeGetCurrent();
        lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        sols->lbfgsTime2 += CFAbsoluteTimeGetCurrent() - lbfgs2;

        
        if (normsg <= opt_outer_tol*normsg0) {
            break;
        }
    }
    
    sols->w = w;
    sols->p_sics = (int) p_sics;

    delete [] w_prev;
    delete [] D;
    delete [] L_grad;
    delete [] L_grad_prev;
    delete [] d_bar;
    delete [] H_diag;
    delete [] H_diag_2;
    delete [] work_set->idxs;
    delete [] work_set->idxs_vec_l;
    delete [] work_set->idxs_vec_u;
    delete work_set;
    delete lR;
    
    return sols;
}
    










