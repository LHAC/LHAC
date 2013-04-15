//
//  l1log.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/27/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "l1log.h"
#include "lhac.h"
#include "myUtilities.h"

#include <vecLib/clapack.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>


static inline void shuffle( work_set_struct* work_set )
{
    unsigned long lens = work_set->numActive;
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* permut = work_set->permut;       
    
    for (unsigned long i = 0; i < lens; i++) {
        unsigned long j = i + rand()%(lens - i);
        unsigned long k1 = idxs[i].i;
        unsigned long k2 = idxs[i].j;
        double vlt = idxs[i].vlt;
        idxs[i].i = idxs[j].i;
        idxs[i].j = idxs[j].j;
        idxs[i].vlt = idxs[j].vlt;
        idxs[j].i = k1;
        idxs[j].j = k2;
        idxs[j].vlt = vlt;
        
        /* update permutation */
        k1 = permut[i];
        permut[i] = permut[j];
        permut[j] = k1;
    }
    
    return;
}


/********* sics version *********/
/*
double l1log::getSubgradient( double* w, double* L_grad, double* lmd )
{
    
    double subgrad = 0.0;
    
    for (unsigned long k = 0, i = 0; i < p; i++, k += p) {
        for (unsigned long j = 0; j <= i; j++) {
            double g = L_grad[k+j];
            if (w[k+j] != 0.0 || (fabs(g) > lmd[k+j])) {
                if (w[k+j] > 0)
                    g += lmd[k+j];
                else if (w[k+j] < 0)
                    g -= lmd[k+j];
                else
                    g = fabs(g) - lmd[k+j];
                subgrad += fabs(g);
            }
        }
    }
    
    return subgrad;
}

void getWorkSet( double* w, double* L_grad, double* lmd, work_set_struct* work_set )
{
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long numActive = 0;
    
    for (unsigned long k = 0, i = 0; i < p; i++, k += p) {
        for (unsigned long j = 0; j <= i; j++) {
            double g = L_grad[k+j];
            if (w[k+j] == 0.0 && (fabs(g) > lmd[k+j])) {
                idxs[numActive].i = (unsigned short) i;
                idxs[numActive].j = (unsigned short) j;
                if (w[k+j] > 0)
                    g += lmd[k+j];
                else if (w[k+j] < 0)
                    g -= lmd[k+j];
                else
                    g = fabs(g) - lmd[k+j];
                idxs[numActive].vlt = fabs(g);
                numActive++;
            }
        }
    }
    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
    
    numActive = (numActive<work_size)?numActive:work_size;
    for (unsigned long k = 0, i = 0; i < p; i++, k += p) {
        for (unsigned long j = 0; j <= i; j++) {
            if (w[k+j] != 0) {
                idxs[numActive].i = (unsigned short) i;
                idxs[numActive].j = (unsigned short) j;
                numActive++;
            }
        }
    }
    
    work_set->numActive = numActive;
    
    return;
}
 */

void l1log::initData(training_set_sp* Dset)
{
    unsigned long dim_x_space = Dset->nnz+Dset->p;
    
    p = Dset->p;
    N = Dset->N;
    y = new double[N];
    X_libsvm = new feature_node*[p];
    x_space = new feature_node[dim_x_space];
    
    memcpy(X_libsvm, Dset->X, p*sizeof(feature_node*));
    memcpy(x_space, Dset->x_space, (dim_x_space)*sizeof(feature_node));
    memcpy(y, Dset->y, N*sizeof(double));
    
    return;
}

void l1log::initData(training_set* Dset)
{
    p = Dset->p;
    N = Dset->N;
    
    X = new double[p*N];
    y = new double[N];
    
    memcpy(X, Dset->X, sizeof(double)*p*N);
    memcpy(y, Dset->y, sizeof(double)*N);
    
    return;
}

void l1log::initVars()
{
    MSG = param->verbose;
    
    w_prev = new double[p];
    w = new double[p];
    L_grad_prev = new double[p];
    L_grad = new double[p];
    D = new double[p];
    H_diag = new double[p]; // p
    d_bar = new double[2*param->l]; // 2*l
    e_ywx = new double[N]; // N
    B = new double[N]; // N
    /* libsvm format */
    Xd = new double[N]; // N
    
    /* initiate */
    memset(w, 0, p*sizeof(double));
    memset(w_prev, 0, p*sizeof(double));
    memset(D, 0, p*sizeof(double));
    /* libsvm format */
    memset(Xd, 0, N*sizeof(double));
    for (unsigned long i = 0; i < N; i++) {
        e_ywx[i] = 1;
    }
    
    timeBegin = clock();
    
    switch (mode) {
        case GENERAL:
            f_current = computeObject(w);
            break;
            
        case LIBSVM:
            f_current = computeObject(w,1.0);
            break;
            
        default:
            break;
    }
    
    computeGradient();
//    printout(" L_grad = ", L_grad, p, ROW_VIEW);
    normsg0 = computeSubgradient();
    memcpy(L_grad_prev, L_grad, p*sizeof(double));
    
    // ista step
    double lmd = param->lmd;
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
        
        /* libsvm format */
        if (mode == LIBSVM) {
            feature_node* xnode = X_libsvm[idx];
            while (xnode->index != -1) {
                int ind = xnode->index-1;
                Xd[ind] += z*(xnode->value);
                xnode++;
            }
        }
    }
    
    lineSearch();
    
    memcpy(L_grad_prev, L_grad, p*sizeof(double));
    
    computeGradient();
    
    return;
}

void l1log::init(training_set* Dset, l1log_param* _param)
{
    param = new l1log_param;
    memcpy(param, _param, sizeof(l1log_param));
    
    mode = GENERAL;
    
    initData(Dset);
    
    initVars();
    
    return;
}

void l1log::init(training_set_sp *Dset, l1log_param *_param)
{
    param = new l1log_param;
    memcpy(param, _param, sizeof(l1log_param));
    
    mode = LIBSVM;
    
    initData(Dset);
    
    initVars();
    
    return;
}

l1log::l1log(training_set* Dset, l1log_param* _param)
{
    init(Dset, _param);
}

l1log::l1log(training_set_sp* Dset, l1log_param* _param)
{
    
    init(Dset, _param);
}

l1log::l1log(training_set_sp* Dset)
{
    /* set parameter */
    l1log_param* _param = new l1log_param;
    _param->l = 8;
    _param->work_size = 10000;
    _param->max_iter = 500;
    _param->max_inner_iter = 20;
    _param->lmd = 0.5;
    _param->opt_inner_tol = 0.05;
    _param->opt_outer_tol = 1e-6;
    _param->max_linesearch_iter = 1000;
    _param->bbeta = 0.5;
    _param->ssigma = 0.001;
    
    init(Dset, _param);
    
    delete _param;
}


l1log::l1log(training_set* Dset)
{
    
    /* set parameter */
    l1log_param* _param = new l1log_param;
    _param->l = 8;
    _param->work_size = 10000;
    _param->max_iter = 500;
    _param->max_inner_iter = 20;
    _param->lmd = 0.5;
    _param->opt_inner_tol = 0.05;
    _param->opt_outer_tol = 1e-5;
    _param->max_linesearch_iter = 1000;
    _param->bbeta = 0.5;
    _param->ssigma = 0.001;
    
    init(Dset, _param);
    
    delete _param;
    
}



l1log::~l1log()
{
    switch (mode) {
        case GENERAL:
            delete [] X;
            break;
        case LIBSVM:
            delete [] X_libsvm;
            delete [] x_space;
            delete [] Xd;
        default:
            break;
    }
    delete [] y;
    delete [] w_prev;
    delete [] w;
    delete [] L_grad;
    delete [] L_grad_prev;
    delete [] D;
    delete [] H_diag;
    delete [] d_bar;
    delete [] e_ywx;
    delete [] B;
    delete param;
}

void l1log::computeWorkSet( work_set_struct* &work_set )
{
    ushort_pair_t* &idxs = work_set->idxs;
    unsigned long numActive = 0;
    
    double lmd = param->lmd;
    unsigned long work_size = param->work_size;
    
    /*** select rule 1 ***/
    for (unsigned long j = 0; j < p; j++) {
        double g = L_grad[j];
        if (w[j] == 0.0 && (fabs(g) > lmd)) {
            idxs[numActive].i = (unsigned short) j;
            idxs[numActive].j = (unsigned short) j;
            g = fabs(g) - lmd;
            idxs[numActive].vlt = fabs(g);
            numActive++;
        }
    }
    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
    
    numActive = (numActive<work_size)?numActive:work_size;
    for (unsigned long j = 0; j < p; j++) {
        if (w[j] != 0) {
            idxs[numActive].i = j;
            idxs[numActive].j = j;
            numActive++;
        }
    }
    
    work_set->numActive = numActive;
    
    
    /*** select rule 2 ***/
//    for (unsigned long j = 0; j < p; j++) {
//        double g = L_grad[j];
//        if (w[j] != 0.0 || (fabs(g) > lmd)) {
//            idxs[numActive].i = (unsigned short) j;
//            idxs[numActive].j = (unsigned short) j;
//            g = fabs(g) - lmd;
//            idxs[numActive].vlt = fabs(g);
//            numActive++;
//        }
//    }
//    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
//    
//    work_set->numActive = numActive;

    
    
    
    /*** select rule 3 ***/
//    for (unsigned long j = 0; j < p; j++) {
//        idxs[numActive].i = (unsigned short) j;
//        idxs[numActive].j = (unsigned short) j;
//        numActive++;
//    }
//    
//    work_set->numActive = numActive;
    
    /* reset permutation */
    for (unsigned long j = 0; j < numActive; j++) {
        work_set->permut[j] = j;
    }
    return;
}

double l1log::computeSubgradient()
{
    
    double subgrad = 0.0;
    
    double lmd = param->lmd;
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

double l1log::computeObject(double* wnew)
{
    double fval = 0.0;
    
//    memset(e_ywx, 0, N*sizeof(double));
    
    double lmd = param->lmd;
    
    double alpha = 1.0;
    double beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, (int)N, (int)p, alpha, X, (int)N, wnew, 1, beta, e_ywx, 1);
    for (unsigned long i = 0; i < N; i++) {
//        for (unsigned long j = 0, k = 0; j < p; j++, k += N) {
//            e_ywx[i] += X[k+i]*wnew[j];
//        }
        e_ywx[i] = exp(e_ywx[i]*y[i]);
        fval += log((1+e_ywx[i])/e_ywx[i]);
    }
    
    for (unsigned long i = 0; i < p; i++) {
        fval += lmd*fabs(wnew[i]);
    }
    
    return fval;
}

double l1log::computeObject(double* wnew, double _a)
{
    double fval = 0.0;
    
    double lmd = param->lmd;
    for (unsigned long i = 0; i < N ; i++) {
        e_ywx[i] = e_ywx[i]*exp(y[i]*_a*Xd[i]);
        fval += log((1+e_ywx[i])/e_ywx[i]);
    }
    
    for (unsigned long i = 0; i < p; i++) {
        fval += lmd*fabs(wnew[i]);
    }
    
    return fval;
}

void l1log::computeGradient()
{
    for (unsigned long i = 0; i < N; i++) {
        B[i] = -y[i]/(1+e_ywx[i]);
    }
//    double alpha = 1.0;
//    double beta = 0.0;
    
    switch (mode) {
        case GENERAL:
            for (unsigned long i = 0, k = 0; i < p; i++, k += N) {
                double gi = 0;
//                for (unsigned long j = 0; j < N; j++) {
//                    gi += B[j]*X[k+j];
//                }
                gi = cblas_ddot((int)N, B, 1, &X[k], 1);
//                printf("%.5e\n", gi-gi2);
                L_grad[i] = gi;
            }
//            cblas_dgemv(CblasColMajor, CblasTrans, (int)N, (int)p, alpha, X, (int)N, B, 1, beta, L_grad, 1);
            break;
        
        case LIBSVM:
            for (unsigned long i = 0; i < p; i++) {
                double gi = 0;
                feature_node* xnode = X_libsvm[i];
                while (xnode->index != -1) {
                    int ind = xnode->index - 1;
                    gi += B[ind]*xnode->value;
                    xnode++;
                }
                L_grad[i] = gi;
            }
            break;
            
        default:
            break;
    }
    
    return;
}

/*
    INPUT:  w,D,L_grad,f_current
    OUTPUT: w,w_prev,f_current
 */
void l1log::lineSearch()
{
    double a = 1;
    
    double l1_current = 0.0;
    double l1_next = 0.0;
    double f1;
    double delta = 0.0;
    
    double lmd = param->lmd;
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
    
//    printout(w, p);
//    printout(D, p);
    
    delta += l1_next - l1_current;
    
//    printout("w = ",w, p);
    unsigned long lineiter;
    switch (mode) {
        case GENERAL:
            f1 = computeObject(w);
            for (lineiter = 0; lineiter < max_linesearch_iter; lineiter++) {
                if (f1 < f_current + a*ssigma*delta) {
                    f_current = f1;
                    break;
                }
                
                a = bbeta*a;
                
                for (unsigned long i = 0; i < p; i++) {
                    w[i] = w_prev[i] + a*D[i];
                }
                
                f1 = computeObject(w);
            }
            break;
            
        case LIBSVM:
            f1 = 0.0;
            for (unsigned long i = 0; i < N ; i++) {
                double e_ywx_;
                e_ywx_ = e_ywx[i]*exp(y[i]*Xd[i]);
                f1 += log((1+e_ywx_)/e_ywx_);
            }
            
            for (unsigned long i = 0; i < p; i++) {
                f1 += lmd*fabs(w[i]);
            }

            for (lineiter = 0; lineiter < max_linesearch_iter; lineiter++) {
                if (f1 < f_current + a*ssigma*delta) {
                    f_current = f1;
                    for (unsigned long i = 0; i < N ; i++) {
                       e_ywx[i] = e_ywx[i]*exp(y[i]*a*Xd[i]);
                    }
                    break;
                }
                
                a = bbeta*a;
                
                for (unsigned long i = 0; i < p; i++) {
                    w[i] = w_prev[i] + a*D[i];
                }
                
                f1 = 0.0;
                for (unsigned long i = 0; i < N ; i++) {
                    double e_ywx_;
                    e_ywx_ = e_ywx[i]*exp(y[i]*a*Xd[i]);
                    f1 += log((1+e_ywx_)/e_ywx_);
                }
                
                for (unsigned long i = 0; i < p; i++) {
                    f1 += lmd*fabs(w[i]);
                }
            }
            break;
            
        default:
            break;
    }

    
    if (MSG >= LHAC_MSG_LINE) {
        printf("\t\t\t Line search step size = %+.4e   Trial step = %ld\n", a, lineiter);
    }
    
//    printout("w_prev = ", w_prev, p);
//    printout("w = ",w, p);
    
    return;
}

void l1log::coordinateDsecent(LBFGS* lR, work_set_struct* work_set)
{
    /* libsvm format */
    if (mode == LIBSVM) {
        memset(Xd, 0, N*sizeof(double));
    }
    
    double lmd = param->lmd;
    unsigned long l = param->l;
    double opt_inner_tol = param->opt_inner_tol;
    
    memset(D, 0, p*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    
    const double* Q = lR->Q;
    const double* Q_bar = lR->Q_bar;
    const unsigned short m = lR->m;
    const double gama = lR->gama;
    
    // Hessian diagonal: H_diag = gama - sum(Q'.*Q_bar);
    
    for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
        H_diag[i] = gama;
        for (unsigned long j = 0, o = 0; j < m; j++, o += work_set->numActive)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[o+i];
    }
//    lR->computeHDiag(H_diag);
    
    //    printout("Q_bar = ", Q_bar, m, p, FULL);
    //    printout("Q = ", Q, p, (unsigned long) m, FULL);
    //    printout("w = ", w, p, FULL);
    //    printout("L_grad = ", L_grad, p, FULL);
    //    printout("H_diag = ", H_diag, p, FULL);
    
    double z = 0.0;
    double Hd_j;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    
    unsigned long max_inneriter;
    max_inneriter = std::min(1 + iter/3, param->max_inner_iter);
    //    if (max_inneriter > (param->max_inner_iter)) {
    //        max_inneriter = param->max_inner_iter;
    //    }
    unsigned long* permut = work_set->permut;
    for (unsigned long inneriter = 1; inneriter <= max_inneriter; inneriter++) {
        double diffd = 0;
        double normd = 0;
        
        //        printout(work_set);
        
        ushort_pair_t* idxs = work_set->idxs;
        for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
            unsigned long idx = idxs[ii].j;
            unsigned long idx_Q = permut[ii];
//            unsigned long idx_Q = idx;
            
            //            printout("d_bar = ", d_bar, m, FULL);
//            Hd_j = gama*D[idx] - cblas_ddot(m, &Q[idx], (int)p, d_bar, 1);
            Hd_j = gama*D[idx] - cblas_ddot(m, &Q[idx_Q], (int)work_set->numActive, d_bar, 1);
//            Hd_j = lR->computeHdj(D[idx], d_bar, idx);
            //            Hd_j = gama*D[idx];
            //            for (unsigned long k = 0, j = 0; j < m; j++, k+=p)
            //                Hd_j = Hd_j - Q[k+idx]*d_bar[j];
            
            Hii = H_diag[idx_Q];
            G = Hd_j + L_grad[idx];
            Gp = G + lmd;
            Gn = G - lmd;
            wpd = w[idx] + D[idx];
            Hwd = Hii * wpd;
            
            z = -wpd;
            if (Gp <= Hwd)
                z = -Gp/Hii;
            if (Gn >= Hwd)
                z = -Gn/Hii;
            
            D[idx] = D[idx] + z;
            
//            lR->updateDbar(d_bar, idx, z);
            for (unsigned long k = idx_Q*m, j = 0; j < m; j++)
//            for (unsigned long k = idx*m, j = 0; j < m; j++)
                d_bar[j] = d_bar[j] + z*Q_bar[k+j];
            
            /* libsvm format */
            if (mode == LIBSVM) {
                feature_node* xnode = X_libsvm[idx];
                while (xnode->index != -1) {
                    int ind = xnode->index-1;
                    Xd[ind] += z*(xnode->value);
                    xnode++;
                }
            }
            
            diffd += fabs(z);
            normd += fabs(D[idx]);
            
        }
        
        if (MSG >= LHAC_MSG_CD) {
            printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                   inneriter, diffd, normd);
        }
        
        if (diffd < opt_inner_tol*normd) {
            break;
        }
        
        shuffle( work_set );
    }
    
    //    printout(D, p);
    
    return;
    
}

// Q: P X m
void l1log::coordinateDescent(double* Q_bar, double* Q,
                       double gama, work_set_struct* work_set,
                       unsigned short m)
{
    /* libsvm format */
    if (mode == LIBSVM) {
        memset(Xd, 0, N*sizeof(double));
    }
    
    double lmd = param->lmd;
    unsigned long l = param->l;
    double opt_inner_tol = param->opt_inner_tol;
    
    memset(D, 0, p*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    
    // Hessian diagonal: H_diag = gama - sum(Q'.*Q_bar);
    
    for (unsigned long k = 0, i = 0; i < p; i++, k += m) {
        H_diag[i] = gama;
        for (unsigned long j = 0, o = 0; j < m; j++, o += p)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[o+i];
    }
    
//    printout("Q_bar = ", Q_bar, m, p, FULL);
//    printout("Q = ", Q, p, (unsigned long) m, FULL);
//    printout("w = ", w, p, FULL);
//    printout("L_grad = ", L_grad, p, FULL);
//    printout("H_diag = ", H_diag, p, FULL);

    double z = 0.0;
    double Hd_j;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    
    unsigned long max_inneriter;
    max_inneriter = std::min(1 + iter/3, param->max_inner_iter);
//    if (max_inneriter > (param->max_inner_iter)) {
//        max_inneriter = param->max_inner_iter;
//    }
    for (unsigned long inneriter = 1; inneriter <= max_inneriter; inneriter++) {
        double diffd = 0;
        double normd = 0;
        
//        printout(work_set);
        
        ushort_pair_t* idxs = work_set->idxs;
        for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
            unsigned short idx = idxs[ii].j;
            
//            printout("d_bar = ", d_bar, m, FULL);
            Hd_j = gama*D[idx] - cblas_ddot(m, &Q[idx], (int)p, d_bar, 1);
//            Hd_j = gama*D[idx];
//            for (unsigned long k = 0, j = 0; j < m; j++, k+=p)
//                Hd_j = Hd_j - Q[k+idx]*d_bar[j];
            
            Hii = H_diag[idx];
            G = Hd_j + L_grad[idx];
            Gp = G + lmd;
            Gn = G - lmd;
            wpd = w[idx] + D[idx];
            Hwd = Hii * wpd;
            
            z = -wpd;
            if (Gp <= Hwd)
                z = -Gp/Hii;
            if (Gn >= Hwd)
                z = -Gn/Hii;
            
            D[idx] = D[idx] + z;
            
            for (unsigned long k = idx*m, j = 0; j < m; j++)
                d_bar[j] = d_bar[j] + z*Q_bar[k+j];
            
            /* libsvm format */
            if (mode == LIBSVM) {
                feature_node* xnode = X_libsvm[idx];
                while (xnode->index != -1) {
                    int ind = xnode->index-1;
                    Xd[ind] += z*(xnode->value);
                    xnode++;
                }
            }
            
            diffd += fabs(z);
            normd += fabs(D[idx]);
            
        }
        
        if (MSG >= LHAC_MSG_CD) {
            printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                   inneriter, diffd, normd);
        }
        
        if (diffd < opt_inner_tol*normd) {
            break;
        }
        
        shuffle( work_set );
    }
    
//    printout(D, p);
    
    return;
}