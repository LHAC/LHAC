//
//  l1log.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/27/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "lhac.h"
#include "myUtilities.h"

#include <CoreFoundation/CoreFoundation.h>

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
    L_prob = new double[p];
    
    /* initiate */
    memset(w, 0, p*sizeof(double));
    memset(w_prev, 0, p*sizeof(double));
    memset(D, 0, p*sizeof(double));

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
    
    /* libsvm format */
    Xd = new double[N]; // N
    memset(Xd, 0, N*sizeof(double));
    
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
    switch (param->active_set) {
        case GREEDY:
            greedySelector(work_set);
            break;
            
        case STD:
            stdSelector(work_set);
            break;
            
        default:
            stdSelector(work_set);
            break;
    }
    
    
    /* reset permutation */
    for (unsigned long j = 0; j < work_set->numActive; j++) {
        work_set->permut[j] = j;
    }
    return;
}

void l1log::stdSelector( work_set_struct* &work_set )
{
    ushort_pair_t* &idxs = work_set->idxs;
    unsigned long numActive = 0;
    
    double lmd = param->lmd;
//    unsigned long work_size = param->work_size;
    
    /*** select rule 2 ***/
    for (unsigned long j = 0; j < p; j++) {
        double g = L_grad[j];
        if (w[j] != 0.0 || (fabs(g) > lmd)) {
            idxs[numActive].i = (unsigned short) j;
            idxs[numActive].j = (unsigned short) j;
//            g = fabs(g) - lmd;
//            idxs[numActive].vlt = fabs(g);
            numActive++;
        }
    }
//    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
//    numActive = (numActive<work_size)?numActive:work_size;
    
    work_set->numActive = numActive;
    
    return;
}

//void l1log::greedySelector( work_set_struct* &work_set )
//{
//    ushort_pair_t* &idxs = work_set->idxs;
//    unsigned long numActive = 0;
//    
//    double lmd = param->lmd;
//    unsigned long work_size = param->work_size;
//    
//    /*** select rule 1 ***/
//    for (unsigned long j = 0; j < p; j++) {
//        double g = L_grad[j];
//        if (w[j] == 0.0 && (fabs(g) > lmd)) {
//            idxs[numActive].i = (unsigned short) j;
//            idxs[numActive].j = (unsigned short) j;
//            g = fabs(g) - lmd;
//            idxs[numActive].vlt = fabs(g);
//            numActive++;
//        }
//    }
//    
//
//    
//    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
//
//    unsigned long nnz = 0;
//    
////    numActive = (numActive<work_size)?numActive:work_size;
//    numActive = (numActive<work_size)?numActive:numActive / work_size;
////    printf("numActive = %ld\n", numActive);
//    for (unsigned long j = 0; j < p; j++) {
//        if (w[j] != 0) {
//            idxs[numActive].i = j;
//            idxs[numActive].j = j;
//            numActive++;
//            nnz++;
//        }
//    }
//    
////    printf("nnz = %ld\n", nnz);
//
//    work_set->numActive = numActive;
//    
//    return;
//}

void l1log::greedySelector( work_set_struct* &work_set )
{
    ushort_pair_t* &idxs = work_set->idxs;
    unsigned long numActive = 0;
    
    double lmd = param->lmd;
    unsigned long work_size = param->work_size;
    
    unsigned long zeroActive = 0;
    
    for (unsigned long j = 0; j < p; j++) {
        double g = L_grad[j];
        if (w[j] != 0.0 || (fabs(g) > lmd)) {
            idxs[numActive].i = (unsigned short) j;
            idxs[numActive].j = (unsigned short) j;
            g = fabs(g) - lmd;
            idxs[numActive].vlt = fabs(g);
            numActive++;
            
            if (w[j] == 0.0) {
                zeroActive++;
            }
        }
    }
    
    
    qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), cmp_by_vlt);
    
    //    numActive = (numActive<work_size)?numActive:work_size;
    
    // zerosActive small means found the nonzeros subspace
    numActive = (zeroActive<10)?numActive:numActive / work_size;
    printf("zero active = %ld\n", zeroActive);
    printf("num active = %ld\n", numActive);
    
    work_set->numActive = numActive;
    
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

double l1log::computeObject()
{
    double fval = 0.0;
    
    
    for (unsigned long i = 0; i < p; i++) {
        w[i] = w_prev[i] + D[i];
    }
    
    double lmd = param->lmd;
    
    double alpha = 1.0;
    double beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, (int)N, (int)p, alpha, X, (int)N, w, 1, beta, e_ywx, 1);
    for (unsigned long i = 0; i < N; i++) {
        double nc1;
        double nc2;
        nc1 = e_ywx[i]*y[i];
        e_ywx[i] = exp(nc1);
        if (nc1 <= 0) {
            nc2 = e_ywx[i];
            fval += log((1+nc2))-nc1;
        }
        else {
            nc2 = exp(-nc1);
            fval += log((1+nc2));
        }

    }
    
    for (unsigned long i = 0; i < p; i++) {
        fval += lmd*fabs(w[i]);
    }
    
    return fval;
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
        double nc1;
        double nc2;
        nc1 = e_ywx[i]*y[i];
        e_ywx[i] = exp(nc1);
        if (nc1 <= 0) {
            nc2 = e_ywx[i];
            fval += log((1+nc2))-nc1;
        }
        else {
            nc2 = exp(-nc1);
            fval += log((1+nc2));
        }
//        e_ywx[i] = exp(e_ywx[i]*y[i]);
//        fval += log((1+e_ywx[i])/e_ywx[i]);
    }
//    printf("fval1 - fval = %.f\n", fval1-fval);
    
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
    
    switch (mode) {
        case GENERAL:
//            for (unsigned long i = 0, k = 0; i < p; i++, k += N) {
//                double gi = 0;
////                for (unsigned long j = 0; j < N; j++) {
////                    gi += B[j]*X[k+j];
////                }
//                gi = cblas_ddot((int)N, B, 1, &X[k], 1);
////                printf("%.5e\n", gi-gi2);
//                L_grad[i] = gi;
//            }
            cblas_dgemv(CblasColMajor, CblasTrans, (int)N, (int)p, 1.0, X, (int)N, B, 1, 0.0, L_grad, 1);
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
    int lineiter;
    static int iter_counter = 0;
    switch (mode) {
        case GENERAL:
            f1 = computeObject(w);
            for (lineiter = 0; lineiter < max_linesearch_iter; lineiter++) {
                if (f1 < f_current + a*ssigma*delta) {
                    f_current = f1;
                    printf(" # of line searches = %d\n", iter_counter);
                    break;
                }
                iter_counter++;
                
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
        printf("\t\t\t Line search step size = %+.4e   Trial step = %d\n", a, lineiter);
    }
    
//    printout("w_prev = ", w_prev, p);
//    printout("w = ",w, p);
    
    return;
}

/*******************************************************************************
 compute model function value
 H = mu*H
 *******************************************************************************/
double l1log::computeModelValue(LBFGS* lR, work_set_struct* work_set, double mu)
{
    double fval = 0;
    double order1 = cblas_ddot((int)p, D, 1, L_grad, 1);
    double order2 = 0;
    double l1norm = 0;
    
    double lmd = param->lmd;
    
    double* Q = lR->Q;
    const unsigned short m = lR->m; // # of cols in Q
    const double gama = lR->gama;
    double* buffer = lR->buff;
    
    int cblas_M = (int) work_set->numActive;
    int cblas_N = (int) m;

    cblas_dgemv(CblasRowMajor, CblasNoTrans, cblas_M, cblas_N, 1.0, Q, cblas_N, d_bar, 1, 0.0, buffer, 1);

    
    double vp = 0;
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* permut = work_set->permut;
    for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
        unsigned long idx = idxs[ii].j;
        unsigned long idx_Q = permut[ii];
        vp += D[idx]*buffer[idx_Q];
    }
    
    order2 = mu*gama*cblas_ddot((int)p, D, 1, D, 1)-vp;
    order2 = order2*0.5;
    
    for (unsigned long i = 0; i < p; i++) {
        l1norm += lmd*(fabs(w_prev[i] + D[i]) - fabs(w_prev[i]));
    }
    
    fval = f_current + order1 + order2 + l1norm;
    
    return fval;
}

bool returnCoordinate(double key, double* kvs, unsigned long range, unsigned long* c) {
    
    
    unsigned long first = 0;
    unsigned long last = range-1;
    unsigned long mid = floor((last + first) / 2);
    
    /* boundary condition */
    if (key < kvs[first]) {
        *c = first;
        return true;
    }
    // key == 1, which is not likely but still possible
    else if (key == kvs[last]) {
        *c = last;
        return true;
    }
    
    while (1) {
        if (last - first <= 1) {
            *c = first;
            return true;
        }
        
        double midKey = kvs[mid];
        if (midKey == key) {
            *c = mid;
            return true;
        }
        else if (midKey > key) {
            last = mid;
        }
        else {
            first = mid;
        }
        
        mid = floor((last + first) / 2);
        
    }
}

unsigned long l1log::randomCoordinateSelector(unsigned long range)
{
    /* uniform random number in [0,1]  */
    double prob = ((double) rand() / (RAND_MAX));
    unsigned long rii = 0;
    
    if (!returnCoordinate(prob, L_prob, range, &rii)) {
        printf("!!!!!!!!!!!!!!!!!!!!\n");
    }
    
    return rii;
    
}

/*******************************************************************************
 suffcient decrease
 *******************************************************************************/
double l1log::suffcientDecrease(LBFGS* lR, work_set_struct* work_set, double mu0, solution* sols)
{
    int max_sd_iters = 20;
    double mu = mu0;
    double rho = param->rho;
    
    double z = 0.0;
    double Hd_j;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    double Qd_bar;
    
//    double d1,d2,d3,d4,d5,d3_,z_square,d6,redc;
    
    double f_trial;
    double f_mdl;
    double rho_trial;
    
    double eTime=0;
    
    memcpy(w_prev, w, p*sizeof(double));
    
    /* libsvm format */
    if (mode == LIBSVM) {
        memset(Xd, 0, N*sizeof(double));
    }
    double lmd = param->lmd;
    unsigned long l = param->l;
    
    double* Q = lR->Q;
    double* Q_bar = lR->Q_bar;
    const unsigned short m = lR->m;
    const double gama = lR->gama;
    
    memset(D, 0, p*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    
    double diag_sum = 0.0;
    for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
        H_diag[i] = mu0*gama;
        for (unsigned long j = 0; j < m; j++)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[k+j];
        
        diag_sum += H_diag[i];
    }
    
    /* cumulative probability */
//    L_prob[0] = H_diag[0] / diag_sum;
//    for (unsigned long i = 1; i < work_set->numActive; i++) {
//        L_prob[i] = H_diag[i] / diag_sum + L_prob[i-1];
//    }
    

    /* mdl value change */
    dQ = 0;
    
//    unsigned long max_cd_pass = std::min(1 + iter/3, param->max_inner_iter);
    unsigned long max_cd_pass = 1 + iter / param->cd_rate;
//    unsigned long max_cd_pass = param->max_inner_iter;
    unsigned long* permut = work_set->permut;
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long cd_pass;
    int sd_iters;
    
    for (sd_iters = 0; sd_iters < max_sd_iters; sd_iters++) {
        
        
        double gama_scale = mu*gama;
        double dH_diag = gama_scale-mu0*gama;
        
        double cdtime = CFAbsoluteTimeGetCurrent();
        for (cd_pass = 1; cd_pass <= max_cd_pass; cd_pass++) {
            double diffd = 0;
            double normd = 0;
            
            for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
//                unsigned long rii = rand()%(work_set->numActive);
//                unsigned long rii = randomCoordinateSelector(work_set->numActive);
                unsigned long rii = ii;
//                printf("%f, %f, %f\n", L_prob[rii], L_prob[rii-1], L_prob[rii+1]);
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
                printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e   Change in Q = %+.4e\n",
                       cd_pass, diffd, normd, dQ);
            }
            
//            shuffle( work_set );
        }
        
        sols->cdTime += CFAbsoluteTimeGetCurrent() - cdtime;
        
        /* add accelerated step */
//        tk1 = (1 + sqrt(1 + 4*tk*tk))/2;
//        double fs = (1 + (tk-1)/tk1);
//        for (unsigned long i = 0; i < work_set->numActive; i++) {
//            unsigned long idx = idxs[i].j;
//            D[idx] = fs * D[idx];
//        }
        
        if (sd_iters == 0) {
            eTime = CFAbsoluteTimeGetCurrent();
        }
        
        double fvaltime = CFAbsoluteTimeGetCurrent();
        f_trial = computeObject();
        f_mdl = computeModelValue(lR, work_set, mu);
        sols->fvalTime += CFAbsoluteTimeGetCurrent() - fvaltime;
        rho_trial = (f_trial-f_current)/(f_mdl-f_current);
        
        printf("\t \t \t # of line searches = %3d; model quality: %+.3f\n", sd_iters, rho_trial);

        
        if (rho_trial > rho) {
            f_current = f_trial;
            break;
        }
        mu = 2*mu;
        
    }
    sols->nls += sd_iters;
    
    eTime = CFAbsoluteTimeGetCurrent() - eTime;
    
    return eTime;
    
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
    
//    max_inneriter = std::min(1 + iter/3, param->max_inner_iter);
    unsigned long max_cd_pass = 1 + iter / param->cd_rate;
    //    if (max_inneriter > (param->max_inner_iter)) {
    //        max_inneriter = param->max_inner_iter;
    //    }
    unsigned long* permut = work_set->permut;
    for (unsigned long inneriter = 1; inneriter <= max_cd_pass; inneriter++) {
        double diffd = 0;
        double normd = 0;
        
        //        printout(work_set);
        
        ushort_pair_t* idxs = work_set->idxs;
        for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
            unsigned long idx = idxs[ii].j;
            unsigned long idx_Q = permut[ii];
            unsigned long Q_idx_m = idx_Q*m;
//            unsigned long idx_Q = idx;
            
            //            printout("d_bar = ", d_bar, m, FULL);
//            Hd_j = gama*D[idx] - cblas_ddot(m, &Q[idx], (int)p, d_bar, 1);
//            Hd_j = gama*D[idx] - cblas_ddot(m, &Q[idx_Q], (int)work_set->numActive, d_bar, 1);
            Hd_j = gama*D[idx] - cblas_ddot(m, &Q[Q_idx_m], 1, d_bar, 1);
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
            for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
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
        
//        shuffle( work_set );
    }
    
    //    printout(D, p);
    
    return;
    
}