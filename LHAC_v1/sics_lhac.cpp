//
//  sics_lhac.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/21/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "sics_lhac.h"
#include "myUtilities.h"

#include <time.h>


unsigned long work_size;
unsigned short max_iter;
unsigned long max_inner_iter;
double* lmd;
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

static inline void shuffle( work_set_struct* work_set )
{
    unsigned long lens = work_set->numActive;
    ushort_pair_t* idxs = work_set->idxs;
    
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
    }
    
    return;
}

//int cmp_by_vlt(const void *a, const void *b)
//{
//    const ushort_pair_t *ia = (ushort_pair_t *)a;
//    const ushort_pair_t *ib = (ushort_pair_t *)b;
//    
//    if (ib->vlt - ia->vlt > 0) {
//        return 1;
//    }
//    else if (ib->vlt - ia->vlt < 0){
//        return -1;
//    }
//    else
//        return 0;
//    
//    //    return (int)(ib->vlt - ia->vlt);
//}

static inline void computeWorkSet( double* w, double* L_grad, work_set_struct* work_set, double* normsg )
{
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long* idxs_vec_l = work_set->idxs_vec_l;
    unsigned long* idxs_vec_u = work_set->idxs_vec_u;
    unsigned long numActive = 0;
    
    double _normsg = 0.0;
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
//            }
//            _normsg += fabs(g);
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
    
    work_set->numActive = numActive;
    
    for (unsigned long ii = 0; ii < numActive; ii++) {
        unsigned long idx = idxs[ii].i;
        unsigned long jdx = idxs[ii].j;
        idxs_vec_u[ii] = jdx*p_sics + idx;
        idxs_vec_l[ii] = idx*p_sics + jdx;
    }
    
    *normsg = _normsg;
    
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
            //            flopsCount++;
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
        //        flopsCount++;
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
        }
//        ptrdiff_t info = 0;
//        ptrdiff_t p0 = p_sics;
//        dpotrf_((char*) "U", (int *)&p0, w, (int *)&p0, (int *)&info);
        for (unsigned long i = 0; i < p_sics; i++) {
            for (unsigned long j = 0; j <= i; j++) {
                double tmp = w[i*p_sics+j];
                w[j*p_sics+i] = tmp;
            }
        }
//        write2mat("w1.mat", "w1", w, p_2, 1);
        int info = 0;
        int p0 = (int) p_sics;
        dpotrf_((char*) "U", &p0, w, &p0, &info);
        if (info != 0) {
            a *= bbeta;
            printf(" a = %f\n", a);
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
    
//    ptrdiff_t info;
//    ptrdiff_t p0 = p_sics;
//    dpotri_((char*) "U", (int *)&p0, w, (int *)&p0, (int *)&info);
    
    int info;
    int p0 = (int) p_sics;
    dpotri_((char*) "U", &p0, w, &p0, &info);
    
    // fill in the lower triangle
    for (unsigned long i = 0; i < p_sics; i++) {
        for (unsigned long j = 0; j <= i; j++) {
            double tmp = w[i*p_sics+j];
            w[j*p_sics+i] = tmp;
        }
    }
    
//    write2mat("invW.mat", "invW", w, p_2, 1);
    
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
//    for (unsigned long i = 0; i < p_2; i++) {
//        L_grad[i] = S[i] - w[i];
//        w[i] = w_prev[i] + a*D[i];
//    }
    
//    write2mat("D.mat", "D", D, p_2, 1);
//    write2mat("w.mat", "w", w, p_2, 1);
//    write2mat("w_prev.mat", "w_prev", w_prev, p_2, 1);
//    write2mat("L_grad.mat", "L_grad", L_grad, p_2, 1);
//    printf(" f_current = %f\n", f_current);
    
    return;
}

static inline void coordinateDescent(double* w, unsigned long iter, LBFGS* lR,  double* L_grad,
                                     work_set_struct* work_set, double* d_bar, double* H_diag, double* D)
{
    
    memset(D, 0, p_2*sizeof(double));
    memset(d_bar, 0, 2*l*sizeof(double));
    
    double* Q = lR->Q;
    double* Q_bar = lR->Q_bar;
    unsigned short m = lR->m;
    const double gama = lR->gama;
    
    // Hessian diagonal: H_diag = gama - sum(Q'.*Q_bar);
    
//    for (unsigned long k = 0, i = 0; i < p_2; i++, k += m) {
//        H_diag[i] = gama;
//        for (unsigned long j = 0, o = 0; j < m; j++, o += p_2)
//            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[o+i];
//    }

    for (unsigned long k = 0, i = 0; i < p_sics; i++, k += m) {
        H_diag[i] = gama;
        for (unsigned long j = 0, o = 0; j < m; j++, o += p_2)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[o+i];
    }
    
//    write2mat("H_diag.mat", "H_diag", H_diag, p_2, 1);
//    write2mat("Qm.mat", "Qm", Q, p_2, m);
//    write2mat("Qm_bar.mat", "Qm_bar", Q_bar, m, p_2);
    //    lR->computeHDiag(H_diag);
    
//    printout("Q_bar = ", Q_bar, m, p_2, COL_VIEW);
//    printout("Q = ", Q, p_2, (unsigned long) m, COL_VIEW);
    //    printout("w = ", w, p, FULL);
    //    printout("L_grad = ", L_grad, p, FULL);
    //    printout("H_diag = ", H_diag, p, FULL);
    
    double z = 0.0;
    double Hd_j;
    double Hd_i;
    double Hii;
    double G;
    double Gp;
    double Gn;
    double wpd;
    double Hwd;
    
    unsigned long max_inneriter;
    max_inneriter = std::min(1 + iter/3, max_inner_iter);
    //    if (max_inneriter > (param->max_inner_iter)) {
    //        max_inneriter = param->max_inner_iter;
    //    }
    for (unsigned long inneriter = 1; inneriter <= max_inneriter; inneriter++) {
        double diffd = 0;
        double normd = 0;
        
        //        printout(work_set);
        
        ushort_pair_t* idxs = work_set->idxs;
        unsigned long* idxs_vec_l = work_set->idxs_vec_l;
        unsigned long* idxs_vec_u = work_set->idxs_vec_u;
        for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
            unsigned long idx = idxs[ii].i;
            unsigned long jdx = idxs[ii].j;
            unsigned long ij = idxs_vec_u[ii];
            unsigned long ji = idxs_vec_l[ii];
            
            if (idx == jdx) {
                Hd_j = gama*D[ij] - cblas_ddot(m, &Q[ij], (int)p_2, d_bar, 1);
                G = Hd_j + L_grad[ij];
                Gp = G + lmd[ij];
                Gn = G - lmd[ij];
                Hii = H_diag[idx];
            }
            else {
                Hd_j = gama*D[ij] - cblas_ddot(m, &Q[ij], (int)p_2, d_bar, 1);
                Hd_i = gama*D[ji] - cblas_ddot(m, &Q[ji], (int)p_2, d_bar, 1);
                G = Hd_i + Hd_j + L_grad[ij] + L_grad[ji];
                Gp = G + 2*lmd[ij];
                Gn = G - 2*lmd[ij];
//                Hii = H_diag[idx] + H_diag[jdx] - 2*cblas_ddot(m, &Q[idx], (int)p_2, &Q_bar[jdx*m], 1);
                Hii = cblas_ddot(m, &Q[idx], (int)p_2, &Q_bar[jdx*m], 1);
                Hii = Hii*(-2);
                Hii += H_diag[idx] + H_diag[jdx];
            }
            
//            printout("d_bar = ", d_bar, m, FULL);
            
            wpd = w[ij] + D[ij];
            Hwd = Hii * wpd;
            z = -wpd;
            if (Gp <= Hwd)
                z = -Gp/Hii;
            if (Gn >= Hwd)
                z = -Gn/Hii;
            
            if (idx == jdx) {
                D[ij] = D[ij] + z;
                for (unsigned long k = ij*m, j = 0; j < m; j++, k++)
                    d_bar[j] = d_bar[j] + z*Q_bar[k];
            }
            else {
                D[ij] = D[ij] + z;
                D[ji] = D[ji] + z;
                for (unsigned long k1 = ij*m, k2 = ji*m, j = 0; j < m; j++, k1++, k2++) {
//                    d_bar[j] = d_bar[j] + z*(Q_bar[k1] + Q_bar[k2]);
                    d_bar[j] = d_bar[j] + 2*z*Q_bar[k1];
                }
            }
            
            //            lR->updateDbar(d_bar, idx, z);

            
            diffd += fabs(z);
            normd += fabs(D[idx]);
            
        }
        
        if (msgFlag >= LHAC_MSG_CD) {
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
    
    p_sics = _p;
    p_2 = p_sics*p_sics;
    
    
    solution* sols = new solution;
    sols->fval = new double[max_iter];
    sols->normgs = new double[max_iter];
    sols->t = new double[max_iter];
    sols->cdTime = 0;
    sols->lbfgsTime1 = 0;
    sols->lbfgsTime2 = 0;
    sols->lsTime = 0;
    

    double timeBegin = 0.0;
    
    double* w = new double[p_2];
    double* w_prev = new double[p_2];
    double* D = new double[p_2];
    double* L_grad = new double[p_2]; // also store the inverse of w
    double* L_grad_prev = new double[p_2];
    double* d_bar = new double[2*l];
    double* H_diag = new double[p_2];
    
//    double f_prev;
    
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
    
    LBFGS* lR = new LBFGS(p_2, l);
    lR->initData(w, w_prev, L_grad, L_grad_prev);
    
    // active set
    work_set_struct* work_set = new work_set_struct;
    work_set->idxs = new ushort_pair_t[(p_sics+1)*p_sics/2];
    work_set->idxs_vec_l = new unsigned long[(p_sics+1)*p_sics/2];
    work_set->idxs_vec_u = new unsigned long[(p_sics+1)*p_sics/2];
    work_set->_p_sics_ = p_sics;
    
    unsigned short newton_iter;
    double normsg;
    double et;
    for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
        computeWorkSet(w, L_grad, work_set, &normsg);
        double elapsedTime = (clock() - timeBegin)/CLOCKS_PER_SEC;
        sols->fval[newton_iter-1] = f_current;
        sols->normgs[newton_iter-1] = normsg;
        sols->t[newton_iter-1] = elapsedTime;
        if (msgFlag >= LHAC_MSG_NEWTON) {
            printf("%.4e  iter %2d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                   elapsedTime, newton_iter, f_current, normsg, work_set->numActive);
        }
        
        et = clock();
        lR->computeLowRankApprox(work_set);
        et = (clock() - et)/CLOCKS_PER_SEC;
        sols->lbfgsTime1 += et;
        
        et = clock();
        coordinateDescent(w, newton_iter, lR, L_grad, work_set, d_bar, H_diag, D);
        et = (clock() - et)/CLOCKS_PER_SEC;
        sols->cdTime += et;
        
        et = clock();
        lineSearch(S, D, L_grad, w, L_grad_prev, w_prev);
        et = (clock() - et)/CLOCKS_PER_SEC;
        sols->lsTime += et;
        
        et = clock();
        lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        et = (clock() - et)/CLOCKS_PER_SEC;
        sols->lbfgsTime2 += et;
        
        if (normsg <= opt_outer_tol*normsg0) {
            break;
        }
    }
    
    sols->size = newton_iter-1;
    
    
    

    delete [] w;
    delete [] w_prev;
    delete [] D;
    delete [] L_grad;
    delete [] L_grad_prev;
    delete [] d_bar;
    delete [] H_diag;
    delete [] work_set->idxs;
    delete [] work_set->idxs_vec_l;
    delete [] work_set->idxs_vec_u;
    delete work_set;
    delete lR;
    
    return sols;
}
    










