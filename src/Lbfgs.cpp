//
//  Lbfgs.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/30/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "Lbfgs.h"
#include "liblapack.h"

#include <math.h>
#include <string.h>
#include <stdio.h>


LMatrix::LMatrix(unsigned long s1, unsigned long s2)
{
    data = new double*[s2];
//    for (unsigned short i = 0; i < s2; i++) {
//        data[i] = new double[s1];
//    }
    data_space = new double[s1*s2];
    for (unsigned long i = 0, k = 0; i < s2; i++, k += s1) {
        data[i] = &data_space[k];
    }
    
    
    rows = 0;
    cols = 0;
    maxcols = s2;
    maxrows = s1;
}

LMatrix::~LMatrix()
{
//    for (unsigned short i = 0; i < maxcols; i++) {
//        delete [] data[i];
//    }
    delete [] data_space;
    delete [] data;
}

void LMatrix::init(double *x, unsigned long n1, unsigned short n2)
{
    rows = n1;
    cols = n2;
    
    double* cl;
    for (unsigned long i = 0, k = 0; i < cols; i++, k+=rows) {
        cl = data[i];
        for (unsigned long j = 0; j < rows; j++) {
            cl[j] = x[k+j];
        }
    }
    
    return;
}

void LMatrix::print()
{
    double* cl;
    for (unsigned long i = 0; i < rows; i++) {
        for (unsigned long j = 0; j < cols; j++) {
            cl = data[j];
            printf( " %6.2f", cl[i] );
        }
        printf( "\n" );
    }
    
    return;
}

void LMatrix::insertRow(double* x)
{
    double* r;
    for (unsigned short i = 0; i < cols; i++) {
        r = data[i];
        r[rows] = x[i];
    }
    
    rows++;
    
    return;
}

void LMatrix::insertCol(double* x)
{
    double* cl = data[cols];
//    for (unsigned long j = 0; j < rows; j++) {
//        cl[j] = x[j];
//    }
    memcpy(cl, x, rows*sizeof(double));
    
    ++cols;
    
    return;
}

void LMatrix::deleteRow()
{
    double* cl;
    rows--;
    for (unsigned short i = 0; i < cols; i++) {
        cl = data[i];
        memmove(cl, cl+1, rows*sizeof(double));
    }
    
    return;
}

void LMatrix::deleteCol()
{
    // save the first column pointer
    double* cl = data[0];
    cols--;
    memmove(data, data+1, cols*sizeof(double*));
    
    // move the first column pointer to the last
    data[cols] = cl;
    
    return;
}

LBFGS::LBFGS(unsigned long _p, unsigned short _l, double _s)
{
    l = _l;
    p = _p;
    shrink = _s;
    
    tQ = 0;
    tR = 0;
    tQ_bar = 0;
    
    Sm = new LMatrix(p, l);
    Tm = new LMatrix(p, l);
    Lm = new LMatrix(l, l);
    STS = new LMatrix(l,l);
    Dm = new double[l];
    permut = new unsigned long[l];
    permut_mx = new double[l*l];
    buff2 = new double[l];
    /* initialize permut and permut matrix */
    for (unsigned long j = 0; j < l; j++) {
        permut[j] = j+1;
    }
    memset(permut_mx, 0, l*l*sizeof(double));
    
    Q = new double[2*l*p];
    Q_bar = new double[2*l*p];
    R = new double[4*l*l];
    
    buff = new double[p];
    
//    lwork = MAX_SY_PAIRS*MAX_SY_PAIRS;
}

void LBFGS::initData(double *w, double *w_prev, double *L_grad, double *L_grad_prev)
{
    /* S = [S obj.w-obj.w_prev]; */
    for (unsigned long i = 0; i < p; i++) {
        buff[i] = w[i] - w_prev[i];
//        printf(" i = %ld\n", i);
    }
    Sm->init(buff, p, 1);
    //    printout("Sm = ", Sm);
    
    double sTs;
    sTs = lcddot((int)p, buff, 1, buff, 1);
    STS->init(&sTs, 1, 1);
    
//    write2mat("STS.mat", "STS", STS);
//    write2mat("Sm.mat", "Sm", Sm);
    
    /* T = [T obj.L_grad-obj.L_grad_prev]; */
    double vv = 0.0;// S(:,end)'*T(:,end)
    double diff;
    for (unsigned long i = 0; i < p; i++) {
        diff = L_grad[i] - L_grad_prev[i];
        vv += buff[i]*diff;
        buff[i] = diff;
    }
    Tm->init(buff, p, 1);
    //    printout("Sm = ", Tm);
    
    Dm[0] = vv;
    
    buff[0] = 0.0;
    Lm->init(buff, 1, 1);
    
    return;
}


void LBFGS::computeQR_v2(work_set_struct* work_set)
{
    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    int p_sics = sqrt(_rows); // sics matrix dimension
    
    double* Tend;
    Tend = Tm->data[_cols-1];
    
    double vv = 0.0;
    vv = lcddot(_rows, Tend, 1, Tend, 1);
    gama = vv / Dm[_cols-1] / shrink;
//    gama = Dm[_cols-1] / vv;
    
    unsigned long numActive = work_set->numActive;
    unsigned long* idxs_vec_u = work_set->idxs_vec_u;
    
    double et;
//    et = clock();
    /* Q */
    //    printout("Sm =", Sm);
    double** S = Sm->data;
    double** T = Tm->data;
    double* cl;
    unsigned long num = 0;
//    for (unsigned long i = 0; i < _cols; i++) {
//        cl = S[i];
//        for (unsigned long j = 0; j < p_sics; j++) {
//            Q[num] = gama*cl[j];
//            num++;
//        }
//        
//        for (unsigned long j = 0; j < numActive; j++) {
//            unsigned long ij = idxs_vec_u[j];
//            Q[num] = gama*cl[ij];
//            num++;
//        }
//    }
//    
//    for (unsigned long i = 0; i < _cols; i++) {
//        cl = T[i];
//        for (unsigned long j = 0; j < p_sics; j++) {
//            Q[num] = cl[j];
//            num++;
//        }
//        
//        for (unsigned long j = 0; j < numActive; j++) {
//            unsigned long ij = idxs_vec_u[j];
//            Q[num] = cl[ij];
//            num++;
//        }
//    }
    
    for (unsigned long i = 0; i < _cols; i++) {
        cl = S[i];
        unsigned long k = 0;
        for (unsigned long j = 0; j < p_sics; j++, k += 2*_cols) {
            Q[i+k] = gama*cl[j];
        }
        
        for (unsigned long j = 0; j < numActive; j++, k += 2*_cols) {
            unsigned long ij = idxs_vec_u[j];
            Q[i+k] = gama*cl[ij];
        }
    }
    
    for (unsigned long i = 0; i < _cols; i++) {
        cl = T[i];
        unsigned long k = 0;
        for (unsigned long j = 0; j < p_sics; j++, k += 2*_cols) {
            Q[i+k+_cols] = cl[j];
        }
        
        for (unsigned long j = 0; j < numActive; j++, k += 2*_cols) {
            unsigned long ij = idxs_vec_u[j];
            Q[i+k+_cols] = cl[ij];
        }
    }
    
    
    
    
//    et = (clock() - et)/CLOCKS_PER_SEC;
//    tQ += et;
    
    //    write2mat("Sm.mat", "Sm", Sm);
    //    write2mat("STS.mat", "STS", STS);
    
    
//    et = clock();
    /* R */
    double* cl1;
    //    double* cl2;
    double** L = Lm->data;
    unsigned short _2cols = 2*_cols;
    memset(R, 0, _2cols*_2cols*sizeof(double));
    // R: 2*_cols X 2*_cols
    //    for (unsigned short i = 0; i < _cols; i++) {
    //        cl1 = S[i];
    //        for (unsigned short j = 0, k = 0; j < i; j++, k += _2cols) {
    //            cl2 = S[j];
    //            R[k+i] = cblas_ddot(_rows, cl1, 1, cl2, 1);
    //            R[k+i] = gama*R[k+i];
    //            R[i*_2cols+j] = R[k+i];
    //        }
    //    }
    
    //    for (unsigned short i = 0, k = 0; i < _cols; i++, k += _2cols) {
    //        cl1 = S[i];
    //        R[k+i] = cblas_ddot(_rows, cl1, 1, cl1, 1);
    //        R[k+i] = gama*R[k+i];
    //    }
    
    double** STSdata = STS->data;
    for (unsigned short i = 0, k = 0; i < _cols; i++, k += _2cols) {
        cl1 = STSdata[i];
        for (unsigned short j = 0; j < i; j++) {
            unsigned short ji = k + j;
            unsigned short ij = j*_2cols+i;
            R[ji] = cl1[j];
            R[ji] = gama*R[ji];
            R[ij] = R[ji];
        }
    }
    
    
    for (unsigned short i = 0, k = 0; i < _cols; i++, k += (_2cols+1)) {
        cl1 = STSdata[i];
        R[k] = cl1[i];
        R[k] = gama*R[k];
    }
    
    
    for (unsigned short i = _cols, k = _cols*_2cols, o = 0; i < _2cols; i++, k += _2cols, o++) {
        cl1 = L[o];
        for (unsigned short j = 0; j < _cols; j++) {
            R[k+j] = cl1[j];
        }
    }
    
    for (unsigned short i = _cols, o = 0; i < _2cols; i++, o++) {
        cl1 = L[o];
        for (unsigned short j = 0, k = 0; j < _cols; j++, k += _2cols) {
            R[k+i] = cl1[j];
        }
    }
    
    for (unsigned short i = _cols, k = _cols*_2cols, j = 0; i < _2cols; i++, k += _2cols, j++) {
        R[k+i] = -Dm[j];
    }
//    et = (clock() - et)/CLOCKS_PER_SEC;
//    tR += et;
    
    return;
    
}

void LBFGS::computeLowRankApprox_v2(work_set_struct* work_set)
{
    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    int _2cols = 2*_cols;
    int p_sics = sqrt(_rows);
    //    unsigned long _p_sics_ = work_set->_p_sics_;
    
    computeQR_v2(work_set);
    
    /* solve R*Q_bar = Q' for Q_bar */
//    double et = clock();
    
//    int info;
//    dgetrf_(&_2cols, &_2cols, R, &_2cols, ipiv, &info);
//    dgetri_(&_2cols, R, &_2cols, ipiv, work, &lwork, &info);
//    lcdgetrf_(R, _2cols, &info);
//    lcdgetri_(R, _2cols, &info);
    inverse(R, _2cols);
    
    /* R now store R-1 */
    int cblas_N = (int) work_set->numActive + p_sics;
//    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, _2cols, cblas_N, _2cols, 1.0, R, _2cols, Q, cblas_N, 0.0, Q_bar, _2cols);
//    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, _2cols, cblas_N, _2cols, 1.0, R, _2cols, Q, _2cols, 0.0, Q_bar, _2cols);
    lcdgemm(R, Q, Q_bar, _2cols, cblas_N);
    
//    et = (clock() - et)/CLOCKS_PER_SEC;
//    tQ_bar += et;
    
    m = _2cols;
    return;
    
}

void LBFGS::updateLBFGS(double* w, double* w_prev, double* L_grad, double* L_grad_prev)
{
    if (Sm->cols >= l) {
        Sm->deleteCol();
        Tm->deleteCol();
        Lm->deleteRow();
        Lm->deleteCol();
        STS->deleteCol();
        STS->deleteRow();
        memmove(Dm, Dm+1, (l-1)*sizeof(double));
    }
    for (unsigned long i = 0; i < p; i++) {
        buff[i] = w[i] - w_prev[i];
    }

    Sm->insertCol(buff);

    for (unsigned long i = 0; i < p; i++) {
        buff[i] = L_grad[i] - L_grad_prev[i];
    }
    Tm->insertCol(buff);
    
    double* cl1 = Sm->data[Sm->cols-1];
    int cblas_N = (int) Tm->rows;
    int cblas_M = (int) Tm->cols;
//    cblas_dgemv(CblasRowMajor, CblasNoTrans, cblas_M, cblas_N, 1.0, Tm->data_space, cblas_N, cl1, 1, 0.0, buff, 1);
    lcdgemv(CblasRowMajor, CblasNoTrans, Tm->data_space, cl1, buff, cblas_M, cblas_N);
    
    if (Sm->cols >= l) {
//        write2mat("Sm.mat", "Sm", Sm);
//        write2mat("Tm.mat", "Tm", Tm);
//        write2mat("buff.mat", "buff", buff, cblas_M, 1);
        /* update permut */
        for (unsigned long j = 0; j < l; j++) {
            if (permut[j] != 0) {
                permut[j]--;
            }
            else
                permut[j] = l-1;
//            printf(" %ld", permut[j]);
        }
//        printf("\n");
        
        /* update permut matrix */
        for (unsigned long j = 0; j < l; j++) {
            unsigned long imx = permut[j];
            unsigned long jmx = j;
            unsigned long ij = jmx*l + imx;
            permut_mx[ij] = 1;
        }
        
        /* permuting buff */
//        cblas_dgemv(CblasColMajor, CblasNoTrans, (int)l, (int)l, 1.0, permut_mx, (int)l, buff, 1, 0.0, buff2, 1);
        lcdgemv(CblasColMajor, CblasNoTrans, permut_mx, buff, buff2, (int)l, (int)l);
        
//        write2mat("permut_mx.mat", "permut_mx", permut_mx, l, l);
//        write2mat("buff_permut.mat", "buff_permut", buff2, cblas_M, 1);
        
        Lm->insertRow(buff2);
        Dm[Lm->rows-1] = buff2[l-1];
        
    }
    else {
        Lm->insertRow(buff);
        Dm[Lm->rows-1] = buff[Tm->cols-1];
    }

    
    memset(buff, 0, Lm->rows*sizeof(double));
    Lm->insertCol(buff);
    
    
    cl1 = Sm->data[Sm->cols-1];
    cblas_N = (int) Sm->rows;
    cblas_M = (int) Sm->cols;
//    cblas_dgemv(CblasRowMajor, CblasNoTrans, cblas_M, cblas_N, 1.0, Sm->data_space, cblas_N, cl1, 1, 0.0, buff, 1);
    lcdgemv(CblasRowMajor, CblasNoTrans, Sm->data_space, cl1, buff, cblas_M, cblas_N);
    
    if (Sm->cols >= l) {
        /* permuting buff */
//        cblas_dgemv(CblasColMajor, CblasNoTrans, (int)l, (int)l, 1.0, permut_mx, (int)l, buff, 1, 0.0, buff2, 1);
        lcdgemv(CblasColMajor, CblasNoTrans, permut_mx, buff, buff2, (int)l, (int)l);
        
        memset(permut_mx, 0, l*l*sizeof(double));        
        
        STS->insertRow(buff2);
        STS->insertCol(buff2);
    }
    else {
        STS->insertRow(buff);
        STS->insertCol(buff);
    }


    return;
}

LBFGS::~LBFGS()
{
    delete Tm;
    delete Lm;
    delete Sm;
    delete STS;
    delete [] Dm;
    delete [] Q;
    delete [] Q_bar;
    delete [] R;
    delete [] buff;
    delete [] buff2;
    delete [] permut_mx;
    delete [] permut;
}











