//
//  Lbfgs.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/30/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "Lbfgs.h"

#include <string.h>
#include <stdio.h>
#include <vecLib/clapack.h>
#include <vecLib/cblas.h>

#include "myUtilities.h"

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
    for (unsigned long j = 0; j < rows; j++) {
        cl[j] = x[j];
    }
    
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

LBFGS::LBFGS(unsigned long _p, unsigned short _l)
{
    l = _l;
    p = _p;
    
    Sm = new LMatrix(p, l);
    Tm = new LMatrix(p, l);
    Lm = new LMatrix(l, l);
    STS = new LMatrix(l,l);
    Dm = new double[l];
    
    Q = new double[2*l*p];
    Q_bar = new double[2*l*p];
    R = new double[4*l*l];
    
    buff = new double[p];
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
    sTs = cblas_ddot((int)p, buff, 1, buff, 1);
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

void LBFGS::computeQR()
{
    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    
    double* Tend;
    Tend = Tm->data[_cols-1];
    
    double vv = 0.0;
    vv = cblas_ddot(_rows, Tend, 1, Tend, 1);
    
    gama = vv / Dm[_cols-1];
    
    double et;
    et = clock();
    /* Q */
    //    printout("Sm =", Sm);
    double** S = Sm->data;
    double** T = Tm->data;
    double* cl;
    unsigned long num = 0;
    for (unsigned long i = 0; i < _cols; i++) {
        cl = S[i];
        for (unsigned long j = 0; j < _rows; j++) {
            Q[num] = gama*cl[j];
            num++;
        }
    }
    
    for (unsigned long i = 0; i < _cols; i++) {
        cl = T[i];
        for (unsigned long j = 0; j < _rows; j++) {
            Q[num] = cl[j];
            num++;
        }
    }
    et = (clock() - et)/CLOCKS_PER_SEC;
    tQ += et;
    
    //    write2mat("Sm.mat", "Sm", Sm);
    //    write2mat("STS.mat", "STS", STS);
    
    
    et = clock();
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
    et = (clock() - et)/CLOCKS_PER_SEC;
    tR += et;
    
    return;
}

void LBFGS::computeQR_v2(work_set_struct* work_set)
{
    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    
    double* Tend;
    Tend = Tm->data[_cols-1];
    
    double vv = 0.0;
    vv = cblas_ddot(_rows, Tend, 1, Tend, 1);
    
    gama = vv / Dm[_cols-1];
    
    ushort_pair_t* idxs = work_set->idxs;
    unsigned long numActive = work_set->numActive;
    
    /* Q */
    //    printout("Sm =", Sm);
    double** S = Sm->data;
    double** T = Tm->data;
    double* cl;
    unsigned long num = 0;
    for (unsigned long i = 0; i < _cols; i++) {
        cl = S[i];
        for (unsigned long jj = 0; jj < numActive; jj++) {
            Q[num] = gama*cl[idxs[jj].j];
            num++;
        }
    }
    
    for (unsigned long i = 0; i < _cols; i++) {
        cl = T[i];
        for (unsigned long jj = 0; jj < numActive; jj++) {
            Q[num] = cl[idxs[jj].j];
            num++;
        }
    }
    
//    write2mat("Sm.mat", "Sm", Sm);
//    write2mat("Tm.mat", "Tm", Tm);
//    write2mat("Dm.mat", "Dm", Dm, _cols, 1);
//    write2mat("Qm.mat", "Qm", Q, numActive, 2*_cols);
//    write2mat("work_set.mat", "work_set", work_set);
    
    /* R */
    double* cl1;
    //    double* cl2;
    double** L = Lm->data;
    unsigned short _2cols = 2*_cols;
    memset(R, 0, _2cols*_2cols*sizeof(double));
    
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
    
    return;
}

void LBFGS::computeLowRankApprox_v2(work_set_struct *work_set)
{
//    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    int _2cols = 2*_cols;
    //    unsigned long _p_sics_ = work_set->_p_sics_;
    
    computeQR_v2(work_set);
    
    /* solve R*Q_bar = Q' for Q_bar */
    
    int info;
    dgetrf_(&_2cols, &_2cols, R, &_2cols, ipiv, &info);
    dgetri_(&_2cols, R, &_2cols, ipiv, work, &lwork, &info);
    /* R now store R-1 */
    
    
    int cblas_N = (int) work_set->numActive;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, _2cols, cblas_N, _2cols, 1.0, R, _2cols, Q, cblas_N, 0.0, Q_bar, _2cols);
//    write2mat("Qm.mat", "Qm", Q, cblas_N, _2cols);
//    write2mat("Rm.mat", "Rm", R, _2cols, _2cols);
//    write2mat("Q_barm.mat", "Q_barm", Q_bar, _2cols, cblas_N);
    
//    ushort_pair_t* idxs = work_set->idxs;
//    for (unsigned long i1 = 0; i1 < work_set->numActive; i1++) {
//        unsigned long i2 = idxs[i1].j;
//        unsigned long i3 = i2*_2cols;
//        
//    }
    
    m = _2cols;
    
    return;
}

void LBFGS::computeLowRankApprox()
{
    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    unsigned short _2cols = 2*_cols;
    
    computeQR();
    
    /* solve R*Q_bar = Q' for Q_bar */
    unsigned long num = 0;
    for (unsigned long i = 0; i < _rows; i++) {
        for (unsigned long j = 0, k = 0; j < _2cols; j++, k += _rows) {
            Q_bar[num] = Q[k+i];
            num++;
        }
    }
    
    //    memcpy(Q_bar, Q, _2cols*_rows*sizeof(double));
    
    //    printout("Dm = ", Dm, _cols, FULL);
    //    printout("R = ", R, _2cols, _2cols, FULL);
    //    printout("Q_bar = ", Q_bar, _2cols, _rows, FULL);
    
    //    write2mat("Rm.mat", "Rm", R, 2*(Lm->rows), 2*(Lm->rows));
    //    write2mat("Qm_tran.mat", "Qm_tran", Q_bar, _2cols, _rows);
    double et = clock();
    
    int n = _2cols, nrhs = _rows, lda = _2cols, ldb = _2cols, info;
    /* Solve the equations A*X = B */
    dgesv_( &n, &nrhs, R, &lda, ipiv, Q_bar, &ldb, &info );
    //    dposv_( (char*) "U", &n, &nrhs, R, &lda, Q_bar, &ldb, &info );
//    dsysv_( (char *) "L", &n, &nrhs, R, &lda, ipiv, Q_bar, &ldb, work, &lwork, &info );
    
    //    printf("\n %d \n", info);
    //    printout("Q_bar = ", Q_bar, _2cols, _rows, FULL);
    et = (clock() - et)/CLOCKS_PER_SEC;
    tQ_bar += et;
    
    m = _2cols;
    
    return;
}

void LBFGS::computeLowRankApprox(work_set_struct* work_set)
{
    int _rows = (int)Tm->rows;
    unsigned short _cols = Tm->cols;
    int _2cols = 2*_cols;
//    unsigned long _p_sics_ = work_set->_p_sics_;
    
    computeQR();
    
    /* solve R*Q_bar = Q' for Q_bar */
    double et = clock();
    
    int info;
    dgetrf_(&_2cols, &_2cols, R, &_2cols, ipiv, &info);
    dgetri_(&_2cols, R, &_2cols, ipiv, work, &lwork, &info);
    /* R now store R-1 */
    
//    unsigned long num = 0;
//    for (unsigned long i = 0; i < _p_sics_; i++) {
//        for (unsigned long j = 0, k = 0; j < _2cols; j++, k += _rows) {
//            Q_bar[num] = Q[k+i];
//            num++;
//        }
//    }
    
//    write2mat("Rm.mat", "Rm", R, _2cols, _2cols);
//    write2mat("Qm.mat", "Qm", Q, p, _2cols);
    
    /*******************************************************************************
     Q_bar = R-1 * Q
     ******************************************************************************/
//    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, _2cols, (int)_p_sics_, _2cols, 1.0, R, _2cols, Q, _rows, 0.0, Q_bar, _2cols);
    
//    for (unsigned long i = 0; i < 1; i++) {
//        int alpha = 1.0;
//        int beta = 0.0;
//        cblas_cgemv(CblasColMajor, CblasNoTrans, _2cols, _2cols, &alpha, R, _2cols, &Q[i], _rows, &beta, &Q_bar[i*_2cols], 1);
//    }
    
//    Q[0] = 3;
//    Q[1] = 4;
//    int alpha = 1.0;
//    int beta = 0.0;
//    cblas_dgemv(CblasColMajor, CblasNoTrans, 2, 2, alpha, R, 2, Q, 1, beta, Q_bar, 1);

//    write2mat("Qm_bar.mat", "Qm_bar", Q_bar, _2cols, _p_sics_);
    
    ushort_pair_t* idxs = work_set->idxs;
    for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
        unsigned long ij = idxs[ii].j;
        unsigned long k1 = ij*_2cols;
        cblas_dgemv(CblasColMajor, CblasNoTrans, _2cols, _2cols, 1.0, R, _2cols, &Q[ij], _rows, 0.0, &Q_bar[k1], 1);
    }
    
    et = (clock() - et)/CLOCKS_PER_SEC;
    tQ_bar += et;
    
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
    //        printout("w = ", mdl->w, _P);
    //        printout("w_prev = ", mdl->w_prev, _P);
//    printout("buff = ", buff, p);
    Sm->insertCol(buff);
    double vv = 0.0;// S(:,end)'*T(:,end)
    double diff;
    for (unsigned long i = 0; i < p; i++) {
        diff = L_grad[i] - L_grad_prev[i];
        vv += buff[i]*diff;
        buff[i] = diff;
    }
    //        printout("L_grad_prev = ", mdl->L_grad_prev, _P);
    //        printout("L_grad = ", mdl->L_grad, _P);
    //        printout("L_grad - L_grad_prev = ", buff, _P);
//    printout("buff = ", buff, p);
    Tm->insertCol(buff);
    double* cl1 = Sm->data[Sm->cols-1];
    double* cl2;
    for (unsigned short i = 0; i < Tm->cols-1; i++) {
        cl2 = Tm->data[i];
        //            printout("cl1 = ", cl1, _P);
        //            printout("cl2 = ", cl2, _P);
        buff[i] = cblas_ddot((int)Tm->rows, cl1, 1, cl2, 1);
    }
    //        printout("Lm =", Lm);
    Lm->insertRow(buff);
    memset(buff, 0, Lm->rows*sizeof(double));
    //        printout("Lm = ", Lm);
    Lm->insertCol(buff);
    //        printout("Lm = ", Lm);
    Dm[Lm->rows-1] = vv;
    
    cl1 = Sm->data[Sm->cols-1];
    for (unsigned short i = 0; i < Sm->cols; i++) {
        cl2 = Sm->data[i];
        buff[i] = cblas_ddot((int)Sm->rows, cl1, 1, cl2, 1);
    }
    STS->insertRow(buff);
    STS->insertCol(buff);
    return;
}

void LBFGS::computeHDiag(double *H_diag)
{
    for (unsigned long k = 0, i = 0; i < p; i++, k += m) {
        H_diag[i] = gama;
        for (unsigned long j = 0, o = 0; j < m; j++, o += p)
            H_diag[i] = H_diag[i] - Q_bar[k+j]*Q[o+i];
    }

    return;
}

double LBFGS::computeHdj(double Di, double *d_bar, unsigned long idx)
{
    double Hd_j;
    Hd_j = gama*Di - cblas_ddot(m, &Q[idx], (int)p, d_bar, 1);
    
    return Hd_j;
}

void LBFGS::updateDbar(double *d_bar, unsigned long idx, double z)
{
    for (unsigned long k = idx*m, j = 0; j < m; j++)
        d_bar[j] = d_bar[j] + z*Q_bar[k+j];
    
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
}











