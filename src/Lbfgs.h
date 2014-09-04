
//  Lbfgs.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/30/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Lbfgs__
#define __LHAC_v1__Lbfgs__

#include "linalg.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    unsigned long i;
    unsigned long j;
    double vlt;
} ushort_pair_t;

typedef struct {
    ushort_pair_t* idxs; //p*(p+1)/2
    unsigned long* idxs_vec_l; // vectorized lower
    unsigned long* idxs_vec_u; // vectorized upper
    unsigned long* permut; //p*(p+1)/2
    unsigned long numActive;
    unsigned long _p_sics_;
} work_set_struct;

inline int cmp_by_vlt(const void *a, const void *b)
{
    const ushort_pair_t *ia = (ushort_pair_t *)a;
    const ushort_pair_t *ib = (ushort_pair_t *)b;
    if (ib->vlt - ia->vlt > 0) return 1;
    else if (ib->vlt - ia->vlt < 0) return -1;
    else return 0;
}


class LMatrix {
public:
    double** data;
    double* data_space;
    unsigned long rows;
    unsigned short cols;
    unsigned long maxrows;
    unsigned short maxcols;
    
    LMatrix(const unsigned long s1, const unsigned long s2) : maxrows(s1), maxcols(s2) {
        data = new double*[s2];
        data_space = new double[s1*s2];
        for (unsigned long i = 0, k = 0; i < s2; i++, k += s1) {
            data[i] = &data_space[k];
        }
        rows = 0; cols = 0;
    }; // initiate to be a matrix s1 X s2
    
    ~LMatrix() {
        delete [] data_space;
        delete [] data;
    };
    
    inline void init(const double* const x, const unsigned long n1,
                     const unsigned short n2) {
        rows = n1; cols = n2;
        double* cl;
        for (unsigned long i = 0, k = 0; i < cols; i++, k+=rows) {
            cl = data[i];
            for (unsigned long j = 0; j < rows; j++) cl[j] = x[k+j];
        }
        return;
    };
    // initialized to be the matrix n1 X n2
    
    inline void print() {
        double* cl;
        for (unsigned long i = 0; i < rows; i++) {
            for (unsigned long j = 0; j < cols; j++) {
                cl = data[j];
                printf( " %6.2f", cl[i] );
            }
            printf( "\n" );
        }
    };
    
    inline void insertRow(const double* const x) {
        double* r;
        for (unsigned short i = 0; i < cols; i++) {
            r = data[i];
            r[rows] = x[i];
        }
        rows++;
    }; // to the bottom
    
    inline void insertCol(const double* const x) {
        double* cl = data[cols++];
        memcpy(cl, x, rows*sizeof(double));
    }; // to the rightmost
    
    inline void deleteRow() {
        double* cl;
        rows--;
        for (unsigned short i = 0; i < cols; i++) {
            cl = data[i];
            memmove(cl, cl+1, rows*sizeof(double));
        }
    }; // first
    
    inline void deleteCol() {
        // save the first column pointer
        double* cl = data[0];
        memmove(data, data+1, (--cols)*sizeof(double*));
        // move the first column pointer to the last
        data[cols] = cl;
    }; // leftmost
};

class LBFGS {
public:
    double* Q;
    double* Q_bar;
    unsigned short m; // no. of cols in Q
    double gama;
    
    // for test
    double tQ;
    double tR;
    double tQ_bar;
    
    double* buff;
    
    double shrink;
    
    LBFGS(const unsigned long _p, const unsigned short _l,
          const double _s) : shrink(_s), p(_p), l(_l) {
        //        l = _l;
        //        p = _p;
        //        shrink = _s;
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
        for (unsigned long j = 0; j < l; j++) permut[j] = j+1;
        memset(permut_mx, 0, l*l*sizeof(double));
        Q = new double[2*l*p];
        Q_bar = new double[2*l*p];
        R = new double[4*l*l];
        buff = new double[p];
    };
    
    ~LBFGS() {
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
    };
    
    inline void initData(const double* const w, const double* const w_prev,
                         const double* const L_grad, const double* const L_grad_prev) {
        /* S = [S obj.w-obj.w_prev]; */
        for (unsigned long i = 0; i < p; i++) buff[i] = w[i] - w_prev[i];
        Sm->init(buff, p, 1);
        double sTs;
        sTs = lcddot((int)p, buff, 1, buff, 1);
        STS->init(&sTs, 1, 1);
        /* T = [T obj.L_grad-obj.L_grad_prev]; */
        double vv = 0.0;// S(:,end)'*T(:,end)
        double diff;
        for (unsigned long i = 0; i < p; i++) {
            diff = L_grad[i] - L_grad_prev[i];
            vv += buff[i]*diff;
            buff[i] = diff;
        }
        Tm->init(buff, p, 1);
        Dm[0] = vv;
        buff[0] = 0.0;
        Lm->init(buff, 1, 1);
    };
    
    /* different from l1log */
    inline void computeLowRankApprox_v2(work_set_struct* work_set) {
        int _rows = (int)Tm->rows;
        unsigned short _cols = Tm->cols;
        int _2cols = 2*_cols;
        const int _p_sics = sqrt(_rows);
        
        computeQR_v2(work_set);
        
        /* solve R*Q_bar = Q' for Q_bar */
        inverse(R, _2cols);
        
        /* R now store R-1 */
        int cblas_N = (int) work_set->numActive + _p_sics;
        lcdgemm(R, Q, Q_bar, _2cols, cblas_N);
        
        m = _2cols;
    };
    
    inline void updateLBFGS(const double* const w, const double* const w_prev,
                            const double* const L_grad, const double* const L_grad_prev) {
        if (Sm->cols >= l) {
            Sm->deleteCol();
            Tm->deleteCol();
            Lm->deleteRow();
            Lm->deleteCol();
            STS->deleteCol();
            STS->deleteRow();
            memmove(Dm, Dm+1, (l-1)*sizeof(double));
        }
        for (unsigned long i = 0; i < p; i++)
            buff[i] = w[i] - w_prev[i];
        Sm->insertCol(buff);
        for (unsigned long i = 0; i < p; i++)
            buff[i] = L_grad[i] - L_grad_prev[i];
        Tm->insertCol(buff);
        double* cl1 = Sm->data[Sm->cols-1];
        int cblas_N = (int) Tm->rows;
        int cblas_M = (int) Tm->cols;
        //    lcdgemv(CblasRowMajor, CblasNoTrans, Tm->data_space, cl1, buff, cblas_M, cblas_N, cblas_N);
        lcdgemv(CblasColMajor, CblasTrans, Tm->data_space, cl1, buff, cblas_N, cblas_M, cblas_N);
        if (Sm->cols >= l) {
            /* update permut */
            for (unsigned long j = 0; j < l; j++)
                if (permut[j] != 0) permut[j]--;
                else permut[j] = l-1;
            /* update permut matrix */
            for (unsigned long j = 0; j < l; j++) {
                unsigned long imx = permut[j];
                unsigned long jmx = j;
                unsigned long ij = jmx*l + imx;
                permut_mx[ij] = 1;
            }
            /* permuting buff */
            lcdgemv(CblasColMajor, CblasNoTrans, permut_mx, buff, buff2, (int)l, (int)l, (int)l);
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
        //    lcdgemv(CblasRowMajor, CblasNoTrans, Sm->data_space, cl1, buff, cblas_M, cblas_N, cblas_N);
        lcdgemv(CblasColMajor, CblasTrans, Sm->data_space, cl1, buff, cblas_N, cblas_M, cblas_N);
        if (Sm->cols >= l) {
            /* permuting buff */
            lcdgemv(CblasColMajor, CblasNoTrans, permut_mx, buff, buff2, (int)l, (int)l, (int)l);
            memset(permut_mx, 0, l*l*sizeof(double));
            STS->insertRow(buff2);
            STS->insertCol(buff2);
        }
        else {
            STS->insertRow(buff);
            STS->insertCol(buff);
        }
    };
    
private:
    LMatrix* Sm;
    LMatrix* Tm;
    LMatrix* Lm;
    LMatrix* STS;
    unsigned long* permut; // for updating lbfgs, length of l
    double* permut_mx; // for updating lbfgs, l*l
    double* buff2; // length of l
    
    double* Dm;
    double* R;
    unsigned long p; // no. of rows in Q
    unsigned short l; // lbfgs param
    
    /* different from l1log */
    inline void computeQR_v2(work_set_struct* work_set) {
        int _rows = (int)Tm->rows;
        unsigned short _cols = Tm->cols;
        const int _p_sics = sqrt(_rows); // sics matrix dimension
        
        double* Tend;
        Tend = Tm->data[_cols-1];
        
        double vv = 0.0;
        vv = lcddot(_rows, Tend, 1, Tend, 1);
        gama = vv / Dm[_cols-1] / shrink;
        //    gama = Dm[_cols-1] / vv;
        
        unsigned long numActive = work_set->numActive;
        unsigned long* idxs_vec_u = work_set->idxs_vec_u;
        
        /* Q in row major */
        double** S = Sm->data;
        double** T = Tm->data;
        double* cl;
        for (unsigned long i = 0; i < _cols; i++) {
            cl = S[i];
            unsigned long k = 0;
            for (unsigned long j = 0; j < _p_sics; j++, k += 2*_cols) {
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
            for (unsigned long j = 0; j < _p_sics; j++, k += 2*_cols) {
                Q[i+k+_cols] = cl[j];
            }
            
            for (unsigned long j = 0; j < numActive; j++, k += 2*_cols) {
                unsigned long ij = idxs_vec_u[j];
                Q[i+k+_cols] = cl[ij];
            }
        }
        
        /* R */
        double* cl1;
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
    };
    
};


#endif /* defined(__LHAC_v1__Lbfgs__) */
