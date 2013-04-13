//
//  myUtilities.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/10/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "myUtilities.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef __MATLAB_API__
#include <mat.h>
#include <mex.h>
#endif

#define MAX_LINE_LEN 1024
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;
    
	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// transpose matrix X from row format to column format
static void transpose(const training_set_sp* prob, training_set_sp* prob_col)
{
	int i;
	unsigned long l = prob->N;
	unsigned long n = prob->p;
	int nnz = 0;
	int *col_ptr = new int[n+1];
	feature_node *x_space;
	prob_col->N = l;
	prob_col->p = n;
	prob_col->y = new double[l];
	prob_col->X = new feature_node*[n];
    prob_col->nnz = prob->nnz;
    
	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];
    
	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->X[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;
    
	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->X[i] = &x_space[col_ptr[i]];
    
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->X[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;
    
	prob_col->x_space = x_space;
    
	delete [] col_ptr;
}

// read in a problem (in libsvm format)
void read_problem(const char *filename, training_set_sp* Dset)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
    feature_node* x_space = NULL;
    
	if(fp == NULL)
	{
		printf("can't open input file %s\n",filename);
		exit(1);
	}
    
	Dset->N = 0;
	elements = 0;
	max_line_len = MAX_LINE_LEN;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label
        
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
//		elements++; // for bias term
		(Dset->N)++;
	}
	rewind(fp);
    
//	prob.bias=bias;
    
//	Dset->y = Malloc(double,Dset->N);
//	Dset->X = Malloc(feature_node*,Dset->N);
//	x_space = Malloc(feature_node,elements+Dset->N);
    
    Dset->nnz = elements;
    
    Dset->y = new double[Dset->N];
	Dset->X = new feature_node*[Dset->N];
	x_space = new feature_node[elements+Dset->N];
    
	max_index = 0;
	j=0;
	for(i=0;i<Dset->N;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		Dset->X[i] = &x_space[j];
		label = strtok(line," \t\n");
//		if(label == NULL) // empty line
//			exit_input_error(i+1);
        
		Dset->y[i] = strtod(label,&endptr);
//		if(endptr == label || *endptr != '\0')
//			exit_input_error(i+1);
        
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
            
			if(val == NULL)
				break;
            
//			errno = 0;
//			x_space[j].index = (int) strtol(idx,&endptr,10);
//			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
//				exit_input_error(i+1);
//			else
//				inst_max_index = x_space[j].index;
//            
//			errno = 0;
//			x_space[j].value = strtod(val,&endptr);
//			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
//				exit_input_error(i+1);
            
            x_space[j].index = (int) strtol(idx,&endptr,10);
            inst_max_index = x_space[j].index;
            x_space[j].value = strtod(val,&endptr);
            
			++j;
		}
        
		if(inst_max_index > max_index)
			max_index = inst_max_index;
        
//		if(prob.bias >= 0)
//			x_space[j++].value = prob.bias;
        
		x_space[j++].index = -1;
	}
    
//	if(prob.bias >= 0)
//	{
//		prob.n=max_index+1;
//		for(i=1;i<Dset->N;i++)
//			(prob.x[i]-2)->index = prob.n; 
//		x_space[j-2].index = prob.n;
//	}
//	else
    Dset->p=max_index;
    Dset->x_space = x_space;
    
	fclose(fp);
    free(line);
}

void readLibsvm(const char *filename, training_set_sp* Dset_col)
{
    training_set_sp* Dset = new training_set_sp;
    read_problem(filename, Dset);
//    printout("Dset = ", Dset, ROW_MAJOR);
    transpose(Dset, Dset_col);
    
    delete [] Dset->y;
    delete [] Dset->X;
    delete [] Dset->x_space;
    delete Dset;
    return;
}

void printout(const char* desc, training_set_sp* Dset, int mode)
{
    unsigned long N = Dset->N;
    unsigned long p = Dset->p;
    
    printf( "\n %s\n", desc );
    switch (mode) {
        case ROW_MAJOR:
            for (unsigned long i = 0; i < N; i++) {
                feature_node* x = Dset->X[i];
                for (int j = 1; j <= p; j++) {
                    if (j != x->index) {
                        printf(" 0.00");
                    }
                    else {
                        printf(" %.2f", x->value);
                        x++;
                    }
                }
                printf("\n");
            }
            break;
            
        case COL_MAJOR:
            for (unsigned long i = 0; i < p; i++) {
                feature_node* x = Dset->X[i];
                for (int j = 1; j <= N; j++) {
                    if (j != x->index) {
                        printf(" 0.00");
                    }
                    else {
                        printf(" %.2f", x->value);
                        x++;
                    }
                }
                printf("\n");
            }
            break;
            
        default:
            break;
    }
    
    return;
}

void printout(const char* desc, double* x, unsigned long s1)
{
    printf( "\n %s\n", desc );
    double colSum = 0.0;
    for (unsigned long i = 0; i < s1; i++) {
        colSum += fabs(x[i]);
    }
    printf(" %+.5e\n", colSum);
}

void printout(const char* desc, double* x, unsigned long s1, int mode)
{
    double colSum = 0.0;
    printf( "\n %s\n", desc );
    switch (mode) {
        case FULL:
            for (unsigned long i = 0; i < s1; i++) {
                printf(" %+.5e", x[i]);
            }
            printf("\n");
            break;
        case ROW_VIEW:
            for (unsigned long i = 0; i < s1; i++) {
                colSum += x[i];
            }
            printf(" %+.5e\n", colSum);
            break;
        default:
            break;
    }
    return;
}

void printout(const char* desc, double* x, unsigned long s1, unsigned long s2)
{
    printf( "\n %s\n", desc );
    for (unsigned long i = 0, k = 0; i < s2; i++, k += s1) {
        double colSum = 0.0;
        for (unsigned long j = 0; j < s1; j++) {
            colSum += x[j+k];
        }
        printf(" %+.5e", colSum);
    }
    printf("\n");
}

void printout(const char* desc, double* x, unsigned long s1, unsigned long s2, int mode)
{
    printf( "%s\n", desc );
    switch (mode) {
        case FULL:
            for (unsigned long i = 0; i < s1; i++) {
                for (unsigned long j = 0, k = 0; j < s2; j++, k += s1) {
                    printf(" %.2f", x[k+i]);
                }
                printf("\n");
            }
            break;
        case COL_VIEW:
            for (unsigned long i = 0, k = 0; i < s2; i++, k += s1) {
                double colSum = 0.0;
                for (unsigned long j = 0; j < s1; j++) {
                    colSum += fabs(x[j+k]);
                }
                printf(" %.2f", colSum);
            }
            printf("\n");
            break;
        default:
            break;
    }
    
    
}

void printout(const char* desc, work_set_struct* work_set)
{
    printf( "\n %s\n", desc );
    ushort_pair_t* idxs = work_set->idxs;
    for (unsigned long i = 0; i < work_set->numActive; i++) {
        printf("i=%ld j=%ld\n", idxs[i].i, idxs[i].j);
    }
    printf("\n");
    
    return;
}

void printout(const char* desc, LMatrix* x, int mode)
{
    printf( "%s\n", desc );
    
    double** data = x->data;
    unsigned short _cols = x->cols;
    unsigned short _rows = x->rows;
    
    double* cl;
    
    switch (mode) {
        case FULL:
            for (unsigned short i = 0; i < _rows; i++) {
                for (unsigned short j = 0; j < _cols; j++) {
                    cl = data[j];
                    printf(" %+.5e", cl[i]);
                }
                printf("\n");
            }
            printf("\n");
            break;
        case COL_VIEW:
            for (unsigned short i = 0; i < _cols; i++) {
                cl = data[i];
                double colSum = 0.0;
                for (unsigned short j = 0; j < _rows; j++) {
                    colSum += cl[j];
                }
                printf(" %+.5e", colSum);
            }
            printf("\n");
            break;
        case ROW_VIEW:
            break;
        default:
            break;
    }
    
    return;
}

void printout(const char* desc, LMatrix* x)
{
    printf( "\n %s\n", desc );
    
    double** data = x->data;
    unsigned short _cols = x->cols;
    unsigned short _rows = x->rows;
    
    double* cl;
    
    for (unsigned short i = 0; i < _cols; i++) {
        cl = data[i];
        double colSum = 0.0;
        for (unsigned short j = 0; j < _rows; j++) {
            colSum += cl[j];
        }
        printf(" %+.5e", colSum);
    }
    printf("\n");
    return;
}

double norm(double* x, unsigned long s1)
{
    double norm = 0.0;
    
    for (unsigned long i = 0; i < s1; i++) {
        norm += fabs(x[i]);
    }
    
    return norm;
}

void generateRandomProb(training_set* Dset,
                        unsigned long p, unsigned long N, double nnz_percent)
{
    double* X = new double[p*N];
    double* y = new double[N];
    
    
    
    for (unsigned long i = 0; i < N; i++) {
        double row_nz = 0;
        while (row_nz == 0) { // ensure no all-zero row
            for (unsigned long j = 0, k = 0; j < p; j++, k += N) {
                double indicator = (double) rand() / (double) RAND_MAX;
                if (indicator > nnz_percent) {
                    X[i+k] = 0.0;
                }
                else {
                    X[i+k] = (double) rand() / (double) RAND_MAX;
                }
                row_nz += X[i+k];
            }
            //            printf(" %f\n", row_nz);
        }
        
    }
    
    for (unsigned long i = 0, k = 0; i < p; i++, k += N) {
        double col_nz = 0.0;
        for (unsigned long j = 0; j < N; j++) {
            col_nz += X[k+j];
        }
        while (col_nz == 0) { // ensure no all-zero columns
            for (unsigned long j = 0; j < N; j++) {
                double indicator = (double) rand() / (double) RAND_MAX;
                if (indicator > nnz_percent) {
                    X[j+k] = 0.0;
                }
                else {
                    X[j+k] = (double) rand() / (double) RAND_MAX;
                }
                col_nz += X[j+k];
            }
        }
    }
    
    double r;
    for (unsigned long i = 0; i < N; i++) {
        r = (double) rand() / (double) RAND_MAX;
        
        if (r < 0.5) {
            y[i] = 1;
        } else
            y[i] = -1;
    }
    
    Dset->X = X;
    Dset->y = y;
    Dset->p = p;
    Dset->N = N;
    
    return;
}

void generateRandomProb(training_set_sp* Dset_sp, training_set* Dset,
                        unsigned long p, unsigned long N, double nnz_percent)
{
    generateRandomProb(Dset, p, N, nnz_percent);
    
    transformToSparseFormat(Dset, Dset_sp);
    
    return;
}

void transformToSparseFormat(training_set* Dset, training_set_sp* Dset_sp)
{
    unsigned long p = Dset->p;
    unsigned long N = Dset->N;
    double* X = Dset->X;
    double* y = Dset->y;
    
    unsigned long nnz = 0;
    
    // count nnz
    for (unsigned long i = 0, k = 0; i < p; i++, k += N) {
        for (unsigned long j = 0; j < N; j++) {
            if (X[j+k] != 0) {
                nnz++;
            }
        }
    }
    
    double* y_sp = new double[N];
    feature_node** X_sp = new feature_node*[p];
    feature_node* x_space = new feature_node[nnz+p];
    
    memcpy(y_sp, y, N*sizeof(double));
    
    unsigned long num = 0;
    for (int i = 0, k = 0; i < p; i++, k += N) {
        X_sp[i] = &x_space[num];
        for (int j = 0; j < N; j++) {
            if (X[j+k] != 0) {
                x_space[num].index = j+1;
                x_space[num].value = X[j+k];
                num++;
            }
        }
        x_space[num].index = -1;
        num++;
    }

    Dset_sp->x_space = x_space;
    Dset_sp->X = X_sp;
    Dset_sp->y = y_sp;
    Dset_sp->p = p;
    Dset_sp->N = N;
    Dset_sp->nnz = nnz;
    
    return;
}

void releaseProb(training_set* Dset)
{
    delete [] Dset->X;
    delete [] Dset->y;
    delete Dset;
}

void releaseProb(training_set_sp* Dset)
{
    delete [] Dset->x_space;
    delete [] Dset->X;
    delete [] Dset->y;
    delete Dset;
}

void writeToFile(training_set* Dset)
{
    FILE *fp;
    double* X = Dset->X;
    double* y = Dset->y;
    unsigned long p = Dset->p;
    unsigned long N = Dset->N;
    
	fp = fopen( "Xmat", "w" );
	if (fp == NULL)
	{
		perror ("Error opening file");
		return;
	}
    
	for (unsigned long i = 0; i < N; i++) {
        for (unsigned long j = 0, k = 0; j < p; j++, k += N) {
            fprintf( fp, " %f", X[k+i]);
        }
        fprintf( fp,"\n");
    }
    
    fclose(fp);
    
    fp = fopen( "ymat", "w" );
	if (fp == NULL)
	{
		perror ("Error opening file");
		return;
	}
    
    for (unsigned long i = 0; i < N; i++) {
        fprintf( fp, "%f\n",y[i]);
    }
    
    fclose(fp);
    
    return;
}

void releaseSolution(solution* sols)
{
    delete [] sols->fval;
    delete [] sols->normgs;
    delete [] sols->t;
    delete sols;
    
    return;
}

void printout(const char* desc, solution* sols)
{
    printf("\n %s \n", desc);
    
    printf("fval\t time\t normgs\n");
    for (unsigned long i = 0; i < sols->size; i++) {
        printf("%+.5e\t %.5e\t %.5e\n", sols->fval[i], sols->t[i], sols->normgs[i]);
    }
    
    printf(" CD Time = %.5e\n", sols->cdTime);
    printf(" LS Time = %.5e\n", sols->lsTime);
    printf(" LBFGS Time 1 = %.5e\n", sols->lbfgsTime1);
    printf(" LBFGS Time 2 = %.5e\n", sols->lbfgsTime2);
    return;
    
}

void transformToDenseFormat(training_set* Dset, training_set_sp* Dset_sp)
{
    unsigned long p = Dset_sp->p;
    unsigned long N = Dset_sp->N;
    double* X = new double[p*N];
    double* y = new double[N];
    
    memcpy(y, Dset_sp->y, N*sizeof(double));
    
    unsigned long num = 0;
    unsigned long nnz = 0;
    for (unsigned long i = 0; i < p; i++) {
        feature_node* xnode = Dset_sp->X[i];
        int ind = xnode->index-1;
        for (unsigned long j = 0; j < N; j++) {
            if (j == ind) {
                X[num] = xnode->value;
                xnode++;
                ind = xnode->index-1;
            }
            else {
                X[num] = 0.0;
                nnz++;
            }
            num++;
        }
    }
    
    printf(" nnz = %ld\n", nnz);
    
    Dset->X = X;
    Dset->y = y;
    Dset->p = p;
    Dset->N = N;
    
    return;
}

int cmp_by_vlt(const void *a, const void *b)
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
    
    //    return (int)(ib->vlt - ia->vlt);
}

#ifdef __MATLAB_API__

void readMatFiles(const char* fileName, double* &S, unsigned long* _p)
{
    MATFile *pmat;
    mxArray *pa;
    pmat = matOpen(fileName, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", fileName);
        return;
    }
    
    
    pa = matGetVariable(pmat, "S");
    if (pa == NULL) {
        printf("Error reading existing matrix X\n");
        return;
    }
    
    S = mxGetPr(pa);
    *_p = mxGetM(pa);
    
    matClose(pmat);
    return;
}

void readMatFiles(const char* fileName, training_set* Dset)
{
    MATFile *pmat;
    mxArray *pa;
    pmat = matOpen(fileName, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", fileName);
        return;
    }
    
    
    pa = matGetVariable(pmat, "X");
    if (pa == NULL) {
        printf("Error reading existing matrix X\n");
        return;
    }
    Dset->X = mxGetPr(pa);
    Dset->N = mxGetM(pa);
    Dset->p = mxGetN(pa);
//    mxDestroyArray(pa);
    
    pa = matGetVariable(pmat, "y");
    Dset->y = mxGetPr(pa);
//    mxDestroyArray(pa);
    
    matClose(pmat);
    return;
}

void write2mat(const char* fileName, const char* name,
               double* x, unsigned long s1, unsigned long s2)
{
    MATFile *pmat;
    mxArray *pa;
    pmat = matOpen(fileName, "w");
    if (pmat == NULL) {
        printf("Error opening file %s\n", fileName);
        return;
    }
    
    pa = mxCreateDoubleMatrix(s1, s2, mxREAL);
    memcpy((void *)(mxGetPr(pa)), (void *)x, s1*s2*sizeof(double));
    matPutVariable(pmat, name, pa);
    
    mxDestroyArray(pa);
    matClose(pmat);
    return;
}

void write2mat(const char* fileName, const char* name, LMatrix* x)
{
    unsigned long s1 = x->rows;
    unsigned long s2 = x->cols;
    double* xarray = new double[s1*s2];
    
    double* col;
    unsigned long num = 0;
    for (unsigned long i = 0; i < s2; i++) {
        col = x->data[i];
        for (unsigned long j = 0; j < s1; j++) {
            xarray[num] = col[j];
            num++;
        }
    }
    
    write2mat(fileName, name, xarray, s1, s2);
    
    delete [] xarray;
    return;
}

#endif










