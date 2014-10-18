//
//  LogReg.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/30/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#include "LogReg.h"
#include <math.h>
#include "linalg.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE_LEN 1024
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char *line = NULL;
static int max_line_len;

typedef struct {
    int index; // starting from 1 (not 0), ending with -1
    double value;
} feature_node;

typedef struct training_set_sp_strct {
    feature_node** X;
    double* y;
    unsigned long p;
    unsigned long N;
    unsigned long nnz; // number of nonzeros
    feature_node* x_space;
    
    ~training_set_sp_strct() {
        delete [] x_space;
        delete [] X;
        delete [] y;
    }
} training_set_sp;

typedef struct training_set_strct {
    double* X;
    double* y;
    unsigned long p;
    unsigned long N;
    
    ~training_set_strct() {
        delete [] X;
        delete [] y;
    }
} training_set;


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
            
            x_space[j].index = (int) strtol(idx,&endptr,10);
            inst_max_index = x_space[j].index;
            x_space[j].value = strtod(val,&endptr);
            
			++j;
		}
        
		if(inst_max_index > max_index)
			max_index = inst_max_index;
        
		x_space[j++].index = -1;
	}
    
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
    
    delete Dset;
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

LogReg::LogReg(const char *filename)
{
    
    // sparse format
    training_set_sp* Dset_sp = new training_set_sp;
    readLibsvm(filename, Dset_sp);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", Dset_sp->p, Dset_sp->N, Dset_sp->nnz);
    
    training_set* Dset = new training_set;
    transformToDenseFormat(Dset, Dset_sp);
    delete Dset_sp;
    
    p = Dset->p;
    N = Dset->N;
    
    X = new double[p*N];
    y = new double[N];
    
    memcpy(X, Dset->X, sizeof(double)*p*N);
    memcpy(y, Dset->y, sizeof(double)*N);
    
    e_ywx = new double[N]; // N
    B = new double[N]; // N
    
}

unsigned long LogReg::getDims() const
{
    return p;
}

double LogReg::computeObject(double* wnew)
{
    double fval = 0;
    
//    double alpha = 1.0;
//    double beta = 0.0;
//    cblas_dgemv(CblasColMajor, CblasNoTrans, (int)N, (int)p, alpha, X, (int)N, wnew, 1, beta, e_ywx, 1);
    lcdgemv(CblasColMajor, CblasNoTrans, X, wnew, e_ywx, (int)N, (int)p, (int)N);
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
    
    return fval;
}

// always computed after computeObject
void LogReg::computeGradient(const double* wnew, double* df)
{
    for (unsigned long i = 0; i < N; i++) {
        B[i] = -y[i]/(1+e_ywx[i]);
    }
//    cblas_dgemv(CblasColMajor, CblasTrans, (int)N, (int)p, 1.0, X, (int)N, B, 1, 0.0, df, 1);
    lcdgemv(CblasColMajor, CblasTrans, X, B, df, (int)N, (int)p, (int)N);
    return;
}

LogReg::~LogReg()
{
    delete [] X;
    delete [] y;
    delete [] e_ywx;
    delete [] B;
}





















