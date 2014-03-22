//
//  myUtilities.cpp
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 2/10/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#include "myUtilities.h"

#include <time.h>
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

void printout(const char* desc, double* x, unsigned long s1)
{
    printf( "\n %s\n", desc );
    double colSum = 0.0;
    for (unsigned long i = 0; i < s1; i++) {
        colSum += fabs(x[i]);
    }
    printf(" %+.5e\n", colSum);
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

void printout(const char* desc, solution* sols, param* param)
{
    FILE *fp;
    time_t current_time;
    
    time(&current_time);
    
    fp = fopen( "LHAC_log", "a" );
	if (fp == NULL)
	{
		perror ("Error opening file");
		return;
	}
    
    /* end of experiment */
    unsigned long nnz = 0;
    unsigned long p_2 = sols->p_sics*sols->p_sics;
    for (unsigned long i = 0; i < p_2; i++) {
        if (sols->w[i] != 0) {
            nnz++;
        }
    }
    
    fprintf(fp, "====================================================\n");
    fprintf(fp, "%s     %s", param->fileName, ctime(&current_time));
    fprintf(fp, "sufficient decrease: \t %d\n", param->sd_flag);
    fprintf(fp, "gamma scale: \t\t\t %.1f\n", param->shrink);
    fprintf(fp, "# max iters: \t\t\t %d\n", param->max_iter);
    fprintf(fp, "opt outer tol: \t\t\t %.0e\n", param->opt_outer_tol);
    fprintf(fp, "lambda = %f\n", param->lmd[0]);
    fprintf(fp, "nnz = %ld\n", nnz);
    fprintf(fp, "sparse ratio = %f%%\n", (double) nnz/p_2*100.0);
    fprintf(fp, "====================================================\n");
    fprintf(fp, "%s \n", desc);
    
    fprintf(fp, "#iter \t fval \t time \t normgs\n");
    for (unsigned long i = 0; i < sols->size; i++) {
        fprintf(fp, "%4d \t %+.8e\t %.5e\t %.5e\n", sols->niter[i], sols->fval[i], sols->t[i], sols->normgs[i]);
    }
    
    fprintf(fp, "\n");
    
    fprintf(fp, "totaltime \t cdtime \t ratio \t #factors \t fvaltime \t gvaltime \t #ls\n");
    fprintf(fp, "%.5e \t %.5e \t %2.1f%% \t %ld \t %.5e \t %.5e \t %ld\n", sols->t[sols->size-1], sols->cdTime, sols->cdTime *100 / sols->t[sols->size-1], sols->record1, sols->fvalTime, sols->gvalTime, sols->nls);
    //    fprintf(fp, " CD Time = %.5e\n", sols->cdTime);
    //    fprintf(fp, " LS Time = %.5e\n", sols->lsTime);
    //    fprintf(fp, " LBFGS Time 1 = %.5e\n", sols->lbfgsTime1);
    //    fprintf(fp, " LBFGS Time 2 = %.5e\n", sols->lbfgsTime2);
    
    fclose(fp);
    
    fp = fopen( "LHAC_brief", "a" );
	if (fp == NULL)
	{
		perror ("Error opening file");
		return;
	}
//    fprintf(fp, "gamma_scale \t #iter \t time \t lstime \t sd\n");
    fprintf(fp, "%3.0f \t %4d \t %+.5e \t %+.5e \t %d \t %s \t %ld",
            param->shrink, sols->niter[sols->size-1], sols->t[sols->size-1], sols->lsTime, param->sd_flag, param->fileName, sols->record1);
    if (param->sd_flag == 1) {
        fprintf(fp, " \t %.2f\n", param->rho);
    }
    else {
        fprintf(fp, " \t --\n");
    }
    fclose(fp);
    
    fp = fopen("LHAC_timeProfile.csv", "a");
    if (fp == NULL)
	{
		perror ("Error opening file");
		return;
	}
    //    fprintf(fp, "gamma_scale \t #iter \t time \t lstime \t sd\n");
    
    /* name || dimension || tolerance || totaltime || cdtime || fvaltime || gvaltime || #iter || #ls || #factors */
    fprintf(fp, "%s, \t",param->fileName);
    fprintf(fp, "%d, \t",sols->p_sics);
    fprintf(fp, "%.0e, \t",param->opt_outer_tol);
    fprintf(fp, "%.5e, \t",sols->t[sols->size-1]);
    fprintf(fp, "%.5e, \t",sols->cdTime);
    fprintf(fp, "%.5e, \t",sols->fvalTime);
    fprintf(fp, "%.5e, \t",sols->gvalTime);
    fprintf(fp, "%.5e, \t",sols->lbfgsTime1);
    fprintf(fp, "%.5e, \t",sols->lbfgsTime2);
    fprintf(fp, "%d, \t",sols->niter[sols->size-1]);
    fprintf(fp, "%ld, \t",sols->nls);
    fprintf(fp, "%ld \t",sols->record1);
    fprintf(fp, "\n");
    fclose(fp);
    return;
    
}

void readTextFile(const char* filename, double* &S, unsigned long* _p)
{
    unsigned long nLines;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *val;
    
    
	if(fp == NULL)
	{
		printf("can't open input file %s\n",filename);
		exit(1);
	}
    
    nLines = 0;
	elements = 0;
	max_line_len = MAX_LINE_LEN;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t");
        if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
            continue;
        
        nLines++;
		
	}
	rewind(fp);
    
    S = new double[nLines*nLines];
    
	j=0;
	while(readline(fp)!=NULL)
	{
//		readline(fp);
        val = strtok(line," \t\n");
        if(val == NULL || *val == '\n')
            continue;
        S[j] = strtod(val,&endptr);
        j++;
        
		while(1)
		{
			val = strtok(NULL," \t");
            
			if(val == NULL || *val == '\n')
				break;
            
            S[j] = strtod(val,&endptr);
            
			++j;
		}
	}
    
    *_p = nLines;
    
	fclose(fp);
    free(line);
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










