//
//  utils.h
//  pepper
//
//  Created by Xiaocheng Tang on 11/11/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef pepper_utils_h
#define pepper_utils_h

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef __MATLAB_API__
#include <mat.h>
#include <mex.h>
#endif

#define MAX_LINE_LEN 2048
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
};


static void readTextFile(const char* filename, double* &S, unsigned long* _p)
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
    if (line == NULL) {
        line = Malloc(char,max_line_len);
    }
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
        val = strtok(line," ,\t\n");
        if(val == NULL || *val == '\n')
            continue;
        S[j] = strtod(val,&endptr);
        j++;
        
        while(1)
        {
            val = strtok(NULL," ,\t");
            
            if(val == NULL || *val == '\n')
                break;
            
            S[j] = strtod(val,&endptr);
            
            ++j;
        }
    }
    
    *_p = nLines;
    
    fclose(fp);
    free(line);
};

#ifdef __MATLAB_API__

static void readMatFiles(const char* fileName, double* &S, unsigned long* _p)
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
};
#endif

#endif
