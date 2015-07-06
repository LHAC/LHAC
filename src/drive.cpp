//
//  main.c
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/21/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "lhac.h"
#include "LogReg.h"
#include "Lasso.h"
#include "utils.h"

void exit_with_help()
{
    printf(
           "Usage: lhac [options] training_set_file or model_file (see option m)\n"
           "options:\n"
           "-m model_file : model_file existence indicator (default false)\n"
           "	true  -- read from model_file without training\n"
           "	false -- train a new model from training_set_file\n"
           "-p test file: apply model on the testing file\n"
           "              and output the result to stdout \n"
           "-d dense format : set matrix format dense or sparse (default 1)\n"
           "	1  -- dense\n"
           "	0  -- sparse\n"
           "-l loss function : set type of loss function (default log)\n"
           "	log    -- logistic regression\n"
           "	square -- least square\n"
           "-c lambda : set the regularization parameter (default 1)\n"
           "-a : pre-compute A^TA in least sqaure  (default true)\n"
           "-i : max number of iterations (default 1000)\n"
           "-b : limited memory in L-BFGS (default 10, maximum 20)\n"
           "-e epsilon : set tolerance of termination criterion\n"
           "             final ista step size <= eps*(initial ista step size)\n"
           "-v : set the verbose level (default 0)\n"
           "	0 -- no output\n"
           "	1 -- outer iteration\n"
           "	2 -- sufficient decrease iteration\n"
           "	3 -- coordinate descent iteration\n"
           );
    exit(1);
}


void parse_command_line(int argc, const char * argv[],
                        Parameter* param)
{
    
    // default value
    param->l = 10;
    param->work_size = 500;
    param->max_iter = 1000;
    param->lmd = 0.001;
    param->max_inner_iter = 100;
    param->opt_inner_tol = 5*1e-6;
    param->opt_outer_tol = 1e-7;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = LHAC_MSG_NO;
    param->sd_flag = 1; // default using suffcient decrease
    param->shrink = 4;
    param->fileName = new char[MAX_LENS];
    param->rho = 0.01;
    param->cd_rate = 5;
    param->active_set = STD;
//    param->active_set = GREEDY_ADDZERO;
    param->loss = LOG;
    param->isCached = false;
    param->posweight = 1.0;
    param->dense = 1;
    param->mdlexist = 0;
    param->method_flag = 2;
    
    // parse options
    int i;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc) {
            printf("wrong input format\n");
            exit_with_help();
        }
			
		switch(argv[i-1][1])
		{
			case 'c':
				param->lmd = atof(argv[i]);
				break;
                
            case 'i':
                param->max_iter = atoi(argv[i]);
                break;
                
            case 'r':
                param->rho = atof(argv[i]);
                break;
                
            case 'v':
                param->verbose = atoi(argv[i]);
                break;
                
            case 'g':
                param->shrink = atof(argv[i]);
                break;
                
            case 'l':
                if (strcmp(argv[i],"square")==0) param->loss = SQUARE;
                else if (strcmp(argv[i],"log")==0) param->loss = LOG;
                break;
                
            case 'a':
                if (atoi(argv[i])!=0) param->isCached = true;
                else param->isCached = false;
                break;
                
            /* solving precision */
            case 'e':
                param->opt_outer_tol = atof(argv[i]);
                break;
                
            case 's':
                param->posweight = atof(argv[i]);
                break;
            
            /* testing file */
            case 'p':
                param->pfile = new char[MAX_LENS];
                strcpy(param->pfile, argv[i]);
                break;
                
            case 'm':
                param->mdlexist = atoi(argv[i]);
                break;
                
            case 'd':
                param->dense = atoi(argv[i]);
                break;
                
            case 'b':
                param->l = atoi(argv[i]);
                break;
                
            case 'q':
                param->method_flag = atoi(argv[i]);
                break;

			default:
				printf("unknown option: -%c\n", argv[i-1][1]);
				break;
		}
	}
    
    if(i>=argc) {
        printf("wrong input format\n");
        exit_with_help();
    }
    
    strcpy(param->fileName, argv[i]);

}

template <typename Derived>
const Solution* optimize(Parameter* param) {
    Objective<Derived>* obj = new Derived(param);
    //    Solution* sols = lhac(obj, param);
    LHAC<Derived>* Alg = new LHAC<Derived>(obj, param);
    Solution* sols = Alg->solve();
    delete obj;
    delete Alg;

    return sols;
}


void predict(const double* w, const unsigned long p, const Parameter* param) {
    if (w != NULL) {
        printf("w\t");
        for (unsigned long i = 0; i < p-1; i++) {
            printf("%f,", w[i]);
        }
        printf("%f\n", w[p-1]);
    }
    if (param->pfile == NULL) return;
//    double* w = sols->w;
//    unsigned long p = sols->p;
//    char* line = new char[MAX_LINE_LEN];
    FILE *fp = fopen(param->pfile,"r");
    if(fp == NULL)
    {
        printf("can't open input file %s\n",param->pfile);
        exit(1);
    }
    unsigned long N = 0;
    unsigned long posN = 0;
    unsigned long negN = 0;
    max_line_len = MAX_LINE_LEN;
    if (line == NULL) {
        line = Malloc(char,max_line_len);
    }
    while(readline(fp)!=NULL) {
        char* label = strtok(line," \t\n");
        char* endptr;
        
        double y_true = strtod(label,&endptr);
        char *idx, *val;
        int index;
        double value;
        double wx = 0.0;
        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");
            
            if(val == NULL)
                break;
            
            index = (int) strtol(idx,&endptr,10);
            value = strtod(val,&endptr);
            
            /* index is not included in training set */
            /* treat w[index] as zero  */
            if (index > p) continue;
            wx += w[index-1]*value;
        }
        if (y_true > 0) {
            N++;
            if (y_true*wx > 0) posN++;
        }
        else
            if (wx > 0) negN++;
        
    }
    printf("total pos\t%ld\n", N);
    printf("neg\t%f\n", (double) negN / N);
    printf("pos\t%f\n", (double) posN / N);
    free(line);
}


void read_model(double* &w, unsigned long &p, Parameter* param) {
    FILE *fp = fopen(param->fileName,"r");
    if(fp == NULL)
    {
        printf("can't open model file %s\n",param->fileName);
        exit(1);
    }
    max_line_len = MAX_LINE_LEN;
    if (line == NULL) {
        line = Malloc(char,max_line_len);
    }
    bool foundw = false;
    p = 0;
    while(readline(fp)!=NULL) {
        char* label = strtok(line," \t\n");
        // find the line starting with w
        if (strcmp(label, "w")) continue;
        foundw = true;
        char* val;
        while (1) {
            val = strtok(NULL," ,");
            if (val != NULL) p++;
                else break;
        }
    }
    if (!foundw) {
        printf("error: model not found in %s!\n", param->fileName);
        exit(1);
    }
    rewind(fp);
    w = new double[p];
    while(readline(fp)!=NULL) {
        char* endptr;
        char* label = strtok(line," \t\n");
        if (strcmp(label, "w")) continue;
        char* val;
        unsigned long i = 0;
        while (1) {
            val = strtok(NULL," ,");
            if (val == NULL) break;
            w[i++] = strtod(val,&endptr);
        }
    }
    return;

}


int main(int argc, const char * argv[])
{
    Parameter* param = new Parameter;
    parse_command_line(argc, argv, param);
    const Solution* sols = NULL;
    
    if (param->mdlexist) {
        double* w = NULL;
        unsigned long p = 0;
        read_model(w, p, param);
        predict(w, p, param);
        exit(0);
    }
    
    switch (param->loss) {
        case SQUARE:
            printf("L1 - square\n");
            sols = optimize<Lasso>(param);
            break;
            
        case LOG:
            printf("L1 - logistic\n");
            sols = optimize<LogReg>(param);
            break;
            
        default:
            printf("Unknown loss: logistic or square!\n");
            break;
    }
    if (sols != NULL && param != NULL) {
//        predict(sols, param);
        predict(sols->w, sols->p, param);
        delete sols;
        delete param;
    }
    
    exit( 0 );
}












