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
#include "myUtilities.h"
#include <CoreFoundation/CoreFoundation.h>

/* Auxiliary routines prototypes */
//extern void print_matrix( char* desc, int m, int n, double* a, int lda );
//extern void print_int_vector( char* desc, int n, int* a );


/* Parameters */
#define _P 4000
#define _N 5000
#define _NNZ_PERC 0.1



//int main(int argc, const char * argv[])
//{
//    training_set_sp* Dset_col = new training_set_sp;
//    readLibsvm("test_file", Dset_col);
//    printout("Dset_col = ", Dset_col, COL_MAJOR);
//    
//    exit(0);
//}

void libsvmExperiment(command_line_param* cparam)
{
    training_set_sp* Dset_sp = new training_set_sp;
    readLibsvm(cparam->fileName, Dset_sp);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", Dset_sp->p, Dset_sp->N, Dset_sp->nnz);
    
    l1log_param* param = new l1log_param;
    param->l = 10;
    param->work_size = 1;
    param->max_iter = cparam->max_iter;
    param->lmd = cparam->lmd;
    param->max_inner_iter = 100;
    param->opt_inner_tol = 5*1e-6;
    param->opt_outer_tol = cparam->opt_outer_tol;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = cparam->verbose;
    param->sd_flag = cparam->sd_flag;
    param->shrink = cparam->shrink;
    param->fileName = cparam->fileName;
    param->rho = cparam->rho;
    param->cd_rate = 5;
    param->active_set = STD;
//    param->active_set = GREEDY;
    
    /* elapsed time (not cputime) */
    time_t start;
    time_t end;
    time(&start);
    double elapsedtime = 0;
    
//    training_set* Dset = new training_set;
//    training_set_sp* Dset_sp = new training_set_sp;
//    readMatFiles(fileName, Dset);
//    transformToSparseFormat(Dset, Dset_sp);
    
    if (cparam->dense) {
        training_set* Dset = new training_set;
        transformToDenseFormat(Dset, Dset_sp);
        releaseProb(Dset_sp);
        
        solution* sols;
        
        l1log* mdl = new l1log(Dset, param);
        //    l1log* mdl = new l1log(Dset);
        
        sols = lhac(mdl);
        
        printout("logs = ", sols, param);
        
        //    releaseProb(Dset);
        releaseProb(Dset);
        releaseSolution(sols);
    }
    else {
        solution* sols;
        
        l1log* mdl = new l1log(Dset_sp, param);
        //    l1log* mdl = new l1log(Dset);
        
        sols = lhac(mdl);
        
        printout("logs = ", sols, param);
        
        //    releaseProb(Dset);
        releaseProb(Dset_sp);
        releaseSolution(sols);
    }
    
    time(&end);
    elapsedtime = difftime(end, start);
    printf("%.f seconds\n", elapsedtime);

    delete param;
    return;
}

void parse_command_line(int argc, const char * argv[],
                        command_line_param* cparam)
{
    
    // default value
    cparam->dense = 1;
    cparam->lmd = 1;
    cparam->max_iter = 400;
    cparam->randomData = 0;
    cparam->random_p = 0;
    cparam->random_N = 0;
    cparam->fileName = new char[MAX_LENS];
    cparam->verbose = LHAC_MSG_CD;
    cparam->sd_flag = 1; // default using suffcient decrease
    cparam->shrink = 4; // default no shrink on gama
    cparam->rho = 0.5;
    cparam->opt_outer_tol = 1e-7;
    
    // parse options
    int i;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc) {
            printf("wrong input format\n");
            exit(1);
        }
			
		switch(argv[i-1][1])
		{
			case 'c':
				cparam->lmd = atoi(argv[i]);
				break;
                
            case 'i':
                cparam->max_iter = atoi(argv[i]);
                break;
                
            case 'd':
                cparam->dense = atoi(argv[i]);
                break;
                
            case 'r':
                cparam->rho = atof(argv[i]);
                break;
                
            case 'v':
                cparam->verbose = atoi(argv[i]);
                break;
                
            case 's':
                cparam->alg = atoi(argv[i]);
                break;
                
            case 'l':
                cparam->sd_flag = atoi(argv[i]);
                break;
                
            case 'g':
                cparam->shrink = atof(argv[i]);
                break;
                
            /* solving precision */
            case 'e':
                cparam->opt_outer_tol = atof(argv[i]);
                break;
                
			default:
				printf("unknown option: -%c\n", argv[i-1][1]);
				break;
		}
	}
    
    
    if (cparam->randomData != 1) {
        // determine filenames
        if(i>=argc) {
            printf("wrong input format\n");
            exit(1);
        }
        
        strcpy(cparam->fileName, argv[i]);
    }
    
}



int main(int argc, const char * argv[])
{
    command_line_param* cparam = new command_line_param;
    parse_command_line(argc, argv, cparam);
    
    libsvmExperiment(cparam);
    
    
    delete [] cparam->fileName;
    delete cparam;
    exit( 0 );
}

/**** compare libsvm format with general format
 on randomly generated data sets ****/
//void randomExperiment(command_line_param* cparam)
//{
//    l1log_param* param = new l1log_param;
//    param->l = 10;
//    param->work_size = 10000;
//    param->max_iter = cparam->max_iter;
//    param->lmd = cparam->lmd;
//    param->max_inner_iter = 20;
//    param->opt_inner_tol = 0.05;
//    param->opt_outer_tol = 1e-5;
//    param->max_linesearch_iter = 1000;
//    param->bbeta = 0.5;
//    param->ssigma = 0.001;
//    param->verbose = cparam->verbose;
//    
//    if (cparam->dense) {
//        training_set* Dset = new training_set;
//        generateRandomProb(Dset, cparam->random_p,
//                           cparam->random_N, cparam->nnz_perc);
//        l1log* mdl = new l1log(Dset, param);
//        
//        solution* sols;
//        sols = lhac(mdl);
//        
//        printout("logs = ", sols);
//        
//        releaseSolution(sols);
//        
//        releaseProb(Dset);
//    }
//    else {
//        training_set_sp* Dset_sp = new training_set_sp;
//        training_set* Dset = new training_set;
//        generateRandomProb(Dset_sp, Dset, cparam->random_p,
//                           cparam->random_N, cparam->nnz_perc);
//        l1log* mdl = new l1log(Dset_sp, param);
//        
//        solution* sols;
//        sols = lhac(mdl);
//        
//        printout("logs = ", sols);
//        
//        releaseSolution(sols);
//        releaseProb(Dset);
//        releaseProb(Dset_sp);
//    }
//    
//    
//    delete param;
//    return;
//    
//    
//}

