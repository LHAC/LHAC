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
#include "l1log.h"
#include "Lbfgs.h"
#include "lhac.h"
#include "myUtilities.h"

/* Auxiliary routines prototypes */
//extern void print_matrix( char* desc, int m, int n, double* a, int lda );
//extern void print_int_vector( char* desc, int n, int* a );


/* Parameters */
#define _P 4000
#define _N 5000
#define _NNZ_PERC 0.1
//#define NRHS 3
//#define LDA 5
//#define LDB 5





/*
    INPUT:  Sm,Tm,Lm,Dm
    OUTPUT: Q,Q_bar,gama
*/

//int main(int argc, const char * argv[])
//{
//    training_set_sp* Dset_col = new training_set_sp;
//    readLibsvm("test_file", Dset_col);
//    printout("Dset_col = ", Dset_col, COL_MAJOR);
//    
//    exit(0);
//}

solution* lhac(l1log* mdl)
{
    double timeBegin = mdl->timeBegin;
    
    l1log_param* param = mdl->param;
    unsigned long l = param->l;
    double opt_outer_tol = param->opt_outer_tol;
    unsigned short max_iter = param->max_iter;
    
    solution* sols = new solution;
    sols->fval = new double[max_iter];
    sols->normgs = new double[max_iter];
    sols->t = new double[max_iter];
    
    unsigned long p = mdl->p;
    
    LBFGS* lR = new LBFGS(p, l);
    
    lR->initData(mdl->w, mdl->w_prev, mdl->L_grad, mdl->L_grad_prev);
    
    // active set
    work_set_struct* work_set = new work_set_struct;
    work_set->idxs = new ushort_pair_t[p];
    
    double normsg = mdl->computeSubgradient();
    unsigned short newton_iter;
    for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
        mdl->iter = newton_iter;
        
        mdl->computeWorkSet(work_set);
        
        double elapsedTime = (clock() - timeBegin)/CLOCKS_PER_SEC;
        sols->fval[newton_iter-1] = mdl->f_current;
        sols->normgs[newton_iter-1] = normsg;
        sols->t[newton_iter-1] = elapsedTime;
        if (mdl->MSG >= LHAC_MSG_NEWTON) {
            printf("%.4e  iter %2d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                   elapsedTime, newton_iter, mdl->f_current, normsg, work_set->numActive);
        }
        
        lR->computeLowRankApprox();
        
//        computeLBFGS(Q, Q_bar, R, Sm, Tm, Lm, Dm, &gama);
        
//        write2mat("w.mat", "w", mdl->w, mdl->p, 1);
//        write2mat("L_grad.mat", "L_grad", mdl->L_grad, mdl->p, 1);
        
//        write2mat("Sm.mat", "Sm", Sm);
//        write2mat("Tm.mat", "Tm", Tm);
//        write2mat("Lm.mat", "Lm", Lm);
//        write2mat("Dm.mat", "Dm", Dm, Lm->rows, 1);
//        write2mat("Qm.mat", "Qm", Q, mdl->p, 2*(Lm->rows));
//        write2mat("Qm_bar.mat", "Qm_bar", Q_bar, 2*(Lm->rows), mdl->p);

        mdl->coordinateDsecent(lR, work_set);
//        mdl->coordinateDescent(Q_bar, Q, gama, work_set, 2*Sm->cols);
        
        double eTime = clock();
        mdl->lineSearch();
        eTime = (clock() - eTime)/CLOCKS_PER_SEC;
//        printf("%.4e\n", eTime);
        
        memcpy(mdl->L_grad_prev, mdl->L_grad, p*sizeof(double));
        
        double gradientTime = clock();
        mdl->computeGradient();
        gradientTime = (clock() - gradientTime)/CLOCKS_PER_SEC;
//        printf("%.4e\n", gradientTime);
        
        /* update LBFGS */
        lR->updateLBFGS(mdl->w, mdl->w_prev, mdl->L_grad, mdl->L_grad_prev);
        
        //        printout("Sm = ", Sm,FULL);
        //        printout("Tm = ", Tm, FULL);
        //        printout("Lm = ", Lm, FULL);
        //        printout("Dm = ", Dm, Lm->rows, FULL);
        
        
        normsg = mdl->computeSubgradient();
        if (normsg <= opt_outer_tol*mdl->normsg0) {
            break;
        }
        
    }
    
    sols->size = newton_iter;
    
    
    delete mdl;
    delete lR;
//    delete Tm;
//    delete Lm;
//    delete Sm;
//    delete [] Dm;
//    delete [] Q;
//    delete [] Q_bar;
//    delete [] R;
//    delete [] buff;
    delete [] work_set->idxs;
    delete work_set;
    return sols;
}


/**** compare libsvm format with general format
       on randomly generated data sets ****/
void randomExperiment(command_line_param* cparam)
{
    l1log_param* param = new l1log_param;
    param->l = 10;
    param->work_size = 10000;
    param->max_iter = cparam->max_iter;
    param->lmd = cparam->lmd;
    param->max_inner_iter = 20;
    param->opt_inner_tol = 0.05;
    param->opt_outer_tol = 1e-5;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = cparam->verbose;
    
    if (cparam->dense) {
        training_set* Dset = new training_set;
        generateRandomProb(Dset, cparam->random_p,
                           cparam->random_N, cparam->nnz_perc);
        l1log* mdl = new l1log(Dset, param);
        
        solution* sols;
        sols = lhac(mdl);
        
        printout("logs = ", sols);
        
        releaseSolution(sols);
        
        releaseProb(Dset);
    }
    else {
        training_set_sp* Dset_sp = new training_set_sp;
        training_set* Dset = new training_set;
        generateRandomProb(Dset_sp, Dset, cparam->random_p,
                           cparam->random_N, cparam->nnz_perc);
        l1log* mdl = new l1log(Dset_sp, param);
        
        solution* sols;
        sols = lhac(mdl);
        
        printout("logs = ", sols);
        
        releaseSolution(sols);
        releaseProb(Dset);
        releaseProb(Dset_sp);
    }
    
    
    delete param;
    return;

   
}

void libsvmExperiment(command_line_param* cparam)
{
    training_set_sp* Dset_sp = new training_set_sp;
    readLibsvm(cparam->fileName, Dset_sp);
    
    /* statistics of the problem */
    printf("p = %ld, N = %ld, nnz = %ld\n", Dset_sp->p, Dset_sp->N, Dset_sp->nnz);
    
    l1log_param* param = new l1log_param;
    param->l = 10;
    param->work_size = 10000;
    param->max_iter = cparam->max_iter;
    param->lmd = cparam->lmd;
    param->max_inner_iter = 20;
    param->opt_inner_tol = 0.05;
    param->opt_outer_tol = 1e-5;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = cparam->verbose;
    
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
        
        printout("logs = ", sols);
        
        //    releaseProb(Dset);
        releaseProb(Dset);
        releaseSolution(sols);
    }
    else {
        solution* sols;
        
        l1log* mdl = new l1log(Dset_sp, param);
        //    l1log* mdl = new l1log(Dset);
        
        sols = lhac(mdl);
        
        printout("logs = ", sols);
        
        //    releaseProb(Dset);
        releaseProb(Dset_sp);
        releaseSolution(sols);
    }

    delete param;
    return;
}

void parse_command_line(int argc, const char * argv[],
                        command_line_param* cparam)
{
    
    // default value
    cparam->dense = 0;
    cparam->lmd = 0.5;
    cparam->max_iter = 300;
    cparam->randomData = 0;
    cparam->random_p = 0;
    cparam->random_N = 0;
    cparam->fileName = new char[MAX_LENS];
    cparam->verbose = LHAC_MSG_CD;
    
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
                cparam->randomData = 1;
                cparam->random_p = atoi(argv[i]);
                if(++i>=argc) {
                    printf("wrong input format\n");
                    exit(1);
                }
                cparam->random_N = atoi(argv[i]);
                if(++i>=argc) {
                    printf("wrong input format\n");
                    exit(1);
                }
                cparam->nnz_perc = atof(argv[i]);
                break;
                
            case 'v':
                cparam->verbose = atoi(argv[i]);
                break;
                
            case 's':
                cparam->alg = atoi(argv[i]);
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

