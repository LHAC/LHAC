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
#include "Lbfgs.h"
#include "lhac.h"
#include "myUtilities.h"
#include "sics_lhac.h"


void sicsExperiment(command_line_param* cparam)
{
    double* S = NULL;
    unsigned long p;
    readMatFiles(cparam->fileName, S, &p);
//    readTextFile(cparam->fileName, S, &p);
    
    param* _param = new param;
    
    
    double lmdi;
    _param->l = 10;
    _param->work_size = 100;
    _param->max_iter = cparam->max_iter;
    lmdi = cparam->lmd;
    _param->max_inner_iter = 50;
    _param->opt_inner_tol = 0.05;
    _param->opt_outer_tol = cparam->opt_outer_tol;
    _param->max_linesearch_iter = 1000;
    _param->bbeta = 0.5;
    _param->ssigma = 0.001;
    _param->verbose = cparam->verbose;
    _param->sd_flag = cparam->sd_flag;
    _param->shrink = cparam->shrink;
    _param->fileName = cparam->fileName;
    _param->rho = cparam->rho;
    _param->cd_rate = 15;
    
    double* lmd_vec = new double[p*p];
    for (unsigned long i = 0; i < p*p; i++) {
        lmd_vec[i] = lmdi;
    }
    _param->lmd = lmd_vec;
    
    solution* sols;
    sols = sics_lhac(S, p, _param);
    
    printout("logs = ", sols, _param);
    
    releaseSolution(sols);
    
//    delete [] S;
    
    return;
}

void parse_command_line(int argc, const char * argv[],
                        command_line_param* cparam)
{
    
    // default value
    cparam->dense = 0;
    cparam->lmd = 0.5;
    cparam->max_iter = 1000;
    cparam->randomData = 0;
    cparam->random_p = 0;
    cparam->random_N = 0;
    cparam->fileName = new char[MAX_LENS];
    cparam->verbose = LHAC_MSG_CD;
    cparam->opt_outer_tol = 1e-6;
    cparam->sd_flag = 1; // default using suffcient decrease
    cparam->shrink = 4; // default gama = gama/4
    cparam->rho = 0.01;
    
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
				cparam->lmd = atof(argv[i]);
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
                
            /* printout level */
            case 'v':
                cparam->verbose = atoi(argv[i]);
                break;
            
            /* solving precision */
            case 'e':
                cparam->opt_outer_tol = atof(argv[i]);
                break;
                
            case 'l':
                cparam->sd_flag = atoi(argv[i]);
                break;
                
            case 'g':
                cparam->shrink = atof(argv[i]);
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
    
    sicsExperiment(cparam);
    
    delete [] cparam->fileName;
    delete cparam;
    
    exit( 0 );
}
