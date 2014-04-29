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
#include "Objective.h"



/* Parameters */
#define _P 4000
#define _N 5000
#define _NNZ_PERC 0.1




void libsvmExperiment(Parameter* param)
{

    
    
    Objective* obj = new Objective(param);
//        write2mat("gisette_x", "X", Dset->X, Dset->N, Dset->p);
//        write2mat("gisette_y", "y", Dset->y, Dset->N, 1);
    
    
//    l1log* mdl = new l1log(Dset, param);
    //    l1log* mdl = new l1log(Dset);
    
    solution* sols = lhac(obj, param);
    
    

    releaseSolution(sols);
    


    delete param;
    return;
}

void parse_command_line(int argc, const char * argv[],
                        Parameter* param)
{
    
    // default value
    param->l = 10;
    param->work_size = 1;
    param->max_iter = 500;
    param->lmd = 1;
    param->max_inner_iter = 100;
    param->opt_inner_tol = 5*1e-6;
    param->opt_outer_tol = 1e-6;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = LHAC_MSG_CD;
    param->sd_flag = 1; // default using suffcient decrease
    param->shrink = 4;
    param->fileName = new char[MAX_LENS];
    param->rho = 0.5;
    param->cd_rate = 5;
    param->active_set = STD;
    
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
				param->lmd = atoi(argv[i]);
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
                
            /* solving precision */
            case 'e':
                param->opt_outer_tol = atof(argv[i]);
                break;
                
			default:
				printf("unknown option: -%c\n", argv[i-1][1]);
				break;
		}
	}
    
    
}



int main(int argc, const char * argv[])
{
    Parameter* param = new Parameter;
    parse_command_line(argc, argv, param);
    
    libsvmExperiment(param);
    
    
    delete [] cparam->fileName;
    delete cparam;
    exit( 0 );
}
