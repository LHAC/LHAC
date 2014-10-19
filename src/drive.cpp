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
    param->opt_outer_tol = 1e-7;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = LHAC_MSG_CD;
    param->sd_flag = 1; // default using suffcient decrease
    param->shrink = 4;
    param->fileName = new char[MAX_LENS];
    param->rho = 0.01;
    param->cd_rate = 5;
    param->active_set = STD;
    param->loss = SQUARE;
    param->isCached = true;
    
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
                
			default:
				printf("unknown option: -%c\n", argv[i-1][1]);
				break;
		}
	}
    
    if(i>=argc) {
        printf("wrong input format\n");
        exit(1);
    }
    
    strcpy(param->fileName, argv[i]);

}

template <typename Derived>
Solution* optimize(Parameter* param) {
    Objective<Derived>* obj = new Derived(param);
    //    Solution* sols = lhac(obj, param);
    LHAC<Derived>* Alg = new LHAC<Derived>(obj, param);
    Solution* sols = Alg->solve();
    delete obj;
    delete Alg;

    return sols;
}



int main(int argc, const char * argv[])
{
    Parameter* param = new Parameter;
    parse_command_line(argc, argv, param);
    Solution* sols = NULL;
    
    
    switch (param->loss) {
        case SQUARE:
            printf("L1 - square\n");
            sols = optimize<Lasso>(param);
            delete sols;
            break;
            
        case LOG:
            printf("L1 - logistic\n");
            sols = optimize<LogReg>(param);
            delete sols;
            break;
            
        default:
            printf("Unknown loss: logistic or square!\n");
            break;
    }
    exit( 0 );
}
