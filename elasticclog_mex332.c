#include "math.h"
#include "mex.h"
#include <stdlib.h>
#include "matrix.h"
#include <stdio.h>
#include <time.h>


#define VERYSMALL 0.000000000001
#define MAXITER 1000

  

/*__inline void matVector(double* a, double* b, double* c);
__inline void matVector2(double* a, double* b, double* c);
__inline void copyActiveset(char* activeset, char* oldset);
__inline void createNewSet(char* newset, const char value);
__inline char setDifference(char* activeset,char* oldset);
__inline void copyBeta(double* beta, double* beta_old) ;
__inline double computeDifference(double* beta, double* beta_old); */
/* __inline void matVector(double* a, double* b, double* c,const int nelem_a,const int nelem_b);
__inline void matVector2(double* a, double* b, double* c,const int nelem_a,const int nelem_b); */

__inline void copyActiveset(char* activeset, char* oldset, const int nvar);
__inline void createNewSet(char* newset, const char value, const int nvar);
__inline char setDifference(char* activeset,char* oldset, const int nvar);
__inline void copyBeta(double* beta, double* beta_old, const int nvar) ;
__inline double computeDifference(double* beta, double* beta_old, const int nvar); 
/*                       double* U, double* Qbeta , double* T, double* Z, char* activeset, 
                          int* nhood, int* nn, double lambda1, double lambda2);
void coorddescent(char* activeset, char* newset, double* V, double* U, double* beta, 
                  double* beta_old, double* w, int* nhood, double* cdiff, int* nn, double lambda2,double nu);
*/
/* Fast elasticnet-graphnet solver for Matlab
% 
% (C) 2015 Jussi Tohka 
% Department of Signal Processing,
% Tampere University of Technology, Finland
% jussi.tohka at tut.fi */



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   
   int iteration_counter;
   char notchanged;
   char *activeset,*newset,*oldset;
   mxArray *mx_activeset, *mx_oldset,*mx_newset;
   char *ifscaled;
   
   double U,V;
   double *beta,*mbeta; /* return variables */
   double *beta_old,*r,*T,*c,*w,*Z; /* internal variables */
   mxArray *mx_beta,*mx_beta_old,*mx_r,*mx_T,*mx_c,*mx_w,*mx_Z; /* internal variables */
   double *initialbeta; /* input variables not global */
   double *cnv; 
   double lambda1,lambda2,nu,olambda1,olambda2;
   const int *dims;
   double tol;
   double ptild,reg; /* for quadratic_approximation */
   int i,iterator,iii,j,seqnum;
   const double *X,*Xsquared,*Y,*nuseq;
   int *nhood,*nn;
   int maxn;
   const int offset = 1; /* if offset is computed too */   
   int nvar,nsamples;
   int nnu;
   char fullcycle;
   
   X = mxGetPr(prhs[0]);
   Y = mxGetPr(prhs[1]);
   dims= mxGetDimensions(prhs[0]);
   nvar = dims[0];
   nsamples = dims[1]; 
   initialbeta = mxGetPr(prhs[2]);
   lambda1 = (double)mxGetScalar(prhs[3]);
   lambda2 = (double)mxGetScalar(prhs[4]);
   nuseq = mxGetPr(prhs[5]);/*(double)mxGetScalar(prhs[5]); */
   nnu = mxGetN(prhs[5]);
   nn = (int*)mxGetData(prhs[6]);
   nhood = (int*)mxGetData(prhs[7]);
   maxn = mxGetM(prhs[7]);
   Xsquared = mxGetPr(prhs[8]);
   ifscaled = (char*)mxGetData(prhs[9]);
   olambda1 = lambda1;
   olambda2 = lambda2;
   
   
   plhs[0]    = mxCreateDoubleMatrix(nvar, nnu, mxREAL);
   mbeta = mxGetPr(plhs[0]);
   plhs[1]    = mxCreateDoubleMatrix(MAXITER,1, mxREAL);
   cnv = mxGetPr(plhs[1]);
  /*  printf("nnu %d nsamples %d \n",nnu, nsamples); */
   
   mx_activeset = mxCreateNumericArray(1,&nvar,mxINT8_CLASS, mxREAL);
   activeset = (char*)mxGetData(mx_activeset);
   mx_oldset = mxCreateNumericArray(1,&nvar,mxINT8_CLASS, mxREAL);
   oldset = (char*)mxGetData(mx_oldset);
   mx_newset = mxCreateNumericArray(1,&nvar,mxINT8_CLASS, mxREAL);
   newset = (char*)mxGetData(mx_newset);
   
   mx_beta = mxCreateNumericArray(1,&nvar,mxDOUBLE_CLASS, mxREAL);
   beta = mxGetPr(mx_beta);
   mx_beta_old = mxCreateNumericArray(1,&nvar,mxDOUBLE_CLASS, mxREAL);
   beta_old = mxGetPr(mx_beta_old);
  /* mx_U = mxCreateNumericArray(1,&nvar,mxDOUBLE_CLASS, mxREAL);
   U = mxGetPr(mx_U); */
   mx_r = mxCreateNumericArray(1,&nsamples,mxDOUBLE_CLASS, mxREAL);  
   r = mxGetPr(mx_r);
  /* mx_V = mxCreateNumericArray(1,&nvar,mxDOUBLE_CLASS, mxREAL);
   V = mxGetPr(mx_V); */
   
   mx_T = mxCreateNumericArray(1,&nvar,mxDOUBLE_CLASS, mxREAL);
   T = mxGetPr(mx_T); 
   
   
   mx_c = mxCreateNumericArray(1,&nsamples,mxDOUBLE_CLASS, mxREAL);
   c = mxGetPr(mx_c);
   mx_w = mxCreateNumericArray(1,&nsamples,mxDOUBLE_CLASS, mxREAL);
   w = mxGetPr(mx_w);
   mx_Z = mxCreateNumericArray(1,&nsamples,mxDOUBLE_CLASS, mxREAL);
   Z = mxGetPr(mx_Z);
  /* mx_cdiff = mxCreateNumericArray(1,&nsamples,mxDOUBLE_CLASS, mxREAL);
   cdiff = mxGetPr(mx_cdiff); */
  /* printf("nvar %d nsamples %d \n",nvar, nsamples); */
   copyBeta(initialbeta,beta,nvar);
   
 
   for(seqnum = 0;seqnum < nnu;seqnum++) {    
   nu = nuseq[seqnum]; 
   if(ifscaled[0]) lambda1 = olambda1*nu;
   if(ifscaled[1]) lambda2 = olambda2*nu;
   printf("nu %f  lambda1 %f lambda2 %f \n",nu,lambda1,lambda2);
   iteration_counter = 0;
   notchanged = 0;
   tol = 0.000001;
   createNewSet(activeset,1,nvar);
   copyBeta(beta,beta_old,nvar);
   
   
   for(i = 0;i < nvar;i++)  {
            T[i] = lambda2*((double)nn[i]);
   }         
   
   copyActiveset(activeset,oldset,nvar);

  /* matVector(beta,X,c,nvar,nsamples); */
   fullcycle = 0;
   while((notchanged < 1) && (iteration_counter < MAXITER)) {
        iteration_counter++;
        
        
 /*    quadratic_approximation(beta,c,w,V,U,Qbeta,T,Z,activeset,nhood, nn,lambda1,lambda2); */
 /*      matVector(beta,X,c,nvar,nsamples); */
       for(i = 0;i < nvar;i++)  {
            if(fabs(beta[i]) > VERYSMALL) {
                activeset[i] = 1;
            } 
            else {
                activeset[i] = 0;
            }    
        } 
       iterator = 0; 
       for(j = 0;j < nsamples;j++)  {
          c[j] = 0;
       }   
       for(i = 0;i < nvar;i++)  {
           if(activeset[i]) {
               iterator = i;
               for(j = 0;j < nsamples;j++)  {
                   c[j] = c[j] + beta[i]*X[iterator];  /* check in what order X is stored; if not in the correct order then  */
                   iterator += nvar;
               }
           }
       }     
       for(i = 0;i < nsamples;i++)  {   
            ptild = 1/(1 + exp(- c[i]));
            if(ptild < 0.00001) ptild = 0;
            if(ptild > 0.99999) ptild = 1;
            w[i] = ptild*(1 - ptild); 
            if(w[i] < 0.00001) { 
                w[i] = 0.00001; 
            }  
            Z[i] = c[i] + (Y[i] - ptild)/w[i];
          
            r[i] = Z[i] - c[i];
            
         }   
        
        if(fullcycle) {
            copyActiveset(activeset,oldset,nvar);
            createNewSet(activeset,1,nvar);    
        }
            
  /*   coorddescent(activeset,newset,V,U,beta, beta_old,w, nhood, cdiff,nn,lambda2,nu); */
        
       createNewSet(newset,0,nvar);

       for(i = nvar - 1;i > (-1);i--)  { /* backtrack */
           if(activeset[i] == 1) {
               if(i == (nvar - offset)) {
                   V = 0;
                   iterator = i;
                   for(j = 0;j < nsamples;j++)  {       
                        V += w[j]*(r[j] + X[iterator]*beta_old[i])*X[iterator]; /* this can be simplified */
                        iterator += nvar;
                   }
                   U = 0;
                   iterator = i;
                   for(j = 0;j < nsamples;j++)  {
                       U = U + w[j]*Xsquared[iterator];  /* check in what order X is stored; if not in the correct order then  */
                       iterator += nvar;
                   } 
                   
                   beta[i] = (V)/U;
                   newset[i] = 1;
               }    
               else {
                   /* compute the gradient */
                 /*  V[i] = sum w_jr_j*x_ij + reg; */
                   beta[i] = 0;
                   V = 0;
                   iterator = i;
                   for(j = 0;j < nsamples;j++)  {       
                        V += w[j]*(r[j] + X[iterator]*beta_old[i])*X[iterator];
             /*  ctmp = (c[j] - X[iterator]*beta[i])*w[j];
               Qbeta[i] += X[iterator]*(Z[j] - ctmp); */
                        iterator += nvar;
                   }        
                   reg = 0;
                   for(iii = 0;iii < nn[i];iii++)  {
                       reg = reg + beta[nhood[i*maxn + iii]];
                   }
                   reg = reg*lambda2;
                   V += reg;
                   /* compute U */
                   U = T[i];
                   iterator = i;
                   for(j = 0;j < nsamples;j++)  {
                       U = U + w[j]*Xsquared[iterator];  /* check in what order X is stored; if not in the correct order then  */
                       iterator += nvar;
                   } 
                   U += lambda1;
                   
                   if((fabs(V) > nu)) {
                       if(V > 0) {
                           beta[i] = (V - nu)/U;
                       }
                       else {
                           beta[i] = (V + nu)/U;
                       }
                       newset[i] = 1;
                    }
               } 
               if(fabs(beta[i] -beta_old[i]) > 0) {
                   /* update r */
                  /* r_j = r_j - x_ijb_i; */
                  iterator = i;
                  for(j = 0;j < nsamples;j++)  {
                      r[j] -= X[iterator]*(beta[i] - beta_old[i]);
                      iterator += nvar;
                  }   
               }  

                
           }
       }   
       copyActiveset(newset,activeset,nvar);
       /* end coord descent */
       if(fullcycle) {
           notchanged = setDifference(activeset,oldset,nvar);
       }    
       cnv[iteration_counter] = computeDifference(beta,beta_old,nvar); 
       if((cnv[iteration_counter] < tol) && (!fullcycle)) {  
           fullcycle = 1;
       } 
       else {
           fullcycle = 0;
       }

     copyBeta(beta,beta_old,nvar);
             
   } 
   /* printf("beta %f %f %f %f \n",beta[0],beta[1],beta[2],beta[3]); */
    copyBeta(beta,&mbeta[seqnum*nvar],nvar); 
    printf("%d iterations \n",iteration_counter);
   } /* end of the main for loop */
   
   return; 
}


/*   product of a vector and matrix */
/*
void matVector(double* a, double* b, double* c ,const int nelem_a, const int nelem_b) 
{
    int iterator,i,j;

    
   iterator = 0; 
   for(i = 0;i < nelem_b;i++)  {
      c[i] = 0;
      for(j = 0;j < nelem_a;j++)  {
          c[i] = c[i] + a[j]*b[iterator];  
          iterator++;
      }
   }
   return;
}


void matVector2(double* a, double* b, double* c, const int nelem_a,const int nelem_b) 
{
    int iterator,i,j;
   
    
   for(i = 0;i < nelem_b;i++)  {
      c[i] = 0;
      iterator = i;
      for(j = 0;j < nelem_a;j++)  {
        
          c[i] = c[i] + a[j]*b[iterator];  
          iterator += nelem_b;
      }     
   }
   return;
}

*/

/* void quadratic_approximation(double* beta, double* c, double* w, double* V, 
                       double* U, double* Qbeta , double* T, double* Z, char* activeset, 
                          int* nhood, int* nn, double lambda1, double lambda2)
{

    double ptild,reg;
    int i,iterator,iii,j;
    double ctmp;
    
   
    return;   
}    
*/
/* void coorddescent(char* activeset, char* newset, double* V, double* U, double* beta, 
                  double* beta_old, double* w, int* nhood, double* cdiff, int* nn, double lambda2,double nu)                      
{
   int i,j,iii,k,iterator;
   
   
/*   printf("cd V %f  V %f beta %f beta %f \n",V[0],V[1],beta[0],beta[1]); */
/*   return;
}    
*/
void copyActiveset(char* activeset, char* oldset ,const int nvar)  
{
   int i;   
    for(i = 0;i < nvar;i++)  { 
         oldset[i] = activeset[i]; 
    }
   return;
}

void createNewSet(char* newset, const char value, const int nvar) 
{
   int i;   
    for(i = 0;i < nvar;i++)  { 
         newset[i] = value; 
    }
    return;
}    


void copyBeta(double* beta, double* beta_old, const int nvar) 
{
    int i;   
    for(i = 0;i < nvar;i++)  { 
         beta_old[i] = beta[i]; 
    }
    return;
}    

double computeDifference(double* beta, double* beta_old, const int nvar) 
{
   int i;
   double returnValue;
   returnValue = 0;
   for(i = 0;i < nvar;i++)  {
       returnValue += fabs(beta[i] - beta_old[i]);
   }
   return returnValue;
}

char setDifference(char* activeset,char* oldset,const int nvar) 
{
   char notchanged;
   int iter;
   
   notchanged = 1;
   iter = 0;
   while(notchanged && (iter < nvar)) {
       if(activeset[iter] != oldset[iter]) {
           notchanged = 0;
       }
       iter++;
   }
   return(notchanged);
}    


    