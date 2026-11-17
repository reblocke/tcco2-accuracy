#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
// Unnormalized sum for each exact requested offset in a chain block.
// Kernel coefficients and the evidence normalization are supplied by Python.
int mixture(int n,int p,const double *mean,const double *inv_sd,const double *log_coeff,
            const double *offsets,double *out){
    double *work=malloc(sizeof(double)*n);
    if(!work)return 1;
    for(int j=0;j<p;j++){
        for(int i=0;i<n;i++){
            double z=(offsets[j]-mean[i])*inv_sd[i];
            work[i]=log_coeff[i]-.5*z*z;
        }
        vvexp(work,work,&n);
        vDSP_sveD(work,1,out+j,n);
    }
    free(work);
    return 0;
}
