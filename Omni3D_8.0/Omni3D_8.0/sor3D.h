#include<afx.h>
#ifndef _INC_SOR3D
#define _INC_SOR3D

extern "C" void sor3D(int Xsize,int Ysize,int Zsize,int NoItr,int *BCScheme,double*x,double*y,double*z,double* p,double *pn,double *RHS,double eps);

#endif  //_INC_PRESSURE2D