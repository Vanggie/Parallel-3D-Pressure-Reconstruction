#ifndef _INC_CALRHS
#define _INC_CALRHS

extern "C" void calRHS(int Xsize,int Ysize,int Zsize,double deltx,double delty,double deltz,double density,double* DuDt,double*DvDt,double*DwDt,double *RHS);

#endif  //_INC_PRESSURE2D