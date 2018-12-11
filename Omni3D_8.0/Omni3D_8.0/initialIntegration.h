#ifndef _INC_INITIALINTEGRATION
#define _INC_INITIALINTEGRATION
extern "C" void initialIntegration(long Xsize,long Ysize,long Zsize,double deltx,double delty,double deltz,double density,double* DuDt,double * DvDt,double* DwDt,double pref,double *p,double* pn,double* pBC);

#endif