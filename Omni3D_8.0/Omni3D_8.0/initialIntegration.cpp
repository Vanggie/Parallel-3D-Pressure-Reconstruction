#include"initialIntegration.h"
#include<afx.h>
#include<math.h>
#include<iostream>
#define weight1 0
#define weight2 0
#define weight 0.8
using namespace std;
extern "C" void initialIntegration(long Xsize,long Ysize,long Zsize,double deltx,double delty,double deltz,double density,double* DuDt,double * DvDt,double* DwDt,double pref,double *p,double* pn,double* pBC)
{
	p[0]=pref;
	pn[0]=pref;
	int i,j,k,temp;
	double rms;
	double Error=0;
	//No.1 0-Xsize/2;0-Ysize/2,0-Zsize/2;
	for(int k=Zsize/2;k>=0;k--)
	{
		for(int j=Ysize/2;j>=0;j--)
		{
			for(int i=Xsize/2;i>=0;i--)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No.2 Xsize/2-Xsize;0-Ysize/2,0-Zsize/2;
	for(int k=Zsize/2;k>=0;k--)
	{
		for(int j=Ysize/2;j>=0;j--)
		{
			for(int i=Xsize/2;i<Xsize;i++)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No.3 0-Xsize/2;Ysize/2-Ysize,0-Zsize/2;
	for(int k=Zsize/2;k>=0;k--)
	{
		for(int j=Ysize/2;j<Ysize;j++)
		{
			for(int i=Xsize/2;i>=0;i--)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No.4 Xsize/2-Xsize;Ysize/2-Ysize,0-Zsize/2;
	for(int k=Zsize/2;k>=0;k--)
	{
		for(int j=Ysize/2;j<Ysize;j++)
		{
			for(int i=Xsize/2;i<Xsize;i++)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind+Xsize*Ysize]+0.5*(DwDt[ind]+DwDt[ind+Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No.5 0-Xsize/2;0-Ysize/2,Zsize/2-Zsize;
	for(int k=Zsize/2;k<Zsize;k++)
	{
		for(int j=Ysize/2;j>=0;j--)
		{
			for(int i=Xsize/2;i>=0;i--)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No.6 0-Xsize/2;0-Ysize/2,Zsize/2-Zsize;
	for(int k=Zsize/2;k<Zsize;k++)
	{
		for(int j=Ysize/2;j>=0;j--)
		{
			for(int i=Xsize/2;i<Xsize;i++)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind+Xsize]+0.5*(DvDt[ind]+DvDt[ind+Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No.7 0-Xsize/2;Ysize/2-Ysize,Zsize/2-Zsize;
	for(int k=Zsize/2;k<Zsize;k++)
	{
		for(int j=Ysize/2;j<Ysize;j++)
		{
			for(int i=Xsize/2;i>=0;i--)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind+1]+0.5*(DuDt[ind]+DuDt[ind+1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
	//No. 8 Xsize/2-Xsize;Ysize/2-Ysize,Zsize/2-Zsize;
	for(int k=Zsize/2;k<Zsize;k++)
	{
		for(int j=Ysize/2;j<Ysize;j++)
		{
			for(int i=Xsize/2;i<Xsize;i++)
			{
				int ind=i+j*Xsize+k*Xsize*Ysize;
				if(j==Ysize/2&&k==Zsize/2&&i!=Xsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
				}
				if(j==Ysize/2&&i==Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
				}
				if(i==Xsize/2&&k==Zsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
				}
				if(i==Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(j==Ysize/2&&i!=Xsize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(k==Zsize/2&&i!=Xsize/2&&j!=Ysize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]=pn[ind]/2.0;
				}
				if(i!=Xsize/2&&j!=Ysize/2&&k!=Zsize/2)
				{
					pn[ind]=pn[ind-1]-0.5*(DuDt[ind]+DuDt[ind-1])*deltx*density;
					pn[ind]+=pn[ind-Xsize]-0.5*(DvDt[ind]+DvDt[ind-Xsize])*delty*density;
					pn[ind]+=pn[ind-Xsize*Ysize]-0.5*(DwDt[ind]+DwDt[ind-Xsize*Ysize])*deltz*density;
					pn[ind]=pn[ind]/3.0;
				}
				p[ind]=pn[ind];
			}
		}
	}
}