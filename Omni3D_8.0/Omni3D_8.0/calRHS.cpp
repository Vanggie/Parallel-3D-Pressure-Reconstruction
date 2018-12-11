#include"calRHS.h"
#include<afx.h>
#include<math.h>
#define weight1 0
#define weight2 0
#define weight 0.8
extern "C" void calRHS(int Xsize,int Ysize,int Zsize,double deltx,double delty,double deltz,double density,double* DuDt,double*DvDt,double*DwDt,double *RHS)
{
	int i,j,k,temp;
	RHS=new double[Xsize*Ysize*Zsize];
	memset(RHS,0,Xsize*Ysize*Zsize*sizeof(double));
	for(int k=0;k<Zsize;k++)
	{
		for(int j=0;j<Ysize;j++)
		{
			for(int i=0;i<Xsize;i++)
			{
				int index=i+j*Xsize+k*Xsize*Ysize;
				if(i==0&&j!=0&&j!=Ysize-1&&k!=0&&k!=Zsize-1)//left face;
				{
					//RHS[index]=(-DuDt[i+j*Xsize+k*Xsize*Ysize]-DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=(DuDt[i+j*Xsize+k*Xsize*Ysize]+DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5*deltx;
					RHS[index]=RHS[index]*density;

				}
				if(i==Xsize-1&&j!=0&&j!=Ysize-1&&k!=0&&k!=Zsize-1)//right face
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(j==0&&i!=0&&i!=Xsize-1&&k!=0&&k!=Zsize-1)//top face
				{

					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(j==Ysize-1&&i!=0&&i!=Xsize-1&&k!=0&&k!=Zsize-1)//bottom face
				{
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(k==0&&i!=0&&i!=Xsize-1&&j!=0&&j!=Ysize-1)
				{

					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(k==Zsize-1&&i!=0&&i!=Xsize-1&&j!=0&&j!=Ysize-1)//Back face
				{
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==0&&j==0&&k!=0&&k!=Zsize-1)//x=0 y=0 side
				{
					//RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==0&&k==0&&j!=0&&j!=Ysize-1)
				{
					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==0&&k==Zsize-1&&j!=0&&j!=Ysize-1)
				{

					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;

					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&j==0&&k!=0&&k!=Zsize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;

					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&j==Ysize-1&&k!=0&&k!=Zsize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&k==0&&j!=0&&j!=Ysize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;

					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&k==Zsize-1&&j!=0&&j!=Ysize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;

					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(j==0&&k==0&&i!=0&&i!=Xsize-1)
				{

					//RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(j==0&&k==Zsize-1&&i!=0&&i!=Xsize-1)
				{

					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(j==Ysize-1&&k==0&&i!=0&&i!=Xsize-1)
				{

					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(j==Ysize-1&&k==Zsize-1&&i!=0&&i!=Xsize-1)
				{

					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==0&&j==0&&k==0)
				{
					//RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(i==0&&j==0&&k==Zsize-1)
				{
					//RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==0&&j==Ysize-1&&k==0)
				{
					//RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(i==0&&j==Ysize-1&&k==Zsize-1)
				{
					//RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&j==0&&k==0)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&j==0&&k==Zsize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&j==Ysize-1&&k==0)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(i==Xsize-1&&j==Ysize-1&&k==Zsize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					//RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					//RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]=RHS[index]*density;
				}
				if(i!=0&&i!=Xsize-1&&j!=0&&j!=Ysize-1&&k!=0&&k!=Zsize-1)
				{
					RHS[index]=(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize])*0.5;
					RHS[index]+=(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize])*0.5;
					RHS[index]+=(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize])*0.5;

					RHS[index]+=0.25*weight1*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+(j-1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i-1+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+(j+1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i-1+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j-1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+1+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j+1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+1+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+(j)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i-1+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i-1+(j)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+j*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i-1+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j)*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+(j)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j)*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+1+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+1+(j)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j)*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+(j)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j)*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+1+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+1+(j)*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+j*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+(j-1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+j*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+(j-1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j)*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j)*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+(j+1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.25*weight1*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j)*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.25*weight1*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j)*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+(j+1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]);


					RHS[index]+=0.125*weight2*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+(j-1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i-1+(j-1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i-1+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+j*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i-1+j*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i-1+(j-1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i-1+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j-1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i-1+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i-1+(j-1)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+(j+1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]-deltx*DuDt[i-1+(j+1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i-1+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+j*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i-1+j*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i-1+(j+1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i-1+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j+1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i-1+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i-1+(j+1)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+(j-1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+j*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i-1+(j-1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i-1+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+j*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i-1+(j-1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i-1+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j-1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i-1+(j-1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i-1+(j-1)*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j-1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+j*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+1+(j-1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+1+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+j*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+1+j*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]-delty*DvDt[i+1+(j-1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+1+j*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+(j-1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]-deltz*DwDt[i+1+j*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+(j-1)*Xsize+(k-1)*Xsize*Ysize]-deltz*DwDt[i+1+(j-1)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j+1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+j*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]+deltx*DuDt[i+1+(j+1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+1+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+j*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+1+j*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]+delty*DvDt[i+1+(j+1)*Xsize+(k-1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(-deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+-deltz*DwDt[i+1+j*Xsize+k*Xsize*Ysize]+-deltz*DwDt[i+(j+1)*Xsize+k*Xsize*Ysize]+-deltz*DwDt[i+j*Xsize+(k-1)*Xsize*Ysize]+-deltz*DwDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]+-deltz*DwDt[i+1+j*Xsize+(k-1)*Xsize*Ysize]+-deltz*DwDt[i+(j+1)*Xsize+(k-1)*Xsize*Ysize]+-deltz*DwDt[i+1+(j+1)*Xsize+(k-1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j-1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+1+(j-1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+1+j*Xsize+k*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+j*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]-delty*DvDt[i+1+j*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]-delty*DvDt[i+1+(j-1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+1+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j-1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+1+(j-1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+1+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+(j-1)*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+1+(j-1)*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(-deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+(j+1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i+j*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]-deltx*DuDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]-deltx*DuDt[i-1+(j+1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i-1+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+j*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i-1+(j+1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i-1+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j+1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i-1+(j+1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i-1+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i-1+(j+1)*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]+=0.125*weight2*(deltx*DuDt[i+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+(j+1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]+deltx*DuDt[i+1+j*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]+deltx*DuDt[i+1+(j+1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(delty*DvDt[i+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+1+j*Xsize+k*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+j*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]+delty*DvDt[i+1+j*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]+delty*DvDt[i+1+(j+1)*Xsize+(k+1)*Xsize*Ysize]);
					RHS[index]+=0.125*weight2*(deltz*DwDt[i+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+1+j*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+(j+1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+1+(j+1)*Xsize+k*Xsize*Ysize]+deltz*DwDt[i+1+j*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+(j+1)*Xsize+(k+1)*Xsize*Ysize]+deltz*DwDt[i+1+(j+1)*Xsize+(k+1)*Xsize*Ysize]);

					RHS[index]=RHS[index]*density;
				}
			}	
		}
	}	
	

}