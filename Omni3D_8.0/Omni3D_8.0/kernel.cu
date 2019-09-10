#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <iomanip>
#include <afx.h>
#include <time.h>
#include"calRHS.h"
#include"sor3D.h"
#include"initialIntegration.h"
#include "io.h"
#include <list>
#include <vector>
using namespace std;
#define weight1 0
#define weight2 0
#define weight 0.8
#define zero 1e-7
#define PI 3.1415926535897932384626433832795
using namespace std;
using std::vector;
//Get all the files in filepath
// Returns false on success, true on error
// Return true if the folder exists, false otherwise
bool folderExists(CString folderName) //Examine if a folder exist
{
	std::string s = CT2A(folderName);
	if (_access(s.c_str(), 0) == -1) {
		//File not found
		return false;
	}

	DWORD attr = GetFileAttributes(folderName);
	if (!(attr & FILE_ATTRIBUTE_DIRECTORY)) {
		// File is not a directory
		return false;
	}

	return true;
}

bool createFolder(CString folderName) {
	list<std::string> folderLevels;
	//    char* c_str = (char*)folderName.c_str();
	if (folderExists(folderName))
	{
		return false;
	}
	else
	{
		CreateDirectory((LPCTSTR)folderName, NULL);
		return true;
	}
	/*Point to end of the string
	char* strPtr = &c_str[strlen(c_str) - 1];

	// Create a list of the folders which do not currently exist
	do {
	if (folderExists(c_str)) {
	break;
	}
	// Break off the last folder name, store in folderLevels list
	do {
	strPtr--;
	} while ((*strPtr != '\\') && (*strPtr != '/') && (strPtr >= c_str));
	folderLevels.push_front(string(strPtr + 1));
	strPtr[1] = 0;
	} while (strPtr >= c_str);

	if (_chdir(c_str)) {
	return true;
	}

	// Create the folders iteratively
	for (list<std::string>::iterator it = folderLevels.begin(); it != folderLevels.end(); it++) {
	if (CreateDirectory((LPCTSTR)it->c_str(), NULL) == 0) {
	return true;
	}
	_chdir(it->c_str());
	}

	return false;*/
}

void getFiles(string path, vector<string> & f)
{
	FILE* pipe = NULL;
	string pCmd = "dir /B /S " + string(path);
	string tmp;
	char buf[1024];

	if (NULL == (pipe = _popen(pCmd.c_str(), "rt")))
	{
		cout << "Shit" << endl;
		return;
	}

	while (!feof(pipe))
	{
		if (fgets(buf, 1024, pipe) != NULL)
		{
			tmp = string(buf);
			tmp.erase(tmp.find_last_not_of("\n") + 1);
			f.push_back(tmp);
		}

	}

	_pclose(pipe);


}

__device__ __host__
void ntoijk(long Xsize, long Ysize, long Zsize, long nout, int* i, int*j, int*k)
{
	int iout, jout, kout;
	if (nout <= Xsize*Ysize - 1)
	{
		kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
	}
	if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
	{
		iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
	}
	if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
	{
		kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
	}
	if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
	{
		jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
		iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
		iout = Xsize - 2 - iout;
		kout = kout + 1;
	}
	if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
	{
		iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
		kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
		kout = Zsize - 2 - kout;
		jout = jout + 1;
	}
	if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
	{
		jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
		iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
		kout = Zsize - 2 - kout;
		iout = iout + 1;
	}
	i[0] = iout; j[0] = jout; k[0] = kout;
}
__device__ __host__ bool crosspoint(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, float k1, float k2, float k3, int* i, int* j, int* k)
{
	int iout, jout, kout;
	float r, x, y, z;
	r = 0; x = 0; y = 0; z = 0;
	bool flag = 0;

	/////case 1, vertical to x-axis
	if (k1 == 0 && k2 != 0 && k3 != 0)
	{
		if (iin >= 0 && iin <= Xsize - 1)
		{
			////four crossing point;y=0;y=max;z=0;z=max;
			r = (0 - jin) / k2; y = 0; z = kin + k3*r;
			if (z <= Zsize - 1 && z >= 0 && r != 0 && flag == 0)//cross y=0;
			{
				iout = iin;
				jout = 0;
				kout = floor(z + 0.5);
				flag = 1;
			}
			r = (Ysize - 1 - jin) / k2; y = Ysize - 1; z = kin + k3*r;
			if (z <= Zsize - 1 && z >= 0 && r != 0 && flag == 0)//y=max;
			{
				iout = iin;
				jout = Ysize - 1;
				kout = floor(z + 0.5);
				flag = 1;
			}
			r = (0 - kin) / k3; z = 0; y = jin + k2*r;
			if (y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)//z=0;
			{

				iout = iin;
				jout = floor(y + 0.5);
				kout = 0;

				flag = 1;
			}
			r = (Zsize - 1 - kin) / k3; z = Zsize - 1; y = jin + k2*r;
			if (y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)
			{
				iout = iin;
				jout = floor(y + 0.5);
				kout = Zsize - 1;
				flag = 1;
			}

		}
		if (iin == Xsize - 1 || iin == 0)
		{
			int jout1 = jin;
			int kout1 = kin;
			r = (0 - jin) / k2; y = 0; float z = kin + k3*r;
			bool flag2 = 0;
			if (z <= Zsize - 1 && z >= 0 && r != 0 && flag == 0)//cross y=0;
			{
				if (flag2 == 0){
					iout = iin;
					jout = 0;
					kout = floor(z + 0.5);
					flag2 = 1;
				}
				else
				{
					iout = iin;
					jout1 = 0;
					kout1 = floor(z + 0.5);
				}
				flag = 1;
			}
			r = (Ysize - 1 - jin) / k2; y = Ysize - 1; z = kin + k3*r;
			if (z <= Zsize - 1 && z >= 0 && r != 0 && flag == 0)//y=max;
			{
				if (flag2 == 0)
				{
					iout = iin;
					jout = Ysize - 1;
					kout = floor(z + 0.5);
					flag2 = 1;
				}
				else
				{
					iout = iin;
					jout1 = Ysize - 1;
					kout1 = floor(z + 0.5);
				}
				flag = 1;
			}
			r = (0 - kin) / k3; z = 0; y = jin + k2*r;
			if (y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)//z=0;
			{
				if (flag2 == 0)
				{
					iout = iin;
					jout = floor(y + 0.5);
					kout = 0;
					flag2 = 1;
				}
				else
				{
					iout = iin;
					jout1 = floor(y + 0.5);
					kout1 = 0;
				}


				flag = 1;
			}
			r = (Zsize - 1 - kin) / k3; z = Zsize - 1; y = jin + k2*r;
			if (y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)
			{
				if (flag2 == 0)
				{
					iout = iin;
					jout = floor(y + 0.5);
					kout = Zsize - 1;
					flag2 = 1;
				}
				else
				{
					iout = iin;
					jout1 = floor(y + 0.5);
					kout1 = Zsize - 1;

				}
				flag = 1;
			}
			if ((jout1 - jin)*(jout1 - jin) + (kout1 - kin)*(kout1 - kin)>(jout - jin)*(jout - jin) + (kout - kin)*(kout - kin))
			{
				jout = jout1; kout = kout1;
			}
		}
	}
	///case 2, vertical to y-axis
	if (k1 != 0 && k2 == 0 && k3 != 0)
	{
		if (jin >= 0 && jin <= Ysize - 1)
		{
			////four crossing point
			r = (0 - iin) / k1; x = 0; z = kin + k3*r;//x=0;
			if (z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
			{
				iout = 0;
				jout = jin;
				kout = floor(z + 0.5);
				flag = 1;
			}
			r = (Xsize - 1 - iin) / k1; x = Xsize - 1; z = kin + k3*r;//x=max
			if (z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
			{
				iout = Xsize - 1;
				jout = jin;
				kout = floor(z + 0.5);
				flag = 1;
			}
			r = (0 - kin) / k3; z = 0; x = iin + k1*r;//z=0;
			if (x <= Xsize - 1 && x >= 0 && r != 0 && flag == 0)
			{
				iout = floor(x + 0.5);
				jout = jin;
				kout = 0;
				flag = 1;
			}
			r = (Zsize - 1 - kin) / k3; z = Zsize - 1; x = iin + k1*r;//z=max;
			if (x <= Xsize - 1 && x >= 0 && r != 0 && flag == 0)
			{
				iout = floor(x + 0.5);
				jout = jin;
				kout = Zsize - 1;
				flag = 1;
			}

		}
		if (jin == 0 || jin == Ysize - 1)
		{
			int iout1 = iin;
			int kout1 = kin;
			bool flag2 = 0;
			r = (0 - iin) / k1; x = 0; z = kin + k3*r;//x=0;
			if (z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
			{
				if (flag2 == 0)
				{
					iout = 0;
					jout = jin;
					kout = floor(z + 0.5);
					flag2 = 1;
				}
				else
				{
					iout1 = 0;
					jout = jin;
					kout1 = floor(z + 0.5);
				}
				flag = 1;
			}
			r = (Xsize - 1 - iin) / k1; x = Xsize - 1; z = kin + k3*r;//x=max
			if (z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
			{
				if (flag2 == 0)
				{
					iout = Xsize - 1;
					jout = jin;
					kout = floor(z + 0.5);
					flag2 = 1;
				}
				else
				{
					iout1 = Xsize - 1;
					jout = jin;
					kout1 = floor(z + 0.5);
				}
				flag = 1;
			}
			r = (0 - kin) / k3; z = 0; x = iin + k1*r;//z=0;
			if (x <= Xsize - 1 && x >= 0 && r != 0 && flag == 0)
			{
				if (flag2 == 0)
				{
					iout = floor(x + 0.5);
					jout = jin;
					kout = 0;
					flag2 = 1;
				}
				else
				{
					iout1 = int(x + 0.5);
					jout = jin;
					kout1 = 0;
				}
				flag = 1;
			}
			r = (Zsize - 1 - kin) / k3; z = Zsize - 1; x = iin + k1*r;//z=max;
			if (x <= Xsize - 1 && x >= 0 && r != 0 && flag == 0)
			{
				if (flag2 == 0)
				{
					iout = floor(x + 0.5);
					jout = jin;
					kout = Zsize - 1;
					flag2 = 1;
				}
				else
				{
					iout1 = floor(x + 0.5);
					jout = jin;
					kout1 = Zsize - 1;
				}
				flag = 1;
			}
			if ((iout1 - iin)*(iout1 - iin) + (kout1 - kin)*(kout1 - kin)>(iout - iin)*(iout - iin) + (kout - kin)*(kout - kin))
			{
				iout = iout1; kout = kout1;
			}
		}
	}
	///case 3, vertical to z-axis
	if (k1 != 0 && k2 != 0 && k3 == 0)
	{
		if (kin >= 0 && kin <= Zsize - 1)
		{
			////four crossing point
			r = (0 - iin) / k1; x = 0; y = jin + k2*r;//x=0;
			if (y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)
			{
				iout = 0;
				jout = floor(y + 0.5);
				kout = kin;
				flag = 1;
			}
			r = (Xsize - 1 - iin) / k1; x = Xsize - 1; y = jin + k2*r;//x=max;
			if (y <= Ysize - 1 && y >= 0 && r != 0 && flag == 0)
			{
				iout = Xsize - 1;
				jout = floor(y + 0.5);
				kout = kin;
				flag = 1;
			}
			r = (0 - jin) / k2; y = 0; x = iin + k1*r;//y=0;
			if (x <= Xsize - 1 && x >= 0 && r != 0 && flag == 0)
			{
				iout = floor(x + 0.5);
				jout = 0;
				kout = kin;
				flag = 1;
			}
			r = (Ysize - 1 - jin) / k2; y = Ysize - 1; x = iin + k1*r;//y=max;
			if (x <= Xsize - 1 && x >= 0 && flag == 0 && r != 0)
			{
				iout = floor(x + 0.5);
				jout = Ysize - 1;
				kout = kin;
				flag = 1;
			}

		}
		if (kin == 0 || kin == Zsize - 1)
		{
			int iout1 = iin;
			int jout1 = jin;
			bool flag2 = 0;
			r = (0 - iin) / k1; x = 0; y = jin + k2*r;//x=0;
			if (y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)
			{
				if (flag2 == 0)
				{
					iout = 0;
					jout = floor(y + 0.5);
					kout = kin;
					flag2 = 1;
				}
				else
				{
					iout1 = 0;
					jout1 = floor(y + 0.5);
					kout = kin;
				}

				flag = 1;
			}
			r = (Xsize - 1 - iin) / k1; x = Xsize - 1; y = jin + k2*r;//x=max;
			if (y <= Ysize - 1 && y >= 0 && r != 0 && flag == 0)
			{
				if (flag2 == 0)
				{
					iout = Xsize - 1;
					jout = floor(y + 0.5);
					kout = kin;
					flag2 = 1;
				}
				else
				{
					iout1 = Xsize - 1;
					jout1 = floor(y + 0.5);
					kout = kin;
				}
				flag = 1;
			}
			r = (0 - jin) / k2; y = 0; x = iin + k1*r;//y=0;
			if (x <= Xsize - 1 && x >= 0 && r != 0 && flag == 0)
			{
				if (flag == 0)
				{
					iout = floor(x + 0.5);
					jout = 0;
					kout = kin;
					flag2 = 1;
				}
				else
				{
					iout1 = floor(x + 0.5);
					jout1 = 0;
					kout = kin;
				}
				flag = 1;
			}
			r = (Ysize - 1 - jin) / k2; y = Ysize - 1; x = iin + k1*r;//y=max;
			if (x <= Xsize - 1 && x >= 0 && flag == 0 && r != 0)
			{
				if (flag2 == 0)
				{
					iout = floor(x + 0.5);
					jout = Ysize - 1;
					kout = kin;
					flag2 = 1;
				}
				else
				{
					iout1 = floor(x + 0.5);
					jout1 = Ysize - 1;
					kout = kin;
				}
				flag = 1;
			}
			if ((iout1 - iin)*(iout1 - iin) + (jout1 - jin)*(jout1 - jin)>(iout - iin)*(iout - iin) + (jout - jin)*(jout - jin))
			{
				iout = iout1; jout = jout1;
			}
		}
	}
	///case 4, vertical to plane IJ
	if (k1 == 0 && k2 == 0 && k3 != 0 && flag == 0)
	{

		if (iin <= Xsize - 1 && iin >= 0 && jin <= Ysize - 1 && jin >= 0)
		{
			iout = iin;
			jout = jin;
			if (kin<Zsize / 2)
			{
				kout = Zsize - 1;
			}
			else
			{
				kout = 0;
			}

			flag = 1;
		}

	}
	///case 5, vertical to IK plane
	if (k1 == 0 && k2 != 0 && k3 == 0 && flag == 0)
	{
		if (iin >= 0 && iin <= Xsize - 1 && kin >= 0 && kin <= Zsize - 1)
		{
			iout = iin; kout = kin;
			if (jin<Ysize / 2)
			{
				jout = Ysize - 1;
			}
			else
			{
				jout = 0;
			}
			flag = 1;
		}

	}
	///case 6, vertical to JK plane
	if (k1 != 0 && k2 == 0 && k3 == 0 && flag == 0)
	{
		if (jin >= 0 && jin <= Ysize - 1 && kin >= 0 && kin <= Zsize - 1)
		{
			jout = jin; kout = kin;
			if (iin<Xsize / 2)
			{
				iout = Xsize - 1;
			}
			else
			{
				iout = 0;
			}
			flag = 1;
		}
	}
	/// case 7, purely inclined
	if (k1 != 0 && k2 != 0 && k3 != 0 && flag == 0)
	{
		/// six crossing point
		r = (0 - iin) / k1; x = 0; y = jin + k2*r; z = kin + k3*r;//x=0
		if (y <= Ysize - 1 && y >= 0 && z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
		{
			iout = 0;
			jout = floor(y + 0.5);
			kout = floor(z + 0.5);
			flag = 1;
		}
		r = (Xsize - 1 - iin) / k1; x = Xsize - 1; y = jin + k2*r; z = kin + k3*r;//x=max
		if (y <= Ysize - 1 && y >= 0 && z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
		{
			iout = Xsize - 1;
			jout = floor(y + 0.5);
			kout = floor(z + 0.5);
			flag = 1;
		}
		r = (0 - jin) / k2; x = iin + k1*r; y = 0; z = kin + k3*r;//y=0;
		if (x <= Xsize - 1 && x >= 0 && z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
		{
			iout = floor(x + 0.5);
			jout = 0;
			kout = floor(z + 0.5);
			flag = 1;
		}
		r = (Ysize - 1 - jin) / k2; x = iin + k1*r; y = Ysize - 1; z = kin + k3*r;//y=max
		if (x <= Xsize - 1 && x >= 0 && z <= Zsize - 1 && z >= 0 && flag == 0 && r != 0)
		{
			iout = floor(x + 0.5);
			jout = Ysize - 1;
			kout = floor(z + 0.5);
			flag = 1;
		}
		r = (0 - kin) / k3; x = iin + k1*r; y = jin + k2*r; z = 0;//z=0;
		if (x <= Xsize - 1 && x >= 0 && y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)
		{
			iout = floor(x + 0.5);
			jout = floor(y + 0.5);
			kout = 0;
			flag = 1;
		}
		r = (Zsize - 1 - kin) / k3; x = iin + k1*r; y = jin + k2*r; z = Zsize - 1;//z=max
		if (x <= Xsize - 1 && x >= 0 && y <= Ysize - 1 && y >= 0 && flag == 0 && r != 0)
		{
			iout = floor(x + 0.5);
			jout = floor(y + 0.5);
			kout = Zsize - 1;
			flag = 1;
		}

	}
	if (flag == 1)
	{
		i[0] = iout;
		j[0] = jout;
		k[0] = kout;
	}
	else
	{
		i[0] = iin;
		j[0] = jin;
		k[0] = kin;
	}
	return flag;
}
__device__ __host__ bool cross2point(long Xsize, long Ysize, long Zsize, int *iin, int *jin, int *kin, float xin, float yin, float zin, float k1, float k2, float k3, int* iout, int* jout, int* kout)
{
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	iin[0] = Xsize; jin[0] = Ysize; kin[0] = Zsize;
	iout[0] = Xsize; jout[0] = Ysize; kout[0] = Zsize;
	//	printf("%f %f %f %f",xin,yin,zin,sqrt(xin*xin+yin*yin+zin*zin));

	if (k1 == 0 && k2 != 0 && k3 != 0)
	{
		if (xin>-center_x&&xin<center_x)
		{
			////four crossing point;y=0;y=max;z=0;z=max;
			float r = (-center_y - yin) / k2; float y1 = -center_y; float z1 = zin + k3*r;
			r = (center_y - yin) / k2; float y2 = center_y; float z2 = zin + k3*r;
			r = (-center_z - zin) / k3; float z3 = -center_z; float y3 = yin + k2*r;
			r = (center_z - zin) / k3; float z4 = center_z; float y4 = yin + k2*r;
			bool flag = 0;
			if (z1 <= center_z&&z1 >= -center_z&&flag == 0)//cross y=0;
			{
				if (flag == 0)
				{
					iin[0] = floor(xin + center_x + 0.5);
					jin[0] = 0;
					kin[0] = floor(z1 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = floor(xin + center_x + 0.5);
					jout[0] = 0;
					kout[0] = floor(z1 + center_z + 0.5);
				}
				flag = 1;
			}
			if (z2 <= center_z&&z2 >= -center_z)//y=max;
			{
				if (flag == 0)
				{
					iin[0] = floor(xin + center_x + 0.5);
					jin[0] = Ysize - 1;
					kin[0] = floor(z2 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = floor(xin + center_x + 0.5);
					jout[0] = Ysize - 1;
					kout[0] = floor(z2 + center_z + 0.5);
				}
				flag = 1;
			}
			if (y3 <= center_y&&y3 >= -center_y)//z=0;
			{
				if (flag == 0)
				{
					iin[0] = floor(xin + center_x + 0.5);
					jin[0] = floor(y3 + center_y + 0.5);
					kin[0] = 0;
				}
				if (flag == 1)
				{
					iout[0] = floor(xin + center_x + 0.5);
					jout[0] = floor(y3 + center_y + 0.5);
					kout[0] = 0;
				}
				flag = 1;
			}
			if (y4 <= center_y&&y4 >= -center_y)
			{
				if (flag == 0)
				{
					iin[0] = floor(xin + center_x + 0.5);
					jin[0] = floor(y4 + center_y + 0.5);
					kin[0] = Zsize - 1;
				}
				if (flag == 1)
				{
					iout[0] = floor(xin + center_x + 0.5);
					jout[0] = floor(y4 + center_y + 0.5);
					kout[0] = Zsize - 1;
				}
			}
			//sorting intersection point by in, out order
			if (flag != 0)
			{
				if ((jout[0] - jin[0])*k2 + (kout[0] - kin[0])*k3<0)
				{
					int temp;
					temp = jin[0]; jin[0] = jout[0]; jout[0] = temp;
					temp = kin[0]; kin[0] = kout[0]; kout[0] = temp;
				}
			}
			return true;
		}
	}
	///case 2, vertical to y-axis
	if (k1 != 0 && k2 == 0 && k3 != 0)
	{
		if (yin>-center_y&&yin<center_y)
		{
			////four crossing point
			float r = (-center_x - xin) / k1; float x1 = -center_x; float z1 = zin + k3*r;//x=0;
			r = (center_x - xin) / k1; float x2 = center_x; float z2 = zin + k3*r;//x=max
			r = (-center_z - zin) / k3; float z3 = -center_z; float x3 = xin + k1*r;//z=0;
			r = (center_z - zin) / k3; float z4 = center_z; float x4 = xin + k1*r;//z=max;
			bool flag = 0;
			if (z1 <= center_z&&z1 >= -center_z)
			{
				if (flag == 0)
				{
					iin[0] = 0;
					jin[0] = floor(yin + center_y + 0.5);
					kin[0] = floor(z1 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = 0;
					jout[0] = floor(yin + center_y + 0.5);
					kout[0] = floor(z1 + center_z + 0.5);
				}
				flag = 1;
			}
			if (z2 <= center_z&&z2 >= -center_z)
			{
				if (flag == 0)
				{
					iin[0] = Xsize - 1;
					jin[0] = floor(yin + center_y + 0.5);
					kin[0] = floor(z2 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = Xsize - 1;
					jout[0] = floor(yin + center_y + 0.5);
					kout[0] = floor(z2 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x3 <= center_x&&x3 >= -center_x)
			{
				if (flag == 0)
				{
					iin[0] = floor(x3 + center_x + 0.5);
					jin[0] = floor(yin + center_y + 0.5);
					kin[0] = 0;
				}
				if (flag == 1)
				{
					iout[0] = floor(x3 + center_x + 0.5);
					jout[0] = floor(yin + center_y + 0.5);
					kout[0] = 0;
				}
				flag = 1;
			}
			if (x4 <= center_x&&x4 >= -center_x)
			{
				if (flag == 0)
				{
					iin[0] = floor(x4 + center_x + 0.5);
					jin[0] = floor(yin + center_y + 0.5);
					kin[0] = Zsize - 1;
				}
				if (flag == 1)
				{
					iout[0] = floor(x4 + center_x + 0.5);
					jout[0] = floor(yin + center_y + 0.5);
					kout[0] = Zsize - 1;
				}
				flag = 1;
			}
			//sorting intersection point by in, out order
			if (flag != 0)
			{
				if ((iout[0] - iin[0])*k1 + (kout[0] - kin[0])*k3<0)
				{
					int temp;
					temp = iin[0]; iin[0] = iout[0]; iout[0] = temp;
					temp = kin[0]; kin[0] = kout[0]; kout[0] = temp;
				}
			}
			return true;
		}
	}
	///case 3, vertical to z-axis
	if (k1 != 0 && k2 != 0 && k3 == 0)
	{
		if (zin>-center_z&&zin<center_z)
		{
			////four crossing point
			float r = (-center_x - xin) / k1; float x1 = -center_x; float y1 = yin + k2*r;//x=0;
			r = (center_x - xin) / k1; float x2 = center_x; float y2 = yin + k2*r;//x=max;
			r = (-center_y - zin) / k2; float y3 = -center_y; float x3 = xin + k1*r;//y=0;
			r = (center_y - zin) / k2; float y4 = center_y; float x4 = xin + k1*r;//y=max;
			bool flag = 0;
			if (y1 <= center_y&&y1 >= -center_y)
			{
				if (flag == 0)
				{
					iin[0] = 0;
					jin[0] = floor(y1 + center_y + 0.5);
					kin[0] = floor(zin + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = 0;
					jout[0] = floor(y1 + center_y + 0.5);
					kout[0] = floor(zin + center_z + 0.5);
				}
				flag = 1;
			}
			if (y2 <= center_y&&y2 >= -center_y)
			{
				if (flag == 0)
				{
					iin[0] = Xsize - 1;
					jin[0] = floor(y2 + center_y + 0.5);
					kin[0] = floor(zin + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = Xsize - 1;
					jout[0] = floor(y2 + center_y + 0.5);
					kout[0] = floor(zin + center_z + 0.5);
				}
				flag = 1;
			}
			if (x3 <= center_x&&x3 >= -center_x)
			{
				if (flag == 0)
				{
					iin[0] = floor(x3 + center_x + 0.5);
					jin[0] = 0;
					kin[0] = floor(zin + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = floor(x3 + center_x + 0.5);
					jout[0] = 0;
					kout[0] = floor(zin + center_z + 0.5);
				}
				flag = 1;
			}
			if (x4 <= center_x&&x4 >= -center_x)
			{
				if (flag == 0)
				{
					iin[0] = floor(x4 + center_x + 0.5);
					jin[0] = Ysize - 1;
					kin[0] = floor(zin + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout[0] = floor(x4 + center_x + 0.5);
					jout[0] = Ysize - 1;
					kout[0] = floor(zin + center_z + 0.5);
				}
				flag = 1;
			}
			//sorting intersection point by in, out order
			if (flag != 0)
			{
				if ((iout[0] - iin[0])*k1 + (jout[0] - jin[0])*k2<0)
				{
					int temp;
					temp = iin[0]; iin[0] = iout[0]; iout[0] = temp;
					temp = jin[0]; jin[0] = jout[0]; jout[0] = temp;
				}
			}
			return true;
		}

	}
	///case 4, vertical to plane IJ
	if (abs(k1)<zero&&abs(k2)<zero&&abs(k3) >= zero)
	{

		if (xin<center_x&&xin>-center_x&&yin<center_y&&yin>-center_y)
		{
			iin[0] = floor(xin + center_x + 0.5); iout[0] = iin[0];
			jin[0] = floor(yin + center_y + 0.5); jout[0] = jin[0];
			if (k3>0)
			{
				kin[0] = 0; kout[0] = Zsize - 1;
			}
			else{
				kin[0] = Zsize - 1; kout[0] = 0;
			}
			return true;
		}

	}
	///case 5, vertical to IK plane
	if (abs(k1)<zero&&abs(k2) >= zero&&abs(k3)<zero)
	{
		if (xin>-center_x&&xin<center_x&&zin>-center_z&&zin<center_z)
		{
			iin[0] = floor(xin + center_x + 0.5); iout[0] = iin[0];
			kin[0] = floor(zin + center_z + 0.5); kout[0] = kin[0];
			if (k2>0)
			{
				jout[0] = Ysize - 1; jin[0] = 0;
			}
			else
			{
				jin[0] = Ysize - 1; jout[0] = 0;
			}
			return true;
		}

	}
	///case 6, vertical to JK plane
	if (abs(k1) >= zero&&abs(k2)<zero&&abs(k3)<zero)
	{
		if (yin>-center_y&&yin<center_y&&zin>-center_z&&zin<center_z)
		{
			jin[0] = floor(yin + center_y + 0.5); jout[0] = jin[0];
			kin[0] = floor(zin + center_z + 0.5); kout[0] = kin[0];
			if (k1>0)
			{
				iout[0] = Xsize - 1; iin[0] = 0;
			}
			else
			{
				iin[0] = Xsize - 1; iout[0] = 0;
			}
		}
		return true;
	}
	/// case 7, purely inclined
	if (abs(k1) >= zero&&abs(k2) >= zero&&abs(k3) >= zero)
	{
		/// six crossing point
		float r;
		float x1, x2, x3, x4, x5, x6;
		float y1, y2, y3, y4, y5, y6;
		float z1, z2, z3, z4, z5, z6;
		r = (-center_x - xin) / k1; x1 = -center_x; y1 = yin + k2*r; z1 = zin + k3*r;//x=0
		r = (center_x - xin) / k1; x2 = center_x; y2 = yin + k2*r; z2 = zin + k3*r;//x=max
		r = (-center_y - yin) / k2; x3 = xin + k1*r; y3 = -center_y; z3 = zin + k3*r;//y=0;
		r = (center_y - yin) / k2; x4 = xin + k1*r; y4 = center_y; z4 = zin + k3*r;//y=max
		r = (-center_z - zin) / k3; x5 = xin + k1*r; y5 = yin + k2*r; z5 = -center_z;//z=0;
		r = (center_z - zin) / k3; x6 = xin + k1*r; y6 = yin + k2*r; z6 = center_z;//z=max
		bool flag = 0;
		if (y1 <= center_y&&y1 >= -center_y&&z1 <= center_z&&z1 >= -center_z)
		{
			if (flag == 0)
			{
				iin[0] = 0;
				jin[0] = floor(y1 + center_y + 0.5);
				kin[0] = floor(z1 + center_z + 0.5);
			}
			if (flag == 1)
			{
				iout[0] = 0;
				jout[0] = floor(y1 + center_y + 0.5);
				kout[0] = floor(z1 + center_z + 0.5);
			}
			flag = 1;
		}
		if (y2 <= center_y&&y2 >= -center_y&&z2 <= center_z&&z2 >= -center_z)
		{
			if (flag == 0)
			{
				iin[0] = Xsize - 1;
				jin[0] = floor(y2 + center_y + 0.5);
				kin[0] = floor(z2 + center_z + 0.5);
			}
			if (flag == 1)
			{
				iout[0] = Xsize - 1;
				jout[0] = floor(y2 + center_y + 0.5);
				kout[0] = floor(z2 + center_z + 0.5);
			}
			flag = 1;
		}
		if (x3 <= center_x&&x3 >= -center_x&&z3 <= center_z&&z3 >= -center_z)
		{
			if (flag == 0)
			{
				iin[0] = floor(x3 + center_x + 0.5);
				jin[0] = 0;
				kin[0] = floor(z3 + center_z + 0.5);
			}
			if (flag == 1)
			{
				iout[0] = floor(x3 + center_x + 0.5);
				jout[0] = 0;
				kout[0] = floor(z3 + center_z + 0.5);
			}
			flag = 1;
		}
		if (x4 <= center_x&&x4 >= -center_x&&z4 <= center_z&&z4 >= -center_z)
		{
			if (flag == 0)
			{
				iin[0] = floor(x4 + center_x + 0.5);
				jin[0] = Ysize - 1;
				kin[0] = floor(z4 + center_z + 0.5);
			}
			if (flag == 1)
			{
				iout[0] = floor(x4 + center_x + 0.5);
				jout[0] = Ysize - 1;
				kout[0] = floor(z4 + center_z + 0.5);
			}
			flag = 1;
		}
		if (x5 <= center_x&&x5 >= -center_x&&y5 <= center_y&&y5 >= -center_y)
		{
			if (flag == 0)
			{
				iin[0] = floor(x5 + center_x + 0.5);
				jin[0] = floor(y5 + center_y + 0.5);
				kin[0] = 0;
			}
			if (flag == 1)
			{
				iout[0] = floor(x5 + center_x + 0.5);
				jout[0] = floor(y5 + center_y + 0.5);
				kout[0] = 0;
			}
			flag = 1;
		}
		if (x6 <= center_x&&x6 >= -center_x&&y6 <= center_y&&y6 >= -center_y)
		{
			if (flag == 0)
			{
				iin[0] = floor(x6 + center_x + 0.5);
				jin[0] = floor(y6 + center_y + 0.5);
				kin[0] = Zsize - 1;
			}
			if (flag == 1)
			{
				iout[0] = floor(x6 + center_x + 0.5);
				jout[0] = floor(y6 + center_y + 0.5);
				kout[0] = Zsize - 1;
			}
			flag = 1;
		}
		//sorting intersection point by in, out order
		if ((iout[0] - iin[0])*k1 + (jout[0] - jin[0])*k2 + (kout[0] - kin[0])*k3<0)
		{
			int temp;
			temp = iin[0]; iin[0] = iout[0]; iout[0] = temp;
			temp = jin[0]; jin[0] = jout[0]; jout[0] = temp;
			temp = kin[0]; kin[0] = kout[0]; kout[0] = temp;
		}
		return true;
	}
	return false;
}
__device__ __host__ bool crosspoint2d(long Xsize, long Ysize, int iin, int jin, float k1, float k2, int *i, int *j)
{
	int iout, jout; bool flag = 0;
	if (k1 == 0 && k2 != 0)
	{
		iout = iin;
		if (jin == 0)
		{
			jout = Ysize - 1;
		}
		else
		{
			jout = 0;
		}
		if (iout == 0 || iout == Xsize - 1)
		{
			if (jin<Ysize / 2)
			{
				jout = Ysize - 1;
			}
			else
			{
				jout = 0;
			}
		}
		flag = 1;
	}
	if (k1 != 0 && k2 == 0)
	{
		jout = jin;
		if (iin == 0)
		{
			iout = Xsize - 1;
		}
		else
		{
			iout = 0;
		}
		if (jout == 0 || jout == Ysize - 1)
		{
			if (iin<Xsize / 2)
			{
				jout = Xsize - 1;
			}
			else
			{
				jout = 0;
			}
		}
		flag = 1;
	}
	if (k1 != 0 && k2 != 0)
	{
		float r, x, y;
		r = (0 - iin) / k1; y = k2*r + jin;
		if (y >= 0 && y <= Ysize - 1 && r != 0 && flag == 0)
		{
			iout = 0;
			jout = int(y + 0.5);
			flag = 1;
		}
		r = (Xsize - 1 - iin) / k1; y = k2*r + jin;
		if (y >= 0 && y <= Ysize - 1 && r != 0 && flag == 0)
		{
			iout = Xsize - 1;
			jout = int(y + 0.5);
			flag = 1;
		}
		r = (0 - jin) / k2; x = k1*r + iin;
		if (x >= 0 && x <= Xsize - 1 && r != 0 && flag == 0)
		{
			jout = 0;
			iout = int(x + 0.5);
			flag = 1;
		}
		r = (Ysize - 1 - jin) / k2; x = k1*r + iin;
		if (x >= 0 && x <= Xsize - 1 && r != 0 && flag == 0)
		{
			jout = Ysize - 1;
			iout = int(x + 0.5);
			flag = 1;
		}
	}
	if (flag == 1)
	{
		i[0] = iout;
		j[0] = jout;
	}
	return flag;

}
__device__ __host__ void ntoij2d(long Xsize, long Ysize, int nin, int *i, int *j)
{
	int iin, jin;
	if (nin <= Xsize - 1)
	{
		iin = nin; jin = 0;
	}
	if (nin>Xsize - 1 && nin <= Xsize + Ysize - 2)
	{
		iin = Xsize - 1; jin = nin - (Xsize - 1);
	}
	if (nin>Xsize + Ysize - 2 && nin <= 2 * Xsize + Ysize - 3)
	{
		iin = Xsize - 1 - (nin - (Xsize + Ysize - 2)); jin = Ysize - 1;
	}
	if (nin>2 * Xsize + Ysize - 3)
	{
		iin = 0; jin = Ysize - 1 - (nin - (2 * Xsize + Ysize - 3));
	}
	i[0] = iin;
	j[0] = jin;
}
__device__ __host__ void ij2dton(long Xsize, long Ysize, int *n, int i, int j)
{
	if (j == 0)
	{
		n[0] = i;
	}
	if (i == Xsize - 1)
	{
		n[0] = i + j;
	}
	if (j == Ysize - 1)
	{
		n[0] = Xsize - 1 + Ysize - 1 + (Xsize - 1 - i);
	}
	if (i == 0 && j != 0)
	{
		n[0] = Xsize - 1 + Ysize - 1 + Xsize - 1 + (Ysize - 1 - j);
	}
}
__device__ __host__ float bodyIntegralFromCenter(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float k1, k2, k3;
	ilast = iin; jlast = jin; klast = kin;
	k1 = iout - jin;
	k2 = jout - jin;
	k3 = kout - kin;
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	bool flag = 0;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, x, y, z;
		r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
		x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
		d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
		r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
		x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
		d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
		r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
		x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
		d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			ilast = inext1;

			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			jlast = jnext2;

			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			klast = knext3;

			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}
__device__ __host__ float bodyIntegral(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	bool flag = 0;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		//why we are not following the real integration path generated????????
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			ilast = inext1;
			//pcountinner[inext1+jnext1*Xsize+knext1*Xsize*Ysize]++;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			jlast = jnext2;
			//pcountinner[inext2+jnext2*Xsize+knext2*Xsize*Ysize]++;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			klast = knext3;
			//pcountinner[inext3+jnext3*Xsize+knext3*Xsize*Ysize]++;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}

__device__ __host__ float bodyIntegralWeighted(long Xsize, long Ysize, long Zsize, int nin, int nout, int n, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float* pcountinner, float* pint, float*pcount, float*pweight)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pinttmp = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		//why we are not following the real integration path generated????????
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pinttmp += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			ilast = inext1;
			curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
			counttmp++;
			//pcountinner[inext1+jnext1*Xsize+knext1*Xsize*Ysize]++;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pinttmp += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			jlast = jnext2;
			curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
			counttmp++;
			//pcountinner[inext2+jnext2*Xsize+knext2*Xsize*Ysize]++;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pinttmp += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			klast = knext3;
			curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
			counttmp++;
			//pcountinner[inext3+jnext3*Xsize+knext3*Xsize*Ysize]++;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	curltmp = curltmp / counttmp;
	if (curltmp != 0)
	{
		pweight[nin + nout*n] += 1 / curltmp;
		pcount[nin + nout*n]++;
		pint[nin + nout*n] += pinttmp;
	}
	return pinttmp;
}
__device__ __host__ float bodyIntegralWeightedMiniCurl(long Xsize, long Ysize, long Zsize, int nin, int nout, int n, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float* pcountinner, float* pint, float*pcount)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pinttmp = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		//why we are not following the real integration path generated????????
		float r, d1, d2, d3, xt, yt, zt;
		/*r=k1*inext1-x*k1+k2*jnext1-k2*y+k3*knext1-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d1=sqrt((xt-inext1)*(xt-inext1)+(yt-jnext1)*(yt-jnext1)+(zt-knext1)*(zt-knext1));
		r=k1*inext2-x*k1+k2*jnext2-k2*y+k3*knext2-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d2=sqrt((xt-inext2)*(xt-inext2)+(yt-jnext2)*(yt-jnext2)+(zt-knext2)*(zt-knext2));
		r=k1*inext3-x*k1+k2*jnext3-k2*y+k3*knext3-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d3=sqrt((xt-inext3)*(xt-inext3)+(yt-jnext3)*(yt-jnext3)+(zt-knext3)*(zt-knext3));*/
		//////End of calculation distance///////////////

		///***calculation of curl in three directions***//////////////////
		d1 = 1e10; d2 = 1e10; d3 = 1e10;
		if (inext1 + jnext1*Xsize + knext1*Xsize*Ysize >= 0 && inext1 + jnext1*Xsize + knext1*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d1 = curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];

		}
		if (inext2 + jnext2*Xsize + knext2*Xsize*Ysize >= 0 && inext2 + jnext2*Xsize + knext2*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d2 = curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
		}
		if (inext3 + jnext3*Xsize + knext3*Xsize*Ysize >= 0 && inext3 + jnext3*Xsize + knext3*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d3 = curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
		}
		//path 1

		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pinttmp += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			ilast = inext1;
			curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
			counttmp++;
			//pcountinner[inext1+jnext1*Xsize+knext1*Xsize*Ysize]++;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pinttmp += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			jlast = jnext2;
			curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
			counttmp++;
			//pcountinner[inext2+jnext2*Xsize+knext2*Xsize*Ysize]++;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pinttmp += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			klast = knext3;
			curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
			counttmp++;
			//pcountinner[inext3+jnext3*Xsize+knext3*Xsize*Ysize]++;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	curltmp = curltmp / counttmp;
	if (curltmp != 0)
	{
		pcount[nin + nout*n] += 1 / curltmp;
		pint[nin + nout*n] += pinttmp;
	}
	return pinttmp;
}

__device__ __host__ float bodyIntegralInner(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float *p, float*pn, float*pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	bool flag = 0;
	pn[ilast + jlast*Xsize + klast*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize];
	pcountinner[ilast + jlast*Xsize + klast*Xsize*Ysize] += 1;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			ilast = inext1;
			pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));
			pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint;
			jlast = jnext2;
			pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			klast = knext3;
			pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}
__device__ __host__ float bodyIntegralInnerStepCount(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float *p, float*pn, float*pcountinner, long*IntegrationStep)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	bool flag = 0;
	int steps = 0;
	pn[ilast + jlast*Xsize + klast*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize];
	pcountinner[ilast + jlast*Xsize + klast*Xsize*Ysize] += 1;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			ilast = inext1;
			pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));
			pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint;
			jlast = jnext2;
			pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			klast = knext3;
			pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}
		else
		{
			steps++;
			IntegrationStep[ilast + jlast*Xsize + klast*Xsize*Ysize] += steps;

		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}

__device__ __host__ float bodyIntegralInnerWeighted(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float *p, float*pn, float*pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	//curltmp=curl[ilast+jlast*Xsize+klast*Xsize*Ysize];
	//counttmp=1;
	//pn[ilast+jlast*Xsize+klast*Xsize*Ysize]+=p[ilast+jlast*Xsize+klast*Xsize*Ysize]*1/curltmp*counttmp;
	//pcountinner[ilast+jlast*Xsize+klast*Xsize*Ysize]+=1/curltmp*counttmp;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1 < Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]/10000;
			counttmp++;
			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			ilast = inext1;
			if (curltmp != 0)
			{
				pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;

			    pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1 / curltmp*counttmp;
		    }
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] / 10000;
			counttmp++;
			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));
			
			if (curltmp != 0)
			{
				pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;

				pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1 / curltmp*counttmp;
			}
			jlast = jnext2;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize]/10000;
			counttmp++;
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			klast = knext3;
			if(curltmp != 0)
			{
				pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;
                pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1 / curltmp*counttmp;
			}
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}
__device__ __host__ float bodyIntegralInnerWeightedFixedBC(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float *p, float*pn, float*pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	//curltmp=curl[ilast+jlast*Xsize+klast*Xsize*Ysize];
	//counttmp=1;
	//pn[ilast+jlast*Xsize+klast*Xsize*Ysize]+=p[ilast+jlast*Xsize+klast*Xsize*Ysize]*1/curltmp*counttmp;
	//pcountinner[ilast+jlast*Xsize+klast*Xsize*Ysize]+=1/curltmp*counttmp;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1 < Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] / 10000;
			counttmp++;
			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			ilast = inext1;
			if (curltmp != 0)
			{
				pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;

				pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1 / curltmp*counttmp;
			}
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 > 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] / 10000;
			counttmp++;
			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));

			if (curltmp != 0)
			{
				pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;

				pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1 / curltmp*counttmp;
			}
			jlast = jnext2;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] / 10000;
			counttmp++;
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			klast = knext3;
			if (curltmp != 0)
			{
				pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;
				pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1 / curltmp*counttmp;
			}
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}

__device__ __host__ float bodyIntegralInnerSelect(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float *p, float*pn, float *curl, float*pcountinner, float threshold)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	bool outthreshold = 0; 
	float p0=0;
	bool flag_p0set = 0;
	if (curl[iin + jin*Xsize + kin*Xsize*Ysize] == 0 || curl[iin + jin*Xsize + kin*Xsize*Ysize]>threshold)
	{
		flag_p0set = 0;
	}
	else
	{
		p0 = p[iin + jin*Xsize + kin*Xsize*Ysize];
		flag_p0set = 1;
		curltmp = 0;
		counttmp = 0;
	}
	
	//curltmp=curl[ilast+jlast*Xsize+klast*Xsize*Ysize];
	//counttmp=1;
	//pn[ilast+jlast*Xsize+klast*Xsize*Ysize]+=p[ilast+jlast*Xsize+klast*Xsize*Ysize]*1/curltmp*counttmp;
	//pcountinner[ilast+jlast*Xsize+klast*Xsize*Ysize]+=1/curltmp*counttmp;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			if (flag_p0set == 0)
			{
				if (curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] != 0 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]<threshold)
				{
					p0 = p[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
					flag_p0set = 1;
					pint = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			else
			{
				if (outthreshold == 0 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] < threshold && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]!=0)
				{
					//passing through low error zone
					pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]/10000;
					counttmp++;
					pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p0 + pint) * 1 / curltmp*counttmp;
					pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1 / curltmp*counttmp;
				}
				if (outthreshold == 0 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] >= threshold || curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]==0)
				{
					///Entering high error zone
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					outthreshold = 1; 
					//pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p0 + pint);
					//pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1;
				}
				if (outthreshold == 1 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] < threshold && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] != 0)
				{
					//reset the starting point of integration. if Exiting the higher error zone.
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					p0 = p[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
					pint = 0;
					outthreshold = 0;
					counttmp = 0;
					curltmp = 0;
					
				}
			}
			
			ilast = inext1;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			if (flag_p0set == 0)
			{
				if (curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] != 0)
				{
					p0 = p[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
					flag_p0set = 1;
					pint = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			else
			{
				if (outthreshold == 0 && curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] < threshold&& curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize]!=0)
				{
					//passing through low error zone
					pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize]/10000;
					counttmp++;
					pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p0 + pint) * 1 / curltmp*counttmp;
					pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1 / curltmp*counttmp;
				}
				if (outthreshold == 0 && curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] >= threshold)
				{
					///Entering high error zone
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					outthreshold = 1;
					//pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p0 + pint);
					//pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1;
				}
				if (outthreshold == 1 && curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] < threshold&& curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] != 0)
				{
					//reset the starting point of integration. if Exiting the higher error zone.
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					p0 = p[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
					pint = 0;
					outthreshold = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
            jlast = jnext2;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			if (flag_p0set == 0)
			{
				if (curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] != 0)
				{
					p0 = p[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
					flag_p0set = 1;
					pint = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			else
			{
				if (outthreshold == 0 && curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] < threshold&&curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] !=0)
				{
					//passing through low error zone
					pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize]/10000;
					counttmp++;
					pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p0 + pint) * 1 / curltmp*counttmp;
					pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1 / curltmp*counttmp;
				}
				if (outthreshold == 0 && curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] >= threshold)
				{
					///Entering high error zone
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					outthreshold = 1; 
					//pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p0 + pint);
					//pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1;
					
				}
				if (outthreshold == 1 && curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] < threshold&&curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize]!= 0)
				{
					//reset the starting point of integration. if Exiting the higher error zone.
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					p0 = p[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
					pint = 0;
					outthreshold = 0;
					

				}
			}

			

			klast = knext3;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
			return 0;
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}
__device__ __host__ float bodyIntegralInnerSelectFixedBC(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float *p, float*pn, float *curl, float*pcountinner, float threshold)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	bool outthreshold = 0;
	float p0 = 0;
	bool flag_p0set = 0;
	if (curl[iin + jin*Xsize + kin*Xsize*Ysize] == 0 || curl[iin + jin*Xsize + kin*Xsize*Ysize]>threshold)
	{
		flag_p0set = 0;
	}
	else
	{
		p0 = p[iin + jin*Xsize + kin*Xsize*Ysize];
		flag_p0set = 1;
		curltmp = 0;
		counttmp = 0;
	}

	//curltmp=curl[ilast+jlast*Xsize+klast*Xsize*Ysize];
	//counttmp=1;
	//pn[ilast+jlast*Xsize+klast*Xsize*Ysize]+=p[ilast+jlast*Xsize+klast*Xsize*Ysize]*1/curltmp*counttmp;
	//pcountinner[ilast+jlast*Xsize+klast*Xsize*Ysize]+=1/curltmp*counttmp;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d1 = sqrt((xt - inext1)*(xt - inext1) + (yt - jnext1)*(yt - jnext1) + (zt - knext1)*(zt - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d2 = sqrt((xt - inext2)*(xt - inext2) + (yt - jnext2)*(yt - jnext2) + (zt - knext2)*(zt - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		xt = x + k1*r; yt = y + k2*r; zt = z + k3*r;
		d3 = sqrt((xt - inext3)*(xt - inext3) + (yt - jnext3)*(yt - jnext3) + (zt - knext3)*(zt - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 < d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize &&jnext1 != 0)
		{
			if (flag_p0set == 0)
			{
				if (curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] != 0 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]<threshold)
				{
					p0 = p[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
					flag_p0set = 1;
					pint = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			else
			{
				if (outthreshold == 0 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] < threshold && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] != 0)
				{
					//passing through low error zone
					pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] / 10000;
					counttmp++;
					pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p0 + pint) * 1 / curltmp*counttmp;
					pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1 / curltmp*counttmp;
				}
				if (outthreshold == 0 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] >= threshold || curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] == 0)
				{
					///Entering high error zone
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					outthreshold = 1;
					//pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p0 + pint);
					//pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1;
				}
				if (outthreshold == 1 && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] < threshold && curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] != 0)
				{
					//reset the starting point of integration. if Exiting the higher error zone.
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					p0 = p[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
					pint = 0;
					outthreshold = 0;
					counttmp = 0;
					curltmp = 0;

				}
			}

			ilast = inext1;
			flag = 1;
		}
		if (d2 <= d1&&d2 <= d3&&jnext2 > 0 && jnext2<Ysize)
		{
			if (flag_p0set == 0)
			{
				if (curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] != 0)
				{
					p0 = p[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
					flag_p0set = 1;
					pint = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			else
			{
				if (outthreshold == 0 && curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] < threshold&& curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] != 0)
				{
					//passing through low error zone
					pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] / 10000;
					counttmp++;
					pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p0 + pint) * 1 / curltmp*counttmp;
					pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1 / curltmp*counttmp;
				}
				if (outthreshold == 0 && curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] >= threshold)
				{
					///Entering high error zone
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					outthreshold = 1;
					//pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p0 + pint);
					//pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1;
				}
				if (outthreshold == 1 && curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] < threshold&& curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] != 0)
				{
					//reset the starting point of integration. if Exiting the higher error zone.
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					p0 = p[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
					pint = 0;
					outthreshold = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			jlast = jnext2;
			flag = 1;
		}
		if (d3 < d1 && d3 < d2 && knext3 >= 0 && knext3<Zsize && jnext3 != 0)
		{
			if (flag_p0set == 0)
			{
				if (curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] != 0)
				{
					p0 = p[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
					flag_p0set = 1;
					pint = 0;
					counttmp = 0;
					curltmp = 0;
				}
			}
			else
			{
				if (outthreshold == 0 && curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] < threshold&&curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] != 0)
				{
					//passing through low error zone
					pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] / 10000;
					counttmp++;
					pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p0 + pint) * 1 / curltmp*counttmp;
					pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1 / curltmp*counttmp;
				}
				if (outthreshold == 0 && curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] >= threshold)
				{
					///Entering high error zone
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					outthreshold = 1;
					//pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p0 + pint);
					//pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1;

				}
				if (outthreshold == 1 && curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] < threshold&&curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] != 0)
				{
					//reset the starting point of integration. if Exiting the higher error zone.
					//pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					p0 = p[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
					pint = 0;
					outthreshold = 0;


				}
			}



			klast = knext3;
			flag = 1;
		}
		/*if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
			return 0;
		}*/

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}

__device__ __host__ float bodyIntegralInnerMiniCurl(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float *p, float*pn, float*pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	bool flag = 0;

	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		/*r=k1*inext1-x*k1+k2*jnext1-k2*y+k3*knext1-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d1=sqrt((xt-inext1)*(xt-inext1)+(yt-jnext1)*(yt-jnext1)+(zt-knext1)*(zt-knext1));
		r=k1*inext2-x*k1+k2*jnext2-k2*y+k3*knext2-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d2=sqrt((xt-inext2)*(xt-inext2)+(yt-jnext2)*(yt-jnext2)+(zt-knext2)*(zt-knext2));
		r=k1*inext3-x*k1+k2*jnext3-k2*y+k3*knext3-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d3=sqrt((xt-inext3)*(xt-inext3)+(yt-jnext3)*(yt-jnext3)+(zt-knext3)*(zt-knext3));*/
		//////End of calculation distance///////////////

		///***calculation of curl in three directions***//////////////////
		d1 = 1e10; d2 = 1e10; d3 = 1e10;
		if (inext1 + jnext1*Xsize + knext1*Xsize*Ysize >= 0 && inext1 + jnext1*Xsize + knext1*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d1 = curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];

		}
		if (inext2 + jnext2*Xsize + knext2*Xsize*Ysize >= 0 && inext2 + jnext2*Xsize + knext2*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d2 = curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
		}
		if (inext3 + jnext3*Xsize + knext3*Xsize*Ysize >= 0 && inext3 + jnext3*Xsize + knext3*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d3 = curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
		}

		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);

			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			ilast = inext1;
			pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]++;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);

			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));
			pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			jlast = jnext2;
			pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize]++;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			klast = knext3;
			pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize]++;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}

__device__ __host__ float bodyIntegralInnerWeightedMiniCurl(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float *p, float*pn, float*pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	float curltmp = 0;
	float counttmp = 0;
	bool flag = 0;
	curltmp = curl[ilast + jlast*Xsize + klast*Xsize*Ysize];
	counttmp = 1;
	pn[ilast + jlast*Xsize + klast*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize] * 1 / curltmp*counttmp;
	pcountinner[ilast + jlast*Xsize + klast*Xsize*Ysize] += 1 / curltmp*counttmp;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3, xt, yt, zt;
		/*r=k1*inext1-x*k1+k2*jnext1-k2*y+k3*knext1-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d1=sqrt((xt-inext1)*(xt-inext1)+(yt-jnext1)*(yt-jnext1)+(zt-knext1)*(zt-knext1));
		r=k1*inext2-x*k1+k2*jnext2-k2*y+k3*knext2-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d2=sqrt((xt-inext2)*(xt-inext2)+(yt-jnext2)*(yt-jnext2)+(zt-knext2)*(zt-knext2));
		r=k1*inext3-x*k1+k2*jnext3-k2*y+k3*knext3-k3*z;
		xt=x+k1*r;yt=y+k2*r;zt=z+k3*r;
		d3=sqrt((xt-inext3)*(xt-inext3)+(yt-jnext3)*(yt-jnext3)+(zt-knext3)*(zt-knext3));*/
		//////End of calculation distance///////////////

		///***calculation of curl in three directions***//////////////////
		d1 = 1e10; d2 = 1e10; d3 = 1e10;
		if (inext1 + jnext1*Xsize + knext1*Xsize*Ysize >= 0 && inext1 + jnext1*Xsize + knext1*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d1 = curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];

		}
		if (inext2 + jnext2*Xsize + knext2*Xsize*Ysize >= 0 && inext2 + jnext2*Xsize + knext2*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d2 = curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
		}
		if (inext3 + jnext3*Xsize + knext3*Xsize*Ysize >= 0 && inext3 + jnext3*Xsize + knext3*Xsize*Ysize<Xsize*Ysize*Zsize)
		{
			d3 = curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
		}

		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			pint += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext1 + jnext1*Xsize + knext1*Xsize*Ysize];
			counttmp++;
			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;
			ilast = inext1;
			pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1 / curltmp*counttmp;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			pint += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext2 + jnext2*Xsize + knext2*Xsize*Ysize];
			counttmp++;
			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));
			pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;
			jlast = jnext2;
			pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1 / curltmp*counttmp;
			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{
			pint += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			curltmp += curl[inext3 + jnext3*Xsize + knext3*Xsize*Ysize];
			counttmp++;
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint) * 1 / curltmp*counttmp;
			klast = knext3;
			pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1 / curltmp*counttmp;
			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}

__device__ __host__ float bodyIntegralInner2(long Xsize, long Ysize, long Zsize, int iin, int jin, int kin, int iout, int jout, int kout, float x, float y, float z, float k1, float k2, float k3, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float *p, float*pn, float*pcountinner)
{
	long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	x = x + center_x;
	y = y + center_y;
	z = z + center_z;
	ilast = iin; jlast = jin; klast = kin;
	if (k1*(iout - iin) + k2*(jout - jin) + k3*(kout - kin)<0)
	{
		k1 = -k1; k2 = -k2; k3 = -k3;
	}
	//k1=iout-iin;k2=jout-jin;k3=kout-kin;
	float pint = 0;
	bool flag = 0;
	do
	{
		if (ilast<iout)
		{
			inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
		}
		if (ilast == iout)
		{
			inext1 = ilast - 60000; jnext1 = jlast; knext1 = klast;
		}
		if (ilast>iout)
		{
			inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
		}
		if (jlast<jout)
		{
			inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
		}
		if (jlast == jout)
		{
			inext2 = ilast; jnext2 = jlast - 60000; knext2 = klast;
		}
		if (jlast>jout)
		{
			inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
		}
		if (klast<kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
		}
		if (klast == kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 60000;
		}
		if (klast>kout)
		{
			inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
		}
		///determine which one is closer to integration path
		float r, d1, d2, d3;
		r = k1*inext1 - x*k1 + k2*jnext1 - k2*y + k3*knext1 - k3*z;
		x = x + k1*r; y = y + k2*r; z = z + k3*r;
		d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
		r = k1*inext2 - x*k1 + k2*jnext2 - k2*y + k3*knext2 - k3*z;
		x = x + k1*r; y = y + k2*r; z = z + k3*r;
		d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
		r = k1*inext3 - x*k1 + k2*jnext3 - k2*y + k3*knext3 - k3*z;
		x = x + k1*r; y = y + k2*r; z = z + k3*r;
		d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
		//////End of calculation distance///////////////
		//path 1
		flag = 0;
		if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize)
		{
			float pinttmp = -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			pint += pinttmp;
			//float w=sqrtf((inext1-iin)*(inext1-iin)+(jnext1-jin)*(jnext1-jin)+(knext1-kin)*(knext1-kin));
			pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint;
			pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1;
			pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize] + pinttmp;
			pcountinner[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += 1;
			ilast = inext1;
			flag = 1;
		}
		if (d2<d1&&d2 <= d3&&jnext2 >= 0 && jnext2<Ysize)
		{
			float pinttmp = -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			pint += pinttmp;
			//float w=sqrtf((inext2-iin)*(inext2-iin)+(jnext2-jin)*(jnext2-jin)+(knext2-kin)*(knext2-kin));
			pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint;
			pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1;
			pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize] + pinttmp;
			pcountinner[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += 1;

			jlast = jnext2;

			flag = 1;
		}
		if (d3<d1&&d3<d2&&knext3 >= 0 && knext3<Zsize)
		{

			float pinttmp = -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
			pint += pinttmp;
			//float w=sqrtf((inext3-iin)*(inext3-iin)+(jnext3-jin)*(jnext3-jin)+(knext3-kin)*(knext3-kin));
			pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint);
			pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1;
			pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += (p[ilast + jlast*Xsize + klast*Xsize*Ysize] + pinttmp);
			pcountinner[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += 1;
			klast = knext3;

			flag = 1;
		}
		if (flag == 0)
		{
			printf("Error! Wrong Point Found!\n");
			if (d3<d1&&d3<d2)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext3, jnext3, knext3, inext1, jnext1, knext1, inext2, jnext2, knext2);

			}
			if (d2<d1&&d2 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext2, jnext2, knext2, inext1, jnext1, knext1, inext3, jnext3, knext3);

			}
			if (d1 <= d2&&d1 <= d3)
			{
				printf("%6.5f %6.5f %6.5f (%02d %02d %02d) (%02d %02d %02d) (%02d %02d %02d)\n(%02d %02d %02d) (%02d %02d %02d)\n", d1, d2, d3, iin, jin, kin, iout, jout, kout, inext1, jnext1, knext1, inext3, jnext3, knext3, inext2, jnext2, knext2);

			}
		}

	} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5&&flag == 1);
	return pint;
}
__global__ void initialIntegration(long Xsize, long Ysize, long Zsize, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* p, float* pn)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	long nout = threadIdx.x + blockIdx.x*blockDim.x;
	while (nout<n)
	{
		int iout, jout, kout;
		ntoijk(Xsize, Ysize, Zsize, nout, &iout, &jout, &kout);
		p[nout] = bodyIntegralFromCenter(Xsize, Ysize, Zsize, Xsize / 2, Ysize / 2, Zsize / 2, iout, jout, kout, deltx, delty, deltz, density, DuDt, DvDt, DwDt);
		nout = nout + blockDim.x*gridDim.x;
	}
}
__global__ void omni3d(long Xsize, long Ysize, long Zsize, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, int*pcountinner)
{
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	long iin, jin, kin, iout, jout, kout, indexin, indexout;
	long nin = blockDim.x*blockIdx.x + threadIdx.x;
	long nout = blockDim.y*blockIdx.y + threadIdx.y;
	while (nin<n&&nout<n)
	{
		long iout, jout, kout;
		long facein, faceout;
		if (nout <= Xsize*Ysize - 1)
		{
			kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			faceout = 1;
		}
		if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
		{
			iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			faceout = 2;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
		{
			kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			faceout = 3;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
		{
			jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
			iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
			iout = Xsize - 2 - iout;
			kout = kout + 1;
			faceout = 4;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
			kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
			kout = Zsize - 2 - kout;
			jout = jout + 1;
			faceout = 5;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
			iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
			kout = Zsize - 2 - kout;
			iout = iout + 1;
			faceout = 6;
		}
		long iin, jin, kin;

		if (nin <= Xsize*Ysize - 1)
		{
			kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
			facein = 1;
		}
		if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
		{
			iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
			facein = 2;
		}
		if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
		{
			kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
			facein = 3;
		}
		if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
		{
			jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
			iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
			iin = Xsize - 2 - iin;
			kin = kin + 1;
			facein = 4;
		}
		if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
			kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
			kin = Zsize - 2 - kin;
			jin = jin + 1;
			facein = 5;
		}
		if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
			iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
			kin = Zsize - 2 - kin;
			iin = iin + 1;
			facein = 6;
		}
		long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
		ilast = iin; jlast = jin; klast = kin;
		if (nin != nout&&nin >= 0 && nin<n&&nout >= 0 && nout<n)
		{
			float k1 = iout - iin;
			float k2 = jout - jin;
			float k3 = kout - kin;
			float l = sqrt(k1*k1 + k2*k2 + k3*k3);
			k1 = k1 / l;
			k2 = k2 / l;
			k3 = k3 / l;
			//cout<<"indexin: "<<nin<<" indexout:"<<nout<<endl;
			//cout<<'('<<iin<<','<<jin<<','<<kin<<")  "<<'('<<iout<<','<<jout<<','<<kout<<")  "<<endl;
			//log<<"indexin: "<<nin<<" indexout:"<<nout<<endl;
			//log<<'('<<iin<<','<<jin<<','<<kin<<")  "<<'('<<iout<<','<<jout<<','<<kout<<")  "<<endl;
			do
			{
				if (ilast<iout)
				{
					inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
				}
				if (ilast == iout)
				{
					inext1 = ilast - 1e6; jnext1 = jlast; knext1 = klast;
				}
				if (ilast>iout)
				{
					inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
				}
				if (jlast<jout)
				{
					inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
				}
				if (jlast == jout)
				{
					inext2 = ilast; jnext2 = jlast - 1e6; knext2 = klast;
				}
				if (jlast>jout)
				{
					inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
				}
				if (klast<kout)
				{
					inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
				}
				if (klast == kout)
				{
					inext3 = ilast; jnext3 = jlast; knext3 = klast - 1e6;
				}
				if (klast>kout)
				{
					inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
				}
				///determine which one is closer to longegration path
				float r, d1, d2, d3, x, y, z;
				r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
				x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
				d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
				r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
				x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
				d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
				r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
				x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
				d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
				//////End of calculation distance///////////////
				//path 1
				if (d1 <= d2&&d1 <= d3)
				{
					pint[nin + nout*n] += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					ilast = inext1;

				}
				if (d2<d1&&d2 <= d3)
				{
					pint[nin + nout*n] += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					jlast = jnext2;
				}
				if (d3<d1&&d3<d2)
				{
					pint[nin + nout*n] += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					klast = knext3;
				}
			} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5);
		}
		nin = nin + blockDim.x*gridDim.x;
		nout = nout + blockDim.y*gridDim.y;
	}
	//////End of calculation of pressure increment////////////////





}
__global__ void omni3dparallellinesEqualSpacing(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount, float* pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}

		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void omni3dparallellinesEqualSpacingWeighted(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float*pweight, float* pcount, float* pcountinner, float*curl)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}

		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegralWeighted(Xsize, Ysize, Zsize, n, nin, nout, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, curl, pcountinner, pint, pcount, pweight);

			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void omni3dparallellinesEqualSpacingWeightedMiniCurl(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount, float* pcountinner, float*curl)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}

		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegralWeightedMiniCurl(Xsize, Ysize, Zsize, n, nin, nout, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, curl, pcountinner, pint, pcount);

			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}

__global__ void omni3dparallellinesEqualSpacingSelect(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount, float* pcountinner, float*curl, float threshold)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}

		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 4 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<4 && kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<4 && kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<4 && kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}
		if (iin >= 0 && iin<Xsize&&jin >= 4 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 4 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}

}
//select iout<iin
__global__ void omni3dparallellinesEqualSpacingSelect2(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount, float* pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}

		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize && (phi<PI / 4 || phi>3 * PI / 4))
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}
		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}

__global__ void omni2dparallellinesOnFace(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount, float* pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.414 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid&&angle<NoAngles)
	{
		float theta = angle / NoAngles * 2 * PI;
		///on XY face
		float k1 = __cosf(theta);
		float k2 = __sinf(theta);
		float k3 = 0;
		float x = __sinf(theta)*(point - NoGrid / 2)*linespacing;
		float y = __cosf(theta)*(point - NoGrid / 2)*linespacing;
		float z = -Zsize / 2.0;
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}
		///on XY face 2
		z = Zsize / 2.0;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				float pincre = bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
				//float w=sqrtf((iout-iin)*(iout-iin)+(jout-jin)*(jout-jin)+(kout-kin)*(kout-kin));
				pint[nin + nout*n] += pincre;
				//pcountinner[iin+jin*Xsize+kin*Xsize*Ysize]++;
				//pcountinner[iout+jout*Xsize+kout*Xsize*Ysize]++;

				pcount[nin + nout*n]++;
				//pint[nout+nin*n]+=-pincre;
				//pcount[nout+nin*n]++;
			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void devidecount(long Xsize, long Ysize, long Zsize, float* pint, float* pcount)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	long tid = threadIdx.x + blockDim.x*blockIdx.x;
	while (tid<n*n)
	{
		if (pcount[tid]>0)
		{
			pint[tid] /= pcount[tid];
		}

		tid += blockDim.x*gridDim.x;
	}
}
__global__ void devidecountWeight(long Xsize, long Ysize, long Zsize, float* pint, float* pcount, float*pweight)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	long tid = threadIdx.x + blockDim.x*blockIdx.x;
	while (tid<n*n)
	{
		if (pcount[tid]>0)
		{
			pint[tid] /= pcount[tid];
			pweight[tid] /= pcount[tid];
		}

		tid += blockDim.x*gridDim.x;
	}
}

__global__ void omni3dvirtual(long Xsize, long Ysize, long Zsize, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount)
{
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	//virtual boundary an ellipsoid
	int a = Xsize - 1;
	int b = Ysize - 1;
	int c = Zsize - 1;
	float delttheta = PI / Zsize / 2;
	float deltbeta = PI / Xsize / 2;
	float xin, yin, zin, xout, yout, zout, k1, k2, k3, x, y, z;
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	indexin = blockDim.x*blockIdx.x + threadIdx.x;
	float thetain = (indexin / (2 * Zsize))*delttheta;
	float betain = (blockDim.x*blockIdx.x + threadIdx.x - 2 * Zsize*(indexin / (2 * Zsize)))*deltbeta;
	indexout = blockDim.y*blockIdx.y + threadIdx.y;
	float thetaout = (indexout / (2 * Xsize))*delttheta;
	float betaout = (blockDim.y*blockIdx.y + threadIdx.y - 2 * Xsize*(indexout / (2 * Xsize)))*deltbeta;
	while (indexin<int(PI / delttheta)*int(PI / deltbeta) * 2 && indexout<int(PI / delttheta)*int(PI / deltbeta) * 2)
	{
		xin = a*sin(thetain)*cos(betain);
		yin = b*sin(thetain)*sin(betain);
		zin = c*cos(thetain);
		xout = a*sin(thetaout)*cos(betaout);
		yout = b*sin(thetaout)*sin(betaout);
		zout = c*cos(thetaout);
		k1 = xout - xin;
		k2 = yout - yin;
		k3 = zout - zin;
		/////case 1, vertical to x-axis
		if (k1 == 0 && k2 != 0 && k3 != 0)
		{
			if (xin >= -center_x&&xin <= center_x)
			{
				////four crossing point;y=0;y=max;z=0;z=max;
				float r = (-center_y - yin) / k2; float y1 = -center_y; float z1 = zin + k3*r;
				r = (center_y - yin) / k2; float y2 = center_y; float z2 = zin + k3*r;
				r = (-center_z - zin) / k3; float z3 = -center_z; float y3 = yin + k2*r;
				r = (center_z - zin) / k3; float z4 = center_z; float y4 = yin + k2*r;
				bool flag = 0;
				if (z1 <= center_z&&z1 >= -center_z&&flag == 0)//cross y=0;
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = 0;
						kin = int(z1 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = 0;
						kout = int(z1 + center_z + 0.5);
					}
					flag = 1;
				}
				if (z2 <= center_z&&z2 >= -center_z)//y=max;
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = Ysize - 1;
						kin = int(z2 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = Ysize - 1;
						kout = int(z2 + center_z + 0.5);
					}
					flag = 1;
				}
				if (y3 <= center_y&&y3 >= -center_y)//z=0;
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = int(y3 + center_y + 0.5);
						kin = 0;
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = int(y3 + center_y + 0.5);
						kout = 0;
					}
					flag = 1;
				}
				if (y4 <= center_y&&y4 >= -center_y)
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = int(y4 + center_y + 0.5);
						kin = Zsize - 1;
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = int(y4 + center_y + 0.5);
						kout = Zsize - 1;
					}
				}
				//sorting intersection point by in, out order
				if (flag != 0)
				{
					if ((jout - jin)*k2 + (kout - kin)*k3<0)
					{
						int temp;
						temp = jin; jin = jout; jout = temp;
						temp = kin; kin = kout; kout = temp;
					}
				}

			}
		}
		///case 2, vertical to y-axis
		if (k1 != 0 && k2 == 0 && k3 != 0)
		{
			if (yin >= -center_y&&yin <= center_y)
			{
				////four crossing point
				float r = (-center_x - xin) / k1; float x1 = -center_x; float z1 = zin + k3*r;//x=0;
				r = (center_x - xin) / k1; float x2 = center_x; float z2 = zin + k3*r;//x=max
				r = (-center_z - zin) / k3; float z3 = -center_z; float x3 = xin + k1*r;//z=0;
				r = (center_z - zin) / k3; float z4 = center_z; float x4 = xin + k1*r;//z=max;
				bool flag = 0;
				if (z1 <= center_z&&z1 >= -center_z)
				{
					if (flag == 0)
					{
						iin = 0;
						jin = int(yin + center_y + 0.5);
						kin = int(z1 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = 0;
						jout = int(yin + center_y + 0.5);
						kout = int(z1 + center_z + 0.5);
					}
					flag = 1;
				}
				if (z2 <= center_z&&z2 >= -center_z)
				{
					if (flag == 0)
					{
						iin = Xsize - 1;
						jin = int(yin + center_y + 0.5);
						kin = int(z2 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = Xsize - 1;
						jout = int(yin + center_y + 0.5);
						kout = int(z2 + center_z + 0.5);
					}
					flag = 1;
				}
				if (x3 <= center_x&&x3 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x3 + center_x + 0.5);
						jin = int(yin + center_y + 0.5);
						kin = 0;
					}
					if (flag == 1)
					{
						iout = int(x3 + center_x + 0.5);
						jout = int(yin + center_y + 0.5);
						kout = 0;
					}
					flag = 1;
				}
				if (x4 <= center_x&&x4 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x4 + center_x + 0.5);
						jin = int(yin + center_y + 0.5);
						kin = Zsize - 1;
					}
					if (flag == 1)
					{
						iout = int(x4 + center_x + 0.5);
						jout = int(yin + center_y + 0.5);
						kout = Zsize - 1;
					}
					flag = 1;
				}
				//sorting intersection point by in, out order
				if (flag != 0)
				{
					if ((iout - iin)*k1 + (kout - kin)*k3<0)
					{
						int temp;
						temp = iin; iin = iout; iout = temp;
						temp = kin; kin = kout; kout = temp;
					}
				}
			}
		}
		///case 3, vertical to z-axis
		if (k1 != 0 && k2 != 0 && k3 == 0)
		{
			if (zin >= -center_z&&zin <= center_z)
			{
				////four crossing point
				float r = (-center_x - xin) / k1; float x1 = -center_x; float y1 = yin + k2*r;//x=0;
				r = (center_x - xin) / k1; float x2 = center_x; float y2 = yin + k2*r;//x=max;
				r = (-center_y - zin) / k2; float y3 = -center_y; float x3 = xin + k1*r;//y=0;
				r = (center_y - zin) / k2; float y4 = center_y; float x4 = xin + k1*r;//y=max;
				bool flag = 0;
				if (y1 <= center_y&&y1 >= -center_y)
				{
					if (flag == 0)
					{
						iin = 0;
						jin = int(y1 + center_y + 0.5);
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = 0;
						jout = int(y1 + center_y + 0.5);
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				if (y2 <= center_y&&y2 >= -center_y)
				{
					if (flag == 0)
					{
						iin = Xsize - 1;
						jin = int(y2 + center_y + 0.5);
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = Xsize - 1;
						jout = int(y2 + center_y + 0.5);
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				if (x3 <= center_x&&x3 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x3 + center_x + 0.5);
						jin = 0;
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(x3 + center_x + 0.5);
						jout = 0;
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				if (x4 <= center_x&&x4 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x4 + center_x + 0.5);
						jin = Ysize - 1;
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(x4 + center_x + 0.5);
						jout = Ysize - 1;
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				//sorting intersection point by in, out order
				if (flag != 0)
				{
					if ((iout - iin)*k1 + (jout - jin)*k2<0)
					{
						int temp;
						temp = iin; iin = iout; iout = temp;
						temp = jin; jin = jout; jout = temp;
					}
				}

			}
		}
		///case 4, vertical to plane IJ
		if (abs(k1)<zero&&abs(k2)<zero&&abs(k3) >= zero)
		{

			if (xin <= center_x&&xin >= -center_x&&yin <= center_y&&yin >= -center_y)
			{
				iin = int(xin + center_x + 0.5); iout = iin;
				jin = int(yin + center_y + 0.5); jout = jin;
				if (k3>0)
				{
					kin = 0; kout = Zsize - 1;
				}
				else{
					kin = Zsize - 1; kout = 0;
				}

			}
		}
		///case 5, vertical to IK plane
		if (abs(k1)<zero&&abs(k2) >= zero&&abs(k3)<zero)
		{
			if (xin >= -center_x&&xin <= center_x&&zin >= -center_z&&zin <= center_z)
			{
				iin = int(xin + center_x + 0.5); iout = iin;
				kin = int(zin + center_z + 0.5); kout = kin;
				if (k2>0)
				{
					jout = Ysize - 1; jin = 0;
				}
				else
				{
					jin = Ysize - 1; jout = 0;
				}

			}
		}
		///case 6, vertical to JK plane
		if (abs(k1) >= zero&&abs(k2)<zero&&abs(k3)<zero)
		{
			if (yin >= -center_y&&yin<center_y&&zin >= -center_z&&zin <= center_z)
			{
				jin = int(yin + center_y + 0.5); jout = jin;
				kin = int(zin + center_z + 0.5); kout = kin;
				if (k1>0)
				{
					iout = Xsize - 1; iin = 0;
				}
				else
				{
					iin = Xsize - 1; iout = 0;
				}
			}

		}
		/// case 7, purely inclined
		if (abs(k1) >= zero&&abs(k2) >= zero&&abs(k3) >= zero)
		{
			/// six crossing point
			float r;
			float x1, x2, x3, x4, x5, x6;
			float y1, y2, y3, y4, y5, y6;
			float z1, z2, z3, z4, z5, z6;
			r = (-center_x - xin) / k1; x1 = -center_x; y1 = yin + k2*r; z1 = zin + k3*r;//x=0
			r = (center_x - xin) / k1; x2 = center_x; y2 = yin + k2*r; z2 = zin + k3*r;//x=max
			r = (-center_y - yin) / k2; x3 = xin + k1*r; y3 = -center_y; z3 = zin + k3*r;//y=0;
			r = (center_y - yin) / k2; x4 = xin + k1*r; y4 = center_y; z4 = zin + k3*r;//y=max
			r = (-center_z - zin) / k3; x5 = xin + k1*r; y5 = yin + k2*r; z5 = -center_z;//z=0;
			r = (center_z - zin) / k3; x6 = xin + k1*r; y6 = yin + k2*r; z6 = center_z;//z=max
			bool flag = 0;
			if (y1 <= center_y&&y1 >= -center_y&&z1 <= center_z&&z1 >= -center_z)
			{
				if (flag == 0)
				{
					iin = 0;
					jin = int(y1 + center_y + 0.5);
					kin = int(z1 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = 0;
					jout = int(y1 + center_y + 0.5);
					kout = int(z1 + center_z + 0.5);
				}
				flag = 1;
			}
			if (y2 <= center_y&&y2 >= -center_y&&z2 <= center_z&&z2 >= -center_z)
			{
				if (flag == 0)
				{
					iin = Xsize - 1;
					jin = int(y2 + center_y + 0.5);
					kin = int(z2 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = Xsize - 1;
					jout = int(y2 + center_y + 0.5);
					kout = int(z2 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x3 <= center_x&&x3 >= -center_x&&z3 <= center_z&&z3 >= -center_z)
			{
				if (flag == 0)
				{
					iin = int(x3 + center_x + 0.5);
					jin = 0;
					kin = int(z3 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = int(x3 + center_x + 0.5);
					jout = 0;
					kout = int(z3 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x4 <= center_x&&x4 >= -center_x&&z4 <= center_z&&z4 >= -center_z)
			{
				if (flag == 0)
				{
					iin = int(x4 + center_x + 0.5);
					jin = Ysize - 1;
					kin = int(z4 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = int(x4 + center_x + 0.5);
					jout = Ysize - 1;
					kout = int(z4 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x5 <= center_x&&x5 >= -center_x&&y5 <= center_y&&y5 >= -center_y)
			{
				if (flag == 0)
				{
					iin = int(x5 + center_x + 0.5);
					jin = int(y5 + center_y + 0.5);
					kin = 0;
				}
				if (flag == 1)
				{
					iout = int(x5 + center_x + 0.5);
					jout = int(y5 + center_y + 0.5);
					kout = 0;
				}
				flag = 1;
			}
			if (x6 <= center_x&&x6 >= -center_x&&y6 <= center_y&&y6 >= -center_y)
			{
				if (flag == 0)
				{
					iin = int(x6 + center_x + 0.5);
					jin = int(y6 + center_y + 0.5);
					kin = Zsize - 1;
				}
				if (flag == 1)
				{
					iout = int(x6 + center_x + 0.5);
					jout = int(y6 + center_y + 0.5);
					kout = Zsize - 1;
				}
				flag = 1;
			}
			//sorting intersection point by in, out order
			if ((iout - iin)*k1 + (jout - jin)*k2 + (kout - kin)*k3<0)
			{
				int temp;
				temp = iin; iin = temp; iout = temp;
				temp = jin; jin = jout; jout = temp;
				temp = kin; kin = kout; kout = temp;
			}
		}
		//////////////////////////////END OF CALCULATING IN AND OUT POINT ON REAL BOUNDARY////////////////////////////////
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize && (iin - center_x - xin)*(iin - center_x - xout) + (jin - center_y - yin)*(jin - center_y - yout) + (kin - center_z - zin)*(kin - center_z - zout)<0 && (iin + jin + kin + iout + jout + kout) != 0 && !(iin == iout&&jin == jout&&kin == kout))
		{
			int nin, nout;
			long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
			ilast = iin; jlast = jin; klast = kin;
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout&&nin<n&&nout<n)
			{
				do
				{
					if (ilast<iout)
					{
						inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
					}
					if (ilast == iout)
					{
						inext1 = ilast - 1e6; jnext1 = jlast; knext1 = klast;
					}
					if (ilast>iout)
					{
						inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
					}
					if (jlast<jout)
					{
						inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
					}
					if (jlast == jout)
					{
						inext2 = ilast; jnext2 = jlast - 1e6; knext2 = klast;
					}
					if (jlast>jout)
					{
						inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
					}
					if (klast<kout)
					{
						inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
					}
					if (klast == kout)
					{
						inext3 = ilast; jnext3 = jlast; knext3 = klast - 1e6;
					}
					if (klast>kout)
					{
						inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
					}
					///determine which one is closer to longegration path
					float r, d1, d2, d3;
					r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
					x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
					d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
					r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
					x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
					d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
					r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
					x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
					d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
					//////End of calculation distance///////////////


					if (d1 <= d2&&d1 <= d3&&inext1 >= 0 && inext1<Xsize&&jnext1 >= 0 && jnext1<Ysize&&knext1 >= 0 && knext1<Zsize)
					{
						pint[nin + nout*n] += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
						ilast = inext1;

					}
					if (d2<d1&&d2 <= d3&&inext2<Xsize&&jnext2 >= 0 && jnext2<Ysize&&knext2 >= 0 && knext2<Zsize)
					{
						pint[nin + nout*n] += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
						jlast = jnext2;
					}
					if (d3<d1&&d3<d2&&inext3<Xsize&&jnext3 >= 0 && jnext3<Ysize&&knext3 >= 0 && knext3<Zsize)
					{
						pint[nin + nout*n] += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
						klast = knext3;
					}
				} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5);
				pcount[nin + nout*n]++;
			}
		}
		indexin = indexin + blockDim.x*gridDim.x;
		indexout = indexout + blockDim.y*gridDim.y;
	}
}
__global__ void omni3virtualgrid(long Xsize, long Ysize, long Zsize, int NoTheta, int NoBeta, long* index, long* ninvir, long *noutvir, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pintvir)
{
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	//virtual boundary an ellipsoid
	int a = Xsize - 1;
	int b = Ysize - 1;
	int c = Zsize - 1;
	float delttheta = PI / NoTheta;
	float deltbeta = 2 * PI / NoBeta;
	float xin, yin, zin, xout, yout, zout, k1, k2, k3, x, y, z;
	//int n=Xsize*Ysize*2+(Zsize-2)*Ysize*2+(Xsize-2)*(Zsize-2)*2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	indexin = blockDim.x*blockIdx.x + threadIdx.x;
	float thetain = (indexin / (NoBeta))*delttheta;
	float betain = (blockDim.x*blockIdx.x + threadIdx.x - NoBeta*(indexin / (NoBeta)))*deltbeta;
	indexout = blockDim.y*blockIdx.y + threadIdx.y;
	float thetaout = (indexout / (NoBeta))*delttheta;
	float betaout = (blockDim.y*blockIdx.y + threadIdx.y - NoBeta*(indexout / (NoBeta)))*deltbeta;
	while (indexin<NoTheta*NoBeta&&indexout<NoTheta*NoBeta)
	{
		xin = a*sin(thetain)*cos(betain);
		yin = b*sin(thetain)*sin(betain);
		zin = c*cos(thetain);
		xout = a*sin(thetaout)*cos(betaout);
		yout = b*sin(thetaout)*sin(betaout);
		zout = c*cos(thetaout);
		k1 = xout - xin;
		k2 = yout - yin;
		k3 = zout - zin;
		/////case 1, vertical to x-axis
		if (k1 == 0 && k2 != 0 && k3 != 0)
		{
			if (xin >= -center_x&&xin <= center_x)
			{
				////four crossing point;y=0;y=max;z=0;z=max;
				float r = (-center_y - yin) / k2; float y1 = -center_y; float z1 = zin + k3*r;
				r = (center_y - yin) / k2; float y2 = center_y; float z2 = zin + k3*r;
				r = (-center_z - zin) / k3; float z3 = -center_z; float y3 = yin + k2*r;
				r = (center_z - zin) / k3; float z4 = center_z; float y4 = yin + k2*r;
				bool flag = 0;
				if (z1 <= center_z&&z1 >= -center_z)//cross y=0;
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = 0;
						kin = int(z1 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = 0;
						kout = int(z1 + center_z + 0.5);
					}
					flag = 1;
				}
				if (z2 <= center_z&&z2 >= -center_z)//y=max;
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = Ysize - 1;
						kin = int(z2 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = Ysize - 1;
						kout = int(z2 + center_z + 0.5);
					}
					flag = 1;
				}
				if (y3 <= center_y&&y3 >= -center_y)//z=0;
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = int(y3 + center_y + 0.5);
						kin = 0;
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = int(y3 + center_y + 0.5);
						kout = 0;
					}
					flag = 1;
				}
				if (y4 <= center_y&&y4 >= -center_y)
				{
					if (flag == 0)
					{
						iin = int(xin + center_x + 0.5);
						jin = int(y4 + center_y + 0.5);
						kin = Zsize - 1;
					}
					if (flag == 1)
					{
						iout = int(xin + center_x + 0.5);
						jout = int(y4 + center_y + 0.5);
						kout = Zsize - 1;
					}
				}
				//sorting intersection point by in, out order
				if (flag != 0)
				{
					if ((jout - jin)*k2 + (kout - kin)*k3<0)
					{
						int temp;
						temp = jin; jin = jout; jout = temp;
						temp = kin; kin = kout; kout = temp;
					}
				}

			}
		}
		///case 2, vertical to y-axis
		if (k1 != 0 && k2 == 0 && k3 != 0)
		{
			if (yin >= -center_y&&yin <= center_y)
			{
				////four crossing point
				float r = (-center_x - xin) / k1; float x1 = -center_x; float z1 = zin + k3*r;//x=0;
				r = (center_x - xin) / k1; float x2 = center_x; float z2 = zin + k3*r;//x=max
				r = (-center_z - zin) / k3; float z3 = -center_z; float x3 = xin + k1*r;//z=0;
				r = (center_z - zin) / k3; float z4 = center_z; float x4 = xin + k1*r;//z=max;
				bool flag = 0;
				if (z1 <= center_z&&z1 >= -center_z)
				{
					if (flag == 0)
					{
						iin = 0;
						jin = int(yin + center_y + 0.5);
						kin = int(z1 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = 0;
						jout = int(yin + center_y + 0.5);
						kout = int(z1 + center_z + 0.5);
					}
					flag = 1;
				}
				if (z2 <= center_z&&z2 >= -center_z)
				{
					if (flag == 0)
					{
						iin = Xsize - 1;
						jin = int(yin + center_y + 0.5);
						kin = int(z2 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = Xsize - 1;
						jout = int(yin + center_y + 0.5);
						kout = int(z2 + center_z + 0.5);
					}
					flag = 1;
				}
				if (x3 <= center_x&&x3 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x3 + center_x + 0.5);
						jin = int(yin + center_y + 0.5);
						kin = 0;
					}
					if (flag == 1)
					{
						iout = int(x3 + center_x + 0.5);
						jout = int(yin + center_y + 0.5);
						kout = 0;
					}
					flag = 1;
				}
				if (x4 <= center_x&&x4 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x4 + center_x + 0.5);
						jin = int(yin + center_y + 0.5);
						kin = Zsize - 1;
					}
					if (flag == 1)
					{
						iout = int(x4 + center_x + 0.5);
						jout = int(yin + center_y + 0.5);
						kout = Zsize - 1;
					}
					flag = 1;
				}
				//sorting intersection point by in, out order
				if (flag != 0)
				{
					if ((iout - iin)*k1 + (kout - kin)*k3<0)
					{
						int temp;
						temp = iin; iin = iout; iout = temp;
						temp = kin; kin = kout; kout = temp;
					}
				}
			}
		}
		///case 3, vertical to z-axis
		if (k1 != 0 && k2 != 0 && k3 == 0)
		{
			if (zin >= -center_z&&zin <= center_z)
			{
				////four crossing point
				float r = (-center_x - xin) / k1; float x1 = -center_x; float y1 = yin + k2*r;//x=0;
				r = (center_x - xin) / k1; float x2 = center_x; float y2 = yin + k2*r;//x=max;
				r = (-center_y - zin) / k2; float y3 = -center_y; float x3 = xin + k1*r;//y=0;
				r = (center_y - zin) / k2; float y4 = center_y; float x4 = xin + k1*r;//y=max;
				bool flag = 0;
				if (y1 <= center_y&&y1 >= -center_y)
				{
					if (flag == 0)
					{
						iin = 0;
						jin = int(y1 + center_y + 0.5);
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = 0;
						jout = int(y1 + center_y + 0.5);
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				if (y2 <= center_y&&y2 >= -center_y)
				{
					if (flag == 0)
					{
						iin = Xsize - 1;
						jin = int(y2 + center_y + 0.5);
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = Xsize - 1;
						jout = int(y2 + center_y + 0.5);
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				if (x3 <= center_x&&x3 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x3 + center_x + 0.5);
						jin = 0;
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(x3 + center_x + 0.5);
						jout = 0;
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				if (x4 <= center_x&&x4 >= -center_x)
				{
					if (flag == 0)
					{
						iin = int(x4 + center_x + 0.5);
						jin = Ysize - 1;
						kin = int(zin + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(x4 + center_x + 0.5);
						jout = Ysize - 1;
						kout = int(zin + center_z + 0.5);
					}
					flag = 1;
				}
				//sorting intersection point by in, out order
				if (flag != 0)
				{
					if ((iout - iin)*k1 + (jout - jin)*k2<0)
					{
						int temp;
						temp = iin; iin = iout; iout = temp;
						temp = jin; jin = jout; jout = temp;
					}
				}

			}
		}
		///case 4, vertical to plane IJ
		if (abs(k1)<zero&&abs(k2)<zero&&abs(k3) >= zero)
		{

			if (xin <= center_x&&xin >= -center_x&&yin <= center_y&&yin >= -center_y)
			{
				iin = int(xin + center_x + 0.5); iout = iin;
				jin = int(yin + center_y + 0.5); jout = jin;
				if (k3>0)
				{
					kin = 0; kout = Zsize - 1;
				}
				else{
					kin = Zsize - 1; kout = 0;
				}

			}
		}
		///case 5, vertical to IK plane
		if (abs(k1)<zero&&abs(k2) >= zero&&abs(k3)<zero)
		{
			if (xin >= -center_x&&xin <= center_x&&zin >= -center_z&&zin <= center_z)
			{
				iin = int(xin + center_x + 0.5); iout = iin;
				kin = int(zin + center_z + 0.5); kout = kin;
				if (k2>0)
				{
					jout = Ysize - 1; jin = 0;
				}
				else
				{
					jin = Ysize - 1; jout = 0;
				}

			}
		}
		///case 6, vertical to JK plane
		if (abs(k1) >= zero&&abs(k2)<zero&&abs(k3)<zero)
		{
			if (yin >= -center_y&&yin<center_y&&zin >= -center_z&&zin <= center_z)
			{
				jin = int(yin + center_y + 0.5); jout = jin;
				kin = int(zin + center_z + 0.5); kout = kin;
			}
			if (k1>0)
			{
				iout = Xsize - 1; iin = 0;
			}
			else
			{
				iin = Xsize - 1; iout = 0;
			}
		}
		/// case 7, purely inclined
		if (abs(k1) >= zero&&abs(k2) >= zero&&abs(k3) >= zero)
		{
			/// six crossing point
			float r;
			float x1, x2, x3, x4, x5, x6;
			float y1, y2, y3, y4, y5, y6;
			float z1, z2, z3, z4, z5, z6;
			r = (-center_x - xin) / k1; x1 = -center_x; y1 = yin + k2*r; z1 = zin + k3*r;//x=0
			r = (center_x - xin) / k1; x2 = center_x; y2 = yin + k2*r; z2 = zin + k3*r;//x=max
			r = (-center_y - yin) / k2; x3 = xin + k1*r; y3 = -center_y; z3 = zin + k3*r;//y=0;
			r = (center_y - yin) / k2; x4 = xin + k1*r; y4 = center_y; z4 = zin + k3*r;//y=max
			r = (-center_z - zin) / k3; x5 = xin + k1*r; y5 = yin + k2*r; z5 = -center_z;//z=0;
			r = (center_z - zin) / k3; x6 = xin + k1*r; y6 = yin + k2*r; z6 = center_z;//z=max
			bool flag = 0;
			if (y1 <= center_y&&y1 >= -center_y&&z1 <= center_z&&z1 >= -center_z)
			{
				if (flag == 0)
				{
					iin = 0;
					jin = int(y1 + center_y + 0.5);
					kin = int(z1 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = 0;
					jout = int(y1 + center_y + 0.5);
					kout = int(z1 + center_z + 0.5);
				}
				flag = 1;
			}
			if (y2 <= center_y&&y2 >= -center_y&&z2 <= center_z&&z2 >= -center_z)
			{
				if (flag == 0)
				{
					iin = Xsize - 1;
					jin = int(y2 + center_y + 0.5);
					kin = int(z2 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = Xsize - 1;
					jout = int(y2 + center_y + 0.5);
					kout = int(z2 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x3 <= center_x&&x3 >= -center_x&&z3 <= center_z&&z3 >= -center_z)
			{
				if (flag == 0)
				{
					iin = int(x3 + center_x + 0.5);
					jin = 0;
					kin = int(z3 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = int(x3 + center_x + 0.5);
					jout = 0;
					kout = int(z3 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x4 <= center_x&&x4 >= -center_x&&z4 <= center_z&&z4 >= -center_z)
			{
				if (flag == 0)
				{
					iin = int(x4 + center_x + 0.5);
					jin = Ysize - 1;
					kin = int(z4 + center_z + 0.5);
				}
				if (flag == 1)
				{
					iout = int(x4 + center_x + 0.5);
					jout = Ysize - 1;
					kout = int(z4 + center_z + 0.5);
				}
				flag = 1;
			}
			if (x5 <= center_x&&x5 >= -center_x&&y5 <= center_y&&y5 >= -center_y)
			{
				if (flag == 0)
				{
					iin = int(x5 + center_x + 0.5);
					jin = int(y5 + center_y + 0.5);
					kin = 0;
				}
				if (flag == 1)
				{
					iout = int(x5 + center_x + 0.5);
					jout = int(y5 + center_y + 0.5);
					kout = 0;
				}
				flag = 1;
			}
			if (x6 <= center_x&&x6 >= -center_x&&y6 <= center_y&&y6 >= -center_y)
			{
				if (flag == 0)
				{
					iin = int(x6 + center_x + 0.5);
					jin = int(y6 + center_y + 0.5);
					kin = Zsize - 1;
				}
				if (flag == 1)
				{
					iout = int(x6 + center_x + 0.5);
					jout = int(y6 + center_y + 0.5);
					kout = Zsize - 1;
				}
				flag = 1;
			}
			//sorting intersection point by in, out order
			if ((iout - iin)*k1 + (jout - jin)*k2 + (kout - kin)*k3<0)
			{
				int temp;
				temp = iin; iin = temp; iout = temp;
				temp = jin; jin = jout; jout = temp;
				temp = kin; kin = kout; kout = temp;
			}
		}
		//////////////////////////////END OF CALCULATING IN AND OUT POINT ON REAL BOUNDARY////////////////////////////////
		if ((iin - center_x - xin)*(iin - center_x - xout) + (jin - center_y - yin)*(jin - center_y - yout) + (kin - center_z - zin)*(kin - center_z - zout)<0 && (iin + jin + kin + iout + jout + kout) != 0 && !(iin == iout&&jin == jout&&kin == kout))
		{
			long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
			ilast = iin; jlast = jin; klast = kin;
			do
			{
				if (ilast<iout)
				{
					inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
				}
				if (ilast == iout)
				{
					inext1 = ilast - 1e6; jnext1 = jlast; knext1 = klast;
				}
				if (ilast>iout)
				{
					inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
				}
				if (jlast<jout)
				{
					inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
				}
				if (jlast == jout)
				{
					inext2 = ilast; jnext2 = jlast - 1e6; knext2 = klast;
				}
				if (jlast>jout)
				{
					inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
				}
				if (klast<kout)
				{
					inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
				}
				if (klast == kout)
				{
					inext3 = ilast; jnext3 = jlast; knext3 = klast - 1e6;
				}
				if (klast>kout)
				{
					inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
				}
				///determine which one is closer to longegration path
				float r, d1, d2, d3;
				r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
				x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
				d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
				r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
				x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
				d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
				r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
				x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
				d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
				//////End of calculation distance///////////////

				ninvir[indexin + indexout*NoTheta*NoBeta] = index[iin + jin*Xsize + kin*Xsize*Ysize];
				noutvir[indexin + indexout*NoTheta*NoBeta] = index[iout + jout*Xsize + kout*Xsize*Ysize];
				if (d1 <= d2&&d1 <= d3)
				{
					pintvir[indexin + indexout*NoTheta*NoBeta] += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					ilast = inext1;

				}
				if (d2<d1&&d2 <= d3)
				{
					pintvir[indexin + indexout*NoTheta*NoBeta] += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					jlast = jnext2;
				}
				if (d3<d1&&d3<d2)
				{
					pintvir[indexin + indexout*NoTheta*NoBeta] += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
					klast = knext3;
				}
			} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5);
		}

		indexin = indexin + blockDim.x*gridDim.x;
		indexout = indexout + blockDim.y*gridDim.y;
	}
}
__global__ void omni3dvirtual2(long Xsize, long Ysize, long Zsize, long* index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint)
{
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	//virtual boundary an ellipsoid
	int a = Xsize - 1;
	int b = Ysize - 1;
	int c = Zsize - 1;
	float delttheta = PI / Zsize / 2;
	float deltbeta = PI / Xsize / 2;
	float xin, yin, zin, xout, yout, zout, k1, k2, k3, x, y, z;
	float r, d1, d2, d3;
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin;
	indexin = blockDim.x*blockIdx.x + threadIdx.x;
	float thetain = (indexin / (2 * Zsize) - 1)*delttheta;
	float betain = (blockDim.x*blockIdx.x + threadIdx.x - 2 * Zsize*(indexin / (2 * Zsize)) - 1)*deltbeta;
	for (float thetaout = 0; thetaout<PI; thetaout += delttheta)
	{
		for (float betaout = 0; betaout<2 * PI; betaout += deltbeta)
		{
			xin = a*sin(thetain)*cos(betain);
			yin = b*sin(thetain)*sin(betain);
			zin = c*cos(thetain);
			xout = a*sin(thetaout)*cos(betaout);
			yout = b*sin(thetaout)*sin(betaout);
			zout = c*cos(thetaout);
			k1 = xout - xin;
			k2 = yout - yin;
			k3 = zout - zin;
			/////case 1, vertical to x-axis
			if (k1 == 0 && k2 != 0 && k3 != 0)
			{
				if (xin >= -center_x&&xin <= center_x)
				{
					////four crossing point;y=0;y=max;z=0;z=max;
					r = (-center_y - yin) / k2; float y1 = -center_y; float z1 = zin + k3*r;
					r = (center_y - yin) / k2; float y2 = center_y; float z2 = zin + k3*r;
					r = (-center_z - zin) / k3; float z3 = -center_z; float y3 = yin + k2*r;
					r = (center_z - zin) / k3; float z4 = center_z; float y4 = yin + k2*r;
					bool flag = 0;
					if (z1 <= center_z&&z1 >= -center_z)//cross y=0;
					{
						if (flag == 0)
						{
							iin = int(xin + center_x + 0.5);
							jin = 0;
							kin = int(z1 + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = int(xin + center_x + 0.5);
							jout = 0;
							kout = int(z1 + center_z + 0.5);
						}
						flag = 1;
					}
					if (z2 <= center_z&&z2 >= -center_z)//y=max;
					{
						if (flag == 0)
						{
							iin = int(xin + center_x + 0.5);
							jin = Ysize - 1;
							kin = int(z2 + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = int(xin + center_x + 0.5);
							jout = Ysize - 1;
							kout = int(z2 + center_z + 0.5);
						}
						flag = 1;
					}
					if (y3 <= center_y&&y3 >= -center_y)//z=0;
					{
						if (flag == 0)
						{
							iin = int(xin + center_x + 0.5);
							jin = int(y3 + center_y + 0.5);
							kin = 0;
						}
						if (flag == 1)
						{
							iout = int(xin + center_x + 0.5);
							jout = int(y3 + center_y + 0.5);
							kout = 0;
						}
						flag = 1;
					}
					if (y4 <= center_y&&y4 >= -center_y)
					{
						if (flag == 0)
						{
							iin = int(xin + center_x + 0.5);
							jin = int(y4 + center_y + 0.5);
							kin = Zsize - 1;
						}
						if (flag == 1)
						{
							iout = int(xin + center_x + 0.5);
							jout = int(y4 + center_y + 0.5);
							kout = Zsize - 1;
						}
					}
					//sorting intersection point by in, out order
					if (flag != 0)
					{
						if ((jout - jin)*k2 + (kout - kin)*k3<0)
						{
							int temp;
							temp = jin; jin = jout; jout = temp;
							temp = kin; kin = kout; kout = temp;
						}
					}

				}
			}
			///case 2, vertical to y-axis
			if (k1 != 0 && k2 == 0 && k3 != 0)
			{
				if (yin >= -center_y&&yin <= center_y)
				{
					////four crossing point
					r = (-center_x - xin) / k1; float x1 = -center_x; float z1 = zin + k3*r;//x=0;
					r = (center_x - xin) / k1; float x2 = center_x; float z2 = zin + k3*r;//x=max
					r = (-center_z - zin) / k3; float z3 = -center_z; float x3 = xin + k1*r;//z=0;
					r = (center_z - zin) / k3; float z4 = center_z; float x4 = xin + k1*r;//z=max;
					bool flag = 0;
					if (z1 <= center_z&&z1 >= -center_z)
					{
						if (flag == 0)
						{
							iin = 0;
							jin = int(yin + center_y + 0.5);
							kin = int(z1 + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = 0;
							jout = int(yin + center_y + 0.5);
							kout = int(z1 + center_z + 0.5);
						}
						flag = 1;
					}
					if (z2 <= center_z&&z2 >= -center_z)
					{
						if (flag == 0)
						{
							iin = Xsize - 1;
							jin = int(yin + center_y + 0.5);
							kin = int(z2 + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = Xsize - 1;
							jout = int(yin + center_y + 0.5);
							kout = int(z2 + center_z + 0.5);
						}
						flag = 1;
					}
					if (x3 <= center_x&&x3 >= -center_x)
					{
						if (flag == 0)
						{
							iin = int(x3 + center_x + 0.5);
							jin = int(yin + center_y + 0.5);
							kin = 0;
						}
						if (flag == 1)
						{
							iout = int(x3 + center_x + 0.5);
							jout = int(yin + center_y + 0.5);
							kout = 0;
						}
						flag = 1;
					}
					if (x4 <= center_x&&x4 >= -center_x)
					{
						if (flag == 0)
						{
							iin = int(x4 + center_x + 0.5);
							jin = int(yin + center_y + 0.5);
							kin = Zsize - 1;
						}
						if (flag == 1)
						{
							iout = int(x4 + center_x + 0.5);
							jout = int(yin + center_y + 0.5);
							kout = Zsize - 1;
						}
						flag = 1;
					}
					//sorting intersection point by in, out order
					if (flag != 0)
					{
						if ((iout - iin)*k1 + (kout - kin)*k3<0)
						{
							int temp;
							temp = iin; iin = iout; iout = temp;
							temp = kin; kin = kout; kout = temp;
						}
					}

				}
			}
			///case 3, vertical to z-axis
			if (k1 != 0 && k2 != 0 && k3 == 0)
			{
				if (zin >= -center_z&&zin <= center_z)
				{
					////four crossing point
					r = (-center_x - xin) / k1; float x1 = -center_x; float y1 = yin + k2*r;//x=0;
					r = (center_x - xin) / k1; float x2 = center_x; float y2 = yin + k2*r;//x=max;
					r = (-center_y - zin) / k2; float y3 = -center_y; float x3 = xin + k1*r;//y=0;
					r = (center_y - zin) / k2; float y4 = center_y; float x4 = xin + k1*r;//y=max;
					bool flag = 0;
					if (y1 <= center_y&&y1 >= -center_y)
					{
						if (flag == 0)
						{
							iin = 0;
							jin = int(y1 + center_y + 0.5);
							kin = int(zin + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = 0;
							jout = int(y1 + center_y + 0.5);
							kout = int(zin + center_z + 0.5);
						}
						flag = 1;
					}
					if (y2 <= center_y&&y2 >= -center_y)
					{
						if (flag == 0)
						{
							iin = Xsize - 1;
							jin = int(y2 + center_y + 0.5);
							kin = int(zin + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = Xsize - 1;
							jout = int(y2 + center_y + 0.5);
							kout = int(zin + center_z + 0.5);
						}
						flag = 1;
					}
					if (x3 <= center_x&&x3 >= -center_x)
					{
						if (flag == 0)
						{
							iin = int(x3 + center_x + 0.5);
							jin = 0;
							kin = int(zin + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = int(x3 + center_x + 0.5);
							jout = 0;
							kout = int(zin + center_z + 0.5);
						}
						flag = 1;
					}
					if (x4 <= center_x&&x4 >= -center_x)
					{
						if (flag == 0)
						{
							iin = int(x4 + center_x + 0.5);
							jin = Ysize - 1;
							kin = int(zin + center_z + 0.5);
						}
						if (flag == 1)
						{
							iout = int(x4 + center_x + 0.5);
							jout = Ysize - 1;
							kout = int(zin + center_z + 0.5);
						}
						flag = 1;
					}
					//sorting intersection point by in, out order
					if (flag != 0)
					{
						if ((iout - iin)*k1 + (jout - jin)*k2<0)
						{
							int temp;
							temp = iin; iin = iout; iout = temp;
							temp = jin; jin = jout; jout = temp;
						}
					}

				}
			}
			///case 4, vertical to plane IJ
			if (abs(k1)<zero&&abs(k2)<zero&&abs(k3) >= zero)
			{

				if (xin <= center_x&&xin >= -center_x&&yin <= center_y&&yin >= -center_y)
				{
					iin = int(xin + center_x + 0.5); iout = iin;
					jin = int(yin + center_y + 0.5); jout = jin;
					if (k3>0)
					{
						kin = 0; kout = Zsize - 1;
					}
					else{
						kin = Zsize - 1; kout = 0;
					}

				}
			}
			///case 5, vertical to IK plane
			if (abs(k1)<zero&&abs(k2) >= zero&&abs(k3)<zero)
			{
				if (xin >= -center_x&&xin <= center_x&&zin >= -center_z&&zin <= center_z)
				{
					iin = int(xin + center_x + 0.5); iout = iin;
					kin = int(zin + center_z + 0.5); kout = kin;
					if (k2>0)
					{
						jout = Ysize - 1; jin = 0;
					}
					else
					{
						jin = Ysize - 1; jout = 0;
					}

				}
			}
			///case 6, vertical to JK plane
			if (abs(k1) >= zero&&abs(k2)<zero&&abs(k3)<zero)
			{
				if (yin >= -center_y&&yin<center_y&&zin >= -center_z&&zin <= center_z)
				{
					jin = int(yin + center_y + 0.5); jout = jin;
					kin = int(zin + center_z + 0.5); kout = kin;
				}
				if (k1>0)
				{
					iout = Xsize - 1; iin = 0;
				}
				else
				{
					iin = Xsize - 1; iout = 0;
				}
			}
			/// case 7, purely inclined
			if (abs(k1) >= zero&&abs(k2) >= zero&&abs(k3) >= zero)
			{
				/// six crossing point
				float x1, x2, x3, x4, x5, x6;
				float y1, y2, y3, y4, y5, y6;
				float z1, z2, z3, z4, z5, z6;
				r = (-center_x - xin) / k1; x1 = -center_x; y1 = yin + k2*r; z1 = zin + k3*r;//x=0
				r = (center_x - xin) / k1; x2 = center_x; y2 = yin + k2*r; z2 = zin + k3*r;//x=max
				r = (-center_y - yin) / k2; x3 = xin + k1*r; y3 = -center_y; z3 = zin + k3*r;//y=0;
				r = (center_y - yin) / k2; x4 = xin + k1*r; y4 = center_y; z4 = zin + k3*r;//y=max
				r = (-center_z - zin) / k3; x5 = xin + k1*r; y5 = yin + k2*r; z5 = -center_z;//z=0;
				r = (center_z - zin) / k3; x6 = xin + k1*r; y6 = yin + k2*r; z6 = center_z;//z=max
				bool flag = 0;
				if (y1 <= center_y&&y1 >= -center_y&&z1 <= center_z&&z1 >= -center_z)
				{
					if (flag == 0)
					{
						iin = 0;
						jin = int(y1 + center_y + 0.5);
						kin = int(z1 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = 0;
						jout = int(y1 + center_y + 0.5);
						kout = int(z1 + center_z + 0.5);
					}
					flag = 1;
				}
				if (y2 <= center_y&&y2 >= -center_y&&z2 <= center_z&&z2 >= -center_z)
				{
					if (flag == 0)
					{
						iin = Xsize - 1;
						jin = int(y2 + center_y + 0.5);
						kin = int(z2 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = Xsize - 1;
						jout = int(y2 + center_y + 0.5);
						kout = int(z2 + center_z + 0.5);
					}
					flag = 1;
				}
				if (x3 <= center_x&&x3 >= -center_x&&z3 <= center_z&&z3 >= -center_z)
				{
					if (flag == 0)
					{
						iin = int(x3 + center_x + 0.5);
						jin = 0;
						kin = int(z3 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(x3 + center_x + 0.5);
						jout = 0;
						kout = int(z3 + center_z + 0.5);
					}
					flag = 1;
				}
				if (x4 <= center_x&&x4 >= -center_x&&z4 <= center_z&&z4 >= -center_z)
				{
					if (flag == 0)
					{
						iin = int(x4 + center_x + 0.5);
						jin = Ysize - 1;
						kin = int(z4 + center_z + 0.5);
					}
					if (flag == 1)
					{
						iout = int(x4 + center_x + 0.5);
						jout = Ysize - 1;
						kout = int(z4 + center_z + 0.5);
					}
					flag = 1;
				}
				if (x5 <= center_x&&x5 >= -center_x&&y5 <= center_y&&y5 >= -center_y)
				{
					if (flag == 0)
					{
						iin = int(x5 + center_x + 0.5);
						jin = int(y5 + center_y + 0.5);
						kin = 0;
					}
					if (flag == 1)
					{
						iout = int(x5 + center_x + 0.5);
						jout = int(y5 + center_y + 0.5);
						kout = 0;
					}
					flag = 1;
				}
				if (x6 <= center_x&&x6 >= -center_x&&y6 <= center_y&&y6 >= -center_y)
				{
					if (flag == 0)
					{
						iin = int(x6 + center_x + 0.5);
						jin = int(y6 + center_y + 0.5);
						kin = Zsize - 1;
					}
					if (flag == 1)
					{
						iout = int(x6 + center_x + 0.5);
						jout = int(y6 + center_y + 0.5);
						kout = Zsize - 1;
					}
					flag = 1;
				}
				//sorting intersection point by in, out order
				if ((iout - iin)*k1 + (jout - jin)*k2 + (kout - kin)*k3<0)
				{
					int temp;
					temp = iin; iin = temp; iout = temp;
					temp = jin; jin = jout; jout = temp;
					temp = kin; kin = kout; kout = temp;
				}
			}
			//////////////////////////////END OF CALCULATING IN AND OUT POINT ON REAL BOUNDARY////////////////////////////////
			if ((iin - center_x - xin)*(iin - center_x - xout) + (jin - center_y - yin)*(jin - center_y - yout) + (kin - center_z - zin)*(kin - center_z - zout)<0 && (iin + jin + kin + iout + jout + kout) != 0 && !(iin == iout&&jin == jout&&kin == kout))
			{
				long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
				ilast = iin; jlast = jin; klast = kin;
				do
				{
					if (ilast<iout)
					{
						inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
					}
					if (ilast == iout)
					{
						inext1 = ilast - 1e6; jnext1 = jlast; knext1 = klast;
					}
					if (ilast>iout)
					{
						inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
					}
					if (jlast<jout)
					{
						inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
					}
					if (jlast == jout)
					{
						inext2 = ilast; jnext2 = jlast - 1e6; knext2 = klast;
					}
					if (jlast>jout)
					{
						inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
					}
					if (klast<kout)
					{
						inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
					}
					if (klast == kout)
					{
						inext3 = ilast; jnext3 = jlast; knext3 = klast - 1e6;
					}
					if (klast>kout)
					{
						inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
					}
					///determine which one is closer to longegration path
					r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
					x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
					d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
					r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
					x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
					d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
					r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
					x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
					d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
					//////End of calculation distance///////////////
					int nin, nout;
					nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
					nout = index[iout + jout*Xsize*kout*Xsize*Ysize];
					if (d1 <= d2&&d1 <= d3)
					{
						pint[nin + nout*n] += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
						ilast = inext1;

					}
					if (d2<d1&&d2 <= d3)
					{
						pint[nin + nout*n] += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
						jlast = jnext2;
					}
					if (d3<d1&&d3<d2)
					{
						pint[nin + nout*n] += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
						klast = knext3;
					}
				} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5);
			}

		}

	}
}
__global__ void BCiteration(long Xsize, long Ysize, long Zsize, float* pint, float *pcount, float *p, float* pn, int itrNo)
{

	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	long nout = blockDim.x*blockIdx.x + threadIdx.x;
	for (int iteration = 0; iteration<itrNo; iteration++)
	{
		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = 0;
			pcount[nout] = 0;
			for (int nin = 0; nin<n; nin++)
			{
				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////

				if (pint[nin + nout*n] != 0)
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n];
					pcount[nout]++;
				}
			}

			pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / pcount[nout];
			//p[iout+jout*Xsize+kout*Xsize*Ysize]=pn[iout+jout*Xsize+kout*Xsize*Ysize];

			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;
			nout = nout + blockDim.x*gridDim.x;
			//nin=nin+blockDim.y*gridDim.y;

		}
		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			nout = nout + blockDim.x*gridDim.x;
		}
		__syncthreads();
	}


}
__global__ void BCiterationFixedBC(long Xsize, long Ysize, long Zsize, float* pint, float *pcount, float *p, float* pn, int itrNo)
{

	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	long nout = blockDim.x*blockIdx.x + threadIdx.x;
	for (int iteration = 0; iteration<itrNo; iteration++)
	{
		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = 0;
			pcount[nout] = 0;
			for (int nin = 0; nin<n; nin++)
			{
				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////
				/// boundary pressure fixed on the top surface....
				if (pint[nin + nout*n] != 0&&jout!=Ysize-1)
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n];
					pcount[nout]++;
				}
			}

			pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / pcount[nout];
			//p[iout+jout*Xsize+kout*Xsize*Ysize]=pn[iout+jout*Xsize+kout*Xsize*Ysize];

			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;
			nout = nout + blockDim.x*gridDim.x;
			//nin=nin+blockDim.y*gridDim.y;

		}
		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			nout = nout + blockDim.x*gridDim.x;
		}
		__syncthreads();
	}


}
__global__ void BCiterationWeighted(long Xsize, long Ysize, long Zsize, float* pint, float *pweight, float *p, float* pn, int itrNo)
{

	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	long nout = blockDim.x*blockIdx.x + threadIdx.x;
	for (int iteration = 0; iteration<itrNo; iteration++)
	{
		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = 0;
			float pcounttmp = 0;
			for (int nin = 0; nin<n; nin++)
			{
				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////

				if (pint[nin + nout*n] != 0)
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n])*pweight[nin + nout*n];
					pcounttmp += pweight[nin + nout*n];
				}
			}
			if (pcounttmp != 0)
			{
				pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / pcounttmp;
			}

			//p[iout+jout*Xsize+kout*Xsize*Ysize]=pn[iout+jout*Xsize+kout*Xsize*Ysize];

			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;
			nout = nout + blockDim.x*gridDim.x;
			//nin=nin+blockDim.y*gridDim.y;

		}

		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = 0;
			nout = nout + blockDim.x*gridDim.x;
		}
		__syncthreads();
	}


}
__global__ void BCiterationWeightedFixedBC(long Xsize, long Ysize, long Zsize, float* pint, float *pweight, float *p, float* pn, int itrNo)
{

	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	long nout = blockDim.x*blockIdx.x + threadIdx.x;
	for (int iteration = 0; iteration<itrNo; iteration++)
	{
		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = 0;
			float pcounttmp = 0;
			for (int nin = 0; nin<n; nin++)
			{
				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////

				if (pint[nin + nout*n] != 0&&jout!=0)
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += (p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n])*pweight[nin + nout*n];
					pcounttmp += pweight[nin + nout*n];
				}
			}
			if (pcounttmp != 0)
			{
				pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / pcounttmp;
			}

			//p[iout+jout*Xsize+kout*Xsize*Ysize]=pn[iout+jout*Xsize+kout*Xsize*Ysize];

			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;
			nout = nout + blockDim.x*gridDim.x;
			//nin=nin+blockDim.y*gridDim.y;

		}

		nout = blockDim.x*blockIdx.x + threadIdx.x;
		while (nout<n)
		{
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = 0;
			nout = nout + blockDim.x*gridDim.x;
		}
		__syncthreads();
	}


}

__global__ void omni3dparallellinesESInner(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInner(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner);
			}

		}
		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void omni3dparallellinesESInnerStepCount(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float*p, float*pn, float*pcountinner, long* IntegrationSteps)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInnerStepCount(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner, IntegrationSteps);
			}

		}
		point += blockDim.x*gridDim.x;
	}
}

__global__ void omni3dparallellinesESInnerWeighted(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInnerWeighted(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, curl, p, pn, pcountinner);

			}

		}
		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void omni3dparallellinesESInnerWeightedMiniCurl(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInnerMiniCurl(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, curl, p, pn, pcountinner);

			}

		}
		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}

__global__ void omni3dparallellinesESInnerSelect(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float*p, float*pn, float*pcountinner, float threshold)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout < Ysize && kout >= 0 && kout<Zsize)
		{

			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInnerSelect(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, curl, pcountinner, threshold);
			}


		}
		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void omni3dparallellinesESInnerSelectFixedBC(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* curl, float*p, float*pn, float*pcountinner, float threshold)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin>=0&&jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize && kout >= 0 && kout<Zsize)
		{

			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInnerSelectFixedBC(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, curl, pcountinner, threshold);
			}


		}
		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}


__global__ void omni3dparallellinesESInnerSelect2(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize && (phi<PI / 4 || phi>3 * PI / 4))
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInner(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner);
			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}

__global__ void omni2dparallellinesOnFaceInner(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.414 / linespacing;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;

	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid&&angle<NoAngles)
	{
		float theta = angle / NoAngles * 2 * PI;
		///on XY face
		float k1 = __cosf(theta);
		float k2 = __sinf(theta);
		float k3 = 0;
		float x = __sinf(theta)*(point - NoGrid / 2)*linespacing;
		float y = __cosf(theta)*(point - NoGrid / 2)*linespacing;
		float z = -Zsize / 2.0;
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				if (nin != nout)
				{
					bodyIntegralInner(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner);
				}
			}

		}
		///on XY face 2
		z = Zsize / 2.0;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInner(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner);
			}

		}

		point += blockDim.x*gridDim.x;
	}
}
__global__ void omni3dparallellinesESInner2(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	long angle = threadIdx.y + blockDim.y*blockIdx.y;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732 / linespacing;
	int point = threadIdx.x + blockDim.x*blockIdx.x;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	while (point<NoGrid*NoGrid&&angle<NoAngles)
	{
		float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
		float k1, k2, k3;
		k1 = k1_d[angle];
		k2 = k2_d[angle];
		k3 = k3_d[angle];
		float theta = acosf(k3);
		float phi = 0;
		if (__sinf(theta) != 0)
		{
			phi = asinf(k2 / __sinf(theta));
			if (k1 / __sinf(theta)<0)
			{
				phi = -phi + PI;
			}
		}
		else
		{
			phi = 0;
		}
		float x = xprime*__cosf(theta)*__cosf(phi) - yprime*__sinf(phi);
		float y = xprime*__cosf(theta)*__sinf(phi) + yprime*__cosf(phi);
		float z = -xprime*__sinf(theta);

		//float k1=__sinf(theta)*__cosf(phi);
		//float k2=__sinf(theta)*__sinf(phi);
		//float k3=__cosf(theta);
		int iin, jin, kin, iout, jout, kout;
		cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
		int nin, nout;
		if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
		{
			nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
			nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
			if (nin != nout)
			{
				bodyIntegralInner2(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner);
			}

		}

		point += blockDim.x*gridDim.x;
		angle += blockDim.y*gridDim.y;
	}
}
__global__ void devidecountInner(long Xsize, long Ysize, long Zsize, float* p, float* pn, float* pcountinner)
{
	long tid = threadIdx.x + blockDim.x*blockIdx.x;
	while (tid<Xsize*Ysize*Zsize)
	{
		if (pcountinner[tid]>0)
		{
			pn[tid] = pn[tid] / pcountinner[tid];
			p[tid] = pn[tid];
			pn[tid] = 0;
		}
		tid += blockDim.x*gridDim.x;
	}
}
__global__ void BCiterationvirtualgrid(long Xsize, long Ysize, long Zsize, int NoTheta, int NoBeta, long* index, long* ninvir, long *noutvir, float* pintvir, float*p, float *pn, int Noitr)

{
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	//virtual boundary an ellipsoid
	int a = Xsize - 1;
	int b = Ysize - 1;
	int c = Zsize - 1;
	float delttheta = PI / NoTheta;
	float deltbeta = 2 * PI / NoBeta;
	float xin, yin, zin, xout, yout, zout, k1, k2, k3, x, y, z;
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	for (int iteration = 0; iteration<Noitr; iteration++)
	{
		indexin = blockDim.x*blockIdx.x + threadIdx.x;
		indexout = blockDim.y*blockIdx.y + threadIdx.y;
		while (indexin<int(PI / delttheta)*int(PI / deltbeta) * 2 && indexout<int(PI / delttheta)*int(PI / deltbeta) * 2)
		{
			int nin, nout;
			nin = ninvir[indexin + indexout*NoTheta*NoBeta];
			nout = noutvir[indexin + indexout*NoTheta*NoBeta];
			long iout, jout, kout, iin, jin, kin;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			if (nin <= Xsize*Ysize - 1)
			{
				kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
			}
			if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
			}
			if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
			}
			if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
				iin = Xsize - 2 - iin;
				kin = kin + 1;
			}
			if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
				kin = Zsize - 2 - kin;
				jin = jin + 1;
			}
			if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
				kin = Zsize - 2 - kin;
				iin = iin + 1;
			}
			int beta = 0;
			if (pintvir[indexin + indexout*NoTheta*NoBeta] != 0)
			{
				pn[iout + jout*Xsize + kout*Xsize*Ysize] = (pn[iout + jout*Xsize + kout*Xsize*Ysize] + p[iin + jin*Xsize + kin*Xsize*Ysize] + pintvir[indexin + indexout*NoTheta*NoBeta])*0.5;
			}
			indexin = indexin + blockDim.x*gridDim.x;
			indexout = indexout + blockDim.y*gridDim.y;
		}
	}


}
__global__ void calCurlofMaterialAcc(long Xsize, long Ysize, long Zsize, float deltx, float delty, float deltz, float* DuDt, float * DvDt, float * DwDt, float * curl)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k = blockDim.z*blockIdx.z + threadIdx.z;
	while (i<Xsize&&j<Ysize&&k<Zsize)
	{
		int i0 = i - 1 >= 0 ? i - 1 : i;
		int j0 = j - 1 >= 0 ? j - 1 : j;
		int k0 = k - 1 >= 0 ? k - 1 : k;
		int ie = i + 1 <= Xsize - 1 ? i + 1 : i;
		int je = j + 1 <= Ysize - 1 ? j + 1 : j;
		int ke = k + 1 <= Zsize - 1 ? k + 1 : k;
		float curlx = (DwDt[i + je*Xsize + k*Xsize*Ysize] - DwDt[i + j0*Xsize + k*Xsize*Ysize]) / (je - j0) / delty;
		curlx += -(DvDt[i + j*Xsize + ke*Xsize*Ysize] - DvDt[i + j*Xsize + k0*Xsize*Ysize]) / (ke - k0) / deltz;
		float curly = -(DwDt[ie + j*Xsize + k*Xsize*Ysize] - DwDt[i0 + j*Xsize + k*Xsize*Ysize]) / (ie - i0) / deltx;
		curly += (DuDt[i + j*Xsize + ke*Xsize*Ysize] - DuDt[i + j*Xsize + k0*Xsize*Ysize]) / (ke - k0) / deltz;
		float curlz = (DvDt[ie + j*Xsize + k*Xsize*Ysize] - DvDt[i0 + j*Xsize + k*Xsize*Ysize]) / (ie - i0) / deltx;
		curlz += -(DuDt[i + je*Xsize + k*Xsize*Ysize] - DuDt[i + j0*Xsize + k*Xsize*Ysize]) / (je - j0) / delty;
		curl[i + j*Xsize + k*Xsize*Ysize] = sqrt(curlx*curlx + curly*curly + curlz*curlz);

		i += blockDim.x*gridDim.x;
		j += blockDim.y*gridDim.y;
		k += blockDim.z*gridDim.z;
	}
}
void omni3virtualcpu(long Xsize, long Ysize, long Zsize, long *index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, long *pcount)
{
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	//virtual boundary an ellipsoid
	int a = Xsize - 1;
	int b = Ysize - 1;
	int c = Zsize - 1;
	float delttheta = PI / 16;
	float deltbeta = PI / 16;
	float xin, yin, zin, xout, yout, zout, k1, k2, k3, x, y, z;
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	int iin, jin, kin, iout, jout, kout, indexin, indexout;
	CStdioFile log;
	log.Open(_T("log.dat"), CFile::modeCreate | CFile::modeWrite);
	for (float thetaout = 0; thetaout<PI; thetaout += delttheta)
	{
		for (float betaout = 0; betaout<2 * PI; betaout += deltbeta)
		{
			for (float thetain = 0; thetain<PI; thetain += delttheta)
			{
				for (float betain = 0; betain<PI; betain += deltbeta)
				{
					xin = a*sin(thetain)*cos(betain);
					yin = b*sin(thetain)*sin(betain);
					zin = c*cos(thetain);
					xout = a*sin(thetaout)*cos(betaout);
					yout = b*sin(thetaout)*sin(betaout);
					zout = c*cos(thetaout);
					k1 = xout - xin;
					k2 = yout - yin;
					k3 = zout - zin;
					iin = 0; iout = 0; jin = 0; jout = 0; kin = 0; kout = 0;
					if (!(k1 == 0 && k2 == 0 && k3 == 0))
					{
						/////case 1, vertical to x-axis
						if (k1 == 0 && k2 != 0 && k3 != 0)
						{
							if (xin >= -center_x&&xin <= center_x)
							{
								////four crossing point;y=0;y=max;z=0;z=max;
								float r = (-center_y - yin) / k2; float y1 = -center_y; float z1 = zin + k3*r;
								r = (center_y - yin) / k2; float y2 = center_y; float z2 = zin + k3*r;
								r = (-center_z - zin) / k3; float z3 = -center_z; float y3 = yin + k2*r;
								r = (center_z - zin) / k3; float z4 = center_z; float y4 = yin + k2*r;
								bool flag = 0;
								if (z1 <= center_z&&z1 >= -center_z)//cross y=0;
								{
									if (flag == 0)
									{
										iin = int(xin + center_x + 0.5);
										jin = 0;
										kin = int(z1 + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = int(xin + center_x + 0.5);
										jout = 0;
										kout = int(z1 + center_z + 0.5);
									}
									flag = 1;
								}
								if (z2 <= center_z&&z2 >= -center_z)//y=max;
								{
									if (flag == 0)
									{
										iin = int(xin + center_x + 0.5);
										jin = Ysize - 1;
										kin = int(z2 + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = int(xin + center_x + 0.5);
										jout = Ysize - 1;
										kout = int(z2 + center_z + 0.5);
									}
									flag = 1;
								}
								if (y3 <= center_y&&y3 >= -center_y)//z=0;
								{
									if (flag == 0)
									{
										iin = int(xin + center_x + 0.5);
										jin = int(y3 + center_y + 0.5);
										kin = 0;
									}
									if (flag == 1)
									{
										iout = int(xin + center_x + 0.5);
										jout = int(y3 + center_y + 0.5);
										kout = 0;
									}
									flag = 1;
								}
								if (y4 <= center_y&&y4 >= -center_y)
								{
									if (flag == 0)
									{
										iin = int(xin + center_x + 0.5);
										jin = int(y4 + center_y + 0.5);
										kin = Zsize - 1;
									}
									if (flag == 1)
									{
										iout = int(xin + center_x + 0.5);
										jout = int(y4 + center_y + 0.5);
										kout = Zsize - 1;
									}
								}
								//sorting intersection point by in, out order
								if (flag != 0)
								{
									if ((jout - jin)*k2 + (kout - kin)*k3<0)
									{
										int temp;
										temp = jin; jin = jout; jout = temp;
										temp = kin; kin = kout; kout = temp;
									}
								}

							}
						}
						///case 2, vertical to y-axis
						if (k1 != 0 && k2 == 0 && k3 != 0)
						{
							if (yin >= -center_y&&yin <= center_y)
							{
								////four crossing point
								float r = (-center_x - xin) / k1; float x1 = -center_x; float z1 = zin + k3*r;//x=0;
								r = (center_x - xin) / k1; float x2 = center_x; float z2 = zin + k3*r;//x=max
								r = (-center_z - zin) / k3; float z3 = -center_z; float x3 = xin + k1*r;//z=0;
								r = (center_z - zin) / k3; float z4 = center_z; float x4 = xin + k1*r;//z=max;
								bool flag = 0;
								if (z1 <= center_z&&z1 >= -center_z)
								{
									if (flag == 0)
									{
										iin = 0;
										jin = int(yin + center_y + 0.5);
										kin = int(z1 + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = 0;
										jout = int(yin + center_y + 0.5);
										kout = int(z1 + center_z + 0.5);
									}
									flag = 1;
								}
								if (z2 <= center_z&&z2 >= -center_z)
								{
									if (flag == 0)
									{
										iin = Xsize - 1;
										jin = int(yin + center_y + 0.5);
										kin = int(z2 + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = Xsize - 1;
										jout = int(yin + center_y + 0.5);
										kout = int(z2 + center_z + 0.5);
									}
									flag = 1;
								}
								if (x3 <= center_x&&x3 >= -center_x)
								{
									if (flag == 0)
									{
										iin = int(x3 + center_x + 0.5);
										jin = int(yin + center_y + 0.5);
										kin = 0;
									}
									if (flag == 1)
									{
										iout = int(x3 + center_x + 0.5);
										jout = int(yin + center_y + 0.5);
										kout = 0;
									}
									flag = 1;
								}
								if (x4 <= center_x&&x4 >= -center_x)
								{
									if (flag == 0)
									{
										iin = int(x4 + center_x + 0.5);
										jin = int(yin + center_y + 0.5);
										kin = Zsize - 1;
									}
									if (flag == 1)
									{
										iout = int(x4 + center_x + 0.5);
										jout = int(yin + center_y + 0.5);
										kout = Zsize - 1;
									}
									flag = 1;
								}
								//sorting intersection point by in, out order
								if (flag != 0)
								{
									if ((iout - iin)*k1 + (kout - kin)*k3<0)
									{
										int temp;
										temp = iin; iin = iout; iout = temp;
										temp = kin; kin = kout; kout = temp;
									}
								}

							}
						}
						///case 3, vertical to z-axis
						if (k1 != 0 && k2 != 0 && k3 == 0)
						{
							if (zin >= -center_z&&zin <= center_z)
							{
								////four crossing point
								float r = (-center_x - xin) / k1; float x1 = -center_x; float y1 = yin + k2*r;//x=0;
								r = (center_x - xin) / k1; float x2 = center_x; float y2 = yin + k2*r;//x=max;
								r = (-center_y - zin) / k2; float y3 = -center_y; float x3 = xin + k1*r;//y=0;
								r = (center_y - zin) / k2; float y4 = center_y; float x4 = xin + k1*r;//y=max;
								bool flag = 0;
								if (y1 <= center_y&&y1 >= -center_y)
								{
									if (flag == 0)
									{
										iin = 0;
										jin = int(y1 + center_y + 0.5);
										kin = int(zin + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = 0;
										jout = int(y1 + center_y + 0.5);
										kout = int(zin + center_z + 0.5);
									}
									flag = 1;
								}
								if (y2 <= center_y&&y2 >= -center_y)
								{
									if (flag == 0)
									{
										iin = Xsize - 1;
										jin = int(y2 + center_y + 0.5);
										kin = int(zin + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = Xsize - 1;
										jout = int(y2 + center_y + 0.5);
										kout = int(zin + center_z + 0.5);
									}
									flag = 1;
								}
								if (x3 <= center_x&&x3 >= -center_x)
								{
									if (flag == 0)
									{
										iin = int(x3 + center_x + 0.5);
										jin = 0;
										kin = int(zin + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = int(x3 + center_x + 0.5);
										jout = 0;
										kout = int(zin + center_z + 0.5);
									}
									flag = 1;
								}
								if (x4 <= center_x&&x4 >= -center_x)
								{
									if (flag == 0)
									{
										iin = int(x4 + center_x + 0.5);
										jin = Ysize - 1;
										kin = int(zin + center_z + 0.5);
									}
									if (flag == 1)
									{
										iout = int(x4 + center_x + 0.5);
										jout = Ysize - 1;
										kout = int(zin + center_z + 0.5);
									}
									flag = 1;
								}
								//sorting intersection point by in, out order
								if (flag != 0)
								{
									if ((iout - iin)*k1 + (jout - jin)*k2<0)
									{
										int temp;
										temp = iin; iin = iout; iout = temp;
										temp = jin; jin = jout; jout = temp;
									}
								}

							}
						}
						///case 4, vertical to plane IJ
						if (abs(k1)<zero&&abs(k2)<zero&&abs(k3) >= zero)
						{

							if (xin <= center_x&&xin >= -center_x&&yin <= center_y&&yin >= -center_y)
							{
								iin = int(xin + center_x + 0.5); iout = iin;
								jin = int(yin + center_y + 0.5); jout = jin;
								if (k3>0)
								{
									kin = 0; kout = Zsize - 1;
								}
								else{
									kin = Zsize - 1; kout = 0;
								}

							}
						}
						///case 5, vertical to IK plane
						if (abs(k1)<zero&&abs(k2) >= zero&&abs(k3)<zero)
						{
							if (xin >= -center_x&&xin <= center_x&&zin >= -center_z&&zin <= center_z)
							{
								iin = int(xin + center_x + 0.5); iout = iin;
								kin = int(zin + center_z + 0.5); kout = kin;
								if (k2>0)
								{
									jout = Ysize - 1; jin = 0;
								}
								else
								{
									jin = Ysize - 1; jout = 0;
								}

							}
						}
						///case 6, vertical to JK plane
						if (abs(k1) >= zero&&abs(k2)<zero&&abs(k3)<zero)
						{
							if (yin >= -center_y&&yin<center_y&&zin >= -center_z&&zin <= center_z)
							{
								jin = int(yin + center_y + 0.5); jout = jin;
								kin = int(zin + center_z + 0.5); kout = kin;
								if (k1>0)
								{
									iout = Xsize - 1; iin = 0;
								}
								else
								{
									iin = Xsize - 1; iout = 0;
								}
							}

						}
						/// case 7, purely inclined
						if (abs(k1) >= zero&&abs(k2) >= zero&&abs(k3) >= zero)
						{
							/// six crossing point
							float r;
							float x1, x2, x3, x4, x5, x6;
							float y1, y2, y3, y4, y5, y6;
							float z1, z2, z3, z4, z5, z6;
							r = (-center_x - xin) / k1; x1 = -center_x; y1 = yin + k2*r; z1 = zin + k3*r;//x=0
							r = (center_x - xin) / k1; x2 = center_x; y2 = yin + k2*r; z2 = zin + k3*r;//x=max
							r = (-center_y - yin) / k2; x3 = xin + k1*r; y3 = -center_y; z3 = zin + k3*r;//y=0;
							r = (center_y - yin) / k2; x4 = xin + k1*r; y4 = center_y; z4 = zin + k3*r;//y=max
							r = (-center_z - zin) / k3; x5 = xin + k1*r; y5 = yin + k2*r; z5 = -center_z;//z=0;
							r = (center_z - zin) / k3; x6 = xin + k1*r; y6 = yin + k2*r; z6 = center_z;//z=max
							bool flag = 0;
							if (y1 <= center_y&&y1 >= -center_y&&z1 <= center_z&&z1 >= -center_z)
							{
								if (flag == 0)
								{
									iin = 0;
									jin = int(y1 + center_y + 0.5);
									kin = int(z1 + center_z + 0.5);
								}
								if (flag == 1)
								{
									iout = 0;
									jout = int(y1 + center_y + 0.5);
									kout = int(z1 + center_z + 0.5);
								}
								flag = 1;
							}
							if (y2 <= center_y&&y2 >= -center_y&&z2 <= center_z&&z2 >= -center_z)
							{
								if (flag == 0)
								{
									iin = Xsize - 1;
									jin = int(y2 + center_y + 0.5);
									kin = int(z2 + center_z + 0.5);
								}
								if (flag == 1)
								{
									iout = Xsize - 1;
									jout = int(y2 + center_y + 0.5);
									kout = int(z2 + center_z + 0.5);
								}
								flag = 1;
							}
							if (x3 <= center_x&&x3 >= -center_x&&z3 <= center_z&&z3 >= -center_z)
							{
								if (flag == 0)
								{
									iin = int(x3 + center_x + 0.5);
									jin = 0;
									kin = int(z3 + center_z + 0.5);
								}
								if (flag == 1)
								{
									iout = int(x3 + center_x + 0.5);
									jout = 0;
									kout = int(z3 + center_z + 0.5);
								}
								flag = 1;
							}
							if (x4 <= center_x&&x4 >= -center_x&&z4 <= center_z&&z4 >= -center_z)
							{
								if (flag == 0)
								{
									iin = int(x4 + center_x + 0.5);
									jin = Ysize - 1;
									kin = int(z4 + center_z + 0.5);
								}
								if (flag == 1)
								{
									iout = int(x4 + center_x + 0.5);
									jout = Ysize - 1;
									kout = int(z4 + center_z + 0.5);
								}
								flag = 1;
							}
							if (x5 <= center_x&&x5 >= -center_x&&y5 <= center_y&&y5 >= -center_y)
							{
								if (flag == 0)
								{
									iin = int(x5 + center_x + 0.5);
									jin = int(y5 + center_y + 0.5);
									kin = 0;
								}
								if (flag == 1)
								{
									iout = int(x5 + center_x + 0.5);
									jout = int(y5 + center_y + 0.5);
									kout = 0;
								}
								flag = 1;
							}
							if (x6 <= center_x&&x6 >= -center_x&&y6 <= center_y&&y6 >= -center_y)
							{
								if (flag == 0)
								{
									iin = int(x6 + center_x + 0.5);
									jin = int(y6 + center_y + 0.5);
									kin = Zsize - 1;
								}
								if (flag == 1)
								{
									iout = int(x6 + center_x + 0.5);
									jout = int(y6 + center_y + 0.5);
									kout = Zsize - 1;
								}
								flag = 1;
							}
							//sorting intersection point by in, out order
							if (flag != 0)
							{
								if ((iout - iin)*k1 + (jout - jin)*k2 + (kout - kin)*k3<0)
								{
									int temp;
									temp = iin; iin = temp; iout = temp;
									temp = jin; jin = jout; jout = temp;
									temp = kin; kin = kout; kout = temp;
								}
							}

						}
						//////////////////////////////END OF CALCULATING IN AND OUT POINT ON REAL BOUNDARY////////////////////////////////
						if ((iin - center_x - xin)*(iin - center_x - xout) + (jin - center_y - yin)*(jin - center_y - yout) + (kin - center_z - zin)*(kin - center_z - zout)<0 && (iin + jin + kin + iout + jout + kout) != 0 && !(iin == iout&&jin == jout&&kin == kout))
						{
							int nin, nout;
							long ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
							ilast = iin; jlast = jin; klast = kin;
							do
							{
								if (ilast<iout)
								{
									inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
								}
								if (ilast == iout)
								{
									inext1 = ilast - 1e6; jnext1 = jlast; knext1 = klast;
								}
								if (ilast>iout)
								{
									inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
								}
								if (jlast<jout)
								{
									inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
								}
								if (jlast == jout)
								{
									inext2 = ilast; jnext2 = jlast - 1e6; knext2 = klast;
								}
								if (jlast>jout)
								{
									inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
								}
								if (klast<kout)
								{
									inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
								}
								if (klast == kout)
								{
									inext3 = ilast; jnext3 = jlast; knext3 = klast - 1e6;
								}
								if (klast>kout)
								{
									inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
								}
								///determine which one is closer to longegration path
								float r, d1, d2, d3;
								r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
								x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
								d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
								r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
								x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
								d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
								r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
								x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
								d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
								//////End of calculation distance///////////////
								nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
								nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
								/*if(kin==0)
								{
								nin=iin+jin*Xsize;
								}
								if(iin==Xsize-1&&kin!=0)
								{
								nin=Xsize*Ysize-1+kin+(Ysize-1-jin)*(Zsize-1);
								}
								if(kin==Zsize-1&&iin!=Xsize-1)
								{
								nin=Xsize*Ysize-1+(Zsize-1)*Ysize+Xsize-1-iin+jin*(Xsize-1);
								}
								if(jin==0&&iin!=Xsize-1&&kin!=0&&kin!=Zsize-1)
								{
								nin=Xsize*Ysize-1+(Zsize-1)*Ysize+Ysize*(Xsize-1)+Xsize-1-iin+(kin-1)*(Xsize-1);//????
								}
								if(iin==0&&jin!=0&&kin!=0&&kin!=Zsize-1)
								{
								nin=Xsize*Ysize-1+(Zsize-1)*Ysize+Ysize*(Xsize-1)+(Xsize-1)*(Zsize-2)+Zsize-1-kin+(jin-1)*(Zsize-2);
								}
								if(jin==Ysize-1&&iin!=0&&iin!=Xsize-1&&kin!=0&&kin!=Zsize-1)
								{
								nin=Xsize*Ysize-1+(Zsize-1)*Ysize+Ysize*(Xsize-1)+(Xsize-1)*(Zsize-2)+(Ysize-1)*(Zsize-2)+iin+(kin-1)*(Xsize-2);
								}
								if(kout==0)
								{
								nout=iout+jout*Xsize;
								}
								if(iout==Xsize-1&&kout!=0)
								{
								nout=Xsize*Ysize-1+kout+(Ysize-1-jout)*(Zsize-1);
								}
								if(kout==Zsize-1&&iout!=Xsize-1)
								{
								nout=Xsize*Ysize-1+(Zsize-1)*Ysize+Xsize-1-iout+jout*(Xsize-1);
								}
								if(jout==0&&iout!=Xsize-1&&kout!=0&&kout!=Zsize-1)
								{
								nout=Xsize*Ysize-1+(Zsize-1)*Ysize+Ysize*(Xsize-1)+Xsize-1-iout+(kout-1)*(Xsize-1);
								}
								if(iout==0&&jout!=0&&kout!=0&&kout!=Zsize-1)
								{
								nout=Xsize*Ysize-1+(Zsize-1)*Ysize+Ysize*(Xsize-1)+(Xsize-1)*(Zsize-2)+Zsize-1-kout+(jout-1)*(Zsize-2);
								}
								if(jout==Ysize-1&&iout!=0&&iout!=Xsize-1&&kout!=0&&kout!=Zsize-1)
								{
								nout=Xsize*Ysize-1+(Zsize-1)*Ysize+Ysize*(Xsize-1)+(Xsize-1)*(Zsize-2)+(Ysize-1)*(Zsize-2)+iout+(kout-1)*(Xsize-2);
								}*/
								if (d1 <= d2&&d1 <= d3)
								{
									pint[nin + nout*n] += -density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);

									ilast = inext1;

								}
								if (d2<d1&&d2 <= d3)
								{
									pint[nin + nout*n] += -density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);

									jlast = jnext2;
								}
								if (d3<d1&&d3<d2)
								{
									pint[nin + nout*n] += -density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);

									klast = knext3;
								}
							} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-5);
							pcount[nin + nout*n]++;
							CString str;
							str.Format(_T("%04d--%04d  (%02d,%02d,%02d) (%02d,%02d,%02d) %10.8f %02d\n"), nin, nout, iin, jin, kin, iout, jout, kout, pint[nin + nout*n], pcount[nin + nout*n]);
							cout << str;
							log.WriteString(str);
						}

					}

				}
			}
		}
	}
	int no = 0;
	for (int k = 0; k<n*n; k++)
	{
		if (pcount[k]>0)
		{
			pint[k] = pint[k] / pcount[k];
			no++;
		}
	}
	cout << no << endl;
	log.Close();
}
float BCIterationCPU(long Xsize, long Ysize, long Zsize, float* pint, float *p, float* pn, float eps, int Noitr)
{
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float pdiffold = 0;
	float pdiffnew = 0;
	float pdiffrela = 100;
	float meanp = 0;
	long iteration = 0;
	while (iteration<Noitr&&pdiffrela>eps)
	{
		meanp = 0;
		for (long nout = n - 1; nout >= 0; nout--)
		{
			long iout, jout, kout;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0;
				kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0;
				jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1;
				kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}

			long beta = 0;
			for (long nin = 0; nin<n; nin++)
			{
				long iin, jin, kin;

				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////
				if (pint[nin + nout*n] != 0)
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n];
					beta++;
				}

			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / (beta + 1);
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			//cout<<pn[iout+jout*Xsize+kout*Xsize*Ysize]<<endl;
		}
		iteration++;
		for (long nout = 0; nout<n; nout++)
		{
			long iout, jout, kout;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			meanp += pn[iout + jout*Xsize + kout*Xsize*Ysize];
			pdiffnew += abs(p[iout + jout*Xsize + kout*Xsize*Ysize] - pn[iout + jout*Xsize + kout*Xsize*Ysize]);
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;

		}
		meanp = meanp / n;
		pdiffnew = pdiffnew / n;
		pdiffrela = abs(pdiffnew - pdiffold);
		pdiffold = pdiffnew; pdiffnew = 0;
	}
	return meanp;
}
float BCIterationCPUFixBC(long Xsize, long Ysize, long Zsize, float* pint, float *p, float* pn, float eps, int Noitr)
{
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float pdiffold = 0;
	float pdiffnew = 0;
	float pdiffrela = 100;
	float meanp = 0;
	long iteration = 0;
	while (iteration<Noitr&&pdiffrela>eps)
	{
		meanp = 0;
		for (long nout = n - 1; nout >= 0; nout--)
		{
			long iout, jout, kout;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			long beta = 0;
			for (long nin = 0; nin<n; nin++)
			{
				long iin, jin, kin;

				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////
				if (pint[nin + nout*n] != 0 && jout != Ysize - 1)
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n];
					beta++;
				}

			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / (beta + 1);
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			//cout<<pn[iout+jout*Xsize+kout*Xsize*Ysize]<<endl;
		}
		iteration++;
		for (long nout = 0; nout<n; nout++)
		{
			long iout, jout, kout;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			meanp += pn[iout + jout*Xsize + kout*Xsize*Ysize];
			pdiffnew += abs(p[iout + jout*Xsize + kout*Xsize*Ysize] - pn[iout + jout*Xsize + kout*Xsize*Ysize]);
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;

		}
		meanp = meanp / n;
		pdiffnew = pdiffnew / n;
		pdiffrela = abs(pdiffnew - pdiffold);
		pdiffold = pdiffnew; pdiffnew = 0;
	}
	return meanp;
}
float BCIterationCPUFixPoint(long Xsize, long Ysize, long Zsize, float* pint, float *p, float* pn, float eps, int Noitr)
{
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float pdiffold = 0;
	float pdiffnew = 0;
	float pdiffrela = 100;
	float meanp = 0;
	long iteration = 0;
	while (iteration<Noitr&&pdiffrela>eps)
	{
		meanp = 0;
		for (long nout = n - 1; nout >= 0; nout--)
		{
			long iout, jout, kout;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			long beta = 0;
			for (long nin = 0; nin<n; nin++)
			{
				long iin, jin, kin;

				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
				}
				///////////
				if (pint[nin + nout*n] != 0 && !(jout == Ysize - 1 && iout == 0 && kout == 0))
				{
					pn[iout + jout*Xsize + kout*Xsize*Ysize] += p[iin + jin*Xsize + kin*Xsize*Ysize] + pint[nin + nout*n];
					beta++;
				}

			}
			pn[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize] / (beta + 1);
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			//cout<<pn[iout+jout*Xsize+kout*Xsize*Ysize]<<endl;
		}
		iteration++;
		for (long nout = 0; nout<n; nout++)
		{
			long iout, jout, kout;
			if (nout <= Xsize*Ysize - 1)
			{
				kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
			}
			if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
			{
				iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
			{
				kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
			{
				jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
				iout = Xsize - 2 - iout;
				kout = kout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
				kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
				kout = Zsize - 2 - kout;
				jout = jout + 1;
			}
			if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
			{
				jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
				iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
				kout = Zsize - 2 - kout;
				iout = iout + 1;
			}
			meanp += pn[iout + jout*Xsize + kout*Xsize*Ysize];
			pdiffnew += abs(p[iout + jout*Xsize + kout*Xsize*Ysize] - pn[iout + jout*Xsize + kout*Xsize*Ysize]);
			p[iout + jout*Xsize + kout*Xsize*Ysize] = pn[iout + jout*Xsize + kout*Xsize*Ysize];
			//pn[iout+jout*Xsize+kout*Xsize*Ysize]=0;

		}
		meanp = meanp / n;
		pdiffnew = pdiffnew / n;
		pdiffrela = abs(pdiffnew - pdiffold);
		pdiffold = pdiffnew; pdiffnew = 0;
	}
	return meanp;
}

void omni3Dinner(long Xsize, long Ysize, long Zsize, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, long *pcount, float *p, float* pn, int itrNo)
{
	int iteration = 0;
	float rms = 0;
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	while (iteration<itrNo)
	{
		for (int nin = 0; nin<n; nin = nin + 1)
		{
			for (int nout = 0; nout<n; nout = nout + 1)
			{
				int iout, jout, kout;
				int facein, faceout;
				if (nout <= Xsize*Ysize - 1)
				{
					kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
					faceout = 1;
				}
				if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
					faceout = 2;
				}
				if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
					faceout = 3;
				}
				if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
					iout = Xsize - 2 - iout;
					kout = kout + 1;
					faceout = 4;
				}
				if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
					kout = Zsize - 2 - kout;
					jout = jout + 1;
					faceout = 5;
				}
				if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
					kout = Zsize - 2 - kout;
					iout = iout + 1;
					faceout = 6;
				}
				int iin, jin, kin;

				if (nin <= Xsize*Ysize - 1)
				{
					kin = 0; jin = nin / Xsize; iin = nin - Xsize*jin;
					facein = 1;
				}
				if (nin>Xsize*Ysize - 1 && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
				{
					iin = Xsize - 1; jin = (nin - Xsize*Ysize) / (Zsize - 1); kin = nin - Xsize*Ysize - jin*(Zsize - 1) + 1;
					facein = 2;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
				{
					kin = Zsize - 1; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - jin*(Xsize - 1); iin = Xsize - 2 - iin;
					facein = 3;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
				{
					jin = 0; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kin*(Xsize - 1);
					iin = Xsize - 2 - iin;
					kin = kin + 1;
					facein = 4;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nin <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					iin = 0; jin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
					kin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jin*(Zsize - 2);
					kin = Zsize - 2 - kin;
					jin = jin + 1;
					facein = 5;
				}
				if (nin>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
				{
					jin = Ysize - 1; kin = (nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
					iin = nin - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kin*(Xsize - 2);
					kin = Zsize - 2 - kin;
					iin = iin + 1;
					facein = 6;
				}
				int ilast, jlast, klast, inext1, inext2, inext3, jnext1, jnext2, jnext3, knext1, knext2, knext3;
				ilast = iin; jlast = jin; klast = kin;
				if (nin != nout&&nin >= 0 && nin<n&&nout >= 0 && nout<n)
				{
					float k1 = iout - iin;
					float k2 = jout - jin;
					float k3 = kout - kin;
					float l = sqrt(k1*k1 + k2*k2 + k3*k3);
					k1 = k1 / l;
					k2 = k2 / l;
					k3 = k3 / l;
					//cout<<"indexin: "<<nin<<" indexout:"<<nout<<endl;
					//cout<<'('<<iin<<','<<jin<<','<<kin<<")  "<<'('<<iout<<','<<jout<<','<<kout<<")  "<<endl;
					//log<<"indexin: "<<nin<<" indexout:"<<nout<<endl;
					//log<<'('<<iin<<','<<jin<<','<<kin<<")  "<<'('<<iout<<','<<jout<<','<<kout<<")  "<<endl;
					do
					{
						if (ilast<iout)
						{
							inext1 = ilast + 1; jnext1 = jlast; knext1 = klast;
						}
						if (ilast == iout)
						{
							inext1 = ilast - 1e6; jnext1 = jlast; knext1 = klast;
						}
						if (ilast>iout)
						{
							inext1 = ilast - 1; jnext1 = jlast; knext1 = klast;
						}
						if (jlast<jout)
						{
							inext2 = ilast; jnext2 = jlast + 1; knext2 = klast;
						}
						if (jlast == jout)
						{
							inext2 = ilast; jnext2 = jlast - 1e6; knext2 = klast;
						}
						if (jlast>jout)
						{
							inext2 = ilast; jnext2 = jlast - 1; knext2 = klast;
						}
						if (klast<kout)
						{
							inext3 = ilast; jnext3 = jlast; knext3 = klast + 1;
						}
						if (klast == kout)
						{
							inext3 = ilast; jnext3 = jlast; knext3 = klast - 1e6;
						}
						if (klast>kout)
						{
							inext3 = ilast; jnext3 = jlast; knext3 = klast - 1;
						}
						///determine which one is closer to integration path
						float r, d1, d2, d3, x, y, z;
						r = k1*inext1 - iin*k1 + k2*jnext1 - k2*jin + k3*knext1 - k3*kin;
						x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
						d1 = sqrt((x - inext1)*(x - inext1) + (y - jnext1)*(y - jnext1) + (z - knext1)*(z - knext1));
						r = k1*inext2 - iin*k1 + k2*jnext2 - k2*jin + k3*knext2 - k3*kin;
						x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
						d2 = sqrt((x - inext2)*(x - inext2) + (y - jnext2)*(y - jnext2) + (z - knext2)*(z - knext2));
						r = k1*inext3 - iin*k1 + k2*jnext3 - k2*jin + k3*knext3 - k3*kin;
						x = iin + k1*r; y = jin + k2*r; z = kin + k3*r;
						d3 = sqrt((x - inext3)*(x - inext3) + (y - jnext3)*(y - jnext3) + (z - knext3)*(z - knext3));
						//////End of calculation distance///////////////
						//path 1
						if (d1 <= d2&&d1 <= d3)
						{
							pn[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize] - density*(inext1 - ilast)*deltx*0.5*(DuDt[inext1 + jnext1*Xsize + knext1*Xsize*Ysize] + DuDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
							pcount[inext1 + jnext1*Xsize + knext1*Xsize*Ysize]++;
							//pint[nin+nout*n]+=-density*(inext1-ilast)*deltx*0.5*(DuDt[inext1+jnext1*Xsize+knext1*Xsize*Ysize]+DuDt[ilast+jlast*Xsize+klast*Xsize*Ysize]);
							ilast = inext1;

						}
						if (d2<d1&&d2 <= d3)
						{
							pn[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize] - density*(jnext2 - jlast)*delty*0.5*(DvDt[inext2 + jnext2*Xsize + knext2*Xsize*Ysize] + DvDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
							pcount[inext2 + jnext2*Xsize + knext2*Xsize*Ysize]++;
							//pint[nin+nout*n]+=-density*(jnext2-jlast)*delty*0.5*(DvDt[inext2+jnext2*Xsize+knext2*Xsize*Ysize]+DvDt[ilast+jlast*Xsize+klast*Xsize*Ysize]);

							jlast = jnext2;
						}
						if (d3<d1&&d3<d2)
						{
							pn[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] += p[ilast + jlast*Xsize + klast*Xsize*Ysize] - density*(knext3 - klast)*deltz*0.5*(DwDt[inext3 + jnext3*Xsize + knext3*Xsize*Ysize] + DwDt[ilast + jlast*Xsize + klast*Xsize*Ysize]);
							pcount[inext3 + jnext3*Xsize + knext3*Xsize*Ysize]++;
							//pint[nin+nout*n]+=-density*(knext3-klast)*deltz*0.5*(DwDt[inext3+jnext3*Xsize+knext3*Xsize*Ysize]+DwDt[ilast+jlast*Xsize+klast*Xsize*Ysize]);
							klast = knext3;
						}
					} while (abs(ilast - iout) + abs(jlast - jout) + abs(klast - kout)>1e-3);


				}

				//cout<<thetain<<' '<<betain<<endl;
				//cout<<thetaout<<' '<<betaout<<endl;
				//cout<<"k1="<<k1<<" k2="<<k2<<" k3="<<k3<<endl;
				//cout<<indexin<<" "<<indexout<<endl;
			}
		}
		rms = 0;
		for (int k = 0; k<Xsize*Ysize*Zsize; k++)
		{
			pn[k] = pn[k] / pcount[k];
			pcount[k] = 0;
			rms += (p[k] - pn[k])*(p[k] - pn[k]);
		}
		rms = sqrt(rms / Xsize / Ysize / Zsize);
		cout << "Iteration: " << iteration << " rms:  " << rms << endl;
		memcpy(p, pn, sizeof(float)*Xsize*Ysize*Zsize);
		memset(pn, 0, sizeof(float)*Xsize*Ysize*Zsize);
		iteration++;
	}

}
void calIndex(long*index, long Xsize, long Ysize, long Zsize)
{
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	for (long nout = n - 1; nout >= 0; nout--)
	{
		long iout, jout, kout;
		if (nout <= Xsize*Ysize - 1)
		{
			kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
		}
		if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
		{
			iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
		{
			kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
		{
			jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
			iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
			iout = Xsize - 2 - iout;
			kout = kout + 1;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
			kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
			kout = Zsize - 2 - kout;
			jout = jout + 1;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			jout = Ysize - 1; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
			iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
			kout = Zsize - 2 - kout;
			iout = iout + 1;
		}
		index[iout + jout*Xsize + kout*Xsize*Ysize] = nout;
	}
}
void omni3dparallellinesEqualSpacingCPU(long Xsize, long Ysize, long Zsize, int NoAngles, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float* pint, float* pcount, float* pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = 16;
	float spacing = 1;
	//CStdioFile log;
	//log.Open("log.dat",CFile::modeCreate|CFile::modeWrite);
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	for (int angle = 0; angle<NoAngles; angle++)
	{

		for (int point = 0; point<NoGrid*NoGrid; point++)
		{
			float xprime = (float(point / NoGrid) - 0.5*NoGrid)*spacing;
			float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*NoGrid)*spacing;
			float k1, k2, k3;
			k1 = k1_d[angle];
			k2 = k2_d[angle];
			k3 = k3_d[angle];
			float theta = acosf(k3);
			float phi = asinf(k2 / sinf(theta));
			if (k1 / sinf(theta)<0)
			{
				phi = -phi + PI;
			}
			float x = xprime*cosf(theta)*cosf(phi) - yprime*sinf(phi);
			float y = xprime*cosf(theta)*sinf(phi) + yprime*cosf(phi);
			float z = -xprime*sinf(theta);
			//float k1=sinf(theta)*cosf(phi);
			//float k2=sinf(theta)*sinf(phi);
			//float k3=cosf(theta);
			int iin, jin, kin, iout, jout, kout;
			cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
			int nin, nout;
			if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
			{
				nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
				nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
				if (nin != nout)
				{
					pint[nin + nout*n] += bodyIntegral(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, pcountinner);
					pcount[nin + nout*n]++;
				}
				//CString str;
				//str.Format(_T("%6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %02d %02d %02d %02d %02d %02d\n"),theta,phi,k1,k2,k3,x,y,z,iin,jin,kin,iout,jout,kout);
				//if(angle==10000/2-1)
				//{
				//	log.WriteString(str);
				//}

			}
		}
	}
	//log.Close();
}
void omni3dparallellinesESInnerCPU(long Xsize, long Ysize, long Zsize, int NoAngles, float linespacing, float* k1_d, float* k2_d, float* k3_d, long*index, float deltx, float delty, float deltz, float density, float* DuDt, float *DvDt, float *DwDt, float*p, float*pn, float*pcountinner)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	float center_x = (Xsize - 1) / 2.0;
	float center_y = (Ysize - 1) / 2.0;
	float center_z = (Zsize - 1) / 2.0;
	int NoGrid = Xsize;
	if (NoGrid<Ysize)
	{
		NoGrid = Ysize;
	}
	if (NoGrid<Zsize)
	{
		NoGrid = Zsize;
	}
	NoGrid = NoGrid*1.732;
	//float spacing=sqrt(float(Xsize*Xsize+Ysize*Ysize+Zsize*Zsize))/NoGrid;
	for (int angle = 0; angle<NoAngles; angle++)
	{
		for (int point = 0; point<NoGrid*NoGrid; point++)
		{
			float xprime = (float(point / NoGrid) - 0.5*(NoGrid - 1))*linespacing;
			float yprime = (float(point - point / NoGrid*NoGrid) - 0.5*(NoGrid - 1))*linespacing;
			float k1, k2, k3;
			k1 = k1_d[angle];
			k2 = k2_d[angle];
			k3 = k3_d[angle];
			float theta = acosf(k3);
			float phi = asinf(k2 / sinf(theta));
			if (k1 / sinf(theta)<0)
			{
				phi = -phi + PI;
			}
			float x = xprime*cosf(theta)*cosf(phi) - yprime*sinf(phi);
			float y = xprime*cosf(theta)*sinf(phi) + yprime*cosf(phi);
			float z = -xprime*sinf(theta);

			//float k1=__sinf(theta)*__cosf(phi);
			//float k2=__sinf(theta)*__sinf(phi);
			//float k3=__cosf(theta);
			int iin, jin, kin, iout, jout, kout;
			cross2point(Xsize, Ysize, Zsize, &iin, &jin, &kin, x, y, z, k1, k2, k3, &iout, &jout, &kout);
			int nin, nout;
			if (iin >= 0 && iin<Xsize&&jin >= 0 && jin<Ysize&&kin >= 0 && kin<Zsize&&iout >= 0 && iout<Xsize&&jout >= 0 && jout<Ysize&&kout >= 0 && kout<Zsize)
			{
				nin = index[iin + jin*Xsize + kin*Xsize*Ysize];
				nout = index[iout + jout*Xsize + kout*Xsize*Ysize];
				if (nin != nout)
				{
					bodyIntegralInner(Xsize, Ysize, Zsize, iin, jin, kin, iout, jout, kout, x, y, z, k1, k2, k3, deltx, delty, deltz, density, DuDt, DvDt, DwDt, p, pn, pcountinner);
				}

			}
		}
	}

}
void devidecountCPU(long Xsize, long Ysize, long Zsize, float* pint, float* pcount)
{
	int n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	for (int tid = 0; tid<n*n; tid++)
	{
		if (pcount[tid]>1)
		{
			pint[tid] /= pcount[tid];
		}

	}
}
void devidecountInnerCPU(long Xsize, long Ysize, long Zsize, float* p, float* pn, float* pcountinner)
{

	for (int tid = 0; tid<Xsize*Ysize*Zsize; tid++)
	{
		if (pcountinner[tid]>1)
		{
			p[tid] = pn[tid] / pcountinner[tid];
			pn[tid] = 0;
		}
	}
}
void calCurlofMaterialAccCPU(long Xsize, long Ysize, long Zsize, float deltx, float delty, float deltz, float* DuDt, float * DvDt, float * DwDt, float * curl,float* mask)
{

	for (int k = 0; k<Zsize; k++)
	{
		for (int j = 0; j<Ysize; j++)
		{
			for (int i = 0; i<Xsize; i++)
			{
				int i0 = i - 1 >= 0 ? i - 1 : i;
				int j0 = j - 1 >= 0 ? j - 1 : j;
				int k0 = k - 1 >= 0 ? k - 1 : k;
				int ie = i + 1 <= Xsize - 1 ? i + 1 : i;
				int je = j + 1 <= Ysize - 1 ? j + 1 : j;
				int ke = k + 1 <= Zsize - 1 ? k + 1 : k;
				float curlx = (DwDt[i + je*Xsize + k*Xsize*Ysize] - DwDt[i + j0*Xsize + k*Xsize*Ysize]) / (je - j0) / delty;
				curlx += -(DvDt[i + j*Xsize + ke*Xsize*Ysize] - DvDt[i + j*Xsize + k0*Xsize*Ysize]) / (ke - k0) / deltz;
				float curly = -(DwDt[ie + j*Xsize + k*Xsize*Ysize] - DwDt[i0 + j*Xsize + k*Xsize*Ysize]) / (ie - i0) / deltx;
				curly += (DuDt[i + j*Xsize + ke*Xsize*Ysize] - DuDt[i + j*Xsize + k0*Xsize*Ysize]) / (ke - k0) / deltz;
				float curlz = (DvDt[ie + j*Xsize + k*Xsize*Ysize] - DvDt[i0 + j*Xsize + k*Xsize*Ysize]) / (ie - i0) / deltx;
				curlz += -(DuDt[i + je*Xsize + k*Xsize*Ysize] - DuDt[i + j0*Xsize + k*Xsize*Ysize]) / (je - j0) / delty;
				curl[i + j*Xsize + k*Xsize*Ysize] = sqrt(curlx*curlx + curly*curly + curlz*curlz)*mask[i + j*Xsize + k*Xsize*Ysize];
				
			}
		}
	}
}

void thredholdHistMaterialAccCPU(int Imax, int Jmax, int Kmax, float* curl, float percentage, float* threshold)
{
	//get min max values for curl;
	float minv = 1e4;
	float maxv = -1e4;
	for (int i = 0; i<Imax*Jmax*Kmax; i++)
	{
		if (minv>curl[i])
		{
			minv = curl[i];
		}
		if (maxv<curl[i])
		{
			maxv = curl[i];
		}
	}
	//generate 1000 bins;
	int * hist; int N = 10000;
	hist = new int[N];
	memset(hist, 0, sizeof(int)*N);
	for (int i = 0; i<Imax*Jmax*Kmax; i++)
	{
		int ind = (int)((curl[i] - minv) / (maxv - minv)*N);
		hist[ind]++;
	}
	float totnum = 0;
	for (int j = N - 1; j >= 0; j++)
	{
		totnum += hist[j];
		if (totnum >= Imax*Jmax*Kmax*percentage)
		{
			threshold[0] = float(j) / N*(maxv - minv) + minv;
			return;
		}
	}
	threshold[0] = maxv;
	delete[] hist;
}

__global__ void calIndexGPU(long*index, long Xsize, long Ysize, long Zsize)
{
	long nout = threadIdx.x + blockIdx.x*blockDim.x;
	long n = Xsize*Ysize * 2 + (Zsize - 2)*Ysize * 2 + (Xsize - 2)*(Zsize - 2) * 2;
	while (nout<n)
	{
		long iout, jout, kout;
		if (nout <= Xsize*Ysize - 1)
		{
			kout = 0; jout = nout / Xsize; iout = nout - Xsize*jout;
		}
		if (nout>Xsize*Ysize - 1 && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1))
		{
			iout = Xsize - 1; jout = (nout - Xsize*Ysize) / (Zsize - 1); kout = nout - Xsize*Ysize - jout*(Zsize - 1) + 1;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize)
		{
			kout = Zsize - 1; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1)) / (Xsize - 1); iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - jout*(Xsize - 1); iout = Xsize - 2 - iout;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize&&nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2))
		{
			jout = 0; kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize) / (Xsize - 1);
			iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - kout*(Xsize - 1);
			iout = Xsize - 2 - iout;
			kout = kout + 1;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) && nout <= Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			iout = 0; jout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2)) / (Zsize - 2);
			kout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - jout*(Zsize - 2);
			kout = Zsize - 2 - kout;
			jout = jout + 1;
		}
		if (nout>Xsize*Ysize - 1 + Ysize*(Zsize - 1) + (Xsize - 1)*Ysize + (Xsize - 1)*(Zsize - 2) + (Ysize - 1)*(Zsize - 2))
		{
			jout = Ysize - 1;
			kout = (nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2)) / (Xsize - 2);
			iout = nout - Xsize*Ysize - Ysize*(Zsize - 1) - (Xsize - 1)*Ysize - (Xsize - 1)*(Zsize - 2) - (Ysize - 1)*(Zsize - 2) - kout*(Xsize - 2);
			kout = Zsize - 2 - kout;
			iout = iout + 1;
		}
		index[iout + jout*Xsize + kout*Xsize*Ysize] = nout;
		nout += blockDim.x*gridDim.x;
	}

}

/////////////////////////////For Experimental Data Version///////////////////////////////
int main()
{
	cudaDeviceProp prop;
	long DeviceNo=0;
	cudaSetDevice(DeviceNo);
	cudaGetDeviceProperties(&prop, DeviceNo) ;
	printf( " ----------- General Information for device %d --------\n", DeviceNo );
	printf( "Name: %s\n", prop.name );
	printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
	printf( "Clock rate: %d\n", prop.clockRate );
	printf( "Device copy overlap: " );
	if (prop.deviceOverlap)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n" );
	printf( "Kernel execition timeout : " );
	if (prop.kernelExecTimeoutEnabled)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n" );
	printf( " ------------ Memory Information for device %d ---------\n", DeviceNo );
	printf( "Total global mem: %ld\n", prop.totalGlobalMem );
	printf( "Total constant Mem: %ld\n", prop.totalConstMem );
	printf( "Max mem pitch: %ld\n", prop.memPitch );
	printf( "Texture Alignment: %ld\n", prop.textureAlignment );
	printf( " --- MP Information for device %d ---\n", DeviceNo );
	printf( "Multiprocessor count: %d\n",
		prop.multiProcessorCount );
	printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
	printf( "Registers per mp: %d\n", prop.regsPerBlock );
	printf( "Threads in warp: %d\n", prop.warpSize );
	printf( "Max threads per block: %d\n",
		prop.maxThreadsPerBlock );
	printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
	printf( "Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
	printf( "\n" );
	ofstream log;
	log.open("log.dat");
	long ImaxOrg,JmaxOrg,KmaxOrg,Imax,Jmax,Kmax,n,PlaneSt,Planedt,PlaneEnd,FileNumSt,FileNumDelt,FileNumEnd;
	float rho,scale,linespacing;
	float density=1;
	float eps=1e-10;
	float meanpcal=0;
	float meanpdns=0;
	float* x,*y,*z,*u,*v,*w,*dudt,*dvdt,*dwdt,*pint,*p,*pn,*pdns,*RHS,*curl,*mask;
	float* k1,*k2,*k3;
	float* dudt_d,*dvdt_d,*dwdt_d,*pint_d,*p_d,*pn_d,*curl_d;
	float* k1_d,*k2_d,*k3_d;
	long *index,*index_d;
	float *pcountinner,*pcountinner_d;
	float* pcount_d,*pcountitr_d,*pcount;
	float* pweight_d;
	float threshold;
	float pref=0;
	int NoAngles=10000;
	int NoItr=100;
	int cutzs,cutze;
	int cutxs,cutxe;
	int cutys,cutye;
	Imax=64;
	Jmax=64;
	Kmax=64;
	float deltx=0.006135923151543;
	float delty=0.006135923151543;
	float deltz=0.006135923151543;
	CString pathpressure,pathacc,fileacc,basefile;
	CString filegrid;
	///////////////////Reading parameters//////////////////////////////
	CStdioFile par;
	CString str;
	if (!par.Open(_T("Parameter_Omni3D.dat"), CFile::modeRead))
	{
		cout << "Parameter input file: \"Parameter_Omni3D.dat\" open error" << endl;
		//MessageBox(NULL, _T("Parameter input file: \"Parameter_Omni3D.dat\" open error"), _T("Omni3D Message"), MB_OK);
		cin >> Imax;
		return;
	}

	par.ReadString(str);ImaxOrg=_wtoi(str);
	par.ReadString(str);JmaxOrg=_wtoi(str);
	par.ReadString(str);KmaxOrg=_wtoi(str);
	par.ReadString(str);deltx=_wtof(str);
	par.ReadString(str);delty=_wtof(str);
	par.ReadString(str);deltz=_wtof(str);
	par.ReadString(str);density=_wtof(str);
	par.ReadString(str);scale=_wtoi(str);
	par.ReadString(str);linespacing=_wtof(str);
	par.ReadString(str);NoAngles=_wtoi(str);
	par.ReadString(filegrid);
    par.ReadString(pathacc);
	par.ReadString(pathpressure);
	par.ReadString(str);NoItr=_wtoi(str);
	par.ReadString(str); threshold = _wtof(str);
	par.ReadString(str); pref = _wtof(str); if (str == ""){ pref = 0; }
	par.ReadString(str); cutxs = _wtoi(str); if (str == ""){ cutxs = 0; }
	par.ReadString(str); cutxe = _wtoi(str); if (str == ""){ cutxe = ImaxOrg-1; }
	par.ReadString(str); cutys = _wtoi(str); if (str == ""){ cutys = 0; }
	par.ReadString(str); cutye = _wtoi(str); if (str == ""){ cutye = JmaxOrg-1; }
	par.ReadString(str); cutzs = _wtoi(str); if (str == ""){ cutzs = 0; }
	par.ReadString(str); cutze = _wtoi(str); if (str == ""){ cutze = KmaxOrg-1; }
	par.Close();
	
	////////////////////////////////Reading parameter completed////////////////////////
	Imax=cutxe-cutxs+1;
	Jmax=cutye-cutys+1;
	Kmax=cutze-cutzs+1;
	x=new float[Imax*Jmax*Kmax];
	y=new float[Imax*Jmax*Kmax];
	z=new float[Imax*Jmax*Kmax];
	u=new float[Imax*Jmax*Kmax];
	v=new float[Imax*Jmax*Kmax];
	w=new float[Imax*Jmax*Kmax];
	
	dudt=new float[Imax*Jmax*Kmax];
	dvdt=new float[Imax*Jmax*Kmax];
	dwdt=new float[Imax*Jmax*Kmax];
	n=Imax*Jmax*2+(Jmax-2)*Kmax*2+(Imax-2)*(Kmax-2)*2;
	p=new float[Imax*Jmax*Kmax];
	pn=new float[Imax*Jmax*Kmax];
	pdns=new float[Imax*Jmax*Kmax];
	RHS=new float[Imax*Jmax*Kmax];
	curl = new float[Imax*Jmax*Kmax];
	mask = new float[Imax*Jmax*Kmax];

	pint=new float[n*n];
	pcountinner=new float[Imax*Jmax*Kmax];
	pcount=new float[n*n];
	index=new long[Imax*Jmax*Kmax];	
	k1=new float[NoAngles];
	k2=new float[NoAngles];
	k3=new float[NoAngles];
	memset(p,0,sizeof(float)*Imax*Jmax*Kmax);
	memset(pn,0,sizeof(float)*Imax*Jmax*Kmax);
	memset(RHS,0,sizeof(float)*Imax*Jmax*Kmax);
	memset(curl, 0, sizeof(float)*Imax*Jmax*Kmax);
	memset(mask, 0, sizeof(float)*Imax*Jmax*Kmax);

	memset(pint,0,sizeof(float)*n*n);
	memset(pcountinner,0,sizeof(float)*Imax*Jmax*Kmax);
	memset(pcount,0,sizeof(float)*n*n);
	//calIndex(index,Imax,Jmax,Kmax);
	CStdioFile fin;		
	////Read Random Numbers;
	cout<<"Reading virtual grid points on the sphere.........."<<endl;
	if (!fin.Open(filegrid, CFile::modeRead))
	{
		cout << "Virtual grid points:" << CT2A(filegrid)<<" open error" << endl;
		//MessageBox(NULL, _T("Parameter input file: \"Parameter_Omni3D.dat\" open error"), _T("Omni3D Message"), MB_OK);
		cin >> Imax;
		return;
	}
	for(int j=0;j<NoAngles;j++)
	{
		long pos;
		fin.ReadString(str);

		pos=str.ReverseFind(' ');
		k3[j]=_wtof(str.Right(str.GetLength()-pos-1));
		for(long m=0;m<1;m++)
		{
			str=str.Left(pos);
			pos=str.ReverseFind(' ');
		}

		k2[j]=_wtof(str.Right(str.GetLength()-pos-1));
		for(long m=0;m<1;m++)
		{
			str=str.Left(pos);
			pos=str.ReverseFind(' ');
		}

		k1[j]=_wtof(str.Right(str.GetLength()-pos-1));
		for(long m=0;m<1;m++)
		{
			str=str.Left(pos);
			pos=str.ReverseFind(' ');
		}
	}
	fin.Close();
	cout << "done" << endl;
	///////////////////////////////////////////////
	cudaMalloc((void **)&dudt_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void **)&dvdt_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void **)&dwdt_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void **)&curl_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void **)&pint_d,sizeof(float)*n*n);
	cudaMalloc((void **)&pcount_d,sizeof(float)*n*n);
	cudaMalloc((void **)&pweight_d,sizeof(float)*n*n);
	cudaMalloc((void **)&p_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void **)&pn_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void **)&index_d,sizeof(long)*Imax*Jmax*Kmax);
	//cudaMalloc((void**)&pcountitr_d,sizeof(int)*n);
	cudaMalloc((void**)&pcountinner_d,sizeof(float)*Imax*Jmax*Kmax);
	cudaMalloc((void**)&k1_d,sizeof(float)*NoAngles);
	cudaMalloc((void**)&k2_d,sizeof(float)*NoAngles);
	cudaMalloc((void**)&k3_d,sizeof(float)*NoAngles);
	
	
	//////////////////////End of allocate memory on GPU//////////////	
	cudaMemcpy(k1_d, k1, sizeof(float)*NoAngles, cudaMemcpyHostToDevice);
	cudaMemcpy(k2_d, k2, sizeof(float)*NoAngles, cudaMemcpyHostToDevice);
	cudaMemcpy(k3_d, k3, sizeof(float)*NoAngles, cudaMemcpyHostToDevice);

	///Read all the files inside a folder////////////////////////////
	//Step.1 judge whether folder exsists///////////////////////////
	if (!folderExists(pathacc))
	{
		cout << "Acceleration folder does not exist: " << CT2A(pathacc)<< endl;
		//MessageBox(NULL, _T("Parameter input file: \"Parameter_Omni3D.dat\" open error"), _T("Omni3D Message"), MB_OK);
		cin >> Imax;
		//MessageBox(NULL, _T("Acceleration folder does not exist"), _T("Omni3D Message"), MB_OK);
		return;
	}
	if (!folderExists(pathpressure))
	{
		createFolder(pathpressure);
	}
	// Step.2 Read all the acceleration filenames////
	vector<string> filesacc;
	vector<string> filesvel;
	string s = CT2A(pathacc);
	getFiles(s, filesacc);
	/////////////////////////////////////////////////////////////////
	int size = filesacc.size();




	cout << "Processing Starts" << endl;
	log << "Processing Starts" << endl;
	for(int FileNum=0;FileNum<size;FileNum++)
	{
		
		fileacc=filesacc[FileNum].c_str();
		int pos = fileacc.ReverseFind('\\');
		cout << CT2A(fileacc.Right(fileacc.GetLength() - pos - 1)) <<endl;
		log  << CT2A(fileacc.Right(fileacc.GetLength() - pos - 1));
		if (!fin.Open(fileacc, CFile::modeRead)){
			log << " failed to open" << endl;
			continue;
		}
		fin.ReadString(str);fin.ReadString(str);fin.ReadString(str);
		
	//fin.ReadString(str);fin.ReadString(str);fin.ReadString(str);
	//With mask files...
	for(long k=0;k<KmaxOrg;k++)
	{
		for(long j=0;j<JmaxOrg;j++)
		{
			for(long i=0;i<ImaxOrg;i++)
			{
				long pos;
				int ind=i-cutxs+(j-cutys)*(cutxe-cutxs+1)+(k-cutzs)*(cutxe-cutxs+1)*(cutye-cutys+1);
				fin.ReadString(str);
				str = str.TrimRight();
				if(i>=cutxs&&i<=cutxe&&j>=cutys&&j<=cutye&&k>=cutzs&&k<=cutze)
				{
					pos = str.ReverseFind(' ');
					mask[ind] = _wtof(str.Right(str.GetLength() - pos - 1));
					for (long m = 0; m<1; m++)
					{
						str = str.Left(pos);
						str = str.TrimRight();
						pos = str.ReverseFind(' ');
					}

					pos=str.ReverseFind(' ');
					dwdt[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					for(long m=0;m<1;m++)
					{
						str=str.Left(pos);
						str = str.TrimRight();
						pos=str.ReverseFind(' ');
					}

					dvdt[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					for(long m=0;m<1;m++)
					{
						str=str.Left(pos);
						str = str.TrimRight();
						pos=str.ReverseFind(' ');
					}

					dudt[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					for(long m=0;m<1;m++)
					{
						str=str.Left(pos);
						str = str.TrimRight();
						pos=str.ReverseFind(' ');
					}

					w[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					str=str.Left(pos);
					pos=str.ReverseFind(' ');
					v[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					str=str.Left(pos);
					pos=str.ReverseFind(' ');
					u[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					str=str.Left(pos);
					pos=str.ReverseFind(' ');
					z[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					for(long m=0;m<1;m++)
					{
						str=str.Left(pos);
						str = str.TrimRight();
						pos=str.ReverseFind(' ');
					}

					y[ind]=_wtof(str.Right(str.GetLength()-pos-1));
					for(long m=0;m<1;m++)
					{
						str=str.Left(pos);
						str = str.TrimRight();
						pos=str.ReverseFind(' ');
					}

					x[ind]=_wtof(str.Right(str.GetLength()-pos-1));
				}
				
			}
		}
	}
	fin.Close();
	cudaMemset(p_d,0,sizeof(float)*Imax*Jmax*Kmax);
	cudaMemset(pn_d,0,sizeof(float)*Imax*Jmax*Kmax);
	cudaMemset(curl_d, 0, sizeof(float)*Imax*Jmax*Kmax);
	cudaMemset(pint_d,0,sizeof(float)*n*n);
	cudaMemset(pcount_d,0,sizeof(float)*n*n);
	cudaMemset(pcountinner_d,0,sizeof(float)*Imax*Jmax*Kmax);
	cudaMemset(pweight_d,0,sizeof(float)*n*n);
	//cudaMemset(pcountitr_d,0,sizeof(int)*n);
	cudaMemcpy(dudt_d,dudt,sizeof(float)*Imax*Jmax*Kmax,cudaMemcpyHostToDevice);
	cudaMemcpy(dvdt_d,dvdt,sizeof(float)*Imax*Jmax*Kmax,cudaMemcpyHostToDevice);
	cudaMemcpy(dwdt_d,dwdt,sizeof(float)*Imax*Jmax*Kmax,cudaMemcpyHostToDevice);
	//set boundary pressure as from Bernoulli equation
	for (int i = 0; i < Imax; i++){
		for (int k = 0; k < Kmax; k++){
			int ind = i + 0*Imax + k*Imax*Jmax;
			p[ind] = pref-0.5*density*(u[ind] * u[ind] + v[ind] * v[ind] + w[ind] * w[ind]);
		}
	}
	cudaMemcpy(p_d,p,sizeof(float)*Imax*Jmax*Kmax,cudaMemcpyHostToDevice);
	cudaMemcpy(pn_d, p, sizeof(float)*Imax*Jmax*Kmax, cudaMemcpyHostToDevice);

	//////////////////////End of allocate memory on GPU//////////////
	dim3 threadPerBlock(8,8);
	dim3 blockPerGrid(512,512);

	dim3 threadPerBlock1(8,8,8);
	dim3 blockPerGrid1(256, 256,256);


	//calCurlofMaterialAcc <<<blockPerGrid1, threadPerBlock1>>>(Imax, Jmax, Kmax, deltx, delty, deltz, dudt_d, dvdt_d, dwdt_d, curl_d);
	calCurlofMaterialAccCPU(Imax, Jmax, Kmax, deltx, delty, deltz, dudt, dvdt, dwdt, curl,mask);
    cudaMemcpy(curl_d, curl, sizeof(float)*Imax*Jmax*Kmax, cudaMemcpyHostToDevice);

	//  thredholdHistMaterialAccCPU(Imax,Jmax,Kmax,curl1,percentage,&threshold);
	////////////////////Start Kernels on GPU////////////////////////////////////////////
	calIndexGPU <<<n/512, 512 >>>(index_d, Imax, Jmax, Kmax);
	//omni3dparallellinesEqualSpacing <<<blockPerGrid, threadPerBlock >>>(Imax, Jmax, Kmax, NoAngles, linespacing, k1_d, k2_d, k3_d, index_d, deltx, delty, deltz, density, dudt_d, dvdt_d, dwdt_d, pint_d, pcount_d, pcountinner_d);

	//omni3dparallellinesEqualSpacingWeighted<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,pint_d,pweight_d,pcount_d,pcountinner_d,curl);
	
	//omni3dparallellinesEqualSpacingSelect<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,pint_d,pcount_d,pcountinner_d);
   // devidecount <<<n/ 512, 512 >>>(Imax, Jmax, Kmax, pint_d, pcount_d);
	//BCiteration <<<n / 512, 512 >>>(Imax, Jmax, Kmax, pint_d, pcount_d, p_d, pn_d, NoItr);
	/////-------------couting time----------------////////////
	//omni3dparallellinesESInner<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,p_d,pn_d,pcountinner_d);
	//omni3dparallellinesESInnerSelect<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,p_d,pn_d,pcountinner_d);
	for (int i = 0; i<NoItr; i++)
	{


		//omni3dparallellinesESInnerWeighted<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,curl_d,p_d,pn_d,pcountinner_d);
		//omni3dparallellinesESInnerStepCount<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,p_d,pn_d,pcountinner_d,IntegrationSteps_d);

		//omni3dparallellinesESInnerWeightedMiniCurl<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,linespacing,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,curl,p_d,pn_d,pcountinner_d);

		omni3dparallellinesESInnerSelect <<<blockPerGrid, threadPerBlock >>>(Imax, Jmax, Kmax, NoAngles, linespacing, k1_d, k2_d, k3_d, index_d, deltx, delty, deltz, density, dudt_d, dvdt_d, dwdt_d, curl_d, p_d, pn_d, pcountinner_d, threshold);
		//omni3dparallellinesESInnerSelectFixedBC << <blockPerGrid, threadPerBlock >> >(Imax, Jmax, Kmax, NoAngles, linespacing, k1_d, k2_d, k3_d, index_d, deltx, delty, deltz, density, dudt_d, dvdt_d, dwdt_d, curl_d, p_d, pn_d, pcountinner_d, threshold);

		//omni2dparallellinesOnFaceInner<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,10000,linespacing,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,p_d,pn_d,pcountinner_d);
		//omni3dparallellinesInner<<<blockPerGrid,threadPerBlock>>>(Imax,Jmax,Kmax,NoAngles,k1_d,k2_d,k3_d,index_d,deltx,delty,deltz,density,dudt_d,dvdt_d,dwdt_d,p_d,pn_d);
		devidecountInner <<<n/512, 512 >>>(Imax, Jmax, Kmax, p_d, pn_d, pcountinner_d);
		if (i == NoItr - 1)
		{
			cudaMemcpy(pcountinner, pcountinner_d, sizeof(float)*Imax*Jmax*Kmax, cudaMemcpyDeviceToHost);
		}
		//cudaMemset(pcountinner_d,1.0,sizeof(float)*Imax*Jmax*Kmax);
		cudaMemset(pcountinner_d, 0, sizeof(int)*Imax*Jmax*Kmax);
	}
	//devidecountInner<<<n/512,512>>>(Imax,Jmax,Kmax,p_d,pn_d,pcountinner_d);
	cudaMemcpy(p, p_d, sizeof(float)*Imax*Jmax*Kmax, cudaMemcpyDeviceToHost);
	//cudaMemcpy(pcountinner, pcountinner_d, sizeof(float)*Imax*Jmax*Kmax, cudaMemcpyDeviceToHost);
	//cudaMemcpy(curl1,curl,sizeof(float)*Imax*Jmax*Kmax,cudaMemcpyDeviceToHost);
	
	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
	CStdioFile fout;	
	CString outfile=pathpressure;
	outfile.AppendFormat(_T("PressureOmni3D_%05d.dat"),FileNum);
	fout.Open(outfile,CFile::modeWrite|CFile::modeCreate);
	////////////////////Write Data to file/////////////////////
	/*
	meanpcal=0;
	int count = 0;
	for(long k=0;k<Kmax;k++)
	{
		for(long j=0;j<Jmax;j++)
		{
			for(long i=0;i<Imax;i++)
			{	
				int ind=i+j*Imax+k*Imax*Jmax;
				if (curl[ind] != 0)
				{
					meanpcal += p[ind];
					count++;
				}
						
			}
		}
	}
	//meanpcal=meanpcal/n;
	meanpcal=meanpcal/count;
	for(long k=0;k<Kmax;k++)
	{
		for(long j=0;j<Jmax;j++)
		{
			for(long i=0;i<Imax;i++)
			{
				int ind=i+j*Imax+k*Imax*Jmax;
				if (curl[ind] != 0)
				{
					p[ind]=p[ind]-meanpcal;
				}
				
			}
		}
	}
	*/
	fout.WriteString(_T("TITLE = \"Pressure Integrated From GPU Based Omni 3D Method\"\n"));
	fout.WriteString(_T("VARIABLES = \"X\",\"Y\",\"Z\",\"P\",\"Count\"\n"));
	str.Format(_T("ZONE I=%i, J=%i, K=%i,F=POINT\n"),Imax,Jmax,Kmax);
	fout.WriteString(str);
	//pmax=1;meanpdns=0;
	
	for(long k=0;k<Kmax;k++)
	{
		for(long j=0;j<Jmax;j++)
		{
			for(long i=0;i<Imax;i++)
			{
				int ind=i+j*Imax+k*Imax*Jmax;
				str.Format(_T("%15.9f %15.9f %15.9f %15.9f %15.9f\n"), x[ind], y[ind], z[ind], p[ind], pcountinner[ind]);
				fout.WriteString(str);
			}
		}
	}

	fout.Close();
	
	/////////////Iteration completed/////////////////////////////////////////////
	if (FileNum == 0)
	{
		fout.Open(_T("CurlofMaterialAcc_Sample.dat"), CFile::modeWrite | CFile::modeCreate);
		fout.WriteString(_T("TITLE = \"Curl of Material Acceleration Multiplied by the Mask\"\n"));
		fout.WriteString(_T("VARIABLES = \"X\",\"Y\",\"Z\",\"Curl of Acceleration\"\n"));
		str.Format(_T("ZONE I=%i, J=%i, K=%i,F=POINT\n"), Imax, Jmax, Kmax);
		fout.WriteString(str);
		//pmax=1;meanpdns=0;

		for (long k = 0; k<Kmax; k++)
		{
			for (long j = 0; j<Jmax; j++)
			{
				for (long i = 0; i<Imax; i++)
				{
					int ind = i + j*Imax + k*Imax*Jmax;
					str.Format(_T("%15.9f %15.9f %15.9f %15.9f\n"), x[ind], y[ind], z[ind], curl[ind]);
					fout.WriteString(str);
				}
			}
		}

		fout.Close();
	}
	
	
	}
	delete []x,y,z,u,v,w,dudt,dvdt,dwdt,pint,p,pn,pdns,RHS,pcount,pcountinner,k1,k2,k3,curl,mask;
	log.close();
	cudaFree(dudt_d);
	cudaFree(dvdt_d);
	cudaFree(dwdt_d);
	cudaFree(pint_d);
	cudaFree(pcount_d);
	cudaFree(p_d);
	cudaFree(pn_d);
	cudaFree(pcountinner_d);
	cudaFree(k1_d);
	cudaFree(k2_d);
	cudaFree(k3_d);
	cudaFree(pweight_d);
	cudaFree(curl_d);
	cudaDeviceReset();
	exit(true ? EXIT_SUCCESS : EXIT_FAILURE);
	return 0;

}