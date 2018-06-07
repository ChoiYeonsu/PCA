// ColorArray.cpp: implementation of the CColorArray class.
//
//////////////////////////////////////////////////////////////////////

#include "ColorArray.h"
#include "stdio.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CColorArray::CColorArray()
{
	m_R = NULL;
	m_G = NULL;
	m_B = NULL;
}

CColorArray::~CColorArray()
{
	Free();
}

void CColorArray::SetSize(int Height, int Width)
{
	m_Height = Height;
	m_Width = Width;
}

void CColorArray::MemoryAllocation()
{
	int i, j;

	m_R = new double *[m_Height];
	m_G = new double *[m_Height];
	m_B = new double *[m_Height];

	for (i = 0; i < m_Height; i++)
	{
		m_R[i] = new double[m_Width];
		m_G[i] = new double[m_Width];
		m_B[i] = new double[m_Width];
	}

	for (i = 0; i < m_Height; i++)
	{
		for (j = 0; j < m_Width; j++)
		{
			m_R[i][j] = 0; m_G[i][j] = 0; m_B[i][j] = 0;
		}
	}


}

void CColorArray::Free()
{
	int i;

	for (i = 0; i < m_Height; i++)
	{
		delete m_R[i];
		delete m_G[i];
		delete m_B[i];
	}

	delete m_R;
	delete m_G;
	delete m_B;
}