// ColorArray.h: interface for the CColorArray class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_COLORARRAY_H__D24209DE_3C87_4F0E_91C1_14CE5C416E2B__INCLUDED_)
#define AFX_COLORARRAY_H__D24209DE_3C87_4F0E_91C1_14CE5C416E2B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CColorArray
{
public:
	void Free();
	void MemoryAllocation();
	void SetSize(int Height, int Width);

	double **m_R;
	double **m_B;
	double **m_G;

	int m_Height;
	int m_Width;

	CColorArray();
	virtual ~CColorArray();
};

#endif // !defined(AFX_COLORARRAY_H__D24209DE_3C87_4F0E_91C1_14CE5C416E2B__INCLUDED_)