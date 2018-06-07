#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ColorArray.h"
#include "Eigen/Dense"
#include <iostream>
#include "pca.h"


using Eigen::MatrixXd;
using namespace std;

int M = 25;
int um = 100;
int ustd = 80;

// 영상 크기 지정
int PADDING_IMAGE_HEIGHT = 112;
int PADDING_IMAGE_WIDTH = 92;

// 원본 파일 이름
char ORIGINAL_FILE1[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\1.raw";
char RESULT_FILE1[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\1.raw";
char ORIGINAL_FILE2[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\2.raw"; 
char RESULT_FILE2[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\2.raw";
char ORIGINAL_FILE3[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\3.raw";
char RESULT_FILE3[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\3.raw";
char ORIGINAL_FILE4[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\4.raw";
char RESULT_FILE4[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\4.raw";
char ORIGINAL_FILE5[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\5.raw";
char RESULT_FILE5[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\5.raw";
char ORIGINAL_FILE6[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\6.raw";
char RESULT_FILE6[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\6.raw";
char ORIGINAL_FILE7[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\7.raw";
char RESULT_FILE7[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\7.raw";
char ORIGINAL_FILE8[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\8.raw";
char RESULT_FILE8[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\8.raw";
char ORIGINAL_FILE9[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\9.raw";
char RESULT_FILE9[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\9.raw";
char ORIGINAL_FILE10[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\10.raw";
char RESULT_FILE10[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\10.raw";
char ORIGINAL_FILE11[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\11.raw";
char RESULT_FILE11[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\11.raw";
char ORIGINAL_FILE12[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\12.raw";
char RESULT_FILE12[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\12.raw";
char ORIGINAL_FILE13[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\13.raw";
char RESULT_FILE13[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\13.raw";
char ORIGINAL_FILE14[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\14.raw";
char RESULT_FILE14[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\14.raw";
char ORIGINAL_FILE15[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\15.raw";
char RESULT_FILE15[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\15.raw";
char ORIGINAL_FILE16[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\16.raw";
char RESULT_FILE16[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\16.raw";
char ORIGINAL_FILE17[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\17.raw";
char RESULT_FILE17[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\17.raw";
char ORIGINAL_FILE18[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\18.raw";
char RESULT_FILE18[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\18.raw";
char ORIGINAL_FILE19[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\19.raw";
char RESULT_FILE19[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\19.raw";
char ORIGINAL_FILE20[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\20.raw";
char RESULT_FILE20[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\20.raw";
char ORIGINAL_FILE21[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\21.raw";
char RESULT_FILE21[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\21.raw";
char ORIGINAL_FILE22[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\22.raw";
char RESULT_FILE22[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\22.raw";
char ORIGINAL_FILE23[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\23.raw";
char RESULT_FILE23[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\23.raw";
char ORIGINAL_FILE24[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\24.raw";
char RESULT_FILE24[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\24.raw";
char ORIGINAL_FILE25[50] = "C:\\Users\\T1\\Desktop\\raw(얼굴)\\25.raw";
char RESULT_FILE25[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\25.raw";
char MEAN_FILE1[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\mean1.raw";
char EIGEN_FILE1[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig1.raw";
char EIGEN_FILE2[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig2.raw";
char EIGEN_FILE3[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig3.raw";
char EIGEN_FILE4[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig4.raw";
char EIGEN_FILE5[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig5.raw";
char EIGEN_FILE6[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig6.raw";
char EIGEN_FILE7[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig7.raw";
char EIGEN_FILE8[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig8.raw";
char EIGEN_FILE9[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig9.raw";
char EIGEN_FILE10[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig10.raw";
char EIGEN_FILE11[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig11.raw";
char EIGEN_FILE12[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig12.raw";
char EIGEN_FILE13[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig13.raw";
char EIGEN_FILE14[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig14.raw";
char EIGEN_FILE15[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig15.raw";
char EIGEN_FILE16[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig16.raw";
char EIGEN_FILE17[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig17.raw";
char EIGEN_FILE18[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig18.raw";
char EIGEN_FILE19[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig19.raw";
char EIGEN_FILE20[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig20.raw";
char EIGEN_FILE21[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig21.raw";
char EIGEN_FILE22[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig22.raw";
char EIGEN_FILE23[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig23.raw";
char EIGEN_FILE24[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig24.raw";
char EIGEN_FILE25[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\eig25.raw";
char NORM_FILE1[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm1.raw";
char NORM_FILE2[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm2.raw";
char NORM_FILE3[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm3.raw";
char NORM_FILE4[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm4.raw";
char NORM_FILE5[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm5.raw";
char NORM_FILE6[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm6.raw";
char NORM_FILE7[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm7.raw";
char NORM_FILE8[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm8.raw";
char NORM_FILE9[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm9.raw";
char NORM_FILE10[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm10.raw";
char NORM_FILE11[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm11.raw";
char NORM_FILE12[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm12.raw";
char NORM_FILE13[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm13.raw";
char NORM_FILE14[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm14.raw";
char NORM_FILE15[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm15.raw";
char NORM_FILE16[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm16.raw";
char NORM_FILE17[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm17.raw";
char NORM_FILE18[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm18.raw";
char NORM_FILE19[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm19.raw";
char NORM_FILE20[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm20.raw";
char NORM_FILE21[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm21.raw";
char NORM_FILE22[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm22.raw";
char NORM_FILE23[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm23.raw";
char NORM_FILE24[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm24.raw";
char NORM_FILE25[50] = "C:\\Users\\T1\\Desktop\\raw(결과)\\norm25.raw";

int main()
{
	int rows = 25;
	int columns = 10304;
	int u, i, j, k;
	double sum_R = 0;
	double sum_G = 0;
	double sum_B = 0;
	int sum;

	FILE *fp1 = NULL;
	FILE *fp2 = NULL;
	FILE *fp3 = NULL;
	FILE *fp4 = NULL;
	FILE *fp5 = NULL;
	FILE *fp6 = NULL;
	FILE *fp7 = NULL;
	FILE *fp8 = NULL;
	FILE *fp9 = NULL;
	FILE *fp10 = NULL;
	FILE *fp11 = NULL;
	FILE *fp12 = NULL;
	FILE *fp13 = NULL;
	FILE *fp14 = NULL;
	FILE *fp15 = NULL;
	FILE *fp16 = NULL;
	FILE *fp17 = NULL;
	FILE *fp18 = NULL;
	FILE *fp19 = NULL;
	FILE *fp20 = NULL;
	FILE *fp21 = NULL;
	FILE *fp22 = NULL;
	FILE *fp23 = NULL;
	FILE *fp24 = NULL;
	FILE *fp25 = NULL;
	FILE *m1 = NULL;
	FILE *eigen1 = NULL;
	FILE *eigen2 = NULL;
	FILE *eigen3 = NULL;
	FILE *eigen4 = NULL;
	FILE *eigen5 = NULL;
	FILE *eigen6 = NULL;
	FILE *eigen7 = NULL;
	FILE *eigen8 = NULL;
	FILE *eigen9 = NULL;
	FILE *eigen10 = NULL;
	FILE *eigen11 = NULL;
	FILE *eigen12 = NULL;
	FILE *eigen13 = NULL;
	FILE *eigen14 = NULL;
	FILE *eigen15 = NULL;
	FILE *eigen16 = NULL;
	FILE *eigen17 = NULL;
	FILE *eigen18 = NULL;
	FILE *eigen19 = NULL;
	FILE *eigen20 = NULL;
	FILE *eigen21 = NULL;
	FILE *eigen22 = NULL;
	FILE *eigen23 = NULL;
	FILE *eigen24 = NULL;
	FILE *eigen25 = NULL;
	FILE *norm1 = NULL;
	FILE *norm2 = NULL;
	FILE *norm3 = NULL;
	FILE *norm4 = NULL;
	FILE *norm5 = NULL;
	FILE *norm6 = NULL;
	FILE *norm7 = NULL;
	FILE *norm8 = NULL;
	FILE *norm9 = NULL;
	FILE *norm10 = NULL;
	FILE *norm11 = NULL;
	FILE *norm12 = NULL;
	FILE *norm13 = NULL;
	FILE *norm14 = NULL;
	FILE *norm15 = NULL;
	FILE *norm16 = NULL;
	FILE *norm17 = NULL;
	FILE *norm18 = NULL;
	FILE *norm19 = NULL;
	FILE *norm20 = NULL;
	FILE *norm21 = NULL;
	FILE *norm22 = NULL;
	FILE *norm23 = NULL;
	FILE *norm24 = NULL;
	FILE *norm25 = NULL;

	CColorArray *CArrayOriginalImage1 = new CColorArray;
	CArrayOriginalImage1->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage1->MemoryAllocation();

	CColorArray *CArrayOriginalImage2 = new CColorArray;
	CArrayOriginalImage2->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage2->MemoryAllocation();

	CColorArray *CArrayOriginalImage3 = new CColorArray;
	CArrayOriginalImage3->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage3->MemoryAllocation();

	CColorArray *CArrayOriginalImage4 = new CColorArray;
	CArrayOriginalImage4->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage4->MemoryAllocation();

	CColorArray *CArrayOriginalImage5 = new CColorArray;
	CArrayOriginalImage5->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage5->MemoryAllocation();

	CColorArray *CArrayOriginalImage6 = new CColorArray;
	CArrayOriginalImage6->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage6->MemoryAllocation();

	CColorArray *CArrayOriginalImage7 = new CColorArray;
	CArrayOriginalImage7->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage7->MemoryAllocation();

	CColorArray *CArrayOriginalImage8 = new CColorArray;
	CArrayOriginalImage8->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage8->MemoryAllocation();

	CColorArray *CArrayOriginalImage9 = new CColorArray;
	CArrayOriginalImage9->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage9->MemoryAllocation();

	CColorArray *CArrayOriginalImage10 = new CColorArray;
	CArrayOriginalImage10->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage10->MemoryAllocation();

	CColorArray *CArrayOriginalImage11 = new CColorArray;
	CArrayOriginalImage11->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage11->MemoryAllocation();

	CColorArray *CArrayOriginalImage12 = new CColorArray;
	CArrayOriginalImage12->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage12->MemoryAllocation();

	CColorArray *CArrayOriginalImage13 = new CColorArray;
	CArrayOriginalImage13->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage13->MemoryAllocation();

	CColorArray *CArrayOriginalImage14 = new CColorArray;
	CArrayOriginalImage14->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage14->MemoryAllocation();

	CColorArray *CArrayOriginalImage15 = new CColorArray;
	CArrayOriginalImage15->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage15->MemoryAllocation();

	CColorArray *CArrayOriginalImage16 = new CColorArray;
	CArrayOriginalImage16->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage16->MemoryAllocation();

	CColorArray *CArrayOriginalImage17 = new CColorArray;
	CArrayOriginalImage17->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage17->MemoryAllocation();

	CColorArray *CArrayOriginalImage18 = new CColorArray;
	CArrayOriginalImage18->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage18->MemoryAllocation();

	CColorArray *CArrayOriginalImage19 = new CColorArray;
	CArrayOriginalImage19->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage19->MemoryAllocation();

	CColorArray *CArrayOriginalImage20 = new CColorArray;
	CArrayOriginalImage20->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage20->MemoryAllocation();

	CColorArray *CArrayOriginalImage21 = new CColorArray;
	CArrayOriginalImage21->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage21->MemoryAllocation();

	CColorArray *CArrayOriginalImage22 = new CColorArray;
	CArrayOriginalImage22->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage22->MemoryAllocation();

	CColorArray *CArrayOriginalImage23 = new CColorArray;
	CArrayOriginalImage23->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage23->MemoryAllocation();

	CColorArray *CArrayOriginalImage24 = new CColorArray;
	CArrayOriginalImage24->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage24->MemoryAllocation();

	CColorArray *CArrayOriginalImage25 = new CColorArray;
	CArrayOriginalImage25->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	CArrayOriginalImage25->MemoryAllocation();

	CColorArray *A1 = new CColorArray;
	A1->SetSize(10304, 25);
	A1->MemoryAllocation();

	CColorArray *img = new CColorArray;
	img->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	img->MemoryAllocation();

	CColorArray *img1 = new CColorArray;
	img1->SetSize(10304, 1);
	img1->MemoryAllocation();

	CColorArray *img2 = new CColorArray;
	img2->SetSize(10304, 25);
	img2->MemoryAllocation();

	CColorArray *A = new CColorArray;
	A->SetSize(25, 10304);
	A->MemoryAllocation();

	CColorArray *u1 = new CColorArray;
	u1->SetSize(10304, 25);
	u1->MemoryAllocation();

	CColorArray *u2 = new CColorArray;
	u2->SetSize(25, 10304);
	u2->MemoryAllocation();

	CColorArray *B1 = new CColorArray;
	B1->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B1->MemoryAllocation();

	CColorArray *B2 = new CColorArray;
	B2->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B2->MemoryAllocation();

	CColorArray *B3 = new CColorArray;
	B3->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B3->MemoryAllocation();

	CColorArray *B4 = new CColorArray;
	B4->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B4->MemoryAllocation();

	CColorArray *B5 = new CColorArray;
	B5->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B5->MemoryAllocation();

	CColorArray *B6 = new CColorArray;
	B6->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B6->MemoryAllocation();

	CColorArray *B7 = new CColorArray;
	B7->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B7->MemoryAllocation();

	CColorArray *B8 = new CColorArray;
	B8->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B8->MemoryAllocation();

	CColorArray *B9 = new CColorArray;
	B9->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B9->MemoryAllocation();

	CColorArray *B10 = new CColorArray;
	B10->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B10->MemoryAllocation();

	CColorArray *B11 = new CColorArray;
	B11->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B11->MemoryAllocation();

	CColorArray *B12 = new CColorArray;
	B12->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B12->MemoryAllocation();

	CColorArray *B13 = new CColorArray;
	B13->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B13->MemoryAllocation();

	CColorArray *B14 = new CColorArray;
	B14->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B14->MemoryAllocation();

	CColorArray *B15 = new CColorArray;
	B15->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B15->MemoryAllocation();

	CColorArray *B16 = new CColorArray;
	B16->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B16->MemoryAllocation();

	CColorArray *B17 = new CColorArray;
	B17->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B17->MemoryAllocation();

	CColorArray *B18 = new CColorArray;
	B18->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B18->MemoryAllocation();

	CColorArray *B19 = new CColorArray;
	B19->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B19->MemoryAllocation();

	CColorArray *B20 = new CColorArray;
	B20->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B20->MemoryAllocation();

	CColorArray *B21 = new CColorArray;
	B21->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B21->MemoryAllocation();

	CColorArray *B22 = new CColorArray;
	B22->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B22->MemoryAllocation();

	CColorArray *B23 = new CColorArray;
	B23->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B23->MemoryAllocation();

	CColorArray *B24 = new CColorArray;
	B24->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B24->MemoryAllocation();

	CColorArray *B25 = new CColorArray;
	B25->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	B25->MemoryAllocation();

	CColorArray *C1 = new CColorArray;
	C1->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C1->MemoryAllocation();

	CColorArray *C2 = new CColorArray;
	C2->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C2->MemoryAllocation();

	CColorArray *C3 = new CColorArray;
	C3->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C3->MemoryAllocation();

	CColorArray *C4 = new CColorArray;
	C4->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C4->MemoryAllocation();

	CColorArray *C5 = new CColorArray;
	C5->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C5->MemoryAllocation();

	CColorArray *C6 = new CColorArray;
	C6->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C6->MemoryAllocation();

	CColorArray *C7 = new CColorArray;
	C7->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C7->MemoryAllocation();

	CColorArray *C8 = new CColorArray;
	C8->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C8->MemoryAllocation();

	CColorArray *C9 = new CColorArray;
	C9->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C9->MemoryAllocation();

	CColorArray *C10 = new CColorArray;
	C10->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C10->MemoryAllocation();

	CColorArray *C11 = new CColorArray;
	C11->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C11->MemoryAllocation();

	CColorArray *C12 = new CColorArray;
	C12->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C12->MemoryAllocation();

	CColorArray *C13 = new CColorArray;
	C13->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C13->MemoryAllocation();

	CColorArray *C14 = new CColorArray;
	C14->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C14->MemoryAllocation();

	CColorArray *C15 = new CColorArray;
	C15->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C15->MemoryAllocation();

	CColorArray *C16 = new CColorArray;
	C16->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C16->MemoryAllocation();

	CColorArray *C17 = new CColorArray;
	C17->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C17->MemoryAllocation();

	CColorArray *C18 = new CColorArray;
	C18->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C18->MemoryAllocation();

	CColorArray *C19 = new CColorArray;
	C19->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C19->MemoryAllocation();

	CColorArray *C20 = new CColorArray;
	C20->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C20->MemoryAllocation();

	CColorArray *C21 = new CColorArray;
	C21->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C21->MemoryAllocation();
	
	CColorArray *C22 = new CColorArray;
	C22->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C22->MemoryAllocation();

	CColorArray *C23 = new CColorArray;
	C23->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C23->MemoryAllocation();

	CColorArray *C24 = new CColorArray;
	C24->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C24->MemoryAllocation();

	CColorArray *C25 = new CColorArray;
	C25->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	C25->MemoryAllocation();

	CColorArray *D1 = new CColorArray;
	D1->SetSize(10304, 25);
	D1->MemoryAllocation();

	CColorArray *E1 = new CColorArray;
	E1->SetSize(columns, rows);
	E1->MemoryAllocation();

	CColorArray *F1 = new CColorArray;
	F1->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F1->MemoryAllocation();

	CColorArray *F2 = new CColorArray;
	F2->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F2->MemoryAllocation();

	CColorArray *F3 = new CColorArray;
	F3->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F3->MemoryAllocation();

	CColorArray *F4 = new CColorArray;
	F4->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F4->MemoryAllocation();

	CColorArray *F5 = new CColorArray;
	F5->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F5->MemoryAllocation();

	CColorArray *F6 = new CColorArray;
	F6->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F6->MemoryAllocation();

	CColorArray *F7 = new CColorArray;
	F7->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F7->MemoryAllocation();

	CColorArray *F8 = new CColorArray;
	F8->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F8->MemoryAllocation();

	CColorArray *F9 = new CColorArray;
	F9->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F9->MemoryAllocation();

	CColorArray *F10 = new CColorArray;
	F10->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F10->MemoryAllocation();

	CColorArray *F11 = new CColorArray;
	F11->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F11->MemoryAllocation();

	CColorArray *F12 = new CColorArray;
	F12->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F12->MemoryAllocation();

	CColorArray *F13 = new CColorArray;
	F13->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F13->MemoryAllocation();

	CColorArray *F14 = new CColorArray;
	F14->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F14->MemoryAllocation();

	CColorArray *F15 = new CColorArray;
	F15->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F15->MemoryAllocation();

	CColorArray *F16 = new CColorArray;
	F16->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F16->MemoryAllocation();

	CColorArray *F17 = new CColorArray;
	F17->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F17->MemoryAllocation();

	CColorArray *F18 = new CColorArray;
	F18->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F18->MemoryAllocation();

	CColorArray *F19 = new CColorArray;
	F19->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F19->MemoryAllocation();

	CColorArray *F20 = new CColorArray;
	F20->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F20->MemoryAllocation();

	CColorArray *F21 = new CColorArray;
	F21->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F21->MemoryAllocation();

	CColorArray *F22 = new CColorArray;
	F22->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F22->MemoryAllocation();

	CColorArray *F23 = new CColorArray;
	F23->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F23->MemoryAllocation();

	CColorArray *F24 = new CColorArray;
	F24->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F24->MemoryAllocation();

	CColorArray *F25 = new CColorArray;
	F25->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	F25->MemoryAllocation();

	CColorArray *G1 = new CColorArray;
	G1->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G1->MemoryAllocation();

	CColorArray *G2 = new CColorArray;
	G2->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G2->MemoryAllocation();

	CColorArray *G3 = new CColorArray;
	G3->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G3->MemoryAllocation();

	CColorArray *G4 = new CColorArray;
	G4->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G4->MemoryAllocation();

	CColorArray *G5 = new CColorArray;
	G5->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G5->MemoryAllocation();

	CColorArray *G6 = new CColorArray;
	G6->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G6->MemoryAllocation();

	CColorArray *G7 = new CColorArray;
	G7->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G7->MemoryAllocation();

	CColorArray *G8 = new CColorArray;
	G8->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G8->MemoryAllocation();

	CColorArray *G9 = new CColorArray;
	G9->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G9->MemoryAllocation();

	CColorArray *G10 = new CColorArray;
	G10->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G10->MemoryAllocation();

	CColorArray *G11 = new CColorArray;
	G11->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G11->MemoryAllocation();

	CColorArray *G12 = new CColorArray;
	G12->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G12->MemoryAllocation();

	CColorArray *G13 = new CColorArray;
	G13->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G13->MemoryAllocation();

	CColorArray *G14 = new CColorArray;
	G14->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G14->MemoryAllocation();

	CColorArray *G15 = new CColorArray;
	G15->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G15->MemoryAllocation();

	CColorArray *G16 = new CColorArray;
	G16->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G16->MemoryAllocation();

	CColorArray *G17 = new CColorArray;
	G17->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G17->MemoryAllocation();

	CColorArray *G18 = new CColorArray;
	G18->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G18->MemoryAllocation();

	CColorArray *G19 = new CColorArray;
	G19->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G19->MemoryAllocation();

	CColorArray *G20 = new CColorArray;
	G20->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G20->MemoryAllocation();

	CColorArray *G21 = new CColorArray;
	G21->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G21->MemoryAllocation();

	CColorArray *G22 = new CColorArray;
	G22->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G22->MemoryAllocation();

	CColorArray *G23 = new CColorArray;
	G23->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G23->MemoryAllocation();

	CColorArray *G24 = new CColorArray;
	G24->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G24->MemoryAllocation();

	CColorArray *G25 = new CColorArray;
	G25->SetSize(PADDING_IMAGE_HEIGHT, PADDING_IMAGE_WIDTH);
	G25->MemoryAllocation();


	CColorArray *p1 = new CColorArray;
	p1->SetSize(rows, rows);
	p1->MemoryAllocation();

	unsigned char file_temp1;
	unsigned char file_temp2;
	unsigned char file_temp3;
	unsigned char file_temp4;
	unsigned char file_temp5;
	unsigned char file_temp6;
	unsigned char file_temp7;
	unsigned char file_temp8;
	unsigned char file_temp9;
	unsigned char file_temp10;
	unsigned char file_temp11;
	unsigned char file_temp12;
	unsigned char file_temp13;
	unsigned char file_temp14;
	unsigned char file_temp15;
	unsigned char file_temp16;
	unsigned char file_temp17;
	unsigned char file_temp18;
	unsigned char file_temp19;
	unsigned char file_temp20;
	unsigned char file_temp21;
	unsigned char file_temp22;
	unsigned char file_temp23;
	unsigned char file_temp24;
	unsigned char file_temp25;

	// Raw 파일 읽기
	fp1 = fopen(ORIGINAL_FILE1, "rb");
	if (fp1 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage1->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage1->m_Width; j++)
		{
			fscanf(fp1, "%c", &file_temp1);
			CArrayOriginalImage1->m_R[i][j] = (double)file_temp1;

			fscanf(fp1, "%c", &file_temp1);
			CArrayOriginalImage1->m_G[i][j] = (double)file_temp1;

			fscanf(fp1, "%c", &file_temp1);
			CArrayOriginalImage1->m_B[i][j] = (double)file_temp1;
		}
	}

	fclose(fp1);

	// 파일 저장하기 

	fp1 = fopen(RESULT_FILE1, "wb");

	for (i = 0; i < CArrayOriginalImage1->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage1->m_Width; j++)
		{
			fprintf(fp1, "%c", (unsigned char)CArrayOriginalImage1->m_R[i][j]);
			fprintf(fp1, "%c", (unsigned char)CArrayOriginalImage1->m_G[i][j]);
			fprintf(fp1, "%c", (unsigned char)CArrayOriginalImage1->m_B[i][j]);
		}
	}

	fclose(fp1);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp2 = fopen(ORIGINAL_FILE2, "rb");

	if (fp2 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage2->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage2->m_Width; j++)
		{
			fscanf(fp2, "%c", &file_temp2);
			CArrayOriginalImage2->m_R[i][j] = (double)file_temp2;

			fscanf(fp2, "%c", &file_temp2);
			CArrayOriginalImage2->m_G[i][j] = (double)file_temp2;

			fscanf(fp2, "%c", &file_temp2);
			CArrayOriginalImage2->m_B[i][j] = (double)file_temp2;
		}
	}

	fclose(fp2);

	// 파일 저장하기 
	fp2 = fopen(RESULT_FILE2, "wb");

	for (i = 0; i < CArrayOriginalImage2->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage2->m_Width; j++)
		{
			fprintf(fp2, "%c", (unsigned char)CArrayOriginalImage2->m_R[i][j]);
			fprintf(fp2, "%c", (unsigned char)CArrayOriginalImage2->m_G[i][j]);
			fprintf(fp2, "%c", (unsigned char)CArrayOriginalImage2->m_B[i][j]);
		}
	}

	fclose(fp2);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp3 = fopen(ORIGINAL_FILE3, "rb");

	if (fp3 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage3->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage3->m_Width; j++)
		{
			fscanf(fp3, "%c", &file_temp3);
			CArrayOriginalImage3->m_R[i][j] = (double)file_temp3;

			fscanf(fp3, "%c", &file_temp3);
			CArrayOriginalImage3->m_G[i][j] = (double)file_temp3;

			fscanf(fp3, "%c", &file_temp3);
			CArrayOriginalImage3->m_B[i][j] = (double)file_temp3;
		}
	}

	fclose(fp3);

	// 파일 저장하기 
	fp3 = fopen(RESULT_FILE3, "wb");

	for (i = 0; i < CArrayOriginalImage3->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage3->m_Width; j++)
		{
			fprintf(fp3, "%c", (unsigned char)CArrayOriginalImage3->m_R[i][j]);
			fprintf(fp3, "%c", (unsigned char)CArrayOriginalImage3->m_G[i][j]);
			fprintf(fp3, "%c", (unsigned char)CArrayOriginalImage3->m_B[i][j]);
		}
	}

	fclose(fp3);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp4 = fopen(ORIGINAL_FILE4, "rb");

	if (fp4 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage4->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage4->m_Width; j++)
		{
			fscanf(fp4, "%c", &file_temp4);
			CArrayOriginalImage4->m_R[i][j] = (double)file_temp4;

			fscanf(fp4, "%c", &file_temp4);
			CArrayOriginalImage4->m_G[i][j] = (double)file_temp4;

			fscanf(fp4, "%c", &file_temp4);
			CArrayOriginalImage4->m_B[i][j] = (double)file_temp4;
		}
	}

	fclose(fp4);

	// 파일 저장하기 
	fp4 = fopen(RESULT_FILE4, "wb");

	for (i = 0; i < CArrayOriginalImage4->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage4->m_Width; j++)
		{
			fprintf(fp4, "%c", (unsigned char)CArrayOriginalImage4->m_R[i][j]);
			fprintf(fp4, "%c", (unsigned char)CArrayOriginalImage4->m_G[i][j]);
			fprintf(fp4, "%c", (unsigned char)CArrayOriginalImage4->m_B[i][j]);
		}
	}

	fclose(fp4);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp5 = fopen(ORIGINAL_FILE5, "rb");

	if (fp5 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage5->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage5->m_Width; j++)
		{
			fscanf(fp5, "%c", &file_temp5);
			CArrayOriginalImage5->m_R[i][j] = (double)file_temp5;

			fscanf(fp5, "%c", &file_temp5);
			CArrayOriginalImage5->m_G[i][j] = (double)file_temp5;

			fscanf(fp5, "%c", &file_temp5);
			CArrayOriginalImage5->m_B[i][j] = (double)file_temp5;
		}
	}

	fclose(fp5);

	// 파일 저장하기 
	fp5 = fopen(RESULT_FILE5, "wb");

	for (i = 0; i < CArrayOriginalImage5->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage5->m_Width; j++)
		{
			fprintf(fp5, "%c", (unsigned char)CArrayOriginalImage5->m_R[i][j]);
			fprintf(fp5, "%c", (unsigned char)CArrayOriginalImage5->m_G[i][j]);
			fprintf(fp5, "%c", (unsigned char)CArrayOriginalImage5->m_B[i][j]);
		}
	}

	fclose(fp5);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp6 = fopen(ORIGINAL_FILE6, "rb");

	if (fp6 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage6->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage6->m_Width; j++)
		{
			fscanf(fp6, "%c", &file_temp6);
			CArrayOriginalImage6->m_R[i][j] = (double)file_temp6;

			fscanf(fp6, "%c", &file_temp6);
			CArrayOriginalImage6->m_G[i][j] = (double)file_temp6;

			fscanf(fp6, "%c", &file_temp6);
			CArrayOriginalImage6->m_B[i][j] = (double)file_temp6;
		}
	}

	fclose(fp6);

	// 파일 저장하기 
	fp6 = fopen(RESULT_FILE6, "wb");

	for (i = 0; i < CArrayOriginalImage6->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage6->m_Width; j++)
		{
			fprintf(fp6, "%c", (unsigned char)CArrayOriginalImage6->m_R[i][j]);
			fprintf(fp6, "%c", (unsigned char)CArrayOriginalImage6->m_G[i][j]);
			fprintf(fp6, "%c", (unsigned char)CArrayOriginalImage6->m_B[i][j]);
		}
	}

	fclose(fp6);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp7 = fopen(ORIGINAL_FILE7, "rb");

	if (fp7 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage7->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage7->m_Width; j++)
		{
			fscanf(fp7, "%c", &file_temp7);
			CArrayOriginalImage7->m_R[i][j] = (double)file_temp7;

			fscanf(fp7, "%c", &file_temp7);
			CArrayOriginalImage7->m_G[i][j] = (double)file_temp7;

			fscanf(fp7, "%c", &file_temp7);
			CArrayOriginalImage7->m_B[i][j] = (double)file_temp7;
		}
	}

	fclose(fp7);

	// 파일 저장하기 
	fp7 = fopen(RESULT_FILE7, "wb");

	for (i = 0; i < CArrayOriginalImage7->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage7->m_Width; j++)
		{
			fprintf(fp7, "%c", (unsigned char)CArrayOriginalImage7->m_R[i][j]);
			fprintf(fp7, "%c", (unsigned char)CArrayOriginalImage7->m_G[i][j]);
			fprintf(fp7, "%c", (unsigned char)CArrayOriginalImage7->m_B[i][j]);
		}
	}

	fclose(fp7);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp8 = fopen(ORIGINAL_FILE8, "rb");

	if (fp8 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage8->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage8->m_Width; j++)
		{
			fscanf(fp8, "%c", &file_temp8);
			CArrayOriginalImage8->m_R[i][j] = (double)file_temp8;

			fscanf(fp8, "%c", &file_temp8);
			CArrayOriginalImage8->m_G[i][j] = (double)file_temp8;

			fscanf(fp8, "%c", &file_temp8);
			CArrayOriginalImage8->m_B[i][j] = (double)file_temp8;
		}
	}

	fclose(fp8);

	// 파일 저장하기 
	fp8 = fopen(RESULT_FILE8, "wb");

	for (i = 0; i < CArrayOriginalImage8->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage8->m_Width; j++)
		{
			fprintf(fp8, "%c", (unsigned char)CArrayOriginalImage8->m_R[i][j]);
			fprintf(fp8, "%c", (unsigned char)CArrayOriginalImage8->m_G[i][j]);
			fprintf(fp8, "%c", (unsigned char)CArrayOriginalImage8->m_B[i][j]);
		}
	}

	fclose(fp8);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp9 = fopen(ORIGINAL_FILE9, "rb");

	if (fp9 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage3->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage3->m_Width; j++)
		{
			fscanf(fp9, "%c", &file_temp9);
			CArrayOriginalImage9->m_R[i][j] = (double)file_temp9;

			fscanf(fp9, "%c", &file_temp9);
			CArrayOriginalImage9->m_G[i][j] = (double)file_temp9;

			fscanf(fp9, "%c", &file_temp9);
			CArrayOriginalImage9->m_B[i][j] = (double)file_temp9;
		}
	}

	fclose(fp9);

	// 파일 저장하기 
	fp9 = fopen(RESULT_FILE9, "wb");

	for (i = 0; i < CArrayOriginalImage9->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage9->m_Width; j++)
		{
			fprintf(fp9, "%c", (unsigned char)CArrayOriginalImage9->m_R[i][j]);
			fprintf(fp9, "%c", (unsigned char)CArrayOriginalImage9->m_G[i][j]);
			fprintf(fp9, "%c", (unsigned char)CArrayOriginalImage9->m_B[i][j]);
		}
	}

	fclose(fp9);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp10 = fopen(ORIGINAL_FILE10, "rb");

	if (fp10 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage10->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage10->m_Width; j++)
		{
			fscanf(fp10, "%c", &file_temp10);
			CArrayOriginalImage10->m_R[i][j] = (double)file_temp10;

			fscanf(fp10, "%c", &file_temp10);
			CArrayOriginalImage10->m_G[i][j] = (double)file_temp10;

			fscanf(fp10, "%c", &file_temp10);
			CArrayOriginalImage10->m_B[i][j] = (double)file_temp10;
		}
	}

	fclose(fp10);

	// 파일 저장하기 
	fp10 = fopen(RESULT_FILE10, "wb");

	for (i = 0; i < CArrayOriginalImage10->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage10->m_Width; j++)
		{
			fprintf(fp10, "%c", (unsigned char)CArrayOriginalImage10->m_R[i][j]);
			fprintf(fp10, "%c", (unsigned char)CArrayOriginalImage10->m_G[i][j]);
			fprintf(fp10, "%c", (unsigned char)CArrayOriginalImage10->m_B[i][j]);
		}
	}

	fclose(fp10);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp11 = fopen(ORIGINAL_FILE11, "rb");

	if (fp11 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage11->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage11->m_Width; j++)
		{
			fscanf(fp11, "%c", &file_temp11);
			CArrayOriginalImage11->m_R[i][j] = (double)file_temp11;

			fscanf(fp11, "%c", &file_temp11);
			CArrayOriginalImage11->m_G[i][j] = (double)file_temp11;

			fscanf(fp11, "%c", &file_temp11);
			CArrayOriginalImage11->m_B[i][j] = (double)file_temp11;
		}
	}

	fclose(fp11);

	// 파일 저장하기 
	fp11 = fopen(RESULT_FILE11, "wb");

	for (i = 0; i < CArrayOriginalImage11->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage11->m_Width; j++)
		{
			fprintf(fp11, "%c", (unsigned char)CArrayOriginalImage11->m_R[i][j]);
			fprintf(fp11, "%c", (unsigned char)CArrayOriginalImage11->m_G[i][j]);
			fprintf(fp11, "%c", (unsigned char)CArrayOriginalImage11->m_B[i][j]);
		}
	}

	fclose(fp11);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp12 = fopen(ORIGINAL_FILE12, "rb");

	if (fp12 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage12->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage12->m_Width; j++)
		{
			fscanf(fp12, "%c", &file_temp12);
			CArrayOriginalImage12->m_R[i][j] = (double)file_temp12;

			fscanf(fp12, "%c", &file_temp12);
			CArrayOriginalImage12->m_G[i][j] = (double)file_temp12;

			fscanf(fp12, "%c", &file_temp12);
			CArrayOriginalImage12->m_B[i][j] = (double)file_temp12;
		}
	}

	fclose(fp12);

	// 파일 저장하기 
	fp12 = fopen(RESULT_FILE12, "wb");

	for (i = 0; i < CArrayOriginalImage12->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage12->m_Width; j++)
		{
			fprintf(fp12, "%c", (unsigned char)CArrayOriginalImage12->m_R[i][j]);
			fprintf(fp12, "%c", (unsigned char)CArrayOriginalImage12->m_G[i][j]);
			fprintf(fp12, "%c", (unsigned char)CArrayOriginalImage12->m_B[i][j]);
		}
	}

	fclose(fp12);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp13 = fopen(ORIGINAL_FILE13, "rb");

	if (fp13 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage13->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage13->m_Width; j++)
		{
			fscanf(fp13, "%c", &file_temp13);
			CArrayOriginalImage13->m_R[i][j] = (double)file_temp13;

			fscanf(fp13, "%c", &file_temp13);
			CArrayOriginalImage13->m_G[i][j] = (double)file_temp13;

			fscanf(fp13, "%c", &file_temp13);
			CArrayOriginalImage13->m_B[i][j] = (double)file_temp13;
		}
	}

	fclose(fp13);

	// 파일 저장하기 
	fp13 = fopen(RESULT_FILE13, "wb");

	for (i = 0; i < CArrayOriginalImage13->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage13->m_Width; j++)
		{
			fprintf(fp13, "%c", (unsigned char)CArrayOriginalImage13->m_R[i][j]);
			fprintf(fp13, "%c", (unsigned char)CArrayOriginalImage13->m_G[i][j]);
			fprintf(fp13, "%c", (unsigned char)CArrayOriginalImage13->m_B[i][j]);
		}
	}

	fclose(fp13);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp14 = fopen(ORIGINAL_FILE14, "rb");

	if (fp14 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage14->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage14->m_Width; j++)
		{
			fscanf(fp14, "%c", &file_temp14);
			CArrayOriginalImage14->m_R[i][j] = (double)file_temp14;

			fscanf(fp14, "%c", &file_temp14);
			CArrayOriginalImage14->m_G[i][j] = (double)file_temp14;

			fscanf(fp14, "%c", &file_temp14);
			CArrayOriginalImage14->m_B[i][j] = (double)file_temp14;
		}
	}

	fclose(fp14);

	// 파일 저장하기 
	fp14 = fopen(RESULT_FILE14, "wb");

	for (i = 0; i < CArrayOriginalImage14->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage14->m_Width; j++)
		{
			fprintf(fp14, "%c", (unsigned char)CArrayOriginalImage14->m_R[i][j]);
			fprintf(fp14, "%c", (unsigned char)CArrayOriginalImage14->m_G[i][j]);
			fprintf(fp14, "%c", (unsigned char)CArrayOriginalImage14->m_B[i][j]);
		}
	}

	fclose(fp14);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp15 = fopen(ORIGINAL_FILE15, "rb");

	if (fp15 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage15->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage15->m_Width; j++)
		{
			fscanf(fp15, "%c", &file_temp15);
			CArrayOriginalImage15->m_R[i][j] = (double)file_temp15;

			fscanf(fp15, "%c", &file_temp15);
			CArrayOriginalImage15->m_G[i][j] = (double)file_temp15;

			fscanf(fp15, "%c", &file_temp15);
			CArrayOriginalImage15->m_B[i][j] = (double)file_temp15;
		}
	}

	fclose(fp15);

	// 파일 저장하기 
	fp15 = fopen(RESULT_FILE15, "wb");

	for (i = 0; i < CArrayOriginalImage15->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage15->m_Width; j++)
		{
			fprintf(fp15, "%c", (unsigned char)CArrayOriginalImage15->m_R[i][j]);
			fprintf(fp15, "%c", (unsigned char)CArrayOriginalImage15->m_G[i][j]);
			fprintf(fp15, "%c", (unsigned char)CArrayOriginalImage15->m_B[i][j]);
		}
	}

	fclose(fp15);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp16 = fopen(ORIGINAL_FILE16, "rb");

	if (fp16 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage16->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage16->m_Width; j++)
		{
			fscanf(fp16, "%c", &file_temp16);
			CArrayOriginalImage16->m_R[i][j] = (double)file_temp16;

			fscanf(fp16, "%c", &file_temp16);
			CArrayOriginalImage16->m_G[i][j] = (double)file_temp16;

			fscanf(fp16, "%c", &file_temp16);
			CArrayOriginalImage16->m_B[i][j] = (double)file_temp16;
		}
	}

	fclose(fp16);

	// 파일 저장하기 
	fp16 = fopen(RESULT_FILE16, "wb");

	for (i = 0; i < CArrayOriginalImage16->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage16->m_Width; j++)
		{
			fprintf(fp16, "%c", (unsigned char)CArrayOriginalImage16->m_R[i][j]);
			fprintf(fp16, "%c", (unsigned char)CArrayOriginalImage16->m_G[i][j]);
			fprintf(fp16, "%c", (unsigned char)CArrayOriginalImage16->m_B[i][j]);
		}
	}

	fclose(fp16);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp17 = fopen(ORIGINAL_FILE17, "rb");

	if (fp17 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage17->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage17->m_Width; j++)
		{
			fscanf(fp17, "%c", &file_temp17);
			CArrayOriginalImage17->m_R[i][j] = (double)file_temp17;

			fscanf(fp17, "%c", &file_temp17);
			CArrayOriginalImage17->m_G[i][j] = (double)file_temp17;

			fscanf(fp17, "%c", &file_temp17);
			CArrayOriginalImage17->m_B[i][j] = (double)file_temp17;
		}
	}

	fclose(fp17);

	// 파일 저장하기 
	fp17 = fopen(RESULT_FILE17, "wb");

	for (i = 0; i < CArrayOriginalImage17->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage17->m_Width; j++)
		{
			fprintf(fp17, "%c", (unsigned char)CArrayOriginalImage17->m_R[i][j]);
			fprintf(fp17, "%c", (unsigned char)CArrayOriginalImage17->m_G[i][j]);
			fprintf(fp17, "%c", (unsigned char)CArrayOriginalImage17->m_B[i][j]);
		}
	}

	fclose(fp17);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp18 = fopen(ORIGINAL_FILE18, "rb");

	if (fp18 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage18->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage18->m_Width; j++)
		{
			fscanf(fp18, "%c", &file_temp18);
			CArrayOriginalImage18->m_R[i][j] = (double)file_temp18;

			fscanf(fp18, "%c", &file_temp18);
			CArrayOriginalImage18->m_G[i][j] = (double)file_temp18;

			fscanf(fp18, "%c", &file_temp18);
			CArrayOriginalImage18->m_B[i][j] = (double)file_temp18;
		}
	}

	fclose(fp18);

	// 파일 저장하기 
	fp18 = fopen(RESULT_FILE18, "wb");

	for (i = 0; i < CArrayOriginalImage18->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage18->m_Width; j++)
		{
			fprintf(fp18, "%c", (unsigned char)CArrayOriginalImage18->m_R[i][j]);
			fprintf(fp18, "%c", (unsigned char)CArrayOriginalImage18->m_G[i][j]);
			fprintf(fp18, "%c", (unsigned char)CArrayOriginalImage18->m_B[i][j]);
		}
	}

	fclose(fp18);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp19 = fopen(ORIGINAL_FILE19, "rb");

	if (fp19 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage19->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage19->m_Width; j++)
		{
			fscanf(fp19, "%c", &file_temp19);
			CArrayOriginalImage19->m_R[i][j] = (double)file_temp19;

			fscanf(fp19, "%c", &file_temp19);
			CArrayOriginalImage19->m_G[i][j] = (double)file_temp19;

			fscanf(fp19, "%c", &file_temp19);
			CArrayOriginalImage19->m_B[i][j] = (double)file_temp19;
		}
	}

	fclose(fp19);

	// 파일 저장하기 
	fp19 = fopen(RESULT_FILE19, "wb");

	for (i = 0; i < CArrayOriginalImage19->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage19->m_Width; j++)
		{
			fprintf(fp19, "%c", (unsigned char)CArrayOriginalImage19->m_R[i][j]);
			fprintf(fp19, "%c", (unsigned char)CArrayOriginalImage19->m_G[i][j]);
			fprintf(fp19, "%c", (unsigned char)CArrayOriginalImage19->m_B[i][j]);
		}
	}

	fclose(fp19);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp20 = fopen(ORIGINAL_FILE20, "rb");

	if (fp20 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage20->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage20->m_Width; j++)
		{
			fscanf(fp20, "%c", &file_temp20);
			CArrayOriginalImage20->m_R[i][j] = (double)file_temp20;

			fscanf(fp20, "%c", &file_temp20);
			CArrayOriginalImage20->m_G[i][j] = (double)file_temp20;

			fscanf(fp20, "%c", &file_temp20);
			CArrayOriginalImage20->m_B[i][j] = (double)file_temp20;
		}
	}

	fclose(fp20);

	// 파일 저장하기 
	fp20 = fopen(RESULT_FILE20, "wb");

	for (i = 0; i < CArrayOriginalImage20->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage20->m_Width; j++)
		{
			fprintf(fp20, "%c", (unsigned char)CArrayOriginalImage20->m_R[i][j]);
			fprintf(fp20, "%c", (unsigned char)CArrayOriginalImage20->m_G[i][j]);
			fprintf(fp20, "%c", (unsigned char)CArrayOriginalImage20->m_B[i][j]);
		}
	}

	fclose(fp20);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp21 = fopen(ORIGINAL_FILE21, "rb");

	if (fp21 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage21->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage21->m_Width; j++)
		{
			fscanf(fp21, "%c", &file_temp21);
			CArrayOriginalImage21->m_R[i][j] = (double)file_temp21;

			fscanf(fp21, "%c", &file_temp21);
			CArrayOriginalImage21->m_G[i][j] = (double)file_temp21;

			fscanf(fp21, "%c", &file_temp21);
			CArrayOriginalImage21->m_B[i][j] = (double)file_temp21;
		}
	}

	fclose(fp21);

	// 파일 저장하기 
	fp21 = fopen(RESULT_FILE21, "wb");

	for (i = 0; i < CArrayOriginalImage21->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage21->m_Width; j++)
		{
			fprintf(fp21, "%c", (unsigned char)CArrayOriginalImage21->m_R[i][j]);
			fprintf(fp21, "%c", (unsigned char)CArrayOriginalImage21->m_G[i][j]);
			fprintf(fp21, "%c", (unsigned char)CArrayOriginalImage21->m_B[i][j]);
		}
	}

	fclose(fp21);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp22 = fopen(ORIGINAL_FILE22, "rb");

	if (fp22 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage22->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage22->m_Width; j++)
		{
			fscanf(fp22, "%c", &file_temp22);
			CArrayOriginalImage22->m_R[i][j] = (double)file_temp22;

			fscanf(fp22, "%c", &file_temp22);
			CArrayOriginalImage22->m_G[i][j] = (double)file_temp22;

			fscanf(fp22, "%c", &file_temp22);
			CArrayOriginalImage22->m_B[i][j] = (double)file_temp22;
		}
	}

	fclose(fp22);

	// 파일 저장하기 
	fp22 = fopen(RESULT_FILE22, "wb");

	for (i = 0; i < CArrayOriginalImage22->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage22->m_Width; j++)
		{
			fprintf(fp22, "%c", (unsigned char)CArrayOriginalImage22->m_R[i][j]);
			fprintf(fp22, "%c", (unsigned char)CArrayOriginalImage22->m_G[i][j]);
			fprintf(fp22, "%c", (unsigned char)CArrayOriginalImage22->m_B[i][j]);
		}
	}

	fclose(fp22);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp23 = fopen(ORIGINAL_FILE23, "rb");

	if (fp23 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage23->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage23->m_Width; j++)
		{
			fscanf(fp23, "%c", &file_temp23);
			CArrayOriginalImage23->m_R[i][j] = (double)file_temp23;

			fscanf(fp23, "%c", &file_temp23);
			CArrayOriginalImage23->m_G[i][j] = (double)file_temp23;

			fscanf(fp23, "%c", &file_temp23);
			CArrayOriginalImage23->m_B[i][j] = (double)file_temp23;
		}
	}

	fclose(fp23);

	// 파일 저장하기 
	fp23 = fopen(RESULT_FILE23, "wb");

	for (i = 0; i < CArrayOriginalImage23->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage23->m_Width; j++)
		{
			fprintf(fp23, "%c", (unsigned char)CArrayOriginalImage23->m_R[i][j]);
			fprintf(fp23, "%c", (unsigned char)CArrayOriginalImage23->m_G[i][j]);
			fprintf(fp23, "%c", (unsigned char)CArrayOriginalImage23->m_B[i][j]);
		}
	}

	fclose(fp23);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp24 = fopen(ORIGINAL_FILE24, "rb");

	if (fp24 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage24->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage24->m_Width; j++)
		{
			fscanf(fp24, "%c", &file_temp24);
			CArrayOriginalImage24->m_R[i][j] = (double)file_temp24;

			fscanf(fp24, "%c", &file_temp24);
			CArrayOriginalImage24->m_G[i][j] = (double)file_temp24;

			fscanf(fp24, "%c", &file_temp24);
			CArrayOriginalImage24->m_B[i][j] = (double)file_temp24;
		}
	}

	fclose(fp24);

	// 파일 저장하기 
	fp24 = fopen(RESULT_FILE24, "wb"); 

	for (i = 0; i < CArrayOriginalImage24->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage24->m_Width; j++)
		{
			fprintf(fp24, "%c", (unsigned char)CArrayOriginalImage24->m_R[i][j]);
			fprintf(fp24, "%c", (unsigned char)CArrayOriginalImage24->m_G[i][j]);
			fprintf(fp24, "%c", (unsigned char)CArrayOriginalImage24->m_B[i][j]);
		}
	}

	fclose(fp24);

	// 동적메모리 반환

	// Raw 파일 읽기
	fp25 = fopen(ORIGINAL_FILE25, "rb");

	if (fp25 == NULL)
	{
		printf("file pointer error!\n");
	}

	for (i = 0; i < CArrayOriginalImage25->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage25->m_Width; j++)
		{
			fscanf(fp25, "%c", &file_temp25);
			CArrayOriginalImage25->m_R[i][j] = (double)file_temp25;

			fscanf(fp25, "%c", &file_temp25);
			CArrayOriginalImage25->m_G[i][j] = (double)file_temp25;

			fscanf(fp25, "%c", &file_temp25);
			CArrayOriginalImage25->m_B[i][j] = (double)file_temp25;
		}
	}

	fclose(fp25);

	// 파일 저장하기 
	fp25 = fopen(RESULT_FILE25, "wb");

	for (i = 0; i < CArrayOriginalImage25->m_Height; i++)
	{
		for (j = 0; j < CArrayOriginalImage25->m_Width; j++)
		{
			fprintf(fp25, "%c", (unsigned char)CArrayOriginalImage25->m_R[i][j]);
			fprintf(fp25, "%c", (unsigned char)CArrayOriginalImage25->m_G[i][j]);
			fprintf(fp25, "%c", (unsigned char)CArrayOriginalImage25->m_B[i][j]);
		}
	}

	fclose(fp25);

	k = 0;
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			A1->m_R[k][0] = CArrayOriginalImage1->m_R[i][j];
			A1->m_R[k][1] = CArrayOriginalImage2->m_R[i][j];
			A1->m_R[k][2] = CArrayOriginalImage3->m_R[i][j];
			A1->m_R[k][3] = CArrayOriginalImage4->m_R[i][j];
			A1->m_R[k][4] = CArrayOriginalImage5->m_R[i][j];
			A1->m_R[k][5] = CArrayOriginalImage6->m_R[i][j];
			A1->m_R[k][6] = CArrayOriginalImage7->m_R[i][j];
			A1->m_R[k][7] = CArrayOriginalImage8->m_R[i][j];
			A1->m_R[k][8] = CArrayOriginalImage9->m_R[i][j];
			A1->m_R[k][9] = CArrayOriginalImage10->m_R[i][j];
			A1->m_R[k][10] = CArrayOriginalImage11->m_R[i][j];
			A1->m_R[k][11] = CArrayOriginalImage12->m_R[i][j];
			A1->m_R[k][12] = CArrayOriginalImage13->m_R[i][j];
			A1->m_R[k][13] = CArrayOriginalImage14->m_R[i][j];
			A1->m_R[k][14] = CArrayOriginalImage15->m_R[i][j];
			A1->m_R[k][15] = CArrayOriginalImage16->m_R[i][j];
			A1->m_R[k][16] = CArrayOriginalImage17->m_R[i][j];
			A1->m_R[k][17] = CArrayOriginalImage18->m_R[i][j];
			A1->m_R[k][18] = CArrayOriginalImage19->m_R[i][j];
			A1->m_R[k][19] = CArrayOriginalImage20->m_R[i][j];
			A1->m_R[k][20] = CArrayOriginalImage21->m_R[i][j];
			A1->m_R[k][21] = CArrayOriginalImage22->m_R[i][j];
			A1->m_R[k][22] = CArrayOriginalImage23->m_R[i][j];
			A1->m_R[k][23] = CArrayOriginalImage24->m_R[i][j];
			A1->m_R[k][24] = CArrayOriginalImage25->m_R[i][j];
			k++;
		}
	}


	// 동적메모리 반환
	m1 = fopen(MEAN_FILE1, "wb");
	
	// 평균 얼굴 계산
	int max = img->m_R[0][0];
	int min = img->m_R[0][0];
	k = 0;
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			sum_R = (A1->m_R[k][0] + A1->m_R[k][1] + A1->m_R[k][2] + A1->m_R[k][3] + A1->m_R[k][4] + A1->m_R[k][5] + A1->m_R[k][6] + A1->m_R[k][7] + A1->m_R[k][8] + A1->m_R[k][9] + A1->m_R[k][10] + A1->m_R[k][11] + A1->m_R[k][12] + A1->m_R[k][13] + A1->m_R[k][14] + A1->m_R[k][15] + A1->m_R[k][16] + A1->m_R[k][17] + A1->m_R[k][18] + A1->m_R[k][19] + A1->m_R[k][20] + A1->m_R[k][21] + A1->m_R[k][22] + A1->m_R[k][23] + A1->m_R[k][24]);
			k++;
			img->m_R[i][j] = sum_R / M;
		}
	}
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			if (max < img->m_R[i][j]) {
				max = img->m_R[i][j];
			}
			if (min > img->m_R[i][j]) {
				min = img->m_R[i][j];
			}
		}
	}
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fprintf(m1, "%c", (unsigned char)(255 * (img->m_R[i][j] - min) / (max - min)));
			fprintf(m1, "%c", (unsigned char)(255 * (img->m_R[i][j] - min) / (max - min)));
			fprintf(m1, "%c", (unsigned char)(255 * (img->m_R[i][j] - min) / (max - min)));
		}
	}
	fclose(m1);

	k = 0;

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			img1->m_R[k][0] = img->m_R[i][j];
			k++;
		}
	}

	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			img2->m_R[j][i] = A1->m_R[j][i] - img1->m_R[j][0];
		}
	}
	MatrixXd m(columns, rows);
	MatrixXd E(rows, rows);

	
	for (i = 0; i < columns; i++) {
		for (j = 0; j < rows; j++) {
			m(i, j) = A1->m_R[i][j];
		}
		
	}
	
	PCA::Compute(m, E);
	

	for (i = 0; i < columns; i++) {
		for (j = 0; j < rows; j++) {
			for (k = 0; k < rows; k++) {
				u1->m_R[i][j] += (A1->m_R[i][k] * E(k, j));
			}
		}
	}
	k = 0;
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			B1->m_R[i][j] = u1->m_R[k][0];
			B2->m_R[i][j] = u1->m_R[k][1];
			B3->m_R[i][j] = u1->m_R[k][2];
			B4->m_R[i][j] = u1->m_R[k][3];
			B5->m_R[i][j] = u1->m_R[k][4];
			B6->m_R[i][j] = u1->m_R[k][5];
			B7->m_R[i][j] = u1->m_R[k][6];
			B8->m_R[i][j] = u1->m_R[k][7];
			B9->m_R[i][j] = u1->m_R[k][8];
			B10->m_R[i][j] = u1->m_R[k][9];
			B11->m_R[i][j] = u1->m_R[k][10];
			B12->m_R[i][j] = u1->m_R[k][11];
			B13->m_R[i][j] = u1->m_R[k][12];
			B14->m_R[i][j] = u1->m_R[k][13];
			B15->m_R[i][j] = u1->m_R[k][14];
			B16->m_R[i][j] = u1->m_R[k][15];
			B17->m_R[i][j] = u1->m_R[k][16];
			B18->m_R[i][j] = u1->m_R[k][17];
			B19->m_R[i][j] = u1->m_R[k][18];
			B20->m_R[i][j] = u1->m_R[k][19];
			B21->m_R[i][j] = u1->m_R[k][20];
			B22->m_R[i][j] = u1->m_R[k][21];
			B23->m_R[i][j] = u1->m_R[k][22];
			B24->m_R[i][j] = u1->m_R[k][23];
			B25->m_R[i][j] = u1->m_R[k][24];
			k++;
		}
	}

	eigen1 = fopen(EIGEN_FILE1, "wb");

	max = B1->m_R[0][0];
	min = B1->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B1->m_R[i][j]) {
				max = B1->m_R[i][j];
			}
			if (min > B1->m_R[i][j]) {
				min = B1->m_R[i][j];
			}
		}
	}
	
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen1, "%c", (unsigned char)(255 * (B1->m_R[i][j] - min) / (max - min)));
			fprintf(eigen1, "%c", (unsigned char)(255 * (B1->m_R[i][j] - min) / (max - min)));
			fprintf(eigen1, "%c", (unsigned char)(255 * (B1->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen1);

	eigen2 = fopen(EIGEN_FILE2, "wb");

	max = B2->m_R[0][0];
	min = B2->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B2->m_R[i][j]) {
				max = B2->m_R[i][j];
			}
			if (min > B2->m_R[i][j]) {
				min = B2->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen2, "%c", (unsigned char)(255 * (B2->m_R[i][j] - min) / (max - min)));
			fprintf(eigen2, "%c", (unsigned char)(255 * (B2->m_R[i][j] - min) / (max - min)));
			fprintf(eigen2, "%c", (unsigned char)(255 * (B2->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen2);


	eigen3 = fopen(EIGEN_FILE3, "wb");

	max = B3->m_R[0][0];
	min = B3->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B3->m_R[i][j]) {
				max = B3->m_R[i][j];
			}
			if (min > B3->m_R[i][j]) {
				min = B3->m_R[i][j];
			}
		}
	}
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen3, "%c", (unsigned char)(255 * (B2->m_R[i][j] - min) / (max - min)));
			fprintf(eigen3, "%c", (unsigned char)(255 * (B2->m_R[i][j] - min) / (max - min)));
			fprintf(eigen3, "%c", (unsigned char)(255 * (B2->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen3);

	eigen4 = fopen(EIGEN_FILE4, "wb");

	max = B4->m_R[0][0];
	min = B4->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B4->m_R[i][j]) {
				max = B4->m_R[i][j];
			}
			if (min > B4->m_R[i][j]) {
				min = B4->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen4, "%c", (unsigned char)(255 * (B4->m_R[i][j] - min) / (max - min)));
			fprintf(eigen4, "%c", (unsigned char)(255 * (B4->m_R[i][j] - min) / (max - min)));
			fprintf(eigen4, "%c", (unsigned char)(255 * (B4->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen4);

	eigen5 = fopen(EIGEN_FILE5, "wb");

	max = B5->m_R[0][0];
	min = B5->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B5->m_R[i][j]) {
				max = B5->m_R[i][j];
			}
			if (min > B5->m_R[i][j]) {
				min = B5->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen5, "%c", (unsigned char)(255 * (B5->m_R[i][j] - min) / (max - min)));
			fprintf(eigen5, "%c", (unsigned char)(255 * (B5->m_R[i][j] - min) / (max - min)));
			fprintf(eigen5, "%c", (unsigned char)(255 * (B5->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen5);

	max = B6->m_R[0][0];
	min = B6->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B6->m_R[i][j]) {
				max = B6->m_R[i][j];
			}
			if (min > B6->m_R[i][j]) {
				min = B6->m_R[i][j];
			}
		}
	}

	eigen6 = fopen(EIGEN_FILE6, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen6, "%c", (unsigned char)(255 * (B6->m_R[i][j] - min) / (max - min)));
			fprintf(eigen6, "%c", (unsigned char)(255 * (B6->m_R[i][j] - min) / (max - min)));
			fprintf(eigen6, "%c", (unsigned char)(255 * (B6->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen6);

	eigen7 = fopen(EIGEN_FILE7, "wb");

	max = B7->m_R[0][0];
	min = B7->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B7->m_R[i][j]) {
				max = B7->m_R[i][j];
			}
			if (min > B7->m_R[i][j]) {
				min = B7->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen7, "%c", (unsigned char)(255 * (B7->m_R[i][j] - min) / (max - min)));
			fprintf(eigen7, "%c", (unsigned char)(255 * (B7->m_R[i][j] - min) / (max - min)));
			fprintf(eigen7, "%c", (unsigned char)(255 * (B7->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen7);

	eigen8 = fopen(EIGEN_FILE8, "wb");

	max = B8->m_R[0][0];
	min = B8->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B8->m_R[i][j]) {
				max = B8->m_R[i][j];
			}
			if (min > B8->m_R[i][j]) {
				min = B8->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen8, "%c", (unsigned char)(255 * (B8->m_R[i][j] - min) / (max - min)));
			fprintf(eigen8, "%c", (unsigned char)(255 * (B8->m_R[i][j] - min) / (max - min)));
			fprintf(eigen8, "%c", (unsigned char)(255 * (B8->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen8);

	eigen9 = fopen(EIGEN_FILE9, "wb");

	max = B9->m_R[0][0];
	min = B9->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B9->m_R[i][j]) {
				max = B9->m_R[i][j];
			}
			if (min > B9->m_R[i][j]) {
				min = B9->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen9, "%c", (unsigned char)(255 * (B9->m_R[i][j] - min) / (max - min)));
			fprintf(eigen9, "%c", (unsigned char)(255 * (B9->m_R[i][j] - min) / (max - min)));
			fprintf(eigen9, "%c", (unsigned char)(255 * (B9->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen9);

	eigen10 = fopen(EIGEN_FILE10, "wb");

	max = B10->m_R[0][0];
	min = B10->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B10->m_R[i][j]) {
				max = B10->m_R[i][j];
			}
			if (min > B10->m_R[i][j]) {
				min = B10->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen10, "%c", (unsigned char)(255 * (B10->m_R[i][j] - min) / (max - min)));
			fprintf(eigen10, "%c", (unsigned char)(255 * (B10->m_R[i][j] - min) / (max - min)));
			fprintf(eigen10, "%c", (unsigned char)(255 * (B10->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen10);

	eigen11 = fopen(EIGEN_FILE11, "wb");

	max = B11->m_R[0][0];
	min = B11->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B11->m_R[i][j]) {
				max = B11->m_R[i][j];
			}
			if (min > B11->m_R[i][j]) {
				min = B11->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen11, "%c", (unsigned char)(255 * (B11->m_R[i][j] - min) / (max - min)));
			fprintf(eigen11, "%c", (unsigned char)(255 * (B11->m_R[i][j] - min) / (max - min)));
			fprintf(eigen11, "%c", (unsigned char)(255 * (B11->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen11);

	eigen12 = fopen(EIGEN_FILE12, "wb");

	max = B12->m_R[0][0];
	min = B12->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B12->m_R[i][j]) {
				max = B12->m_R[i][j];
			}
			if (min > B12->m_R[i][j]) {
				min = B12->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen12, "%c", (unsigned char)(255 * (B12->m_R[i][j] - min) / (max - min)));
			fprintf(eigen12, "%c", (unsigned char)(255 * (B12->m_R[i][j] - min) / (max - min)));
			fprintf(eigen12, "%c", (unsigned char)(255 * (B12->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen12);

	eigen13 = fopen(EIGEN_FILE13, "wb");

	max = B13->m_R[0][0];
	min = B13->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B13->m_R[i][j]) {
				max = B13->m_R[i][j];
			}
			if (min > B13->m_R[i][j]) {
				min = B13->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen13, "%c", (unsigned char)(255 * (B13->m_R[i][j] - min) / (max - min)));
			fprintf(eigen13, "%c", (unsigned char)(255 * (B13->m_R[i][j] - min) / (max - min)));
			fprintf(eigen13, "%c", (unsigned char)(255 * (B13->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen13);

	eigen14 = fopen(EIGEN_FILE14, "wb");

	max = B14->m_R[0][0];
	min = B14->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B14->m_R[i][j]) {
				max = B14->m_R[i][j];
			}
			if (min > B14->m_R[i][j]) {
				min = B14->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen14, "%c", (unsigned char)(255 * (B14->m_R[i][j] - min) / (max - min)));
			fprintf(eigen14, "%c", (unsigned char)(255 * (B14->m_R[i][j] - min) / (max - min)));
			fprintf(eigen14, "%c", (unsigned char)(255 * (B14->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen14);

	eigen15 = fopen(EIGEN_FILE15, "wb");

	max = B15->m_R[0][0];
	min = B15->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B15->m_R[i][j]) {
				max = B15->m_R[i][j];
			}
			if (min > B15->m_R[i][j]) {
				min = B15->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen15, "%c", (unsigned char)(255 * (B15->m_R[i][j] - min) / (max - min)));
			fprintf(eigen15, "%c", (unsigned char)(255 * (B15->m_R[i][j] - min) / (max - min)));
			fprintf(eigen15, "%c", (unsigned char)(255 * (B15->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen15);

	eigen16 = fopen(EIGEN_FILE16, "wb");

	max = B16->m_R[0][0];
	min = B16->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B16->m_R[i][j]) {
				max = B16->m_R[i][j];
			}
			if (min > B16->m_R[i][j]) {
				min = B16->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen16, "%c", (unsigned char)(255 * (B16->m_R[i][j] - min) / (max - min)));
			fprintf(eigen16, "%c", (unsigned char)(255 * (B16->m_R[i][j] - min) / (max - min)));
			fprintf(eigen16, "%c", (unsigned char)(255 * (B16->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen16);

	eigen17 = fopen(EIGEN_FILE17, "wb");

	max = B17->m_R[0][0];
	min = B17->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B17->m_R[i][j]) {
				max = B17->m_R[i][j];
			}
			if (min > B17->m_R[i][j]) {
				min = B17->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen17, "%c", (unsigned char)(255 * (B17->m_R[i][j] - min) / (max - min)));
			fprintf(eigen17, "%c", (unsigned char)(255 * (B17->m_R[i][j] - min) / (max - min)));
			fprintf(eigen17, "%c", (unsigned char)(255 * (B17->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen17);

	eigen18 = fopen(EIGEN_FILE18, "wb");

	max = B18->m_R[0][0];
	min = B18->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B18->m_R[i][j]) {
				max = B18->m_R[i][j];
			}
			if (min > B18->m_R[i][j]) {
				min = B18->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen18, "%c", (unsigned char)(255 * (B18->m_R[i][j] - min) / (max - min)));
			fprintf(eigen18, "%c", (unsigned char)(255 * (B18->m_R[i][j] - min) / (max - min)));
			fprintf(eigen18, "%c", (unsigned char)(255 * (B18->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen18);

	eigen19 = fopen(EIGEN_FILE19, "wb");

	max = B19->m_R[0][0];
	min = B19->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B19->m_R[i][j]) {
				max = B19->m_R[i][j];
			}
			if (min > B19->m_R[i][j]) {
				min = B19->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen19, "%c", (unsigned char)(255 * (B19->m_R[i][j] - min) / (max - min)));
			fprintf(eigen19, "%c", (unsigned char)(255 * (B19->m_R[i][j] - min) / (max - min)));
			fprintf(eigen19, "%c", (unsigned char)(255 * (B19->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen19);

	eigen20 = fopen(EIGEN_FILE20, "wb");

	max = B20->m_R[0][0];
	min = B20->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B20->m_R[i][j]) {
				max = B20->m_R[i][j];
			}
			if (min > B20->m_R[i][j]) {
				min = B20->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen20, "%c", (unsigned char)(255 * (B20->m_R[i][j] - min) / (max - min)));
			fprintf(eigen20, "%c", (unsigned char)(255 * (B20->m_R[i][j] - min) / (max - min)));
			fprintf(eigen20, "%c", (unsigned char)(255 * (B20->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen20);

	eigen21 = fopen(EIGEN_FILE21, "wb");

	max = B21->m_R[0][0];
	min = B21->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B21->m_R[i][j]) {
				max = B21->m_R[i][j];
			}
			if (min > B21->m_R[i][j]) {
				min = B21->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen21, "%c", (unsigned char)(255 * (B21->m_R[i][j] - min) / (max - min)));
			fprintf(eigen21, "%c", (unsigned char)(255 * (B21->m_R[i][j] - min) / (max - min)));
			fprintf(eigen21, "%c", (unsigned char)(255 * (B21->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen21);

	eigen22 = fopen(EIGEN_FILE22, "wb");

	max = B22->m_R[0][0];
	min = B22->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B22->m_R[i][j]) {
				max = B22->m_R[i][j];
			}
			if (min > B22->m_R[i][j]) {
				min = B22->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen22, "%c", (unsigned char)(255 * (B22->m_R[i][j] - min) / (max - min)));
			fprintf(eigen22, "%c", (unsigned char)(255 * (B22->m_R[i][j] - min) / (max - min)));
			fprintf(eigen22, "%c", (unsigned char)(255 * (B22->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen22);

	eigen23 = fopen(EIGEN_FILE23, "wb");

	max = B23->m_R[0][0];
	min = B23->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B23->m_R[i][j]) {
				max = B23->m_R[i][j];
			}
			if (min > B23->m_R[i][j]) {
				min = B23->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen23, "%c", (unsigned char)(255 * (B23->m_R[i][j] - min) / (max - min)));
			fprintf(eigen23, "%c", (unsigned char)(255 * (B23->m_R[i][j] - min) / (max - min)));
			fprintf(eigen23, "%c", (unsigned char)(255 * (B23->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen23); 

	eigen24 = fopen(EIGEN_FILE24, "wb");

	max = B24->m_R[0][0];
	min = B24->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B24->m_R[i][j]) {
				max = B24->m_R[i][j];
			}
			if (min > B24->m_R[i][j]) {
				min = B24->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen24, "%c", (unsigned char)(255 * (B24->m_R[i][j] - min) / (max - min)));
			fprintf(eigen24, "%c", (unsigned char)(255 * (B24->m_R[i][j] - min) / (max - min)));
			fprintf(eigen24, "%c", (unsigned char)(255 * (B24->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen24);

	eigen25 = fopen(EIGEN_FILE25, "wb");

	max = B25->m_R[0][0];
	min = B25->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < B25->m_R[i][j]) {
				max = B25->m_R[i][j];
			}
			if (min > B25->m_R[i][j]) {
				min = B25->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(eigen25, "%c", (unsigned char)(255 * (B25->m_R[i][j] - min) / (max - min)));
			fprintf(eigen25, "%c", (unsigned char)(255 * (B25->m_R[i][j] - min) / (max - min)));
			fprintf(eigen25, "%c", (unsigned char)(255 * (B25->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(eigen25);

	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			u2->m_R[i][j] = u1->m_R[j][i];
		}
	}
;
	/*
	double F1[25];
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			F1[i] += (u2->m_R[i][j] * img2->m_R[j][i]);
		}
	}
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			E1->m_R[j][i] = F1[i] * u1->m_R[j][i];		
		}
	}

	k = 0;
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			G1->m_R[i][j] = E1->m_R[k][0];
			G2->m_R[i][j] = E1->m_R[k][1];
			G3->m_R[i][j] = E1->m_R[k][2];
			G4->m_R[i][j] = E1->m_R[k][3];
			G5->m_R[i][j] = E1->m_R[k][4];
			G6->m_R[i][j] = E1->m_R[k][5];
			G7->m_R[i][j] = E1->m_R[k][6];
			G8->m_R[i][j] = E1->m_R[k][7];
			G9->m_R[i][j] = E1->m_R[k][8];
			G10->m_R[i][j] = E1->m_R[k][9];
			G11->m_R[i][j] = E1->m_R[k][10];
			G12->m_R[i][j] = E1->m_R[k][11];
			G13->m_R[i][j] = E1->m_R[k][12];
			G14->m_R[i][j] = E1->m_R[k][13];
			G15->m_R[i][j] = E1->m_R[k][14];
			G16->m_R[i][j] = E1->m_R[k][15];
			G17->m_R[i][j] = E1->m_R[k][16];
			G18->m_R[i][j] = E1->m_R[k][17];
			G19->m_R[i][j] = E1->m_R[k][18];
			G20->m_R[i][j] = E1->m_R[k][19];
			G21->m_R[i][j] = E1->m_R[k][20];
			G22->m_R[i][j] = E1->m_R[k][21];
			G23->m_R[i][j] = E1->m_R[k][22];
			G24->m_R[i][j] = E1->m_R[k][23];
			G25->m_R[i][j] = E1->m_R[k][24];
			k++;
		}
	}
	*/
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			C1->m_R[i][j] = CArrayOriginalImage1->m_R[i][j] - img->m_R[i][j];
			C2->m_R[i][j] = CArrayOriginalImage2->m_R[i][j] - img->m_R[i][j];
			C3->m_R[i][j] = CArrayOriginalImage3->m_R[i][j] - img->m_R[i][j];
			C4->m_R[i][j] = CArrayOriginalImage4->m_R[i][j] - img->m_R[i][j];
			C5->m_R[i][j] = CArrayOriginalImage5->m_R[i][j] - img->m_R[i][j];
			C6->m_R[i][j] = CArrayOriginalImage6->m_R[i][j] - img->m_R[i][j];
			C7->m_R[i][j] = CArrayOriginalImage7->m_R[i][j] - img->m_R[i][j];
			C8->m_R[i][j] = CArrayOriginalImage8->m_R[i][j] - img->m_R[i][j];
			C9->m_R[i][j] = CArrayOriginalImage9->m_R[i][j] - img->m_R[i][j];
			C10->m_R[i][j] = CArrayOriginalImage10->m_R[i][j] - img->m_R[i][j];
			C11->m_R[i][j] = CArrayOriginalImage11->m_R[i][j] - img->m_R[i][j];
			C12->m_R[i][j] = CArrayOriginalImage12->m_R[i][j] - img->m_R[i][j];
			C13->m_R[i][j] = CArrayOriginalImage13->m_R[i][j] - img->m_R[i][j];
			C14->m_R[i][j] = CArrayOriginalImage14->m_R[i][j] - img->m_R[i][j];
			C15->m_R[i][j] = CArrayOriginalImage15->m_R[i][j] - img->m_R[i][j];
			C16->m_R[i][j] = CArrayOriginalImage16->m_R[i][j] - img->m_R[i][j];
			C17->m_R[i][j] = CArrayOriginalImage17->m_R[i][j] - img->m_R[i][j];
			C18->m_R[i][j] = CArrayOriginalImage18->m_R[i][j] - img->m_R[i][j];
			C19->m_R[i][j] = CArrayOriginalImage19->m_R[i][j] - img->m_R[i][j];
			C20->m_R[i][j] = CArrayOriginalImage20->m_R[i][j] - img->m_R[i][j];
			C21->m_R[i][j] = CArrayOriginalImage21->m_R[i][j] - img->m_R[i][j];
			C22->m_R[i][j] = CArrayOriginalImage22->m_R[i][j] - img->m_R[i][j];
			C23->m_R[i][j] = CArrayOriginalImage23->m_R[i][j] - img->m_R[i][j];
			C24->m_R[i][j] = CArrayOriginalImage24->m_R[i][j] - img->m_R[i][j];
			C25->m_R[i][j] = CArrayOriginalImage25->m_R[i][j] - img->m_R[i][j];
		}
	}

	k = 0;

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			D1->m_R[k][0] = C1->m_R[i][j];
			D1->m_R[k][1] = C2->m_R[i][j];
			D1->m_R[k][2] = C3->m_R[i][j];
			D1->m_R[k][3] = C4->m_R[i][j];
			D1->m_R[k][4] = C5->m_R[i][j];
			D1->m_R[k][5] = C6->m_R[i][j];
			D1->m_R[k][6] = C7->m_R[i][j];
			D1->m_R[k][7] = C8->m_R[i][j];
			D1->m_R[k][8] = C9->m_R[i][j];
			D1->m_R[k][9] = C10->m_R[i][j];
			D1->m_R[k][10] = C11->m_R[i][j];
			D1->m_R[k][11] = C12->m_R[i][j];
			D1->m_R[k][12] = C13->m_R[i][j];
			D1->m_R[k][13] = C14->m_R[i][j];
			D1->m_R[k][14] = C15->m_R[i][j];
			D1->m_R[k][15] = C16->m_R[i][j];
			D1->m_R[k][16] = C17->m_R[i][j];
			D1->m_R[k][17] = C18->m_R[i][j];
			D1->m_R[k][18] = C19->m_R[i][j];
			D1->m_R[k][19] = C20->m_R[i][j];
			D1->m_R[k][20] = C21->m_R[i][j];
			D1->m_R[k][21] = C22->m_R[i][j];
			D1->m_R[k][22] = C23->m_R[i][j];
			D1->m_R[k][23] = C24->m_R[i][j];
			D1->m_R[k][24] = C25->m_R[i][j];
			k++;
		}
	}
	
	for (i = 0; i < rows; i++) {
		for (j = 0; j < rows; j++) {
			for (k = 0; k < columns; k++) {
				p1->m_R[i][j] += (u2->m_R[i][k] * D1->m_R[k][j]);
			}
		}
	}

	for (i = 0; i < columns; i++) {
		for (j = 0; j < rows; j++) {
			for (k = 0; k < rows; k++) {
				E1->m_R[i][j] += (u1->m_R[i][k] * p1->m_R[k][j]);
			}
		}
	}
	
	k = 0;

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			F1->m_R[i][j] = E1->m_R[k][0];
			F2->m_R[i][j] = E1->m_R[k][1];
			F3->m_R[i][j] = E1->m_R[k][2];
			F4->m_R[i][j] = E1->m_R[k][3];
			F5->m_R[i][j] = E1->m_R[k][4];
			F6->m_R[i][j] = E1->m_R[k][5];
			F7->m_R[i][j] = E1->m_R[k][6];
			F8->m_R[i][j] = E1->m_R[k][7];
			F9->m_R[i][j] = E1->m_R[k][8];
			F10->m_R[i][j] = E1->m_R[k][9];
			F11->m_R[i][j] = E1->m_R[k][10];
			F12->m_R[i][j] = E1->m_R[k][11];
			F13->m_R[i][j] = E1->m_R[k][12];
			F14->m_R[i][j] = E1->m_R[k][13];
			F15->m_R[i][j] = E1->m_R[k][14];
			F16->m_R[i][j] = E1->m_R[k][15];
			F17->m_R[i][j] = E1->m_R[k][16];
			F18->m_R[i][j] = E1->m_R[k][17];
			F19->m_R[i][j] = E1->m_R[k][18];
			F20->m_R[i][j] = E1->m_R[k][19];
			F21->m_R[i][j] = E1->m_R[k][20];
			F22->m_R[i][j] = E1->m_R[k][21];
			F23->m_R[i][j] = E1->m_R[k][22];
			F24->m_R[i][j] = E1->m_R[k][23];
			F25->m_R[i][j] = E1->m_R[k][24];
			k++;
		}
	}
	
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			G1->m_R[i][j] = F1->m_R[i][j] + img->m_R[i][j];
			G2->m_R[i][j] = F2->m_R[i][j] + img->m_R[i][j];
			G3->m_R[i][j] = F3->m_R[i][j] + img->m_R[i][j];
			G4->m_R[i][j] = F4->m_R[i][j] + img->m_R[i][j];
			G5->m_R[i][j] = F5->m_R[i][j] + img->m_R[i][j];
			G6->m_R[i][j] = F6->m_R[i][j] + img->m_R[i][j];
			G7->m_R[i][j] = F7->m_R[i][j] + img->m_R[i][j];
			G8->m_R[i][j] = F8->m_R[i][j] + img->m_R[i][j];
			G9->m_R[i][j] = F9->m_R[i][j] + img->m_R[i][j];
			G10->m_R[i][j] = F10->m_R[i][j] + img->m_R[i][j];
			G11->m_R[i][j] = F11->m_R[i][j] + img->m_R[i][j];
			G12->m_R[i][j] = F12->m_R[i][j] + img->m_R[i][j];
			G13->m_R[i][j] = F13->m_R[i][j] + img->m_R[i][j];
			G14->m_R[i][j] = F14->m_R[i][j] + img->m_R[i][j];
			G15->m_R[i][j] = F15->m_R[i][j] + img->m_R[i][j];
			G16->m_R[i][j] = F16->m_R[i][j] + img->m_R[i][j];
			G17->m_R[i][j] = F17->m_R[i][j] + img->m_R[i][j];
			G18->m_R[i][j] = F18->m_R[i][j] + img->m_R[i][j];
			G19->m_R[i][j] = F19->m_R[i][j] + img->m_R[i][j];
			G20->m_R[i][j] = F20->m_R[i][j] + img->m_R[i][j];
			G21->m_R[i][j] = F21->m_R[i][j] + img->m_R[i][j];
			G22->m_R[i][j] = F22->m_R[i][j] + img->m_R[i][j];
			G23->m_R[i][j] = F23->m_R[i][j] + img->m_R[i][j];
			G24->m_R[i][j] = F24->m_R[i][j] + img->m_R[i][j];
			G25->m_R[i][j] = F25->m_R[i][j] + img->m_R[i][j];
		}
	}
	
	norm1 = fopen(NORM_FILE1, "wb");

	max = G1->m_R[0][0];
	min = G1->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G1->m_R[i][j]) {
				max = G1->m_R[i][j];
			}
			if (min > G1->m_R[i][j]) {
				min = G1->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm1, "%c", (unsigned char)(255 * (G1->m_R[i][j] - min) / (max - min)));
			fprintf(norm1, "%c", (unsigned char)(255 * (G1->m_R[i][j] - min) / (max - min)));
			fprintf(norm1, "%c", (unsigned char)(255 * (G1->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm1);

	norm2 = fopen(NORM_FILE2, "wb");

	max = G2->m_R[0][0];
	min = G2->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G2->m_R[i][j]) {
				max = G2->m_R[i][j];
			}
			if (min > G2->m_R[i][j]) {
				min = G2->m_R[i][j];
			}
		}
	}

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm2, "%c", (unsigned char)(255 * (G2->m_R[i][j] - min) / (max - min)));
			fprintf(norm2, "%c", (unsigned char)(255 * (G2->m_R[i][j] - min) / (max - min)));
			fprintf(norm2, "%c", (unsigned char)(255 * (G2->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm2);

	norm3 = fopen(NORM_FILE3, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm3, "%c", (unsigned char)G3->m_R[i][j]);
			fprintf(norm3, "%c", (unsigned char)G3->m_R[i][j]);
			fprintf(norm3, "%c", (unsigned char)G3->m_R[i][j]);
		}
	}

	fclose(norm3);

	norm4 = fopen(NORM_FILE4, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm4, "%c", (unsigned char)G4->m_R[i][j]);
			fprintf(norm4, "%c", (unsigned char)G4->m_R[i][j]);
			fprintf(norm4, "%c", (unsigned char)G4->m_R[i][j]);
		}
	}

	fclose(norm4);

	norm5 = fopen(NORM_FILE5, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm5, "%c", (unsigned char)G5->m_R[i][j]);
			fprintf(norm5, "%c", (unsigned char)G5->m_R[i][j]);
			fprintf(norm5, "%c", (unsigned char)G5->m_R[i][j]);
		}
	}

	fclose(norm5);

	norm6 = fopen(NORM_FILE6, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm6, "%c", (unsigned char)G6->m_R[i][j]);
			fprintf(norm6, "%c", (unsigned char)G6->m_R[i][j]);
			fprintf(norm6, "%c", (unsigned char)G6->m_R[i][j]);
		}
	}


	norm7 = fopen(NORM_FILE7, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm7, "%c", (unsigned char)G7->m_R[i][j]);
			fprintf(norm7, "%c", (unsigned char)G7->m_R[i][j]);
			fprintf(norm7, "%c", (unsigned char)G7->m_R[i][j]);
		}
	}

	fclose(norm7);

	norm8 = fopen(NORM_FILE8, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm8, "%c", (unsigned char)G8->m_R[i][j]);
			fprintf(norm8, "%c", (unsigned char)G8->m_R[i][j]);
			fprintf(norm8, "%c", (unsigned char)G8->m_R[i][j]);
		}
	}

	fclose(norm8);

	norm9 = fopen(NORM_FILE9, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm9, "%c", (unsigned char)G9->m_R[i][j]);
			fprintf(norm9, "%c", (unsigned char)G9->m_R[i][j]);
			fprintf(norm9, "%c", (unsigned char)G9->m_R[i][j]);
		}
	}

	fclose(norm9);

	norm10 = fopen(NORM_FILE10, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm10, "%c", (unsigned char)G10->m_R[i][j]);
			fprintf(norm10, "%c", (unsigned char)G10->m_R[i][j]);
			fprintf(norm10, "%c", (unsigned char)G10->m_R[i][j]);
		}
	}


	norm11 = fopen(NORM_FILE11, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm11, "%c", (unsigned char)G11->m_R[i][j]);
			fprintf(norm11, "%c", (unsigned char)G11->m_R[i][j]);
			fprintf(norm11, "%c", (unsigned char)G11->m_R[i][j]);
		}
	}

	fclose(norm11);

	norm12 = fopen(NORM_FILE12, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm12, "%c", (unsigned char)G12->m_R[i][j]);
			fprintf(norm12, "%c", (unsigned char)G12->m_R[i][j]); 
			fprintf(norm12, "%c", (unsigned char)G12->m_R[i][j]);
		}
	}

	fclose(norm12);

	norm13 = fopen(NORM_FILE13, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm13, "%c", (unsigned char)G13->m_R[i][j]);
			fprintf(norm13, "%c", (unsigned char)G13->m_R[i][j]);
			fprintf(norm13, "%c", (unsigned char)G13->m_R[i][j]);
		}
	}

	fclose(norm13);

	norm14 = fopen(NORM_FILE14, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm14, "%c", (unsigned char)G14->m_R[i][j]);
			fprintf(norm14, "%c", (unsigned char)G14->m_R[i][j]);
			fprintf(norm14, "%c", (unsigned char)G14->m_R[i][j]);
		}
	}

	fclose(norm14);

	norm15 = fopen(NORM_FILE15, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm15, "%c", (unsigned char)G15->m_R[i][j]);
			fprintf(norm15, "%c", (unsigned char)G15->m_R[i][j]);
			fprintf(norm15, "%c", (unsigned char)G15->m_R[i][j]);
		}
	}

	fclose(norm15);

	norm16 = fopen(NORM_FILE16, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm16, "%c", (unsigned char)G16->m_R[i][j]);
			fprintf(norm16, "%c", (unsigned char)G16->m_R[i][j]);
			fprintf(norm16, "%c", (unsigned char)G16->m_R[i][j]);
		}
	}

	fclose(norm16);

	norm16 = fopen(NORM_FILE16, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm16, "%c", (unsigned char)G16->m_R[i][j]);
			fprintf(norm16, "%c", (unsigned char)G16->m_R[i][j]);
			fprintf(norm16, "%c", (unsigned char)G16->m_R[i][j]);
		}
	}

	fclose(norm16);

	norm17 = fopen(NORM_FILE17, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm17, "%c", (unsigned char)G17->m_R[i][j]);
			fprintf(norm17, "%c", (unsigned char)G17->m_R[i][j]);
			fprintf(norm17, "%c", (unsigned char)G17->m_R[i][j]);
		}
	}

	fclose(norm17);

	norm18 = fopen(NORM_FILE18, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm18, "%c", (unsigned char)G18->m_R[i][j]);
			fprintf(norm18, "%c", (unsigned char)G18->m_R[i][j]);
			fprintf(norm18, "%c", (unsigned char)G18->m_R[i][j]);
		}
	}

	fclose(norm18);

	norm19 = fopen(NORM_FILE19, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm19, "%c", (unsigned char)G19->m_R[i][j]);
			fprintf(norm19, "%c", (unsigned char)G19->m_R[i][j]);
			fprintf(norm19, "%c", (unsigned char)G19->m_R[i][j]);
		}
	}

	fclose(norm19);

	norm20 = fopen(NORM_FILE20, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm20, "%c", (unsigned char)G20->m_R[i][j]);
			fprintf(norm20, "%c", (unsigned char)G20->m_R[i][j]);
			fprintf(norm20, "%c", (unsigned char)G20->m_R[i][j]);
		}
	}

	fclose(norm20);

	norm21 = fopen(NORM_FILE21, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm21, "%c", (unsigned char)G21->m_R[i][j]);
			fprintf(norm21, "%c", (unsigned char)G21->m_R[i][j]);
			fprintf(norm21, "%c", (unsigned char)G21->m_R[i][j]);
		}
	}

	fclose(norm21);

	norm22 = fopen(NORM_FILE22, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm22, "%c", (unsigned char)G22->m_R[i][j]);
			fprintf(norm22, "%c", (unsigned char)G22->m_R[i][j]);
			fprintf(norm22, "%c", (unsigned char)G22->m_R[i][j]);
		}
	}

	fclose(norm22);

	norm23 = fopen(NORM_FILE23, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm23, "%c", (unsigned char)G23->m_R[i][j]);
			fprintf(norm23, "%c", (unsigned char)G23->m_R[i][j]);
			fprintf(norm23, "%c", (unsigned char)G23->m_R[i][j]);
		}
	}

	fclose(norm23);

	norm24 = fopen(NORM_FILE24, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm24, "%c", (unsigned char)G24->m_R[i][j]);
			fprintf(norm24, "%c", (unsigned char)G24->m_R[i][j]);
			fprintf(norm24, "%c", (unsigned char)G24->m_R[i][j]);
		}
	}

	fclose(norm24);

	norm25 = fopen(NORM_FILE25, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm25, "%c", (unsigned char)G25->m_R[i][j]);
			fprintf(norm25, "%c", (unsigned char)G25->m_R[i][j]);
			fprintf(norm25, "%c", (unsigned char)G25->m_R[i][j]);
		}
	}

	fclose(norm25);
	
/*
	FILE *o1 = fopen("a1.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o1, "%lf", &(G1->m_R[i][j]));
		}
	}
	fclose(o1);

	max = G1->m_R[0][0];
	min = G1->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G1->m_R[i][j]) {
				max = G1->m_R[i][j];
			}
			if (min > G1->m_R[i][j]) {
				min = G1->m_R[i][j];
			}
		}
	}
	norm1 = fopen(NORM_FILE1, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm1, "%c", (unsigned char)(255 * (G1->m_R[i][j] - min) / (max - min)));
			fprintf(norm1, "%c", (unsigned char)(255 * (G1->m_R[i][j] - min) / (max - min)));
			fprintf(norm1, "%c", (unsigned char)(255 * (G1->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm1);

	FILE *o2 = fopen("a2.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o2, "%lf", &(G2->m_R[i][j]));
		}
	}
	fclose(o2);

	max = G2->m_R[0][0];
	min = G2->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G2->m_R[i][j]) {
				max = G2->m_R[i][j];
			}
			if (min > G2->m_R[i][j]) {
				min = G2->m_R[i][j];
			}
		}
	}
	norm2 = fopen(NORM_FILE2, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm2, "%c", (unsigned char)(255 * (G2->m_R[i][j] - min) / (max - min)));
			fprintf(norm2, "%c", (unsigned char)(255 * (G2->m_R[i][j] - min) / (max - min)));
			fprintf(norm2, "%c", (unsigned char)(255 * (G2->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm2);

	FILE *o3 = fopen("a3.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o1, "%lf", &(G3->m_R[i][j]));
		}
	}
	fclose(o3);

	max = G3->m_R[0][0];
	min = G3->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G3->m_R[i][j]) {
				max = G3->m_R[i][j];
			}
			if (min > G3->m_R[i][j]) {
				min = G3->m_R[i][j];
			}
		}
	}
	norm3 = fopen(NORM_FILE3, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm3, "%c", (unsigned char)(255 * (G3->m_R[i][j] - min) / (max - min)));
			fprintf(norm3, "%c", (unsigned char)(255 * (G3->m_R[i][j] - min) / (max - min)));
			fprintf(norm3, "%c", (unsigned char)(255 * (G3->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm3);

	FILE *o4 = fopen("a4.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o1, "%lf", &(G4->m_R[i][j]));
		}
	}
	fclose(o4);

	max = G4->m_R[0][0];
	min = G4->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G4->m_R[i][j]) {
				max = G4->m_R[i][j];
			}
			if (min > G4->m_R[i][j]) {
				min = G4->m_R[i][j];
			}
		}
	}
	norm4 = fopen(NORM_FILE4, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm4, "%c", (unsigned char)(255 * (G4->m_R[i][j] - min) / (max - min)));
			fprintf(norm4, "%c", (unsigned char)(255 * (G4->m_R[i][j] - min) / (max - min)));
			fprintf(norm4, "%c", (unsigned char)(255 * (G4->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm4);

	FILE *o5 = fopen("a5.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o5, "%lf", &(G5->m_R[i][j]));		
		}
	}
	fclose(o5);

	max = G5->m_R[0][0];
	min = G5->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G5->m_R[i][j]) {
				max = G5->m_R[i][j];
			}
			if (min > G5->m_R[i][j]) {
				min = G5->m_R[i][j];
			}
		}
	}
	norm5 = fopen(NORM_FILE5, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm5, "%c", (unsigned char)(255 * (G5->m_R[i][j] - min) / (max - min)));
			fprintf(norm5, "%c", (unsigned char)(255 * (G5->m_R[i][j] - min) / (max - min)));
			fprintf(norm5, "%c", (unsigned char)(255 * (G5->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm5);

	FILE *o6 = fopen("a6.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o6, "%lf", &(G6->m_R[i][j]));
		}
	}
	fclose(o6);

	max = G6->m_R[0][0];
	min = G6->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G6->m_R[i][j]) {
				max = G6->m_R[i][j];
			}
			if (min > G6->m_R[i][j]) {
				min = G6->m_R[i][j];
			}
		}
	}
	norm6 = fopen(NORM_FILE6, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm6, "%c", (unsigned char)(255 * (G6->m_R[i][j] - min) / (max - min)));
			fprintf(norm6, "%c", (unsigned char)(255 * (G6->m_R[i][j] - min) / (max - min)));
			fprintf(norm6, "%c", (unsigned char)(255 * (G6->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm6);

	FILE *o7 = fopen("a7.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o7, "%lf", &(G7->m_R[i][j]));
		}
	}
	fclose(o7);

	max = G7->m_R[0][0];
	min = G7->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G7->m_R[i][j]) {
				max = G7->m_R[i][j];
			}
			if (min > G7->m_R[i][j]) {
				min = G7->m_R[i][j];
			}
		}
	}
	norm7 = fopen(NORM_FILE7, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm7, "%c", (unsigned char)(255 * (G7->m_R[i][j] - min) / (max - min)));
			fprintf(norm7, "%c", (unsigned char)(255 * (G7->m_R[i][j] - min) / (max - min)));
			fprintf(norm7, "%c", (unsigned char)(255 * (G7->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm7);

	FILE *o8 = fopen("a8.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o8, "%lf", &(G8->m_R[i][j]));
		}
	}
	fclose(o8);

	max = G8->m_R[0][0];
	min = G8->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G8->m_R[i][j]) {
				max = G8->m_R[i][j];
			}
			if (min > G8->m_R[i][j]) {
				min = G8->m_R[i][j];
			}
		}
	}
	norm8 = fopen(NORM_FILE8, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm8, "%c", (unsigned char)(255 * (G8->m_R[i][j] - min) / (max - min)));
			fprintf(norm8, "%c", (unsigned char)(255 * (G8->m_R[i][j] - min) / (max - min)));
			fprintf(norm8, "%c", (unsigned char)(255 * (G8->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm8);

	FILE *o9 = fopen("a9.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o9, "%lf", &(G9->m_R[i][j]));
		}
	}
	fclose(o9);

	max = G9->m_R[0][0];
	min = G9->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G9->m_R[i][j]) {
				max = G9->m_R[i][j];
			}
			if (min > G9->m_R[i][j]) {
				min = G9->m_R[i][j];
			}
		}
	}
	norm9 = fopen(NORM_FILE9, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm9, "%c", (unsigned char)(255 * (G9->m_R[i][j] - min) / (max - min)));
			fprintf(norm9, "%c", (unsigned char)(255 * (G9->m_R[i][j] - min) / (max - min)));
			fprintf(norm9, "%c", (unsigned char)(255 * (G9->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm9);

	FILE *o10 = fopen("a10.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o10, "%lf", &(G10->m_R[i][j]));
		}
	}
	fclose(o10);

	max = G10->m_R[0][0];
	min = G10->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G10->m_R[i][j]) {
				max = G10->m_R[i][j];
			}
			if (min > G10->m_R[i][j]) {
				min = G10->m_R[i][j];
			}
		}
	}
	norm10 = fopen(NORM_FILE10, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm10, "%c", (unsigned char)(255 * (G10->m_R[i][j] - min) / (max - min)));
			fprintf(norm10, "%c", (unsigned char)(255 * (G10->m_R[i][j] - min) / (max - min)));
			fprintf(norm10, "%c", (unsigned char)(255 * (G10->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm10);

	FILE *o11 = fopen("a11.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o11, "%lf", &(G11->m_R[i][j]));
		}
	}
	fclose(o11);

	max = G11->m_R[0][0];
	min = G11->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G11->m_R[i][j]) {
				max = G11->m_R[i][j];
			}
			if (min > G11->m_R[i][j]) {
				min = G11->m_R[i][j];
			}
		}
	}
	norm11 = fopen(NORM_FILE11, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm11, "%c", (unsigned char)(255 * (G11->m_R[i][j] - min) / (max - min)));
			fprintf(norm11, "%c", (unsigned char)(255 * (G11->m_R[i][j] - min) / (max - min)));
			fprintf(norm11, "%c", (unsigned char)(255 * (G11->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm11);

	FILE *o12 = fopen("a12.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o12, "%lf", &(G12->m_R[i][j]));
		}
	}
	fclose(o12);

	max = G12->m_R[0][0];
	min = G12->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G12->m_R[i][j]) {
				max = G12->m_R[i][j];
			}
			if (min > G12->m_R[i][j]) {
				min = G12->m_R[i][j];
			}
		}
	}
	norm12 = fopen(NORM_FILE12, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm12, "%c", (unsigned char)(255 * (G12->m_R[i][j] - min) / (max - min)));
			fprintf(norm12, "%c", (unsigned char)(255 * (G12->m_R[i][j] - min) / (max - min)));
			fprintf(norm12, "%c", (unsigned char)(255 * (G12->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm12);

	FILE *o13 = fopen("a13.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o13, "%lf", &(G13->m_R[i][j]));
		}
	}
	fclose(o13);

	max = G13->m_R[0][0];
	min = G13->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G13->m_R[i][j]) {
				max = G13->m_R[i][j];
			}
			if (min > G13->m_R[i][j]) {
				min = G13->m_R[i][j];
			}
		}
	}
	norm13 = fopen(NORM_FILE13, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm13, "%c", (unsigned char)(255 * (G13->m_R[i][j] - min) / (max - min)));
			fprintf(norm13, "%c", (unsigned char)(255 * (G13->m_R[i][j] - min) / (max - min)));
			fprintf(norm13, "%c", (unsigned char)(255 * (G13->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm13);

	FILE *o14 = fopen("a14.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o14, "%lf", &(G14->m_R[i][j]));
		}
	}
	fclose(o14);

	max = G14->m_R[0][0];
	min = G14->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G14->m_R[i][j]) {
				max = G14->m_R[i][j];
			}
			if (min > G14->m_R[i][j]) {
				min = G14->m_R[i][j];
			}
		}
	}
	norm14 = fopen(NORM_FILE14, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm14, "%c", (unsigned char)(255 * (G14->m_R[i][j] - min) / (max - min)));
			fprintf(norm14, "%c", (unsigned char)(255 * (G14->m_R[i][j] - min) / (max - min)));
			fprintf(norm14, "%c", (unsigned char)(255 * (G14->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm14);

	FILE *o15 = fopen("a15.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o15, "%lf", &(G15->m_R[i][j]));
		}
	}
	fclose(o15);

	max = G15->m_R[0][0];
	min = G15->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G15->m_R[i][j]) {
				max = G15->m_R[i][j];
			}
			if (min > G15->m_R[i][j]) {
				min = G15->m_R[i][j];
			}
		}
	}
	norm15 = fopen(NORM_FILE15, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm15, "%c", (unsigned char)(255 * (G15->m_R[i][j] - min) / (max - min)));
			fprintf(norm15, "%c", (unsigned char)(255 * (G15->m_R[i][j] - min) / (max - min)));
			fprintf(norm15, "%c", (unsigned char)(255 * (G15->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm15);

	FILE *o16 = fopen("a16.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o16, "%lf", &(G16->m_R[i][j]));
		}
	}
	fclose(o16);

	max = G16->m_R[0][0];
	min = G16->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G16->m_R[i][j]) {
				max = G16->m_R[i][j];
			}
			if (min > G16->m_R[i][j]) {
				min = G16->m_R[i][j];
			}
		}
	}
	norm16 = fopen(NORM_FILE1, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm16, "%c", (unsigned char)(255 * (G16->m_R[i][j] - min) / (max - min)));
			fprintf(norm16, "%c", (unsigned char)(255 * (G16->m_R[i][j] - min) / (max - min)));
			fprintf(norm16, "%c", (unsigned char)(255 * (G16->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm16);

	FILE *o17 = fopen("a17.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o17, "%lf", &(G17->m_R[i][j]));
		}
	}
	fclose(o17);

	max = G17->m_R[0][0];
	min = G17->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G17->m_R[i][j]) {
				max = G17->m_R[i][j];
			}
			if (min > G17->m_R[i][j]) {
				min = G17->m_R[i][j];
			}
		}
	}
	norm17 = fopen(NORM_FILE17, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm17, "%c", (unsigned char)(255 * (G17->m_R[i][j] - min) / (max - min)));
			fprintf(norm17, "%c", (unsigned char)(255 * (G17->m_R[i][j] - min) / (max - min)));
			fprintf(norm17, "%c", (unsigned char)(255 * (G17->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm17);

	FILE *o18 = fopen("a18.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o18, "%lf", &(G18->m_R[i][j]));
		}
	}
	fclose(o18);

	max = G18->m_R[0][0];
	min = G18->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G18->m_R[i][j]) {
				max = G18->m_R[i][j];
			}
			if (min > G18->m_R[i][j]) {
				min = G18->m_R[i][j];
			}
		}
	}
	norm18 = fopen(NORM_FILE18, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm18, "%c", (unsigned char)(255 * (G18->m_R[i][j] - min) / (max - min)));
			fprintf(norm18, "%c", (unsigned char)(255 * (G18->m_R[i][j] - min) / (max - min)));
			fprintf(norm18, "%c", (unsigned char)(255 * (G18->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm18);

	FILE *o19 = fopen("a19.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o19, "%lf", &(G19->m_R[i][j]));
		}
	}
	fclose(o19);

	max = G19->m_R[0][0];
	min = G19->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G19->m_R[i][j]) {
				max = G19->m_R[i][j];
			}
			if (min > G19->m_R[i][j]) {
				min = G19->m_R[i][j];
			}
		}
	}
	norm19 = fopen(NORM_FILE19, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm19, "%c", (unsigned char)(255 * (G19->m_R[i][j] - min) / (max - min)));
			fprintf(norm19, "%c", (unsigned char)(255 * (G19->m_R[i][j] - min) / (max - min)));
			fprintf(norm19, "%c", (unsigned char)(255 * (G19->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm19);

	FILE *o20 = fopen("a20.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o20, "%lf", &(G20->m_R[i][j]));
		}
	}
	fclose(o20);

	max = G20->m_R[0][0];
	min = G20->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G20->m_R[i][j]) {
				max = G20->m_R[i][j];
			}
			if (min > G20->m_R[i][j]) {
				min = G20->m_R[i][j];
			}
		}
	}
	norm20 = fopen(NORM_FILE20, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm20, "%c", (unsigned char)(255 * (G20->m_R[i][j] - min) / (max - min)));
			fprintf(norm20, "%c", (unsigned char)(255 * (G20->m_R[i][j] - min) / (max - min)));
			fprintf(norm20, "%c", (unsigned char)(255 * (G20->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm20);

	FILE *o21 = fopen("a21.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o21, "%lf", &(G21->m_R[i][j]));
		}
	}
	fclose(o21);

	max = G21->m_R[0][0];
	min = G21->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G21->m_R[i][j]) {
				max = G21->m_R[i][j];
			}
			if (min > G21->m_R[i][j]) {
				min = G21->m_R[i][j];
			}
		}
	}
	norm21 = fopen(NORM_FILE21, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm21, "%c", (unsigned char)(255 * (G21->m_R[i][j] - min) / (max - min)));
			fprintf(norm21, "%c", (unsigned char)(255 * (G21->m_R[i][j] - min) / (max - min)));
			fprintf(norm21, "%c", (unsigned char)(255 * (G21->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm21);

	FILE *o22 = fopen("a22.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o22, "%lf", &(G22->m_R[i][j]));
		}
	}
	fclose(o22);

	max = G22->m_R[0][0];
	min = G22->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G22->m_R[i][j]) {
				max = G22->m_R[i][j];
			}
			if (min > G22->m_R[i][j]) {
				min = G22->m_R[i][j];
			}
		}
	}
	norm22 = fopen(NORM_FILE22, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm22, "%c", (unsigned char)(255 * (G22->m_R[i][j] - min) / (max - min)));
			fprintf(norm22, "%c", (unsigned char)(255 * (G22->m_R[i][j] - min) / (max - min)));
			fprintf(norm22, "%c", (unsigned char)(255 * (G22->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm22);

	FILE *o23 = fopen("a23.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o23, "%lf", &(G23->m_R[i][j]));
		}
	}
	fclose(o23);

	max = G23->m_R[0][0];
	min = G23->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G23->m_R[i][j]) {
				max = G23->m_R[i][j];
			}
			if (min > G23->m_R[i][j]) {
				min = G23->m_R[i][j];
			}
		}
	}
	norm23 = fopen(NORM_FILE23, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm23, "%c", (unsigned char)(255 * (G23->m_R[i][j] - min) / (max - min)));
			fprintf(norm23, "%c", (unsigned char)(255 * (G23->m_R[i][j] - min) / (max - min)));
			fprintf(norm23, "%c", (unsigned char)(255 * (G23->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm23);

	FILE *o24 = fopen("a24.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o24, "%lf", &(G24->m_R[i][j]));
		}
	}
	fclose(o24);

	max = G24->m_R[0][0];
	min = G24->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G24->m_R[i][j]) {
				max = G24->m_R[i][j];
			}
			if (min > G24->m_R[i][j]) {
				min = G24->m_R[i][j];
			}
		}
	}
	norm24 = fopen(NORM_FILE24, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm24, "%c", (unsigned char)(255 * (G24->m_R[i][j] - min) / (max - min)));
			fprintf(norm24, "%c", (unsigned char)(255 * (G24->m_R[i][j] - min) / (max - min)));
			fprintf(norm24, "%c", (unsigned char)(255 * (G24->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm24);

	FILE *o25 = fopen("a25.txt", "r");
	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++) {
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++) {
			fscanf(o25, "%lf", &(G25->m_R[i][j]));
		}
	}
	fclose(o25);

	max = G25->m_R[0][0];
	min = G25->m_R[0][0];

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			if (max < G25->m_R[i][j]) {
				max = G25->m_R[i][j];
			}
			if (min > G25->m_R[i][j]) {
				min = G25->m_R[i][j];
			}
		}
	}
	norm25 = fopen(NORM_FILE25, "wb");

	for (i = 0; i < PADDING_IMAGE_HEIGHT; i++)
	{
		for (j = 0; j < PADDING_IMAGE_WIDTH; j++)
		{
			fprintf(norm25, "%c", (unsigned char)(255 * (G25->m_R[i][j] - min) / (max - min)));
			fprintf(norm25, "%c", (unsigned char)(255 * (G25->m_R[i][j] - min) / (max - min)));
			fprintf(norm25, "%c", (unsigned char)(255 * (G25->m_R[i][j] - min) / (max - min)));
		}
	}

	fclose(norm25);
	*/
}
