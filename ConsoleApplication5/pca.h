#ifndef PCA_H
#define PCA_H

#include <iostream>
#include <assert.h>
#include <vector>
#include <math.h>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace Eigen;
using namespace std;

class PCA
{
public:

	/*
	* Computes the princcipal component of a given matrix. Computation steps:
	* Assert that the input matrix is square matrix.
	* Compute the mean image.
	* Subtract mean image from the data set to get mean centered data vector
	* Compute the covariance matrix from the mean centered data matrix
	* Calculate the eigenvalues and eigenvectors for the covariance matrix
	* Normalize the eigen vectors
	* Find out an eigenvector with the largest eigenvalue
	*
	* @input MatrixXd D the data samples matrix.
	*
	* @returns VectorXd The principal component vector
	*/
	//static MatrixXd Compute(MatrixXd D)
	static void Compute(MatrixXd D, MatrixXd &Ee)
	{
		// The matrix must be square matrix.
		//assert(D.rows() == D.cols());
		int M = D.rows();
		int N = D.cols();

		// 1. Compute the mean image
		MatrixXd mean(1, N);
		mean.setZero();

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				mean(0, j) += D(i, j) / M;
			}
		}


		// 2. Subtract mean image from the data set to get mean centered data vector
		MatrixXd U = D;

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				U(i, j) -= mean(0, j);
			}
		}
		// 3. Compute the covariance matrix from the mean centered data matrix
		MatrixXd covariance = (U.adjoint() * U)  / (double)(N);
		//cout << covariance << endl;
		// 4. Calculate the eigenvalues and eigenvectors for the covariance matrix
		EigenSolver<MatrixXd> solver(covariance);
		MatrixXd eigenVectors = solver.eigenvectors().real();
		VectorXd eigenValues = solver.eigenvalues().real();
	
		//cout << eigenVectors << endl;
		
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				Ee(i,j) = eigenVectors(i, j);
				
			}
		}
		eigenVectors.normalize();

		
		//cout << Ee << endl;

		//cout << eigenVectors << endl;
		//cout << eigenValues << endl;

		// 6. Find out an eigenvector with the largest eigenvalue
		//    which distingushes the data
		sort(eigenValues.derived().data(), eigenValues.derived().data() + eigenValues.derived().size());
		short index = eigenValues.size() - 1;
		VectorXd featureVector = eigenVectors.row(index);
		//return featureVector;
		//return eigenVectors;

	}
};

#endif // PCA_H