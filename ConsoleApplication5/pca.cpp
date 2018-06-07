#include <iostream>
#include "Eigen/Dense"
#include "pca.h"

using Eigen::MatrixXd;
using namespace std;

/*
* This is the main entry for the PCA program.
*/
int main()
{
	// The number of rows in the input matrix
	int rows = 25;

	// The number of columns in the input matrix
	int columns = 10304;

	// Initialize the matrix

	MatrixXd m(rows, columns);
	

	MatrixXd Ee(columns, columns);


	// Set the matrix values.
	m << 1, 1, 1,
		2, 2, 2, 
		3, 3, 3,
		4, 4, 4,
		5, 5, 5;

	// Compute the principal component and show results.
	//cout << PCA::Compute(m,E) << endl;
	PCA::Compute(m, Ee);
	cout << Ee << endl;
}