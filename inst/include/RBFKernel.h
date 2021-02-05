// kkgroups.h
#ifndef _KKG_RBF_KERNEL_H // include guard
#define _KKG_RBF_KERNEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

#include <stdlib.h>
#include "Kernel.h"

using namespace arma;
using namespace std;



namespace kkg {


/**
 * Class RBFKernel
 */
class RBFKernel : public Kernel {
private:
  double sigma;
  double getSigma(const mat &X, double exponent) const;
public:
  RBFKernel();
  RBFKernel(const double _sigma);
  RBFKernel(const mat &X);
  RBFKernel(const mat &X, double exponent);
  mat getKernelMatrix(const mat &X) const override;
};



RBFKernel::RBFKernel() {
  sigma = 1;
}

RBFKernel::RBFKernel(const double _sigma) {
  sigma = _sigma;
}

RBFKernel::RBFKernel(const mat &X) {
  sigma = getSigma(X, 1);
}

RBFKernel::RBFKernel(const mat &X, double exponent) {
  sigma = getSigma(X, exponent);
  // cout << "Sigma: " << sigma << endl;
}

double RBFKernel::getSigma(const mat &X, double exponent) const {
  if (X.is_empty())
    return 0;

  int m = X.n_rows;
  mat  fil_ig = repmat(X.row(0) * trans(X.row(0)), 1, m);

  for (int i = 1; i < m; i++) {
    fil_ig = join_cols(fil_ig, repmat(X.row(i) * trans(X.row(i)), 1, m));
  }

  mat col_ig = trans(fil_ig);
  mat cross = X * trans(X);
  fil_ig = (fil_ig + col_ig - 2 * cross);
  fil_ig.diag(0).fill(0);

  vec v = sort(fil_ig.elem(find(fil_ig > 10e-10)));
  int n = v.size();
  int n1 = floor((float(n + 1) / float(2)));
  int n2 = ceil((float(n + 1) / float(2)));

  return pow(sqrt(0.5 * v(n1 - 1) + 0.5 * v(n2 - 1)), exponent);
}


mat RBFKernel::getKernelMatrix(const mat &X) const {
  mat matrix(X.n_rows, X.n_rows, fill::zeros);
  if (!X.is_empty()) {
#pragma omp parallel for shared(matrix)
    for (uword i = 0; i < X.n_rows; i++) {
      for (uword j = 0; j <= i; j++) {
        // matrix(i, j) = exp(-1 * (1/(2*sigma*sigma)) * (2 *
        // matrix(i, j) = exp(1/(2*sigma*sigma) * (2 *
        //   // matrix(i, j) = exp(sigma * (2 *
        //   dot(X.row(i).t(), X.row(j).t()) -
        //   dot(X.row(i).t(), X.row(i).t()) -
        //   dot(X.row(j).t(), X.row(j).t())));
        matrix(i,j) = exp(-1/(2*sigma*sigma) * pow(norm(X.row(i) - X.row(j), 2), 2));
        matrix(j, i) = matrix(i, j);
      }
    }
  }
  return matrix;
}


}

#endif
