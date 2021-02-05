// kkgroups.h
#ifndef _KKG_ENERGY_KERNEL_H // include guard
#define _KKG_ENERGY_KERNEL_H

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
 * Class EnergyKernel
 */
class EnergyKernel : public Kernel {
private:
  double alpha;
  vec x0;
public:
  EnergyKernel();
  EnergyKernel(const double _alpha);
  EnergyKernel(const double _alpha, const vec _x0);
  mat getKernelMatrix(const mat &X) const override;
};


EnergyKernel::EnergyKernel() {
  alpha = 1;
}

EnergyKernel::EnergyKernel(const double _alpha) {
  alpha = _alpha;
}

EnergyKernel::EnergyKernel(const double _alpha, const vec _x0) {
  alpha = _alpha;
  x0 = _x0;
}

mat EnergyKernel::getKernelMatrix(const mat &X) const {
  vec _x0 = x0;
  if (x0.is_empty() || X.n_cols != x0.n_elem)
    _x0.zeros(X.n_cols);
    // throw invalid_argument("x0 length and X number of columns must be the same.");
  mat matrix(X.n_rows, X.n_rows, fill::zeros);
#pragma omp parallel for shared(matrix)
  for (uword i = 0; i < X.n_rows; i++) {
    for (uword j = 0; j <= i; j++) {
      matrix(i, j) =
        // (pow(norm(X.row(i).t() - _x0,   2), 2) +
        // pow(norm(X.row(j).t() - _x0,    2), 2) -
        pow(1 + dot(X.row(i).t(), X.row(j).t()), 2);
      // matrix(i, j) = -(0.5 *
      //   (pow(norm(X.row(i).t() - _x0,   2), 2) +
      //   pow(norm(X.row(j).t() - _x0,    2), 2) -
      //   2 * pow(dot(X.row(i).t(), X.row(j).t()), 1)));
      // matrix(i, j) = (0.5 *
      //   (pow(norm(X.row(i).t() - _x0,   2), alpha) +
      //   pow(norm(X.row(j).t() - _x0,    2), alpha) -
      //   pow(norm(X.row(i).t() - X.row(j).t(), 2), alpha)));
      matrix(j, i) = matrix(i, j);
    }
  }
  return matrix;
}


}

#endif
