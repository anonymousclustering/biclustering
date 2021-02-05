// kkgroups.h
#ifndef _KKG_RANDOM_INITIALIZATION_H // include guard
#define _KKG_RANDOM_INITIALIZATION_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include "InitializationMethod.h"

using namespace arma;



namespace kkg {




/**
 * Class RandomInitialization
 */
class RandomInitialization : public InitializationMethod {
public:
  mat getLabelMatrix(const mat& X, const uword k) const override;
};


mat RandomInitialization::getLabelMatrix(const mat& X, const uword k) const {
  mat Z(X.n_rows, k, fill::zeros);
  for (uword i = 0; i < X.n_rows; i++) {
    Z(i, rand() % k) = 1;
  }
  return Z;
}


}

#endif
