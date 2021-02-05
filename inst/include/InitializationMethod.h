// kkgroups.h
#ifndef _KKG_INITIALIZATION_H // include guard
#define _KKG_INITIALIZATION_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

using namespace arma;
using namespace std;



namespace kkg {


/**
 * Class InitializationMethod
 */
class InitializationMethod {
public:
  virtual mat getLabelMatrix(const mat& X, const uword k) const = 0;
};


}

#endif
