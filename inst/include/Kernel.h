// kkgroups.h
#ifndef _KKG_KERNEL_H // include guard
#define _KKG_KERNEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

using namespace arma;
using namespace std;



namespace kkg {

/**
 * Class Kernel
 */
class Kernel {
public:
  virtual mat getKernelMatrix(const mat &X) const = 0;
};


}

#endif
