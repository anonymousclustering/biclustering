// kkgroups.h
#ifndef _KBC_KERNEL_FACTORY_H // include guard
#define _KBC_KERNEL_FACTORY_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

#include <stdlib.h>
#include "Kernel.h"

using namespace arma;
using namespace kkg;
using namespace std;



namespace kbc {

/**
 * Class Kernel
 */
class KernelFactory {
public:
  virtual Kernel *createKernel() const = 0;
  virtual Kernel *createKernel(const double d) const = 0;
  virtual Kernel *createKernel(const double d, const vec &v) const = 0;
  virtual Kernel *createKernel(const mat &m) const = 0;
  virtual Kernel *createKernel(const mat &m, const double d) const = 0;
};


}

#endif
