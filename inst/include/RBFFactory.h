// kkgroups.h
#ifndef _KBC_RBF_FACTORY_H // include guard
#define _KBC_RBF_FACTORY_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <stdlib.h>
#include "Kernel.h"
#include "RBFKernel.h"
#include "KernelFactory.h"

using namespace arma;
using namespace kkg;
using namespace std;



namespace kbc {

/**
 * Class RBFFactory
 */
class RBFFactory : public KernelFactory {
public:
  Kernel *createKernel() const override;
  Kernel *createKernel(const double d) const override;
  Kernel *createKernel(const double d, const vec &v) const override;
  Kernel *createKernel(const mat &m) const override;
  Kernel *createKernel(const mat &m, const double d) const override;
};

Kernel *RBFFactory::createKernel() const {
  return new RBFKernel();
}

Kernel *RBFFactory::createKernel(const double d) const {
  return new RBFKernel(d);
}

Kernel *RBFFactory::createKernel(const double d, const vec &v) const {
  throw invalid_argument("Unsupported method for RBF kernel" );
}

Kernel *RBFFactory::createKernel(const mat &m) const {
  return new RBFKernel(m);
}

Kernel *RBFFactory::createKernel(const mat &m, const double d) const {
  return new RBFKernel(m, d);
}




}

#endif
