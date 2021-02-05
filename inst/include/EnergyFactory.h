// kkgroups.h
#ifndef _KBC_ENERGY_FACTORY_H // include guard
#define _KBC_ENERGY_FACTORY_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <stdlib.h>
#include "Kernel.h"
#include "EnergyKernel.h"
#include "KernelFactory.h"

using namespace arma;
using namespace kkg;
using namespace std;



namespace kbc {

/**
 * Class EnergyFactory
 */
class EnergyFactory : public KernelFactory {
public:
  Kernel *createKernel() const override;
  Kernel *createKernel(const double d) const override;
  Kernel *createKernel(const double d, const vec &v) const override;
  Kernel *createKernel(const mat &m) const override;
  Kernel *createKernel(const mat &m, const double d) const override;
};

Kernel *EnergyFactory::createKernel() const {
  return new EnergyKernel();
}

Kernel *EnergyFactory::createKernel(const double d) const {
  return new EnergyKernel(d);
}

Kernel *EnergyFactory::createKernel(const double d, const vec &v) const {
  return new EnergyKernel(d, v);
}

Kernel *EnergyFactory::createKernel(const mat &m) const {
  return new EnergyKernel();
}

Kernel *EnergyFactory::createKernel(const mat &m, const double d) const {
  return new EnergyKernel(d);
}

}

#endif
