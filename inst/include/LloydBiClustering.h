// kkgroups.h
#ifndef _KERNEL_LLOYD_BI_CLUSTERING_H // include guard
#define _KERNEL_LLOYD_BI_CLUSTERING_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <stdlib.h>
#include <limits>
#include <math.h>
#include <ctype.h>
#include <math.h>


#include "Kernel.h"
#include "RBFKernel.h"
#include "EnergyKernel.h"
#include "ClusteringMethod.h"
#include "HartiganClustering.h"
#include "LloydClustering.h"
#include "InitializationMethod.h"
#include "RandomInitialization.h"
#include "KppInitialization.h"
#include "BiClustering.h"

using namespace arma;
using namespace std;
using namespace kkg;


namespace kbc {



/**
 * Class LloydBiClustering
 */
class LloydBiClustering : public BiClusteringMethod {
private:
  ClusteringMethod* clustering_method = new LloydClustering();
public:
  ClusteringMethod* getClusteringMethod() const override;
  void doOptimizationLoop(const vec& w, const uword k, const uword iterations, const field<mat>& WGW, mat& Z, vec& q, vec& s) const override;
};


ClusteringMethod* LloydBiClustering::getClusteringMethod() const {
  return clustering_method;
}

void LloydBiClustering::doOptimizationLoop(const vec& w, const uword k, const uword iterations, const field<mat>& WGW, mat& Z, vec& q, vec& s) const {
  for (uword iter = 0; iter < iterations; iter++) {
    bool flag = 1;
    for (uword i = 0; i < Z.n_rows; i++) {
      uword j = Z.row(i).index_max();

      vec j_stars(k);
      // j_stars.fill(DBL_MAX);
      for (uword l = 0; l < k; l++) {
        j_stars[l] = 1 / (s(l) * s(l)) * q(l) - 2 / s(l) * dot(WGW(l).row(i), Z.col(l));
      }
      uword j_star = j_stars.index_min();

      if (j != j_star) {
        // cout << "Move " << i << " from C" << j << " to C" << j_star << endl;
        flag = 0;
        q(j)         = q(j) - 2 * dot(WGW(j).row(i), Z.col(j));
        Z(i, j)      = 0;
        Z(i, j_star) = 1;
        s(j)         = s(j) - w(i);
        s(j_star)    = s(j_star) + w(i);
        q(j_star)    = q(j_star) + 2 * dot(WGW(j).row(i), Z.col(j_star));
      }
    }
    if (flag) break;
  }
}

}

#endif
