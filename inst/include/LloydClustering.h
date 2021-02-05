// kkgroups.h
#ifndef _KKG_LLOYD_CLUSTERING_H // include guard
#define _KKG_LLOYD_CLUSTERING_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

#include <stdlib.h>
#include <math.h>
#include <ctype.h>
// #include <time.h>

#include "Kernel.h"
#include "RBFKernel.h"
#include "EnergyKernel.h"
#include "InitializationMethod.h"
#include "RandomInitialization.h"
#include "KppInitialization.h"
#include "ClusteringMethod.h"

using namespace arma;
using namespace std;



namespace kkg {



/**
 * Class LloydClustering
 */
class LloydClustering : public ClusteringMethod {
public:
  void doOptimizationLoop(const mat& X, const vec& w, const uword k, const uword iterations, const mat& WGW, mat& Z, vec& q, vec& s) const override;
};





/** Weighted version of kernel k-means algorithm based on Lloyd method to find local solutions to the optimization problem.
 * @param X The data points matrix.
 * @param w A vector of weights associated to data points.
 * @param k The number of clusters.
 * @param iterations The number of iterations.
 * @param restarts The number of re-starts.
 * @param G The Gram matrix.
 * @param Z The label matrix.
 */
void LloydClustering::doOptimizationLoop(const mat& X, const vec& w, const uword k, const uword iterations, const mat& WGW, mat& Z, vec& q, vec& s) const {
  for (uword iter = 0; iter < iterations; iter++) {
    bool flag = 1;
    for (uword i = 0; i < X.n_rows; i++) {
      uword j = Z.row(i).index_max();

      vec j_stars(k);
      // j_stars.fill(DBL_MAX);
      j_stars.fill(std::numeric_limits<double>::infinity());
      for (uword l = 0; l < k; l++) {
        j_stars[l] = normalize_cost(1 / (s(l) * s(l)) * q(l) - 2 / s(l) * dot(WGW.row(i), Z.col(l)));
      }
      uword j_star = j_stars.index_min();

      if (j != j_star) {
        flag = 0;
        q(j)         = q(j) - 2 * dot(WGW.row(i), Z.col(j));
        s(j)         = s(j) - w(i);
        Z(i, j)      = 0;
        Z(i, j_star) = 1;
        s(j_star)    = s(j_star) + w(i);
        q(j_star)    = q(j_star) + 2 * dot(WGW.row(i), Z.col(j_star));
      }
    }
    if (flag) break;
  }
}




}

#endif
