// kkgroups.h
#ifndef _KKG_HARTIGAN_CLUSTERING_H // include guard
#define _KKG_HARTIGAN_CLUSTERING_H

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
 * Class HartiganClustering
 */
class HartiganClustering : public ClusteringMethod {
public:
  void doOptimizationLoop(const mat& X, const vec& w, const uword k, const uword iterations, const mat& WGW, mat& Z, vec& q, vec& s) const override;
};



/** Weighted version of kernel k-means algorithm based on Hartigan method to find local solutions to the optimization problem.
 * @param X The data points matrix.
 * @param w A vector of weights associated to data points.
 * @param k The number of clusters.
 * @param iterations The number of iterations.
 * @param restarts The number of re-starts.
 * @param G The Gram matrix.
 * @param Z The label matrix.
 */
void HartiganClustering::doOptimizationLoop(const mat& X, const vec& w, const uword k, const uword iterations, const mat& WGW, mat& Z, vec& q, vec& s) const {
  // cout << "Z: \n" << Z << endl;
  // cout << "q: \n" << q << endl;
  // cout << "s: \n" << s << endl;

  for (uword iter = 0; iter < iterations; iter++) {
    // cout << "***** Iteration " << iter << endl;
    bool flag = 1;
    for (uword i = 0; i < X.n_rows; i++) {
      uword j = Z.row(i).index_max();
      // cout << "Row " << i << " in cluster " << j << endl;
      vec j_stars(k, fill::zeros);
      double tmp = (1 / (s(j) - w(i))) * (w(i) / s(j) * q(j) - 2 * dot(WGW.row(i), Z.col(j)) + WGW(i,i));
      // #pragma omp parallel for shared(i,j,tmp)
      for (uword l = 0; l < k; l++) {
        if (l != j) {
          //   j_stars[l] = 0;
          // // j_stars[l] = -DBL_MAX;
          // } else {
          j_stars[l] =  normalize_cost(tmp - (1 / (s(l) + w(i))) * (w(i) / s(l) * q(l) - 2 * dot(WGW.row(i), Z.col(l)) - WGW(i,i)));

        }
      }
      uword j_star = j_stars.index_max();

      // cout << "j_stars:" << endl;
      // pp(j_stars);

      if (j_stars[j_star] > 0) {
        // cout << "Mov row " << i << "from C" << j << " to C" << j_star << endl;
        flag = 0;
        q(j)        = q(j) - 2 * dot(WGW.row(i), Z.col(j));
        s(j)        = s(j) - w(i);
        Z(i, j)      = 0;
        Z(i, j_star) = 1;
        q(j_star)   = q(j_star) + 2 * dot(WGW.row(i), Z.col(j_star));
        s(j_star)   = s(j_star) + w(i);
      }
    }
    if (flag) break;
  }
}



}

#endif
