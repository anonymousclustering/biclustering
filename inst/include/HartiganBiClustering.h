// kkgroups.h
#ifndef _KERNEL_HARTIGAN_BI_CLUSTERING_H // include guard
#define _KERNEL_HARTIGAN_BI_CLUSTERING_H

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
 * Class HartiganBiClustering
 */
class HartiganBiClustering : public BiClusteringMethod {
private:
  ClusteringMethod* clustering_method = new HartiganClustering();
public:
  ClusteringMethod* getClusteringMethod() const override;
  void doOptimizationLoop(const vec& w, const uword k, const uword iterations, const field<mat>& WGW, mat& Z, vec& q, vec& s) const override;
};



ClusteringMethod* HartiganBiClustering::getClusteringMethod() const {
  return clustering_method;
}


void HartiganBiClustering::doOptimizationLoop(const vec& w, const uword k, const uword iterations, const field<mat>& WGW, mat& Z, vec& q, vec& s) const {
  // cout << "**** Loop with fixed column ***" << endl;
  for (uword iter = 0; iter < iterations; iter++) {
    bool flag = 1;
    for (uword i = 0; i < Z.n_rows; i++) {
      uword j = Z.row(i).index_max();

      double tmp = (1 / (s(j) - w(i))) * (w(i) / s(j) * q(j) - 2 * dot(WGW(j).row(i), Z.col(j)) + WGW(j)(i,i));
      // cout << "**** Q" << j << "(x" << i << ")-: " << tmp << endl;
      // cout << "**** " << (1 / (s(j) - w(i))) << " * [" << w(i) / s(j) * q(j) << " - "<< 2 * dot(WGW(j).row(i), Z.col(j)) << " + " << WGW(j)(i,i) << "]"  << endl;
      vec j_stars(k, fill::zeros);
      for (uword l = 0; l < k; l++) {
        if (l != j) {
          // cout << "**** Q" << l << "(x" << i << ")+: " << (1 / (s(l) + w(i))) * (w(i) / s(l) * q(l) - 2 * dot(WGW(l).row(i), Z.col(l)))  << endl;
          // cout << "**** " << - 1 * (1 / (s(l) + w(i))) << " * [" << w(i) / s(l) * q(l) << " - "<< 2 * dot(WGW(l).row(i), Z.col(l)) << " - " << WGW(j)(i,i) << "]"  << endl;
          j_stars[l] =  normalize_cost(tmp - (1 / (s(l) + w(i))) * (w(i) / s(l) * q(l) - 2 * dot(WGW(l).row(i), Z.col(l)) - WGW(j)(i,i)));
        }
      }
      uword j_star = j_stars.index_max();
      // cout << "**** j*  " << ss(j_stars) << endl;

      if (j_stars[j_star] > 0) {
        // cout << "**** Move x" << i << " from C" << j << " to C" << j_star << endl;
        flag = 0;
        q(j)         = q(j) - 2 * dot(WGW(j).row(i), Z.col(j));
        Z(i, j)      = 0;
        Z(i, j_star) = 1;
        s(j)         = s(j) - w(i);
        s(j_star)    = s(j_star) + w(i);
        q(j_star)    = q(j_star) + 2 * dot(WGW(j_star).row(i), Z.col(j_star));
      }
    }
    if (flag) break;
  }
}




}

#endif
