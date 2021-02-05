// kkgroups.h
#ifndef _KKG_CLUSTERING_METHOD_H // include guard
#define _KKG_CLUSTERING_METHOD_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

#include <stdlib.h>
#include <math.h>
#include <ctype.h>

#include "Kernel.h"
#include "RBFKernel.h"
#include "EnergyKernel.h"
#include "InitializationMethod.h"

using namespace arma;
using namespace std;



namespace kkg {

struct Clustering {
  uvec cluster;
  uword k;
  uword iterations;
  uword restarts;
  double cost;
};


template <class T>
inline void pp(T X) {
  for (uword i = 0; i < X.n_elem; i++)
    cout << X(i) << " ";
  cout << endl;
}

template<typename T>
inline string ss(Col<T> &X) {
  stringstream sstm;
  for (uword i = 0; i < X.n_elem; i++)
    sstm << X(i) << " ";
  return sstm.str();
  // string str = "";
  // for (uword i = 0; i < X.n_elem; i++)
  //   str = str.append(std::to_string(X(i))).append(" ");
  // return str;
}

template<typename T>
inline string ss(Row<T> &X) {
  stringstream sstm;
  for (uword i = 0; i < X.n_elem; i++)
    sstm << X(i) << " ";
  return sstm.str();
  // string str = "";
  // for (uword i = 0; i < X.n_elem; i++)
  //   str = str.append(std::to_string(X(i))).append(" ");
  // return str;
}


inline double normalize_cost(double cost) {
  if (isinf(cost) || isnan(cost)) {
    return 0;
    // return -DBL_MAX;
  }
  return cost;
}


/**
 * Class Method
 */
class ClusteringMethod {

public:
  Clustering doClustering(const mat& X, const vec& w, const uword k, const uword iterations, const uword restarts, const InitializationMethod& initialization, const Kernel& kernel) const;
protected:
  virtual void doOptimizationLoop(const mat& X, const vec& w, const uword k, const uword iterations, const mat& WGW, mat& Z, vec& q, vec& s) const = 0;
private:
  void checkParamenters(mat X, vec w, uword k, uword iterations, uword restarts) const;

};





void ClusteringMethod::checkParamenters(mat X, vec w, uword k, uword iterations, uword restarts) const {
  if (X.is_empty())
    throw invalid_argument("Parameter X is undefined.");
  if (X.n_rows != w.n_elem)
    throw invalid_argument("w length and X number of rows must be the same.");
  if (k == 0)
    throw invalid_argument( "k must be greater than zero" );
  if (iterations == 0)
    throw invalid_argument( "iterations must be greater than zero" );
  if (restarts == 0)
    throw invalid_argument( "restarts must be greater than zero" );
}




Clustering ClusteringMethod::doClustering(const mat& X, const vec& w, const uword k, const uword iterations, const uword restarts,
                    const InitializationMethod& initialization, const Kernel& kernel) const {

  checkParamenters(X, w, k, iterations, restarts);

  umat clusters(X.n_rows, restarts, fill::zeros);
  vec costs(restarts, fill::zeros);

  const mat WGW = diagmat(w) * kernel.getKernelMatrix(X) * diagmat(w);

  // mat G = kernel.getKernelMatrix(X); // Def 19
  // mat WGW = diagmat(w) * G * diagmat(w);
  // cout << "########################" << endl;
  // cout << "G: \n" << G << endl;
  // cout << "########################" << endl;
  // cout << "WGW: \n" << WGW << endl;
  // cout << "########################" << endl;

#pragma omp parallel for
  for (uword i = 0; i < restarts; i++) {
    // cout << "Restart " << i << endl;
    mat Z = initialization.getLabelMatrix(X, k); // Def 20

    // Initialize q() and s()
    vec q(k, fill::zeros); // Def 41
    vec s(k, fill::zeros); // Def 14
    for (uword j = 0; j < k; j++) {
      uvec idx = find(Z.col(j) == 1);
      q(j) = accu(WGW.rows(idx) * Z.col(j));
      s(j) = accu(w(idx));
    }

    doOptimizationLoop(X, w, k, iterations, WGW, Z, q, s);

    uvec cluster(X.n_rows);
#pragma omp parallel for shared(Z)
    for (uword i = 0; i < X.n_rows; i++) {
      cluster(i) = index_max(Z.row(i));
    }

    clusters.col(i) = cluster;
    costs(i) = normalize_cost(sum(q / s));
  }

  uword index = costs.index_max();

  Clustering result;
  result.cluster = clusters.col(index);
  result.cost = costs(index);
  result.k = k;
  result.iterations = iterations;
  result.restarts = restarts;

  return result;

}




}

#endif
