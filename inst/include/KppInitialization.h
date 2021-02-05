// kkgroups.h
#ifndef _KKG_KPP_INITIALIZATION_H // include guard
#define _KKG_KPP_INITIALIZATION_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include "InitializationMethod.h"

using namespace arma;
using namespace std;



namespace kkg {



/**
 * Class KppInitialization
 */
class KppInitialization : public InitializationMethod {
public:
  mat getLabelMatrix(const mat& X, const uword k) const override;
private:
  uword  nearest(const uword i, const mat& centroidist, const uword n_cluster) const;
  double nearestDistance(const uword i, const mat& centroidist, const uword n_cluster) const;
  rowvec computeDistances(const uword centroid, const mat& X) const;
};



mat KppInitialization::getLabelMatrix(const mat& X, const uword k) const {
  Row<uword> centroids(k);
  mat centroidist(k, X.n_rows, fill::zeros);
  centroids(0) = rand() % X.n_rows;
  centroidist.row(0) = computeDistances(centroids(0), X);

  // cout << "(0) centroid 0: " << centroids(0) << endl;
  // time_t my_time = time(NULL);
  vec dist(X.n_rows, fill::zeros);
  double sum = 0;
  for (uword i = 1; i < k; i++) {

    // my_time = time(NULL);
    // cout << "Time 1.1: " << my_time << endl << flush;
#pragma omp parallel for reduction(+:sum) shared(dist, centroidist, i)
    for (uword j = 0; j < X.n_rows; j++) {
      dist(j) = nearestDistance(j, centroidist, i);
      sum += dist(j);
      // cout << "(1) nearest distance " << i << ", " << j << ": " << dist(j) << " -- sum: " << sum << endl << flush;
    }
    // my_time = time(NULL);
    // cout << "Time 1.2: " << my_time << endl << flush;
    // cout << "(2) sum: " << sum << endl;
    sum =  sum * rand() / (RAND_MAX - 1.);
    // cout << "(3) sum: " << sum << endl;
    for (uword j = 0; j < X.n_rows; j++) {
      // cout << "(4) sum: " << sum << endl;
      if ((sum -= dist(j)) > 0) continue;
      // cout << "(5) centroid " << i << ": " << j << endl;
      centroids(i) = j;
      centroidist.row(i) = computeDistances(j, X);
      break;
    }
    // cout << "(6) centroids(" << i << "): " << centroids(i) << endl;
  }
  // my_time = time(NULL);
  // cout << "Time 1.3: " << my_time << endl << flush;
  // cout << "(6) centroids: \n" << centroids << endl << flush;
  mat Z(X.n_rows, k, fill::zeros);
#pragma omp parallel for shared(centroidist)
  for (uword i = 0; i < X.n_rows; i++) {
    // cout << "(7) i: " << i << ", " << nearest(i, centroids, X, k) << endl;
    Z(i, nearest(i, centroidist, k)) = 1;
  }

  // my_time = time(NULL);
  // cout << "Time 1.4: " << my_time << endl << flush;
  // cout << "(8)" << endl;
  return Z;
}



uword KppInitialization::nearest(const uword i, const mat& centroidist, const uword n_cluster) const {
  double minD = std::numeric_limits<double>::infinity(), dist;
  int index = 0;
  for (uword j = 0; j < n_cluster; j++) {
    // cout << "(8) distance(X, " << others(j) << ", " << i << "): " << distance(X, others(j), i) << endl;
    if (minD > (dist = centroidist(j, i))) {
      // if (minD > (dist = distance(X, others(j), i))) {
      minD = dist;
      index = j;
    }
  }
  // cout << "(8) index: " << index << endl;
  return index;
}


double KppInitialization::nearestDistance(const uword i, const mat& centroidist, const uword n_cluster) const {
  // cout << "(3) i: " << i << ", n_cluster: " << n_cluster << endl;
  double minD = std::numeric_limits<double>::infinity(), dist;
  for (uword j = 0; j < n_cluster; j++) {
    // cout << "(3) distance(X, " << others(j) << ", " << i << "): " << distance(X, others(j), i) << endl;
    if (minD > (dist = centroidist(j, i)))
      minD = dist;
  }
  return minD;
}


rowvec KppInitialization::computeDistances(const uword centroid, const mat& X) const {
  rowvec row(X.n_rows);
#pragma omp parallel for
  for (uword i = 0; i < X.n_rows; i++) {
    row(i) = norm(X.row(centroid).t() - X.row(i).t(), 2);
  }
  return row;
}




}

#endif
