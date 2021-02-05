// kkgroups.h
#ifndef _KERNEL_BI_CLUSTERING_H // include guard
#define _KERNEL_BI_CLUSTERING_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <stdlib.h>
#include <unordered_map>
#include <limits>
#include <math.h>
#include <ctype.h>
#include <math.h>


#include "Kernel.h"
#include "KernelFactory.h"
#include "ClusteringMethod.h"
#include "HartiganClustering.h"
#include "LloydClustering.h"
#include "InitializationMethod.h"
#include "RandomInitialization.h"
#include "KppInitialization.h"


using namespace arma;
using namespace std;
using namespace kkg;


namespace kbc {


struct BiClustering {
  uvec row_labels;
  uvec col_labels;
  uword k;
  uword iterations;
  uword restarts;
  double row_cost;
  double col_cost;
};


/**
 * Class BiClusteringMethod
 */
class BiClusteringMethod {
public:
  // BiClustering doClustering(const double sigma_r, const double sigma_c, const mat &X, const uword k, const uword iterations, const uword restarts, const InitializationMethod &initialization, const KernelFactory &factory);
  BiClustering doClustering(const mat &X, const uword k, const uword iterations, const uword restarts, const InitializationMethod &initialization, const KernelFactory &factory);
protected:
  uvec doOptimization(const string &prefix, const mat &X, const vec &w, const uword k, const uword iterations, const uword restarts, const Kernel &kernel, const uvec &row_labels, const uvec &col_labels);
  // uvec doOptimization(const string & prefix, const mat &X, const vec &w, const uword k, const uword iterations, const uword restarts, const KernelFactory &factory, const double exponent, const uvec &row_labels, const uvec &col_labels);
  virtual void doOptimizationLoop(const vec &w, const uword k, const uword iterations, const field<mat> &WGW, mat &Z, vec &q, vec &s) const = 0;
  virtual ClusteringMethod* getClusteringMethod() const = 0;
private:
  std::unordered_map<string, field<mat>> umap;
  void putWGW(const string key, field<mat> &value);
  field<mat> getWGW(const string key) const;
  void checkParamenters(const mat &X, const uword k, const uword iterations, const uword restarts) const;
  double getWithinDispersion(const mat &X, const uword k, const uvec &cluster) const;
  double getWithinDispersion(const string prefix, const vec &w, const mat &X, const uword k, const Kernel &kernel, const uvec &cluster1, const uvec &cluster2);
  // double getWithinDispersion(const string prefix, const KernelFactory &factory, const double kernel_param, const vec &w, const mat &X, const uword k, const uvec &cluster1, const uvec &cluster2);
  double doKernelOptimization(const mat &X, const uword k, const uword iterations, const uword restarts, const InitializationMethod &initialization, const KernelFactory &factory) const;
  bool equals(uvec &a, uvec &b) const;
  string getKey(const string &prefix, const uvec &v) const;
};




void BiClusteringMethod::putWGW(const string key, field<mat> &value) {
  umap.insert(make_pair(key, value));
}

field<mat> BiClusteringMethod::getWGW(const string key) const {
  auto it = umap.find(key);
  if (it == umap.end())
    return field<mat>();
  else
    return it -> second;
}


// BiClustering BiClusteringMethod::doClustering(const double sigma_r, const double sigma_c, const mat &X, const uword k, const uword iterations, const uword restarts, const InitializationMethod &initialization, const KernelFactory &factory) {
BiClustering BiClusteringMethod::doClustering(const mat &X, const uword k, const uword iterations, const uword restarts, const InitializationMethod &initialization, const KernelFactory &factory) {
    checkParamenters(X, k, iterations, restarts);

  umat best_row_labels(X.n_rows, restarts, fill::zeros);
  umat best_col_labels(X.n_cols, restarts, fill::zeros);

  vec best_row_cost(restarts);
  best_row_cost.fill(0);
  // best_row_cost.fill(datum::inf);
  vec best_col_cost(restarts);
  // best_col_cost.fill(datum::inf);

  mat Xt = X.t();
  vec row_w = ones<vec>(X.n_rows);
  vec col_w = ones<vec>(X.n_cols);

  // HartiganMethod method;
  // double row_kernel_param = doKernelOptimization(X, k, 20, 20, initialization, factory);
  // double col_kernel_param = doKernelOptimization(Xt, k, 20, 20, initialization, factory);
  double row_kernel_param = 1;
  double col_kernel_param = 1;

  Kernel *row_kernel = factory.createKernel(X, row_kernel_param);
  Kernel *col_kernel = factory.createKernel(Xt, col_kernel_param);
  // cout << "best row_kernel_param: " << row_kernel_param << endl;
  // cout << "best col_kernel_param: " << col_kernel_param << endl;

#pragma omp parallel for
  for (uword i = 0; i < restarts; i++) {
    // cout << "Restart: " << i << endl << flush;

    // Perform kernel k-groups clustering on rows
    // Kernel *row_kernel = factory.createKernel(X, row_kernel_param);
    // Kernel *row_kernel = factory.createKernel(sigma_r);
    // RBFKernel row_kernel(X, row_kernel_param);
    Clustering result = getClusteringMethod() -> doClustering(X, row_w, k, iterations, restarts, initialization, *row_kernel);
    uvec row_labels = result.cluster;
    // cout << "Initial row labels: " << ss(row_labels) << endl;


    // Perform kernel k-groups clustering on columns
    // Kernel *col_kernel = factory.createKernel(Xt, col_kernel_param);
    // Kernel *col_kernel = factory.createKernel(sigma_c);
    // RBFKernel col_kernel(Xt, col_kernel_param);
    result = getClusteringMethod() -> doClustering(Xt, col_w, k, iterations, restarts, initialization, *col_kernel);
    uvec col_labels = result.cluster;
    // cout << "Initial col labels: " << ss(col_labels) << endl;


    uvec old_row_labels = row_labels;
    uvec old_col_labels = col_labels;
    for (uword iter = 0; iter < iterations; iter++) {
      // cout << "Iteration: " << iter << endl;
      // With fixed columns
      // row_labels = doOptimization(sigma_r, "row", X, row_w, k, iterations, restarts, factory, row_kernel_param, row_labels, col_labels);
      row_labels = doOptimization("row", X, row_w, k, iterations, restarts, *row_kernel, row_labels, col_labels);
      // double row_cost = getWithinDispersion(X, k, row_labels, col_labels);
      // cout << "Row labels: " << ss(row_labels) << " --- " << ss(col_labels) << endl;


      // With fixed rows
      // col_labels = doOptimization(sigma_c, "col", Xt, col_w, k, iterations, restarts, factory, col_kernel_param, col_labels, row_labels);
      col_labels = doOptimization("col", Xt, col_w, k, iterations, restarts, *col_kernel, col_labels, row_labels);
      // double col_cost = getWithinDispersion(Xt, k, col_labels, row_labels);
      // cout << "Col labels: " << ss(col_labels) << " --- " << ss(row_labels) << endl;

      double row_cost = getWithinDispersion("row", row_w, X, k, *row_kernel, row_labels, col_labels);
      // cout << "Cost: " << row_cost << endl;

      // cout << "Cost: (" << row_cost << ", " << col_cost << ") vs (" << best_row_cost << ", " << best_col_cost << ")"  << endl;
      if (row_cost >= best_row_cost(i)) {
        // if (row_cost <= best_row_cost(i) && col_cost <= best_col_cost(i)) {
        best_row_cost(i) = row_cost;
        // best_col_cost(i) = col_cost;
        best_row_labels.col(i) = row_labels;
        best_col_labels.col(i) = col_labels;
      }

      if (equals(old_row_labels,row_labels) && equals(old_col_labels, col_labels))
        break;

      old_row_labels = row_labels;
      old_col_labels = col_labels;
    }
  }

  // uword index = best_col_cost.index_min();
  uword index = best_row_cost.index_max();

  BiClustering clustering;
  clustering.row_labels = best_row_labels.col(index);
  clustering.col_labels = best_col_labels.col(index);
  clustering.k = k;
  clustering.iterations = iterations;
  clustering.restarts = restarts;
  clustering.row_cost = best_row_cost(index);
  // clustering.col_cost = best_col_cost(index);
  return clustering;
}

string BiClusteringMethod::getKey(const string &prefix, const uvec &v) const {
  stringstream key;
  key << prefix;
  for (uword i = 0; i < v.n_elem; i++)
    key << v(i) << "-";
  return key.str();
}



uvec BiClusteringMethod::doOptimization(const string &prefix, const mat &X, const vec &w, const uword k, const uword iterations, const uword restarts, const Kernel &kernel, const uvec &row_labels, const uvec &col_labels) {
  // cout << "** Optimization with fixed column ***" << endl;
  // uvec rl = row_labels;
  // cout << "** row_labels: " << ss(rl) << endl;
  // uvec cl = col_labels;
  // cout << "** col_labels: " << ss(cl) << endl;

  string key = getKey(prefix, col_labels);
  field<mat> WGW = getWGW(key);
  if (WGW.is_empty()) {
    field<mat> G(k);
    for (uword i = 0; i < k; i++) {
      // Get column indices
      uvec columns = find(col_labels == i);
      // Create a kernel for the specified columns
      // Kernel *kernel = factory.createKernel(sigma);
      // Kernel *kernel = factory.createKernel(X.cols(columns), kernel_param);
      // RBFKernel kernel(X.cols(columns), exponent);
      // Get the gram matrix for this subset of columns
      G(i) = kernel.getKernelMatrix(X.cols(columns) / sqrt(columns.n_elem));
    }

    WGW = field<mat>(k);
    for (uword i = 0; i < k; i++) {
      WGW(i) = diagmat(w) * G(i) * diagmat(w);
    }
    putWGW(key, WGW);
  }

  // cout << "Gram: \n" << WGW << endl;

  mat Z(X.n_rows, k, fill::zeros);
  for (uword i = 0; i < X.n_rows; i++) {
    Z(i, row_labels(i)) = 1;
  }

  // Initialize q() and s()
  vec q(k, fill::zeros); // Def 41
  vec s(k, fill::zeros); // Def 14
  for (uword j = 0; j < k; j++) {
    uvec idx = find(Z.col(j) == 1);
    q(j) = accu(WGW(j).rows(idx) * Z.col(j));
    s(j) = accu(w(idx));
  }


  doOptimizationLoop(w, k, iterations, WGW, Z, q, s);

  uvec cluster(X.n_rows);
// #pragma omp parallel for shared(Z)
  for (uword i = 0; i < X.n_rows; i++) {
    cluster(i) = index_max(Z.row(i));
  }

  //     Clustering result;
  //     result.cluster = cluster;
  //     result.k = k;
  //     result.iterations = iterations;
  //     result.restarts = restarts;
  //     result.cost = sum(q / s);
  return cluster;
}


void BiClusteringMethod::checkParamenters(const mat &X, const uword k, const uword iterations, const uword restarts) const {
  if (X.is_empty())
    throw invalid_argument("Parameter X is undefined.");
  if (k < 1)
    throw invalid_argument("k must be positive.");
  if (iterations == 0)
    throw invalid_argument("iterations must be positive.");
  if (restarts == 0)
    throw invalid_argument("restarts must be positive.");
}

double BiClusteringMethod::getWithinDispersion(const mat &X, const uword k, const uvec &cluster) const {
  vec clusters_cost(k);
// #pragma omp parallel for shared(clusters_cost)
  for (uword j = 0; j < k; j++) {
    uvec idx = find(cluster == j);
    if (idx.is_empty()) {
      clusters_cost(j) = std::numeric_limits<double>::infinity();
    } else {
      double sum = 0;
      // #pragma omp parallel for reduction(+:sum) collapse(2) shared(idx)
      for (uword i1 = 0; i1 < idx.n_elem; i1++) {
        for (uword i2 = 0; i2 < idx.n_elem; i2++) {
          sum += norm(X.row(idx(i1)) - X.row(idx(i2)), 2);
          // sum += pow(norm(X.row(idx(i1)) - X.row(idx(i2)), 2), 2);
        }
      }
      clusters_cost(j) = sum / (idx.n_elem);
    }
  }
  return sum(clusters_cost);
}

// double BiClusteringMethod::getWithinDispersion(const string prefix, const Kernel &kernel, const vec &w, const mat &X, const uword k, const uvec &cluster1, const uvec &cluster2) {
double BiClusteringMethod::getWithinDispersion(const string prefix, const vec &w, const mat &X, const uword k, const Kernel &kernel, const uvec &cluster1, const uvec &cluster2) {
    vec clusters_cost(k);

  string key = getKey(prefix, cluster2);
  field<mat> WGW = getWGW(key);

  if (WGW.is_empty()) {
    field<mat> G(k);
    for (uword i = 0; i < k; i++) {
      // Get column indices
      uvec columns = find(cluster2 == i);
      // Create a kernel for the specified columns
      // Kernel *kernel = factory.createKernel(sigma);
      // Kernel *kernel = factory.createKernel(X.cols(columns), kernel_param);
      // RBFKernel kernel(X.cols(columns), exponent);
      // Get the gram matrix for this subset of columns
      G(i) = kernel.getKernelMatrix(X.cols(columns) / sqrt(columns.n_elem));
    }

    WGW = field<mat>(k);
    for (uword i = 0; i < k; i++) {
      WGW(i) = diagmat(w) * G(i) * diagmat(w);
    }
    putWGW(key, WGW);
  }

  // #pragma omp parallel for shared(clusters_cost)
  for (uword j = 0; j < k; j++) {
    uvec idx1 = find(cluster1 == j);
    uvec idx2 = find(cluster2 == j);
    if (idx1.is_empty()) {
      clusters_cost(j) = std::numeric_limits<double>::infinity();
    } else {
      if (idx2.is_empty()) {
        clusters_cost(j) = std::numeric_limits<double>::infinity();
      } else {
        double sum = 0;
        // #pragma omp parallel for reduction(+:sum) collapse(2) shared(idx1, idx2)
        for (uword i1 = 0; i1 < idx1.n_elem; i1++) {
          for (uword i2 = 0; i2 < idx1.n_elem; i2++) {
            // rowvec rv1 = X.row(idx1(i1));
            // rowvec rv2 = X.row(idx1(i2));
            sum += WGW(j)(idx1(i1), idx1(i2));
            // sum += norm((rv1.cols(idx2) - rv2.cols(idx2))/sqrt(idx2.n_elem), 2);
            // sum += pow(norm((rv1.cols(idx2) - rv2.cols(idx2))/sqrt(idx2.n_elem), 2), 2);
          }
        }
        clusters_cost(j) = sum / (idx1.n_elem);
      }
    }
  }
  return sum(clusters_cost);
}

double BiClusteringMethod::doKernelOptimization(const mat &X, const uword k, const uword iterations, const uword restarts, const InitializationMethod &initialization, const KernelFactory &factory) const {
  uword n = 20;
  double step = 0.1;
  vec exponent_cost(n);
  exponent_cost.fill(datum::inf);
  const vec w(X.n_rows, fill::ones);
#pragma omp parallel for
  for (uword j=1; j<n; j++) {
    double i = j * step;
    // i=1.9;
    Kernel *kernel = factory.createKernel(X, i);
    // RBFKernel kernel(X, i);
    Clustering result = getClusteringMethod() -> doClustering(X, w, k, iterations, restarts, initialization, *kernel);
    exponent_cost(j) = normalize_cost(getWithinDispersion(X, k, result.cluster));
    // cout << "Exponent: " << i << ": " << exponent_cost(i) << " vs " << result.cost << endl << flush;
  }

  // for (uword i = 1; i < n; i++)
  // cout << "Exponent: " << i * step << ": " << exponent_cost(i) << endl << flush;

  uword imax = exponent_cost.index_min();
  return imax * step;
}

bool BiClusteringMethod::equals(uvec &a, uvec &b) const {
  bool flag = true;
  for (uword i=0; i < a.n_elem; i++) {
    if (a(i) != b(i))
      return false;
  }
  return flag;
}

}

#endif
