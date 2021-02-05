#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]

#include "InitializationMethod.h"
#include "RandomInitialization.h"
#include "KppInitialization.h"

#include "KernelFactory.h"
#include "RBFFactory.h"
#include "EnergyFactory.h"

#include "BiClustering.h"
#include "LloydBiClustering.h"
#include "HartiganBiClustering.h"

//' Weighted version of kernel k-means algorithm to find local solutions to the optimization problem.
//'
//' @param X The data points matrix.
//' @param w A vector of weights associated to data points.
//' @param k The number of clusters.
//' @param iterations The number of iterations.
//' @param restarts The number of re-starts.
//' @param init_method The initialization method: "random" or "kpp".
//' @param method_name The optimization method: "lloyd" or "hardigan".
//' @export
// [[Rcpp::export]]
Rcpp::List kernel_biclustering(
    const arma::mat X,
    const arma::uword k = 3,
    const arma::uword iterations = 10,
    const arma::uword restarts = 1,
    const std::string init_method = "random",
    const std::string method_name = "hartigan",
    const std::string kernel_name = "rbf") {


  kkg::InitializationMethod *init;
  if (init_method == "random")
    init = new kkg::RandomInitialization();
  else if (init_method == "kpp")
    init = new kkg::KppInitialization();
  else
    throw invalid_argument("No such initialization method: Use 'random' or 'kpp'" );

  kbc::KernelFactory *factory;
  if (kernel_name == "rbf")
    factory = new kbc::RBFFactory();
  else if (kernel_name == "energy")
    factory = new kbc::EnergyFactory();
  else
    throw invalid_argument("No such kernel method: Use 'rbf' or 'energy'" );

  kbc::BiClusteringMethod *biclustering;
  if (method_name == "hartigan")
    biclustering = new kbc::HartiganBiClustering();
  else if (method_name == "lloyd")
    biclustering = new kbc::LloydBiClustering();
  else
    throw invalid_argument("No such method: Use 'hartigan' or 'lloyd'" );

  // auto started = std::chrono::high_resolution_clock::now();

  // kbc::BiClustering result = biclustering -> doClustering(sigma_r, sigma_c, X, k, iterations, restarts, *init, *factory);
  kbc::BiClustering result = biclustering -> doClustering(X, k, iterations, restarts, *init, *factory);

  // auto done = std::chrono::high_resolution_clock::now();

  // cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms" << endl;
  // cout << "Row labels: " << ss(result.row_labels) << endl;
  // cout << "Col labels: " << ss(result.col_labels) << endl;

  return Rcpp::List::create(
    Rcpp::_["row_labels"] = result.row_labels.t()+1,
    Rcpp::_["col_labels"] = result.col_labels.t()+1,
    Rcpp::_["k"] = k,
    Rcpp::_["iterations"] = iterations,
    Rcpp::_["restarts"] = restarts,
    Rcpp::_["cost"] = result.row_cost
    // Rcpp::_["col_cost"] = result.col_cost
  );
}










