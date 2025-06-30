#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
double add_edge_loss_prob( double pois_par, int n_active, double base_prob) {
  double ret = 1.0;
  double term = std::exp( - pois_par);
    ret -= term;
    for (int i = 1; i <= n_active; ++i) {
      term *= pois_par / i;
      ret  -= term;
      }
  return ret*base_prob;
}

// [[Rcpp::export]]
double cohesion( double pois_par, int n_active) {
  double ret = 1.0;
  return ret;
}

// [[Rcpp::export]]
double gX( double X, int rho) {
  double ret = 1.0;
  return ret;
}
