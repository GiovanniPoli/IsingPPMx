#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "Ising.h"
#include "randomgen.h"
#include "utils.h"
#include "MCMC.h"
#include "update_steps.h"
#include "struct.h"

// [[Rcpp::export]]
Rcpp::NumericVector rpg( const arma::colvec z, const int trunc = 200 ) {
  arma::colvec w =  cpp_polyagamma_h1_truncated(z,trunc);
  return Rcpp::NumericVector(w.begin(), w.end());
}


// [[Rcpp::export]]
int cpp_find_idx_j(const arma::uvec & v, int j) {
  for ( int i = (v.n_elem-1); i > 0; --i) {
    if (v[i] == j) {
      return i;
    }
  }
  throw std::runtime_error("index lost during the mapping");
}

// Helper: convert a cluster_parameter into a R list.
static Rcpp::List cluster_to_list(const cluster_parameter & cp) {

  arma::uvec ones_r  = cp.ones.is_empty()  ? arma::uvec() : cp.ones  + 1;
  arma::uvec zeros_r = cp.zeros.is_empty() ? arma::uvec() : cp.zeros + 1;

  Rcpp::List mapping_r(cp.mapping.size());
  for (std::size_t r = 0; r < cp.mapping.size(); ++r) {
    arma::uvec tmp = cp.mapping[r];
    if (!tmp.is_empty()) tmp += 1;
    mapping_r[r] = tmp;
  }

  return Rcpp::List::create(
    Rcpp::Named("Beta")    = cp.Beta,
    Rcpp::Named("alpha")   = cp.alpha,
    Rcpp::Named("ones")    = ones_r,
    Rcpp::Named("zeros")   = zeros_r,
    Rcpp::Named("mapping") = mapping_r
  );
}

// Wrapper 1 : default constructor (no arguments)
// [[Rcpp::export]]
Rcpp::List cp_default_wrapper() {
  cluster_parameter cp;
  return cluster_to_list(cp);
}

// Wrapper 2 : empty-graph constructor
// [[Rcpp::export]]
Rcpp::List cp_empty_wrapper(const arma::uword dim) {
  cluster_parameter cp(dim);
  return cluster_to_list(cp);
}

// Wrapper 3 : prior-sampling constructor
// [[Rcpp::export]]
Rcpp::List cp_prior_wrapper(const arma::uword    dim,
                      const arma::colvec & Qx,
                      const double         sd_diag,
                      const double         sd_offdiag,
                      const double         rho) {
  cluster_parameter cp(dim, Qx, sd_diag, sd_offdiag, rho);
  return cluster_to_list(cp);
}

// Wrapper 4 : test_update_alpha_and_beta
// [[Rcpp::export]]
Rcpp::List test_update_alpha_and_beta(
                            const arma::mat    & YY,
                            arma::mat    & BETA,
                            arma::colvec & alpha,
                            const arma::uvec   & mapping_vector,
                            const double sd_offdiag,
                            const double sd_diag,
                            const double rho) {

  cpp_update_beta_and_alpha( YY, BETA, alpha, mapping_vector,
                             sd_offdiag * sd_offdiag,
                             sd_diag    * sd_diag,
                             rho );

  return Rcpp::List::create(
      Rcpp::Named("Beta")    = BETA,
      Rcpp::Named("alpha")   = alpha);
}
