#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "Ising.h"
#include "randomgen.h"
#include "utils.h"
#include "MCMC.h"

// [[Rcpp::export]]
Rcpp::NumericVector rpg(const arma::colvec z, const int trunc = 200) {
  arma::colvec w =  cpp_polyagamma_h1_truncated(z,trunc);
  return Rcpp::NumericVector(w.begin(), w.end()); ;
}

// [[Rcpp::export]]
void w_variable_selection_step( arma::mat & gamma, arma::mat & omega, const int p,
                                const arma::colvec & y,  const arma::mat & X,
                                      const double var_int, const double var_coef,
                                      const double pi_slab ) {
  arma::subview_col<double> tilde_gamma = gamma.col(p) ;
  arma::subview_col<double> tilde_beta  = omega.col(p) ;

  cpp_update_variable_selection( tilde_gamma, tilde_beta, y, X,  pi_slab, var_int ) ;
}

// [[Rcpp::export]]
void w_variable_selection_step_v2( arma::mat & gamma, arma::mat & omega, const int p,
                                       const arma::colvec & y,  const arma::mat & X,
                                       int & edges,
                                       const double var_int, const double var_coef,
                                       const double pi_slab, const double par_pi, const double a, const double b) {
  arma::subview_col<double> tilde_gamma     = gamma.col(p) ;
  arma::subview_col<double> tilde_beta      = omega.col(p) ;
  arma::subview_row<double> tilde_gamma_tr  = gamma.row(p) ;


  cpp_update_variable_selection_v2( p, tilde_gamma, tilde_beta, tilde_gamma_tr, edges,
                                                   y, X, var_int, par_pi, a, b) ;

}

// [[Rcpp::export]]
void w_cpp_update_Omega( const arma::mat & gamma,
                         arma::mat & omega,
                         const int p,
                         const arma::colvec & y,
                         const arma::mat    & X,
                         const double var_slab, const double var_int) {

  arma::subview_col<double> tilde_gamma = gamma.col(p) ;
  arma::subview_col<double> tilde_beta  = omega.col(p) ;

  cpp_update_Omega( p, tilde_gamma, tilde_beta, y, X, var_slab, var_int) ;
}
