#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "randomgen.h"
#include "utils.h"

using namespace Rcpp ;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]


// [[Rcpp::export]]
double node_wise_pseudo_ll( const arma::colvec & y_it, arma::mat Omega) {

  arma::colvec diag = Omega.diag() ;
  Omega.diag().zeros() ;
  arma::colvec phi = diag + Omega * y_it ;

  return arma::sum( y_it % phi -  arma::log( 1 + arma::exp(phi) ) ) ;
}

// [[Rcpp::export]]
double node_wise_generalized_ll( const arma::colvec y_it,
                                 arma::mat Omega, const int a) {
  arma::colvec diag = Omega.diag() ;
  Omega.diag().zeros() ;
  arma::colvec phi = diag + Omega * y_it ;
  arma::uvec nonzero_rows = arma::sum(Omega, 1) != 0;
  return arma::sum( y_it % phi -  arma::log( 1 + arma::exp(phi) ) ) ;
}

// [[Rcpp::export]]
double log_likelihood_ratio_flip( const arma::colvec & tilde_y,
                                  const arma::mat    & tilde_X,
                                  const arma::colvec & tilde_beta,
                                  const double new_beta, const int node) {

  arma::colvec tilde_xp  = tilde_X.col(node) ;
  arma::colvec XB        = tilde_X * tilde_beta ;
  arma::colvec delta_vec = new_beta * tilde_xp ;
  arma::colvec XB_new    = XB + delta_vec ;                    // always x_ip (beta - 0)


  return arma::accu( tilde_y % delta_vec -
                     arma::log1p(arma::exp(XB_new)) +
                     arma::log1p(arma::exp(XB)) ) ;   //log(1 + ex)
}


// [[Rcpp::export]]
double log_likelihood_ratio_swap( const arma::colvec & tilde_y,
                                  const arma::mat    & tilde_X,
                                  const arma::colvec & tilde_beta,
                                  const double new_beta,
                                  const int old_node_1,
                                  const int new_node_1 ) {

  arma::colvec tilde_xp_old = tilde_X.col(old_node_1) ;
  arma::colvec tilde_xp_new = tilde_X.col(new_node_1) ;

  const double old_beta = tilde_beta(old_node_1) ;

  arma::colvec XB        = tilde_X * tilde_beta ;
  arma::colvec delta_vec = new_beta * tilde_xp_new - old_beta  * tilde_xp_old ;
  arma::colvec XB_new    = XB + delta_vec ;

  return arma::accu( tilde_y % delta_vec - arma::log1p(arma::exp(XB_new)) + arma::log1p(arma::exp(XB)) ) ;   //log(1 + ex)
}

// [[Rcpp::export]]
double log_likelihood_ratio_swap_v2( const arma::colvec & tilde_y,
                                     const arma::mat    & tilde_X,
                                     const arma::colvec & tilde_beta,
                                     const double new_beta,
                                     const int    old_one,
                                     const int    new_one ) {

  arma::colvec new_tilde_beta = tilde_beta ;
  new_tilde_beta(new_one) = new_beta ;
  new_tilde_beta(old_one) = 0.0 ;


  arma::colvec XB        = tilde_X * tilde_beta ;
  arma::colvec XB_new    = tilde_X * new_tilde_beta ;

  Rcpp::Rcout << tilde_beta << std::endl ;
  Rcpp::Rcout << new_tilde_beta << std::endl ;

  double s1 = arma::accu( tilde_y %  (XB_new - XB ) )  ;
  double s2 = arma::accu( arma::log1p(arma::exp(XB_new)) ) ;
  double s3 = arma::accu( arma::log1p(arma::exp(XB)) ) ;

  Rcpp::Rcout << s1 << std::endl ;
  Rcpp::Rcout << s2 << std::endl ;
  Rcpp::Rcout << s3 << std::endl ;

  return arma::accu( tilde_y % XB_new - arma::log1p(arma::exp(XB_new)) -
                     tilde_y % XB     + arma::log1p(arma::exp(XB)) ) ;   //log(1 + ex)
}




