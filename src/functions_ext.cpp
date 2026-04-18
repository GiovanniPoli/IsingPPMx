#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
double log_beta( double x1, double x2){
  return std::lgamma(x1) + std::lgamma(x2) - std::lgamma(x1 + x2) ;
}

// [[Rcpp::export]]
double log_gX( const int n, const arma::rowvec & x,  const arma::rowvec & Sx) {
  int q = x.n_elem ;

  arma::rowvec alpha  = 1.0 / q +     Sx ;
  arma::rowvec beta   = 1.0 / q + n - Sx ;

  double ret =  arma::accu( q * std::lgamma(2.0) +
                            arma::lgamma(x + alpha) +
                            arma::lgamma(1.0 - x + beta) -
                            arma::lgamma(1.0 + alpha + beta) -
                            arma::lgamma(1.0 + x) -
                            arma::lgamma(2.0 - x) +
                            arma::lgamma(alpha + beta) -
                            arma::lgamma(alpha) -
                            arma::lgamma(beta) ) ;
  return ret;
}

// [[Rcpp::export]]
double empty_log_gX(const arma::rowvec & x) {
  int q = x.n_elem ;

  arma::rowvec a(q) ;
  a.fill(1.0 / q) ;

  double ret =  arma::accu( q *  std::lgamma( 2.0 ) +
                            arma::lgamma( x + a ) +
                            arma::lgamma( 1.0 - x + a ) -
                            arma::lgamma( 1.0 + 2*a )  -
                            arma::lgamma( 1.0 + x ) -
                            arma::lgamma( 2.0 - x ) +
                            arma::lgamma( 2 * a ) -
                            2.0 * arma::lgamma( a ) )  ;
  return ret;
}

// [[Rcpp::export]]
double log_cohesion( const double nh, const double sigma){
  return std::log( nh - sigma) ;
}

// [[Rcpp::export]]
double empty_log_cohesion( const double M, const double H, const double sigma){
  return std::log( M + H * sigma);
}

// [[Rcpp::export]]
double log_choose_std(int n, int k) {
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

// [[Rcpp::export]]
double log_gammas_mid_1( int g1, int g2, double c){
  return log_choose_std( 2, g1 + g2) +
    log_beta( g1 + g2 + 1/c, 2 - g1 - g2 + 1/c ) -
    log_beta( 1/c, 1/c);
}

// [[Rcpp::export]]
arma::uvec cpp_random_pair_fixed_j(arma::uword dim, arma::uword j) {
  arma::uword i = arma::randi<int>(arma::distr_param(0, dim - 2));
  i += (i >= j);
  arma::uvec pair = {i, j};
  return pair;
}
