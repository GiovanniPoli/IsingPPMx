#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <truncnorm.h>
#include <mvnorm.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "randomgen.h"
#include "Ising.h"
#include "utils.h"
#include "functions_ext.h"
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat bayes_logit( const arma::colvec & y,  const arma::mat & X,
                       const arma::colvec & b0, const arma::mat & B0,
                       const arma::colvec & bstart,
                       const int sample, const int burn = 0, const int thinning = 1){
  const int N = y.n_elem ;
  const int P = X.n_cols ;

  arma::colvec n(N, arma::fill::ones); ;
  arma::colvec  alpha = (y - .5) % n ;
  arma::mat    Xalpha = X;
  Xalpha.each_col() %= alpha ;
  arma::colvec Z = arma::sum(Xalpha,0).t() ;

  arma::mat    Prec0  = arma::inv(B0) ;
  arma::mat    P0b0   = Prec0 * b0 ;

  arma::colvec w(N, arma::fill::zeros) ;
  arma::colvec beta = bstart ;

  arma::colvec psi ;
  arma::mat Xw(N,P);
  arma::mat Precn(P,P) ;
  arma::mat Varn(P,P)  ;
  arma::colvec Meann(P) ;

  const int S    = burn + thinning*sample;
  int ss = 0;

  MyTimePoint t0 = myClock::now();
  arma::mat BETA(sample, P) ;

  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();

    catIter(s, S, t0) ;
    psi    = X*beta ;
    w      = cpp_polyagamma_h1_devroye(psi) ;
    Xw     = X.each_col()%w ;
    Precn  = X.t()*Xw+Prec0 ;
    Varn   = arma::inv_sympd(Precn) ;
    Meann  = Varn*(Z+P0b0) ;

    beta = cpp_mvrnormArma1( Meann, Varn);

    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      BETA.row(ss) =  beta.t() ;
      ss += 1 ;
    }
    catIter(S, S, t0) ;
  }
  return BETA ;
}

void cpp_update_variable_selection( arma::subview_col<double> tilde_gamma,
                                    arma::subview_col<double> tilde_beta,
                                    const arma::colvec & y,  const arma::mat & X,
                                    const double pi_slab, const double var_slab) {
  // TO DELATE
  int P = tilde_gamma.n_elem;

  arma::vec u       = arma::randu(2);
  double log_ar     = std::log(u(1)) ;
  double log_alpha  = 0.0 ;

  arma::uvec ones ;
  arma::uvec zeros  = arma::find(tilde_gamma == 0) ;

  int node      = arma::as_scalar(arma::randi(arma::distr_param(0, tilde_beta.n_rows-1)));
  int old_value = tilde_gamma(node) ;

  int c1 = static_cast<int>( (u(0) > 0.5) & ((zeros.n_elem != 0) & (zeros.n_elem != P)) );

  int CASE = (c1 << 1) | old_value;

  int node_old ;
  int node_old_index ;
  double beta_proposed ;

  switch (CASE) {
  case 0:
    // "flip 0 -> 1";
    beta_proposed = arma::randn(arma::distr_param(0.0, var_slab)) ;
    log_alpha    += log_likelihood_ratio_flip( y, X, tilde_gamma, beta_proposed, node) ;
    log_alpha    += -.5 * beta_proposed * beta_proposed / var_slab ;
    log_alpha    += std::log(       pi_slab ) ;
    log_alpha    -= std::log( 1.0 - pi_slab ) ;
    if(log_alpha > log_ar){
      tilde_beta(node)  = beta_proposed;
      tilde_gamma(node) = 1 ;
    }
    break;
  case 1:
    // "flip 1 -> 0";
    log_alpha    -= log_likelihood_ratio_flip( y, X, tilde_gamma, tilde_beta(node), node) ;
    log_alpha    -= - 0.5 * tilde_beta(node) * tilde_beta(node) / var_slab ;
    log_alpha    -= std::log(       pi_slab ) ;
    log_alpha    += std::log( 1.0 - pi_slab ) ;
    if(log_alpha > log_ar){
      tilde_beta(node)  = 0.0;
      tilde_gamma(node) = 0 ;
    }
    break;
  case 2:
    // "swap 0 -> 1";
    ones           = arma::find(tilde_gamma) ;
    node_old       = ones( arma::as_scalar(arma::randi(arma::distr_param(0,  ones.n_rows-1))) ) ;
    beta_proposed  =                       arma::randn(arma::distr_param(0.0, var_slab)) ;
    log_alpha     += log_likelihood_ratio_swap( y, X, tilde_beta, beta_proposed,
                                                node_old, node);
    log_alpha     += -.5 *    beta_proposed * beta_proposed    / var_slab ;
    log_alpha     -= -.5 * tilde_beta(node_old) * tilde_beta(node_old) / var_slab ;

    if(log_alpha > log_ar){
      tilde_beta(node)  = beta_proposed ;
      tilde_gamma(node) = 1 ;
      tilde_beta(node_old)  = 0.0 ;
      tilde_gamma(node_old) = 0 ;
    }
    break;
  case 3:
    // "swap 1 -> 0";
    node_old       = zeros(arma::as_scalar(arma::randi(arma::distr_param(0, zeros.n_rows-1))));
    beta_proposed  = arma::randn(arma::distr_param(0.0, var_slab)) ;
    log_alpha     += log_likelihood_ratio_swap( y, X, tilde_beta, beta_proposed,
                                                node, node_old) ;
    log_alpha     += -.5 *    beta_proposed * beta_proposed    / var_slab ;
    log_alpha     -= -.5 * tilde_beta(node) * tilde_beta(node) / var_slab ;

    if(log_alpha > log_ar){
      tilde_beta(node)  = 0.0;
      tilde_gamma(node) = 0 ;
      tilde_beta(node_old)  = beta_proposed;
      tilde_gamma(node_old) = 1 ;
    }
    break;
  default:
    Rcpp::stop("Unexpected value used for variable selection step; it must be 0 or 1.");
  break;
  }
}


void cpp_update_variable_selection_v2( const int p,
                                       arma::subview_col<double> tilde_gamma,
                                       arma::subview_col<double> tilde_beta,
                                       arma::subview_row<double> tilde_gamma_tr,
                                       int & edges,
                                       const arma::colvec & y,  const arma::mat & X,
                                       const double var_slab,
                                       const double par_pi, const double a, const double b) {

  int P = tilde_gamma.n_elem;

  arma::vec u       = arma::randu(2);
  double log_ar     = std::log(u(1)) ;
  double log_alpha  = 0.0 ;

  arma::uvec ones ;
  arma::uvec zeros  = arma::find(tilde_gamma == 0) ;

  int n_1st = arma::as_scalar(arma::randi(arma::distr_param(0, tilde_beta.n_rows-2))); // not p
  if( n_1st >= p) n_1st +=1;

  int    old_value_1st = tilde_gamma(n_1st) ;

  double ref_1st  = tilde_gamma_tr(n_1st) ;
  double sign_1st = 2.0 * static_cast<int>( old_value_1st == ref_1st ) - 1.0;

  int c1 = static_cast<int>( (u(0) > 0.5) & ((zeros.n_elem > 0) & (zeros.n_elem < P - 1 )) );
  int CASE = (c1 << 1) | old_value_1st ;

  int    index_n_2nd ;
  int    n_2nd ;
  double ref_2nd ;
  double beta_proposed ;
  double sign_2nd ;
  double pi_slab = add_edge_loss_prob( a, edges, par_pi ) ;
  Rcpp::Rcout << edges << std::endl ;

  switch (CASE) {
  case 0: // "flip 0 -> 1";
    beta_proposed = arma::randn(arma::distr_param(0.0, var_slab)) ;

    log_alpha    += log_likelihood_ratio_flip( y, X, tilde_gamma, beta_proposed, n_1st ) ;
    log_alpha    += -.5 * beta_proposed * beta_proposed / var_slab ;
    log_alpha    += std::log(       pi_slab ) ;
    log_alpha    -= std::log( 1.0 - pi_slab ) ;
    log_alpha    -= std::log( b ) * sign_1st  ;


    if(log_alpha > log_ar){
      tilde_beta(n_1st)  = beta_proposed;
      tilde_gamma(n_1st) = 1 ;
      edges += 1 ;
    }
    break;
  case 1: // "flip 1 -> 0";

    log_alpha    -= log_likelihood_ratio_flip( y, X, tilde_gamma, tilde_beta(n_1st), n_1st) ;
    log_alpha    -= - 0.5 * tilde_beta(n_1st) * tilde_beta(n_1st) / var_slab ;
    log_alpha    -= std::log(       pi_slab ) ;
    log_alpha    += std::log( 1.0 - pi_slab ) ;
    log_alpha    -= std::log( b ) * sign_1st  ;

    if(log_alpha > log_ar){
      tilde_beta(n_1st)  = 0.0;
      tilde_gamma(n_1st) = 0 ;
      edges -= 1 ;
    }
    break;
  case 2:
    // "swap 0 -> 1";
    beta_proposed  = arma::randn(arma::distr_param(0.0, var_slab)) ;

    ones        = arma::find(tilde_gamma == 1 ) ;
    index_n_2nd = arma::as_scalar(arma::randi(arma::distr_param(0,  ones.n_rows-2))) ;
    n_2nd   = ones( index_n_2nd ) ;
    if( n_2nd == p ) n_2nd   = ones( index_n_2nd + 1 ) ;

    ref_2nd = tilde_gamma_tr( n_2nd ) ;
    sign_2nd = 2.0 * static_cast<int>( tilde_gamma( n_2nd ) == ref_2nd ) - 1.0;


    log_alpha     += log_likelihood_ratio_swap( y, X, tilde_beta, beta_proposed, n_2nd, n_1st);
    log_alpha     += -.5 *    beta_proposed * beta_proposed    / var_slab ;
    log_alpha     -= -.5 * tilde_beta(n_2nd) * tilde_beta(n_2nd) / var_slab ;
    log_alpha     -= std::log( b ) * sign_1st ;
    log_alpha     -= std::log( b ) * sign_2nd ;

    if(log_alpha > log_ar){
      tilde_beta(  n_1st ) = beta_proposed ;
      tilde_gamma( n_1st ) = 1 ;
      tilde_beta(  n_2nd ) = 0.0 ;
      tilde_gamma( n_2nd ) = 0 ;
    }
    break;
  case 3: // "swap 1 -> 0";
    beta_proposed  = arma::randn(arma::distr_param(0.0, var_slab)) ;

    n_2nd   = zeros(arma::as_scalar(arma::randi(arma::distr_param(0, zeros.n_rows-1))));
    ref_2nd = tilde_gamma_tr( n_2nd ) ;
    sign_2nd = 2.0 * static_cast<int>( tilde_gamma( n_2nd ) == ref_2nd ) - 1.0;

    log_alpha     += log_likelihood_ratio_swap( y, X, tilde_beta, beta_proposed, n_1st, n_2nd) ;
    log_alpha     += -.5 *    beta_proposed * beta_proposed    / var_slab ;
    log_alpha     -= -.5 * tilde_beta(n_1st) * tilde_beta(n_1st) / var_slab ;
    log_alpha     -= std::log( b ) * sign_1st ;
    log_alpha     -= std::log( b ) * sign_2nd ;

    if(log_alpha > log_ar){
      tilde_beta(  n_1st ) = 0.0;
      tilde_gamma( n_1st ) = 0 ;
      tilde_beta(  n_2nd ) = beta_proposed;
      tilde_gamma( n_2nd ) = 1 ;
    }
    break;
  default:
    Rcpp::stop("Unexpected value used for variable selection step; it must be 0 or 1.");
  break;
  }
}


void cpp_update_Omega( const int p,
                       const arma::subview_col<double> tilde_gamma,
                       arma::subview_col<double> tilde_beta,
                       const arma::colvec & y,
                       const arma::mat    & X,
                       const double var_slab,
                       const double var_int) {

  const int P_tot = tilde_gamma.n_elem;
  const int N = y.n_elem ;

  arma::colvec new_beta(P_tot, arma::fill::zeros);
  arma::uvec   ones    = arma::find(tilde_gamma == 1) ;
  arma::uvec   int_pos = arma::find( ones == p, 1 );

  const int P = ones.n_elem;

  const arma::mat Xreduced = X.cols(ones);

  arma::colvec  alpha = y - .5;
  arma::mat    Xalpha = Xreduced;
  Xalpha.each_col() %= alpha ;
  arma::colvec Z = arma::sum(Xalpha,0).t() ;

  arma::mat Prec0(P,P, arma::fill::eye ) ;
  Prec0.diag() = Prec0.diag() * 1.0 / var_slab ;
  Prec0( int_pos(0), int_pos(0)) = 1.0 / var_int;

  arma::colvec beta = arma::nonzeros(tilde_beta) ;

  arma::colvec psi ;
  arma::mat    Xw(N,P);
  arma::mat    Precn(P,P) ;
  arma::mat    Varn(P,P)  ;
  arma::colvec Meann(P) ;

  psi    = Xreduced*beta ;
  arma::colvec w = cpp_polyagamma_h1_devroye(psi) ;
  Xw     = Xreduced.each_col()%w ;
  Precn  = Xreduced.t()*Xw+Prec0 ;
  Varn   = arma::inv_sympd(Precn) ;
  Meann  = Varn*(Z) ;

  new_beta(ones) = cpp_mvrnormArma1( Meann, Varn);

  tilde_beta = new_beta ;

}

// [[Rcpp::export]]
Rcpp::List quasi_Ising( const arma::mat & X,
                        const double var_int, const double var_coef,
                        const double par_pi, const double a, const double b,
                        const int sample,     const int burn = 0,    const int thinning = 1){
  const int N = X.n_rows ;
  const int P = X.n_cols ;

  const int S = burn + thinning * sample;
  int ss = 0;

  MyTimePoint t0 = myClock::now();

  arma::cube  OMEGA(P, P, sample) ;
  arma::cube  DELTA(P, P, sample) ;

  arma::mat    omega(P, P, arma::fill::eye);
  arma::mat    delta(P, P, arma::fill::eye);
  omega = omega * 1e-2 ;

  int current_edges = 0 ;

  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  for( int s = 0; s<S; ++s){
    catIter(s, S, t0) ;
    for( int p = 0; p < P; ++p ){
      Rcpp::checkUserInterrupt();

      tilde_X = X ;
      tilde_y = X.col(p) ;
      tilde_X.col(p).fill(1.0) ;

      arma::subview_col<double> tilde_gamma     = delta.col(p) ;
      arma::subview_row<double> tilde_gamma_tr  = delta.row(p) ;
      arma::subview_col<double> tilde_beta      = omega.col(p) ;

      cpp_update_Omega( p, tilde_gamma, tilde_beta, tilde_y, tilde_X, var_coef, var_int ) ;
      cpp_update_variable_selection_v2( p, tilde_gamma, tilde_beta, tilde_gamma_tr, current_edges, tilde_y, tilde_X, var_coef, par_pi, a, b );
    }
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      OMEGA.slice(ss) = omega ;
      DELTA.slice(ss) = delta ;
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return Rcpp::List::create( Rcpp::Named("delta") = DELTA,
                             Rcpp::Named("beta")  = OMEGA ) ;
}

