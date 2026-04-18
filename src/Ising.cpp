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
double Ising_pseudologlikelihood(const arma::mat& Y, const arma::mat& Omega) {
  arma::vec diag = Omega.diag();
  arma::mat Off  = Omega;
  Off.diag().zeros();
  arma::mat Phi = arma::repmat(diag.t(), Y.n_rows, 1) + Y * Off.t();
  arma::mat term = Y % Phi - arma::log1p(arma::exp(Phi));
  return arma::accu(term);
}

// [[Rcpp::export]]
double node_wise_pseudo_ll( const arma::colvec & y_it, arma::mat Omega) {

  arma::colvec diag = Omega.diag() ;
  Omega.diag().zeros() ;
  arma::colvec phi = diag + Omega * y_it ;

  return arma::sum( y_it % phi -  arma::log( 1 + arma::exp(phi) ) ) ;
}

// [[Rcpp::export]]
double log_likelihood_ratio_add( const arma::colvec & tilde_y,
                                 const arma::mat    & tilde_X,
                                 const arma::colvec & tilde_beta,
                                 const double new_beta,
                                 const int    add_node ) {

  arma::colvec XB        = tilde_X * tilde_beta ;
  arma::colvec delta_vec = new_beta * tilde_X.col(add_node) ;

  return arma::accu( tilde_y % delta_vec -
                     arma::log1p( arma::exp( XB + delta_vec )) +
                     arma::log1p( arma::exp( XB )) ) ;
}

// [[Rcpp::export]]
double log_likelihood_ratio_swap( const arma::colvec & tilde_y,
                                  const arma::mat    & tilde_X,
                                  const arma::colvec & tilde_beta,
                                  const int & node_0_t0_1, const double & new_tilde_beta,
                                  const int & node_1_t0_0, const double & old_tilde_beta ){

  arma::colvec XB        = tilde_X * tilde_beta ;
  arma::colvec delta_vec = tilde_X.col( node_0_t0_1 ) * new_tilde_beta -
                           tilde_X.col( node_1_t0_0 ) * old_tilde_beta ;

  return arma::accu( tilde_y % delta_vec -
                     arma::log1p( arma::exp( XB + delta_vec )) +
                     arma::log1p( arma::exp( XB )) ) ;
}

// [[Rcpp::export]]
double log_likelihood_ratio_global_flip(
    const arma::mat  & Y,
    const arma::mat  & Omega,
    const arma::uvec & pair,
    const arma::uvec & reg_for_pos0,
    const arma::uvec & reg_for_pos1,
    const double beta_new_pos0,
    const double beta_new_pos1) {
  arma::uvec v0 = {pair(0)};
  double old_beta_pos0 = Omega(pair(1),pair(0)) ;
  arma::mat tildeX0 = Y.cols( reg_for_pos0 ) ;
  tildeX0.cols( arma::find(reg_for_pos0 == pair(0)) ).fill(1)  ;
  auto beta0_reduced = Omega.submat(reg_for_pos0, v0 );
  arma::colvec XB0 = tildeX0 * beta0_reduced  ;
  arma::colvec delta0 = Y.col(pair(1)) * (beta_new_pos0 - old_beta_pos0 ) ;

  arma::uvec v1 = {pair(1)};
  double old_beta_pos1 = Omega(pair(0),pair(1)) ;
  arma::mat tildeX1 = Y.cols( reg_for_pos1 ) ;
  tildeX1.cols( arma::find(reg_for_pos1 == pair(1)) ).fill(1)  ;
  auto beta1_reduced = Omega.submat(reg_for_pos1, v1 );
  arma::colvec XB1 = tildeX1 * beta1_reduced  ;
  arma::colvec delta1 = Y.col(pair(0)) * (beta_new_pos1 - old_beta_pos1 ) ;

  return arma::accu(
    Y.col(pair(0)) % delta0 - arma::log1p( arma::exp(XB0 + delta0) ) + arma::log1p( arma::exp(XB0) ) +
    Y.col(pair(1)) % delta1 - arma::log1p( arma::exp(XB1 + delta1) ) + arma::log1p( arma::exp(XB1) )
    );
}


// [[Rcpp::export]]
double log_likelihood_ratio_global_swap(
     const arma::mat  & Y,
     const arma::mat  & Omega,
     const arma::uvec & pair01,
     const arma::uvec & pair23,
     const double beta_new_pos0,
     const arma::uvec & indx0,
     const double beta_new_pos1,
     const arma::uvec & indx1,
     const double beta_new_pos2,
     const arma::uvec & indx2,
     const double beta_new_pos3,
     const arma::uvec & indx3) {


   arma::colvec changed_betas = { beta_new_pos0, beta_new_pos1,
                                  beta_new_pos2, beta_new_pos3 };
  std::vector<arma::uvec> idx_list ;
  idx_list.reserve(4) ;
  idx_list.emplace_back(indx0);
  idx_list.emplace_back(indx1);
  idx_list.emplace_back(indx2);
  idx_list.emplace_back(indx3);

   std::vector<std::tuple<int, arma::uvec, arma::colvec, arma::uvec>> map = map_pairs_into_regs( pair01, pair23, changed_betas,idx_list);

   double ll = 0 ;
   for( std::size_t i = 0; i < map.size(); ++i){
     arma::uword  n1  = std::get<0>(map[i]) ;
     arma::uvec   node = {n1};
     arma::uvec   & to_change   = std::get<1>(map[i]); // len: 1 or 2
     arma::colvec & new_betas   = std::get<2>(map[i]);
     arma::uvec   & idx_active  = std::get<3>(map[i]);

     auto & tildey       = Y.col( n1 ) ;
     auto & beta_old_all = Omega.submat(idx_active, node);
     auto & beta_old_ch  = Omega.submat(to_change, node) ;
     arma::mat tildeX  = Y.cols(idx_active);
     tildeX.cols( arma::find(idx_active == n1)).fill(1) ;

     arma::colvec XB = tildeX * beta_old_all ;
     arma::colvec delta = Y.cols(to_change) * (new_betas - beta_old_ch) ;
     ll +=arma::accu( Y.col(n1) % delta - arma::log1p( arma::exp(XB + delta) ) + arma::log1p( arma::exp(XB) ));
     }

   return ll;
}


