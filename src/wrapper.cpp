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

// [[Rcpp::export]]
Rcpp::NumericVector rpg( const arma::colvec z, const int trunc = 200 ) {
  arma::colvec w =  cpp_polyagamma_h1_truncated(z,trunc);
  return Rcpp::NumericVector(w.begin(), w.end());
}

// [[Rcpp::export]]
Rcpp::List rG0_v0( const int dim,
                   const double sd_off_diag,
                   const double sd_diag,
                   const double sparsity){
  std::pair<arma::mat,arma::mat> ret = cpp_rG0_v0( dim, sparsity, sd_off_diag, sd_diag );
  return Rcpp::List::create( Rcpp::Named("Coef")    = ret.first,
                             Rcpp::Named("Betas")   = ret.second );
}

// [[Rcpp::export]]
Rcpp::List rG0_v1( const int dim,
                   const double sd_off_diag,
                   const double sd_diag,
                   const double sparsity,
                   const double c ){
  std::pair<arma::mat,arma::mat> ret = cpp_rG0_v1( dim, sparsity, c, sd_off_diag, sd_diag );
  return Rcpp::List::create( Rcpp::Named("Coef")    = ret.first,
                             Rcpp::Named("Betas")   = ret.second );
}

// [[Rcpp::export]]
Rcpp::List rG0_v2( const int dim,
                   const arma::colvec & Qx,
                   const double sd_off_diag,
                   const double sd_diag,
                   const double c ){
  std::tuple<arma::mat,arma::mat,arma::mat> ret = cpp_rG0_v2( dim, Qx, c, sd_off_diag, sd_diag );
  return Rcpp::List::create( Rcpp::Named("Coef")    = std::get<0>(ret),
                             Rcpp::Named("Betas")   = std::get<1>(ret),
                             Rcpp::Named("Actives") = std::get<2>(ret) );
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

 // [[Rcpp::export]]
 Rcpp::List w_rG0_v3(const int dim,
                     const arma::colvec &Qx,
                     double sd_offdiag,
                     double sd_diag) {
   auto res = cpp_rG0_v3(dim, Qx, sd_offdiag, sd_diag);
   arma::mat ret1 = std::get<0>(res);
   arma::mat ret2 = std::get<1>(res);
   arma::uvec ones = std::get<2>(res);
   arma::uvec zeros = std::get<3>(res);
   std::vector<arma::uvec> Map_for_Ising = std::get<4>(res);


   Rcpp::List map_list = vector_to_list(Map_for_Ising) ;
   return Rcpp::List::create(
     Rcpp::Named("ret1") = ret1,
     Rcpp::Named("ret2") = ret2,
     Rcpp::Named("ones") = ones,
     Rcpp::Named("zeros") = zeros,
     Rcpp::Named("Map_for_Ising") = map_list
   );
 }


