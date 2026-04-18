#ifndef STRUCT_H
#define STRUCT_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <truncnorm.h>
#include <mvnorm.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "randomgen.h"
#include "utils.h"

struct cluster_parameter {
  arma::mat    Beta;
  arma::colvec alpha;
  arma::uvec   ones;
  arma::uvec   zeros;
  std::vector<arma::uvec> mapping;

  cluster_parameter() {
    Beta  = arma::mat(0,0);
    alpha = {};
    ones  = {};
    zeros = {};
    mapping = std::vector<arma::uvec>(0);
  }

  cluster_parameter( const arma::uword dim) {
    Beta.zeros(dim, dim);
    alpha.zeros(dim);
    arma::uword M = dim*(dim-1)/2 ;
    arma::uvec ordered = arma::regspace<arma::uvec>(0,M) ;
    arma::uword v;
    mapping.resize(dim);
    for( unsigned p = 0; p < dim; ++p){
      v = p ;
      mapping[p] = arma::uvec({v});
    }
    ones = {};
    zeros = ordered ;
  }

  cluster_parameter( const arma::uword dim,
                     const arma::colvec & Qx,
                     const double sd_diag,
                     const double sd_offdiag,
                     const double rho ) {
    Beta.zeros(dim, dim);
    alpha = arma::randn<arma::colvec>(dim, arma::distr_param(0.0, sd_diag));
    const int       M  = dim * (dim - 1) / 2;
    const double disc  = 2.0 *  dim - 1.0;
    const double disc2 = disc*disc;
    arma::uvec   perm    = arma::randperm<arma::uvec>(M) ;
    arma::uvec   ordered = arma::regspace<arma::uvec>(0,M) ;
    double EE = cpp_sample(ordered, 1, Qx)(0) ;
    arma::uword  k, i, j, v;
    mapping.resize(dim) ;
    for( unsigned p = 0; p < dim; ++p){
      v = p ;
      mapping[p] = arma::uvec({v});
    }
    arma::uvec pair ;
    for (int pos = 0; pos < EE; ++pos) {
      pair = index_to_pair( perm(pos) ) ;
      i = pair(0) ;
      j = pair(1) ;
      push_back_j(mapping[i], j );
      push_back_j(mapping[j], i );
      Beta(j,i) = arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
      Beta(i,j) = arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
    }
    if( EE == 0){
      ones = {};
      zeros = perm ;
    }else if(EE == M){
      zeros = {};
      ones = perm ;
    }else{
      ones   = perm.subvec(0, EE-1);
      zeros  = perm.subvec(EE, M-1);
    }
  }
};

#endif
