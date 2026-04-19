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

// =============================================================================
// cluster_parameter
// -----------------------------------------------------------------------------
// Container for the unique parameters associated with a single cluster in the
// nonparametric mixture of quasi-Ising graphical models.
//
// Each cluster is characterised by:
//   - a vector of node-wise intercepts  alpha  (length p), corresponding to
//     the diagonal terms {alpha_r} of the node-wise logistic regressions;
//   - a matrix of pairwise interaction coefficients  Beta  (p x p);
//   - a binary adjacency structure Gamma, summarised here by two index sets
//     ('ones' and 'zeros') that partition the L = p(p-1)/2 potential edges
//     into active (E_l = 1) and inactive (E_l = 0).
// =============================================================================


struct cluster_parameter {
  arma::mat    Beta;  // Pairwise interaction matrix beta (p x p).
  arma::colvec alpha; // Vector of node-specific intercepts
  arma::uvec   ones;  // Mapped indices (in {0, ..., L-1}, L=p(p-1)/2) of the currently ACTIVE edges
  arma::uvec   zeros; // Mapped indices (in {0, ..., L-1}, L=p(p-1)/2) of the currently INACTIVE edges
  std::vector<arma::uvec> mapping; // Indices of sparse adjacency representation for each node r in {0, ..., p-1}

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
    arma::uword L = dim*(dim-1)/2 ;
    arma::uvec ordered = arma::regspace<arma::uvec>(0,L-1) ;
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
    double b0;
    alpha = arma::randn<arma::colvec>(dim, arma::distr_param(0.0, sd_diag));
    const int       M  = dim * (dim - 1) / 2;
    arma::uvec   perm    = arma::randperm<arma::uvec>(M) ;
    arma::uvec   ordered = arma::regspace<arma::uvec>(0,M) ;
    int EE = cpp_sample(ordered, 1, Qx)(0) ;
    arma::uword  k, i, j, v;
    mapping.resize(dim) ;
    for( k = 0; k < dim; ++k){
      v = k ;
      mapping[k] = arma::uvec({v});
    }
    arma::uvec pair ;
    for (int pos = 0; pos < EE; ++pos) {
      pair = index_to_pair( perm(pos) ) ;
      i = pair(0) ;
      j = pair(1) ;
      push_back_j(mapping[i], j );
      push_back_j(mapping[j], i );
      b0 = arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
      Beta(j,i) = b0;
      Beta(i,j) = arma::randn<double>( arma::distr_param( b0*rho, (1+1e-10 - rho*rho)*sd_offdiag*sd_offdiag));
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
