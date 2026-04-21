#ifndef Ising_H
#define Ising_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>


double Ising_pseudologlikelihood(const arma::mat& Y, const arma::mat& Omega) ;
double node_wise_pseudo_ll(const arma::colvec & y_it, arma::mat Omega) ;


double cpp_ll_ratio_global_flip(
    const arma::mat & YY,
    const arma::mat & Beta, const arma::colvec & alpha,
    const arma::uword n1,   const arma::uvec   & mapping_n1,
    const double flipped_beta_n2_reg_n1,
    const arma::uword n2,   const arma::uvec   & mapping_n2,
    const double flipped_beta_n1_reg_n2 );

double cpp_ll_ratio_global_swap(
    const arma::mat    & YY,
    const arma::mat    & Beta,
    const arma::colvec & alpha,
    const arma::uword n1, const arma::uvec & mapping_n1,
    const double delta_b_reg_n1,
    const arma::uword n2, const arma::uvec & mapping_n2,
    const double delta_b_reg_n2,
    const arma::uword n3, const arma::uvec & mapping_n3,
    const double delta_b_reg_n3,
    const arma::uword n4, const arma::uvec & mapping_n4,
    const double delta_b_reg_n4 );


double log_likelihood_ratio_add( const arma::colvec & tilde_y, const arma::mat & tilde_X,
                                 const arma::colvec & tilde_beta, const double new_beta, const int add_node );

double log_likelihood_ratio_swap( const arma::colvec & tilde_y, const arma::mat & tilde_X,
                                  const arma::colvec & tilde_beta,
                                  const int & node_0_t0_1, const double & new_tilde_beta,
                                  const int & node_1_t0_0, const double & old_tilde_beta ) ;

double log_likelihood_ratio_global_flip( const arma::mat  & Y, const arma::mat  & Omega, const arma::uvec & pair,
    const arma::uvec & reg_for_pos0, const arma::uvec & reg_for_pos1,
    const double beta_new_pos0, const double beta_new_pos1);


double log_likelihood_ratio_global_swap(const arma::mat  & Y,const arma::mat  & Omega,
                                        const arma::uvec & pair01,  const arma::uvec & pair23,
                                        const double beta_new_pos0, const arma::uvec & indx0,
                                        const double beta_new_pos1, const arma::uvec & indx1,
                                        const double beta_new_pos2, const arma::uvec & indx2,
                                        const double beta_new_pos3, const arma::uvec & indx3);
#endif
