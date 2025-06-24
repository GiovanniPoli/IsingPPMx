#ifndef Ising_H
#define Ising_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>



double node_wise_pseudo_ll(const arma::colvec & y_it, arma::mat Omega) ;
double node_wise_generalized_ll(const arma::colvec & y_it, arma::mat Omega, const int a) ;
double log_likelihood_ratio_swap( const arma::colvec & tilde_y, const arma::mat & tilde_X, const arma::colvec & tilde_beta, const double add_beta, const int remove_node, const int add_node );
double log_likelihood_ratio_flip(const arma::colvec & tilde_y, const arma::mat & tilde_X, const arma::colvec & tilde_beta, const double new_beta, const int node) ;
double log_likelihood_ratio_swap_v2( const arma::colvec & tilde_y, const arma::mat & tilde_X, const arma::colvec & tilde_beta, const double new_beta, const int old_one, const int new_one );

#endif
