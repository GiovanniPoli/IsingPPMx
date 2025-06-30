#ifndef MCMC_H
#define MCMC_H

#include <RcppArmadillo.h>
#include <iostream>
#include <chrono>
#include <thread>

void cpp_update_variable_selection( arma::subview_col<double> tilde_gamma,
                                    arma::subview_col<double> tilde_beta,
                                    const arma::colvec & y,  const arma::mat & X,
                                    const double pi_slab,
                                    const double var_slab ) ;


void cpp_update_Omega( const int p,
                       const arma::subview_col<double> tilde_gamma,
                       arma::subview_col<double> tilde_beta,
                       const arma::colvec & y,
                       const arma::mat    & X,
                       const double var_slab,
                       const double var_int ) ;


void cpp_update_variable_selection_v2( const int p,
                                       arma::subview_col<double> tilde_gamma,
                                       arma::subview_col<double> tilde_beta,
                                       arma::subview_row<double> tilde_gamma_tr,
                                       int & edges,
                                       const arma::colvec & y,  const arma::mat & X,
                                       const double var_slab,
                                       const double par_pi, const double a, const double b)  ;

#endif
