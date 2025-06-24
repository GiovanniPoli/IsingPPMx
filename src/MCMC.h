#ifndef MCMC_H
#define MCMC_H

#include <RcppArmadillo.h>
#include <iostream>
#include <chrono>
#include <thread>

std::string cpp_update_variable_selection( arma::subview_col<double> tilde_gamma,
                                    arma::subview_col<double> tilde_beta,
                                    const arma::colvec & y,  const arma::mat & X,
                                    const double pi_slab,
                                    const double var_slab ) ;


std::string cpp_update_Omega( const arma::subview_col<double> tilde_gamma,
                              arma::subview_col<double> tilde_beta,
                              const arma::colvec & y,
                              const arma::mat    & X,
                              const double pi_slab, const double var_slab) ;


std::string cpp_update_variable_selection_v2( arma::subview_col<double> tilde_gamma,
                                              arma::subview_col<double> tilde_beta,
                                              arma::subview_row<double> tilde_gamma_tr,
                                              const arma::colvec & y,  const arma::mat & X,
                                              const double pi_slab, const double var_slab,
                                              const double beta) ;

#endif
