#ifndef UPDATE_STEPS_H
#define UPDATE_STEPS_H

#include <RcppArmadillo.h>
#include <iostream>
#include <chrono>
#include <thread>


void cpp_update_beta_and_alpha( const arma::mat  & YY,
                                arma::mat        & BETA,
                                arma::colvec     & alpha,
                                const arma::uvec & map_ones,
                                const double var_slab,
                                const double var_int,
                                const double rho );

void cpp_update_Omega( const int p, // diff var
                       const arma::subview_col<double> tilde_gamma,
                             arma::subview_col<double> tilde_beta,
                       const arma::colvec & y, const arma::mat & X,
                       const double var_slab,  const double var_int );

void cpp_variable_selection_v0( const int p,
                                arma::subview_col<double> tilde_gamma,
                                arma::subview_col<double> tilde_beta,
                                const arma::colvec & y,  const arma::mat & X,
                                const double var_slab,
                                const double par_pi );

void cpp_variable_selection_v2( const int p,
                                arma::subview_col<double> tilde_gamma,
                                arma::subview_col<double> tilde_beta,
                                arma::subview_row<double> tilde_gamma_tr,
                                arma::mat & Emat,
                                arma::uword & S_N,
                                const arma::colvec & Qx,
                                const arma::colvec & y,  const arma::mat & X,
                                const double var_slab,
                                const double c );

void cpp_update_cluster_v0( arma::uvec & rho,   arma::uvec & table, arma::mat& Sx, int & H,
                            arma::cube & BETAS, arma::cube & GAMMAS,
                            arma::cube & B_ext, arma::cube & G_ext,
                            unsigned index,
                            const arma::mat & Y, const arma::mat & Z,
                            const double M, const double sigma,  const double c_par,
                            const int C,    const double pi_par, const double var_int,
                            const double var_coef ) ;

void cpp_update_cluster_v2( unsigned index,
                            arma::uvec & rho,   arma::uvec & table,
                            arma::mat& Sx, int & H,
                            arma::cube & BETAS, arma::cube & GAMMAS,
                            arma::cube & B_ext, arma::cube & G_ext,
                            arma::cube & Ecube, arma::cube & E_ext,
                            arma::uvec & SN,
                            const arma::mat & Y, const arma::mat & Z,
                            const double M, const double sigma,
                            const arma::colvec & Qx, const double c_par,
                            const double var_int, const double var_coef, const int C) ;

unsigned cpp_update_S_N( const int sum_gammas, const arma::colvec & Qx) ;



void cpp_variable_selection_v3( arma::mat & BETA,
                                arma::mat & GAMMA,
                                arma::uvec & Map_to_zeros,
                                arma::uvec & Map_to_ones,
                                std::vector<arma::uvec> & Map_for_current_Ising,
                                arma::uword & S_N,
                                const arma::colvec & Qx, const arma::mat & YY,
                                const double var_slab) ;

void cpp_update_Omega_v3( const int p,
                          const arma::subview_col<double> tilde_gamma,
                          arma::subview_col<double> tilde_beta,
                          arma::subview_row<double> tilde_beta_tr,
                          const arma::uvec & Ones,
                          const arma::colvec & y, const arma::mat & X,
                          const double var_slab,  const double var_int ) ;

void cpp_update_cluster_v3( unsigned index,
                            arma::uvec & rho,
                            arma::uvec & table, arma::mat& Sx, int & H,
                            arma::cube & BETAS, arma::cube & GAMMAS,
                            arma::cube & B_ext, arma::cube & G_ext,
                            arma::uvec & S_N,
                            std::vector<arma::uvec> & Map_to_ones,
                            std::vector<arma::uvec> & Map_to_zeros,
                            std::vector<std::vector<arma::uvec>> & logits_maps,
                            std::vector<arma::uvec> & Map_to_ones_EXT,
                            std::vector<arma::uvec> & Map_to_zeros_EXT,
                            std::vector<std::vector<arma::uvec>> & logits_maps_EXT,
                            const arma::mat & Y, const arma::mat & Z,
                            const arma::colvec & Qx,
                            const double M, const double sigma,
                            const int C,
                            const double var_int, const double var_coef );

void   cpp_update_Omega_v5( arma::mat & BETA,
                               double & alpha,
                               const arma::uword p,
                               const arma::mat  & YY,
                               const arma::uvec & map_ones,
                               const double var_slab,
                               const double var_int ) ;
#endif
