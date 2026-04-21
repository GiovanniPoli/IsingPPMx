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
#include "update_steps.h"
#include "struct.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]]

//' Bayesian Logistic Regression via Polya-Gamma Augmentation
//'
//' Fits a Bayesian logistic regression model using the Polya-Gamma data
//' augmentation scheme of Polson, Scott, and Windle (2013). The posterior
//' of the coefficient vector \eqn{\boldsymbol{\beta}} is conjugate given the
//' augmented variables, and each MCMC iteration reduces to a Gaussian update.
//' Used internally as a building block for the node-wise coefficient updates
//' of the quasi-Ising samplers.
//'
//' @param y      Binary response vector of length \eqn{n}, with entries in
//'               \eqn{\{0, 1\}}.
//' @param X      Design matrix of dimension \eqn{n \times p}. For node-wise
//'               quasi-Ising updates, the column corresponding to the response
//'               node is replaced by a column of ones (intercept).
//' @param b0     Prior mean vector of length \eqn{p} for
//'               \eqn{\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{b}_0, B_0)}.
//' @param B0     Prior covariance matrix \eqn{p \times p} for
//'               \eqn{\boldsymbol{\beta}}.
//' @param bstart Starting value for \eqn{\boldsymbol{\beta}}, a vector of
//'               length \eqn{p}.
//' @param sample Number of MCMC draws to retain after burn-in and thinning.
//' @param burn   Number of initial iterations to discard as burn-in.
//'               Default \code{0}.
//' @param thinning Thinning interval: one draw is stored every
//'               \code{thinning} iterations. Default \code{1} (no thinning).
//'
//' @return A numeric matrix of dimension \eqn{n} \eqn{\times} \eqn{p}.
//'   Row \eqn{s} contains the \eqn{s}-th posterior draw of
//'   \eqn{\boldsymbol{\beta}}.
//'
//' @details
//' The sampler runs for \code{burn + thinning * sample} total iterations.
//' At each step:
//' \enumerate{
//'   \item Compute the linear predictor \eqn{\boldsymbol{\psi} = X\boldsymbol{\beta}}.
//'   \item Draw Polya-Gamma weights \eqn{\omega_i \sim \mathrm{PG}(1, \psi_i)}.
//'   \item Update \eqn{\boldsymbol{\beta}} from the resulting Gaussian full conditional distribution.
//' }
//'
//' @references
//' Polson, N. G., Scott, J. G., and Windle, J. B. (2013).
//' Bayesian inference for logistic models using Polya-Gamma latent variables.
//' \emph{Journal of the American Statistical Association}, 108(504), 1339--1349.
//'
//' @seealso \code{\link{qIsing}}, \code{\link{qIsing_PPMx}}
//'
//' @export
// [[Rcpp::export]]
arma::mat logit_mcmc( const arma::colvec & y,  const arma::mat & X,
                       const arma::colvec & b0, const arma::mat & B0,
                       const arma::colvec & bstart,
                       const int sample,
                       const int burn = 0,
                       const int thinning = 1){
  const int N = y.n_elem ;
  const int P = X.n_cols ;

  arma::colvec n(N, arma::fill::ones); ;
  arma::colvec alpha = (y - .5) % n ;
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
  }
  catIter(S, S, t0) ;
  return BETA ;
}

//' Bayesian Quasi-Ising Graphical Model (Single Population)
//'
//' Fits a single-population quasi-Ising graphical model via Metropolis-within-Gibbs
//' MCMC. The joint distribution is approximated by the pseudo-likelihood of
//' Besag (1975), which factorises into \eqn{p} independent node-wise logistic
//' regressions. Graph structure is learned through a finite-exchangeable-sequence
//' (FES) prior on the edge count
//' Paired interaction coefficients share information through bivariate Normal prior with correlation \eqn{\rho}.
//'
//' @param Y      Binary data matrix of dimension \eqn{p \times p}. Rows are
//'               observations, columns are nodes of the graph.
//' @param Qx     Probability vector of length \eqn{L+1}, where
//'               \eqn{L = p(p-1)/2}, encoding the FES prior on the number of
//'               active edges: \eqn{Q_x(k) = \Pr(K = k)} for
//'               \eqn{k = 0, \ldots, L}.
//' @param sd_int  Prior standard deviation for the node-wise intercepts
//'               \eqn{\alpha_j \sim \mathcal{N}(0, \texttt{sd\_int}^2)}.
//' @param sd_coef Prior standard deviation for the slab component of the
//'               interaction coefficients
//'               \eqn{\beta_{r,c} \mid \gamma_{r,c}=1 \sim \mathcal{N}(0, \texttt{sd\_coef}^2)}.
//' @param rho    Correlation parameter of the quasi-symmetric Normal prior on
//'               the paired coefficients \eqn{(\beta_{r,c}, \beta_{c,r})}.
//'               \eqn{\rho = 0} gives independence; \eqn{\rho \to 1} forces
//'               \eqn{\beta_{r,c} = \beta_{c,r}} a priori.
//' @param sample Number of MCMC draws to retain after burn-in and thinning.
//' @param burn   Number of initial iterations to discard as burn-in.
//'               Default \code{0}.
//' @param thinning Thinning interval. Default \code{1} (no thinning).
//'
//' @return A named \code{list} of length \code{sample}. Each element is itself
//'   a named list with four entries:
//'   \describe{
//'     \item{\code{Beta}}{Numeric \eqn{p \times p} matrix of interaction
//'       coefficients. Entry \eqn{(r,c)} is \eqn{\hat\beta_{r,c}}; the diagonal
//'       is zero.}
//'     \item{\code{alpha}}{Numeric vector of length \eqn{p} of node-wise
//'       intercepts.}
//'     \item{\code{ones}}{Integer vector of active-edge linear indices
//'       (1-based, \eqn{\subseteq \{1,\ldots,L\}}). Together with
//'       \code{zeros} it partitions the full edge set.}
//'     \item{\code{zeros}}{Integer vector of inactive-edge linear indices
//'       (1-based).}
//'   }
//'
//' @details
//' The sampler alternates two steps at each iteration:
//' \enumerate{
//'   \item \strong{Coefficient-smooting update step} (\eqn{\alpha}, \eqn{\beta}).
//'   \item \strong{Graph update} (\eqn{\Gamma}).
//'     A Global Add / Delete / Swap Metropolis step proposes a new graph from the
//'     FES prior \code{Qx} and accepts according to the quasi-likelihood ratio.
//' }
//'
//' @references
//' Besag, J. (1975). Statistical analysis of non-lattice data.
//' \emph{The Statistician}, 24(3), 179--195.
//'
//' Polson, N. G., Scott, J. G., and Windle, J. B. (2013).
//' Bayesian inference for logistic models using Polya-Gamma latent variables.
//' \emph{JASA}, 108(504), 1339--1349.
//'
//' @seealso \code{\link{bayes_logistic_regression}}, \code{\link{qIsing_PPMx_v5}}
//'
//' @export
// [[Rcpp::export]]
Rcpp::List qIsing_mcmc( const arma::mat  & Y,
                         const arma::colvec & logQx,
                         const double sd_int,
                         const double sd_coef,
                         const double rho,
                         const int sample,
                         const int burn     = 0,
                         const int thinning = 1 )  {

  const arma::uword N = Y.n_rows;
  const arma::uword P = Y.n_cols;
  const arma::uword L = P * (P - 1) / 2;
  const arma::uword S = burn + thinning * static_cast<arma::uword>(sample);

  const double var_slab = sd_coef * sd_coef ;
  const double var_int  = sd_int  * sd_int  ;

  arma::uword ss = 0;
  Rcpp::List  return_list(sample);

  arma::mat    Beta(P, P, arma::fill::zeros);
  arma::colvec Alpha(P, arma::fill::zeros);
  arma::uvec   ones  = arma::uvec();
  arma::uvec   zeros = arma::regspace<arma::uvec>(0, L - 1);

  std::vector<arma::uvec> MAP(P);
  for (arma::uword r = 0; r < P; ++r) {
    MAP[r] = arma::uvec({r});
  }

  arma::uword current_edges = 0;
  MyTimePoint t0 = myClock::now();

  for (arma::uword s = 0; s < S; ++s) {
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0);

    for (arma::uword r = 0; r < P; ++r) {

      cpp_update_beta_and_alpha( Y,
                                 Beta,
                                 Alpha,
                                 MAP[r],
                                 var_slab,
                                 var_int,
                                 rho );
    }

    for (arma::uword r = 0; r < P; ++r) {

    }




    // Store
    if ( ((s + 1) > static_cast<arma::uword>(burn)) &&
         ((s + 1 - burn) % static_cast<arma::uword>(thinning) == 0) ) {
      return_list[ss] = Rcpp::List::create(
        Rcpp::Named("Beta")  = Beta,
        Rcpp::Named("alpha") = Alpha,
        Rcpp::Named("ones")  = ones  + 1,   // shift for R
        Rcpp::Named("zeros") = zeros + 1
      );
      ss += 1;
    }
  }
  catIter(S, S, t0);
  return return_list;
}












//
// // [[Rcpp::export]]
// Rcpp::List qIsing_mcmc( const arma::mat & Y,
//                    const double var_int, const double var_coef, const double par_pi,
//                    const int sample, const int burn = 0, const int thinning = 1){
//   const int N = Y.n_rows ;
//   const int P = Y.n_cols ;
//
//   const int S = burn + thinning * sample;
//   int ss = 0;
//
//   MyTimePoint t0 = myClock::now();
//
//   arma::cube  OMEGA(P, P, sample) ;
//   arma::cube  DELTA(P, P, sample) ;
//
//   arma::mat    omega(P, P, arma::fill::eye);
//   arma::mat    delta(P, P, arma::fill::eye);
//
//   omega = omega * 1e-5 ;
//
//   int current_edges = 0 ;
//
//   arma::mat    tilde_X ;
//   arma::colvec tilde_y ;
//
//   for( int s = 0; s<S; ++s){
//     catIter(s, S, t0) ;
//     for( int p = 0; p < P; ++p ){
//       Rcpp::checkUserInterrupt();
//
//       tilde_X = Y ;
//       tilde_y = Y.col(p) ;
//       tilde_X.col(p).fill(1.0) ;
//
//       arma::subview_col<double> tilde_gamma     = delta.col(p) ;
//       arma::subview_col<double> tilde_beta      = omega.col(p) ;
//
//       cpp_update_Omega( p, tilde_gamma, tilde_beta,
//                         tilde_y, tilde_X, var_coef, var_int ) ;
//
//       cpp_variable_selection_v0( p, tilde_gamma, tilde_beta,
//                                  tilde_y, tilde_X, var_coef, par_pi ) ;
//
//    }
//     if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//       OMEGA.slice(ss) = omega ;
//       DELTA.slice(ss) = delta ;
//       ss += 1 ;
//     }
//   }
//   catIter(S, S, t0) ;
//   return Rcpp::List::create( Rcpp::Named("delta") = DELTA,
//                              Rcpp::Named("beta")  = OMEGA ) ;
// }
//
//n// [[Rcpp::export]]
//nRcpp::List qIsing_PPMx_v0( const arma::mat & Y,     const arma::mat & Z,
//n                           const double & var_int,  const double & var_coef, const double & par_pi,
//n                           const double & M,        const double & sigma,
//n                           const int sample = 1000, const int burn = 0, const int thinning = 1,
//n                           const int C = 5){
//n
//n  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations";
//n
//n  const unsigned totN = Y.n_rows ;
//n  const unsigned P    = Y.n_cols ;
//n
//n  const unsigned S    = burn + thinning * sample;
//n  unsigned ss = 0;
//n
//n  const arma::mat eye_mat(P,P, arma::fill::eye);
//n
//n  arma::cube  BETAS_it (P, P, totN, arma::fill::zeros) ;
//n  arma::cube  GAMMAS_it(P, P, totN, arma::fill::zeros) ;
//n
//n  arma::cube  BETAS_ext (P, P, C) ;
//n  arma::cube  GAMMAS_ext(P, P, C) ;
//n
//n  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  GAMMAS_it.each_slice()  += eye_mat ;
//n  GAMMAS_ext.each_slice() += eye_mat ;
//n
//n  int H = totN;
//n
//n  arma::uvec rho_it = arma::linspace<arma::uvec>(0,H-1,H) ;
//n  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
//n  arma::mat  Sx = Z;
//n
//n  Rcpp::List return_list(sample) ;
//n
//n  arma::mat    Gh_beta  ;
//n  arma::mat    Gh_gamma ;
//n  arma::mat    tilde_X  ;
//n  arma::colvec tilde_y  ;
//n
//n  Rcpp::Rcout << ". (Done!)" << std::endl ;
//n
//n  MyTimePoint t0 = myClock::now();
//n
//n  for( int s = 0; s<S; ++s){
//n    Rcpp::checkUserInterrupt();
//n    catIter(s, S, t0) ;
//n
//n    // Parameter updates
//n    for( int h = 0; h < H; ++h ){
//n      arma::mat & Gh_beta   = BETAS_it.slice(h) ;
//n      arma::mat & Gh_gamma  = GAMMAS_it.slice(h);
//n      const arma::uvec & index_h = arma::find( rho_it == h ) ;
//n      const arma::mat  & subY = Y.rows( index_h );
//n
//n      for( int p = 0; p < P; ++p ){
//n
//n        tilde_X = subY;
//n        tilde_y = subY.col(p) ;
//n        tilde_X.col(p).fill(1.0) ;
//n
//n        arma::subview_col<double> tilde_gamma     = Gh_gamma.col(p) ;
//n        arma::subview_col<double> tilde_beta      = Gh_beta.col(p) ;
//n
//n        cpp_update_Omega( p, tilde_gamma, tilde_beta,
//n                          tilde_y, tilde_X, var_coef, var_int ) ;
//n
//n        cpp_variable_selection_v0( p, tilde_gamma, tilde_beta,
//n                                   tilde_y, tilde_X, var_coef, par_pi ) ;
//n
//n
//n      }
//n    }
//n    // Clusters updates
//n    for( int i = 0; i<totN; ++i ){
//n    }
//n
//n    // Store
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
//n                                            Rcpp::Named("Beta")  = BETAS_it,
//n                                            Rcpp::Named("Gamma") = GAMMAS_it,
//n                                            Rcpp::Named("Sx")    = Sx,
//n                                            Rcpp::Named("H")     = H) ;
//n      ss += 1 ;
//n    }
//n   }
//n  catIter(S, S, t0) ;
//n  return return_list ;
//n}
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_v1( const arma::mat & Y,
//n                      const double var_int, const double var_coef,
//n                      const double par_pi,  const double c,
//n                      const int sample, const int burn = 0, const int thinning = 1){
//n  const int N = Y.n_rows ;
//n  const int P = Y.n_cols ;
//n
//n  const int S = burn + thinning * sample;
//n  int ss = 0;
//n
//n  MyTimePoint t0 = myClock::now();
//n
//n  arma::cube  OMEGA(P, P, sample) ;
//n  arma::cube  DELTA(P, P, sample) ;
//n
//n  arma::mat    omega(P, P, arma::fill::eye);
//n  arma::mat    delta(P, P, arma::fill::eye);
//n
//n  omega = omega * 1e-5 ;
//n
//n  int current_edges = 0 ;
//n
//n  arma::mat    tilde_X ;
//n  arma::colvec tilde_y ;
//n
//n  for( int s = 0; s<S; ++s){
//n    catIter(s, S, t0) ;
//n    for( int p = 0; p < P; ++p ){
//n      Rcpp::checkUserInterrupt();
//n
//n      tilde_X = Y ;
//n      tilde_y = Y.col(p) ;
//n      tilde_X.col(p).fill(1.0) ;
//n
//n      arma::subview_col<double> tilde_gamma     = delta.col(p) ;
//n      arma::subview_col<double> tilde_beta      = omega.col(p) ;
//n
//n      cpp_update_Omega( p, tilde_gamma, tilde_beta,
//n                        tilde_y, tilde_X, var_coef, var_int ) ;
//n
//n      cpp_variable_selection_v0( p, tilde_gamma, tilde_beta,
//n                                 tilde_y, tilde_X, var_coef, par_pi ) ;
//n
//n    }
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      OMEGA.slice(ss) = omega ;
//n      DELTA.slice(ss) = delta ;
//n      ss += 1 ;
//n    }
//n  }
//n  catIter(S, S, t0) ;
//n  return Rcpp::List::create( Rcpp::Named("delta") = DELTA,
//n                             Rcpp::Named("beta")  = OMEGA ) ;
//n}
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_PPMx_v1( const arma::mat & Y,     const arma::mat & Z,
//n                           const double & var_int,  const double & var_coef,
//n                           const double & par_pi,   const double & c,
//n                           const double & M,        const double & sigma,
//n                           const int sample = 1000, const int burn = 0, const int thinning = 1,
//n                           const int C = 5){
//n
//n  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations";
//n
//n  const unsigned totN = Y.n_rows ;
//n  const unsigned P    = Y.n_cols ;
//n
//n  const unsigned S    = burn + thinning * sample;
//n  unsigned ss = 0;
//n
//n  const arma::mat eye_mat(P,P, arma::fill::eye);
//n
//n  arma::cube  BETAS_it (P, P, totN, arma::fill::zeros) ;
//n  arma::cube  GAMMAS_it(P, P, totN, arma::fill::zeros) ;
//n
//n  arma::cube  BETAS_ext (P, P, C) ;
//n  arma::cube  GAMMAS_ext(P, P, C) ;
//n
//n  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, var_int));
//n  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, var_int));
//n  GAMMAS_it.each_slice()  += eye_mat ;
//n  GAMMAS_ext.each_slice() += eye_mat ;
//n
//n  int H = totN;
//n  arma::uvec rho_it = arma::linspace<arma::uvec>(0,H-1,H) ;
//n  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
//n  arma::uvec current_edges(H, arma::fill::zeros) ;
//n  arma::mat  Sx = Z;
//n
//n  Rcpp::List return_list(sample) ;
//n
//n  arma::mat    Gh_beta ;
//n  arma::mat    Gh_gamma ;
//n  arma::mat    tilde_X ;
//n  arma::colvec tilde_y ;
//n
//n  Rcpp::Rcout << ". (Done!)" << std::endl ;
//n
//n  MyTimePoint t0 = myClock::now();
//n
//n  for( int s = 0; s<S; ++s){
//n    Rcpp::checkUserInterrupt();
//n    catIter(s, S, t0) ;
//n
//n    // Parameter updates
//n    for( int h = 0; h < H; ++h ){
//n
//n      arma::mat & Gh_beta   = BETAS_it.slice(h) ;
//n      arma::mat & Gh_gamma  = GAMMAS_it.slice(h);
//n      const arma::mat & subY = Y.rows( arma::find( rho_it == h ));
//n
//n      for( int p = 0; p < P; ++p ){
//n
//n        tilde_X = subY;
//n        tilde_y = subY.col(p) ;
//n        tilde_X.col(p).fill(1.0) ;
//n        arma::uword & edges_h = current_edges(h);
//n
//n        arma::subview_col<double> tilde_gamma     = Gh_gamma.col(p) ;
//n        arma::subview_row<double> tilde_gamma_tr  = Gh_gamma.row(p) ;
//n        arma::subview_col<double> tilde_beta      = Gh_beta.col(p) ;
//n
//n      }
//n    }
//n
//n    // Clusters updates
//n    for( int i = 0; i<totN; ++i ){
//n    }
//n    // Store
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
//n                                            Rcpp::Named("Beta")  = BETAS_it,
//n                                            Rcpp::Named("Gamma") = GAMMAS_it,
//n                                            Rcpp::Named("Sx")    = Sx,
//n                                            Rcpp::Named("H")     = H) ;
//n      ss += 1 ;
//n    }
//n  }
//n  catIter(S, S, t0) ;
//n  return return_list ;
//n}
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_PPMx_v2( const arma::mat & Y,       const arma::mat & Z,
//n                           const double & var_int,    const double & var_coef,
//n                           const arma::colvec & Qx,   const double & c,
//n                           const double & M,          const double & sigma,
//n                           const int sample = 1000,   const int burn = 0,    const int thinning = 1,
//n                           const int C = 5){
//n
//n  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations";
//n
//n  const unsigned totN = Y.n_rows ;
//n  const unsigned P    = Y.n_cols ;
//n
//n  const unsigned S    = burn + thinning * sample;
//n  unsigned ss = 0;
//n
//n  const arma::mat eye_mat(P,P, arma::fill::eye);
//n
//n  arma::cube BETAS_it (P, P, totN, arma::fill::zeros) ;
//n  arma::cube GAMMAS_it(P, P, totN, arma::fill::zeros) ;
//n  arma::cube E_it(P, P, totN, arma::fill::zeros) ;
//n
//n  arma::cube BETAS_ext (P, P, C) ;
//n  arma::cube GAMMAS_ext(P, P, C) ;
//n  arma::cube E_ext(P, P, C) ;
//n
//n  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  GAMMAS_it.each_slice()  += eye_mat ;
//n  GAMMAS_ext.each_slice() += eye_mat ;
//n
//n  int H = totN;
//n  arma::uvec rho_it = arma::linspace<arma::uvec>(0, H-1, H) ;
//n  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
//n  arma::uvec   S_N( H, arma::fill::zeros )  ;
//n
//n  arma::mat  Sx = Z;
//n
//n  Rcpp::List return_list(sample) ;
//n
//n  arma::mat    Gh_beta ;
//n  arma::mat    Gh_gamma ;
//n  arma::mat    tilde_X ;
//n  arma::colvec tilde_y ;
//n
//n  Rcpp::Rcout << ". (Done!)" << std::endl ;
//n
//n  MyTimePoint t0 = myClock::now();
//n
//n  for( int s = 0; s<S; ++s){
//n    Rcpp::checkUserInterrupt();
//n    catIter(s, S, t0) ;
//n    // Parameter updates
//n    for( int h = 0; h < H; ++h ){
//n
//n      arma::mat & Gh_beta    = BETAS_it.slice(h) ;
//n      arma::mat & Gh_gamma   = GAMMAS_it.slice(h);
//n      arma::mat & Gh_E       = E_it.slice(h);
//n      const arma::mat & subY = Y.rows( arma::find( rho_it == h ));
//n      arma::uword & SN_h =   S_N(h);
//n
//n      for(int p = 0; p < P; ++p ){
//n
//n        tilde_X = subY;
//n        tilde_y = subY.col(p) ;
//n        tilde_X.col(p).fill(1.0) ;
//n
//n        arma::subview_col<double> tilde_gamma     = Gh_gamma.col(p) ;
//n        arma::subview_row<double> tilde_gamma_tr  = Gh_gamma.row(p) ;
//n        arma::subview_col<double> tilde_beta      = Gh_beta.col(p) ;
//n
//n        cpp_update_Omega( p, tilde_gamma, tilde_beta, tilde_y, tilde_X, var_coef, var_int ) ;
//n
//n
//n        cpp_variable_selection_v2( p,
//n                                   tilde_gamma, tilde_beta, tilde_gamma_tr,
//n                                   Gh_E, SN_h,
//n                                   Qx, tilde_y, tilde_X,  var_coef, c ) ;
//n
//n
//n      }
//n
//n    }
//n
//n    // Clusters updates
//n    for( int i = 0; i<totN; ++i ){
//n      cpp_update_cluster_v2( i, rho_it,
//n                             gTable, Sx, H,
//n                             BETAS_it,  GAMMAS_it,
//n                             BETAS_ext, GAMMAS_ext,
//n                             E_it, E_ext, S_N,
//n                             Y, Z,
//n                             M, sigma,
//n                             Qx,
//n                             c, var_int, var_coef, C) ;
//n
//n    }
//n    // Store
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
//n                                            Rcpp::Named("Beta")  = BETAS_it,
//n                                            Rcpp::Named("Gamma") = GAMMAS_it,
//n                                            Rcpp::Named("Sx")    = Sx,
//n                                            Rcpp::Named("H")     = H) ;
//n      ss += 1 ;
//n    }
//n  }
//n  catIter(S, S, t0) ;
//n  return return_list ;
//n}
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_PPMx_v3( const arma::mat & Y,      const arma::mat & Z,
//n                           const arma::colvec & Qx,
//n                           const double var_int,
//n                           const double var_coef,
//n                           const double M,
//n                           const double sigma,
//n                           const int sample = 1000,  const int burn = 0,    const int thinning = 1,
//n                           const int C = 5,
//n                           const bool verb = true){
//n  if(verb) Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations...";
//n
//n  const unsigned totN = Y.n_rows  ;
//n  const unsigned P    = Y.n_cols  ;
//n
//n  const unsigned S    = burn + thinning * sample;
//n  unsigned ss = 0;
//n
//n  const arma::mat eye_mat(P,P, arma::fill::eye);
//n
//n  arma::cube BETAS_it (P, P, totN, arma::fill::zeros) ;
//n  arma::cube GAMMAS_it(P, P, totN, arma::fill::zeros) ;
//n
//n  arma::cube BETAS_ext (P, P, C);
//n  arma::cube GAMMAS_ext(P, P, C);
//n
//n  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  GAMMAS_it.each_slice()  += eye_mat ;
//n  GAMMAS_ext.each_slice() += eye_mat ;
//n
//n  int H = totN;
//n  arma::uvec rho_it = arma::linspace<arma::uvec>(0, H-1, H) ;
//n  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
//n  arma::uvec   S_N( H, arma::fill::zeros )  ;
//n
//n  arma::mat  Sx = Z;
//n  Rcpp::List return_list(sample) ;
//n  arma::mat    Gh_beta ;
//n  arma::mat    Gh_gamma ;
//n  arma::mat    tilde_X ;
//n  arma::colvec tilde_y ;
//n
//n  std::vector<arma::uvec> Map_for_an_Ising(P) ;
//n
//n  // Creating Mapping constant for pars
//n  std::vector<arma::uvec> Map_to_ones(totN) ;
//n  std::vector<arma::uvec> Map_to_zeros(totN) ;
//n  std::vector<std::vector<arma::uvec>> Map_for_all_Ising(totN) ;
//n  for (int i = 0; i < totN; ++i) {
//n    Map_to_ones[i]  = arma::uvec() ;
//n    Map_to_zeros[i] = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
//n    for( int p = 0; p < P; ++p){
//n      arma::uword v = p ;
//n      Map_for_an_Ising[p] = arma::uvec({v});
//n    }
//n    Map_for_all_Ising[i] = Map_for_an_Ising ;
//n  }
//n
//n  // Creating Mapping for augemented pars
//n  std::vector<arma::uvec> Map_to_ones_EXT(C) ;
//n  std::vector<arma::uvec> Map_to_zeros_EXT(C) ;
//n  std::vector<std::vector<arma::uvec>> Map_for_all_Ising_EXT(C);
//n  for (int c = 0; c < C; ++c) {
//n    Map_to_ones_EXT[c]  = arma::uvec() ;
//n    Map_to_zeros_EXT[c] = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
//n    for( int p = 0; p < P; ++p){
//n      arma::uword v = p ;
//n      Map_for_an_Ising[p] = arma::uvec({v});
//n    }
//n    Map_for_all_Ising_EXT[c] = Map_for_an_Ising;
//n  }
//n
//n  if(verb) Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulations. (Done)" << std::endl ;
//n  MyTimePoint t0;
//n  if(verb) t0 = myClock::now();
//n  for( int s = 0; s<S; ++s){
//n    Rcpp::checkUserInterrupt();
//n    if(verb) catIter(s, S, t0);
//n    // Parameter updates
//n    for( int h = 0; h < H; ++h ){
//n      const arma::mat & subY = Y.rows( arma::find( rho_it == h ) );
//n      arma::mat  & Gh_beta  = BETAS_it.slice(h);
//n      arma::mat  & Gh_gamma = GAMMAS_it.slice(h);
//n      arma::uvec & v_zeros  = Map_to_zeros[h];
//n      arma::uvec & v_ones   = Map_to_ones[h];
//n      std::vector<arma::uvec> & Map_for_current_Ising = Map_for_all_Ising[h];
//n      arma::uword & SN_h = S_N(h);
//n
//n      for( int p = 0; p < P; ++p){
//n        cpp_variable_selection_v3( Gh_beta, Gh_gamma, v_zeros, v_ones,
//n                                   Map_for_current_Ising,
//n                                   SN_h, Qx, Y, var_coef) ;
//n      }
//n
//n      for( int p = 0; p < P; ++p ){
//n         tilde_X = subY;
//n         tilde_y = subY.col(p);
//n         tilde_X.col(p).fill(1.0);
//n         arma::subview_col<double> tilde_gamma   = Gh_gamma.col(p);
//n         arma::subview_col<double> tilde_beta    = Gh_beta.col(p);
//n         arma::subview_row<double> tilde_beta_tr = Gh_beta.row(p);
//n         arma::uvec & Map_for_the_logit = Map_for_current_Ising[p] ;
//n
//n         cpp_update_Omega_v3( p, tilde_gamma,
//n                              tilde_beta, tilde_beta_tr,
//n                              Map_for_the_logit,
//n                              tilde_y, tilde_X,
//n                              var_coef, var_int ) ;
//n      }
//n    }
//n
//n    // Clusters updates
//n    for( int i = 0; i<totN; ++i ){
//n
//n      cpp_update_cluster_v3( i, rho_it, gTable , Sx, H,
//n                                BETAS_it, GAMMAS_it,
//n                                BETAS_ext, GAMMAS_ext,
//n                                S_N,
//n                                Map_to_ones,     Map_to_zeros,      Map_for_all_Ising,
//n                                Map_to_ones_EXT, Map_to_zeros_EXT,  Map_for_all_Ising_EXT,
//n                                Y, Z,
//n                                Qx, M, sigma,
//n                                C, var_int, var_coef ) ;
//n
//n    }
//n
//n    // Store
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
//n                                            Rcpp::Named("Beta")  = BETAS_it,
//n                                            Rcpp::Named("Gamma") = GAMMAS_it,
//n                                            Rcpp::Named("Sx")    = Sx,
//n                                            Rcpp::Named("H")     = H,
//n                                            Rcpp::Named("table") = gTable,
//n                                            Rcpp::Named("S_n")   = S_N,
//n                                            Rcpp::Named("out")   = vector_to_list( Map_to_zeros ),
//n                                            Rcpp::Named("out2")  = vector_to_list( Map_to_ones ),
//n                                            Rcpp::Named("out3")  = vector_of_vectors_to_list(Map_for_all_Ising),
//n                                            Rcpp::Named("out4")  = vector_to_list( Map_to_zeros_EXT),
//n                                            Rcpp::Named("out5")  = vector_to_list( Map_to_ones_EXT),
//n                                            Rcpp::Named("out6")  = vector_of_vectors_to_list(Map_for_all_Ising_EXT));
//n      ss += 1 ;
//n    }
//n  }
//n  if(verb) catIter(S, S, t0) ;
//n  return  return_list;
//n}
//n
//n
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_v3( const arma::mat & Y,
//n                      const arma::colvec & Qx,
//n                      const double var_int,     const double var_coef,
//n                      const int sample = 1000,  const int burn = 0,    const int thinning = 1,
//n                      const int C = 5){
//n
//n  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations...";
//n  const unsigned N = Y.n_rows  ;
//n  const unsigned P = Y.n_cols  ;
//n
//n
//n  const unsigned S    = burn + thinning * sample;
//n  Rcpp::List return_list(S);
//n
//n  unsigned ss = 0;
//n
//n
//n  arma::mat GAMMA(P,P,arma::fill::zeros) ;
//n  arma::mat BETA(P,P,arma::fill::zeros);
//n
//n  GAMMA.diag().fill(1);
//n  BETA.diag() = arma::randn<arma::colvec>(P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  arma::uword current_edges = 0 ;
//n
//n  arma::mat    tilde_X ;
//n  arma::colvec tilde_y ;
//n
//n  arma::uvec Map_to_ones  = arma::uvec() ;
//n  arma::uvec Map_to_zeros = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
//n  std::vector<arma::uvec> Map_for_the_Ising(P) ;
//n
//n  for( int p = 0; p < P; ++p){
//n    arma::uword v = p ;
//n    Map_for_the_Ising[p] = arma::uvec({v});
//n  }
//n
//n  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulations. (Done)" << std::endl ;
//n  MyTimePoint t0 = myClock::now();
//n  for( int s = 0; s<S; ++s){
//n    Rcpp::checkUserInterrupt();
//n    catIter(s, S, t0) ;
//n
//n    // Parameters
//n    for( int p = 0; p < P; ++p){
//n        cpp_variable_selection_v3( BETA, GAMMA,
//n                                   Map_to_zeros, Map_to_ones,
//n                                   Map_for_the_Ising,
//n                                   current_edges, Qx, Y, var_coef) ;
//n      }
//n    for( int p = 0; p < P; ++p ){
//n        tilde_X = Y;
//n        tilde_y = Y.col(p);
//n        tilde_X.col(p).fill(1.0);
//n        arma::subview_col<double> tilde_gamma   = GAMMA.col(p);
//n        arma::subview_col<double> tilde_beta    = BETA.col(p);
//n        arma::subview_row<double> tilde_beta_tr = BETA.row(p);
//n        arma::uvec & Map_for_the_logit = Map_for_the_Ising[p] ;
//n
//n        cpp_update_Omega_v3( p, tilde_gamma,
//n                             tilde_beta, tilde_beta_tr,
//n                             Map_for_the_logit,
//n                             tilde_y, tilde_X,
//n                             var_coef, var_int ) ;
//n      }
//n    // Store
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      return_list[ss] = Rcpp::List::create( Rcpp::Named("Beta")  = BETA,
//n                                            Rcpp::Named("Gamma") = GAMMA );
//n      ss += 1 ;
//n    }
//n  }
//n  catIter(S, S, t0) ;
//n  return  return_list;
//n}
//n
//n
//n// [[Rcpp::depends(RcppArmadillo)]]
//n
//n// --- Esempio di funzioni cohesion con firme diverse ---
//ndouble cohesionA(const arma::vec& x, const arma::mat& Sx, double sigma, double M) {
//n  return arma::sum(x) + sigma + M;
//n}
//n
//ndouble cohesionB(const arma::vec& x, const arma::mat& Sx, double sigma, double M, double p0, double rho) {
//n  return arma::accu(Sx) + sigma + M + p0 + rho;
//n}
//n
//n// [[Rcpp::export]]
//ndouble MCMC(bool useA) {
//n  // --- Parametri fissi ---
//n  double sigma = 1.2, M = 2.0, p0 = 0.5, rho = 0.8;
//n  arma::vec x = {1.0, 2.0, 3.0};
//n  arma::mat Sx = arma::eye(3, 3);
//n
//n  // --- Definizione del riferimento a funzione con firma unificata ---
//n  std::function<double(const arma::mat&, const arma::vec&)> cohesion;
//n
//n  if (useA) {
//n    cohesion = [=](const arma::mat& Sx, const arma::vec& x) {
//n      return cohesionA(x, Sx, sigma, M);
//n    };
//n  } else {
//n    cohesion = [=](const arma::mat& Sx, const arma::vec& x) {
//n      return cohesionB(x, Sx, sigma, M, p0, rho);
//n    };
//n  }
//n
//n  // --- Uso tipico dentro un ciclo MCMC ---
//n  double result = 0.0;
//n  for (int iter = 0; iter < 5; ++iter) {
//n    result += cohesion(Sx, x);  // chiamata uniforme
//n    x += 0.1;                   // esempio: aggiorno x
//n  }
//n
//n  return result;
//n}
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_PPMx_v5( const arma::mat & Y,
//n                           const arma::colvec & predefined_groups,
//n                           const arma::mat & Z,
//n                           const arma::colvec & Qx,
//n                           const double var_int,     const double var_coef,
//n                           const double M,           const double sigma,
//n                           const int sample = 1000,  const int burn = 0,    const int thinning = 1,
//n                           const int C = 5){
//n
//n  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulation...";
//n
//n  // 0.1 Useful Constants
//n  const arma::uword  totN = Y.n_rows  ;
//n  const arma::uword     P = Y.n_cols  ;
//n  const double sd_diag    = std::sqrt(var_int);
//n  const double sd_offdiag = std::sqrt(var_coef);
//n
//n  // 0.2 Initialize Partition and via predefined_groups
//n  arma::uvec  current_partition = arma::conv_to<arma::uvec>::from(predefined_groups);
//n  arma::uvec  rho = arma::conv_to<arma::uvec>::from(arma::unique(predefined_groups)) ;
//n  const arma::uword MaxClusters = rho.n_elem ;
//n  arma::uword H = totN;
//n  arma::uvec  gTable = arma::uvec(H, arma::fill::ones) ;
//n  std::vector<arma::uvec> tmp_map(MaxClusters);
//n  for (unsigned i = 0; i < MaxClusters; ++i) {
//n    arma::uvec sub_partition = arma::find(predefined_groups == i);
//n    tmp_map[i] = sub_partition;
//n  }
//n  const std::vector<arma::uvec> map_for_partition = std::move(tmp_map);
//n  // 0.3.1 Initialize "active" atoms parameters
//n  std::unordered_map<int, arma::uvec> map_for_clusters;
//n  std::unordered_map<int, cluster_parameter> THETA;
//n  THETA.max_load_factor(1.0);
//n  THETA.reserve(MaxClusters);
//n  map_for_clusters.max_load_factor(1.0);
//n  map_for_clusters.reserve(MaxClusters);
//n  arma::uvec tmp_partion ;
//n  for( unsigned i = 0; i < MaxClusters; ++i){
//n    tmp_partion = map_for_partition[i];
//n    map_for_clusters[i] = arma::join_cols(map_for_clusters[i], tmp_partion);
//n    THETA.emplace(i, cluster_parameter( P, Qx, sd_diag, sd_offdiag, 0.0));
//n  }
//n  // 0.3.1 Initialize "augmented" atoms parameters
//n  std::vector<cluster_parameter> ext_THETA;
//n  ext_THETA.resize(C) ;
//n  for( unsigned i = 0; i < C; ++i){
//n    ext_THETA[i] = cluster_parameter( P, Qx, sd_diag, sd_offdiag, 0.0);
//n  }
//n  // 0.4 MCMC constant
//n  const unsigned long long S = burn+thinning*sample;
//n  Rcpp::List return_list(sample);
//n  unsigned ss = 0;
//n  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulation. (Done)" << std::endl ;
//n  // 1 MCMC loop
//n  MyTimePoint t0 = myClock::now();
//n  for( unsigned long long s = 0; s<S; ++s ){
//n    Rcpp::checkUserInterrupt();
//n    catIter(s, S, t0);
//n    // 1.1 Parameter updates
//n    for( arma::uword h = 0; h < H; ++h ){
//n      arma::mat Ysubset = Y.rows(map_for_clusters[h]);
//n      cluster_parameter & PARS = THETA[h];
//n      // 1.1.1 Beta smoothing via Polya Gamma
//n      for( arma::uword p = 0; p < P; ++p){
//n        arma::mat        & BETA_it = PARS.Beta ;
//n        double          & alpha_it = PARS.alpha(p) ;
//n        const arma::uvec & ones_it = PARS.ones ;
//n
//n        cpp_update_Omega_v5( BETA_it, alpha_it,
//n                             p, Ysubset, ones_it,
//n                             var_coef, var_int );
//n      }
//n      // 1.1.1 Stochastic random search for symmetric matrices
//n      for( arma::uword p = 0; p < P; ++p){
//n      }
//n    }
//n    // 1.2 Clusters updates
//n    for( int i = 0; i<totN; ++i ){
//n    }
//n    // Store 1.3
//n    if( ((s+1)>burn) & ((s+1-burn)%thinning==0) ){
//n      return_list[ss] = Rcpp::List::create();
//n      ss += 1 ;
//n    }
//n  }
//n  catIter(S, S, t0) ;
//n  return  return_list;
//n}
//n
//n// [[Rcpp::export]]
//nRcpp::List qIsing_v5( const arma::mat & Y,
//n                      const arma::colvec & Qx,
//n                      const double var_int,     const double var_coef,
//n                      const int sample = 1000,  const int burn = 0,    const int thinning = 1,
//n                      const int C = 5){
//n
//n  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations...";
//n  const unsigned N = Y.n_rows  ;
//n  const unsigned P = Y.n_cols  ;
//n
//n
//n  const unsigned S    = burn + thinning * sample;
//n  Rcpp::List return_list(S);
//n
//n  unsigned ss = 0;
//n
//n
//n  arma::mat GAMMA(P,P,arma::fill::zeros) ;
//n  arma::mat BETA(P,P,arma::fill::zeros);
//n
//n  GAMMA.diag().fill(1);
//n  BETA.diag() = arma::randn<arma::colvec>(P, arma::distr_param(0.0, std::sqrt(var_int)));
//n  arma::uword current_edges = 0 ;
//n
//n  arma::mat    tilde_X ;
//n  arma::colvec tilde_y ;
//n
//n  arma::uvec Map_to_ones  = arma::uvec() ;
//n  arma::uvec Map_to_zeros = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
//n  std::vector<arma::uvec> Map_for_the_Ising(P) ;
//n
//n  for( int p = 0; p < P; ++p){
//n    arma::uword v = p ;
//n    Map_for_the_Ising[p] = arma::uvec({v});
//n  }
//n
//n  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulations. (Done)" << std::endl ;
//n  MyTimePoint t0 = myClock::now();
//n  for( int s = 0; s<S; ++s){
//n    Rcpp::checkUserInterrupt();
//n    catIter(s, S, t0) ;
//n    // Parameters
//n    for( int p = 0; p < P; ++p){
//n      cpp_variable_selection_v3( BETA, GAMMA,
//n                                 Map_to_zeros, Map_to_ones,
//n                                 Map_for_the_Ising,
//n                                 current_edges, Qx, Y, var_coef) ;
//n    }
//n    for( int p = 0; p < P; ++p ){
//n      tilde_X = Y;
//n      tilde_y = Y.col(p);
//n      tilde_X.col(p).fill(1.0);
//n      arma::subview_col<double> tilde_gamma   = GAMMA.col(p);
//n      arma::subview_col<double> tilde_beta    = BETA.col(p);
//n      arma::subview_row<double> tilde_beta_tr = BETA.row(p);
//n      arma::uvec & Map_for_the_logit = Map_for_the_Ising[p] ;
//n
//n      cpp_update_Omega_v3( p, tilde_gamma,
//n                           tilde_beta, tilde_beta_tr,
//n                           Map_for_the_logit,
//n                           tilde_y, tilde_X,
//n                           var_coef, var_int ) ;
//n    }
//n    // Store
//n    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
//n      return_list[ss] = Rcpp::List::create( Rcpp::Named("Beta")  = BETA,
//n                                            Rcpp::Named("Gamma") = GAMMA );
//n      ss += 1 ;
//n    }
//n  }
//n  catIter(S, S, t0) ;
//n  return  return_list;
//n}

