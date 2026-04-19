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
//' @param y      Binary response vector of length \eqn{N}, with entries in
//'               \eqn{\{0, 1\}}.
//' @param X      Design matrix of dimension \eqn{N \times P}. For node-wise
//'               quasi-Ising updates, the column corresponding to the response
//'               node is replaced by a column of ones (intercept).
//' @param b0     Prior mean vector of length \eqn{P} for
//'               \eqn{\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{b}_0, B_0)}.
//' @param B0     Prior covariance matrix \eqn{P \times P} for
//'               \eqn{\boldsymbol{\beta}}.
//' @param bstart Starting value for \eqn{\boldsymbol{\beta}}, a vector of
//'               length \eqn{P}.
//' @param sample Number of MCMC draws to retain after burn-in and thinning.
//' @param burn   Number of initial iterations to discard as burn-in.
//'               Default \code{0}.
//' @param thinning Thinning interval: one draw is stored every
//'               \code{thinning} iterations. Default \code{1} (no thinning).
//'
//' @return A numeric matrix of dimension \code{sample} \eqn{\times} \eqn{P}.
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
arma::mat bayes_logistic_regression( const arma::colvec & y,  const arma::mat & X,
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
//' Besag (1975), which factorises into \eqn{P} independent node-wise logistic
//' regressions. Graph structure is learned through a finite-exchangeable-sequence
//' (FES) prior on the edge count
//' Paired interaction coefficients share information through bivariate Normal prior with correlation \eqn{\rho}.
//'
//' @param Y      Binary data matrix of dimension \eqn{N \times P}. Rows are
//'               observations, columns are nodes of the graph.
//' @param Qx     Probability vector of length \eqn{L+1}, where
//'               \eqn{L = P(P-1)/2}, encoding the FES prior on the number of
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
//'     \item{\code{Beta}}{Numeric \eqn{P \times P} matrix of interaction
//'       coefficients. Entry \eqn{(r,c)} is \eqn{\hat\beta_{r,c}}; the diagonal
//'       is zero.}
//'     \item{\code{alpha}}{Numeric vector of length \eqn{P} of node-wise
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
Rcpp::List bayes_qIsing( const arma::mat  & Y,
                         const arma::colvec & Qx,
                         const double sd_int,
                         const double sd_coef,
                         const double rho,
                         const int sample,
                         const int burn     = 0,
                         const int thinning = 1 ) {

  const arma::uword N = Y.n_rows;
  const arma::uword P = Y.n_cols;
  const arma::uword L = P * (P - 1) / 2;
  const arma::uword S = burn + thinning * static_cast<arma::uword>(sample);

  arma::uword ss = 0;
  Rcpp::List  return_list(sample);

  arma::mat    Beta(P, P, arma::fill::zeros);
  arma::colvec alpha(P, arma::fill::zeros);
  arma::uvec   ones  = arma::uvec();
  arma::uvec   zeros = arma::regspace<arma::uvec>(0, L - 1);

  std::vector<arma::uvec> mapping(P);
  for (arma::uword r = 0; r < P; ++r) {
    mapping[r] = arma::uvec({r});
  }

  arma::uword current_edges = 0;

  MyTimePoint t0 = myClock::now();

  for (arma::uword s = 0; s < S; ++s) {
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0);

    for (arma::uword r = 0; r < P; ++r) {
      // TODO: replace with the new helper once available.
      // Expected signature (to be confirmed):
      //   cpp_update_Omega_v6( Beta, alpha,
      //                        r, mapping[r],
      //                        Y,
      //                        sd_coef, sd_int, rho );
    }



    // Store
    if ( ((s + 1) > static_cast<arma::uword>(burn)) &&
         ((s + 1 - burn) % static_cast<arma::uword>(thinning) == 0) ) {
      return_list[ss] = Rcpp::List::create(
        Rcpp::Named("Beta")  = Beta,
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("ones")  = ones  + 1,   // shift for R
        Rcpp::Named("zeros") = zeros + 1
      );
      ss += 1;
    }
  }
  catIter(S, S, t0);
  return return_list;
}













// [[Rcpp::export]]
Rcpp::List qIsing( const arma::mat & Y,
                   const double var_int, const double var_coef, const double par_pi,
                   const int sample, const int burn = 0, const int thinning = 1){
  const int N = Y.n_rows ;
  const int P = Y.n_cols ;

  const int S = burn + thinning * sample;
  int ss = 0;

  MyTimePoint t0 = myClock::now();

  arma::cube  OMEGA(P, P, sample) ;
  arma::cube  DELTA(P, P, sample) ;

  arma::mat    omega(P, P, arma::fill::eye);
  arma::mat    delta(P, P, arma::fill::eye);

  omega = omega * 1e-5 ;

  int current_edges = 0 ;

  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  for( int s = 0; s<S; ++s){
    catIter(s, S, t0) ;
    for( int p = 0; p < P; ++p ){
      Rcpp::checkUserInterrupt();

      tilde_X = Y ;
      tilde_y = Y.col(p) ;
      tilde_X.col(p).fill(1.0) ;

      arma::subview_col<double> tilde_gamma     = delta.col(p) ;
      arma::subview_col<double> tilde_beta      = omega.col(p) ;

      cpp_update_Omega( p, tilde_gamma, tilde_beta,
                        tilde_y, tilde_X, var_coef, var_int ) ;

      cpp_variable_selection_v0( p, tilde_gamma, tilde_beta,
                                 tilde_y, tilde_X, var_coef, par_pi ) ;

   }
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      OMEGA.slice(ss) = omega ;
      DELTA.slice(ss) = delta ;
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return Rcpp::List::create( Rcpp::Named("delta") = DELTA,
                             Rcpp::Named("beta")  = OMEGA ) ;
}

// [[Rcpp::export]]
Rcpp::List qIsing_PPMx_v0( const arma::mat & Y,     const arma::mat & Z,
                           const double & var_int,  const double & var_coef, const double & par_pi,
                           const double & M,        const double & sigma,
                           const int sample = 1000, const int burn = 0, const int thinning = 1,
                           const int C = 5){

  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations";

  const unsigned totN = Y.n_rows ;
  const unsigned P    = Y.n_cols ;

  const unsigned S    = burn + thinning * sample;
  unsigned ss = 0;

  const arma::mat eye_mat(P,P, arma::fill::eye);

  arma::cube  BETAS_it (P, P, totN, arma::fill::zeros) ;
  arma::cube  GAMMAS_it(P, P, totN, arma::fill::zeros) ;

  arma::cube  BETAS_ext (P, P, C) ;
  arma::cube  GAMMAS_ext(P, P, C) ;

  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
  GAMMAS_it.each_slice()  += eye_mat ;
  GAMMAS_ext.each_slice() += eye_mat ;

  int H = totN;

  arma::uvec rho_it = arma::linspace<arma::uvec>(0,H-1,H) ;
  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
  arma::mat  Sx = Z;

  Rcpp::List return_list(sample) ;

  arma::mat    Gh_beta  ;
  arma::mat    Gh_gamma ;
  arma::mat    tilde_X  ;
  arma::colvec tilde_y  ;

  Rcpp::Rcout << ". (Done!)" << std::endl ;

  MyTimePoint t0 = myClock::now();

  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;

    // Parameter updates
    for( int h = 0; h < H; ++h ){
      arma::mat & Gh_beta   = BETAS_it.slice(h) ;
      arma::mat & Gh_gamma  = GAMMAS_it.slice(h);
      const arma::uvec & index_h = arma::find( rho_it == h ) ;
      const arma::mat  & subY = Y.rows( index_h );

      for( int p = 0; p < P; ++p ){

        tilde_X = subY;
        tilde_y = subY.col(p) ;
        tilde_X.col(p).fill(1.0) ;

        arma::subview_col<double> tilde_gamma     = Gh_gamma.col(p) ;
        arma::subview_col<double> tilde_beta      = Gh_beta.col(p) ;

        cpp_update_Omega( p, tilde_gamma, tilde_beta,
                          tilde_y, tilde_X, var_coef, var_int ) ;

        cpp_variable_selection_v0( p, tilde_gamma, tilde_beta,
                                   tilde_y, tilde_X, var_coef, par_pi ) ;


      }
    }
    // Clusters updates
    for( int i = 0; i<totN; ++i ){
    }

    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
                                            Rcpp::Named("Beta")  = BETAS_it,
                                            Rcpp::Named("Gamma") = GAMMAS_it,
                                            Rcpp::Named("Sx")    = Sx,
                                            Rcpp::Named("H")     = H) ;
      ss += 1 ;
    }
   }
  catIter(S, S, t0) ;
  return return_list ;
}

// [[Rcpp::export]]
Rcpp::List qIsing_v1( const arma::mat & Y,
                      const double var_int, const double var_coef,
                      const double par_pi,  const double c,
                      const int sample, const int burn = 0, const int thinning = 1){
  const int N = Y.n_rows ;
  const int P = Y.n_cols ;

  const int S = burn + thinning * sample;
  int ss = 0;

  MyTimePoint t0 = myClock::now();

  arma::cube  OMEGA(P, P, sample) ;
  arma::cube  DELTA(P, P, sample) ;

  arma::mat    omega(P, P, arma::fill::eye);
  arma::mat    delta(P, P, arma::fill::eye);

  omega = omega * 1e-5 ;

  int current_edges = 0 ;

  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  for( int s = 0; s<S; ++s){
    catIter(s, S, t0) ;
    for( int p = 0; p < P; ++p ){
      Rcpp::checkUserInterrupt();

      tilde_X = Y ;
      tilde_y = Y.col(p) ;
      tilde_X.col(p).fill(1.0) ;

      arma::subview_col<double> tilde_gamma     = delta.col(p) ;
      arma::subview_col<double> tilde_beta      = omega.col(p) ;

      cpp_update_Omega( p, tilde_gamma, tilde_beta,
                        tilde_y, tilde_X, var_coef, var_int ) ;

      cpp_variable_selection_v0( p, tilde_gamma, tilde_beta,
                                 tilde_y, tilde_X, var_coef, par_pi ) ;

    }
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      OMEGA.slice(ss) = omega ;
      DELTA.slice(ss) = delta ;
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return Rcpp::List::create( Rcpp::Named("delta") = DELTA,
                             Rcpp::Named("beta")  = OMEGA ) ;
}

// [[Rcpp::export]]
Rcpp::List qIsing_PPMx_v1( const arma::mat & Y,     const arma::mat & Z,
                           const double & var_int,  const double & var_coef,
                           const double & par_pi,   const double & c,
                           const double & M,        const double & sigma,
                           const int sample = 1000, const int burn = 0, const int thinning = 1,
                           const int C = 5){

  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations";

  const unsigned totN = Y.n_rows ;
  const unsigned P    = Y.n_cols ;

  const unsigned S    = burn + thinning * sample;
  unsigned ss = 0;

  const arma::mat eye_mat(P,P, arma::fill::eye);

  arma::cube  BETAS_it (P, P, totN, arma::fill::zeros) ;
  arma::cube  GAMMAS_it(P, P, totN, arma::fill::zeros) ;

  arma::cube  BETAS_ext (P, P, C) ;
  arma::cube  GAMMAS_ext(P, P, C) ;

  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, var_int));
  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, var_int));
  GAMMAS_it.each_slice()  += eye_mat ;
  GAMMAS_ext.each_slice() += eye_mat ;

  int H = totN;
  arma::uvec rho_it = arma::linspace<arma::uvec>(0,H-1,H) ;
  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
  arma::uvec current_edges(H, arma::fill::zeros) ;
  arma::mat  Sx = Z;

  Rcpp::List return_list(sample) ;

  arma::mat    Gh_beta ;
  arma::mat    Gh_gamma ;
  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  Rcpp::Rcout << ". (Done!)" << std::endl ;

  MyTimePoint t0 = myClock::now();

  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;

    // Parameter updates
    for( int h = 0; h < H; ++h ){

      arma::mat & Gh_beta   = BETAS_it.slice(h) ;
      arma::mat & Gh_gamma  = GAMMAS_it.slice(h);
      const arma::mat & subY = Y.rows( arma::find( rho_it == h ));

      for( int p = 0; p < P; ++p ){

        tilde_X = subY;
        tilde_y = subY.col(p) ;
        tilde_X.col(p).fill(1.0) ;
        arma::uword & edges_h = current_edges(h);

        arma::subview_col<double> tilde_gamma     = Gh_gamma.col(p) ;
        arma::subview_row<double> tilde_gamma_tr  = Gh_gamma.row(p) ;
        arma::subview_col<double> tilde_beta      = Gh_beta.col(p) ;

      }
    }

    // Clusters updates
    for( int i = 0; i<totN; ++i ){
    }
    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
                                            Rcpp::Named("Beta")  = BETAS_it,
                                            Rcpp::Named("Gamma") = GAMMAS_it,
                                            Rcpp::Named("Sx")    = Sx,
                                            Rcpp::Named("H")     = H) ;
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return return_list ;
}

// [[Rcpp::export]]
Rcpp::List qIsing_PPMx_v2( const arma::mat & Y,       const arma::mat & Z,
                           const double & var_int,    const double & var_coef,
                           const arma::colvec & Qx,   const double & c,
                           const double & M,          const double & sigma,
                           const int sample = 1000,   const int burn = 0,    const int thinning = 1,
                           const int C = 5){

  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations";

  const unsigned totN = Y.n_rows ;
  const unsigned P    = Y.n_cols ;

  const unsigned S    = burn + thinning * sample;
  unsigned ss = 0;

  const arma::mat eye_mat(P,P, arma::fill::eye);

  arma::cube BETAS_it (P, P, totN, arma::fill::zeros) ;
  arma::cube GAMMAS_it(P, P, totN, arma::fill::zeros) ;
  arma::cube E_it(P, P, totN, arma::fill::zeros) ;

  arma::cube BETAS_ext (P, P, C) ;
  arma::cube GAMMAS_ext(P, P, C) ;
  arma::cube E_ext(P, P, C) ;

  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
  GAMMAS_it.each_slice()  += eye_mat ;
  GAMMAS_ext.each_slice() += eye_mat ;

  int H = totN;
  arma::uvec rho_it = arma::linspace<arma::uvec>(0, H-1, H) ;
  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
  arma::uvec   S_N( H, arma::fill::zeros )  ;

  arma::mat  Sx = Z;

  Rcpp::List return_list(sample) ;

  arma::mat    Gh_beta ;
  arma::mat    Gh_gamma ;
  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  Rcpp::Rcout << ". (Done!)" << std::endl ;

  MyTimePoint t0 = myClock::now();

  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;
    // Parameter updates
    for( int h = 0; h < H; ++h ){

      arma::mat & Gh_beta    = BETAS_it.slice(h) ;
      arma::mat & Gh_gamma   = GAMMAS_it.slice(h);
      arma::mat & Gh_E       = E_it.slice(h);
      const arma::mat & subY = Y.rows( arma::find( rho_it == h ));
      arma::uword & SN_h =   S_N(h);

      for(int p = 0; p < P; ++p ){

        tilde_X = subY;
        tilde_y = subY.col(p) ;
        tilde_X.col(p).fill(1.0) ;

        arma::subview_col<double> tilde_gamma     = Gh_gamma.col(p) ;
        arma::subview_row<double> tilde_gamma_tr  = Gh_gamma.row(p) ;
        arma::subview_col<double> tilde_beta      = Gh_beta.col(p) ;

        cpp_update_Omega( p, tilde_gamma, tilde_beta, tilde_y, tilde_X, var_coef, var_int ) ;


        cpp_variable_selection_v2( p,
                                   tilde_gamma, tilde_beta, tilde_gamma_tr,
                                   Gh_E, SN_h,
                                   Qx, tilde_y, tilde_X,  var_coef, c ) ;


      }

    }

    // Clusters updates
    for( int i = 0; i<totN; ++i ){
      cpp_update_cluster_v2( i, rho_it,
                             gTable, Sx, H,
                             BETAS_it,  GAMMAS_it,
                             BETAS_ext, GAMMAS_ext,
                             E_it, E_ext, S_N,
                             Y, Z,
                             M, sigma,
                             Qx,
                             c, var_int, var_coef, C) ;

    }
    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
                                            Rcpp::Named("Beta")  = BETAS_it,
                                            Rcpp::Named("Gamma") = GAMMAS_it,
                                            Rcpp::Named("Sx")    = Sx,
                                            Rcpp::Named("H")     = H) ;
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return return_list ;
}

// [[Rcpp::export]]
Rcpp::List qIsing_PPMx_v3( const arma::mat & Y,      const arma::mat & Z,
                           const arma::colvec & Qx,
                           const double var_int,
                           const double var_coef,
                           const double M,
                           const double sigma,
                           const int sample = 1000,  const int burn = 0,    const int thinning = 1,
                           const int C = 5,
                           const bool verb = true){
  if(verb) Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations...";

  const unsigned totN = Y.n_rows  ;
  const unsigned P    = Y.n_cols  ;

  const unsigned S    = burn + thinning * sample;
  unsigned ss = 0;

  const arma::mat eye_mat(P,P, arma::fill::eye);

  arma::cube BETAS_it (P, P, totN, arma::fill::zeros) ;
  arma::cube GAMMAS_it(P, P, totN, arma::fill::zeros) ;

  arma::cube BETAS_ext (P, P, C);
  arma::cube GAMMAS_ext(P, P, C);

  BETAS_it.each_slice()   += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
  BETAS_ext.each_slice()  += eye_mat % arma::randn(P, P, arma::distr_param(0.0, std::sqrt(var_int)));
  GAMMAS_it.each_slice()  += eye_mat ;
  GAMMAS_ext.each_slice() += eye_mat ;

  int H = totN;
  arma::uvec rho_it = arma::linspace<arma::uvec>(0, H-1, H) ;
  arma::uvec gTable = arma::uvec(H, arma::fill::ones) ;
  arma::uvec   S_N( H, arma::fill::zeros )  ;

  arma::mat  Sx = Z;
  Rcpp::List return_list(sample) ;
  arma::mat    Gh_beta ;
  arma::mat    Gh_gamma ;
  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  std::vector<arma::uvec> Map_for_an_Ising(P) ;

  // Creating Mapping constant for pars
  std::vector<arma::uvec> Map_to_ones(totN) ;
  std::vector<arma::uvec> Map_to_zeros(totN) ;
  std::vector<std::vector<arma::uvec>> Map_for_all_Ising(totN) ;
  for (int i = 0; i < totN; ++i) {
    Map_to_ones[i]  = arma::uvec() ;
    Map_to_zeros[i] = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
    for( int p = 0; p < P; ++p){
      arma::uword v = p ;
      Map_for_an_Ising[p] = arma::uvec({v});
    }
    Map_for_all_Ising[i] = Map_for_an_Ising ;
  }

  // Creating Mapping for augemented pars
  std::vector<arma::uvec> Map_to_ones_EXT(C) ;
  std::vector<arma::uvec> Map_to_zeros_EXT(C) ;
  std::vector<std::vector<arma::uvec>> Map_for_all_Ising_EXT(C);
  for (int c = 0; c < C; ++c) {
    Map_to_ones_EXT[c]  = arma::uvec() ;
    Map_to_zeros_EXT[c] = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
    for( int p = 0; p < P; ++p){
      arma::uword v = p ;
      Map_for_an_Ising[p] = arma::uvec({v});
    }
    Map_for_all_Ising_EXT[c] = Map_for_an_Ising;
  }

  if(verb) Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulations. (Done)" << std::endl ;
  MyTimePoint t0;
  if(verb) t0 = myClock::now();
  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    if(verb) catIter(s, S, t0);
    // Parameter updates
    for( int h = 0; h < H; ++h ){
      const arma::mat & subY = Y.rows( arma::find( rho_it == h ) );
      arma::mat  & Gh_beta  = BETAS_it.slice(h);
      arma::mat  & Gh_gamma = GAMMAS_it.slice(h);
      arma::uvec & v_zeros  = Map_to_zeros[h];
      arma::uvec & v_ones   = Map_to_ones[h];
      std::vector<arma::uvec> & Map_for_current_Ising = Map_for_all_Ising[h];
      arma::uword & SN_h = S_N(h);

      for( int p = 0; p < P; ++p){
        cpp_variable_selection_v3( Gh_beta, Gh_gamma, v_zeros, v_ones,
                                   Map_for_current_Ising,
                                   SN_h, Qx, Y, var_coef) ;
      }

      for( int p = 0; p < P; ++p ){
         tilde_X = subY;
         tilde_y = subY.col(p);
         tilde_X.col(p).fill(1.0);
         arma::subview_col<double> tilde_gamma   = Gh_gamma.col(p);
         arma::subview_col<double> tilde_beta    = Gh_beta.col(p);
         arma::subview_row<double> tilde_beta_tr = Gh_beta.row(p);
         arma::uvec & Map_for_the_logit = Map_for_current_Ising[p] ;

         cpp_update_Omega_v3( p, tilde_gamma,
                              tilde_beta, tilde_beta_tr,
                              Map_for_the_logit,
                              tilde_y, tilde_X,
                              var_coef, var_int ) ;
      }
    }

    // Clusters updates
    for( int i = 0; i<totN; ++i ){

      cpp_update_cluster_v3( i, rho_it, gTable , Sx, H,
                                BETAS_it, GAMMAS_it,
                                BETAS_ext, GAMMAS_ext,
                                S_N,
                                Map_to_ones,     Map_to_zeros,      Map_for_all_Ising,
                                Map_to_ones_EXT, Map_to_zeros_EXT,  Map_for_all_Ising_EXT,
                                Y, Z,
                                Qx, M, sigma,
                                C, var_int, var_coef ) ;

    }

    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      return_list[ss] = Rcpp::List::create( Rcpp::Named("rho")   = rho_it,
                                            Rcpp::Named("Beta")  = BETAS_it,
                                            Rcpp::Named("Gamma") = GAMMAS_it,
                                            Rcpp::Named("Sx")    = Sx,
                                            Rcpp::Named("H")     = H,
                                            Rcpp::Named("table") = gTable,
                                            Rcpp::Named("S_n")   = S_N,
                                            Rcpp::Named("out")   = vector_to_list( Map_to_zeros ),
                                            Rcpp::Named("out2")  = vector_to_list( Map_to_ones ),
                                            Rcpp::Named("out3")  = vector_of_vectors_to_list(Map_for_all_Ising),
                                            Rcpp::Named("out4")  = vector_to_list( Map_to_zeros_EXT),
                                            Rcpp::Named("out5")  = vector_to_list( Map_to_ones_EXT),
                                            Rcpp::Named("out6")  = vector_of_vectors_to_list(Map_for_all_Ising_EXT));
      ss += 1 ;
    }
  }
  if(verb) catIter(S, S, t0) ;
  return  return_list;
}



// [[Rcpp::export]]
Rcpp::List qIsing_v3( const arma::mat & Y,
                      const arma::colvec & Qx,
                      const double var_int,     const double var_coef,
                      const int sample = 1000,  const int burn = 0,    const int thinning = 1,
                      const int C = 5){

  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations...";
  const unsigned N = Y.n_rows  ;
  const unsigned P = Y.n_cols  ;


  const unsigned S    = burn + thinning * sample;
  Rcpp::List return_list(S);

  unsigned ss = 0;


  arma::mat GAMMA(P,P,arma::fill::zeros) ;
  arma::mat BETA(P,P,arma::fill::zeros);

  GAMMA.diag().fill(1);
  BETA.diag() = arma::randn<arma::colvec>(P, arma::distr_param(0.0, std::sqrt(var_int)));
  arma::uword current_edges = 0 ;

  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  arma::uvec Map_to_ones  = arma::uvec() ;
  arma::uvec Map_to_zeros = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
  std::vector<arma::uvec> Map_for_the_Ising(P) ;

  for( int p = 0; p < P; ++p){
    arma::uword v = p ;
    Map_for_the_Ising[p] = arma::uvec({v});
  }

  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulations. (Done)" << std::endl ;
  MyTimePoint t0 = myClock::now();
  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;

    // Parameters
    for( int p = 0; p < P; ++p){
        cpp_variable_selection_v3( BETA, GAMMA,
                                   Map_to_zeros, Map_to_ones,
                                   Map_for_the_Ising,
                                   current_edges, Qx, Y, var_coef) ;
      }
    for( int p = 0; p < P; ++p ){
        tilde_X = Y;
        tilde_y = Y.col(p);
        tilde_X.col(p).fill(1.0);
        arma::subview_col<double> tilde_gamma   = GAMMA.col(p);
        arma::subview_col<double> tilde_beta    = BETA.col(p);
        arma::subview_row<double> tilde_beta_tr = BETA.row(p);
        arma::uvec & Map_for_the_logit = Map_for_the_Ising[p] ;

        cpp_update_Omega_v3( p, tilde_gamma,
                             tilde_beta, tilde_beta_tr,
                             Map_for_the_logit,
                             tilde_y, tilde_X,
                             var_coef, var_int ) ;
      }
    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      return_list[ss] = Rcpp::List::create( Rcpp::Named("Beta")  = BETA,
                                            Rcpp::Named("Gamma") = GAMMA );
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return  return_list;
}


// [[Rcpp::depends(RcppArmadillo)]]

// --- Esempio di funzioni cohesion con firme diverse ---
double cohesionA(const arma::vec& x, const arma::mat& Sx, double sigma, double M) {
  return arma::sum(x) + sigma + M;
}

double cohesionB(const arma::vec& x, const arma::mat& Sx, double sigma, double M, double p0, double rho) {
  return arma::accu(Sx) + sigma + M + p0 + rho;
}

// [[Rcpp::export]]
double MCMC(bool useA) {
  // --- Parametri fissi ---
  double sigma = 1.2, M = 2.0, p0 = 0.5, rho = 0.8;
  arma::vec x = {1.0, 2.0, 3.0};
  arma::mat Sx = arma::eye(3, 3);

  // --- Definizione del riferimento a funzione con firma unificata ---
  std::function<double(const arma::mat&, const arma::vec&)> cohesion;

  if (useA) {
    cohesion = [=](const arma::mat& Sx, const arma::vec& x) {
      return cohesionA(x, Sx, sigma, M);
    };
  } else {
    cohesion = [=](const arma::mat& Sx, const arma::vec& x) {
      return cohesionB(x, Sx, sigma, M, p0, rho);
    };
  }

  // --- Uso tipico dentro un ciclo MCMC ---
  double result = 0.0;
  for (int iter = 0; iter < 5; ++iter) {
    result += cohesion(Sx, x);  // chiamata uniforme
    x += 0.1;                   // esempio: aggiorno x
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::List qIsing_PPMx_v5( const arma::mat & Y,
                           const arma::colvec & predefined_groups,
                           const arma::mat & Z,
                           const arma::colvec & Qx,
                           const double var_int,     const double var_coef,
                           const double M,           const double sigma,
                           const int sample = 1000,  const int burn = 0,    const int thinning = 1,
                           const int C = 5){

  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulation...";

  // 0.1 Useful Constants
  const arma::uword  totN = Y.n_rows  ;
  const arma::uword     P = Y.n_cols  ;
  const double sd_diag    = std::sqrt(var_int);
  const double sd_offdiag = std::sqrt(var_coef);

  // 0.2 Initialize Partition and via predefined_groups
  arma::uvec  current_partition = arma::conv_to<arma::uvec>::from(predefined_groups);
  arma::uvec  rho = arma::conv_to<arma::uvec>::from(arma::unique(predefined_groups)) ;
  const arma::uword MaxClusters = rho.n_elem ;
  arma::uword H = totN;
  arma::uvec  gTable = arma::uvec(H, arma::fill::ones) ;
  std::vector<arma::uvec> tmp_map(MaxClusters);
  for (unsigned i = 0; i < MaxClusters; ++i) {
    arma::uvec sub_partition = arma::find(predefined_groups == i);
    tmp_map[i] = sub_partition;
  }
  const std::vector<arma::uvec> map_for_partition = std::move(tmp_map);
  // 0.3.1 Initialize "active" atoms parameters
  std::unordered_map<int, arma::uvec> map_for_clusters;
  std::unordered_map<int, cluster_parameter> THETA;
  THETA.max_load_factor(1.0);
  THETA.reserve(MaxClusters);
  map_for_clusters.max_load_factor(1.0);
  map_for_clusters.reserve(MaxClusters);
  arma::uvec tmp_partion ;
  for( unsigned i = 0; i < MaxClusters; ++i){
    tmp_partion = map_for_partition[i];
    map_for_clusters[i] = arma::join_cols(map_for_clusters[i], tmp_partion);
    THETA.emplace(i, cluster_parameter( P, Qx, sd_diag, sd_offdiag, 0.0));
  }
  // 0.3.1 Initialize "augmented" atoms parameters
  std::vector<cluster_parameter> ext_THETA;
  ext_THETA.resize(C) ;
  for( unsigned i = 0; i < C; ++i){
    ext_THETA[i] = cluster_parameter( P, Qx, sd_diag, sd_offdiag, 0.0);
  }
  // 0.4 MCMC constant
  const unsigned long long S = burn+thinning*sample;
  Rcpp::List return_list(sample);
  unsigned ss = 0;
  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulation. (Done)" << std::endl ;
  // 1 MCMC loop
  MyTimePoint t0 = myClock::now();
  for( unsigned long long s = 0; s<S; ++s ){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0);
    // 1.1 Parameter updates
    for( arma::uword h = 0; h < H; ++h ){
      arma::mat Ysubset = Y.rows(map_for_clusters[h]);
      cluster_parameter & PARS = THETA[h];
      // 1.1.1 Beta smoothing via Polya Gamma
      for( arma::uword p = 0; p < P; ++p){
        arma::mat        & BETA_it = PARS.Beta ;
        double          & alpha_it = PARS.alpha(p) ;
        const arma::uvec & ones_it = PARS.ones ;

        cpp_update_Omega_v5( BETA_it, alpha_it,
                             p, Ysubset, ones_it,
                             var_coef, var_int );
      }
      // 1.1.1 Stochastic random search for symmetric matrices
      for( arma::uword p = 0; p < P; ++p){
      }
    }
    // 1.2 Clusters updates
    for( int i = 0; i<totN; ++i ){
    }
    // Store 1.3
    if( ((s+1)>burn) & ((s+1-burn)%thinning==0) ){
      return_list[ss] = Rcpp::List::create();
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return  return_list;
}

// [[Rcpp::export]]
Rcpp::List qIsing_v5( const arma::mat & Y,
                      const arma::colvec & Qx,
                      const double var_int,     const double var_coef,
                      const int sample = 1000,  const int burn = 0,    const int thinning = 1,
                      const int C = 5){

  Rcpp::Rcout << "Inzializing starting values and constant for MCMC simulations...";
  const unsigned N = Y.n_rows  ;
  const unsigned P = Y.n_cols  ;


  const unsigned S    = burn + thinning * sample;
  Rcpp::List return_list(S);

  unsigned ss = 0;


  arma::mat GAMMA(P,P,arma::fill::zeros) ;
  arma::mat BETA(P,P,arma::fill::zeros);

  GAMMA.diag().fill(1);
  BETA.diag() = arma::randn<arma::colvec>(P, arma::distr_param(0.0, std::sqrt(var_int)));
  arma::uword current_edges = 0 ;

  arma::mat    tilde_X ;
  arma::colvec tilde_y ;

  arma::uvec Map_to_ones  = arma::uvec() ;
  arma::uvec Map_to_zeros = arma::regspace<arma::uvec>(0, P*(P-1)/2-1);
  std::vector<arma::uvec> Map_for_the_Ising(P) ;

  for( int p = 0; p < P; ++p){
    arma::uword v = p ;
    Map_for_the_Ising[p] = arma::uvec({v});
  }

  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC simulations. (Done)" << std::endl ;
  MyTimePoint t0 = myClock::now();
  for( int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;
    // Parameters
    for( int p = 0; p < P; ++p){
      cpp_variable_selection_v3( BETA, GAMMA,
                                 Map_to_zeros, Map_to_ones,
                                 Map_for_the_Ising,
                                 current_edges, Qx, Y, var_coef) ;
    }
    for( int p = 0; p < P; ++p ){
      tilde_X = Y;
      tilde_y = Y.col(p);
      tilde_X.col(p).fill(1.0);
      arma::subview_col<double> tilde_gamma   = GAMMA.col(p);
      arma::subview_col<double> tilde_beta    = BETA.col(p);
      arma::subview_row<double> tilde_beta_tr = BETA.row(p);
      arma::uvec & Map_for_the_logit = Map_for_the_Ising[p] ;

      cpp_update_Omega_v3( p, tilde_gamma,
                           tilde_beta, tilde_beta_tr,
                           Map_for_the_logit,
                           tilde_y, tilde_X,
                           var_coef, var_int ) ;
    }
    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      return_list[ss] = Rcpp::List::create( Rcpp::Named("Beta")  = BETA,
                                            Rcpp::Named("Gamma") = GAMMA );
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return  return_list;
}

