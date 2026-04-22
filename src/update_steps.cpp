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

using namespace Rcpp;

// Step to update Logit Regression par
void cpp_update_beta_and_alpha( const arma::mat  & YY,
                                arma::mat        & BETA,
                                arma::colvec     & alpha,
                                const arma::uvec & ones,
                                const double var_slab,
                                const double var_int,
                                const double rho) {
  const arma::uword p = ones(0);
  const arma::uvec node = arma::uvec{p};
  const auto & tilde_y      = YY.col(p);
  const auto & beta_reduced = BETA.submat(ones, node ).as_col() ;

  arma::uword _p    = ones.n_rows;
  arma::mat tilde_X = YY.cols(ones);
  tilde_X.col(0).ones();

  arma::colvec  kappa = tilde_y - .5;

  // Creating implied prior on parameters
  const arma::colvec b0 = rho * BETA.submat(node, ones).as_col();
  arma::colvec prec0_diag(_p, arma::fill::value( 1.0 / (var_slab * (1.0 + 1e-10 - rho*rho))));
  prec0_diag(0) = 1.0 / var_int;
  arma::mat Prec0 = arma::diagmat(prec0_diag);

  arma::mat    Xw( YY.n_rows, _p);
  arma::mat    Precn(_p,_p) ;
  arma::mat    Varn(_p,_p)  ;
  arma::colvec Meann(_p) ;


  // Update (Sample):
  // P.G. Augmentation,
  arma::colvec psi = tilde_X*beta_reduced + alpha(p) ;
  arma::colvec w   = cpp_polyagamma_h1_devroye(psi) ;

  // Normal-Normal conjugate models,
  Precn = tilde_X.t()* arma::diagmat(w) * tilde_X + Prec0 ;
  Varn  = arma::inv_sympd(Precn) ;
  Meann = Varn*( tilde_X.t() * kappa + Prec0 * b0 ) ;
  arma::colvec new_alpha_and_beta = cpp_mvrnormArma1(Meann, Varn);

  // Update (Allocation):
  alpha(p) = new_alpha_and_beta(0) ;
  new_alpha_and_beta(0) = 0.0 ;
  BETA.submat(ones, node) = new_alpha_and_beta ;
}

// [[Rcpp::export]]
void cpp_update_global_SRS_debug(
    const arma::mat               & YY,
    arma::mat                     & BETA,
    const arma::colvec            & alpha,
    arma::uvec                    & ones,
    arma::uvec                    & zeros,
    std::vector<arma::uvec>       & mapping,
    const arma::colvec            & logQx,
    const double var_slab, const double rho ) {

  const arma::uword N = YY.n_rows;
  const arma::uword P = YY.n_cols;
  const arma::uword L = P * (P - 1) / 2;
  const arma::uword K  = ones.n_elem;
  const arma::uword K0 = zeros.n_elem;

  double       log_alpha = 0.0;
  const double log_u     = std::log(arma::randu<double>());

  const arma::mat    S0 = {{var_slab, rho}, {rho, var_slab}};
  const arma::colvec b0 = {0.0, 0.0};

  enum MoveType { FORCED_ADD, FORCED_DELATE, FLIP_ADD, FLIP_DELETE, SWAP };
  MoveType move;
  if (K == 0) {
    move = FORCED_ADD;
  } else if (K0 == 0) {
    move = FORCED_DELATE;
  } else {
    const double u = arma::randu<double>();
    if (u > 0.5) {
      move = SWAP;
    } else {
      move = ( 2.0 * u * L < K0 ) ? FLIP_ADD : FLIP_DELETE;
    }
  }

  if (move == FORCED_ADD) {

    const arma::uword pos  = arma::randi<arma::uword>(arma::distr_param(0, L - 1));
    const arma::uword ell  = zeros(pos);
    const auto        edge = index_to_pair(ell);
    arma::uword n1 = edge.first;
    arma::uword n2 = edge.second;

    const arma::colvec new_beta = cpp_mvrnormArma1(b0, S0);
    const double b_reg_n1 = new_beta(0);
    const double b_reg_n2 = new_beta(1);

    log_alpha += cpp_ll_ratio_global_flip(YY, BETA, alpha,
                                          n1, mapping[n1], b_reg_n1,
                                          n2, mapping[n2], b_reg_n2);

    if(log_alpha > log_u){
      // A:
    }

  } else if (move == FORCED_DELATE) {

    const arma::uword pos  = arma::randi<arma::uword>(arma::distr_param(0, L - 1));
    const arma::uword ell  = ones(pos);
    const auto        edge = index_to_pair(ell);
    arma::uword n1 = edge.first;
    arma::uword n2 = edge.second;

    const double b_reg_n1 = BETA(n2, n1);
    const double b_reg_n2 = BETA(n1, n2);

    log_alpha += cpp_ll_ratio_global_flip(YY, BETA, alpha,
                                          n1, mapping[n1], -b_reg_n1,
                                          n2, mapping[n2], -b_reg_n2);
    if(log_alpha > log_u){
      // B:
    }
  } else if (move == FLIP_ADD) {

    const arma::uword pos  = arma::randi<arma::uword>(arma::distr_param(0, K0 - 1));
    const arma::uword ell  = zeros(pos);
    const auto        edge = index_to_pair(ell);
    arma::uword n1 = edge.first;
    arma::uword n2 = edge.second;

    const arma::colvec new_beta = cpp_mvrnormArma1(b0, S0);
    const double b_reg_n1 = new_beta(0);
    const double b_reg_n2 = new_beta(1);

    log_alpha += cpp_ll_ratio_global_flip(YY, BETA, alpha,
                                          n1, mapping[n1], b_reg_n1,
                                          n2, mapping[n2], b_reg_n2);
    if(log_alpha > log_u){
      // C:
    }

  } else if (move == FLIP_DELETE) {

    const arma::uword pos  = arma::randi<arma::uword>(arma::distr_param(0, K - 1));
    const arma::uword ell  = ones(pos);
    const auto        edge = index_to_pair(ell);
    arma::uword n1 = edge.first;
    arma::uword n2 = edge.second;

    const double b_reg_n1 = BETA(n2, n1);
    const double b_reg_n2 = BETA(n1, n2);

    log_alpha += cpp_ll_ratio_global_flip(YY, BETA, alpha,
                                          n1, mapping[n1], -b_reg_n1,
                                          n2, mapping[n2], -b_reg_n2);
    if(log_alpha > log_u){
      // D:
    }

  } else if (move == SWAP) {

    const arma::uword pos_del = arma::randi<arma::uword>(arma::distr_param(0, K - 1));
    const arma::uword ell_del = ones(pos_del);
    const auto edge_del       = index_to_pair(ell_del);
    arma::uword n1 = edge_del.first;
    arma::uword n2 = edge_del.second;

    const arma::uword pos_add = arma::randi<arma::uword>(arma::distr_param(0, K0 - 1));
    const arma::uword ell_add = zeros(pos_add);
    const auto edge_add       = index_to_pair(ell_add);
    arma::uword n3 = edge_add.first;
    arma::uword n4 = edge_add.second;

    const double b_reg_n1_old = BETA(n2, n1);
    const double b_reg_n2_old = BETA(n1, n2);

    const arma::colvec new_beta = cpp_mvrnormArma1(b0, S0);
    const double b_reg_n3_new = new_beta(0);
    const double b_reg_n4_new = new_beta(1);

    log_alpha += cpp_ll_ratio_global_swap(YY, BETA, alpha,
                                          n1, mapping[n1], -b_reg_n1_old,
                                          n2, mapping[n2], -b_reg_n2_old,
                                          n3, mapping[n3],  b_reg_n3_new,
                                          n4, mapping[n4],  b_reg_n4_new);
    if(log_alpha > log_u){
      // E:
    }
  }
}
// void cpp_update_Omega( const int p,
//                        const arma::subview_col<double> tilde_gamma,
//                              arma::subview_col<double> tilde_beta,
//                        const arma::colvec & y, const arma::mat & X,
//                        const double var_slab,  const double var_int) {
//
//   const int P_tot = tilde_gamma.n_elem;
//   const int     N = y.n_elem ;
//
//   arma::colvec new_beta(P_tot, arma::fill::zeros);
//   arma::uvec ones    = arma::find(tilde_gamma == 1) ;
//   arma::uvec int_pos = arma::find( ones == p, 1 );
//
//   const int P = ones.n_elem;
//
//   const arma::mat Xreduced = X.cols(ones);
//
//   arma::colvec  alpha = y - .5;
//   arma::mat    Xalpha = Xreduced;
//
//   Xalpha.each_col()  %= alpha ;
//   arma::colvec Z      = arma::sum(Xalpha,0).t() ;
//
//   arma::mat Prec0(P,P, arma::fill::eye ) ;
//   Prec0.diag()                   = Prec0.diag() * 1.0 / var_slab ;
//   Prec0( int_pos(0), int_pos(0)) = 1.0 / var_int;
//
//   arma::colvec beta = arma::nonzeros(tilde_beta) ;
//
//   arma::colvec psi ;
//   arma::mat    Xw(N,P);
//   arma::mat    Precn(P,P) ;
//   arma::mat    Varn(P,P)  ;
//   arma::colvec Meann(P) ;
//
//   psi            = Xreduced*beta ;
//   arma::colvec w = cpp_polyagamma_h1_devroye(psi) ;
//   Xw             = Xreduced.each_col()%w ;
//   Precn          = Xreduced.t()*Xw+Prec0 ;
//   Varn           = arma::inv_sympd(Precn) ;
//   Meann          = Varn*(Z) ;
//
//   new_beta(ones) = cpp_mvrnormArma1( Meann, Varn);
//   tilde_beta     = new_beta ;
// }
//
//
//
// void cpp_update_Omega_v3( const int p,
//                           const arma::subview_col<double> tilde_gamma,
//                           arma::subview_col<double> tilde_beta,
//                           arma::subview_row<double> tilde_beta_tr,
//                           const arma::uvec & ones,
//                           const arma::colvec & y, const arma::mat & X,
//                           const double var_slab,  const double var_int) {
//
//
//   const int P_tot = tilde_gamma.n_elem;
//   const int     N = y.n_elem ;
//
//   arma::colvec new_beta(P_tot, arma::fill::zeros);
//   arma::uword int_pos = 0;
//
//   const int P = ones.n_elem;
//
//   arma::mat Xreduced = X.cols(ones);
//
//   arma::colvec  alpha = y - .5;
//   arma::mat    Xalpha = Xreduced;
//
//   Xalpha.each_col()  %= alpha ;
//   arma::colvec Z      = arma::sum(Xalpha,0).t() ;
//
//   arma::mat Prec0(P,P, arma::fill::eye ) ;
//   Prec0.diag() = Prec0.diag() * 1.0 / var_slab ;
//   Prec0( int_pos, int_pos ) = 1.0 / var_int;
//
//   arma::colvec beta_all = tilde_beta ;
//   arma::colvec beta = beta_all.rows( ones ) ;
//
//   arma::colvec psi ;
//   arma::mat    Xw(N,P);
//   arma::mat    Precn(P,P) ;
//   arma::mat    Varn(P,P)  ;
//   arma::colvec Meann(P) ;
//
//   psi            = Xreduced*beta ;
//
//
//
//   arma::colvec w = cpp_polyagamma_h1_devroye(psi) ;
//   Xw             = Xreduced.each_col()%w ;
//   Precn          = Xreduced.t()*Xw+Prec0 ;
//   Varn           = arma::inv_sympd(Precn) ;
//   Meann          = Varn*(Z) ;
//
//   new_beta(ones) = cpp_mvrnormArma1( Meann, Varn);
//   tilde_beta     = new_beta ;
// }
//
//
// void cpp_variable_selection_v0( const int p,
//                                 arma::subview_col<double> tilde_gamma,
//                                 arma::subview_col<double> tilde_beta,
//                                 const arma::colvec & y,  const arma::mat & X,
//                                 const double var_slab,
//                                 const double par_pi ) {
//
//   int P = tilde_gamma.n_elem;
//
//   arma::vec u       = arma::randu(2);
//   double log_ar     = std::log(u(1)) ;
//   double log_alpha  = 0.0 ;
//
//   arma::uvec ones ;
//   arma::uvec zeros  = arma::find(tilde_gamma == 0) ;
//
//   int n_1st = arma::as_scalar(arma::randi(arma::distr_param(0, tilde_beta.n_rows-2))); // not p
//   if( n_1st >= p) n_1st +=1;
//
//   int old_value_1st = tilde_gamma(n_1st) ;
//
//   int c1 = static_cast<int>( (u(0) > 0.5) & ((zeros.n_elem > 0) & (zeros.n_elem < (P - 1) )) );
//   int CASE = (c1 << 1) | old_value_1st ;
//
//   int    index_n_2nd ;
//   int    n_2nd ;
//   double beta_proposed ;
//
//   switch (CASE) {
//   case 0: // "flip 0 -> 1";
//     beta_proposed = arma::randn(arma::distr_param(0.0, std::sqrt(var_slab) )) ;
//     log_alpha    += log_likelihood_ratio_add( y, X, tilde_gamma, beta_proposed, n_1st ) ;
//     log_alpha    += -.5 * beta_proposed * beta_proposed / var_slab ;
//     log_alpha    += std::log( par_pi ) ;
//     log_alpha    -= std::log( 1.0 - par_pi ) ;
//     if(log_alpha > log_ar){
//       tilde_beta(n_1st)  = beta_proposed;
//       tilde_gamma(n_1st) = 1 ;
//     }
//     break;
//   case 1: // "flip 1 -> 0";
//     log_alpha    -= log_likelihood_ratio_add( y, X, tilde_gamma, tilde_beta(n_1st), n_1st) ;
//     log_alpha    -= - 0.5 * tilde_beta(n_1st) * tilde_beta(n_1st) / var_slab ;
//     log_alpha    += std::log( par_pi ) - std::log( 1.0 - par_pi ) ;
//     log_alpha    -= std::log( par_pi ) ;
//     log_alpha    += std::log( 1.0 - par_pi ) ;
//     if( log_alpha > log_ar ){
//       tilde_beta(n_1st)  = 0.0;
//       tilde_gamma(n_1st) = 0 ;
//     }
//     break;
//
//   case 2: // "swap 0 -> 1";
//     beta_proposed  = arma::randn(arma::distr_param(0.0, std::sqrt(var_slab))) ;
//     ones  = arma::find(tilde_gamma == 1 ) ;
//     n_2nd = p ;
//     while(n_2nd == p){
//       index_n_2nd = arma::as_scalar(arma::randi(arma::distr_param(0,  ones.n_rows-1))) ;
//       n_2nd       = ones( index_n_2nd ) ;
//     }
//     log_alpha     += log_likelihood_ratio_swap( y, X,
//                                                 tilde_beta,
//                                                 n_1st, beta_proposed,
//                                                 n_2nd, tilde_beta(n_2nd) );
//     log_alpha     += -.5 *    beta_proposed * beta_proposed    / var_slab ;
//     log_alpha     -= -.5 * tilde_beta(n_2nd) * tilde_beta(n_2nd) / var_slab ;
//     if(log_alpha > log_ar){
//       tilde_beta(  n_1st ) = beta_proposed ;
//       tilde_gamma( n_1st ) = 1 ;
//       tilde_beta(  n_2nd ) = 0.0 ;
//       tilde_gamma( n_2nd ) = 0 ;
//     }
//     break;
//   case 3: // "swap 1 -> 0";
//     beta_proposed  = arma::randn(arma::distr_param(0.0, std::sqrt(var_slab))) ;
//     n_2nd   = zeros(arma::as_scalar(arma::randi(arma::distr_param(0, zeros.n_rows-1))));
//
//     log_alpha     += log_likelihood_ratio_swap( y, X,
//                                                 tilde_beta,
//                                                 n_2nd, beta_proposed,
//                                                 n_1st, tilde_beta(n_1st) );
//     log_alpha     += -.5 *     beta_proposed * beta_proposed     / var_slab ;
//     log_alpha     -= -.5 * tilde_beta(n_1st) * tilde_beta(n_1st) / var_slab ;
//
//     if(log_alpha > log_ar){
//       tilde_beta(  n_1st ) = 0.0;
//       tilde_gamma( n_1st ) = 0 ;
//       tilde_beta(  n_2nd ) = beta_proposed;
//       tilde_gamma( n_2nd ) = 1 ;
//     }
//     break;
//   default:
//     Rcpp::stop("Unexpected value used for variable selection step; it must be 0 or 1.");
//   break;
//   }
// }
//
// // [[Rcpp::export]]
// void cpp_update_cluster_v0( arma::uvec & rho,   arma::uvec & table, arma::mat& Sx, int & H,
//                             arma::cube & BETAS, arma::cube & GAMMAS,
//                             arma::cube & B_ext, arma::cube & G_ext,
//                             unsigned index,
//                             const arma::mat & Y, const arma::mat & Z,
//                             const double M, const double sigma, const double c_par,
//                             const int C, const double pi_par, const double var_int, const double var_coef ){
//
//   const unsigned old_ri = rho(index) ;
//   const arma::colvec & yi     = Y.row(index).t() ;
//   const arma::rowvec & row_zi = Z.row(index) ;
//   arma::rowvec row_sx ;
//
//   if( table(old_ri) == 1 ){
//     // Reuse
//     unsigned cc = arma::randi<unsigned>( arma::distr_param(0,C-1) );
//     G_ext.slice(cc) = GAMMAS.slice( old_ri ) ;
//     B_ext.slice(cc) = BETAS.slice( old_ri )  ;
//     // Remove
//     Sx.shed_row(old_ri)    ;
//     table.shed_row(old_ri) ;
//     BETAS.shed_slice(old_ri) ;
//     GAMMAS.shed_slice(old_ri) ;
//     rho( arma::find( rho > old_ri ) ) -= 1 ;
//     H -= 1 ;
//   }else{
//     table(old_ri)  -= 1 ;
//     Sx.row(old_ri) -= Z.row(index) ;
//   } // Rescale
//   const unsigned prob_size = H + C;
//   arma::colvec log_prob(prob_size, arma::fill::zeros) ; // Assigned
//   for( int h = 0; h < H; ++h){
//     row_sx = Sx.row(h);
//     double& value = log_prob(h) ;
//     value += log_gX( table(h), row_zi, row_sx );
//     value += log_cohesion(table(h), sigma) ;
//     value += node_wise_pseudo_ll(yi, BETAS.slice(h)) ;
//   }
//   for( int c = 0; c < C; ++c){   // Augmented + reuse
//     double& value = log_prob(H+c) ;
//     value += empty_log_gX(row_zi);
//     value += empty_log_cohesion(M, H, sigma) ;
//     value += node_wise_pseudo_ll(yi, B_ext.slice(c)) ;
//   }
//   // new ri
//   arma::colvec probs     = arma::exp( log_prob - log_prob.max() ) ;
//   arma::uvec indices     = arma::regspace<arma::uvec>(0, probs.n_elem - 1) ;
//   unsigned sampled_index = cpp_sample_1( indices, probs)  ;
//
//   if( sampled_index >= H){
//
//     BETAS.insert_slices(  H, 1);
//     GAMMAS.insert_slices( H, 1);
//     BETAS.slice( H ) += B_ext.slice( sampled_index - H ) ;
//     GAMMAS.slice( H )+= G_ext.slice( sampled_index - H ) ;
//     // Add
//     table.insert_rows(H,1) ;
//     Sx.insert_rows(H,1) ;
//     Sx.row(H) += row_zi ;
//     table(H)  += 1 ;
//     int n_edges = arma::accu(GAMMAS.slice( H ))  - GAMMAS.n_rows ;
//     rho(index) = H ;
//
//     // Replace
//     std::pair<arma::mat,arma::mat> new_par_pair = cpp_rG0_v0( Y.n_cols, pi_par,
//                                                               std::sqrt(var_int),
//                                                               std::sqrt(var_coef) );
//     G_ext.slice( sampled_index - H ) = new_par_pair.first  ;
//     B_ext.slice( sampled_index - H ) = new_par_pair.second ;
//     H += 1 ;
//   }else{
//     table(sampled_index)  += 1 ;
//     Sx.row(sampled_index) += row_zi ;
//     rho(index)             = sampled_index ;
//   }
// }
//
//
// void cpp_variable_selection_v2( const int p,
//                                 arma::subview_col<double> tilde_gamma,
//                                 arma::subview_col<double> tilde_beta,
//                                 arma::subview_row<double> tilde_gamma_tr,
//                                 arma::mat & Emat,
//                                 arma::uword & S_N,
//                                 const arma::colvec & Qx,
//                                 const arma::colvec & y,  const arma::mat & X,
//                                 const double var_slab,
//                                 const double c ) {
//
//   int P = tilde_gamma.n_elem;
//   int old_SN = S_N;
//
//   arma::vec u       = arma::randu(3);
//   double log_ar     = std::log(u(1)) ;
//   double log_alpha  = 0.0 ;
//
//   arma::uvec ones ;
//   arma::uvec zeros  = arma::find(tilde_gamma == 0) ;
//
//   int n_1st = arma::as_scalar(arma::randi(arma::distr_param(0, tilde_beta.n_rows-2))); // not p
//   if( n_1st >= p) n_1st +=1;
//
//   int old_value_1st = tilde_gamma(n_1st) ;
//   int value_1st_tr   = tilde_gamma_tr(n_1st) ;
//
//   int c1 = static_cast<int>( (u(0) > 0.5) & ((zeros.n_elem > 0) & (zeros.n_elem < (P - 1) )) );
//   int CASE = (c1 << 1) | old_value_1st ;
//
//   int index_n_2nd ;
//   int n_2nd ;
//   int Ei_new ;
//   int Ei_old ;
//   int S_Nnew ;
//
//   double beta_proposed ;
//
//   switch (CASE) {
//   case 0: // "flip 0 -> 1";
//
//     Ei_new = 1 ;              // Need to be active
//     Ei_old = Emat(p, n_1st) ; // it can be 0/1
//     S_Nnew = S_N - Ei_old + Ei_new ;
//     // old_value_1st is 0,
//     // Ei = 0/1 (old)
//     //  if 0, Pr(gi = 0 (old) | Ei = 0 (old)) = 1
//     //  if 1, Pr(gi = 0 (old) | Ei = 1 (old)) = f(x,x)
//
//     beta_proposed = arma::randn(arma::distr_param(0.0, std::sqrt(var_slab))) ;
//     log_alpha    += log_likelihood_ratio_add( y, X, tilde_gamma, beta_proposed, n_1st ) ;
//     log_alpha    += -.5 * beta_proposed * beta_proposed / var_slab ;
//     log_alpha    += log_gammas_mid_1( 1, value_1st_tr, c) ;
//     log_alpha    -= log_gammas_mid_1( 0, value_1st_tr, c) * Ei_old ; // +  (1-Ei(old))* log(1)=0
//     log_alpha    += std::log( Qx(S_Nnew) ) -   std::log( Qx(S_N) ) ;
//     log_alpha    += std::log(0.5); // to active always Ei=1
//
//
//     if(log_alpha > log_ar){
//       tilde_beta(n_1st)  = beta_proposed;
//       tilde_gamma(n_1st) = 1 ;
//       S_N = S_Nnew ;
//       Emat(p, n_1st) = Ei_new ;
//       Emat(n_1st, p) = Ei_new ;
//     }
//     break;
//   case 1: // "flip 1 -> 0";
//     Ei_new = u(2)> 0.5 ;
//     Ei_old = Emat(p, n_1st) ; // it is always 1
//     S_Nnew = S_N - Ei_old + Ei_new ;
//
//     log_alpha    -= log_likelihood_ratio_add( y, X, tilde_gamma, tilde_beta(n_1st), n_1st) ;
//     log_alpha    -= - 0.5 * tilde_beta(n_1st) * tilde_beta(n_1st) / var_slab ;
//     log_alpha    += log_gammas_mid_1( 0, value_1st_tr, c) * Ei_new; // else + log(1)=0
//     log_alpha    -= log_gammas_mid_1( 1, value_1st_tr, c) ; // old_value_1st shold be 1
//     log_alpha    += std::log( Qx(S_Nnew) ) - std::log( Qx(S_N) ) ;
//     log_alpha    += std::log(2); // to active we always have Ei=1, 1/(1/2) => 2
//
//
//     if( log_alpha > log_ar ){
//       tilde_beta(n_1st)  = 0.0;
//       tilde_gamma(n_1st) = 0 ;
//       S_N = S_Nnew ;
//       Emat(p, n_1st) = Ei_new ;
//       Emat(n_1st, p) = Ei_new ;
//     }
//     break;
//
//   case 2: // "swap 0 -> 1";
//     beta_proposed  = arma::randn(arma::distr_param(0.0, std::sqrt(var_slab))) ;
//     ones  = arma::find(tilde_gamma == 1 ) ;
//     n_2nd = p ;
//     while(n_2nd == p){
//       index_n_2nd = arma::as_scalar(arma::randi(arma::distr_param(0,  ones.n_rows-1))) ;
//       n_2nd       = ones( index_n_2nd ) ;
//     }
//
//     Ei_old = Emat( p, n_1st ) ; // From p-> n_1st to be active
//     Ei_new = Emat( p, n_2nd ) ; // From p-> n_2nd, always 1
//
//
//     log_alpha     += log_likelihood_ratio_swap( y, X,
//                                                 tilde_beta,
//                                                 n_1st, beta_proposed,
//                                                 n_2nd, tilde_beta(n_2nd) );
//     log_alpha     += -.5 *    beta_proposed * beta_proposed    / var_slab ;
//     log_alpha     -= -.5 * tilde_beta(n_2nd) * tilde_beta(n_2nd) / var_slab ;
//
//     if(log_alpha > log_ar){
//       tilde_beta(  n_1st ) = beta_proposed ;
//       tilde_gamma( n_1st ) = 1 ;
//       tilde_beta(  n_2nd ) = 0.0 ;
//       tilde_gamma( n_2nd ) = 0 ;
//       Emat(p, n_1st) = Ei_new;
//       Emat(n_1st, p) = Ei_new;
//       Emat(p, n_2nd) = Ei_old;
//       Emat(n_2nd, p) = Ei_old;
//     }
//     break;
//   case 3: // "swap 1 -> 0";
//     beta_proposed  = arma::randn(arma::distr_param(0.0, std::sqrt(var_slab))) ;
//     n_2nd   = zeros(arma::as_scalar(arma::randi(arma::distr_param(0, zeros.n_rows-1))));
//
//     Ei_old = Emat( p, n_1st ) ; // From p-> n_1st to be active, always 1
//     Ei_new = Emat( p, n_2nd ) ; // From p-> n_2nd,
//
//
//     log_alpha     += log_likelihood_ratio_swap( y, X,
//                                                 tilde_beta,
//                                                 n_2nd, beta_proposed,
//                                                 n_1st, tilde_beta(n_1st) );
//     log_alpha     += -.5 *     beta_proposed * beta_proposed     / var_slab ;
//     log_alpha     -= -.5 * tilde_beta(n_1st) * tilde_beta(n_1st) / var_slab ;
//
//     if(log_alpha > log_ar){
//       tilde_beta(  n_1st ) = 0.0;
//       tilde_gamma( n_1st ) = 0 ;
//       tilde_beta(  n_2nd ) = beta_proposed;
//       tilde_gamma( n_2nd ) = 1 ;
//       Emat(p, n_1st) = Ei_new;
//       Emat(n_1st, p) = Ei_new;
//       Emat(p, n_2nd) = Ei_old;
//       Emat(n_2nd, p) = Ei_old;
//     }
//     break;
//   default:
//     Rcpp::stop("Unexpected value used for variable selection step; it must be 0 or 1.");
//   break;
//   }
// }
//
//
//
// void cpp_variable_selection_v3( arma::mat & BETA, arma::mat & GAMMA,
//                                 arma::uvec & Map_to_zeros, arma::uvec & Map_to_ones,
//                                 std::vector<arma::uvec> & Map_for_current_Ising,
//                                 arma::uword & S_N, const arma::colvec & Qx, const arma::mat & YY,
//                                 const double var_slab) {
//   const int P = BETA.n_cols;
//   const int M = P*(P-1)/2 ;
//
//   double u_step = arma::randu<double>();
//   double u      = arma::randu<double>();
//
//   int random_step   = static_cast<int>(u_step > 0.5);
//   int boundary_case = static_cast<int>( (S_N == 0) || (S_N == M) );
//   int CASE          = static_cast<int>( boundary_case*2 + (1-boundary_case)*random_step );
//
//
//   int i1, j1, i2, j2, k1, k2, place_holder;
//   arma::uvec pair1, pair2;
//   double log_alpha_MH = 0;
//   double b0,b1 ;
//
//   switch (CASE) {
//     case 0: // "Pair Flip"
//       k1 = arma::randi<unsigned>(arma::distr_param(0,M-1));
//       pair1 = index_to_pair(k1);
//
//       if( k1 <  Map_to_ones.n_elem  ){
//         // (1,1) -> (0,0)
//         pair1 = index_to_pair( Map_to_ones( k1 ) ) ;
//
//         log_alpha_MH += log_likelihood_ratio_global_flip( YY, BETA, pair1,
//                                                   Map_for_current_Ising[pair1(0)],
//                                                   Map_for_current_Ising[pair1(1)],
//                                                   0.0, 0.0 ) ;
//         log_alpha_MH += .5 * std::pow(BETA(pair1(1), pair1(0)),2) / var_slab ;
//         log_alpha_MH += .5 * std::pow(BETA(pair1(0), pair1(1)),2) / var_slab ;
//         log_alpha_MH += std::log( Qx(S_N - 1) ) -   std::log( Qx(S_N) ) ;
//
//         if( std::exp(log_alpha_MH) > u ){
//           // Elements Mapping
//           remove_j(Map_for_current_Ising[pair1(0)], pair1(1));
//           remove_j(Map_for_current_Ising[pair1(1)], pair1(0));
//           Map_to_zeros.insert_rows( Map_to_zeros.n_elem, 1 ) ;
//           Map_to_zeros( Map_to_zeros.n_elem-1 ) += Map_to_ones( k1 );
//           Map_to_ones.shed_row( k1 );
//           S_N -= 1 ;
//           // Parameter Update
//           BETA(pair1(0),pair1(1)) = 0.0;
//           BETA(pair1(1),pair1(0)) = 0.0;
//           GAMMA(pair1(0),pair1(1)) = 0;
//           GAMMA(pair1(1),pair1(0)) = 0;
//         }
//       }else{
//         // (0,0) -> (1,1)
//         pair1 = index_to_pair( Map_to_zeros( k1 - Map_to_ones.n_elem) ) ;
//
//         b0 = arma::randn<double>(arma::distr_param(0.0, std::sqrt(var_slab) )) ;
//         b1 = arma::randn<double>(arma::distr_param(0.0, std::sqrt(var_slab) )) ;
//
//         log_alpha_MH += log_likelihood_ratio_global_flip( YY, BETA, pair1,
//                                                           Map_for_current_Ising[pair1(0)],
//                                                           Map_for_current_Ising[pair1(1)],
//                                                           b0, b1 ) ;
//         log_alpha_MH += -.5 * std::pow(b0,2) / var_slab ;
//         log_alpha_MH += -.5 * std::pow(b1,2) / var_slab ;
//         log_alpha_MH += std::log( Qx(S_N+1) ) -   std::log( Qx(S_N) ) ;
//
//         if( std::exp(log_alpha_MH) > u){
//
//           // Elements Mapping
//           push_back_j(Map_for_current_Ising[pair1(0)], pair1(1));
//           push_back_j(Map_for_current_Ising[pair1(1)], pair1(0));
//           Map_to_ones.insert_rows( Map_to_ones.n_elem, 1 ) ;
//           Map_to_ones( Map_to_ones.n_elem-1 ) += Map_to_zeros( k1 - Map_to_ones.n_elem +1);
//           Map_to_zeros.shed_row(   k1 - Map_to_ones.n_elem + 1 );
//           S_N += 1 ;
//           // Parameter Update
//           BETA(pair1(0),pair1(1))  = b0;
//           BETA(pair1(1),pair1(0))  = b1;
//           GAMMA(pair1(0),pair1(1)) = 1;
//           GAMMA(pair1(1),pair1(0)) = 1;
//         }
//       }
//       break;
//     case 1: // "Pair Swap"
//       k1 = arma::randi<unsigned>(arma::distr_param(0, Map_to_ones.n_elem-1));
//       k2 = arma::randi<unsigned>(arma::distr_param(0, Map_to_zeros.n_elem-1));
//
//       pair1 = index_to_pair( Map_to_ones(k1)) ;
//       pair2 = index_to_pair( Map_to_zeros(k2)) ;
//
//       b0 = arma::randn<double>(arma::distr_param(0.0, std::sqrt(var_slab) )) ;
//       b1 = arma::randn<double>(arma::distr_param(0.0, std::sqrt(var_slab) )) ;
//
//       log_alpha_MH += log_likelihood_ratio_global_swap( YY, BETA,
//                                                         pair1, pair2,
//                                                         0.0, Map_for_current_Ising[pair1(0)],
//                                                         0.0, Map_for_current_Ising[pair1(1)],
//                                                         b0,  Map_for_current_Ising[pair2(0)],
//                                                         b1,  Map_for_current_Ising[pair2(1)] );
//
//       log_alpha_MH += -.5 * std::pow(b0,2) / var_slab ;
//       log_alpha_MH += -.5 * std::pow(b1,2) / var_slab ;
//       log_alpha_MH -= -.5 * std::pow(BETA(pair1(1), pair1(0)),2) / var_slab ;
//       log_alpha_MH -= -.5 * std::pow(BETA(pair1(0), pair1(1)),2) / var_slab ;
//
//
//       if( std::exp(log_alpha_MH)  > u ){
//         //Element Mapping
//         push_back_j(Map_for_current_Ising[pair2(0)], pair2(1));
//         push_back_j(Map_for_current_Ising[pair2(1)], pair2(0));
//         remove_j(Map_for_current_Ising[pair1(0)], pair1(1));
//         remove_j(Map_for_current_Ising[pair1(1)], pair1(0));
//         place_holder     = Map_to_ones(k1) ;
//         Map_to_ones(k1 ) = Map_to_zeros(k2) ;
//         Map_to_zeros(k2) = place_holder ;
//         // Parameter Update
//         BETA(pair1(0),pair1(1))  = 0.0;
//         BETA(pair1(1),pair1(0))  = 0.0;
//         GAMMA(pair1(0),pair1(1)) = 0;
//         GAMMA(pair1(1),pair1(0)) = 0;
//
//         BETA(pair2(0),pair2(1))  = b0;
//         BETA(pair2(1),pair2(0))  = b1;
//         GAMMA(pair2(0),pair2(1)) = 1;
//         GAMMA(pair2(1),pair2(0)) = 1;
//       }
//       break;
//     case 2: // "Pair forced Flip"
//
//       k1 = arma::randi<unsigned>(arma::distr_param(0,M-1));
//
//       if( S_N == M ){
//         // Forced (1,1) -> (0,0)
//         pair1 = index_to_pair( Map_to_ones(k1)) ;
//         log_alpha_MH += log_likelihood_ratio_global_flip( YY, BETA, pair1,
//                                                           Map_for_current_Ising[pair1(0)],
//                                                           Map_for_current_Ising[pair1(1)],
//                                                           0.0, 0.0 ) ;
//         log_alpha_MH += .5 * std::pow(BETA(pair1(1), pair1(0)),2) / var_slab ;
//         log_alpha_MH += .5 * std::pow(BETA(pair1(0), pair1(1)),2) / var_slab ;
//         log_alpha_MH += std::log( Qx(S_N - 1) ) -   std::log( Qx(S_N) ) ;
//
//         if( std::exp(log_alpha_MH) > u){
//           // Mapping
//           remove_j(Map_for_current_Ising[pair1(0)], pair1(1));
//           remove_j(Map_for_current_Ising[pair1(1)], pair1(0));
//           Map_to_zeros.insert_rows( 0, 1 ) ;
//           Map_to_zeros(0) += Map_to_ones( k1 );
//           Map_to_ones.shed_row( k1 );
//           S_N -= 1 ;
//           // Parameters
//           BETA(pair1(0),pair1(1))  = 0.0;
//           BETA(pair1(1),pair1(0))  = 0.0;
//           GAMMA(pair1(0),pair1(1)) = 0;
//           GAMMA(pair1(1),pair1(0)) = 0;
//         }
//       }else{
//         //Forced (0,0) -> (1,1)
//
//         pair1 = index_to_pair( Map_to_zeros(k1)) ;
//         log_alpha_MH += log_likelihood_ratio_global_flip( YY, BETA, pair1,
//                                                           Map_for_current_Ising[pair1(0)],
//                                                           Map_for_current_Ising[pair1(1)],
//                                                           b0, b1 ) ;
//         log_alpha_MH += -.5 * std::pow(b0,2) / var_slab ;
//         log_alpha_MH += -.5 * std::pow(b1,2) / var_slab ;
//         log_alpha_MH += std::log( Qx(S_N+1) ) -   std::log( Qx(S_N) ) ;
//
//         if( std::exp(log_alpha_MH) > u ){
//           // Mapping
//           push_back_j(Map_for_current_Ising[pair1(0)], pair1(1));
//           push_back_j(Map_for_current_Ising[pair1(1)], pair1(0));
//           Map_to_ones.insert_rows( 0, 1 ) ;
//           Map_to_ones(0) += Map_to_zeros(k1);
//           Map_to_zeros.shed_row(k1);
//           S_N += 1 ;
//           // Update
//           BETA(pair1(0),pair1(1))  = b0;
//           BETA(pair1(1),pair1(0))  = b1;
//           GAMMA(pair1(0),pair1(1)) = 1;
//           GAMMA(pair1(1),pair1(0)) = 1;
//         }
//       }
//     break;
//   }
// }
//
// // [[Rcpp::export]]
// void cpp_update_cluster_v2( unsigned index, // index for  subject to update rho_i
//                             arma::uvec & rho,   arma::uvec & table,
//                             arma::mat  & Sx, int & H,
//                             arma::cube & BETAS, arma::cube & GAMMAS,
//                             arma::cube & B_ext, arma::cube & G_ext,
//                             arma::cube & Ecube, arma::cube & E_ext,
//                             arma::uvec & SN,
//                             const arma::mat & Y, const arma::mat & Z,
//                             const double M, const double sigma,
//                             const arma::colvec & Qx, const double c_par,
//                             const double var_int, const double var_coef, const int C){
//
//   const unsigned old_ri = rho(index) ;
//   const arma::colvec & yi     = Y.row(index).t() ;
//   const arma::rowvec & row_zi = Z.row(index) ;
//   arma::rowvec row_sx ;
//
//   if( table(old_ri) == 1 ){
//     // Reuse
//     unsigned cc = arma::randi<unsigned>( arma::distr_param(0,C-1) );
//     G_ext.slice(cc) = GAMMAS.slice( old_ri ) ;
//     B_ext.slice(cc) = BETAS.slice( old_ri )  ;
//     E_ext.slice(cc) = Ecube.slice( old_ri )  ;
//
//     // Remove
//     Sx.shed_row(old_ri)    ;
//     table.shed_row(old_ri) ;
//     BETAS.shed_slice(old_ri) ;
//     GAMMAS.shed_slice(old_ri) ;
//     Ecube.shed_slice(old_ri) ;
//     SN.shed_row( old_ri ) ;
//
//     rho( arma::find( rho > old_ri ) ) -= 1 ;
//     H -= 1 ;
//   }else{
//     table(old_ri)  -= 1 ;
//     Sx.row(old_ri) -= Z.row(index) ;
//   }
//   // Evaluate Neal 8th
//   const unsigned prob_size = H + C;
//   arma::colvec log_prob(prob_size, arma::fill::zeros) ;
//
//   for( int h = 0; h < H; ++h){
//     row_sx = Sx.row(h);
//     double & value = log_prob(h) ;
//     value += log_gX( table(h), row_zi, row_sx );
//     value += log_cohesion( table(h), sigma) ;
//     value += node_wise_pseudo_ll( yi, BETAS.slice(h)) ;
//   }
//
//   for( int c = 0; c < C; ++c){   // Augmented + reuse
//     double & value = log_prob(H+c) ;
//     value += empty_log_gX(row_zi);
//     value += empty_log_cohesion(M, H, sigma) ;
//     value += node_wise_pseudo_ll(yi, B_ext.slice(c)) ;
//   }
//
//   arma::colvec probs     = arma::exp( log_prob - log_prob.max() ) ;
//   arma::uvec indices     = arma::regspace<arma::uvec>(0, probs.n_elem - 1) ;
//   unsigned sampled_index = cpp_sample_1( indices, probs)  ;
//
//   if( sampled_index >= H){
//
//     BETAS.insert_slices(  H, 1);
//     GAMMAS.insert_slices( H, 1);
//     Ecube.insert_slices( H, 1) ;
//
//     BETAS.slice( H )  += B_ext.slice( sampled_index - H ) ; // value = c + H => c = value - h
//     GAMMAS.slice( H ) += G_ext.slice( sampled_index - H ) ;
//     Ecube.slice( H )  += E_ext.slice( sampled_index - H ) ;
//
//     // Add
//     SN.insert_rows( H, 1) ;
//     SN( H ) = static_cast<int>( arma::accu(Ecube.slice( H ))) ;
//
//     Sx.insert_rows(H,1) ;
//     Sx.row(H) += row_zi ;
//
//     table.insert_rows(H,1) ;
//     table(H)  += 1 ;
//     // Replace
//     std::tuple<arma::mat,arma::mat, arma::mat> new_par_tuple = cpp_rG0_v2( Y.n_cols, Qx, c_par, std::sqrt(var_int), std::sqrt(var_coef) );
//
//     G_ext.slice( sampled_index - H ) = std::get<0>(new_par_tuple) ;
//     B_ext.slice( sampled_index - H ) = std::get<1>(new_par_tuple) ;
//     E_ext.slice( sampled_index - H ) = std::get<2>(new_par_tuple) ;
//
//     // rho_i update!
//     rho(index) = H ;
//     H += 1 ;
//
//   }else{
//     table(sampled_index)  += 1 ;
//     Sx.row(sampled_index) += row_zi ;
//     rho(index)             = sampled_index ;
//   }
// }
//
// unsigned cpp_update_S_N(  const int sum_gammas, const arma::colvec & Qx){
//     const double trunc = Qx.n_elem - sum_gammas ;
//     arma::colvec  probs  = Qx.tail( trunc ) ;
//     arma::uvec v = arma::regspace<arma::uvec>( 0, trunc - 1 );
//     return cpp_sample_1(v, probs) + sum_gammas ;
// }
//
//
// void cpp_update_cluster_v3( unsigned index,
//                             arma::uvec & rho,
//                             arma::uvec & table, arma::mat& Sx, int & H,
//                             arma::cube & BETAS, arma::cube & GAMMAS,
//                             arma::cube & B_ext, arma::cube & G_ext,
//                             arma::uvec & S_N,
//                             std::vector<arma::uvec> & Map_to_ones,
//                             std::vector<arma::uvec> & Map_to_zeros,
//                             std::vector<std::vector<arma::uvec>> & logits_maps,
//                             std::vector<arma::uvec> & Map_to_ones_EXT,
//                             std::vector<arma::uvec> & Map_to_zeros_EXT,
//                             std::vector<std::vector<arma::uvec>> & logits_maps_EXT,
//                             const arma::mat & Y, const arma::mat & Z,
//                             const arma::colvec & Qx,
//                             const double M, const double sigma,
//                             const int C,
//                             const double var_int, const double var_coef ){
//   const unsigned old_ri = rho(index) ;
//   const arma::colvec & yi     = Y.row(index).t() ;
//   const arma::rowvec & row_zi = Z.row(index) ;
//   arma::rowvec row_sx ;
//   const int P =  GAMMAS.n_cols ;
//   if( table(old_ri) == 1 ){
//
//     unsigned cc = arma::randi<unsigned>( arma::distr_param(0,C-1) );
//     G_ext.slice(cc) = GAMMAS.slice( old_ri ) ;
//     B_ext.slice(cc) = BETAS.slice( old_ri )  ;
//     Map_to_ones_EXT[cc]  = Map_to_ones[old_ri];
//     Map_to_zeros_EXT[cc] = Map_to_zeros[old_ri];
//     logits_maps_EXT[cc]  = logits_maps[old_ri];
//
//     Sx.shed_row(old_ri)    ;
//     table.shed_row(old_ri) ;
//     BETAS.shed_slice(old_ri) ;
//     GAMMAS.shed_slice(old_ri) ;
//     S_N.shed_row(old_ri) ;
//
//     Map_to_ones.erase(Map_to_ones.begin()   + old_ri);
//     Map_to_zeros.erase(Map_to_zeros.begin() + old_ri);
//     logits_maps.erase(logits_maps.begin()   + old_ri);
//
//     rho( arma::find( rho > old_ri ) ) -= 1 ;
//     H -= 1 ;
//   }else{
//     table(old_ri)  -= 1 ;
//     Sx.row(old_ri) -= Z.row(index) ;
//   }
//
//   // Evaluate Neal 8th
//   const unsigned prob_size = H + C;
//   arma::colvec log_prob(prob_size, arma::fill::zeros) ;
//   for( int h = 0; h < H; ++h){
//     row_sx = Sx.row(h);
//     double & value = log_prob(h) ;
//     value += log_gX( table(h), row_zi, row_sx );
//     value += log_cohesion( table(h), sigma) ;
//     value += node_wise_pseudo_ll( yi, BETAS.slice(h)) ;
//   }
//   for( int c = 0; c < C; ++c){
//     double & value = log_prob(H+c) ;
//     value += empty_log_gX(row_zi);
//     value += empty_log_cohesion(M, H, sigma) ;
//     value += node_wise_pseudo_ll(yi, B_ext.slice(c)) ;
//   }
//
//   arma::colvec probs     = arma::exp( log_prob - log_prob.max() ) ;
//   arma::uvec indices     = arma::regspace<arma::uvec>(0, probs.n_elem - 1) ;
//   unsigned sampled_index = cpp_sample_1( indices, probs)  ;
//
//   if( sampled_index >= H){
//
//     arma::uword ext_c = sampled_index - H ;
//
//     BETAS.insert_slices(  H, 1);
//     GAMMAS.insert_slices( H, 1);
//
//     BETAS.slice( H )  += B_ext.slice( ext_c ) ;
//     GAMMAS.slice( H ) += G_ext.slice( ext_c ) ;
//
//     // Add
//     S_N.insert_rows( H, 1) ;
//     S_N(H) = static_cast<int>(  (arma::accu( GAMMAS.slice(H) ) - P)/2) ;
//
//     Sx.insert_rows(H,1) ;
//     Sx.row(H) += row_zi ;
//
//     table.insert_rows(H,1) ;
//     table(H)  += 1 ;
//
//     Map_to_ones.push_back(  Map_to_ones_EXT[ext_c]);
//     Map_to_zeros.push_back( Map_to_zeros_EXT[ext_c]);
//     logits_maps.push_back(  logits_maps_EXT[ext_c]);
//
//     // Replace
//     std::tuple<arma::mat,  arma::mat, arma::uvec, arma::uvec, std::vector<arma::uvec>> new_tuple ;
//     new_tuple = cpp_rG0_v3( P, Qx, std::sqrt(var_coef), std::sqrt(var_int) );
//     G_ext.slice( ext_c )    = std::get<0>(new_tuple) ;
//     B_ext.slice( ext_c )    = std::get<1>(new_tuple) ;
//     Map_to_ones_EXT[ext_c]  = std::get<2>(new_tuple) ;
//     Map_to_zeros_EXT[ext_c] = std::get<3>(new_tuple) ;
//     logits_maps_EXT[ext_c]  = std::get<4>(new_tuple) ;
//
//     // rho_i update!
//     rho(index) = H ;
//     H += 1 ;
//
//   }else{
//     table(sampled_index)  += 1 ;
//     Sx.row(sampled_index) += row_zi ;
//     rho(index)             = sampled_index ;
//   }
// }
//
//
// void cpp_update_Omega_v5( arma::mat & BETA,
//                           double & alpha,
//                           const arma::uword p,
//                           const arma::mat & YY,
//                           const arma::uvec & map_ones,
//                           const double var_slab,
//                           const double var_int ) {
//
//   const auto & tilde_y = YY.col(p);
//   arma::uword _p = map_ones.n_rows;
//   arma::mat tilde_X = YY.cols(map_ones);
//
//   // Polya Gamma Constants
//   // must be re-evaluated as can change the subject associated with the cluster
//   arma::colvec  alpha_pg = tilde_y - .5;
//   arma::mat       Xalpha = tilde_X;
//   Xalpha.each_col()  %= alpha_pg ;
//   arma::colvec Z      = arma::sum(Xalpha,0).t() ;
//
//   // old_values values
//   arma::colvec beta_reduced = BETA.submat(map_ones, arma::uvec{p}).as_col() ;
//
//   // Creating implied prior on parameters
//   arma::mat Prec0( _p, _p, arma::fill::eye ) ;
//   Prec0.diag() = Prec0.diag() * 1.0 / var_slab ;
//   Prec0(0,0)   = 1.0 / var_int;
//
//   arma::mat    Xw( YY.n_rows, _p);
//   arma::mat    Precn(_p,_p) ;
//   arma::mat    Varn(_p,_p)  ;
//   arma::colvec Meann(_p) ;
//
//   arma::colvec psi = tilde_X*beta_reduced + alpha ;
//
//   arma::colvec w = cpp_polyagamma_h1_devroye(psi) ;
//   Xw    = tilde_X.each_col()%w ;
//   Precn = tilde_X.t()*Xw+Prec0 ;
//   Varn  = arma::inv_sympd(Precn) ;
//   Meann = Varn*(Z) ;
//
//   BETA.submat(map_ones, arma::uvec{p}) = cpp_mvrnormArma1(Meann, Varn);
//   alpha = BETA(p,p) ;
//   BETA(p,p) = 0.0 ;
// }
//
