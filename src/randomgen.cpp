#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <truncnorm.h>
#include <mvnorm.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <RcppArmadilloExtensions/sample.h>
#include "randomgen.h"
#include "utils.h"


using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
arma::uvec cpp_sample( const arma::uvec & vec, const int & size, const arma::colvec & prob) {
  arma::uvec ret = RcppArmadillo::sample(vec, size, true, prob);
  return(ret);
}

// [[Rcpp::export]]
arma::colvec cpp_mvrnormArma1(const arma::colvec & mu, const arma::mat & sigma) {
  int ncols = sigma.n_cols;
  arma::colvec Z = arma::randn(ncols);
  return mu + arma::chol(sigma).t() * Z;
}

// [[Rcpp::export]]
arma::colvec cpp_polyagamma_h1_truncated(const arma::colvec & z, const int & trunc) {
  int num = z.n_elem ;
  arma::rowvec c_i = arma::square( arma::regspace<arma::rowvec>(1, trunc) - 0.5 )* MyPi2x4;
  arma::vec z2     = arma::square(z);
  arma::mat a_i(num, trunc);
  a_i.each_row()   = c_i;
  a_i.each_col()  += z2;
  a_i = arma::pow(a_i, -1);
  arma::mat    gamma_samples = arma::randg<arma::mat>(num, trunc) ;
  arma::colvec w = 2.0 * arma::sum( gamma_samples % a_i, 1 );
  return w ;
}
double a_coef(int n, double x) {
  double k = n + 0.5;
  if (x > pgTRUNC) {
    return MyPI * k * std::exp(-0.5 * k * k * MyPI * MyPI * x);
  } else {
    double factor = std::pow(2.0 / (MyPI * x), 1.5);
    return factor * MyPI * k * std::exp(-2.0 * k * k / x);
  }
}
double mass_texpon(double Z) {
  double x = 0.64;
  double fz = MyPI * MyPI / 8.0 + 0.5 * Z * Z;

  double sqrt_1_over_x = std::sqrt(1.0 / x);
  double b =   sqrt_1_over_x * (x * Z - 1.0);
  double a = - sqrt_1_over_x * (x * Z + 1.0);

  double x0 = std::log(fz) + fz * x;
  double xb = x0 - Z + std::log(arma::normcdf(b));
  double xa = x0 + Z + std::log(arma::normcdf(a));

  double qdivp = 4.0 / MyPI * (std::exp(xb) + std::exp(xa));
  return 1.0 / (1.0 + qdivp);
}
double rtigauss(double Z, double R = 0.64) {

  Z = std::abs(Z);
  double mu = 1.0 / Z;
  double X = R + 1.0;

  if (mu > R) {

    double E1;
    double E2;
    double alpha = 0.0;

    while( arma::randu() > alpha){

      E1 = arma::randg();
      E2 = arma::randg();
      while ( (E1 * E1) > (2.0 * E2/R) ){
        E1 = arma::randg();
        E2 = arma::randg();
      }

      X     = R / std::pow( 1.0 + R*E1, 2.0);
      alpha = std::exp( -0.5* Z*Z * X);
    }
  }else {
    while (X > R) {

      double lambda = 1.0;
      double Ys = arma::randn() ;
      double Y  = Ys * Ys ;

      X = mu + 0.5 * mu*mu/lambda * Y - 0.5 * mu/lambda * ( std::sqrt(4.0 * mu * lambda * Y + (mu * Y) * (mu * Y))) ;

      if (arma::randu() > mu / (mu + X) ) {
        X = mu*mu / X;
      }

    }
  }
  return X;
}
double rpg_devroye_1(double Z) {
  Z = std::abs(Z) * 0.5;
  double fz = MyPI * MyPI / 8.0 + Z * Z / 2.0;
  int n;
  double X, S, Y;

  while (true) {

    if (arma::randu() < mass_texpon(Z)) {
      X =  pgTRUNC + arma::randg() / fz;
    } else {
      X = rtigauss(Z);
    }

    S = a_coef(0, X);
    Y = arma::randu() * S;

    n = 0;
    while (true) {
      n += 1 ;
      if (n % 2 == 1) {
        S -= a_coef(n, X);
        if (Y <= S) break;
      } else {
        S += a_coef(n, X);
        if (Y > S) break;
      }
    }

    if (Y <= S)
      break;
  }

  return 0.25 * X ;
}

// [[Rcpp::export]]
arma::colvec cpp_polyagamma_h1_devroye(const arma::colvec & z) {
  int num = z.n_elem ;
  arma::colvec w(num) ;
  for( int i = 0; i < num; ++i ){
    w(i) = rpg_devroye_1( z(i) ) ;
  }
  return w ;
}

// [[Rcpp::export]]
arma::uvec cpp_sample( const arma::uvec vec, const int size, const arma::colvec prob) {
  arma::uvec ret = RcppArmadillo::sample(vec, size, true, prob);
  return ret ;
}

// [[Rcpp::export]]
unsigned cpp_sample_1( const arma::uvec & vec, const arma::colvec & prob) {
  unsigned ret = RcppArmadillo::sample(vec, 1, true, prob)(0);
  return ret ;
}


std::pair<arma::mat, arma::mat> cpp_rG0_v0(int dim,
                                           const double off_diagonal_sparsity,
                                           const double sd_offdiag,
                                           const double sd_diag ) {
  arma::sp_mat mat = arma::sprandn<arma::sp_mat>(dim, dim, off_diagonal_sparsity) * sd_offdiag;
  mat.diag()       = arma::randn(dim) * sd_diag;

  arma::mat dense_mat = arma::conv_to<arma::mat>::from(mat);
  arma::mat mask_mat  = arma::conv_to<arma::mat>::from(spones(mat));

  return std::make_pair(std::move(mask_mat), std::move(dense_mat));
}

std::pair<arma::mat,arma::mat> cpp_rG0_v1( const int dim,
                                           const double sparsity,
                                           const double c,
                                           double sd_offdiag,
                                           double sd_diag ) {
  arma::mat ret1 = arma::eye( dim, dim ) ;
  arma::mat ret2( dim, dim, arma::fill::zeros);
  ret2.diag() = arma::randn<arma::vec>(dim, arma::distr_param(0.0, sd_diag));

  // MAPPING CONSTANT
  const int M = dim * (dim - 1) / 2;
  const double disc  = 2.0 * dim - 1.0;
  const double disc2 = disc*disc;

  // Matrix Order
  arma::uvec perm = arma::randperm<arma::uvec>(M) ;
  const double a = ( 1-sparsity ) / c ;
  const double b = (   sparsity ) / c ;

  arma::vec u_t(2);
  int d1, d2, k, i, j;
  double pi_t, off_before;

  for (int pos = 0; pos < M; ++pos) {
    k = static_cast<int>( perm(pos)+1 ) ;
    i = static_cast<int>( std::ceil( (disc - std::sqrt(disc2 - 8.0*k)) / 2.0 ) ) - 1;
    off_before  = (i * (2.0*dim-i + 1))/2.0;
    j = dim-i- static_cast<int>(k-off_before);

    pi_t = r_4beta(a, b, 0.0, 1.0);
    u_t  = arma::randu(2) ;
    d1 = ( pi_t > u_t(0)) ? 0 : 1 ;
    d2 = ( pi_t > u_t(1)) ? 0 : 1 ;
    ret1(j,i) = d1 ;
    ret1(i,j) = d2 ;
    ret2(j,i) = d1 * arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
    ret2(i,j) = d2 * arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
  }
  return std::make_pair(ret1, ret2);
}


std::tuple<arma::mat, arma::mat, arma::mat> cpp_rG0_v2( const int dim,
                                             const arma::colvec & Qx,
                                             const double c,
                                             double sd_offdiag,
                                             double sd_diag ) {
  arma::mat ret1 = arma::eye( dim, dim ) ;
  arma::mat ret2( dim, dim, arma::fill::zeros);
  arma::mat ret3( dim, dim, arma::fill::zeros);

  ret2.diag() = arma::randn<arma::vec>(dim, arma::distr_param(0.0, sd_diag));

  // MAPPING CONSTANT
  const int       M  = dim * (dim - 1) / 2;
  const double disc  = 2.0 *  dim - 1.0;
  const double disc2 = disc*disc;

  // Matrix Order
  arma::uvec   perm    = arma::randperm<arma::uvec>(M) ;
  arma::colvec ordered = arma::regspace( 0, M) ;
  double EE = RcppArmadillo::sample(ordered, 1, true, Qx)(0) ;

  const double a = .5 / c ;
  const double b = .5 / c ;

  arma::vec u_t(2);
  int d1, d2, k, i, j;
  double pi_t, off_before;

  for (int pos = 0; pos < EE; ++pos) {
    k = static_cast<int>( perm(pos)+1 ) ;
    i = static_cast<int>( std::ceil( (disc - std::sqrt(disc2 - 8.0*k)) / 2.0 ) ) - 1;
    off_before  = (i * (2.0*dim-i + 1))/2.0;
    j = dim-i- static_cast<int>(k-off_before);

    ret3(j,i) = 1 ;
    ret3(i,j) = 1 ;

    pi_t = r_4beta(a, b, 0.0, 1.0);

    u_t  = arma::randu(2) ;

    d1 = ( pi_t > u_t(0) ) ? 0 : 1 ;
    d2 = ( pi_t > u_t(1) ) ? 0 : 1 ;

    ret1(j,i) = d1 ;
    ret1(i,j) = d2 ;

    ret2(j,i) = d1 * arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
    ret2(i,j) = d2 * arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
  }
  return std::make_tuple(ret1, ret2, ret3);
}

std::tuple<arma::mat,  arma::mat,
           arma::uvec, arma::uvec,
           std::vector<arma::uvec>> cpp_rG0_v3( const int dim,
                                                const arma::colvec & Qx,
                                                double sd_offdiag,
                                                double sd_diag ) {
  arma::mat ret1 = arma::eye( dim, dim ) ;
  arma::mat ret2( dim, dim, arma::fill::zeros);
  ret2.diag() = arma::randn<arma::vec>(dim, arma::distr_param(0.0, sd_diag));

  const int       M  = dim * (dim - 1) / 2;
  const double disc  = 2.0 *  dim - 1.0;
  const double disc2 = disc*disc;

  arma::uvec   perm    = arma::randperm<arma::uvec>(M) ;
  arma::colvec ordered = arma::regspace( 0, M) ;
  double EE = RcppArmadillo::sample(ordered, 1, true, Qx)(0) ;

  int  k, i, j;

  std::vector<arma::uvec> Map_for_Ising(dim) ;

  for( int p = 0; p < dim; ++p){
      arma::uword v = p ;
      Map_for_Ising[p] = arma::uvec({v});
  }

  arma::uvec pair ;
  for (int pos = 0; pos < EE; ++pos) {
    pair = index_to_pair( perm(pos) ) ;
    i = pair(0) ;
    j = pair(1) ;
    push_back_j(Map_for_Ising[i], j );
    push_back_j(Map_for_Ising[j], i );
    ret1(j,i) = 1 ;
    ret1(i,j) = 1 ;

    ret2(j,i) = arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
    ret2(i,j) = arma::randn<double>( arma::distr_param(0.0, sd_offdiag));
  }

  arma::uvec ones;
  arma::uvec zeros;

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
  return std::make_tuple(ret1, ret2, ones, zeros, Map_for_Ising) ;
}

