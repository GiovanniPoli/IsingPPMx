#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <truncnorm.h>
#include <mvnorm.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <RcppArmadilloExtensions/sample.h>

#include "randomgen.h"

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

arma::uvec   cpp_sample( const arma::uvec & vec, const int & size, const arma::colvec & prob) {
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

