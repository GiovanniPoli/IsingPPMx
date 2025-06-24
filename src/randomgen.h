#ifndef randomgen_H
#define randomgen_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

const double MyPI = 3.141592653589793238462643383279502884197;
const double pgTRUNC = 0.64 ;
const double MyPi2x4 = 4 * MyPI * MyPI ;


arma::uvec   cpp_sample(const arma::uvec vec, const int size, const arma::colvec prob);
arma::colvec cpp_polyagamma_h1_truncated(const arma::colvec& z, const int& trunc);
arma::colvec cpp_polyagamma_h1_devroye(const arma::colvec &  z) ;
arma::colvec cpp_mvrnormArma1(const arma::colvec & mu, const arma::mat & sigma) ;

#endif
