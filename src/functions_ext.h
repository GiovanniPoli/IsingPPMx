#ifndef FUNCTIONS_EXT_H
#define FUNCTIONS_EXT_H

#include <RcppArmadillo.h>

double log_beta( double x1, double x2);
double       log_gX( const int n, const arma::rowvec & x,  const arma::rowvec & Sx);
double empty_log_gX(              const arma::rowvec & x) ;
double       log_cohesion( const double nh, const double sigma);
double empty_log_cohesion( const double M,  const double H, const double sigma);
double log_gammas_mid_1( int g1, int g2, double c) ;
#endif
