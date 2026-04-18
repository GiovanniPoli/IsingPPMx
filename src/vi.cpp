#include <RcppArmadillo.h>
#include "vi.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]]

// [[Rcpp::export]]
double entropy(const arma::uvec & cl) {
  int n = cl.n_elem;
  arma::colvec counts(n, arma::fill::zeros);
  for(int i = 0; i < n ; i++){
    counts(cl(i)) += 1.0;
  }
  arma::colvec non_zero_ratios = arma::nonzeros(counts)/n;
  return - arma::accu( non_zero_ratios%log2(non_zero_ratios));
}

// [[Rcpp::export]]
double VI( const arma::uvec & c0,
           const arma::uvec & c1) {

  int n = c0.n_elem;

  arma::mat    joint_counts(n, n, arma::fill::zeros);
  arma::colvec counts0(n, arma::fill::zeros);
  arma::colvec counts1(n, arma::fill::zeros);
  arma::uword k0,k1;

  for (int i = 0; i < n; i++) {
    k0 = c0(i);
    k1 = c1(i);

    joint_counts(k0, k1) += 1.0;
    counts0(k0)          += 1.0;
    counts1(k1)          += 1.0;
  }

  arma::colvec non_zero_joint   = arma::nonzeros(joint_counts) / n;
  arma::colvec non_zero_ratios0 = arma::nonzeros(counts0) / n;
  arma::colvec non_zero_ratios1 = arma::nonzeros(counts1) / n;

  double mH0 = arma::accu(non_zero_ratios0 % arma::log2(non_zero_ratios0));
  double mH1 = arma::accu(non_zero_ratios1 % arma::log2(non_zero_ratios1));
  double joint = -2.0 * arma::accu(non_zero_joint % arma::log2(non_zero_joint));

  return joint + mH0 + mH1;
}
/*
// [[Rcpp::export]]
arma::vec marginal_vi(const arma::uvec & cluster,
                      const arma::uvec & c0,
                      arma::uword H,
                      arma::uword obs){
  arma::uvec clust_minus = cluster;
  arma::uvec C_0_minus = c0;

  arma::uvec unique_v = arma::unique(c0);
  arma::uword H_C0    = unique_v.n_elem;

  double n = c0.n_elem;
  double entropy = 0;
  double H_cc    = 0;

  arma::uvec dist(H, arma::fill::zeros);

  arma::mat     counts_int(H,H_C0, arma::fill::zeros);
  arma::colvec  counts_set(H, arma::fill::zeros);

  for(int i = 0; i < n; i++)
  {
    if(i == obs)
    {
      // Skip subject
      continue;
    }
    else
    {
      // Compute block sizes for the current clustering minus the observation obs
      counts_set(clust_minus[i]) += 1;

      // Compute sizes of intersection withouth the observation obs
      counts_int(clust_minus[i], C_0_minus[i]) += 1;

    }
  }

  counts_set(find(counts_set > 0))= counts_set(find(counts_set > 0))/n;
  counts_int(find(counts_int > 0)) = counts_int(find(counts_int > 0))/n ;

  //--------------------------------------//

  for(int h = 0; h < H; h++){

    // Compute entropy of clustering
    H_c = -((counts_set(h) + 1/n)*log2(counts_set(h) + 1/n)) -
      sum(nonzeros(counts_set)%log2(nonzeros(counts_set))) +
      ((counts_set(h) > 0) ? (counts_set(h)*log2(counts_set(h))) : (0));


    // Compute mutual information
    H_cc = -(sum(nonzeros(counts_int)%log2(nonzeros(counts_int)))) +
      ((counts_int(h, C_0_minus(obs)) > 0) ? (counts_int(h, C_0_minus(obs))*log2(counts_int(h, C_0_minus(obs)))) : (0)) -
      ((counts_int(h, C_0_minus(obs)) +1/n)*log2(counts_int(h, C_0_minus(obs)) + 1/n));

    // VI distance (proportional up to H_0)
    dist(h) = 2*H_cc - H_c ;
  }

  return(dist);
 */
