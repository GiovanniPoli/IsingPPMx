#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>
#include <iostream>
#include <chrono>
#include <thread>


using myClock = std::chrono::steady_clock;
using MyTimePoint = std::chrono::time_point<myClock>;

int  get_console_width() ;
void catProgressBar(int progress, int total, MyTimePoint start_time)  ;
void catIter(int progress, int total, MyTimePoint start_time) ;

void remove_j(arma::uvec & v, int j) ;
void push_back_j(arma::uvec & v, int j) ;


arma::uword pair_to_index(arma::uword i, arma::uword j);
std::pair<arma::uword,arma::uword> index_to_pair(int k);



Rcpp::List wrap_unordered_map_as_list(const std::unordered_map<int, arma::uvec> &m);
std::vector<std::tuple<int, arma::uvec, arma::colvec, arma::uvec>> map_pairs_into_regs(const arma::uvec& pair1, const arma::uvec& pair2, const arma::colvec& changed_betas, const std::vector<arma::uvec> & idx_regs);

Rcpp::List wrap_nested_unordered_map_of_uvec(const std::unordered_map<int, std::unordered_map<int, arma::uvec>> &nested_map);

Rcpp::List vector_to_list(const std::vector<arma::uvec>& vec);
Rcpp::List vector_of_vectors_to_list(const std::vector<std::vector<arma::uvec>>& vecvec);
void print_vector(const std::vector<arma::uvec>& vec);
void print_vector_of_vectors(const std::vector<std::vector<arma::uvec>>& vecvec);
#endif
