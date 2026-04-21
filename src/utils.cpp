#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "utils.h"

using namespace Rcpp;

using myClock = std::chrono::steady_clock;
using MyTimePoint = std::chrono::time_point<myClock>;

int get_console_width() {
  Function getOption("getOption");
  int width = as<int>(getOption("width"));
  return width;
}
void catProgressBar(int progress, int total, MyTimePoint start_time) {
  int barWidth = get_console_width()-50;
  float ratio = (float)progress / total;
  int pos = barWidth * ratio;

  auto now = myClock::now();
  double elapsed = std::chrono::duration<double>(now - start_time).count()/progress * (total-progress);

  std::string unit ;
  if(elapsed < 60.0){
    unit = "s" ;
  }else if( (elapsed >= 60.0) & (elapsed < 3600.0) ){
    unit = "m" ;
    elapsed = elapsed/60.0;
  }else{
    unit = "h" ;
    elapsed = elapsed/3600.0;

  }

  std::cout << "[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << int(ratio * 100.0) << " % - ETA: ";
  std::cout << std::fixed << std::setprecision(2) << elapsed << unit <<" \r";
  std::cout.flush();
}
void catIter(int progress, int total, MyTimePoint start_time) {

  auto       now = myClock::now();
  double elapsed = std::chrono::duration<double>(now - start_time).count()/progress * (total-progress);

  std::string unit ;
  if(elapsed < 60.0){
    unit = "s" ;
  }else if( (elapsed >= 60.0) & (elapsed < 3600.0) ){
    unit = "m" ;
    elapsed = elapsed/60.0;
  }else{
    unit = "h" ;
    elapsed = elapsed/3600.0;

  }

  std::cout << "[";
  std::cout << progress;
  std::cout << "/";
  std::cout << total;
  std::cout << "] - ";
  std::cout << std::fixed << std::setprecision(2)<< "ETA: " << elapsed  << unit <<"                 \r";
  std::cout.flush();
}

// [[Rcpp::export]]
arma::mat coocorence( const arma::mat& MCMC_clusters ){
  const int N = MCMC_clusters.n_cols ;
  const int S = MCMC_clusters.n_rows ;
  arma::mat ret(N,N, arma::fill::eye) ;

  for( int i = 0; i<N; ++i){
    for( int j = i+1; j<N; ++j){
      for( int s = 0; s<S; ++s){
        if( MCMC_clusters(s,i) == MCMC_clusters(s,j) ){
          ret(i,j) += 1.0 / S ;
          ret(j,i) += 1.0 / S ;
        }
      }
    }
  }
  return ret ;
}

arma::uword pair_to_index(arma::uword i, arma::uword j) {
  arma::uword min = std::min(i,j) ;
  arma::uword max = std::max(i,j) ;
  return max *(max - 1) / 2 + min;
}

std::pair<arma::uword,arma::uword> index_to_pair(int k) {
  arma::uword i = static_cast<unsigned>( std::floor((1 + std::sqrt(1 + 8.0 * k)) / 2.0) );
  arma::uword j = k - i * (i - 1) / 2;
  return std::make_pair(i,j);
}

void remove_j(arma::uvec & v, int j) {
  int idx ;
  for (int i = v.n_elem - 1; i >= 0; --i) {
    if (v[i] == j) {
      idx = i;
      break;
    }
  }
  v(idx) = v(v.n_elem - 1);
  v.resize(v.n_elem - 1);
}

void push_back_j(arma::uvec& v, int j) {
  v.resize(v.n_elem + 1);
  v(v.n_elem - 1) = j;
}


std::vector<std::pair<int,arma::uvec>> map_for_variable_selection(const arma::uvec & x) {
  arma::uvec unique_values = arma::unique(x);
  std::vector<std::pair<int,arma::uvec>> ret(unique_values.n_elem);
  for (int i = 0; i < unique_values.n_elem; ++i) {
    ret[i] = std::make_pair(unique_values(i), unique_values(arma::find(unique_values != unique_values(i)  )));
  }

  return ret;
}

std::vector<std::tuple<int, arma::uvec, arma::colvec, arma::uvec>> map_pairs_into_regs(
                                                                           const arma::uvec& pair1,
                                                                           const arma::uvec& pair2,
                                                                           const arma::colvec& changed_betas,
                                                                           const std::vector<arma::uvec> & idx_regs) {
  arma::uvec all    = arma::join_cols(pair1, pair2);
  arma::uvec common = arma::intersect(pair1, pair2);

  std::vector<std::tuple<int, arma::uvec, arma::colvec, arma::uvec>> ret ;
  arma::uvec others ;
  arma::colvec beta ;
  arma::uword common_int ;
  arma::uvec othr_comm;
  arma::uvec beta_othr_idx(2);
  bool first = true ;
  for (arma::uword i = 0; i < all.n_elem; ++i) {
    int val = all(i);
    if(arma::any(common == val)) {
      if(first){
        common_int = val ;
        othr_comm = all(arma::find(all != val));
        beta_othr_idx(0) = i;
        first = false ;
      }else{
        beta_othr_idx(1) = i;
      }
    }else{
    if(arma::any(pair1 == val)) {
        others = pair1(arma::find(pair1 != val));
        beta   = {changed_betas(i)} ;
        ret.push_back(std::make_tuple(val, others, beta, idx_regs[i]));
      }else{
        others = pair2(arma::find(pair2 != val));
        beta   = {changed_betas(i)};
        ret.push_back(std::make_tuple(val, others, beta, idx_regs[i]));
      }
    }
  }
  if( common.n_elem != 0){
    beta = changed_betas.rows(beta_othr_idx) ;
    ret.push_back( std::make_tuple(common_int, othr_comm, beta, idx_regs[beta_othr_idx(0)] ));
  }
  return ret;
}

Rcpp::List wrap_unordered_map_as_list(const std::unordered_map<int, arma::uvec> &m) {
  Rcpp::List out;
  for (const auto &kv : m) {
    out[std::to_string(kv.first)] = Rcpp::wrap(kv.second) ;
  }
  return out;
}

Rcpp::List wrap_nested_unordered_map_of_uvec(
    const std::unordered_map<int, std::unordered_map<int, arma::uvec>> &nested_map){
  Rcpp::List outer_list(nested_map.size());
  Rcpp::CharacterVector outer_names(nested_map.size());
  int outer_idx = 0;
  for (const auto &outer_pair : nested_map) {
    const auto &inner_map = outer_pair.second;
    Rcpp::List inner_list(inner_map.size());
    Rcpp::CharacterVector inner_names(inner_map.size());
    int inner_idx = 0;
    for (const auto &inner_pair : inner_map) {
      inner_list[inner_idx] = Rcpp::wrap(inner_pair.second);
      inner_names[inner_idx] = std::to_string(inner_pair.first);
      ++inner_idx;
    }
    inner_list.attr("names") = inner_names;
    outer_list[outer_idx] = inner_list;
    outer_names[outer_idx] = std::to_string(outer_pair.first);
    ++outer_idx;
  }
  outer_list.attr("names") = outer_names;
  return outer_list;
}


Rcpp::List vector_to_list(const std::vector<arma::uvec>& vec) {
  Rcpp::List out(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    out[i] = vec[i];
  }
  return out;
}

Rcpp::List vector_of_vectors_to_list(const std::vector<std::vector<arma::uvec>>& vecvec) {
  Rcpp::List outer_list(vecvec.size());
  for (size_t i = 0; i < vecvec.size(); ++i) {
    const auto& inner_vec = vecvec[i];
    Rcpp::List inner_list(inner_vec.size());
    for (size_t j = 0; j < inner_vec.size(); ++j) {
      inner_list[j] = inner_vec[j];
    }
    outer_list[i] = inner_list;
  }
  return outer_list;
}

void print_vector(const std::vector<arma::uvec>& vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    Rcpp::Rcout << "Element " << i << ": " << vec[i].t();
  }
}

void print_vector_of_vectors(const std::vector<std::vector<arma::uvec>>& vecvec) {
  for (size_t i = 0; i < vecvec.size(); ++i) {
    Rcpp::Rcout << "Outer element " << i << ":\n";
    const auto& inner_vec = vecvec[i];
    for (size_t j = 0; j < inner_vec.size(); ++j) {
      Rcpp::Rcout << "  Inner element " << j << ": " << inner_vec[j].t();
    }
  }
}
