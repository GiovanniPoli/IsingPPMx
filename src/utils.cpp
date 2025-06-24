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
