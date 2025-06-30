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

#endif
