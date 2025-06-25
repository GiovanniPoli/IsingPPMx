#### Likliehood -----
if(FALSE){
  {
  ## TEST 1
  SUM1  = NULL
  SUM2  = NULL
  DIFF1 = NULL
  DIFF2 = NULL
  r        = 5
  p        = 100
  n        = 1000
  S        = 100
  var_diag   = 0
  var_values = 0
  for(test in 1:S){
    y = cbind( matrix( c(rep(c(1,0),n)), ncol = 2, nrow = n, byrow = TRUE),
               matrix(rbinom(n*(p-2), 1,.5), ncol = p-2, nrow = n))

    OmegaNULL = diag(rep(0,100))
    Omega     = diag(rep(0,100))
    diag(Omega)     = 0
    diag(OmegaNULL) = 0

    Omega[1,2] = Omega[2,1] = rnorm(1,mean = -1, sd = sqrt(var_values))

    diag(OmegaNULL) = rnorm(p,mean = 0, sd = sqrt(var_diag))
    diag(Omega)     = rnorm(p,mean = 0, sd = sqrt(var_diag))


    s = sample.int(100, r)
    k = sample.int(100, r)
    OmegaNULL[s,k] = OmegaNULL[k,s] = rnorm(1,mean = 0, sd = sqrt(var_values))
    isSymmetric.matrix(OmegaNULL)


    s1 = 0
    d1 = 0
    for(i in 1:100){


      s1 = s1 + as.numeric(  IsingGraph::node_wise_pseudo_ll(y[i,], OmegaNULL) <
                             IsingGraph::node_wise_pseudo_ll(y[i,], Omega) )
      d1 = d1 + as.numeric(  IsingGraph::node_wise_pseudo_ll(y[i,], OmegaNULL) -
                             IsingGraph::node_wise_pseudo_ll(y[i,], Omega) )

    }
    d2 = s2 = 0
    for(i in 1:100){
      s2 = s2 + as.numeric( IsingGraph::node_wise_generalized_ll(y[i,], OmegaNULL, .5 ) <
                            IsingGraph::node_wise_generalized_ll(y[i,], Omega , .5) )
      d2 = d2 + as.numeric( IsingGraph::node_wise_generalized_ll(y[i,], OmegaNULL, .5 ) -
                            IsingGraph::node_wise_generalized_ll(y[i,], Omega, .5 ))
    }
    DIFF1 = c(DIFF1, d1/100)
    DIFF2 = c(DIFF2, d2/100)
    SUM1  = c(SUM1 , s1/100)
    SUM2  = c(SUM2 , s2/100)

    cat("\r",paste0("[", as.integer(test),"/",as.integer(S),"]             "))

  }

  plot(density(SUM1), lwd = 2)
  lines(density(SUM2)$x, density(SUM2)$y, lwd = 2, col = "red")
  }
}
#### LOSS ----
if(FALSE){

{
  library(scales)
  end = 10
  cols = scales::hue_pal()(27)
  int = 1
  alpha = 0.2
  plot(NULL, ylab = "p", xlab = "y", xlim = c(1,end), ylim = c(0,1), main = beta)
  for(beta in log(1:10)){
      r = function(d) (.5 * alpha*(1+d)^(-beta) + (1- alpha*.5)/2)*alpha
      lines(1:end,r(1:end), pch = 20, ylim = c(0,1), col = cols[int], type = "l")
      abline(h = c(0,alpha,1-alpha), pch = 20, lty = 2)
      int = int + 1
      abline(v = seq(from = 0, to = 100, by = 1), col = "orange", lty = 2)
      points(1,1, pch = 20, col = cols[27])
  }
}






1- round(exp(-(1/1:100)^2),2)



}
### RPG_1 ----
if(FALSE){
  b = function(num = 1, h = 1, z = 0, trunc = 200){
    n = h
    w = rep(0, num)
    c.i = (1:trunc - 1/2)^2 * pi^2 * 4
    a.i = c.i + z^2
    for (i in 1:num) {
      w[i] = 2 * sum(rgamma(trunc, n)/a.i)
    }
    w
  }
  a = function ( num = 1, h = 1, z = 0) {
    n = h
    z = array(z, num)
    n = array(n, num)
    total.trials = 0
    x = rep(0, num)
    for (i in 1:num) {
      x[i] = 0
      for (j in 1:n[i]) {
        temp = c(z[i])
        x[i] = x[i] + temp$x
        total.trials = total.trials + temp$n
      }
    }
    x
  }
  e = function(Z){
    x = 0.64
    fz = pi^2/8 + Z^2/2
    b = sqrt(1/x) * (x * Z - 1)
    a = -1 * sqrt(1/x) * (x * Z + 1)
    x0 = log(fz) + fz * 0.64
    xb = x0 - Z + pnorm(b, log.p = TRUE)
    xa = x0 + Z + pnorm(a, log.p = TRUE)
    qdivp = 4/pi * (exp(xb) + exp(xa))
    1/(1 + qdivp)
  }
  c = function(Z){
    Z = abs(Z) * 0.5
    fz = pi^2/8 + Z^2/2
    num.trials = 0
    total.iter = 0
    while (TRUE) {
      num.trials = num.trials + 1
      if (runif(1) < e(Z)) {
        X = 0.64 + rexp(1)/fz
      }
      else {
        X = d(Z)
      }
      S = f(0, X)
      Y = runif(1) * S
      n = 0
      while (TRUE) {
        n = n + 1
        total.iter = total.iter + 1
        if (n%%2 == 1) {
          S = S - f(n, X)
          if (Y <= S)
            break
        }
        else {
          S = S + f(n, X)
          if (Y > S)
            break
        }
      }
      if (Y <= S)
        break
    }
    list(x = 0.25 * X, n = num.trials, total.iter = total.iter)
  }
  d = function (Z, R = 0.64){
    Z = abs(Z)
    mu = 1/Z
    X = R + 1
    if (mu > R) {
      alpha = 0
      while (runif(1) > alpha) {
        E = rexp(2)
        while (E[1]^2 > 2 * E[2]/R) {
          E = rexp(2)
        }
        X = R/(1 + R * E[1])^2
        alpha = exp(-0.5 * Z^2 * X)
      }
    }
    else {
      while (X > R) {
        lambda = 1
        Y = rnorm(1)^2
        X = mu + 0.5 * mu^2/lambda * Y - 0.5 * mu/lambda * sqrt(4 * mu * lambda * Y + (mu * Y)^2)
        if (runif(1) > mu/(mu + X)) {
          X = mu^2/X
        }
      }
    }
    X
  }
  f = function (n, x){
    if (x > 0.64 )
      pi * (n + 0.5) * exp(-(n + 0.5)^2 * pi^2 * x/2)
    else
      (2/pi/x)^1.5 * pi * (n + 0.5) * exp(-2 * (n + 0.5)^2/x)
  }


  {
    t1 = Sys.time()
    d2  = density(BayesLogit::rpg.gamma(1000000,1,2))
    t2 = Sys.time()
    d = density( c(IsingGraph::rpg(rep(2,1000000))))
    t3 = Sys.time()
    # d3 = density( c(IsingGraph:::alpha cpp_polyagamma_h1_truncated(rep(2,100))))
    t4  = Sys.time()
  }

  t2-t1
  t3-t2
  t4-t3
  plot(d2)
  lines(d$x,d$y,col = "red")
  lines(d3$x,d3$y,col = "green")

  plot(density(   replicate( 100000, BayesLogit:::rpg.devroye.1(100)$x)))
  d = density(    cpp_polyagamma_h1_devroye(rep(100,100000)) )



  lines(d$x,d$y, col = "red")

  BayesLogit:::rpg.devroye.R


  X = matrix(1:12, nrow = 6, ncol = 2)
  X*rep(c(0,2),each = 3)


  y =  c( rbinom(50,1, .75),
          rbinom(50,1, .25) )
  X = matrix ( c( rep(c(1,0), 50),
                  rep(c(1,1), 50)), ncol = 2, nrow = 100, byrow = TRUE)

  b.0 = c(0,0)
  B.0 = diag(rep(10,2))
  b.start =  c(-5,-5)

  sample = 100+12*1
  Bayes_logit_GIBBS( y, X, b.0, B.0, b.start,100000)

}
if(FALSE){

  {

  t0 = Sys.time()
  replicas = list()


  for(BB in c( seq(from = 1, to = 2, by = .25), 10 ) ){

    n = 15
    a =  c( rbinom(n,1, .5))
    X = cbind(a,1-a, rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5))

    P = 15
    p = 0
    gamma = diag(rep(0,P))
    gamma[1,2] = gamma[2,1] = 0

    omega = matrix(rnorm(P^2,sd = .1), ncol = P, nrow = P) * gamma

    GRAPHS = list()

    for(replica in 1:50){

      gamma_sum = omega_sum = matrix(0, ncol = 15, nrow = 15)

      for( i in 1:11000){
        for(p in 0:14){
          Xtilde = X
          ytilde = X[,p+1]
          Xtilde[,p+1] = 1

          pi_slab = 1.0 - ppois(  sum(gamma[,1+p]) , 1 )

          IsingGraph::w_cpp_update_Omega( gamma, omega, p, ytilde, Xtilde, 1, 1)
          IsingGraph::w_variable_selection_step_v2( gamma, omega, p, ytilde, Xtilde, 10, 10, pi_slab,
                                                         BB )

        }

    if(i >= 1000){
      gamma_sum = gamma_sum + gamma
      omega_sum = omega_sum + omega
      }
    }
      GRAPHS[[as.character(replica)]] =   gamma_sum / 10000

    }
    replicas[[as.character(BB)]] =  GRAPHS

  }
  t1 = Sys.time()

  }

}




if(FALSE){

  {
    n = 1e2
    gamma_sum = omega_sum = matrix(0, ncol = 15, nrow = 15)
    a =  c( rbinom(n,1, .5))
    X = cbind( a, 1-a, rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5) )

    a = b= c()
    P = 15
    p = 0
    gamma = diag(rep(0,P))
    gamma[1,2] = gamma[2,1] = 1

    omega = matrix(rnorm(P^2,sd = .1), ncol = P, nrow = P) * gamma

    remove(a, b, n, p, P)


    i=0
    {
      i = i + 1
    {
      p = sample.int(15,1)-1
      cat("col",p,"\n")

      Xtilde = X
      ytilde = X[,p+1]
      Xtilde[,p+1] = 1

      selected = gamma[ sample.int(15,1)-1 , p]
      others   = which( drop(gamma[, p]) != selected )
      new_beta = rnorm(1)
      if( length(others) != 0 ){

        old_node = others[sample.int(length(others),1)]

        if( selected == 1 ){

          IsingGraph::log_likelihood_ratio_swap(
            ytilde,
            Xtilde,
            omega[,p],
            new_beta,
            old_node-1,
            selected)
        }else{

          IsingGraph::log_likelihood_ratio_swap(
            ytilde,
            Xtilde,
            omega[,p],
            new_beta,
            old_node -1,
            selected)

          IsingGraph::log_likelihood_ratio_swap_v2(
            ytilde,
            Xtilde,
            omega[,p],
            new_beta,
            old_node -1,
            selected)




        Limit = 0.001
        Prs = NULL
        for(i in 1:p){
          Prs = c(Pr + .1)
        }


        }
        cat("\n")

      }

      }
      gamma_sum = gamma_sum + gamma
      omega_sum = omega_sum + omega
    }

    gamma_sum / 1000
    omega_sum / 1000

  }
}


if(FALSE){

  P = 3

  gamma = diag(rep(0,P))
  gamma[1,1] = 1
  omega = matrix(rnorm(P^2), ncol = P, nrow = P)*gamma
  omega[1,1] = 1.0
  SUM   = diag(c(0,0,0))



  y =  c( rbinom(2,1, .5))
  X =  cbind(1,1-y, rbinom(2,1,.5))



  gamma[2,1] = 1
  gamma[1,2] = 1
  gamma[1,1] = 1

  new_omega = omega[,1]
  new_omega[2] = -2
  new_omega[1] = 0
  omega

  sum( y* log ( exp( X %*% new_omega) / (exp(X %*% new_omega) + 1 ) ) -
       y* log ( exp( X %*% omega[,1]) / (exp(X %*% omega[,1]) + 1 ) ))


  a = IsingGraph::w_variable_selection_step( gamma, omega, 0, y, X, 1,.1, 1)
  b = IsingGraph::w_cpp_update_Omega( gamma, omega, 0, y, X, 1,.1)

  b = omega[,1]

  log_likelihood_ratio_swap(y, X, b,
                            10, # beta
                            1,   # remove_node
                            0)   # add_node

  log_likelihood_ratio_swap(y, X, b,
                            b, # beta
                            1,   # remove_node
                            0)   # add_node



  cbind(y,X)
  X = X[1:50,]
  y = y[1:50]
  b.0 = c(0,0)
  B.0 = diag(rep(10,2))


}

