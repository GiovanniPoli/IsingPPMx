if(FALSE){

t0 = Sys.time()
RETURN = list()

for(probs in c(1, 0.75, .5)){
  replicas = list()

for(BB in c( seq(from = 1, to = 2, by = .5), 10 ) ){
  GRAPHS = list()

  for(replica in 1:20){

    n = 15
    a =  c( rbinom(n,1, .5))
    b =  rbinom(length(a),1, ifelse(a == 1, 1-probs, probs))


    X = cbind(a,1-a,
              rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5),
              rbinom(n,1,.5),rbinom(n,1,.5),rbinom(n,1,.5))

    P = 15
    p = 0
    gamma = diag(rep(0,P))
    gamma[1,2] = gamma[2,1] = 0

    omega = matrix(rnorm(P^2,sd = .1), ncol = P, nrow = P) * gamma


    gamma_sum = omega_sum = matrix(0, ncol = 15, nrow = 15)

    for( i in 1:11000){
      for(p in 0:14){
        Xtilde = X
        ytilde = X[,p+1]
        Xtilde[,p+1] = 1

        pi_slab = 1.0 - ppois(  sum(gamma[,1+p]) , 1 )

        IsingGraph::w_cpp_update_Omega( gamma, omega, p, ytilde, Xtilde, 1, 1)
        IsingGraph::w_variable_selection_step_v2( gamma, omega, p, ytilde, Xtilde, 10, 10, pi_slab, BB )

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
  RETURN[[as.character(probs)]] = replicas
}
}
