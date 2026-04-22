if(FALSE){
{
  # Ising parameters:
  N = 10
  Graph <- matrix(sample(0:1,N^2,TRUE,prob = c(0.7, 0.3)),N,N) * rnorm(N^2, sd = .1)
  Graph <- pmax(Graph,t(Graph)) / N
  diag(Graph) <- 0
  Thresh <- -(rnorm(N)^2)
  Beta <- 1

  YY = matrix( rbinom(N*100, 1, .5), nrow = 100, ncol = 10)

  get_mapping <- function(Beta, r) {
    # active neighbours: non-zero entries in column r (excluding diagonal)
    idx <- which(Beta[, r] != 0)
    idx <- idx[idx != r]             # defensive: exclude self
    as.integer(c(r,idx) - 1L)             # 0-based for C++
  }

  Graph_old = Graph
  Graph_new = Graph

  beta_new_n1n2 <-  0.1
  beta_new_n2n1 <-  0.1

  n1 = 2
  n2 = 3

  if( Graph[n1,n2] != 0 | Graph[n2,n1] != 0 ){
    cat("ERRORE")
  }

  Graph_new[n1,n2] = beta_new_n1n2
  Graph_new[n2,n1] = beta_new_n2n1

  map_n1 <- get_mapping(Graph_new, n1)
  map_n2 <- get_mapping(Graph_new, n2)

  p1 = qIsing(YY, Graph_new, alpha = Thresh, log = TRUE) - qIsing(YY, Graph_old, alpha = Thresh, log = TRUE)
  p2 = IsingSampler::IsingPL(YY,Graph_new, Thresh, Beta, c(0,1)) -
       IsingSampler::IsingPL(YY,Graph_old, Thresh, Beta, c(0,1))
  p3 = cpp_ll_ratio_global_flip(
      YY            = YY,
      Beta          = Graph_old,
      alpha         = Thresh,
      n1            = n1 - 1L,             # 0-based for C++
      mapping_n1    = map_n1,
      n2            = n2 - 1L,
      mapping_n2    = map_n2,
      beta_new_n1n2 = beta_new_n1n2,
      beta_new_n2n1 = beta_new_n2n1
  )
  cat(p1,p2,p3)
}

{
    # Ising parameters:
    N = 10
    Graph <- matrix(sample(0:1,N^2,TRUE,prob = c(0.7, 0.3)),N,N) * rnorm(N^2, sd = .1)
    Graph <- pmax(Graph,t(Graph)) / N
    diag(Graph) <- 0
    Thresh <- -(rnorm(N)^2)
    Beta <- 1

    YY = matrix( rbinom(N*100, 1, .5), nrow = 100, ncol = 10)

    get_mapping <- function(Beta, r) {
      # active neighbours: non-zero entries in column r (excluding diagonal)
      idx <- which(Beta[, r] != 0)
      idx <- idx[idx != r]             # defensive: exclude self
      as.integer(c(r,idx) - 1L)             # 0-based for C++
    }

    Graph_old = Graph
    Graph_new = Graph


    n1 = 0
    n2 = 1


    beta_new_n1n2 <-  Graph[n1+1,n2+1]
    beta_new_n2n1 <-  Graph[n2+1,n1+1]

    if( Graph[n1+1,n2+1] != 0 | Graph[n2+1,n1+1] != 0 ){
      cat("OK")
    }

    Graph_new[n1+1,n2+1] = 0
    Graph_new[n2+1,n1+1] = 0

  map_n1 <- get_mapping(Graph_new, n1)
  map_n2 <- get_mapping(Graph_new, n2)

  p1 = qIsing(YY, Graph_new, alpha = Thresh, log = TRUE) -
       qIsing(YY, Graph_old, alpha = Thresh, log = TRUE)
  p2 = IsingSampler::IsingPL(YY,Graph_new, Thresh, Beta, c(0,1)) -
       IsingSampler::IsingPL(YY,Graph_old, Thresh, Beta, c(0,1))
  p3 = - cpp_ll_ratio_global_flip(
    YY            = YY,
    Beta          = Graph_old,
    alpha         = Thresh,
    n1            = n1 - 1L,             # 0-based for C++
    mapping_n1    = map_n1,
    n2            = n2 - 1L,
    mapping_n2    = map_n2,
    beta_new_n1n2 = beta_new_n1n2,
    beta_new_n2n1 = beta_new_n2n1
  )
  cat(p1,p2,p3)
}
{
  # Ising parameters:
  N = 10
  L = 10 * 9 / 2
  Graph <- matrix(sample(0:1,N^2,TRUE,prob = c(0.7, 0.3)),N,N) * rnorm(N^2, sd = .5)
  Graph <- pmax(Graph,t(Graph)) / N
  diag(Graph) <- 0

  map = apply(Graph,2, function(x) which(x!=0), simplify = FALSE)
  map_c = lapply(1:N, function(x) c(x,map[[x]])-1 )
  n_test = 0

  ones  = c()
  zeros = c()
  TESTm = matrix(0, nrow = N, ncol = N)
  for(l in 0:(L-1)){
    pair = index_to_pair_R(l)
    if(Graph[pair[1,],pair[2,]]==0){
      zeros = c(zeros, pair_to_index_R(pair[1,],pair[2,]))
    }else{
      ones  = c(ones,pair_to_index_R(pair[1,],pair[2,]))
      TESTm[pair[1,],pair[2,]]=1

    }
  }
  round(Graph,4)



  Thresh <- rep(0,10)
  Beta <- 1

  logQx = dpois(0:L,2, log =TRUE)

  YY = cbind(1,0,matrix(rbinom((N-2)*100, 1, .5), nrow = 100, ncol = 8))

  RET = cpp_update_global_SRS_debug(YY,Graph, Thresh, ones, zeros, map_c,logQx,1,0.75)
  NE = length(ones)
  if( RET$move == "FLIP_ADD"){
    NEnew = NE + 1
  } else {
    NEnew = NE - 1
  }




  ll =  qIsing(YY,Omega = RET$BETA_new, alpha = Thresh) -
        qIsing(YY,Omega = RET$BETA_old, alpha = Thresh)
RET$log_alpha
ll
}
}
