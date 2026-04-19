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


    n1 = 2
    n2 = 3


    beta_new_n1n2 <-  Graph[2,3]
    beta_new_n2n1 <-  Graph[3,2]

    if( Graph[n1,n2] != 0 | Graph[n2,n1] != 0 ){
      cat("OK")
    }

    Graph_new[n1,n2] = 0
    Graph_new[n2,n1] = 0

  map_n1 <- get_mapping(Graph_new, n1)
  map_n2 <- get_mapping(Graph_new, n2)

  p1 = qIsing(YY, Graph_new, alpha = Thresh, log = TRUE) - qIsing(YY, Graph_old, alpha = Thresh, log = TRUE)
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
}
