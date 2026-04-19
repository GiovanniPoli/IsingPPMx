#' Quasi-Ising Pseudo-likelihood
#'
#' Computes the log pseudo-likelihood of a binary matrix under the quasi-Ising
#' graphical model.
#'
#' @param Y Binary data matrix of dimension \eqn{n \times p}, entries in
#'   \{0, 1\}.
#' @param ... Model parameters. Two forms are accepted:
#'   \describe{
#'     \item{Omega parametrisation}{Supply \code{Omega}, a
#'       \eqn{P \times P} matrix with intercepts on the diagonal and
#'       interactions off-diagonal.}
#'     \item{alpha + Beta parametrisation}{Supply \code{alpha} (length \eqn{P})
#'       and \code{Beta} (\eqn{P \times P} matrix, zero diagonal).}
#'   }
#'
#' @return Scalar pseudo-likelihood.
#'
#' @details
#' The two parametrisations are equivalent through
#' \eqn{\alpha_r = \Omega_{rr}} and \eqn{\beta_{r,c} = \Omega_{r,c}} for
#' \eqn{r \ne c}.
#'
#' @examples
#' \dontrun{
#'   pseudo_ll(Y, Omega = Om)
#'   pseudo_ll(Y, alpha = a, Beta = B)
#' }
#'
#' @export
qIsing = function(Y, Omega, alpha = NULL, log = TRUE) {

  if (!is.matrix(Y) || !all(Y %in% c(0, 1))) stop("Y must be a binary matrix with entries in {0, 1}.")
  P = ncol(Y)

  if (!is.matrix(Omega) || any(dim(Omega) != c(P, P))) stop("'Omega' must be a (P x P) matrix.")
  if (is.null(alpha)) {
    ret = cpp_pseudo_ll_Omega(Y, Omega)
  } else {
    if (length(alpha) != P) stop("'alpha' must have length P = ncol(Y).")
    if (any(diag(Omega) != 0)) {
      diag(Omega) = 0
      warning("'Omega' had non-zero diagonal entries; they were set to 0.")
    }
    ret = cpp_pseudo_ll_alpha_beta(Y, alpha, Omega)
  }

  if (log) ret else exp(ret)
}





if(FALSE){

  library(IsingPPMx)
  library(IsingSampler)
  library(pheatmap)
  library(ggplot2)

  fhetmap = function(x, ...){
    pheatmap( x,   scale = "none",
              cluster_rows = FALSE,
              cluster_cols = FALSE,
              ... )
  }

  # Input:
    p       = 25  # Number of nodes
    M       = p * (p-1) / 2

    nSample = 100 # Number of samples

    Graph  = matrix( sample(0:1,p^2, TRUE, prob = c(0.99, 0.01)), p, p) * rnorm(p^2) *5
    Graph1 = (Graph + t(Graph))
    Graph  = matrix(sample(0:1,p^2,TRUE,prob = c(0.99, 0.01)),p,p) * rnorm(p^2) * 5
    Graph2 = (Graph + t(Graph))

    diag(Graph1) = diag(Graph2) = 0
    Thresholds1  = - rowSums(Graph1) / 2
    Thresholds2  = - rowSums(Graph2) / 2

    Beta         = 1
    Resp         = c(0L,1L)

    Data1 = IsingSampler(nSample, Graph1, Thresholds1)
    Data2 = IsingSampler(nSample, Graph2, Thresholds2)

    data = rbind(Data1, Data2)

    Cov  = rbind( matrix(1, ncol = 4, nrow = nSample),
                  matrix(0, ncol = 4, nrow = nSample))

    {
      # Freq Est
      t0 = Sys.time()

      # Pseudolikelihood:
      resPL         =  EstimateIsing( rbind(Data1, Data2), method = "pl")
      resPL_oracle1 =  EstimateIsing( Data1, method = "pl")
      resPL_oracle2 =  EstimateIsing( Data2, method = "pl")

      # Ising Lasso
      lasso_resPL         =  IsingFit( data )
      lasso_resPL_oracle1 =  IsingFit( Data1 )
      lasso_resPL_oracle2 =  IsingFit( Data2 )
    }
    fhetmap( lasso_resPL$weiadj )

    fhetmap( Graph1, main = "Grafo Vero (1)" )
    fhetmap( resPL_oracle1$graph, main = "Grafo Oracolo (1)" )
    fhetmap( lasso_resPL_oracle1$weiadj, main = "Lasso Oracolo (1)")

    fhetmap( Graph2, main = "Grafo vero (2)")
    fhetmap( resPL_oracle2$graph, main = "Grafo Oracolo (2)")
    fhetmap( lasso_resPL_oracle2$weiadj, main = "Lasso Oracolo (2)")



    c1 = cor( Graph1[upper.tri(Graph1)],
              resPL$graph[upper.tri(resPL$graph)])
    c2 = cor( Graph2[upper.tri(Graph2)],
              resPL$graph[upper.tri(resPL$graph)])

    c1_gs = cor( Graph1[upper.tri(Graph1)],
                 resPL_oracle1$graph[upper.tri(resPL_oracle1$graph)])

    c2_gs = cor( Graph2[upper.tri(Graph2)],
                 resPL_oracle2$graph[upper.tri(resPL_oracle2$graph)])





    Qx_test   = dpois(0:M, M*0.01*2)
    Ret = IsingPPMx::qIsing_PPMx_v2( Y = data,
                                     Z = Cov,
                                     var_coef = 1,
                                     var_int  = 1,
                                     Qx       = Qx_test,
                                     c        = 1000000,
                                     M        = .5,
                                     sigma    = 0,
                                     sample   = 1000,
                                     burn     = 1000,
                                     thinning = 10,
                                     C        = 10 )
    corr_sim_1 = NULL
    for( i in 1:(nSample)){

      A = lapply(Ret, function(x)  x$Beta[,, x$rho[i,]+1 ] )
      B = lapply(Ret, function(x) x$Gamma[,, x$rho[i,]+1 ] )
      d = length(A)

      M_i           = Reduce("+", A) / d
      M_i_structure = Reduce("+", B) / d

      corr_sim_1 = c( corr_sim_1,
                    cor(Graph1[upper.tri(Graph1)],
                          M_i[upper.tri(M_i)]) )
    }

    corr_sim_2 = NULL
    for( i in (nSample+1):(2*nSample)){
      A = lapply( Ret, function(x)  x$Beta[,, x$rho[i,]+1 ] )
      B = lapply( Ret, function(x) x$Gamma[,, x$rho[i,]+1 ] )
      d = length(A)

      M_i_v2           = Reduce("+", A)/d
      M_i_v2_structure = Reduce("+", B)/d

      corr_sim_2 = c( corr_sim_2, cor( Graph2[upper.tri(Graph2)],
                                       M_i_v2[upper.tri(M_i_v2)])
                      )

    }

    pheatmap(coocorence(t(sapply(Ret,  function(x) x$rho))), scale = "none", family = "serif")

    fhetmap( M_i, main = "Graph a posteriori")
    fhetmap( Graph1,
             main = "Grafo Vero (1)" )

    A = IsingFit(Data2)
    fhetmap(A$weiadj)

    fhetmap( resPL_oracle1$graph,
             main = "Grafo Oracolo (1)" )



    fhetmap( Graph1,
             main = "Grafo vero (2)")
    fhetmap( M_i_structure,
             main = "AA")
    fhetmap( lasso_resPL_oracle1$asymm.weights,
             main = "Grafo Oracolo (2)")




    fhetmap( Graph2,
             main = "Grafo vero (2)")
    fhetmap( M_i_v2_structure,
             main = "AA")
    fhetmap( lasso_resPL_oracle2$asymm.weights,
             main = "Grafo Oracolo (2)")


    A = sapply(Ret, function(x)  x$Gamma[4,15, x$rho[i,]+1 ] )
    plot(A)
    mean(A)
    t1 = Sys.time()
    t1-t0
    saveRDS(list(
      "G1" = Graph1 ,
      "G2" = Graph2 ,
      "resPL" = resPL,
      "resPL_oracle1" = resPL_oracle1,
      "resPL_oracle2" = resPL_oracle2,
      "PPMX"          = Ret),
      paste0("sim_",s,".rds"))
  }



