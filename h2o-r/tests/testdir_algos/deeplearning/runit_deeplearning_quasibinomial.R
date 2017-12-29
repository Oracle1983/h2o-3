setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")


LogLikelihood<- function(beta, Y, X){
  pi<- plogis( X%*%beta )
  pi[pi==0] <- .Machine$double.neg.eps
  pi[pi==1] <- 1-.Machine$double.neg.eps
  logLike<- sum( Y*log(pi)  + (1-Y)*log(1-pi)  )
  return(-logLike)
}
###########
# grad- corresponding function to calculate the gradient
# other optimization routines (e.g. Nelder-Mead) do not use the gradient
######
grad<- function(beta, Y, X){
  pi<- plogis( X%*%beta )        # P(Y|A,W)= expit(beta0 + beta1*X1+beta2*X2...)
  pi[pi==0] <- .Machine$double.neg.eps        # for consistency with above
  pi[pi==1] <- 1-.Machine$double.neg.eps
  gr<- crossprod(X, Y-pi)        # gradient is -residual*covariates
  return(-gr)
}


test.DL.quasi_binomial <- function() {
  # First calculate paper version
  # From TLMSE paper (Estimating Effects on Rare Outcomes: Knowledge is Power, Laura B. Balzer, Mark J. van der Laan)
  # Example: Data generating experiment for Simulation 1
  set.seed(123)
  n=2500
  W1<- rnorm(n, 0, .25)
  W2<- runif(n, 0, 1)
  W3<- rbinom(n, size=1, 0.5)
  A<- rbinom(n, size=1, prob= plogis(-.5+ W1+W2+W3) )
  pi<- plogis(-3+ 2*A + 1*W1+2*W2-4*W3 + .5*A*W1)/15
  Y<- rbinom(n, size=1, prob= pi)
  sum(Y)
  # 29
  # Qbounds (l,u)= (0,0.065)
  l=0; u=0.065
  #create the design matrix
  X <- model.matrix(as.formula(Y~W1+W2+W3+A*W1))
  # transform Y to Y.tilde in between (l,u)
  Y.tilde<- (Y - l)/(u-l)
  summary(Y.tilde)
  # Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
  # 0.0000  0.0000  0.0000  0.1785  0.0000 15.3800
  # call to the optim function.
  # par: initial parameter estimates; f:function to minimize; gr: gradient
  # arguments to LogLikelihood() & grad() are Y and X
  optim.out <- optim(par=rep(0, ncol(X)), fn=LogLikelihood, gr=grad,
                     Y=Y.tilde, X=X, method="BFGS")
  # see optim help files for more details and other optimization routines
  # get parameter estimates
  beta<- optim.out$par

  # now H2O
  hf = as.h2o(cbind(Y.tilde,X))
  x = 2:7
  y = 1
  m_glm = h2o.glm(training_frame = hf,x=x,y=y,family='quasibinomial',standardize=F,lambda=0)
  beta_h2o_1 = m_glm@model$coefficients

  h2o_glm_pred = plogis(X%*%beta_h2o_1)
  h2o_glm_pred[h2o_glm_pred==0] <- .Machine$double.neg.eps
  h2o_glm_pred[h2o_glm_pred==1] <- 1-.Machine$double.neg.eps

  betas = cbind(beta,beta_h2o_1)
  colnames(betas) <- c("R","H2O-GLM")
  print(betas)

  l0 = LogLikelihood(beta,Y.tilde,X)
  l1 = LogLikelihood(beta_h2o_1,Y.tilde,X)
  expect_equal(mean(h2o_glm_pred), mean(Y.tilde), tolerance=1e-4)
  expect_equal(l0,l1,tolerance=1e-4)

  # fit h2o deeplearning model
  m_deeplearning = h2o.deeplearning(training_frame = hf, x=x, y=y, distribution='quasibinomial', epochs=10,
                                    hidden=c(20,20,20), seed=1234, nfolds=10, stopping_rounds=10,stopping_tolerance=0)

  # compute log-likelihood for h2o deeplearning predictions
  h2o_deeplearning_pred = as.data.frame(h2o.predict(m_deeplearning, as.h2o(X)))
  h2o_deeplearning_pred[h2o_deeplearning_pred==0] <- .Machine$double.neg.eps
  h2o_deeplearning_pred[h2o_deeplearning_pred==1] <- 1-.Machine$double.neg.eps
  l2 <- -sum( Y.tilde*log(h2o_deeplearning_pred)  + (1-Y.tilde)*log(1-h2o_deeplearning_pred)  )

  ls = c(l0,l1,l2)
  names(ls) <- c(colnames(betas), "H2O DL")
  print(ls)

  preds = cbind(h2o_deeplearning_pred, h2o_glm_pred, Y.tilde)
  names(preds) <- c("H2O DL preds", "H2O GLM preds", "Actual Y")
  print(head(preds, 100))
  print(summary(preds))

  expect_equal(mean(h2o_deeplearning_pred[,1]), mean(Y.tilde), tolerance=2.5e-3)  # distribution means match
  expect_true(l0 > l2)  # should fit better than GLM (on training data)
}

doTest("DL Test: quasi binomial", test.DL.quasi_binomial)
