

## QRFSL algorithm ##

# 0) Loading required packages
library(quantregForest)
library(doParallel)
library(foreach)


# 1) Training
QRFSL <- function(
  # feature matrix
  x,        
  # target vector
  y,        
  # k-fold cv (default is 10)
  k = 10,      
  # prob for quantile
  prob = c(
    0.025, seq(0.05, 0.95, by = 0.05), 0.975
  ),
  # random seed (default is 0)
  seed = 0,
  # parellel option (default is single)
  mc = 1,
  # SL type (0: weights, 1: rf),
  sl = 0
) {
  
  # cv settings 
  cv.folds <- function (n, folds = 10) {
    split(sample(1:n), rep(1:folds, length = n))
  }
  set.seed(seed)
  i_cv <- cv.folds(
    n = length(y),
    folds = k
  )
  
  # cv object
  tdat_x_cv <- lapply(i_cv, function(i) x[-i,,drop=F])
  tdat_y_cv <- lapply(i_cv, function(i) as.matrix(y[-i]))
  
  idat_x_cv <- lapply(i_cv, function(i) x[+i,,drop=F])
  idat_y_cv <- lapply(i_cv, function(i) as.matrix(y[+i]))
  
  # do cv
  if(mc > 1) {
    cl <- makeCluster(mc); registerDoParallel(cl);
    pred_y_cv <- foreach(i = 1:k, .packages='quantregForest') %dopar% {
      set.seed(i)
      qreg_one <- quantregForest(
        x = tdat_x_cv[[i]],
        y = as.vector(tdat_y_cv[[i]])
      )
      predict(
        qreg_one,
        newdata = idat_x_cv[[i]],
        what = prob
      )
    }
    stopCluster(cl)
    
  } else {
    pred_y_cv <- foreach(i = 1:k, .packages='quantregForest') %do% {
      set.seed(i)
      qreg_one <- quantregForest(
        x = tdat_x_cv[[i]],
        y = as.vector(tdat_y_cv[[i]])
      )
      predict(
        qreg_one,
        newdata = idat_x_cv[[i]],
        what = prob
      )
    }
    
  }
  pred_cv <- do.call("rbind", pred_y_cv)
  colnames(pred_cv) <- gsub(
    pattern = "\\s",
    replacement = "",
    x = colnames(pred_cv)
  )
  yact_cv <- do.call("rbind", idat_y_cv)
  
  # training super-learner
  set.seed(seed)
  sl <- suppressWarnings(
    randomForest(
      x = pred_cv,
      y = yact_cv,
      ntree = 1000
    )
  )
  
  # saving objects
  return(
    list(
      sl = sl, 
      x = x, 
      y = y, 
      prob = prob
    )
  )
}


# 2) Prediction 
predict.QRFSL <- function(
  # QRFSL object
  object, 
  # new feature data
  newx, 
  # random seed (default is 0)
  seed = 0
) {
  
  # Quantile model
  set.seed(seed)
  qmod <- quantregForest(
    x = object$x,
    y = object$y
  )
  
  # Predicted quantiles
  qpred <- predict(
    qmod, 
    newdata = newx,
    what = object$prob
  )
  colnames(qpred) <- gsub(
    pattern = "\\s",   
    replacement = "",
    x = colnames(qpred)
  )
  
  # Prediction by SL
  pred <- predict(
    object = object$sl, 
    newdata = qpred, 
    type = "response"
  )
  
  # saving objects
  return(
    list(
      qmod = qmod,
      qpred = qpred,
      pred = pred
    )
  )
}
## ## ## ## ##




## Example dataset ##
data(mtcars)
head(mtcars)


# Setting dataset (training / test)
set.seed(2)
idx <- sample(x = 1:nrow(mtcars), size = 5, replace = F)
tdat_x <- mtcars[-idx,-1]
tdat_y <- mtcars[-idx,+1]
head(tdat_x); head(tdat_y);

idat_x <- mtcars[+idx,-1]
idat_y <- mtcars[+idx,+1]
head(idat_x); head(idat_y);


# Training a 10-fold CV-based QRFSL 
fobj <- QRFSL(
  x = tdat_x,        
  y = tdat_y,        
  k = 10,      
  seed = 1,
  mc = 1
)

# Prediction by using the trained model
pobj <- predict.QRFSL(
  object = fobj, 
  newx = idat_x, 
  seed = 2019
)
pred <- pobj$pred
pred

# Checking predictive performance
yact <- idat_y
cbind.data.frame(
  rmse = prettyNum(
    sqrt(mean((yact - pred)^2)),
    big.mark=","
  ), 
  corr = round(
    x = cor(yact, pred)^2,
    digits = 3
  )
)
## ## ## ## ##

# End..

