
#' Removing identical features
#'
#' @param train required. data.frame or similar classes
#' @param test optional. data.frame or similar classes
#'
#' @return None
#' @export
#' 
#' 



rm_ident <- function(train, test=NULL) {
  features_pair <- combn(names(train), 2, simplify = F)
  toRemove <- c()
  for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
      if (all(train[[f1]] == train[[f2]])) {
        cat(f1, "and", f2, "are equals.\n")
        toRemove <- c(toRemove, f2)
      }
    }
  }
  train <- train[,!colnames(train) %in% toRemove]
  if(!missing(test)){
    test  <- test[,!colnames(test) %in% toRemove]
  }
}

