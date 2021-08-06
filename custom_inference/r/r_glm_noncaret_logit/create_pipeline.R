create_pipeline<-function(X, y, model_type='regression') {

  # set up data.frame for modeling
  train_df <- X
  train_df$target <- unlist(y)
  if (model_type == 'classification'){
    train_df$target <- as.factor(train_df$target)
  }


  # Run a logistic regression using builtin glm 
  model <- glm(target~., data=train_df, family = "binomial")
  return(model)
}
