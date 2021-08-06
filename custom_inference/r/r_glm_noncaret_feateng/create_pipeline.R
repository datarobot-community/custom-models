source("preprocess.R") # not working in DR yet!
source("rmcons.R")
source("rmident.R")

create_pipeline<-function(X, y, model_type='regression') {
  # Clean
  X <- rm_ident(X)
  X <- rm_cons(X)
  X <- preprocess(X)

  # set up dataframe for modeling
  train_df <- X
  train_df$target <- unlist(y)
  if (model_type == 'classification'){
    train_df$target <- as.factor(train_df$target)
  }

  # Run the model using builtin glm function
  model <- glm(target~.,data=train_df)
  return(model)
}
