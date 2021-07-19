# This is the simplest POC that Composable ML / DRUM can
#   work with non-caret R models.
# Here I just take the caret example and replace the 
#   caret model with a very naive GLM() that uses the default
#   settings for family, link, prediction scale, etc.
#         - Jason

create_pipeline<-function(X, y, model_type='regression') {

  # set up dataframe for modeling
  train_df <- X
  train_df$target <- unlist(y)
  if (model_type == 'classification'){
    train_df$target <- as.factor(train_df$target)
  }


  # Run the model using builtin glm to see if we can get around using caret
  model <- glm(target~.,data=train_df)
  return(model)
}
