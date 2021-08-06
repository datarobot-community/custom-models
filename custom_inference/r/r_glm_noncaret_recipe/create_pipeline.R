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

  # set up the modeling pipeline
  model_recipe <- recipe(target ~ ., data = train_df) %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    
    # Numeric preprocessing
    step_medianimpute(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

  # Run the model using the builtin glm function 
  model <- glm(target~.,data=train_df)
  return(model)
}
