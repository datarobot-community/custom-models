#' Provides factor handling + mean imputation for numeric, etc 
#'
#' @param train data.frame or similar classes
#' @param test data.frame or similar
#' @param exclude character scalar or vector of any column names to ignore  
#' @return None
#' @export
#' 

# Purge / Impute Missing Data ---------------------------------------------
preprocess <- function(train, test, exclude){
  # preprocess() provides factor handling and mean imputation for numeric
  
  # Coherent handling of factors when unknown new levels may occur
  #
  # Replaces NA's in numeric values
  #
  # Character are converted to factor
  #
  # Integer are converted to numeric
  #
  # Supports up to 30 factor levels, for > 30 please re-encode as a best practice
  #
  # Ignores other data classes (date, etc)
  
  if(missing(exclude)){
    exclude <- c("next_term_retention", "group", "oos_Date",
      "Term_Start_Date")
  } else{
    if(!class(exclude)=="character"){
      cat("exclude must be a character or character vector")
      
    }
  }
  
  for(i in 1:length(colnames(train))){
    if(class(train[,i])  %in% c("Date", "POSIXct", "POSIXt")){
        next
      }
    if("integer"  %in% class(train[,i])){
      train[,i] <- as.numeric(train[,i])
    }
    if("numeric"  %in% class(train[,i])){
      train[,i] <- na.roughfix(train[,i])
    }
    if(("factor" %in% class(train[,i])) |
       ("character" %in% class(train[,i]))){
      if(colnames(train)[i] %in% exclude){
        next
      } else{
        train[,i]  <- as.character(train[,i])
        train[1,i] <- NA
        train[,i][is.na(train[,i])] <- "Missing"
        train[,i]  <- as.factor(train[,i])
      }
      if(length(levels(train[,i])) > 30){
        print("TOO MANY LEVELS!")
        print(colnames(train)[i])
        print(length(levels(train[,i])))
      }
    }
  }
  
  test <- test[,colnames(test) %in% colnames(train)]
  
  cn <- colnames(test)
  for(i in 1:length(cn)){
    if("integer"  %in% class(test[,i])){
      test[,i] <- as.numeric(test[,i])
    }
    if("numeric"  %in% class(test[,i])){
      test[,i] <- na.roughfix(test[,i])
    }
    if(("factor" %in% class(test[,i])) |
       ("character" %in% class(test[,i]))){
      if(colnames(test)[i] %in% exclude){
        next
      } else{
        test[,i]                          <- as.character(test[,i])
        test[1,i]                         <- NA
        test[,i][is.na(test[,i])] <- "Missing"
        test[,i]                          <- as.factor(test[,i])
      }
      
      if(!identical(levels(test[,i]),
                    levels(train[,colnames(test)[i]]))){
        l1 <- levels(train[,colnames(test)[i]])
        l2 <- levels(test[,i])
        
        test[,i] <- as.character(test[,i])
        test[,i] <- ifelse(test[,i] %in% l1,
                                 as.character(test[,i]),
                                 "Missing")
        
        if(length(l2) < length(l1) |
           (length(l2) == length(l1) &
            !(identical(levels(as.factor(test[,i])),
                        levels(train[,colnames(test)[i]]))))
        ){
          for(n in 1:length(l1[!l1 %in% l2])){
            test[10+n,i] <- l1[!l1 %in% l2][n]
          }
        }
        
        test[,i] <- as.factor(test[,i])
        
        if(!identical(levels(test[,i]),
                      levels(train[,colnames(test)[i]]))){
          print("There is a problem with factor levels!")
          print(i)
        }
      }
      
      if(length(levels(test[,i])) > 30){
        print(colnames(test)[i])
        print(length(levels(test[,i])))
      }
    }
  }
  
  for(i in 1:ncol(train)){
    if(!(identical(
      colnames(train)[i],
      colnames(test)[i]
    ))){
      print(colnames(train)[i])
      print(i)
    }
    if(!(identical(
      class(train)[i],
      class(test)[i]
    ))){
      print("There is a problem with:")
      print(colnames(train)[i])
      print("which is col number:")
      print(i)
    }
    if(!(identical(
      levels(train)[i],
      levels(test)[i]
    ))){
      print("There is a problem with:")
      print(colnames(train)[i])
      print(i)
    }
  }
  if(sum(is.na(test)) > 0){print("!!!WARNING!!! THERE IS A PROBLEM WITH MISSING DATA IN test")}
}
