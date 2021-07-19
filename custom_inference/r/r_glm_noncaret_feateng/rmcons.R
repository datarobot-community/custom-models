#' Removes constant columns
#'
#' @param x data.frame or similar classes
#'
#' @return None
#' @export
#' 
#' 
rm_cons <- function(x){
  cat("\n## Removing the constant features.\n")
  for (f in names(x)) {
    if (length(unique(x[[f]])) == 1) {
      cat(f, "is constant. It is has been deleted.\n")
      x[[f]] <- NULL
    }
  }
}
