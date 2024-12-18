
GenWalsh <- function(n) {
  numrow <- 2
  Wtemp <- matrix(0, n, n)
  W <- Wtemp
  W[1,1] <- 1
  W[1,2] <- 1
  W[2,1] <- 1
  W[2,2] <- -1
 
  while (numrow < n) {
    k <- 1
    for (i in 1:numrow) {
      for (j in 1:numrow) {
        Wtemp[k,j] <- W[i,j]
      }
      k <- k + 1
      for (j in 1:numrow) {
        Wtemp[k,j] <- W[i,j]
      }
      k <- k + 1
    }
   
    for (i in 1:(numrow*2)) {
      for (j in (numrow+1):(numrow*2)) {
        if ((i+1) %% 2 == 1) {
          Wtemp[i,j] <- Wtemp[i,j-numrow] * -1
        } else {
          Wtemp[i,j] <- Wtemp[i,j-numrow]
        }
      }
    }
    numrow <- numrow * 2
    W[1:numrow, 1:numrow] <- Wtemp[1:numrow, 1:numrow]
  }
 
  return(W)
}

matrix <- GenWalsh(16)
write.table(matrix, file="mymatrix.txt", row.names=FALSE, col.names=FALSE)

