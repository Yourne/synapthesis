#### FUNCTION DEFINITIONS ####
# interquartile scaling
IQR.transform <- function(df) {
  for(i in 1:ncol(df)) {
    df[, i] <- df[, i] / IQR(df[, i])
  }
  return(df)
}
#  boxcox transform
boxcox.transform <- function(x, lambda) {
  if (lambda == 0) {
    return(log(x))
  } else {
    return((x^lambda - 1) / lambda)
  }
}

#### LOAD DATA SETS ####
df <- read.table("data/aperta.csv", header = TRUE, sep = ",")
# df <- read.table("data10/contracts.csv", header = TRUE, sep = ",", row.names = "index")
lot_id <- df$id_lotto
# subset features
df <- subset(df, select = c("amount","be_med_ann_revenue", 
                            "pa_med_ann_expenditure", "duration"))

# load sample set
sample <- read.table("data/subset_aperta.csv", header = TRUE, sep = ",")
y <- as.logical(sample$outlier)
sample <- subset(sample, select = c("amount","be_med_ann_revenue", 
                                    "pa_med_ann_expenditure", "duration"))

#### PREPROCESS ####
# apparently, there are a few duplicates 
lot_id <- lot_id[!duplicated(df)]
df <- unique(df)

# replace duration 0 with 1
df[, "duration"][df[, "duration"] == 0] <- 1
sample[, "duration"][sample[, "duration"] == 0] <- 1

# apply interquartile scaling
df <- IQR.transform(df)
sample <- IQR.transform(sample)

# apply boxcox transformation
lambdas <- c(0, 0, 0, 0)
for (i in 1:ncol(df)) {
    # estimate lambda paramter with MLE
    out <- MASS::boxcox(df[, i] ~ 1, lambda = seq(-2, 2, 0.001), plotit = FALSE)
    lambdas[i] <- out$x[which.max(out$y)]
    # trasform data
    df[, i] <- boxcox.transform(df[, i], lambdas[i])
}
# boxcox transformation with lambdas computed on the whole dataset
for (i in 1:length(sample)) {
  sample[, i] <- boxcox.transform(sample[, i], lambdas[i])
}



#### COMPUTE BANDWIDTH MATRICES ####
# Normal scale bandwidth
Hns <- ks::Hns(x = df)

# plug-in bandwidth
bin.size <- 15 # it should equal the default
Hpi <- ks::Hpi(x = df, nstage = 2, bgridsize = rep(bin.size, length(df))) # like 1 minute

# Least Squares cross validation bandwidth
start.time <- Sys.time()
Hlscv <- ks::Hlscv(x = data.matrix(df))
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time

#### model evaluation ####
# normal scale bandwith
# bin.size <- 15
# fhat <- ks::kde(x = df, H = Hns, gridsize = rep(bin.size, length(df)))
# # compute prediction prob
# preds <- predict(fhat, x=sample)
# # compute AUC
# roc.Hns <- pROC::roc(y~preds, direction = "<")
# # best coordinates
# # pROC compute ROC as sensitivity vs specificity.
# # I want hit rate and false alarm rate
# # exploit relation: false alarm rate = 1 - specificity;  
# 1 - pROC::auc(roc.Hns)
# pROC::coords(roc.Hns, x="best")
# use verification package to plot the "right" ROC curve
# verification::roc.plot(y, preds)
# alternative: set the direction in the roc object
bin.size <- 15
print("Hns")
fhat <- ks::kde(x = df, H = Hns, gridsize = rep(bin.size, length(df)))
preds <- predict(fhat, x=sample)
roc.Hns <- pROC::roc(y~preds, direction = "<")
pROC::auc(roc.Hns)
pROC::coords(roc.Hns, x="best")
pROC::plot.roc(roc.Hns)

# plug-in bandwdith 
print("Hpi")
fhat <- ks::kde(x = df, H = Hpi, gridsize = rep(bin.size, length(df)))
preds <- predict(fhat, x=sample)
# roc.Hpi <- pROC::roc(y~preds, direction = "<")
# pROC::auc(roc.Hpi)
# pROC::coords(roc.Hpi, x="best")
# pROC::plot.roc(roc.Hpi)
roc.Hpi <- ROCit::rocit(preds, class=y, )
plot(roc.Hpi)

# least squares cross validation bandwidth
fhat <- ks::kde(x = df, H = Hlscv, gridsize = rep(bin.size, length(df)))
preds <- predict(fhat, x=sample)
roc.Hlscv <- pROC::roc(y~preds, direction = "<")
pROC::auc(roc.Hlscv)
pROC::coords(roc.Hlscv, x="best")
pROC::plot.roc(roc.Hlscv)

#### OUTPUT TO GOOGLE SHEET ####
fhat <- ks::kde(x = df, H = Hpi, gridsize = rep(bin.size, length(df)))
out <- data.frame(lot_id, predict(fhat, x = df))


#### MISCELLANEA ####
# # quantile quantile plot
# for (i in 1:ncol(df)) {
#   name <- names(df)[i]
#   car::qqPlot(df[, i], main = name, ylab = name)
# }

# eval.points decides where to evaluate fhat
# binned = FALSE imposes to use non-evenly spaced eval.points
# binned = TRUE (default) interpolates eval.points to have estimates of fhat
# fhat <- ks::kde(x = dt, H = Hns, eval.point=dt, binned=FALSE)

# now fhat may be vary overfitted. Let's compute the binned version
# then, get estimates for a given sample set
# predict(fhat, x=df)

# understand the cont attribute of object fhat
# sum(predict(fhat, x=df) <= fhat[["cont"]][["1%"]], na.rm = TRUE)
