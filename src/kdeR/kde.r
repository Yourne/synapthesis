library(MASS)
library(ggplot2)

dt <- read.table("data/aperta.csv", header = TRUE, sep = ",")

# subset features
dt <- subset(dt, select = c("amount",
    "be_med_ann_revenue", "pa_med_ann_expenditure", "duration"))

# replace duration 0 with 1
dt[, "duration"][dt[, "duration"] == 0] <- 1

# interquantile scaling
for(i in 1:ncol(dt)) {
    dt[, i] <- dt[, i] / IQR(dt[, i])
}

#  define boxcox transform
boxcox.transform <- function(x, lambda) {
    if (lambda == 0) {
        return(log(x))
    } else {
        return((x^lambda - 1) / lambda)
    }
}

# apply boxcox transformatiion
for (i in 1:ncol(dt)) {
    # estimate lambda paramter with MLE
    out <- boxcox(dt[, i] ~ 1, lambda = seq(-2, 2, 0.001), plotit = FALSE)
    lambda <- out$x[which.max(out$y)]
    # trasform data
    dt[, i] <- boxcox.transform(dt[, i], lambda)
}

# Normal scale bandwidth
(Hns <- ks::Hns(x = dt))

# Kernel density estimation
# eval.points decides where to evaluate fhat
# binned = FALSE imposes to use non-evenly spaced eval .points
# binned = TRUE (default) interpolates eval.points to have estimates of fhat
fhat <- ks::kde(x = dt, H = Hns, eval.point=dt, binned=FALSE)

# now fhat may be vary overfitted. Let's compute the binned version
fhat <- ks::kde(x= dt, H = Hns)
# then, estimate with
predict(fhat, x=dt)

# understand the cont attribute of object fhat
sum(predict(fhat, x=dt) <= fhat[["cont"]][["1%"]], na.rm = TRUE)

