library(MASS)

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

# ks

