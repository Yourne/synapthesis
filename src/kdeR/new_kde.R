plot_roc_curves <- function(prediction.matrix, labels.matrix) {
  roc.preds.mat <- ROCR::prediction(prediction.matrix, labels=labels.matrix)
  roc.perf.mat <- ROCR::performance(roc.preds.mat, "tpr", "fpr")
  plot(roc.perf.mat, col=as.list(1:3), lty=c(1, 2, 4))
  abline(0, 1, lty=3)
  grid()
  legend(x=0.5, y=0.4, legend = c("rule amount", "extreme amount", "extreme duration"), 
         col=c(1, 2, 3), lty=c(1, 2, 4), title = "Outlier type")
  roc.perf.mat.auc <- ROCR::performance(roc.preds.mat, "auc")
  auc.Hns <- roc.perf.mat.auc@y.values
}


#### LOAD DATA SETS ####
# df <- read.table("data10/processed_data.csv", header = TRUE, sep = ",", row.names = "index")
X_train <- read.csv("data10/train_test_open_full/X_train.csv")
X_test <- read.csv("data10/train_test_open_full/X_test.csv")
y_train <- read.csv("data10/train_test_open_full/y_train.csv")
y_test <- read.csv("data10/train_test_open_full/y_test.csv")

X_train <- read.csv("data10/train_test_open/X_train.csv")
X_test <- read.csv("data10/train_test_open/X_test.csv")
y_train <- read.csv("data10/train_test_open/y_train.csv")
y_test <- read.csv("data10/train_test_open/y_test.csv")

# select most relevant features
features <- c("be_amount", "pa_amount", "be_duration", "pa_duration")
X_train <- X_train[, features]
X_test <- X_test[, features]

# drop "rule_duration" columns as it contains only inlier
y_train <- y_train[, c("rule_amount", "extreme_amount", "extreme_duration")]
y_test <- y_test[, c("rule_amount", "extreme_amount", "extreme_duration")]

#### COMPUTE BANDWIDTH MATRICES ####
# Normal scale bandwidth
Hns <- ks::Hns(x = X_train)

# plug-in bandwidth
bin.size <- 15 # it should equal the default
start.time <- Sys.time()
Hpi <- ks::Hpi(x = X_train, nstage = 2) # 2.66 min
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)

# Least Squares cross validation bandwidth
start.time <- Sys.time()
Hlscv <- ks::Hlscv(x = X_train)
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)

#### MODEL EVALUATIONS ####
# normal scale bandwith
fhat.Hns <- ks::kde(x = X_test, H = Hns)
preds.Hns <- predict(fhat.Hns, x=X_test)
auc.Hns <- plot_roc_curves(cbind(preds.Hns, preds.Hns, preds.Hns), y_test)

# plug-in bandwdith
fhat.Hpi <- ks::kde(x = X_test, H = Hpi)
preds.Hpi <- predict(fhat.Hpi, x=X_test)
auc.Hpi <- plot_roc_curves(cbind(preds.Hpi, preds.Hpi, preds.Hpi), y_test)


# least squares cross validation bandwidth
fhat.Hlscv <- ks::kde(x = X_test, H = Hlscv)
preds.Hlscv <- predict(fhat.Hlscv, x=X_test)
auc.Hlscv <- plot_roc_curves(cbind(preds.Hlscv, preds.Hlscv, preds.Hlscv), y_test)

#### output to file ####
out <- data.frame(predict(fhat, x = aperta), row.names = row.names(aperta))
write.csv(out, file=paste(Sys.time(), "Hns_aperta.csv", sep="_"))
