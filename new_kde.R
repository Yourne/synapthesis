#### LOAD DATA SETS ####
df <- read.table("data10/processed_data.csv", header = TRUE, sep = ",", row.names = "index")
aperta <- df[df$id_award_procedure==1, ]
rm(df)
# drop the award procedure column
aperta <- aperta[, c("be_amount", "pa_amount", "be_duration", "pa_duration")]
# aperta <- aperta[, c("be_amount", "pa_amount", "duration")]
# remove duplicates
aperta <- unique(aperta)

# load outliers
y.rule_amount <- read.table("data10/rule_amount.csv", sep=',')
y.rule_amount <- y.rule_amount[row.names(aperta), ]
# y.rule_duration <- read.table("data10/rule_duration.csv", sep=',')
# y.rule_duration <- y.rule_duration[row.names(aperta), ]
y.extreme_amount <- read.table("data10/extreme_amount.csv", sep = ',')
y.extreme_amount <- y.extreme_amount[row.names(aperta), ]
y.extreme_duration <- read.table("data10/extreme_duration.csv", sep = ',')
y.extreme_duration <- y.extreme_duration[row.names(aperta), ]
outfnames = list("rule_amount.png", "extreme_amount.png", "extreme_duration.png")


#### COMPUTE BANDWIDTH MATRICES ####
# Normal scale bandwidth
Hns <- ks::Hns(x = aperta)

# plug-in bandwidth
bin.size <- 15 # it should equal the default
start.time <- Sys.time()
Hpi <- ks::Hpi(x = aperta, nstage = 2, bgridsize = rep(bin.size, length(df)))
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)

# Least Squares cross validation bandwidth
start.time <- Sys.time()
Hlscv <- ks::Hlscv(x = data.matrix(aperta))
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)

#### MODEL EVALUATIONS ####
# normal scale bandwith
print("Hns")
start.time <- Sys.time()
fhat.Hns <- ks::kde(x = aperta, H = Hns, binned=FALSE)
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)
preds.Hns <- predict(fhat.Hns, x=aperta)
# library(ROCR)
i = 0
for (y in list(y.rule_amount, y.extreme_amount, y.extreme_duration)) {
  i = i + 1
  roc.pred <- prediction(preds.Hns, y)
  roc.perf <- performance(roc.pred, "tpr", "fpr")
  png(paste("images/roc/kde/HNS", outfnames[i], sep="-"))
  plot(roc.perf)
  abline(a=0, b=1)
  dev.off()
  print(outfnames[i])
  print(performance(roc.pred, measure="auc")@y.values[[1]])
}

# plug-in bandwdith
start.time <- Sys.time()
fhat.Hpi <- ks::kde(x = df, H = Hpi, binned=FALSE)
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)
preds.Hpi <- predict(fhat.Hpi, x=aperta)
i = 0
for (y in list(y.rule_amount, y.extreme_amount, y.extreme_duration)) {
  i = i + 1
  roc.pred <- prediction(preds.Hpi, y)
  roc.perf <- performance(roc.pred, "tpr", "fpr")
  png(paste("images/roc/kde/HPI", outfnames[i], sep="-"), width = 480, height = 480)
  plot(roc.perf)
  abline(a=0, b=1)
  dev.off()
  print(outfnames[i])
  print(performance(roc.pred, measure="auc")@y.values[[1]])
}


# least squares cross validation bandwidth
start.time <- Sys.time()
fhat.Hlscv <- ks::kde(x = df, H = Hlscv, binned=FALSE)
stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
print(time.elapsed)
preds.Hlscv <- predict(fhat.Hlscv, x=aperta)
i = 0
for (y in list(y.rule_amount, y.extreme_amount, y.extreme_duration)) {
  i = i + 1
  roc.pred <- prediction(preds.Hlscv, y)
  roc.perf <- performance(roc.pred, "tpr", "fpr")
  png(paste("images/roc/kde/HLSCV", outfnames[i], sep="-"), width = 480, height = 480)
  plot(roc.perf)
  abline(0, 1)
  dev.off()
  print(outfnames[i])
  print(performance(roc.pred, measure="auc")@y.values[[1]])
}

#### output to file ####
out <- data.frame(predict(fhat, x = aperta), row.names = row.names(aperta))
write.csv(out, file=paste(Sys.time(), "Hns_aperta.csv", sep="_"))
