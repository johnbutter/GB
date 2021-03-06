
cat("Loading libraries...\n");
library(xgboost)
library(data.table)
library(Matrix)
library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(treemap)
library(Ckmeans.1d.dp)
library(party)
library(rpart)
library(randomForest)
library(rpart.plot)
library(Metrics)
library(arules)
library(arulesViz)
library(moments)

cat("Reading CSV files and loading frames...\n");
completeData <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/train.csv", header = T, stringsAsFactors = T))
client <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/cliente_tabla.csv")
product <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/producto_tabla.csv")
town <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/town_state.csv")
completeTest <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/test.csv", header = T, stringsAsFactors = T))
returnsample <- as.data.table(fread("~/Desktop/Data/Kaggle/GrupoBimbo/returnsample.csv", header = T, stringsAsFactors = T))
  
# Read in Training data
hier_train <- fread("~/Desktop/Data/Kaggle/GrupoBimbo/train.csv",select = c('Agencia_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil')
                      , colClasses=c(Agencia_ID="numeric", Ruta_SAK="numeric", Cliente_ID="numeric",Producto_ID="numeric",Demanda_uni_equil="numeric"))

# Read in Test data 
hier_test <- fread('~/Desktop/Data/Kaggle/GrupoBimbo/test.csv', 
                     select = c('id','Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Cliente_ID'),
                     colClasses=c(Producto_ID="numeric", Agencia_ID="numeric", Ruta_SAK="numeric", Cliente_ID="numeric"))

# Read in Week 9 Holdout Test data 
week_nine <- fread('~/Desktop/Data/Kaggle/GrupoBimbo/WEEK9.csv', 
                     select = c('Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Cliente_ID', 'Demanda_uni_equil'),
                     colClasses=c(Producto_ID="numeric", Agencia_ID="numeric", Ruta_SAK="numeric", Cliente_ID="numeric", Demanda_uni_equil="numeric"))


##########################################
### SAMPLING LOGIC HERE - MODIFY AS NEEDED
##########################################
# Here is where we should set up and select which sample we test - default is random
# Change The Sampling Fraction to adjust the size of the training data set
############### PICK ONE OR SERIES OF FOLLOWING ####################
cat("Splitting data...\n");
# actual demand between 0 and 1000 covers over 60M records
clients<-subset(completeData, Demanda_uni_equil<=2000&Demanda_uni_equil>-1)

# Break Training into WEEK9 testing and WEEK3_8 for training
testweek9<-subset(clients, Semana==9)
week3_8<-subset(clients, Semana<9)

# Random sample subset for training set to desired %
cat("Number of rows:",nrow(clients),"\n")
train<- clients %>% sample_frac(0.01) #settled on .1-10% for the model build
test<-completeTest

###############
# Feature Build 
###############
##################################################
# Building Feature Tables for Product Client pairs
# For each PC pair, calculate the week's total, mean, median
# This will be used to create an new feature for each item purchased
# to have their previous 3 weeks data included on the line.  Additionally,
# we are creating an indicator flag that states whether the item was purchased
# previously.  As well as a product client mean, median for all of their weeks.
# We will determine which of these features helps us predict as we train on 3-8
# and test on week 9 - prior to finalizing the features and approach for 
# final model submission.
# Input Brand Feature created by preprocessing product table.
###################################################
xgb_train<-as.data.table(train)
setkey(xgb_train,Producto_ID,Cliente_ID,Semana)
feature_PCW_summary<-xgb_train[,list(
                      PCW_total=sum(Demanda_uni_equil),
                      PCW_mean=mean(Demanda_uni_equil),
                      PCW_median=as.numeric(median(Demanda_uni_equil))),
                by=.(Producto_ID, Cliente_ID,Semana)
                ] 
feature_PC<-xgb_train[,list(Mult_Wks=if_else(is.na(sd(Demanda_uni_equil)),0,1),
                    PC_total=sum(Demanda_uni_equil),
                   PC_mean=mean(Demanda_uni_equil),
                   PC_median=as.numeric(median(Demanda_uni_equil))),
             by=.(Producto_ID, Cliente_ID)
             ] 
feature_brand <- as.data.table(fread("~/Desktop/Data/Kaggle/GrupoBimbo/brand.csv", header = T, stringsAsFactors = T))
setnames(feature_brand, "PID", "Producto_ID")
setkey(feature_brand, by="Producto_ID")
setkey(feature_PC,Producto_ID,Cliente_ID)
setkey(feature_PCW_summary,Producto_ID,Cliente_ID,Semana)
# feature_PC
# feature_brand 
# feature_PCW_summary 


############################
# Market Basket Set Analysis
############################
# Can use any week or combined weeks...change testweek9 to week3_8
marketbasket<-data.table(testweek9)
marketbasket<-data.table(distinct(marketbasket, Producto_ID, Cliente_ID))
i <- split(marketbasket$Producto_ID, marketbasket$Cliente_ID)
sapply(head(i), function(x) sum(duplicated(x))) # check for duplicates in basket
txn <- as(i, "transactions")
basket_rules <- apriori(txn, parameter = list(sup = 0.01, conf = 0.5, target="rules", maxlen=5))
itemFrequencyPlot(txn, topN = 25)

plot(basket_rules)
inspect(head(sort(basket_rules, by= "lift"), 20))

subrules2 <- head(sort(basket_rules, by= "lift"), 100)
plot(subrules2, method="graph",control=list(type="items",main=""))


###########################################
# EDA of completeDataSet (random fractions)
###########################################
ggplot(completeData %>% sample_frac(0.005))+
  geom_histogram(aes(x=Semana), color="black", fill="cyan", alpha=0.5)+
  scale_x_continuous(breaks=1:10)+
  scale_y_continuous(name="Client / Product deliveries")+
  theme_bw()

ggplot(completeData %>% sample_frac(0.005))+
  geom_histogram(aes(x=Agencia_ID), color="black", fill="orange", alpha=0.5)+
  scale_x_continuous(breaks=1:30)+
  scale_y_continuous(name="Frequency")+
  theme_bw()



# Client Sales
sales <- completeData %>%
  group_by(Cliente_ID) %>%
  summarise(Units = sum(Venta_uni_hoy),
            Pesos = sum(Venta_hoy),
            Return_Units = sum(Dev_uni_proxima),
            Return_Pesos = sum(Dev_proxima),
            Net = sum(Demanda_uni_equil)) %>%
  mutate(Return_Rate = Return_Units / (Units+Return_Units),
         Avg_Pesos = Pesos / Units) %>%
  mutate(Net_Pesos = Pesos - Return_Pesos) %>%
  inner_join(client, by="Cliente_ID") %>%
  arrange(desc(Pesos))

treemap(sales[1:50, ], 
        index=c("NombreCliente"), vSize="Units", vColor="Return_Rate", 
        palette=c("#AAEEEE","#AAEEEE","#AA0000"),
        type="value", title.legend="Units return %", title="Top 50 clients")

# Canals x Agencies
agencias.canals <- completeData %>%
  group_by(Agencia_ID) %>%
  summarise(n_canals = n_distinct(Canal_ID))

ggplot(agencias.canals)+
  geom_histogram(aes(x=n_canals), fill="cyan", color="black", alpha="0.5", binwidth=0.5)+
  scale_x_continuous(name="Number of canals", breaks=1:5)+
  scale_y_continuous(name="Number of agencies")+
  theme(axis.text.x=element_text(hjust=1))+
  theme_bw()

# Client Routes
client.routes <- completeData %>%
  group_by(Cliente_ID) %>%
  summarise(n_routes = n_distinct(Ruta_SAK))

ggplot(client.routes)+
  geom_histogram(aes(x=n_routes), fill="cyan", color="black", alpha="0.5", binwidth=1)+
  scale_x_continuous(name="Number of clients")+
  scale_y_continuous(name="Number of routes", labels=function(x)paste(x/1000, "k"))+
  theme_bw()

# Products x Clients
client.Product <- completeData %>%
  group_by(Cliente_ID) %>%
  summarise(n_products = n_distinct(Producto_ID))

ggplot(client.Product)+
  geom_histogram(aes(x=n_products), fill="cyan", color="black", alpha="0.5", binwidth=1)+
  scale_x_continuous(name="Number of products")+
  scale_y_continuous(name="Number of clients", labels=function(x)paste(x/1000, "k"))+
  theme_bw()

# Products x Routes
client.Product <- completeData %>%
  group_by(Ruta_SAK) %>%
  summarise(n_products = n_distinct(Producto_ID))

ggplot(client.Product)+
  geom_histogram(aes(x=n_products), fill="cyan", color="black", alpha="0.5", binwidth=1)+
  scale_x_continuous(name="Number of products")+
  scale_y_continuous(name="Number of routes", labels=function(x)paste(x/1000, "k"))+
  theme_bw()

##########################################
# MODEL 1 - HIERARCHICAL
# Create Median and Means Prediction Table
##########################################
kaggle<-TRUE
PCAcf1<-0.70
PCAcf2<-0.7
PCRcf1<-0.70
PCRcf2<-0.7
PAcf1<-0.75
PAcf2<-0.1
PRcf1<-0.76
PRcf2<-0.1
Pcf1<-0.76
Pcf2<-0.25
Cf1<-0.76
Cf2<-0.25

# Use Next 3 Lines if we want to use Week9 instead of Kaggle TEST file
#hier_test<-week_nine
#hier_test$Demanda_uni_equil<-NULL
#kaggle==FALSE

#global mean/median (+modification)
global <- hier_train[, mean(Demanda_uni_equil)]

#table of product -client -agent mean 
setkey(hier_train, Producto_ID, Cliente_ID, Agencia_ID)
mean_Prod_Client_Agent <- hier_train[, exp(mean(log1p(Demanda_uni_equil)))*PCAcf1+PCAcf2,by = .(Producto_ID,Cliente_ID,Agencia_ID)]
setnames(mean_Prod_Client_Agent,"V1","PCA")

#table of product -client -route mean 
setkey(hier_train, Producto_ID, Cliente_ID, Ruta_SAK)
mean_Prod_Client_Ruta <- hier_train[, exp(mean(log1p(Demanda_uni_equil)))*PCRcf1+PCRcf2,by = .(Producto_ID,Cliente_ID,Ruta_SAK)]
setnames(mean_Prod_Client_Ruta,"V1","PCR")

#table of product -route mean
setkey(hier_train, Producto_ID, Ruta_SAK)
mean_Prod_Ruta <- hier_train[, exp(mean(log1p(Demanda_uni_equil)))*PRcf1+PRcf2,by = .(Producto_ID, Ruta_SAK)]
setnames(mean_Prod_Ruta,"V1","PR")

#table of product -agency mean
setkey(hier_train, Producto_ID, Agencia_ID)
mean_Prod_Agency <- hier_train[, exp(mean(log1p(Demanda_uni_equil)))*PAcf1+PAcf2,by = .(Producto_ID, Agencia_ID)]
setnames(mean_Prod_Agency,"V1","PA")

#table of product overall mean 
mean_Prod <- hier_train[, exp(mean(log1p(Demanda_uni_equil)))*Pcf1+Pcf2, by = .(Producto_ID)]
setnames(mean_Prod,"V1","P")

#table of product -client -rEturns mean 
setkey(preturns, Producto_ID, Cliente_ID)
mean_Prod_Client_Returns <- preturns[, exp(mean(log1p(Dev_uni_proxima)))-1,by = .(Producto_ID,Cliente_ID)]
setnames(mean_Prod_Client_Returns,"V1","PCE")

#table of client overall mean 
setkey(hier_train, Cliente_ID)
mean_Client <- hier_train[, exp(mean(log1p(Demanda_uni_equil)))*Cf1+Cf2, by = .(Cliente_ID)]
setnames(mean_Client,"V1","C")

# add columns PCA, PCR, PR, PA, P, C, global for mean 
hold<-NULL
hold <- merge(hier_test, mean_Prod_Client_Agent, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"), all.x = TRUE)
hold <- merge(hold, mean_Prod_Client_Ruta, by = c("Producto_ID", "Cliente_ID", "Ruta_SAK"), all.x = TRUE)
hold <- merge(hold, mean_Prod_Ruta, by = c("Producto_ID", "Ruta_SAK"), all.x = TRUE)
hold <- merge(hold, mean_Prod_Agency, by = c("Producto_ID", "Agencia_ID"), all.x = TRUE)
hold <- merge(hold, mean_Prod, by = "Producto_ID", all.x = TRUE)
hold <- merge(hold, mean_Client, by = "Cliente_ID", all.x = TRUE)

######################
# Hierarchy 1 PCA-PR-P
######################
hold$Pred1<-NULL
hold$Pred1 <- hold$PCA
hold[is.na(Pred1)]$Pred1 <- hold[is.na(Pred1)]$PR
hold[is.na(Pred1)]$Pred1 <- hold[is.na(Pred1)]$P
hold[is.na(Pred1)]$Pred1 <- global

######################
# Hierarchy 2 PCR-PA-P
######################
hold$Pred2<-NULL
hold$Pred2 <- hold$PCR
hold[is.na(Pred2)]$Pred2 <- hold[is.na(Pred2)]$PA
hold[is.na(Pred2)]$Pred2 <- hold[is.na(Pred2)]$P
hold[is.na(Pred2)]$Pred2 <- global

######################
# Hierarchy 3 PCR-PA-C-PR-P
######################
hold$Pred3<-NULL
hold$Pred3 <- hold$PCR
hold[is.na(Pred3)]$Pred3 <- hold[is.na(Pred3)]$PA
hold[is.na(Pred3)]$Pred3 <- hold[is.na(Pred3)]$C
hold[is.na(Pred3)]$Pred3 <- hold[is.na(Pred3)]$P
hold[is.na(Pred3)]$Pred3 <- global


# If doing Kaggle Submission sort by ID else others
if (kaggle ==TRUE) {
  setkey(hold, id)
}  else {
    setkey(week_nine, Producto_ID, Cliente_ID, Agencia_ID, Ruta_SAK)
    setkey(hold,Producto_ID, Cliente_ID, Agencia_ID, Ruta_SAK )
    xr1<-rmsle(week_nine$Demanda_uni_equil, hold$Pred1)
    xr2<-rmsle(week_nine$Demanda_uni_equil, hold$Pred2)
    at<-sum(week_nine$Demanda_uni_equil)
    pt1<-sum(hold$Pred1)
    pt2<-sum(hold$Pred2)
    cat("Actual Demand Total:", at,"\n")
    cat("Predicted HIER1 Demand Total:", pt1,"\n")
    cat("HIER1 RMSLE:", xr1,"\n")
    cat("Predicted HIER2 Demand Total:", pt2,"\n")
    cat("HIER2 RMSLE:", xr2,"\n")
}

####### THIS IS HIERARCHY PREDICTION SETS 1-2 ###############
hierpred <- data.frame(id=hold$id, hp1=hold$Pred1, hp2=hold$Pred2, hp3=hold$Pred3)
#############################################################

################
# MODEL 2
# Boosting Model
################
keep.id<-test$id
train$id <- NULL;
test$id <- NULL;

ohetrain<-train
ohetest<-test

cat("Creating new features...\n");
## Must have at least 1 numeric field in the matrix for xgboost to work
ohetrain$Canal_ID<-1.0 
ohetest$Canal_ID<-1.0
## Brand
ohetrain <- merge(ohetrain, feature_brand, by = c("Producto_ID"), all.x = TRUE)
ohetest <- merge(ohetest, feature_brand, by = c("Producto_ID"), all.x = TRUE)
## Prod-Client pair total demand, mean, median, and whether it was purchased more than 1
ohetrain <- merge(ohetrain, feature_PC, by = c("Producto_ID", "Cliente_ID"), all.x = TRUE)
ohetest <- merge(ohetest, feature_PC, by = c("Producto_ID", "Cliente_ID"), all.x = TRUE)
## Add key columns for previous week
ohetrain$Prev1<-as.integer(ohetrain$Semana-1)
ohetrain$Prev2<-as.integer(ohetrain$Semana-2)
ohetrain$Prev3<-as.integer(ohetrain$Semana-3)
ohetest$Prev1<-as.integer(9)
ohetest$Prev2<-as.integer(ohetest$Semana-2)
ohetest$Prev3<-as.integer(ohetest$Semana-3)
## Add Previous 1 Week, first change column names
setnames(feature_PCW_summary, c("PCW_total","PCW_mean","PCW_median"),c("PCW1_total","PCW1_mean","PCW1_median"))
ohetrain <- merge(ohetrain, feature_PCW_summary, by.x = c("Producto_ID", "Cliente_ID", "Prev1"), by.y= c("Producto_ID", "Cliente_ID", "Semana"), all.x = TRUE)
ohetest <- merge(ohetest, feature_PCW_summary, by.x = c("Producto_ID", "Cliente_ID", "Prev1"), by.y= c("Producto_ID", "Cliente_ID", "Semana"), all.x = TRUE)
## Add Previous 2 Week, first change column names
setnames(feature_PCW_summary, c("PCW1_total","PCW1_mean","PCW1_median"),c("PCW2_total","PCW2_mean","PCW2_median"))
ohetrain <- merge(ohetrain, feature_PCW_summary, by.x = c("Producto_ID", "Cliente_ID", "Prev2"), by.y= c("Producto_ID", "Cliente_ID", "Semana"), all.x = TRUE)
ohetest <- merge(ohetest, feature_PCW_summary, by.x = c("Producto_ID", "Cliente_ID", "Prev2"), by.y= c("Producto_ID", "Cliente_ID", "Semana"), all.x = TRUE)
## Add Previous 3 Week, first change column names
setnames(feature_PCW_summary, c("PCW2_total","PCW2_mean","PCW2_median"),c("PCW3_total","PCW3_mean","PCW3_median"))
ohetrain <- merge(ohetrain, feature_PCW_summary, by.x = c("Producto_ID", "Cliente_ID", "Prev3"), by.y= c("Producto_ID", "Cliente_ID", "Semana"), all.x = TRUE)
ohetest <- merge(ohetest, feature_PCW_summary, by.x = c("Producto_ID", "Cliente_ID", "Prev3"), by.y= c("Producto_ID", "Cliente_ID", "Semana"), all.x = TRUE)

cat("Treating features...\n");
## impute train / test here
# Fix NA brand with 16 (no brand brand)
ohetrain$CATBRAND[is.na(ohetrain$BRAND)] <- 16 # no identification
ohetest$CATBRAND[is.na(ohetest$BRAND)] <- 16 # no identification
# Fix NA for all new features
ohetrain$Mult_Wks[is.na(ohetrain$Mult_Wks)] <- 0
ohetest$Mult_Wks[is.na(ohetest$Mult_Wks)] <- 0 
ohetrain$Mult_Wks<-as.integer(ohetrain$Mult_Wks)
ohetest$Mult_Wks<-as.integer(ohetest$Mult_Wks)
ohetest$PC_total[is.na(ohetest$PC_total)] <- 0
ohetest$PC_mean[is.na(ohetest$PC_mean)] <- 0
ohetest$PC_median[is.na(ohetest$PC_median)] <- 0
ohetrain$PCW1_total[is.na(ohetrain$PCW1_total)] <- 0
ohetrain$PCW1_mean[is.na(ohetrain$PCW1_mean)] <- 0
ohetrain$PCW1_median[is.na(ohetrain$PCW1_median)] <- 0
ohetrain$PCW2_total[is.na(ohetrain$PCW2_total)] <- 0
ohetrain$PCW2_mean[is.na(ohetrain$PCW2_mean)] <- 0
ohetrain$PCW2_median[is.na(ohetrain$PCW2_median)] <- 0
ohetrain$PCW3_total[is.na(ohetrain$PCW3_total)] <- 0
ohetrain$PCW3_mean[is.na(ohetrain$PCW3_mean)] <- 0
ohetrain$PCW3_median[is.na(ohetrain$PCW3_median)] <- 0
ohetest$PCW1_total[is.na(ohetest$PCW1_total)] <- 0
ohetest$PCW1_mean[is.na(ohetest$PCW1_mean)] <- 0
ohetest$PCW1_median[is.na(ohetest$PCW1_median)] <- 0
ohetest$PCW2_total[is.na(ohetest$PCW2_total)] <- 0
ohetest$PCW2_mean[is.na(ohetest$PCW2_mean)] <- 0
ohetest$PCW2_median[is.na(ohetest$PCW2_median)] <- 0
ohetest$PCW3_total[is.na(ohetest$PCW3_total)] <- 0
ohetest$PCW3_mean[is.na(ohetest$PCW3_mean)] <- 0
ohetest$PCW3_median[is.na(ohetest$PCW3_median)] <- 0

## drop from train and test here if not used or deemed not useful
cat("Dropping features...\n");
# Boost Performance was 1.03 with brand so must remove
ohetrain$BRAND<-NULL
ohetest$BRAND<-NULL
ohetrain$Venta_uni_hoy<-NULL
ohetrain$Venta_hoy<-NULL
ohetrain$Dev_uni_proxima<-NULL
ohetrain$Dev_proxima<-NULL

#Retain Log Transformed Target Demand - added one here and remove 1 when prediction saved
target = log(ohetrain$Demanda_uni_equil+1);
target[target<0] = 0

#Build sparse matrix training with one hot encoding
sparse_matrix_train<-NULL
sparse_matrix_train <- sparse.model.matrix(PC_total +PC_mean+PC_median
                                           +PCW1_total+PCW1_mean+PCW1_median
                                           +PCW2_total+PCW2_mean+PCW2_median
                                           +PCW3_total+PCW3_mean+PCW3_median
                                             ~ Prev3+ Prev2+Prev1+Mult_Wks+Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID  -1, data = ohetrain)
merge_totals_with_ohe<-NULL
merge_totals_with_ohe<-subset(ohetrain, select=c(PCW1_total,PCW1_mean,PCW1_median
                                                ,PCW2_total,PCW2_mean,PCW2_median
                                                ,PCW3_total,PCW3_mean,PCW3_median))
merge_totals_with_ohe<-data.matrix(merge_totals_with_ohe)
sparse_matrix_train<-cbind2(sparse_matrix_train,merge_totals_with_ohe)

# Build sparse Test matrix with one hot encoding
sparse_matrix_test<-NULL
sparse_matrix_test <- sparse.model.matrix( PC_total +PC_mean+PC_median
                                           +PCW1_total+PCW1_mean+PCW1_median
                                           +PCW2_total+PCW2_mean+PCW2_median
                                           +PCW3_total+PCW3_mean+PCW3_median
                                           ~ Prev3+ Prev2+Prev1+Mult_Wks+Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID  -1, data = ohetest)

merge_totals_with_ohe<-NULL
merge_totals_with_ohe<-subset(ohetest, select=c(PCW1_total,PCW1_mean,PCW1_median
                                                 ,PCW2_total,PCW2_mean,PCW2_median
                                                 ,PCW3_total,PCW3_mean,PCW3_median))
merge_totals_with_ohe<-data.matrix(merge_totals_with_ohe)
sparse_matrix_test<-cbind2(sparse_matrix_test,merge_totals_with_ohe)

# build data matrix with ohe encoded training file
cat("Creating data.matrix...\n");
preds <- rep(0,nrow(test));
trainM<-data.matrix(sparse_matrix_train, rownames.force = NA); #playing around with OHE
cat("Creating DMatrix for xgboost...\n");
dtrain <- xgb.DMatrix(data=trainM, label=target, missing = NaN);
watchlist <- list(trainM=dtrain);

################# Iterate 50-100 Times through cross validation to find best 
################# parameter list. Modified to linear regression #######################################

nloops<-100
best_param = list()
best_seednumber = 1969
best_error = Inf
best_error_index = 0
library(mlr)
for (iter in 1:nloops) {
  param <- list(objective = "reg:linear",
                max_depth = sample(10:50, 1), #25 produced a 0.53
                eta = runif(1, .1, .9), #0.2784 
                gamma = runif(1, 0.0, 0.2), #0.134885
                subsample = runif(1, .6, .9), #0.7742556
                colsample_bytree = runif(1, .5, .8), #0.5917445
                min_child_weight = sample(1:40, 1), #9
                max_delta_step = sample(1:10, 1) #4
  )
  seed.number = 1240#sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  cv.nround = 50
  cv.nfold = 5
  mdcv <- xgb.cv(data=dtrain, params = param, watchlist = watchlist, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early.stop.round=2, maximize=FALSE)
  
  min_error = min(mdcv$test.rmse.mean)
  min_error_index = which.min( mdcv$test.rmse.mean )
  
  if (min_error < best_error) {
    best_error = min_error
    best_error_index = min_error_index
    best_seednumber = seed.number
    best_param = param
  }
  cat("Loop:", iter,"\n");
}

nround = best_error_index
set.seed(best_seednumber)
cat("Best round:", nround,"\n");
cat("Best result:",best_error,"\n");

clf <- xgb.train(   params              = best_param, 
                    data                = dtrain, 
                    nrounds             = nround, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

##########################
# Test Variable Importance
##########################
importance <- xgb.importance(feature_names = colnames(trainM), model = clf)
xgb.plot.importance(importance)
print(importance)

testM <-data.matrix(sparse_matrix_test, rownames.force = NA);
preds <- round(predict(clf, testM),10);
preds <- exp(preds)-1

#Fix Negatives
preds[preds<0] = 0

####### THIS IS XGBOOST PREDICTION SET and Parameters ###############
write.csv(data.frame(best_param), "~/Desktop/Data/Kaggle/GrupoBimbo/XGBPARAM.csv", row.names = F)
xgbpred <- data.frame(id=keep.id, xgbp=preds);
#############################################################


############################
# BUILD ENSEMBLE
############################

# merge XGBOOST and HIERARCHIES by id
ensemble<-NULL
ensemble<-merge(xgbpred, hierpred, by = c("id"), all.x = TRUE)
ensemble$minp<-apply(ensemble[,2:5],1,min)
ensemble$maxp<-apply(ensemble[,2:5],1,max)

#straight averages
ensemble$averagex123<-(ensemble$hp3+ensemble$hp2+ensemble$hp1+ensemble$xgbp)/4
ensemble$averagex12<-(ensemble$hp2+ensemble$hp1+ensemble$xgbp)/3
ensemble$averagex3<-(ensemble$hp3+ensemble$xgbp)/2
ensemble$averagex2<-(ensemble$hp2+ensemble$xgbp)/2
ensemble$averagex1<-(ensemble$hp1+ensemble$xgbp)/2
ensemble$averageh12<-(ensemble$hp2+ensemble$hp1)/2
ensemble$averageh123<-(ensemble$hp3+ensemble$hp2+ensemble$hp1)/3
ensemble$average.min<-(ensemble$hp3+ensemble$hp2+ensemble$hp1+ensemble$xgbp-ensemble$minp)/3
ensemble$average.max<-(ensemble$hp3+ensemble$hp2+ensemble$hp1+ensemble$xgbp-ensemble$maxp)/3

############################################################
# write out files for KAggle
cat("Saving the submission files\n");
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$hp1), "~/Desktop/Data/Kaggle/GrupoBimbo/HIER1.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$hp2), "~/Desktop/Data/Kaggle/GrupoBimbo/HIER2.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$hp3), "~/Desktop/Data/Kaggle/GrupoBimbo/HIER3.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$xgbp), "~/Desktop/Data/Kaggle/GrupoBimbo/XGB.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$averagex1), "~/Desktop/Data/Kaggle/GrupoBimbo/XGBHIER1.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$averagex12), "~/Desktop/Data/Kaggle/GrupoBimbo/XGBHIER12.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$averagex123), "~/Desktop/Data/Kaggle/GrupoBimbo/XGBHIER123.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$averageh12), "~/Desktop/Data/Kaggle/GrupoBimbo/HIER12.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$averageh123), "~/Desktop/Data/Kaggle/GrupoBimbo/HIER123.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$averagex2), "~/Desktop/Data/Kaggle/GrupoBimbo/XGBHIER2.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$average.min), "~/Desktop/Data/Kaggle/GrupoBimbo/3MINUSLOW.csv", row.names = F)
write.csv(data.frame("id"=ensemble$id, "Demanda_uni_equil"=ensemble$average.max), "~/Desktop/Data/Kaggle/GrupoBimbo/3MINUSHIGH.csv", row.names = F)
########## Completed with KAGGLE evaluation #################



##########################################################
######### Code to Use completed training on Week 9 Holdout
######### Compare results using RMSLE to actual and 
######### other evaluations such as total cost, etc.
##########################################################
# MODEL 2 - XGBOOST
##########################################################
actual9<-data.frame(Semana=testweek9$Semana, Agencia_ID=testweek9$Agencia_ID,
                    Ruta_SAK=testweek9$Ruta_SAK,Cliente_ID=testweek9$Cliente_ID,
                    Producto_ID=testweek9$Producto_ID,Demanda_uni_equil=testweek9$Demanda_uni_equil)
test9<-actual9
test9$Canal_ID<-1.0
test9$Demanda_uni_equil<-NULL
sparse_matrix_test <- sparse.model.matrix( ~ Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID -1, data = test9)
testM <-data.matrix(sparse_matrix_test, rownames.force = NA);
preds <- round(predict(clf, testM),10);
preds <- exp(preds)-1

#Fix Negatives
preds[preds<0] = 0

#Truncate anything larger than 1000
#preds[preds>1000] = 1000

# use rmsle to compare actual from week 9 to predicted
# this should be close or better than the xgb results for KAGGLE
# to confirm reasonableness of model at predicting for any week.
xr<-rmsle(actual9$Demanda_uni_equil, preds)
at<-sum(actual9$Demanda_uni_equil)
pt<-sum(preds)
cat("Actual Demand Total:", at,"\n")
cat("Predicted Demand Total:", pt,"\n")
cat("XGBOOST WEEK 9 RMSLE:", xr,"\n")
####### END WEEK 9 FOR MODEL 2


##################################
# MODEL 1 - HIERARCHICAL
##################################
hier_test<-week_nine
hier_test$Demanda_uni_equil<-NULL
# sort by key
setkey(hier_train, Producto_ID, Cliente_ID, Agencia_ID)

#table of overall mean (+modification)
medianp <- hier_train[, mean(Demanda_uni_equil)]

###### DO NOT NEED TO REBUILD THE TRAINED MEANS UNLESS YOU WANT TO CHANGE THEM
#################################
PCAcf1<-0.7
PCAcf2<-0.7
PRcf1<-0.8
PRcf2<-0.1
Pcf1<-1.1
Pcf2<-.25

#table of product -client -agent mean .75+.75
mean_Prod_Client_Agent <- hier_train[, exp(mean(log(Demanda_uni_equil+1)))*PCAcf1+PCAcf2,by = .(Producto_ID,Cliente_ID,Agencia_ID)]
setnames(mean_Prod_Client_Agent,"V1","PCA")

#table of product -route mean .8+.1
mean_Prod_Ruta <- hier_train[, exp(mean(log(Demanda_uni_equil+1)))*PRcf1+PRcf2,by = .(Producto_ID, Ruta_SAK)]
setnames(mean_Prod_Ruta,"V1","PR")

#table of product overall mean 1.15+.25
mean_Prod <- hier_train[, exp(mean(log(Demanda_uni_equil+1)))*Pcf1+Pcf2, by = .(Producto_ID)]
setnames(mean_Prod,"V1","P")

########################################
# WEEK 9 HIER SKIP TO HERE
# add column PCA for mean by product client agent
hold<-NULL
hold <- merge(hier_test, mean_Prod_Client_Agent, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"), all.x = TRUE)

# add column PR for mean by product client 
hold <- merge(hold, mean_Prod_Ruta, by = c("Producto_ID", "Ruta_SAK"), all.x = TRUE)

# add column P that contains mean by Product
hold <- merge(hold, mean_Prod, by = "Producto_ID", all.x = TRUE)

# Prediction intially set to be MNPA 
hold$Pred <- hold$PCA

# if still NA use mean product route
hold[is.na(Pred)]$Pred <- hold[is.na(Pred)]$PR

# if still NA use mean product
hold[is.na(Pred)]$Pred <- hold[is.na(Pred)]$P

# if still NA use overall median
hold[is.na(Pred)]$Pred <- medianp
  
setkey(week_nine, Producto_ID, Cliente_ID, Agencia_ID, Ruta_SAK)
setkey(hold,Producto_ID, Cliente_ID, Agencia_ID, Ruta_SAK )
xr<-rmsle(week_nine$Demanda_uni_equil, hold$Pred)
at<-sum(week_nine$Demanda_uni_equil)
pt<-sum(hold$Pred)
cat("Actual Demand Total:", at,"\n")
cat("Predicted Demand Total:", pt,"\n")
cat("HIERARCHICAL WEEK 9 RMSLE:", xr,"\n")
### END WEEK 9 TEST FOR MODEL 1
