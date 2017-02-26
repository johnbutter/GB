
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
#library(rattle)
library(rpart.plot)
library(Metrics)
#library(RColorBrewer)


cat("Reading CSV files and loading frames...\n");
completeData <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/train.csv", header = T, stringsAsFactors = T))
client <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/cliente_tabla.csv")
product <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/producto_tabla.csv")
town <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/town_state.csv")
brand <- as.data.table(fread("~/Desktop/Data/Kaggle/GrupoBimbo/brand.csv", header = T, stringsAsFactors = T))
  setkey(brand, by="PID")
completeTest <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/test.csv", header = T, stringsAsFactors = T))

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
# actual demand between 1 and 1000 covers over 60M records
clients<-subset(completeData, Demanda_uni_equil<1000&Demanda_uni_equil>0)
# weeks 3-8 (consider holding out week 9 ~7M records to compare results)
testweek9<-subset(clients, Semana==9)
clients<-subset(clients, Semana<9 &Semana>0)
week8<-subset(clients, Semana==8)
week7<-subset(clients, Semana==7)
week6<-subset(clients, Semana==6)

# Random sample subset for training set to desired %
cat("Number of rows:",nrow(clients),"\n")
train<- clients %>% sample_frac(0.01) #settled on .1-10% for the model build
test<-completeTest

############################
# Market Basket Set Analysis
############################
library("dplyr")
# Can use any week or combined weeks...
marketbasket<-data.table(testweek9)
marketbasket<-data.table(distinct(marketbasket, Producto_ID, Cliente_ID))
i <- split(marketbasket$Producto_ID, marketbasket$Cliente_ID)
sapply(head(i), function(x) sum(duplicated(x))) # check for duplicates in basket
library("arules")
library("arulesViz")
txn <- as(i, "transactions")
basket_rules <- apriori(txn, parameter = list(sup = 0.1, conf = 0.5, target="rules", maxlen=5))
itemFrequencyPlot(txn, topN = 25)
plot(basket_rules)
inspect(head(sort(basket_rules, by= "lift"), 20))


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
# Use Next 3 Lines if we want to use Week9 instead of Kaggle TEST file
#hier_test<-week_nine
#hier_test$Demanda_uni_equil<-NULL
#kaggle==FALSE

# sort by key
setkey(hier_train, Producto_ID, Cliente_ID, Agencia_ID)

#table of overall median (+modification)
medianp <- hier_train[, median(Demanda_uni_equil)]

#table of product -client -agent mean 
mean_Prod_Client_Agent <- hier_train[, exp(mean(log(Demanda_uni_equil+1)))*0.75+0.75,by = .(Producto_ID,Cliente_ID,Agencia_ID)]
setnames(mean_Prod_Client_Agent,"V1","PCA")

#table of product -route mean
mean_Prod_Ruta <- hier_train[, exp(mean(log(Demanda_uni_equil+1)))*0.8+0.1,by = .(Producto_ID, Ruta_SAK)]
setnames(mean_Prod_Ruta,"V1","PR")

#table of product overall mean 
mean_Prod <- hier_train[, exp(mean(log(Demanda_uni_equil+1)))*1.15+0.25, by = .(Producto_ID)]
setnames(mean_Prod,"V1","P")

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

# If doing Kaggle Submission sort by ID else others
if (kaggle ==TRUE) {
  setkey(hold, id)
}  else {
    setkey(week_nine, Producto_ID, Cliente_ID, Agencia_ID, Ruta_SAK)
    setkey(hold,Producto_ID, Cliente_ID, Agencia_ID, Ruta_SAK )
    xr<-rmsle(week_nine$Demanda_uni_equil, hold$Pred)
    at<-sum(week_nine$Demanda_uni_equil)
    pt<-sum(hold$Pred)
    cat("Actual Demand Total:", at,"\n")
    cat("Predicted Demand Total:", pt,"\n")
    cat("RMSLE:", xr,"\n")
}


# drop all but id, Median_Pred from hold use to ensemble later
####### THIS IS MEDIAN PREDICTION SET ###############
hierpred <- data.frame(id=hold$id, mpred=hold$Pred)


################
# MODEL 2
# Boosting Model
################
keep.id<-test$id
train$id <- NULL;
test$id <- NULL;

# Branch train and test to be used by different model techniques
# For OHE of categorical variables
ohetrain<-train
ohetest<-test

cat("Creating new features...\n");
## Must have at least 1 numeric field in the matrix for xgboost to work
ohetrain$Canal_ID<-1.0 
ohetest$Canal_ID<-1.0

###################################
## Working on new section for additional features
## Brand
## Basket
## add means to the records of train/test then boost
###################################

# add column PCA, PR, P for mean by product client agent to TRAIN and TEST
ohetrain <- merge(ohetrain, mean_Prod_Client_Agent, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"), all.x = TRUE)
ohetrain <- merge(ohetrain, mean_Prod_Ruta, by = c("Producto_ID", "Ruta_SAK"), all.x = TRUE)
ohetrain <- merge(ohetrain, mean_Prod, by = "Producto_ID", all.x = TRUE)

ohetest <- merge(ohetest, mean_Prod_Client_Agent, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"), all.x = TRUE)
ohetest <- merge(ohetest, mean_Prod_Ruta, by = c("Producto_ID", "Ruta_SAK"), all.x = TRUE)
ohetest <- merge(ohetest, mean_Prod, by = "Producto_ID", all.x = TRUE)


cat("Treating features...\n");
## impute train / test here
# if TRAIN/TEST are missing values in PCA, PR, or P, use P or mean (7)
ohetrain$PCA[is.na(ohetrain$PCA)] <- ohetrain$P[is.na(ohetrain$PCA)]
ohetrain$PR[is.na(ohetrain$PR)] <- ohetrain$P[is.na(ohetrain$PR)]
ohetrain$P[is.na(ohetrain$P)] <- 7


ohetest$PCA[is.na(ohetest$PCA)] <- ohetest$P[is.na(ohetest$PCA)]
ohetest$PR[is.na(ohetest$PR)] <- ohetest$P[is.na(ohetest$PR)]
ohetest$P[is.na(ohetest$P)] <- 7


ohetrain$Demanda_uni_equil[ohetrain$Demanda_uni_equil>1000] = 1000
ohetrain$Demanda_uni_equil[is.na(ohetrain$Demanda_uni_equil)]<-median(ohetrain$Demanda_uni_equil)

#Retain Log Transformed Target Demand
target = log(ohetrain$Demanda_uni_equil+1);

## drop from train and test heree
cat("Dropping features...\n");
ohetrain$Cliente_ID<-NULL
ohetrain$Producto_ID<-NULL
ohetrain$Agencia_ID<-NULL
ohetrain$Semana<-NULL
ohetest$Cliente_ID<-NULL
ohetest$Producto_ID<-NULL
ohetest$Agencia_ID<-NULL
ohetest$Semana<-NULL



#one hot encoding of matrix to categorize integer dimensions
sparse_matrix_train <- sparse.model.matrix( ~ Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID -1, data = ohetrain)
sparse_matrix_test <- sparse.model.matrix( ~ Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID -1, data = ohetest)

# combine PCA, PR, P columns back to OHE matrix
mean_cols <- subset(ohetrain,select= c(PCA,PR,P))
mean_cols<-data.matrix(mean_cols)
sparse_matrix_train<-cbind2(sparse_matrix_train,mean_cols)

# ohe test combine
mean_cols <- subset(ohetest,select= c(PCA,PR,P))
mean_cols<-data.matrix(mean_cols)
sparse_matrix_test<-cbind2(sparse_matrix_test,mean_cols)


# build data matrix with ohe encoded training file
cat("Creating data.matrix...\n");
preds <- rep(0,nrow(test));
trainM<-data.matrix(sparse_matrix_train, rownames.force = NA); #playing around with OHE
cat("Creating DMatrix for xgboost...\n");
dtrain <- xgb.DMatrix(data=trainM, label=target, missing = NaN);
watchlist <- list(trainM=dtrain);

################# Iterate 50-100 Times through cross validation to find best 
################# parameter list. Modified from to linear regression #######################################

nloops<-10
best_param = list()
best_seednumber = 1969
best_error = Inf
best_error_index = 0
library(mlr)
for (iter in 1:nloops) {
  param <- list(objective = "reg:linear",
                max_depth = sample(6:10, 1), #10
                eta = runif(1, .01, .3), #0.2784
                gamma = runif(1, 0.0, 0.2), #0.134885
                subsample = runif(1, .6, .9), #0.7742556
                colsample_bytree = runif(1, .5, .8), #0.5917445
                min_child_weight = sample(1:40, 1), #9
                max_delta_step = sample(1:10, 1) #4
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, watchlist = watchlist, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early.stop.round=8, maximize=FALSE)
  
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
preds <- round(predict(clf, testM, missing=NA),10);
preds <- exp(preds)-1

#Fix Negatives with Median
preds[preds<0] = 3

#Truncate anything larger than 1000
preds[preds>1000] = 1000

# build data table with id, xgbpredicted demand
# this is the XGBoost model output
xgbpred <- data.frame(id=keep.id, Demanda_uni_equil=preds);



############################
# BUILD ENSEMBLE
# Start Ensembling Procedure
############################

# merge xgboost and medianpred by id
ensemble<-merge(xgbpred, hierpred, by = c("id"), all.x = TRUE)

#calculate new demand prediction (1/2 * the difference between XGB and Median subtracted from Median)
ensemble$NewDemand<-ensemble$mpred-((ensemble$mpred-ensemble$Demanda_uni_equil)/2)

#set column names for the submission files and drop unneeded fields
setnames(hierpred,"mpred","Demanda_uni_equil")
ensemble$mpred<-NULL
ensemble$Demanda_uni_equil<-NULL
setnames(ensemble,"NewDemand","Demanda_uni_equil")

############################################################
# write out files for KAggle
cat("Saving the submission files\n");
write.csv(xgbpred, "~/Desktop/Data/Kaggle/GrupoBimbo/basicBOOST.csv", row.names = F)
write.csv(hierpred, "~/Desktop/Data/Kaggle/GrupoBimbo/basicHIER.csv", row.names = F)
write.csv(ensemble, "~/Desktop/Data/Kaggle/GrupoBimbo/ENSEMBLESTACK.csv", row.names = F)
########## Completed with KAGGLE evaluation #################



##########################################################
######### Code to Use completed training on Week 9 Holdout
######### Compare results using RMSLE to actual and 
######### other evaluations such as total cost, etc.
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

#Fix Negatives with Median
preds[preds<0] = 3

#Truncate anything larger than 1000
preds[preds>1000] = 1000

# use rmsle to compare actual from week 9 to predicted
# this should be close or better than the xgb results for KAGGLE
# to confirm reasonableness of model at predicting for any week.
xr<-rmsle(actual9$Demanda_uni_equil, preds)
at<-sum(actual9$Demanda_uni_equil)
pt<-sum(preds)
cat("Actual Demand Total:", at,"\n")
cat("Predicted Demand Total:", pt,"\n")
cat("RMSLE:", xr,"\n")

