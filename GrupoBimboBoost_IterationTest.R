
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


#Custom Error Function For the Competition
myerror <- function(af, pf) {
  s<-0
  n<-0
  for (col in colnames(af)){
    a<-af[,col]
    p<-pf[,col]
    x<-!is.na(a)
    n<-n+sum(x)
    s<-s+sum((log1p(p(p[x])-log1p(a[x]))^2))
  }
    return(sqrt(s/n))
}

cat("Reading CSV files and loading frames...\n");
completeData <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/train.csv", header = T, stringsAsFactors = T))
client <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/cliente_tabla.csv")
product <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/producto_tabla.csv")
town <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/town_state.csv")
brand <- as.data.table(fread("~/Desktop/Data/Kaggle/GrupoBimbo/brand.csv", header = T, stringsAsFactors = T))
  setkey(brand, by="PID")
completeTest <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/test.csv", header = T, stringsAsFactors = T))

# Read in Training data
median_train <- fread("~/Desktop/Data/Kaggle/GrupoBimbo/train.csv",select = c('Agencia_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil')
                      , colClasses=c(Agencia_ID="numeric", Ruta_SAK="numeric", Cliente_ID="numeric",Producto_ID="numeric",Demanda_uni_equil="numeric"))

# Read in Test data 
median_test <- fread('~/Desktop/Data/Kaggle/GrupoBimbo/test.csv', 
                     select = c('id','Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Cliente_ID'),
                     colClasses=c(Producto_ID="numeric", Agencia_ID="numeric", Ruta_SAK="numeric", Cliente_ID="numeric"))


##########################################
### SAMPLING LOGIC HERE - MODIFY AS NEEDED
##########################################
# Here is where we should set up and select which sample we test - default is random
# Change The Sampling Fraction to adjust the size of the training data set
############### PICK ONE OR SERIES OF FOLLOWING ####################
cat("Splitting data...\n");
# units between 1 and 1000 covers over 60M records
clients<-subset(completeData, Demanda_uni_equil<1000&Demanda_uni_equil>0)
testweek9<-subset(clients, Semana==9)
week9<-subset(clients, Semana==9)
week8<-subset(clients, Semana==8)
week7<-subset(clients, Semana==7)
week6<-subset(clients, Semana==6)
# weeks 3-8 (consider holding out week 9 ~7M records to compare results)
clients<-subset(clients, Semana<9 &Semana>0)

# Use the holdout week variable which contains records in test 
# (weeks 10, 11 predictions) which we can use any known week's data
# will be used to establish baseline rmsle.  Logic is if we can predict
# week 9 data good, then it should be just as easy to do 10,11
holdout_week_test<-completeTest


# Random sample subset for training set to desired %
cat("Number of rows:",nrow(clients),"\n")
train<- clients %>% sample_frac(0.01) #settled on .1-10% for the model build
test<-completeTest

############################
# Market Basket Set Analysis
############################
library("dplyr")
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
# MODEL 1
# Create Median and Means Prediction Table
##########################################

#coefficients (consider using automated to tune that minimizes RMSLE)

# sort by key
setkey(median_train, Producto_ID, Agencia_ID, Cliente_ID)

#table of overall median (+modification)
medianp <- median_train[, median(Demanda_uni_equil)]

#table of product overall mean OMN
mean_Prod <- median_train[, exp(mean(log(Demanda_uni_equil+1)))*1.15+0.25, by = .(Producto_ID)]
setnames(mean_Prod,"V1","OMN")

#table of agent / product mean MNPA
mean_Agent_Prod <- median_train[, exp(mean(log(Demanda_uni_equil+1)))*0.75+0.75,by = .(Producto_ID,Agencia_ID,Cliente_ID)]
setnames(mean_Agent_Prod,"V1","MNPA")

#table of client / product median MEDPC
median_Client_Prod <- median_train[, exp(mean(log(Demanda_uni_equil+1)))*0.8+0.1,by = .(Producto_ID, Cliente_ID)]
setnames(median_Client_Prod,"V1","MEDPC")

# add column MNPA for mean by agent product
hold <- merge(median_test, mean_Agent_Prod, by = c("Producto_ID","Agencia_ID","Cliente_ID"), all.x = TRUE)

# add column MEDPC for median by client product
hold <- merge(hold, median_Client_Prod, by = c("Producto_ID", "Cliente_ID"), all.x = TRUE)

# add column OMN that contains mean by Product
hold <- merge(hold, mean_Prod, by = "Producto_ID", all.x = TRUE)

# Prediction intially set to be MNPA 
hold$Pred <- hold$MNPA

# if still NA use median product client
hold[is.na(Pred)]$Pred <- hold[is.na(Pred)]$MEDPC

# if still NA use overall mean
hold[is.na(Pred)]$Pred <- hold[is.na(Pred)]$OMN

# if still NA use overall median
hold[is.na(Pred)]$Pred <- medianp

# sort by id
setkey(hold, id)

# drop all but id, Median_Pred from hold use to ensemble later
####### THIS IS MEDIAN PREDICTION SET ###############
medianpred <- data.frame(id=hold$id, mpred=hold$Pred)



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

factortrain<-train
factortest<-test

cat("Creating new features...\n");
## add new features here
# Variables as Factors
factortrain$Producto_ID<-as.factor(factortrain$Producto_ID)
factortest$Producto_ID<-as.factor(factortest$Producto_ID)
factortrain$Cliente_ID<-as.factor(factortrain$Cliente_ID)
factortest$Cliente_ID<-as.factor(factortest$Cliente_ID)
factortrain$Ruta_SAK<-as.factor(factortrain$Ruta_SAK)
factortest$Ruta_SAK<-as.factor(factortest$Ruta_SAK)
factortrain$Canal_ID<-1.0 
factortest$Canal_ID<-1.0
ohetrain$Canal_ID<-1.0 
ohetest$Canal_ID<-1.0


cat("Treating features...\n");
## impute train and test here
factortrain$Demanda_uni_equil[factortrain$Demanda_uni_equil>100] = 100


#Retain Target Demand
factortrain$Demanda_uni_equil[is.na(factortrain$Demanda_uni_equil)]<-median(factortrain$Demanda_uni_equil)
target = log(factortrain$Demanda_uni_equil+1);


#one hot encoding
sparse_matrix_train <- sparse.model.matrix( ~ Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID -1, data = ohetrain)
sparse_matrix_test <- sparse.model.matrix( ~ Semana +Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID -1, data = ohetest)


cat("Dropping features...\n");
## drop from train and test heree
factortrain$Demanda_uni_equil <- NULL;
factortrain$Venta_hoy<-NULL
factortrain$Venta_uni_hoy<-NULL
factortrain$Dev_proxima<-NULL
factortrain$Dev_uni_proxima<-NULL
# dropped these next...
factortrain$Semana<-NULL
factortest$Semana<-NULL
factortrain$Agencia_ID<-NULL
factortest$Agencia_ID<-NULL

dim(factortrain)
dim(factortest)



cat("Creating data.matrix...\n");
preds <- rep(0,nrow(test));
#trainM<-data.matrix(factortrain, rownames.force = NA); # use this line if not using OHE
trainM<-data.matrix(sparse_matrix_train, rownames.force = NA); #playing around with OHE
cat("Creating DMatrix for xgboost...\n");
dtrain <- xgb.DMatrix(data=trainM, label=target, missing = NaN);
watchlist <- list(trainM=dtrain);

################# Iterate 50-100 Times through cross validation to find best 
################# parameter list. Modified from to linear regression #######################################
nloops<-1
best_param = list()
best_seednumber = 1969
best_error = Inf
best_error_index = 0
library(mlr)
for (iter in 1:nloops) {
  param <- list(objective = "reg:linear",
                #eval_metric = "mlogloss", use default until function works
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
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
preds <- round(predict(clf, testM),10);
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
ensemble<-merge(xgbpred, medianpred, by = c("id"), all.x = TRUE)

#calculate new demand prediction (1/2 * the difference between XGB and Median subtracted from Median)
ensemble$NewDemand<-ensemble$mpred-((ensemble$mpred-ensemble$Demanda_uni_equil)/2)

#set column names for the submission files and drop unneeded fields
setnames(medianpred,"mpred","Demanda_uni_equil")
ensemble$mpred<-NULL
ensemble$Demanda_uni_equil<-NULL
setnames(ensemble,"NewDemand","Demanda_uni_equil")


cat("Saving the submission files\n");
write.csv(xgbpred, "~/Desktop/Data/Kaggle/GrupoBimbo/basicBOOST.csv", row.names = F)
write.csv(medianpred, "~/Desktop/Data/Kaggle/GrupoBimbo/basicMEDIAN.csv", row.names = F)
write.csv(ensemble, "~/Desktop/Data/Kaggle/GrupoBimbo/ENSEMBLE.csv", row.names = F)

