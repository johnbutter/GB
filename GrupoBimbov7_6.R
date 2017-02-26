
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

#Custom Error Function For the Competition
evalerror = function (preds, dtrain){
  ademand = getinfo(dtrain, "label")
  pdemand = getinfo(preds, "Demanda_uni_equil")
  terms_to_sum = for(i in 1:length(pdemand) [(log(pdemand[i] + 1) - log(max(0,ademand[i]) + 1)) ^ 2.0])
    err = as.numberic ((sum(terms_to_sum) * (1.0/length(preds))) ** 0.5)
    return (list(metric = "Error:", value = err))
}

cat("Reading CSV file...\n");
completeData <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/train.csv", header = T, stringsAsFactors = T))
client <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/cliente_tabla.csv")
product <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/producto_tabla.csv")
town <- read_csv("~/Desktop/Data/Kaggle/GrupoBimbo/town_state.csv")

# EDA of completeDataSet (random fractions)
ggplot(completeData %>% sample_frac(0.005))+
  geom_histogram(aes(x=Semana), color="black", fill="cyan", alpha=0.5)+
  scale_x_continuous(breaks=1:10)+
  scale_y_continuous(name="Client / Product deliveries")+
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
clients.routes <- completeData %>%
  group_by(Cliente_ID) %>%
  summarise(n_routes = n_distinct(Ruta_SAK))

ggplot(clients.routes)+
  geom_histogram(aes(x=n_routes), fill="cyan", color="black", alpha="0.5", binwidth=1)+
  scale_x_continuous(name="Number of clients")+
  scale_y_continuous(name="Number of routes", labels=function(x)paste(x/1000, "k"))+
  theme_bw()


#Start Model Iteration Here
#############
# Here is where we should set up and select which sample we test - default is random
# Change The Sampling Fraction to adjust the size of the training data set

cat("Splitting data...\n");
# random sample full set....
clients <- completeData %>% sample_frac(0.01)

############### PICK ONE OR SERIES OF FOLLOWING ####################
#specific clients
clients<-subset(completeData, Cliente_ID==c('15766','1265976','1451681','2233526'))
# units between 1 and 15 covers over 52M records
clients<-subset(completeData, Demanda_uni_equil<15 &Demanda_uni_equil>1)
# weeks 3-8 (consider holding out week 9 ~7M records to compare results)
testweek9<-subset(clients, Semana==9)
clients<-subset(clients, Semana<9 &Semana>0)

# Random sample subset for training set to desired %
cat("Number of rows:",nrow(clients),"\n")
train<- clients %>% sample_frac(0.1) #settled on 10% for the model build

################
# Ensemble 1
# Boosting Model
################
test <- as.data.frame(fread("~/Desktop/Data/Kaggle/GrupoBimbo/test.csv", header = T, stringsAsFactors = T))
keep.id<-test$id

train$id <- NULL;
test$id <- NULL;

cat("Creating new features...\n");
## add new features to train and test here



cat("Treating features...\n");
## impute train and test here
# if not using OHE then you should execute these 2 next lines
train$Demanda_uni_equil[train$Demanda_uni_equil>100] = 100
train$Producto_ID<-as.factor(train$Producto_ID)
test$Producto_ID<-as.factor(test$Producto_ID)

#if going to use OHE then must do the following 3 lines
train$Canal_ID<-1.0 
test$Canal_ID<-1.0
target = train$Demanda_uni_equil;


#one hot encoding
sparse_matrix_train <- sparse.model.matrix(Canal_ID ~ Agencia_ID +Ruta_SAK +Producto_ID +Cliente_ID -1, data = train)


cat("Dropping features...\n");
## drop from train and test heree
train$Demanda_uni_equil <- NULL;
train$Venta_hoy<-NULL
train$Venta_uni_hoy<-NULL
train$Dev_proxima<-NULL
train$Dev_uni_proxima<-NULL
# dropped these next...
train$Semana<-NULL
test$Semana<-NULL
train$Agencia_ID<-NULL
test$Agencia_ID<-NULL

dim(train)
dim(test)




cat("Creating data.matrix...\n");
preds <- rep(0,nrow(test));
trainM<-data.matrix(train, rownames.force = NA); # use this line if not using OHE
#trainM<-data.matrix(sparse_matrix_train, rownames.force = NA); #playing around with OHE
cat("Creating DMatrix for xgboost...\n");
dtrain <- xgb.DMatrix(data=trainM, label=target, missing = NaN);
watchlist <- list(trainM=dtrain);

# Need to change to get best results
set.seed(1969);


param <- list(  objective           = "count:poisson", 
                #booster             = "gblinear",
                #eval_metric         ="mae",
                eta                 = 0.035,
                max_depth           = 6,
                subsample           = 1.00,
                colsample_bytree    = 0.40
)

clf <- xgb.cv(  params              = param, 
                data                = dtrain, 
                nrounds             = 50, 
                verbose             = 1,
                watchlist           = watchlist,
                maximize            = FALSE,
                nfold               = 3,
                early.stop.round    = 10,
                print.every.n       = 1
);

bestRound <- which.min( as.matrix(clf)[,3] );
cat("Best round:", bestRound,"\n");
cat("Best result:",min(as.matrix(clf)[,3]),"\n");

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = bestRound, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

##########################################
# Test Variable Importance 
importance <- xgb.importance(feature_names = colnames(dtrain), model = clf)
xgb.plot.importance(importance)
print(importance)

testM <-data.matrix(test, rownames.force = NA);
preds <- round(predict(clf, testM),10);

#Fix Negatives with Median
preds[preds<0] = 3

#Truncate anything larger than 1000
preds[preds>1000] = 1000

submission <- data.frame(id=keep.id, Demanda_uni_equil=preds);
cat("Saving the submission file\n");
write.csv(submission, "~/Desktop/Data/Kaggle/GrupoBimbo/basicXGBoost.csv", row.names = F)

#check number of rows
cat("Number of rows:",nrow(submission),"\n")

