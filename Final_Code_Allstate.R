#libraries required for this code 

library(caret)
library(e1071)
library(PCAmixdata)
library(FactoMineR)
require(ggplot2)
library(factoextra)
library(data.table)
library(mlr)
library(caTools)
library(xgboost)
library(Matrix)
library(methods)
library(forecast)


# Reading the data
Train = read.csv("allstate_train.csv", header = T)
Test = read.csv("allstate_test.csv", header = T)

# Combining Train and Test datasets
all_data = rbind(Train[,1:131], Test)


### Data Pre-Processing
## Treating Continous variables

#Running correlation on continuous variables 
corr = cor(all_data[,118:131], method = c('pearson'))
corr.df = as.data.frame(as.table(corr))
subset(corr.df[order(-abs(corr.df$Freq)),], abs(Freq) > 0.8)

#Removing Varibles with correlations greater than 80% - cont 9, cont 10, cont12, cont13
cont_vars_all = all_data[,118:131]
cont_vars_2_all = cont_vars_all[,-c(9,10,12,13)]


## Treating Binary variables (PCA) 

#extracting the binary variables from all_data
binary_all <- all_data[,2:73]

# Converting the matrix to numeric 
binaryNum_all <- sapply(binary_all, as.numeric)

#performing pca on the numeric matrix of binary variables 
pca_all <- prcomp(binaryNum_all, scale=F)
Binary_vars_all = as.data.frame(pca_all$x[,1:25])


## Treating the 3-9 level categorical variables (Categorical PCA)

#extracting the 3-9 level categorocal variables from the whole dataset
categorical_all = all_data[74:117]

#creating two categories 
cat1_all = categorical_all[ ,1:16]
cat2_all = categorical_all[ ,c(17:26, 30)]

#running pca on both categories separately 
pca_cat1_all = PCAmix(X.quali = cat1_all , ndim = 30, rename.level = TRUE)
pca_cat2_all = PCAmix(X.quali = cat2_all , ndim = 50, rename.level = TRUE)

#extracting the scores to create a dataframe of the new variables from the step above
cat1_vars_all = data.frame(pca_cat1_all$scores)
cat2_vars_all = data.frame(pca_cat2_all$scores)

#changing the column names so as to not mix with cat1 variables 
colnames(cat2_vars_all) <- paste("X", colnames(cat2_vars_all), sep = "_")


## Treating the above 9 level categorical variables (converting into numeric) 

#extracting all more than 9 level variables and converting them into numeric values 
high_level_vars_all = categorical_all[,27:44]
high_level_vars_numeric_all <-  as.data.frame(sapply(high_level_vars_all, as.numeric))

#final dataset
final.dataset.all = cbind(Binary_vars_all, cat1_vars_all, cat2_vars_all, high_level_vars_numeric_all, cont_vars_2_all)

#writing the final dataset to your computer as a csv file 
write.csv(final.dataset.all, file = "Final_dataset_all.csv")

##now splitting the train and test data from the pre-processed final dataset
Test_data = tail(final.dataset.all, 100)
Train_data = final.dataset.all[1:188218,]

#convert data frame to data table
setDT(Train_data) 
setDT(Test_data)

#adding the target variable to the training dataset
target = Train$loss
target = as.data.frame(target)
Train_data = cbind(Train_data, target)

#splitting training into 2- train and valid
set.seed(505)
Train_data$split = sample.split(Train_data, SplitRatio = 0.8)
head(Train_data)
tr = subset(Train_data, Train_data$split == TRUE)
ts = subset(Train_data, Train_data$split == FALSE)

#subsetting the columns we need
tr = tr[,1:134]
ts = ts[,1:134]


#### Model preparation ####

#extracting the target variable from the training set 
tr_label = tr$target
ts_label = ts$target

#plotting the histogram to check the distribution
hist(tr_label)
hist(ts_label)

#Part of feature engineering- transforming the target variable to obtain normal-like distribution
tr_label_log = log(tr$target + 200)
ts_label_log = log(ts$target + 200)

#plotting the histogram to check the new distribution
hist(tr_label_log)
hist(ts_label_log)

#converting the train and test dataframes to a matrix
tr_matrix = as.matrix(tr, rownames.force = NA)
ts_matrix = as.matrix(ts, rownames.force = NA)

#converting the train and test dataframes to a sparse matrix
tr_sparse = as(tr_matrix, "sparseMatrix")
ts_sparse = as(ts_matrix, "sparseMatrix")

#For xgboost, using xgb.DMatrix to convert data table into a matrix is most recommended
dtrain = xgb.DMatrix(data = tr_sparse[,1:133], label = tr_label_log )
dtest = xgb.DMatrix(data = ts_sparse[,1:133], label = ts_label_log )


## preparation for xgboost model 

#defining default parameters
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, 
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, lambda=0, alpha=1)

#Using the inbuilt xgb.cv function to calculate the best nround for this model. 
#In addition, this function also returns CV error, which is an estimate of test error.

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)


# finding the best nrounds value
xgbcv$best_iteration

#training the model on default parameters with nrounds = 100 
xgb1 <- xgb.train (params = params, 
                   data = dtrain, 
                   nrounds = 100, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, 
                   early_stopping_rounds = 10, 
                   maximize = F , 
                   eval_metric = "mae")


#model prediction
xgbpred = predict(xgb1,dtest)

#taking the antilog of the predictions and subtracting 200 to get the error value in terms of the original target variable
preds = exp(xgbpred)-200

#checking the accuracy of the the model using MAE
accuracy(preds, ts_label)

#the MAE value given by this model is 1161.97 which is very close to the MAE of the second Kaggle winner of this competition
#Also, the MAE may vary a little if you are trying to run this code in your system 
#setting a different seed might also cause the MAE to vary a little

