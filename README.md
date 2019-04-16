# Machine Learning Project: Predicting Claim Severity for Allstate Insurance (Kaggle Competition)

The purpose of this project is to predict the continuous variable ‘loss’ which quantifies the severity of a claim to the insurance company. The information in this dataset has been completely stripped of its business context and simply contains both categorical and continuous attributes without any variable descriptions or hints. There are 116 categorical variables (named cat1-cat116) and 14 continuous attributes (named cont1-cont14).

The broad steps involved in solving this problem -

### Summarized the data by using dimension reduction techniques like PCA
I first performed correlation analysis on the subset of numeric variables (14 of them) in the dataset to exclude variables having more than 89% correlation. Second, I took a subset of variables having less than 9 category levels and performed a categorical principle component analysis on them to reduce their dimention by removing redundant variables. Third, I took all the above 9-category-level variables and converted them into numeric. This helped transform the entire dataset to having contnuous variables. 

### Performed feature engineering to transform the target variable to normal distribution 
I plotted the target variable (loss) to visualize its distribution and I saw that it was quite heavily skewed. To correct for that, I transformed the variable using log transformations to achieve a more symmetrical distribution. This helped improve the prediction too.

### Built a xgboost model in R to predict the proxy variable for claims severity to the insurance company 
I used an xgboost model in R to predict the continuous variable, loss. This model uses the GBM framework but works 10 times faster than the gradient boosting model It is good in many ways, to name a few-

1. Parallel Computing: It is enabled with parallel processing (using OpenMP); i.e., when you run xgboost, by default, it would use all the cores of your laptop/machine.
2. Regularization: I believe this is the biggest advantage of xgboost. GBM has no provision for regularization. Regularization is a technique used to avoid overfitting in linear and tree-based models.
3. Enabled Cross Validation: In R, we usually use external packages such as caret and mlr to obtain CV results. But, xgboost is enabled with internal CV function 
4. Tree Pruning: Unlike GBM, where tree pruning stops once a negative loss is encountered, XGBoost grows the tree upto max_depth and then prune backward until the improvement in loss function is below a threshold.

### Tuned the model parameters to obtain optimized predictive ability 
I first used the in-built xgb.cv function to find the best nrounds for the defaults parameter model. I then ran then trained the xgboost model with the default parameters and the best nrounds value. I used the predict funtion to predict the loss values given by the model
and then checked their accuracy against the actual values using the mean absolute error metric. 


