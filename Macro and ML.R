library(dplyr) # data manipulation library


#FRED data
file <- "data/2023-12 - quarter.csv" 
data.with.definitions <- read.csv(file = file)
data <- data.with.definitions[3:nrow(data.with.definitions), ] # we drop the first two rows which are not data
rownames(data) <- NULL # we reset the indices of the rows

colnames(data)

x <- data$sasdate
# we drop the rows which have no date
data <- data[(x!="Transform:" & nchar(x)>2),]
t<-data[,1]

print(paste("Deleted", length(t) - length(x), "row(s) because of absent date"), quote=FALSE)



# Variable names
series <- colnames(data[,2:length(data)])

#Selection of variables of interest : à faire


#drop na
data.complete<-na.omit(data)
print(paste("Deleted", length(t) - length(data.complete[,1]), "row(s) because of NAs"), quote=FALSE)
# Considering only fully complete rows leads to deleting 189 rows out of 259.
# Thus it seems better to first choose a limited set of variables to consider.

# Let us see what variables are present in the dataset at each date,
# and see if it seems reasonable to consider only those variables.
data.complete.columns <- data[, colSums(is.na(data)) == 0]
print(paste("Deleted", length(data) - length(data.complete.columns), "variable(s) because of NAs"), quote=FALSE)



####### ATTENTION au caractère time-series des données
####### il ne faut pas oublier de prendre en compte les dates précédentes
####### et pas seulement les variables contemporaines


### Penalized linear model : Ridge Regression

#install.packages("glmnet")
library(ggplot2)
library(glmnet)

set.seed(123)


## As a first step and benchmark, we first reason in a "cross-sectional" way.
## However, we don't forget that the main aim of our work is forecasting
## So we'll try to predict GDP using the value of other variables at date t-1

# we exclude the last row of x so as to induce a shift between dependent and independent variables



# x <- data.matrix(data.complete.columns[-nrow(data.complete.columns),c("GPDIC1","pcecc96","INDPRO","GCEC1","EXPGSC1","IMPGSC1","CPIAUCSL",
#                                  "FEDFUNDS","UNRATE","S_P_500","NASDAQCOM","EXUSEU","EXJPUS","M1REAL", "M2REAL",
#                                  "GS10","BUSLOANSx","consumerx")])
# we exclude the first row of y for the same reason
y <- data.complete.columns[-1,"GDPC1"]
# We reset the indices of y
rownames(y) <- NULL

# x is not defined because some of the variables we want to use have been deleted, because they contained NAs
# let's see which variables are still available
v_noNA_all <- colnames(data.complete.columns)
# Don't forget to include GDPC1 in the list of covariates: it would be too bad not to use the past values of it to predict its future values
v_noNA <- intersect(v_noNA_all, c("GDPC1", "GPDIC1","pcecc96","INDPRO","GCEC1","EXPGSC1","IMPGSC1","CPIAUCSL",
            "FEDFUNDS","UNRATE","S_P_500","NASDAQCOM","EXUSEU","EXJPUS","M1REAL", "M2REAL",
            "GS10","BUSLOANSx","consumerx"))
print(paste("Variables available: ", paste(v_noNA, sep=" ", collapse = ", "), sep=""), quote=FALSE)

x <- data.matrix(data.complete.columns[-nrow(data.complete.columns),v_noNA])
# note : we still exclude the last row of x so as to induce a shift between dependent and independent variables
# it is useless to reset the indices of x since we only removed the last row
# rownames(x) <- NULL


# Split data between train and test samples
# Remark: the fact that each y_t corresponds to a single date x_{t-1} allows to
#         split randomly without worrying about the autocorrelation of the data

train_indices <- sample(1:nrow(x), 0.7 * nrow(x))

x_train <- x[train_indices,]
y_train <- y[train_indices]

x_test <- x[-train_indices,]
y_test <- y[-train_indices]


# Normalizing data
x_train <- scale(x_train)
x_test <- scale(x_test)

# Setting the range of lambda values
lambda_seq <- 10^seq(2, -2, by = -.1)

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq) # alpha = 0 for ridge regression
# Best lambda value
best_lambda <- ridge_cv$lambda.min


#Choosing the best model
best_fit <- ridge_cv$glmnet.fit


# Predict on the test sample
test_predictions <- predict(best_fit, newx = x_test)

# RMSE
rmse <- sqrt(mean((test_predictions - y_test)^2))

# R²
SST <- sum((y_test - mean(y_test))^2)
SSR <- sum((test_predictions - mean(y_test))^2)
rsquared <- SSR / SST

print(paste("RMSE:", rmse))
print(paste("R-squared:", rsquared))




## Now, let's take more deeply into account the time-series aspect of our data

# To keep using usual syntax of ML algorithms,
# where X and Y are "contemporaneous",
# we need to transform our data, especially the "independent" variables,
# so that X features the lagged values

# First, let's remove the date column
data_nodate <- data[, -1]

# # we keep in X the first column, which corresponds to y, in order to use its past values
# X_primitive <- data_nodate
# Y_primitive <- data_nodate[, 1] # GDPC1


# # For the moment, let's set a parameter p to account for the number of lags that we consider.
# # Then, we can improve the process and make a function from this.
# parameter_p <- 4

# # We bind the lagged values of the explanatory variables
# X_cross_sectional <- X_primitive[parameter_p:(nrow(X_primitive)-1), ] # here lag == 1
# cols <- paste(colnames(X_primitive), "_lag1", sep="")
# for (lag in 2:parameter_p) {
#     X_cross_sectional <- cbind(X_cross_sectional, X_primitive[(parameter_p+1-lag):(nrow(X_primitive)-lag), ])
#     cols <- c(cols, paste(colnames(X_primitive), "_lag", lag, sep=""))
# }
# colnames(X_cross_sectional) <- cols

# Y_cross_sectional <- Y_primitive[(parameter_p+1):length(Y_primitive)]



# A function to produce pseudo-cross-sectional data from time-series data
    # data: dataframe containing the data
    # p: number of lags to consider ; p >= 1
    # Y_variable: index of the column of data containing the dependent variable
    # X_variables: indices of the columns of data containing the independent variables. Don't forget to include Y if you want to use its past values
    # data_has_date_column: whether the first column of data contains the dates or not
produce_cross_sectional_data <- function(data, p, Y_variable=1, X_variables=1:ncol(data), data_has_date_column=FALSE) {
    if (data_has_date_column) {
        data_nodate <- data[, -1]
    } else {
        data_nodate <- data
    }

    X_primitive <- data_nodate[, X_variables]
    Y_primitive <- data_nodate[, Y_variable]

    X_cross_sectional <- X_primitive[p:(nrow(X_primitive)-1), ] # here lag == 1
    cols <- paste(colnames(X_primitive), "_lag1", sep="")
    for (lag in 2:p) {
        X_cross_sectional <- cbind(X_cross_sectional, X_primitive[(p+1-lag):(nrow(X_primitive)-lag), ])
        cols <- c(cols, paste(colnames(X_primitive), "_lag", lag, sep=""))
    }
    colnames(X_cross_sectional) <- cols

    Y_cross_sectional <- Y_primitive[(p+1):length(Y_primitive)]

    return(list(X_cross_sectional, Y_cross_sectional))
}



### We are now coming to 2 sensitive points :
## - the train/test split with time-series data
## - the choice of a strategy for cross-validation, involving train/validation splits

# Let us fix a maximum lag parameter
# keeping in mind that our data is quarterly
max_p <- 12 # i.e. 3 years

## From there, let us define our train/test strategy
# We will test our model on the end of the data
test_size <- 0.1
test_indices <- ceiling((1-test_size) * nrow(data_nodate)):nrow(data_nodate)

## Difficulty here : the cross-sectional data is produced on-the-fly for each p
Y_cross_sectional_test <- Y_cross_sectional[test_indices]
X_cross_sectional_test <- X_cross_sectional[test_indices, ]

# Then, let us get our global training set well separated from the test set
# The earliest possible Y in X_t is Y_{t-max_p},
# so, with t_0 the first date in the test set,
# the latest totally separate date to include in the training set is t_0 - max_p - 1
train_indices <- 1:(test_indices[1] - max_p - 1)

Y_cross_sectional_train <- Y_cross_sectional[train_indices]
X_cross_sectional_train <- X_cross_sectional[train_indices, ]

## Now, let us define our cross-validation strategy
# We choose a LOOCV strategy
# For each t, we will train on all the data except indices (t-max_p):(t+max_p)


# A function to get the train and validation sets for a given t
#   X: matrix of covariates
#   Y: observations of the dependent variable
#   t: studied date
#   max_p: maximum lag for the whole CV process
get_train_validation_sets <- function(X, Y, t, max_p=max_p) {
    not_train_indices <- max(0,t-max_p):min(t+max_p, nrow(Y)) # don't forget the beginning and end bounds
    return(list(X[-not_train_indices, ], Y[-not_train_indices], X[t, ], Y[t]))
}





### Now, let's adapt the previous scheme to the specific time-series structure
# using the same variables but with their lagged values

v_noNA_lags <- paste(v_noNA, "_lag1", sep="")
for (lag in 2:parameter_p) {
    v_noNA_lags <- c(v_noNA_lags, paste(v_noNA, "_lag", lag, sep=""))
}


x_penalized_linear_2 <- X_cross_sectional[, v_noNA_lags]
y_penalized_linear_2 <- Y_cross_sectional








### Neural Networks
library(tensorflow)
library(keras)

# given the randomness of the train/test split, we can reuse it
