rm(list=ls()) # clear the environment

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
lambda_seq_global <- 10^seq(2, -2, by = -.1)

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq_global) # alpha = 0 for ridge regression
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
data_nodate_global <- data.complete.columns[, -1]

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
    print("Producing pseudo-cross-sectional data...")
    if (data_has_date_column) {
        data_nodate <- data[, -1]
    } else {
        data_nodate <- data
    }

    X_primitive <- data_nodate[, X_variables]
    Y_primitive <- data_nodate[, Y_variable]

    X_cross_sectional <- data.frame(nrow=1:(nrow(X_primitive)-p)) # an empty dataframe, but not NULL since cbind doesn't like NULL for dataframes
    cols <- NULL
    for (lag in 1:p) {
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
p_max_global <- 12 # i.e. 3 years

## From there, let us define our train/test strategy
# We will test our model on the end of the data
test_size_global <- 0.1
# # test indices regarding the initial dataset
# test_indices <- ceiling((1-test_size) * nrow(data_nodate)):nrow(data_nodate)

test_amount_global <- ceiling(test_size_global * nrow(data_nodate_global)) # number of observations in the test set, no matter the lag parameter

# # Difficulty here : the cross-sectional data is produced on-the-fly for each p
# Y_cross_sectional_test <- Y_cross_sectional[test_indices]
# X_cross_sectional_test <- X_cross_sectional[test_indices, ]

# Then, let us get our global training set well separated from the test set
# The earliest possible Y in X_t is Y_{t-max_p},
# so, with t_0 the first date in the test set,
# the latest totally separate date to include in the training set is t_0 - max_p - 1
# train_indices <- 1:(test_indices[1] - max_p - 1)

# Y_cross_sectional_train <- Y_cross_sectional[train_indices]
# X_cross_sectional_train <- X_cross_sectional[train_indices, ]

## Now, let us define our cross-validation strategy
# We choose a LOOCV strategy
# For each t, we will train on all the data except indices (t-max_p):(t+max_p)


# A function to get the train and validation sets for a given date t and number of lags p
#   X: matrix of covariates
#   Y: observations of the dependent variable
#   t: studied date
#   p: number of lags to consider ; p >= 1
get_train_validation_sets <- function(X, Y, t, p) {
    print(paste("Getting train and validation sets for date t = ", t, "...", sep=""))
    not_train_indices <- max(0,t-p):min(t+p, length(Y)) # don't forget the beginning and end bounds
    return(list(X[-not_train_indices, ], Y[-not_train_indices], X[t, ], Y[t]))
}




### Now, let's adapt the previous scheme to the specific time-series structure
# using the same variables but with their lagged values

# ## One step of cross-validation
# ### A function doing a whole step of cross-validation for parameter p
# data: dataframe containing the original data
# Y_index: column index of the dependent variable
# X_indices: column indices of the covariates
# p: number of lags to consider ; p >= 1
# test_amount: number of observations in the test set (no matter the lag parameter)
# lambda_seq: sequence of lambda values to consider (for the penalized linear model)
#
# This function will :
# - produce pseudo-cross-sectional data adapted to the chosen lag parameter p
# - determine the related global training set
# - for each date t in the training set: determine the related train/validation sets, train the model, and compute the mean squared error
# - return the global mean squared error
cross_validation_step_linear <- function(data=data_nodate_global, Y_index=1, X_indices=1:ncol(data_nodate_global), p=1, test_amount=test_amount_global, lambda_seq=lambda_seq_global, validation_proportion=0.05) {
    print(paste("Cross-validation -- step p = ", p, " : starting", sep=""))
    # Produce pseudo-cross-sectional data, specially adapted to lag p
    res <- produce_cross_sectional_data(data, p, Y_index, X_indices)
    X_cross_sectional <- res[[1]]
    Y_cross_sectional <- res[[2]]

    # Determine the global training set
    # The first date in the corresponding test set is nrow(Y_cross_sectional) - test_amount + 1
    # The earliest Y in X_{t_0} is Y_{t_0-p}, so dates before t_0 - p - 1 are safe
    train_indices <- 1:(length(Y_cross_sectional) - test_amount - p)

    X_cross_sectional_train <- X_cross_sectional[train_indices, ]
    Y_cross_sectional_train <- Y_cross_sectional[train_indices]

    # For each date t in the training set: determine the related train/validation sets, train the model, and compute the mean squared error
    mse <- 0
    validation_indices <- sample(train_indices, floor(validation_proportion * length(train_indices)))
    for (i_t in 1:length(validation_indices)) {
        t <- validation_indices[i_t]
        print(paste("Cross-validation -- step p = ", p, " : validation date t = ", t, " i.e. substep ", i_t, " / ", length(validation_indices), sep=""))
        # Get the train and validation sets for date t (NB : the validation set is a single observation)
        res <- get_train_validation_sets(X_cross_sectional, Y_cross_sectional, t, p)
        X_train <- res[[1]]
        Y_train <- res[[2]]
        X_validation <- res[[3]]
        Y_validation <- res[[4]]

        X_train <- data.matrix(X_train)
        X_validation <- data.matrix(X_validation)

        # Train the model
        print("training model...")
        ridge_cv <- cv.glmnet(X_train, Y_train, alpha = 0, lambda = lambda_seq) # alpha = 0 for ridge regression
        ## We let the provided model run its own cross-validation protocol, even though it is not a-priori adapted to our time-series structure
        # Best lambda value
        best_lambda <- ridge_cv$lambda.min
        #Choosing the best model
        best_fit <- ridge_cv$glmnet.fit
        print("model trained")

        # Predict on the validation observation
        pred <- predict(best_fit, newx = X_validation)

        # Actualize the mean squared error
        mse <- mse + (pred - Y_validation)^2
    }
    mse <- mse / length(validation_indices)
    return(mse)
}

# # The computation of wanted column names is no longer necessary as we select the columns before producing the pseudo-cross-sectional data
# v_noNA_lags <- paste(v_noNA, "_lag1", sep="")
# for (lag in 2:parameter_p) {
#     v_noNA_lags <- c(v_noNA_lags, paste(v_noNA, "_lag", lag, sep=""))
# }
# x_penalized_linear_2 <- X_cross_sectional[, v_noNA_lags]
# y_penalized_linear_2 <- Y_cross_sectional



# ## Whole training and CV (linear model)
# ### This function proceeds to the whole cross-validation process and produces the best model
# data: dataframe containing the original data. (It is assumed not to contain date as first column)
# Y_index: column index of the dependent variable
# X_indices: column indices of the covariates
# p_max: maximum number of lags to consider, across the cross-validation process (p will vary from 1 to p_max)
# test_amount: number of observations in the test set (no matter the lag parameter)
# lambda_seq: sequence of lambda values to consider (for the penalized linear model)
#
# This function will :
# - execute cross-validation steps for each p from 1 to p_max, and store the mean squared errors of each step
# - fit the model on the complete training set, for the best p
# - test the model on the test set and compute the mean squared error
# - return the best model itself, its mean squared error (on the test set), the best p, the mean squared errors for each p
whole_training_linear <- function(data=data_nodate_global, Y_index=1, X_indices=1:ncol(data_nodate_global), p_max=p_max_global, test_amount=test_amount_global, lambda_seq=lambda_seq_global, validation_proportion=0.05) {
    # Execute cross-validation steps for each p from 1 to p_max, and store the mean squared errors of each step
    mse_cv <- rep(0, p_max)
    for (p in 1:p_max) {
        mse_cv[p] <- cross_validation_step_linear(data, Y_index, X_indices, p, test_amount, lambda_seq, validation_proportion)
        print(paste("Cross-validation -- step p = ", p, " : done", sep=""))
        print(paste("---> mean squared error (p = ", p, ") : ", mse_cv[p], sep=""))
        print("-----------------------------------------------")
    }

    print("")

    # Determine the best p
    best_p <- which.min(mse_cv)
    print(paste("Cross-validation done. Best p = ", best_p, sep=""))

    # Produce pseudo-cross-sectional data, specially adapted to lag p
    res <- produce_cross_sectional_data(data, best_p, Y_index, X_indices)
    X_cross_sectional <- res[[1]]
    Y_cross_sectional <- res[[2]]

    # Determine the training set (there will be no validation sets this time)
    # The first date in the test set is length(Y_cross_sectional) - test_amount + 1
    # The earliest Y in X_{t_0} is Y_{t_0-best_p}, so dates before t_0 - best_p - 1 are safe
    train_indices <- 1:(length(Y_cross_sectional) - test_amount - best_p)

    X_cross_sectional_train <- X_cross_sectional[train_indices, ]
    Y_cross_sectional_train <- Y_cross_sectional[train_indices]

    X_cross_sectional_train <- data.matrix(X_cross_sectional_train)

    # Train the model
    print("training (final) model...")
    ridge_cv <- cv.glmnet(X_cross_sectional_train, Y_cross_sectional_train, alpha = 0, lambda = lambda_seq) # alpha = 0 for ridge regression
    ## We let the provided model run its own cross-validation protocol, even though it is not a-priori adapted to our time-series structure
    # Best lambda value
    best_lambda <- ridge_cv$lambda.min
    #Choosing the best model
    best_fit <- ridge_cv$glmnet.fit
    print("(final) model trained")

    # Determine the test set
    test_indices <- (length(Y_cross_sectional) - test_amount + 1):length(Y_cross_sectional)
    X_test <- X_cross_sectional[test_indices, ]
    Y_test <- Y_cross_sectional[test_indices]

    X_test <- data.matrix(X_test)

    # Predict on the test set
    pred <- predict(best_fit, newx = X_test)

    # MSE
    mse_global <- mean((pred - Y_test)^2)

    return(list(best_fit, mse_global, best_p, mse_cv))
}

res_whole_training_linear <- whole_training_linear()
best_fit_linear <- res_whole_training_linear[[1]]
mse_global_linear <- res_whole_training_linear[[2]]
best_p_linear <- res_whole_training_linear[[3]]
mse_cv_linear <- res_whole_training_linear[[4]]

print(paste("Mean squared error (on the test set) of the best model : ", mse_global_linear, sep=""))
print(paste("Best p : ", best_p_linear, sep=""))
print(paste("Mean squared errors (validation process) for each p : ", paste(mse_cv_linear, sep="", collapse=", "), sep=""))



### Neural Networks
library(tensorflow)
library(keras)

# given the randomness of the train/test split, we can reuse it
