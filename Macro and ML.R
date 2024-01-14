library(dplyr)


#FRED data
file <- "data/2023-12 - quarter.csv" 
data <- read.csv(file = file)

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


#Penalized linear model : Ridge Regression
#install.packages("glmnet")
library(ggplot2)
library(glmnet)

set.seed(123)

# Split data between train and test samples
train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]


x_train <- data.matrix(train_data[,c("GPDIC1","pcecc96","INDPRO","GCEC1","EXPGSC1","IMPGSC1","CPIAUCSL",
                                  "FEDFUNDS","UNRATE","S_P_500","NASDAQCOM","EXUSEU","EXJPUS","M1REAL", "M2REAL",
                                  "GS10","BUSLOANSx","consumerx")])

y_train <- train_data$GDPC1

x_test <- data.matrix(test_data[,c("GPDIC1","pcecc96","INDPRO","GCEC1","EXPGSC1","IMPGSC1","CPIAUCSL",
                                   "FEDFUNDS","UNRATE","S_P_500","NASDAQCOM","EXUSEU","EXJPUS","M1REAL", "M2REAL",
                                   "GS10","BUSLOANSx","consumerx")])
y_test <- test_data$GDPC1


# Normalizing data
x_train <- scale(x_train)
x_test <- scale(x_test)

# Setting the range of lambda values
lambda_seq <- 10^seq(2, -2, by = -.1)

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq)
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




