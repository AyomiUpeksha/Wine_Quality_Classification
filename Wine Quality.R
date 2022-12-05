library(tidyverse)
library(EnvStats) # For Outlier detection
library(rsample)    # for creating validation splits
library(magrittr)
library(GGally) # For generate pairewise correlation
#library(caret)       # for fitting KNN models
library(class)  #for fitting KNN models
library(rpart) # For fitting Decision Trees
library(randomForest) # For Fitting random forest
library(randomForestExplainer)
library(e1071) # For SVM


#### Import Data Set
df1 <- read.csv("C:/Users/Dell/Desktop/My Projects/Wine Quality Classification - R/wine.csv")

head(df1)

###### Data Wrangling
df1 <- as_tibble(df1)
str(df1)

# 1. Checking whether there are any missing value exists.
anyNA(df1)
#is.na(df1)

# 2. Changing the type of "quality" character to factor
df1$quality <- as.factor(df1$quality)

summary(df1)
# 3. Checking the outliers
p1 <- ggplot(df1, aes(x = df1$fixed.acidity)) +
  geom_boxplot()

p2 <- ggplot(df1, aes(x = df1$volatile.acidity)) +
  geom_boxplot() # Outlier exists

p3 <- ggplot(df1, aes(x = df1$citric.acid)) +
  geom_boxplot() # Outlier exists

p4 <- ggplot(df1, aes(x = df1$residual.sugar)) +
  geom_boxplot() 

p5 <- ggplot(df1, aes(x = df1$chlorides)) +
  geom_boxplot()

p6 <- ggplot(df1, aes(x = df1$free.sulfur.dioxide)) +
  geom_boxplot() 

p7 <- ggplot(df1, aes(x = df1$total.sulfur.dioxide)) +
  geom_boxplot()

p8 <- ggplot(df1, aes(x = df1$density)) +
  geom_boxplot() 

p9 <- ggplot(df1, aes(x = df1$pH)) +
  geom_boxplot()

p10 <- ggplot(df1, aes(x = df1$sulphates)) +
  geom_boxplot()

p11 <- ggplot(df1, aes(x = df1$alcohol)) +
  geom_boxplot()

## Seems outliers exsits in many variables - IQR Filtering
# 3.1 Getting the quantile 25% & 75%
quartiles_fixed <- quantile(df1$fixed.acidity, probs=c(.25, .75), na.rm = FALSE)
quartiles_volatile <- quantile(df1$volatile.acidity, probs=c(.25, .75), na.rm = FALSE)
quartiles_citric <- quantile(df1$citric.acid, probs=c(.25, .75), na.rm = FALSE)
quartiles_sugar <- quantile(df1$residual.sugar, probs=c(.25, .75), na.rm = FALSE)
quartiles_chlorides <- quantile(df1$chlorides, probs=c(.25, .75), na.rm = FALSE)
quartiles_free.sulfur <- quantile(df1$free.sulfur.dioxide, probs=c(.25, .75), na.rm = FALSE)
quartiles_total.sulfur <- quantile(df1$total.sulfur.dioxide, probs=c(.25, .75), na.rm = FALSE)
quartiles_density <- quantile(df1$density, probs=c(.25, .75), na.rm = FALSE)
quartiles_pH <- quantile(df1$pH, probs=c(.25, .75), na.rm = FALSE)
quartiles_sulphates <- quantile(df1$sulphates, probs=c(.25, .75), na.rm = FALSE)
quartiles_alcohol <- quantile(df1$alcohol, probs=c(.25, .75), na.rm = FALSE)

# 3.2 Getting IQR
IQR_fixed.acidity <- IQR(df1$fixed.acidity) 
IQR_volatile.acidity <- IQR(df1$volatile.acidity)
IQR_citric.acid <- IQR(df1$citric.acid)
IQR_residual.sugar <- IQR(df1$residual.sugar)
IQR_chlorides <- IQR(df1$chlorides)
IQR_free.sulfur.dioxide <- IQR(df1$free.sulfur.dioxide)
IQR_total.sulfur.dioxide <- IQR(df1$total.sulfur.dioxide)
IQR_density <- IQR(df1$density)
IQR_pH <- IQR(df1$pH)
IQR_sulphates <- IQR(df1$sulphates)
IQR_alcohol <- IQR(df1$alcohol)

# 3.3 Getting Upper & Lower Limits
Lower_fixed <- quartiles_fixed[1] - 1.5*IQR_fixed.acidity # For Fixed
Upper_fixed <- quartiles_fixed[2] + 1.5*IQR_fixed.acidity 

Lower_volatile <- quartiles_volatile[1] - 1.5*IQR_volatile.acidity # For Volatile
Upper_volatile <- quartiles_volatile[2] + 1.5*IQR_volatile.acidity 

Lower_citric <- quartiles_citric[1] - 1.5*IQR_citric.acid # For citric 
Upper_citric <- quartiles_citric[2] + 1.5*IQR_citric.acid 

Lower_sugar <- quartiles_citric[1] - 1.5*IQR_residual.sugar # For sugar 
Upper_sugar <- quartiles_citric[2] + 1.5*IQR_residual.sugar 

Lower_chlorides <- quartiles_chlorides[1] - 1.5*IQR_chlorides # For chlorides 
Upper_chlorides <- quartiles_chlorides[2] + 1.5*IQR_chlorides 

Lower_free.sulfur <- quartiles_free.sulfur[1] - 1.5*IQR_free.sulfur.dioxide # For free sulfur 
Upper_free.sulfur <- quartiles_free.sulfur[2] + 1.5*IQR_free.sulfur.dioxide 

Lower_total.sulfur <- quartiles_total.sulfur[1] - 1.5*IQR_total.sulfur.dioxide # For total sulfur 
Upper_total.sulfur <- quartiles_total.sulfur[2] + 1.5*IQR_total.sulfur.dioxide 

Lower_density <- quartiles_density[1] - 1.5*IQR_density # For density 
Upper_density <- quartiles_density[2] + 1.5*IQR_density 

Lower_pH <- quartiles_pH[1] - 1.5*IQR_pH # For pH 
Upper_pH <- quartiles_pH[2] + 1.5*IQR_pH 

Lower_sulphates <- quartiles_sulphates[1] - 1.5*IQR_sulphates # For sulphates 
Upper_sulphates <- quartiles_sulphates[2] + 1.5*IQR_sulphates 

Lower_alcohol <- quartiles_alcohol[1] - 1.5*IQR_alcohol # For alcohol 
Upper_alcohol <- quartiles_alcohol[2] + 1.5*IQR_alcohol 

# 3.4 Outliers are removed from the dataset
L1 <- df1$fixed.acidity > Lower_fixed & df1$fixed.acidity < Upper_fixed
L2 <- df1$volatile.acidity > Lower_volatile & df1$volatile.acidity < Upper_volatile
L3 <- df1$citric.acid > Lower_citric & df1$citric.acid < Upper_citric
L4 <- df1$residual.sugar > Lower_sugar & df1$residual.sugar < Upper_sugar
L5 <- df1$chlorides > Lower_chlorides & df1$chlorides < Upper_chlorides
L6 <- df1$free.sulfur.dioxide > Lower_free.sulfur & df1$free.sulfur.dioxide < Upper_free.sulfur
L7 <- df1$total.sulfur.dioxide > Lower_total.sulfur & df1$total.sulfur.dioxide < Upper_total.sulfur
L8 <- df1$density > Lower_density & df1$density < Upper_density
L9 <- df1$pH > Lower_pH & df1$pH < Upper_pH
L10 <- df1$sulphates > Lower_sulphates & df1$sulphates < Upper_sulphates
L11 <- df1$alcohol > Lower_alcohol & df1$alcohol < Upper_alcohol


data_no_outlier <- subset(df1,(L1 & L2 & L3 & L4 & L5 & L6 & L7 & L8 & L9 & L10 & L11))

dim(data_no_outlier)

#### 4. Splitting Training (70%) and testing set(30%)
set.seed(2022)
wine_split <- initial_split(data_no_outlier, prop = 0.7)
wine_train <- training(wine_split)
wine_test <- testing(wine_split)

#### 5. EDA

wine_train %>% filter(is.na(quality) == FALSE) %>%
  ggpairs(columns = c("fixed.acidity","volatile.acidity","citric.acid",
                      "residual.sugar","chlorides",
                      "free.sulfur.dioxide","total.sulfur.dioxide",
                      "density","pH","sulphates",
                      "alcohol"),
          mapping = aes(color = quality))

wine_train %>% ggplot(aes(x = fixed.acidity, y = quality, fill = quality, col = quality )) +
  geom_boxplot(alpha = 0.2) + geom_jitter(aes(col = quality ))
# Assumptions - all predictor variables are independent each other.

#### 6. Building a model

## 6.1 KNN algorithm --------- Method 01 ------------------

X_train <-wine_train[, 1:11]
x_test <- wine_test[, 1:11]

# K= 1
classifier_knn_1 <- knn(train = X_train,
                      test = x_test,
                      cl = wine_train$quality,
                      k = 1)
classifier_knn_1

# Confusion Matrix
cm_1 <- table(wine_test$quality, classifier_knn_1)
cm_1

misClassError_1 <- mean(classifier_knn_1 != wine_test$quality)


# K= 3
classifier_knn_3 <- knn(train = X_train,
                        test = x_test,
                        cl = wine_train$quality,
                        k = 3)
cm_3 <- table(wine_test$quality, classifier_knn_3)
misClassError_3 <- mean(classifier_knn_3 != wine_test$quality)

# K= 5
classifier_knn_5 <- knn(train = X_train,
                        test = x_test,
                        cl = wine_train$quality,
                        k = 5)
cm_5 <- table(wine_test$quality, classifier_knn_5)
misClassError_5 <- mean(classifier_knn_5 != wine_test$quality)

# K= 7
classifier_knn_7 <- knn(train = X_train,
                        test = x_test,
                        cl = wine_train$quality,
                        k = 7)
cm_7 <- table(wine_test$quality, classifier_knn_7)
misClassError_7 <- mean(classifier_knn_7 != wine_test$quality)

# Calculate out of Sample error

print(paste('Accuracy (K = 1) =', 1-misClassError_1))
print(paste('Accuracy (K = 3) =', 1-misClassError_3))
print(paste('Accuracy (K = 5) =', 1-misClassError_5))
print(paste('Accuracy (K = 7) =', 1-misClassError_7))

# According to the output, it can be said that k increases then the accuracy of 
# the model also decreases.

# Most appropriate when K = 1

## 6.2 KNN Prediction 
head(classifier_knn_3, 5)

# ------------------------------------------------------------------

### 6.2 Decision Trees ---------- Method 02 -----------------------

classifier_tree = rpart(formula = quality ~ .,
                   data = wine_train)

# Predicting the Test set results
y_pred = predict(classifier_tree,
                 newdata = wine_test,
                 type = 'class')
# Making the Confusion Matrix
cm_tree = table(wine_test$quality, y_pred)

# Checking the accuracy of the model
accuracy_Test <- sum(diag(cm_tree)) / sum(cm_tree)

print(paste('Accuracy for Decision Tree', accuracy_Test))

### 6.2 Random Forest ---------- Method 03 -----------------------

classifier_rf <- randomForest(quality ~ ., 
                        data = wine_train, 
                        importance = TRUE,
                        proximity = TRUE)

plot(classifier_rf)

# Prediction & Confusion Matrix – train data
p1 <- predict(classifier_rf, wine_train)
confusionMatrix(p1, wine_train$quality)
# Train data accuracy is 100% that indicates all the values classified correctly.

# Prediction & Confusion Matrix – test data
p2 <- predict(classifier_rf, wine_test)
confusionMatrix(p2, wine_test$quality)
# Test data accuracy is approximately 64%
# So we need to tune the tree
# Select mtry value with minimum out of bag(OOB) error(28%).

mtry <- tuneRF(wine_train[-1],wine_train$quality, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)

#best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
#print(mtry)
#print(best.m)

classifier_rf_2 <- randomForest(quality ~ ., 
                              data = wine_train, 
                              importance = TRUE,
                              proximity = TRUE,
                              mtry=2)

p3 <- predict(classifier_rf_2, wine_test)
confusionMatrix(p3, wine_test$quality)
# Still 63% accuracy on the testing test

#Variable Importance
varImpPlot(classifier_rf,
           sort = T,
           n.var = 11,
           main = "Top 11 - Variable Importance")

importance(classifier_rf)
# According to the informations obtained alcohol is the most influence variable 
# which is affected the wine quality

### 6.4 SVM ---------- Method 04 -----------------------

classifier_SVM = svm(formula = quality ~ .,
                 data = wine_train,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred_SVM = predict(classifier_SVM, newdata = wine_test)

# Making the Confusion Matrix
cm_SVM = table(wine_test$quality, y_pred_SVM)

confusionMatrix(cm_SVM)
# Accuracy is approximately 64%.

### 6.5 Naive Bayes Theorem ---------- Method 04 ----------------------- 

set.seed(2022)  # Setting Seed
classifier_NB <- naiveBayes(quality ~ ., data = wine_train)
classifier_NB

# Predicting on test data'
y_pred_NB <- predict(classifier_NB, newdata = wine_test)

# Confusion Matrix
cm_NB <- table(wine_test$quality, y_pred_NB)

# Model Evaluation
confusionMatrix(cm_NB)

# Accuracy is approximately 55%.




