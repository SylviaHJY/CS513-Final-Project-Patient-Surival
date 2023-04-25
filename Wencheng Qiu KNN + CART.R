#################################################
#  Project    : Final Project
#  Purpose    : Patient Survival After One Year of Treatment Prediction
#  Name       : Wencheng Qiu
#  CWID			  : 20011469

rm(list=ls())

# Install.packages("Metrics")
library("FactoMineR")
library("factoextra")
library("corrplot")
library(MLmetrics)


# Read the csv file
# setwd("~/Desktop")
# dataset <- read.csv("Patient_Survive.csv", header = TRUE, stringsAsFactors = FALSE, na.strings = "?")
filename<-file.choose()
dataset <-read.csv(filename, na.strings=c("?"))
View(dataset)

# delete the rows with missing value
pharma_data <- na.omit(dataset)

# Pre-processing, convert columns to numeric for pharma_data
# Eliminate the 3rd, 8th, 14th column for identifier
pharma_data <- pharma_data[,-3]
pharma_data <- pharma_data[,-8]
pharma_data <- pharma_data[,-14]
pharma_data$Patient_Smoker <- ifelse(pharma_data$Patient_Smoker == "YES", 1, 0)
pharma_data$Patient_Rural_Urban <- ifelse(pharma_data$Patient_Rural_Urban == "URBAN", 1, 0)
#View(pharma_data)

# Pre-processing, normalize columns using z-score standardization
pharma_data$ID_Patient_Care_Situation <- scale(pharma_data$ID_Patient_Care_Situation)
pharma_data$Diagnosed_Condition <- scale(pharma_data$Diagnosed_Condition)
pharma_data$Treated_with_drugs <-scale(pharma_data$Treated_with_drugs)
pharma_data$Patient_Age <- scale(pharma_data$Patient_Age)
pharma_data$Patient_Body_Mass_Index <- scale(pharma_data$Patient_Body_Mass_Index)
pharma_data$Number_of_prev_cond <- scale(pharma_data$Number_of_prev_cond)
View(pharma_data)

# Correlation plot
cor.mat <- round(cor(pharma_data),2)

# PCA
res.pca <- PCA(pharma_data, graph = TRUE)
var <- get_pca_var(res.pca)
corrplot(var$cos2, is.corr=FALSE)
eig.val <- get_eigenvalue(res.pca)
fviz_pca_var(res.pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))


###################################################################

# TEST 1: KNN Method
# eliminates columns
library(class)
new_data <- pharma_data[,c(4,6,12,14,15)]
# draw the correlation plot
cor.mat <- round(cor(new_data),2)
corrplot(cor.mat, type="upper", order="hclust", tl.col="black", tl.srt=45)
# split the data into 70% training and 30% testing sets
set.seed(211)
trainIndex <- sample(1:nrow(new_data), 0.7 * nrow(new_data))
trainData <- new_data[trainIndex, ]
testData <- new_data[-trainIndex, ]
# define the range of k values
k_values <- c(1, 3, 5, 7, 9, 11, 13, 15, 17, 20)
# create an empty vector to store the accuxracy values
accuracy_values <- numeric(length(k_values))
# Create empty vectors to store evaluation metrics
f1_values <- numeric(length(k_values))
# Iterate through each k value and calculate the accuracy
for (i in 1:length(k_values)) {
  knnModel <- knn(train = trainData[, 1:4], test = testData[, 1:4], cl = trainData$Survived_1_year, k = k_values[i])
  accuracy_values[i] <- sum(knnModel == testData$Survived_1_year) / nrow(testData)
  f1_values[i] <- F1_Score(knnModel, testData$Survived_1_year)
}
# print the accuracy values for each k value
for (i in 1:length(k_values)) {
  cat("Accuracy of KNN with k=", k_values[i], ":", accuracy_values[i],", F1-Score:", f1_values[i], "\n")
}
# plot the accuracies
par(mar=c(5,4,4,2))
plot(k_values, accuracy_values, type = "b", xlab = "k", ylab = "Accuracy", main = "Accuracy of KNN models")


################################################################

# TEST 3: CART Method
# install.packages("ggplot2")
library(rpart)
library(rpart.plot) 
library(caret)
# Split the data into training and testing sets
set.seed(100)
train_indices <- sample(nrow(pharma_data), 0.7 * nrow(pharma_data))
trainning <- pharma_data[train_indices, ]
test <- pharma_data[-train_indices, ]
# Grow the tree
fit_Dtree <-rpart(Survived_1_year~., data = trainning, method="class")
# display the results
printcp(fit_Dtree)
# detailed summary of splits
summary(fit_Dtree)
# Plot the tree
par(mar=c(1,1,1,1))
rpart.plot(fit_Dtree)
# make predictions on the test data
pred_Dtree <- predict(fit_Dtree, newdata = test, type="class")
# create the frequency table
accuray_Dtree <- table(Actual = test[,"Survived_1_year"], CART = pred_Dtree)
accuray_Dtree
# calculate accuracy
Dtree_error_rate <- (sum(accuray_Dtree) - sum(diag(accuray_Dtree))) / sum(accuray_Dtree)
Dtree_error_rate
print(paste0('Accuracy: ', (1 - Dtree_error_rate)*100))
# calculate F1_Score
F1_Score(pred_Dtree,test$Survived_1_year)
