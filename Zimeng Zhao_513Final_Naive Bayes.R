#  Project    : Final Project
#  Purpose    : Patient Survival After One Year of Treatment Prediction
#  Name       : Zimeng Zhao
#  CWID			  : 20012231

library("FactoMineR")
library("factoextra")
library("corrplot")
install.packages("Metrics")
library(Metrics)


rm(list=ls())

#Load the data
filename<-file.choose()
mydata<-read.csv(filename, na.strings=c("?"))
View(mydata)
summary(mydata)

# delete the rows with missing value
pharma_data <- na.omit(mydata)

#  Pre-processing, convert categorical columns to numeric for pharma_data
pharma_data <- pharma_data[,-3] # Eliminate the 3rd column for identifier
pharma_data$Patient_Smoker <- ifelse(pharma_data$Patient_Smoker == "YES", 1, 0)
pharma_data$Patient_Rural_Urban <- ifelse(pharma_data$Patient_Rural_Urban == "URBAN", 1, 0)
pharma_data$Patient_mental_condition <- ifelse(pharma_data$Patient_mental_condition == "Stable", 1, 0)


#Define z_score normalization function
pharma_data$Number_of_prev_cond <- scale(pharma_data$Number_of_prev_cond)
pharma_data$Patient_Age <- scale(pharma_data$Patient_Age)
pharma_data$Diagnosed_Condition <- scale(pharma_data$Diagnosed_Condition)
pharma_data$Patient_Body_Mass_Index <- scale(pharma_data$Patient_Body_Mass_Index)
pharma_data$ID_Patient_Care_Situation <- scale(pharma_data$ID_Patient_Care_Situation)
View(pharma_data)

#Correlation plot
cor.mat <- round(cor(pharma_data),2)

#PCA
res.pca <- PCA(pharma_data, graph = TRUE)
var <- get_pca_var(res.pca)
corrplot(var$cos2, is.corr=FALSE)
eig.val <- get_eigenvalue(res.pca)
eig.val
fviz_pca_var(res.pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

###################################################################

# TEST 1: KNN Method
#Eliminate the 8th, 14th column for identifier
pharma_data <- pharma_data[,-8]
pharma_data <- pharma_data[,-14]

# eliminates columns
library(class)
new_data <- pharma_data[,c(4,6,12,14,15)]

# draw the correlation plot
cor.mat <- round(cor(new_data),2)
corrplot(cor.mat, type="upper", order="hclust", tl.col="black", tl.srt=45)

# split the data into 70% training and 30% testing sets
set.seed(100)
trainIndex <- sample(1:nrow(new_data), 0.7 * nrow(new_data))
trainData <- new_data[trainIndex, ]
testData <- new_data[-trainIndex, ]

# define the range of k values
k_values <- c(1, 3, 5, 7, 9, 11, 13, 15, 17, 20)

# create an empty vector to store the accuracy values
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

###################################################################
# TEST 2: Naive Bayes

###################################################################



#install.packages('e1071', dependencies = TRUE)
library(e1071)
library(class) 
library(caret)
library(mlbench)
library(ROCR)

newdata<- pharma_data[1:17]
summary(newdata)

# sample size: 70% 
samp <- floor(0.70 * nrow(newdata))

#Set the seed,generates a sequence of integers from 1 to the number of rows in newdata
set.seed(123)
train_ind <- sample(seq_len(nrow(newdata)), size = samp)

#Loading 70% record in training dataset
training <- newdata[train_ind, ]

#Loading 30% Breast cancer in testing dataset
testing <- newdata[-train_ind, ]

#Implementing NaiveBayes
model_naive<- naiveBayes(Survived_1_year ~ ., data = training)

#Predicting target class for the Validation set
predict_naive <- predict(model_naive, testing)

#Confusion matrix
conf_matrix <- table(predict_nb=predict_naive,Survived_1_year=testing$Survived_1_year)
print(conf_matrix)

# Extract values from the confusion matrix
tp <- conf_matrix[2, 2] # True positives
fp <- conf_matrix[1, 2] # False positives
tn <- conf_matrix[1, 1] # True negatives
fn <- conf_matrix[2, 1] # False negatives

# Calculate accuracy
accuracy <- (tp + tn) / (tp + fp + tn + fn)
accuracy

# Calculate precision
precision <- tp / (tp + fp)
precision

# Calculate recall
recall <- tp / (tp + fn)
recall

# Calculate F1-score
f1 <- 2 * precision * recall / (precision + recall)
f1
