#  Project    : Final Project
#  Purpose    : Patient Survival After One Year of Treatment Prediction
#  Name       : Zimeng Zhao
#  CWID			  : 20012231

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
View(pharma_data)

#Define z_score normalization function
pharma_data$Number_of_prev_cond <- scale(pharma_data$Number_of_prev_cond)
pharma_data$Patient_Age <- scale(pharma_data$Patient_Age)
pharma_data$Diagnosed_Condition <- scale(pharma_data$Diagnosed_Condition)
pharma_data$Patient_Body_Mass_Index <- scale(pharma_data$Patient_Body_Mass_Index)
pharma_data$ID_Patient_Care_Situation <- scale(pharma_data$ID_Patient_Care_Situation)

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
