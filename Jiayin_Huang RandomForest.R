#################################################
#  Company    : Stevens 
#  Project    : CS513-Data Mining
#  Purpose    : Final-Project-Random Forest
#  First Name  : Jiayin
#  Last Name  : Huang
#  Id			    : 10477088


rm(list=ls())
#################################################
#Load the data
filename<-file.choose()
mydata<-read.csv(filename, na.strings=c("?"))
View(mydata)
summary(mydata)

# delete the rows with missing value
pharma_data <- na.omit(mydata)

library(randomForest)
set.seed(123)

#  Pre-processing, convert categorical columns to numeric for pharma_data
pharma_data <- pharma_data[,-3] # Eliminate the 3rd column for identifier
pharma_data$Patient_Smoker <- ifelse(pharma_data$Patient_Smoker == "YES", 1, 0)
pharma_data$Patient_Rural_Urban <- ifelse(pharma_data$Patient_Rural_Urban == "URBAN", 1, 0)
pharma_data$Patient_mental_condition <- ifelse(pharma_data$Patient_mental_condition == "Stable", 1, 0)
View(pharma_data)

#Define z_score normalization function
pharma_data$Number_of_prev_cond_zscore <- scale(pharma_data$Number_of_prev_cond)
pharma_data$Patient_Age_zscore <- scale(pharma_data$Patient_Age)
pharma_data$Diagnosed_Condition_zscore <- scale(pharma_data$Diagnosed_Condition)
pharma_data$Patient_Body_Mass_Index_zscore <- scale(pharma_data$Patient_Body_Mass_Index)
pharma_data$ID_Patient_Care_Situation_zscore <- scale(pharma_data$ID_Patient_Care_Situation)

pharma_data$Survived_1_year <- factor(pharma_data$Survived_1_year, levels = c(0, 1), labels = c("Not_Survived", "Survived"))
idx<-sort(sample(nrow(pharma_data),round(.70*nrow(pharma_data))))
training <- pharma_data[idx,]
test <- pharma_data[-idx,]


#6.2 Use the Random Forest methodology to develop a classification model for the Diagnosis and identify important features.
fit <- randomForest(Survived_1_year ~ ., data=training, importance=TRUE, ntree=1000)
importance(fit)
varImpPlot(fit)

# Create a confusion matrix to compare the predicted results with the actual results
#predict with test
Prediction <- predict(fit, test)
conf_matrix <- table(actual=test$Survived_1_year, Prediction)

accuracy <- function (x) {sum(diag(x) / (sum(rowSums(x))) * 100);};
accuracy(conf_matrix)

# Calculate precision (positive predictive value)
precision <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[1, 2])

# Calculate recall (sensitivity)
recall <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[2, 1])

# Calculate F1 score
F1_score <- 2 * ((precision * recall) / (precision + recall))

#error rate
wrong<- (test$Survived_1_year!=Prediction)
errorRate<-sum(wrong)/length(wrong)

