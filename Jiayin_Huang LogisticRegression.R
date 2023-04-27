#################################################
#  Company    : Stevens 
#  Project    : CS513-Data Mining
#  Purpose    : Final-Project-Random Forest
#  First Name  : Jiayin
#  Last Name  : Huang
#  Id			    : 10477088


rm(list=ls())
#################################################
filename<-file.choose()
pharma_data<-read.csv(filename, na.strings=c("?"))
View(pharma_data)

# delete the rows with missing value
pharma_data <- na.omit(pharma_data)
pharma_data<- pharma_data[,c(2,4,5,6,7,17,18)]
View(pharma_data)
# Convert Survived_1_year to a factor
pharma_data$Survived_1_year <- factor(pharma_data$Survived_1_year, levels = c(0, 1), labels = c("Not_Survived", "Survived"))

# Create training and test sets (70% training, 30% test)
idx <- sort(sample(nrow(pharma_data), round(.70 * nrow(pharma_data))))
training <- pharma_data[idx,]
test <- pharma_data[-idx,]

# Perform logistic regression
install.packages("rlang", dependencies = TRUE)
library("glmnet")
install.packages("caret")
library(caret)

# Create logistic regression model
# 在创建逻辑回归模型时，使用 family = "binomial" 即可将线性模型转换为逻辑回归模型。
log_model <- glm(Survived_1_year ~ ., family = "binomial", data = training)

# Summary of the logistic regression model
summary(log_model)

#predict with test
Prediction <- predict(log_model, test)
# Choose a threshold (e.g., 0.5) to classify predicted probabilities
pred_results <- ifelse(Prediction > 0.5,"Survived","Not_Survived")

# Compute accuracy, confusion matrix, and other performance metrics
#confusionMatrix(as.factor(pred_results), as.factor(test$Survived_1_year))
conf_matrix <- table(actual=test$Survived_1_year, pred_results)

confusionMatrix(as.factor(pred_results), test$Survived_1_year)

accuracy <- function (x) {sum(diag(x) / (sum(rowSums(x))) * 100);};
accuracy(conf_matrix)
# Calculate precision (positive predictive value)
precision <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[1, 2])

# Calculate recall (sensitivity)
recall <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[2, 1])

# Calculate F1 score
F1_score <- 2 * ((precision * recall) / (precision + recall))
#error rate
wrong<- (test$Survived_1_year!=pred_results)
errorRate<-sum(wrong)/length(wrong)

