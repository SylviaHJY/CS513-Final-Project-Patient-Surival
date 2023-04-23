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

#newdata<- pharma_data[1:17]
#summary(newdata)

# sample size: 70% 
samp <- floor(0.70 * nrow(pharma_data))

#Set the seed,generates a sequence of integers from 1 to the number of rows in newdata
set.seed(123)
train_ind <- sample(seq_len(nrow(pharma_data)), size = samp)

#Loading 70% record in training dataset
training <- pharma_data[train_ind, ]

#Loading 30% Breast cancer in testing dataset
testing <- pharma_data[-train_ind, ]
summary(testing)
#ANN methodology
#install.packages("neuralnet")
library("neuralnet")

#Train the neural network with five (5) nodes in the hidden layer
#"Output~." specifies the formula for the neural network model. In this case, "Output" is the dependent variable (i.e., the variable we want to predict), and the "~." notation indicates that all other variables in the "predata" data frame should be used as independent variables in the model.
#"hidden=5" specifies that the neural network should have one hidden layer with five nodes.
#"threshold=0.01" specifies the threshold value for the partial derivatives of the error function. This threshold determines the stopping criteria for the neural network training algorithm. 
nnm<- neuralnet(Survived_1_year~., training, hidden=5, threshold = 0.05)
print(nnm)
prediction <-predict(nnm , testing)
pred_cat <- ifelse(prediction<0.5,0,1)
conf_matrix <-table(Actual = testing$Survived_1_year, Prediction = pred_cat)

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

error_rate<- 1-accuracy
error_rate

