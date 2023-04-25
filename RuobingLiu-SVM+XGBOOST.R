
rm(list=ls())
library("FactoMineR")
library("factoextra")
library("corrplot")
library(RColorBrewer)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(MLmetrics)
library(car)
library(e1071)
library(rpart)
library(pROC)
?PCA

#Load the data
filename<-file.choose()
mydata<-read.csv(filename, na.strings=c("?"))
View(mydata)
summary(mydata)

# delete the rows with missing value
pharma_data <- na.omit(mydata)
pharma_data$Patient_Smoker <- ifelse(pharma_data$Patient_Smoker == "YES", 1, 0)
pharma_data$Patient_Rural_Urban <- ifelse(pharma_data$Patient_Rural_Urban == "URBAN", 1, 0)
#  Pre-processing, convert categorical columns to numeric for pharma_data
pharma_data <- pharma_data[,-3] # Eliminate the 3rd column for identifier
pharma_data <- pharma_data[,-8]
pharma_data <- pharma_data[,-14]
View(pharma_data)

#Define z_score normalization function
pharma_data$Number_of_prev_cond <- scale(pharma_data$Number_of_prev_cond)
pharma_data$Patient_Age <- scale(pharma_data$Patient_Age)
pharma_data$Diagnosed_Condition <- scale(pharma_data$Diagnosed_Condition)
pharma_data$Patient_Body_Mass_Index <- scale(pharma_data$Patient_Body_Mass_Index)
pharma_data$Treated_with_drugs <-scale(pharma_data$Treated_with_drugs)



#PCA
res.pca <- PCA(pharma_data, graph = TRUE)
var <- get_pca_var(res.pca)
corrplot(var$cos2, is.corr=FALSE)
eig.val <- get_eigenvalue(res.pca)
eig.val
fviz_pca_var(res.pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)

#Official Data
data<- pharma_data[,c(4,6,12,14,15)]
data_val <- sort(sample(nrow(data), size = as.integer(.70 * nrow(pharma_data))))
training <- data[data_val,]
#View(training)
test <- data[-data_val,]
View(test)
#Correlation plot
cor.mat <- round(cor(pharma_data),2)
corrplot(cor.mat, type="upper", order="hclust", 
         tl.col="black", tl.srt=45)
# create test and training dataset
# use 30% test 70% training data
set.seed(100)
data_val <- sort(sample(nrow(pharma_data), size = as.integer(.70 * nrow(pharma_data))))
training <- pharma_data[data_val,]
View(training)
test <- pharma_data[-data_val,]
View(test)

## svm
svm.model <- svm( Survived_1_year~ ., data =training  )
svm.pred <- predict(svm.model,  test )
svm.pred <-ifelse(svm.pred >0.5,1,0)
#table(actual=test[,15],svm.pred )
SVM_wrong<- (test$Survived_1_year!=svm.pred)
rate<-sum(SVM_wrong)/length(SVM_wrong)
conf_matrix<-table(svm.pred,test$Survived_1_year)
accuracy <- function(x){sum(diag(x)/(sum(colSums(x))))}
accuracy(conf_matrix)

cat('SVM model case:\n')
w <- t(svm.model$coefs) %*% svm.model$SV                 # weight vectors
w <- apply(w, 2, function(v){sqrt(sum(v^2))})  # weight
w <- sort(w, decreasing = T)
print(w)
barplot(w)
##ROC 
roc_score=roc(test[,15], svm.pred) #AUC score
plot(roc_score ,main ="ROC curve -- SVM ")
F1_Score(svm.pred,test$Survived_1_year)
## XGBOOST
bst <- xgboost(data = as.matrix(training[-15]), 
               label =training$Survived_1_year,
               max.depth = 2, eta = 1, 
               nthread = 2, nrounds = 2, 
               objective = "binary:logistic", 
               verbose = 1)
pred <- predict(bst, as.matrix(test[-15]))
pred <-ifelse(pred >0.5,1,0)
F1_Score(pred,test$Survived_1_year)
SVM_wrong<- (test$Survived_1_year!=pred)
rate<-sum(SVM_wrong)/length(SVM_wrong)
conf_matrix<-table(pred,test$Survived_1_year)
accuracy <- function(x){sum(diag(x)/(sum(colSums(x))))}
accuracy(conf_matrix)

model <- xgb.dump(bst, with_stats = T)
model[1:10] #This statement prints top 10 nodes of the model
# names of attributes
names <- dimnames(data.matrix(training[,-15]))[[2]]
# ca
importance_matrix <- xgb.importance(names, model =bst)

# plot
xgb.plot.importance(importance_matrix[1:10,])


