install.packages("arules")
library(arules)
install.packages("arulesViz")
library(arulesViz)
install.packages("plyr")
library(plyr)
install.packages("caret")
library(caret)
install.packages("randomForest")
library(randomForest)
install.packages("rattle")
library(rattle)
install.packages("rpart")
library(rpart)

#Load the BlackFriday Dataset
BFdf<- read.csv("c://BlackFriday.csv")


#Examine the data
#--------------------------------------------
#Structure of data
str(BFdf)
summary(BFdf)

#View first 4 records to see how the data looks like
head(BFdf,4)

#View the dataset
View(BFdf)

#Check for duplicate data
nrow(BFdf[duplicated(BFdf),])


#Interesting nuggest about the data
#---------------------------------------------------------------
#Product
UniqueProd <- length(unique(BFdf[,2]))
HighPurchase <- max(BFdf[,12])
LowPurchase <-  min(BFdf[,12])

cat("Total number of unique product is ", UniqueProd)
cat("The smallest and the largest purchase made that day were $" , LowPurchase, " and $", HighPurchase)

#Gender
GenderAggregate <- aggregate(cbind(Category1 = BFdf$Product_Category_1, Category2 = BFdf$Product_Category_2, Category3 = BFdf$Product_Category_3) , by=list(Gender = BFdf$Gender),FUN=sum,na.rm=TRUE)

Gender_barplot <-barplot(as.matrix(GenderAggregate[,2:4]), main="Purchase by Category by Gender", col=c("pink","blue"))

legend("topright",legend=c("Female","Male"), cex=.3, fill=c("pink","blue")) 


#Age
#Piechart
PurchaseByAge <-aggregate(cbind(Purchase = BFdf$Purchase) , by=list(Gender = BFdf$Age),FUN=sum)

lab<- as.character(unlist(PurchaseByAge[1]))
x <- as.numeric(unlist(PurchaseByAge[2]))
piepercent<- round(100*x/sum(x), 1)

pie(x,labels=piepercent,main="Purchase by Age Group",col = rainbow(length(x)))
legend("topleft",lab,cex=0.4,fill=rainbow(length(x)))


#Association Rule Mining
#---------------------------------------------------------------

#Make a copy for the data 
BF_AR <- BFdf

#There are a lot of numeric data that need to be converted to discrete data

#Convert numeric data to nominal data
BF_AR$PurchaseCategory<-cut(BF_AR$Purchase,breaks=c(1,5000,8000,12000,25000), labels=c("Small","Medium","Large","Very Large"))

# discretization of numeric columns
cols <-c(5,8:11)
BF_AR[,cols] <- lapply(BF_AR[,cols], factor)

#Remove UserID and Purchase and category cols
BF_AR <- BF_AR[,c(-1,-2,-9,-10,-11,-12)]
str(BF_AR)

#Convert the record dataset to a transactional dataset
fac_var <- sapply(BF_AR, is.factor)
BlackFriTransData <- as(BF_AR[, fac_var], "transactions")

#Inspect the transaction data first
itemFrequencyPlot(BlackFriTransData,topN=5,type="absolute")


#Apply the Apriori AR rules, start with min support and confidence
rules<- apriori(BlackFriTransData, parameter=list(supp=.02,conf=0.8,minlen=2))

#Inspect the rules
inspect(rules[1:5])

#Summary of rules
summary(rules) # we have about 81 rules

#Trim the support/conf/limit to 2 digits
options(digits=2)

#Inspect rules - sorting by confidence and then lift
inspect(head(sort(rules, by="confidence", decreasing = TRUE),10))

inspect(head(sort(rules, by="lift", decreasing = TRUE),10))


#Using Associated Rule Mining as Supervised Learning Method by forcing the target callsification attribute on the RHS of associateion rules
#---------------------------------------------------------------

#Target classification Purchase Category as the rhs rule
rules<- apriori(BlackFriTransData, parameter=list(supp=.01,conf=.3,minlen=2),control=list(verbose=F),appearance= list(default="lhs",rhs=c("PurchaseCategory=Small","PurchaseCategory=Medium","PurchaseCategory=Large","PurchaseCategory=Very Large")))

#Summary of rules
summary(rules)

#Inspect the rules
inspect(rules[1:20])

#Remove redundant rules that is rules that are subset
subset_rules <- which(colSums(is.subset(rules, rules)) >1 )
rules <- sort(rules[-subset_rules], by="lift",decreasing = TRUE)

#Summary of rules
summary(rules)

#Inspect the rules
inspect(rules)

#Classification Algorithms
#--------------------------------------------

#Setup the Train and Test dataset
#This data set is quite large and the time take to train models with caret is very high, so I have decided to use a sample of the original data for my project

BF <- BF_AR[sample(nrow(BF_AR),100000),]
#Using the Hold out method, I will split the dataset into train which will have 80% of data and test which will have the remaining 20%
split_index <- createDataPartition(BF$Gender, p=0.8, list=F)

trainds <- BF[split_index, ]
testds <- BF[-split_index, ]

nrow(trainds)
nrow(testds)

#memory.limit()
#memory.limit(size=32000)

#Random Forest
#---------------------------------------------------------
#Model with default parameters
set.seed(101)
model_RandomForest <- randomForest(Gender~., trainds)
print(model_RandomForest)
#We see that the model has used ntree=500 and mtry=2 to with OOB error rate is 17.33%

RandomForest_Pred <- predict(model_RandomForest, testds)

CM_RF<-confusionMatrix(testds$Gender, RandomForest_Pred) 
print(CM_RF)

importance(model_RandomForest)
varImpPlot(model_RandomForest)

cat("Accuracy for Random Forest Default Model is ", CM_RF$overall[1])


#Plot error
plot(model_RandomForest)

#From the plot we see that the OOB error dips when ntree= 10 and then stabilizes. 

#tune mtry
tune1<-tuneRF(trainds[,2:ncol(trainds)],trainds[,1],stepFactor=0.5,ntreeTry=400,improve=0.05,plot=TRUE)

tune2<-tuneRF(trainds[,2:ncol(trainds)],trainds[,1],stepFactor=0.5,ntreeTry=500,improve=0.05,plot=TRUE)

tune3<-tuneRF(trainds[,2:ncol(trainds)],trainds[,1],stepFactor=0.5,ntreeTry=600,improve=0.05,plot=TRUE)


#Let's tune the model
model_RandomForest_tune <- randomForest(Gender~., trainds, ntree=600, mtry=4)
print(model_RandomForest_tune)


#Predict Model

RandomForest_Pred_tune <- predict(model_RandomForest_tune, testds,importance=TRUE)

CM_RF_tune<-confusionMatrix(testds$Gender, RandomForest_Pred_tune) 
print(CM_RF_tune)

cat("Accuracy for Random Forest tuned Model is ", CM_RF_tune$overall[1])

#Naive Bayes
#--------------------------------------
#Default model
set.seed(250)
model_nb <- suppressWarnings(train(
  Gender~.,
  data=trainds,
  method = "nb", trControl =  trainControl(method = "boot", number = 5)
))

print(model_nb)
predictnb <- suppressWarnings(predict(model_nb,newdata=testds))

CM_NB <- confusionMatrix(predictnb,testds$Gender)

cat("Accuracy for Naive Bayes default Model is ", CM_NB$overall[1])

#Tune parameters
##First Model
model_nb_tune <- suppressWarnings(train(Gender ~ ., data = trainds,
                                    method = "nb",
                                    trControl =  trainControl(method = "boot", number = 5),
                                    tuneGrid = expand.grid(
                                      usekernel = FALSE,
                                      fL = 1,
                                      adjust =1
                                    )))

predictnb_tune <- suppressWarnings(predict(model_nb_tune,newdata=testds))

CM_NB_tune <- confusionMatrix(predictnb_tune,testds$Gender)

cat("Accuracy for Naive Bayes with userkernel=FALSE, fL=1, adjust=1 is ", CM_NB_tune$overall[1])

#Second Model
model_nb_tune1 <- suppressWarnings(train(Gender ~ ., data = trainds,
                                        method = "nb",
                                        trControl =  trainControl(method = "boot", number = 5),
                                        tuneGrid = expand.grid(
                                          usekernel = TRUE,
                                          fL = 1,
                                          adjust =1
                                        )))

print(model_nb_tune1)
predictnb_tune1 <- suppressWarnings(predict(model_nb_tune1,newdata=testds))

CM_NB_tune1 <- confusionMatrix(predictnb_tune1,testds$Gender)

cat("Accuracy for Naive Bayes with userkernel=TRUE, fL=1, adjust=1 is ", CM_NB_tune1$overall[1])

#Third Model
model_nb_tune2 <- suppressWarnings(train(Gender ~ ., data = trainds,
                                         method = "nb",
                                         trControl =  trainControl(method = "boot", number = 5),
                                         tuneGrid = expand.grid(
                                           usekernel = TRUE,
                                           fL = 1,
                                           adjust =3
                                         )))

print(model_nb_tune2)

predict_nb_tune2 <- suppressWarnings(predict(model_nb_tune2,newdata=testds))
CM_NB_tune2 <-confusionMatrix(predict_nb_tune2,testds$Gender)

cat("Accuracy for Naive Bayes with userkernel=TRUE, fL=1, adjust=3 is ", CM_NB_tune2$overall[1])

#Fourth Model
model_nb_tune3 <- suppressWarnings(train(Gender ~ ., data = trainds,
                                         method = "nb",
                                         trControl =  trainControl(method = "boot", number = 5),
                                         tuneGrid = expand.grid(
                                           usekernel = TRUE,
                                           fL = 1,
                                           adjust =5
                                         )))

print(model_nb_tune3)


predict_nb_tune3 <- suppressWarnings(predict(model_nb_tune3,newdata=testds))
CM_NB_tune3 <-confusionMatrix(predict_nb_tune3,testds$Gender)



#knn 
##----------------------------------------------------
#PreProcessing : Since knn is a distance based algorithm, we need to convert all column to numeric
#Make a copy for the data 
BF_data <- BFdf

#Remove UserID and Purchase and category cols
BF_data <- BF_data[,c(-1,-2,-9,-10,-11,-12)]
str(BF_data)

#Convert the colums to numeric
BF_num <- lapply(BF_data[,2:ncol(BF_data)], as.numeric)
BF_num <- cbind(BF_data[1],BF_num)
str(BF_num)

#Setup the Train and Test dataset
#This data set is quite large and the time take to train models with caret is very high, so I have decided to use a sample of the original data for my project

BF_ds <- BF_num[sample(nrow(BF_num),50000),]
#Using the Hold out method, I will split the dataset into train which will have 80% of data and test which will have the remaining 20%
split_index <- createDataPartition(BF_ds$Gender, p=0.8, list=F)

training <- BF_num[split_index, ]
testing <- BF_num[-split_index, ]

str(training)
nrow(testing)


#Default Mode
set.seed(890)

#Normalizing numeric data is important since knn is a distance based algorithm
preprocess <- preProcess(training,method=c("scale","center"))
print(preprocess)

train_preprocess <- predict(preprocess,newdata=training)
test_preprocess <- predict(preprocess,newdata=testing)
summary(train_preprocess)

model_knn_preprocess <- train(Gender~., data=train_preprocess, method="knn")
print(model_knn_preprocess)
predict_knn_preprocess <- predict(model_knn_preprocess, newdata=test_preprocess)
confusionMatrix(predict_knn_preprocess,test_preprocess$Gender)

#Let's tune the parameters
#The value for k is generally chosen as the square root of the number of observations


#k <- round(sqrt(nrow(training))) 
model_knn_tune <- train(Gender~., data=train_preprocess, method="knn", tuneGrid=data.frame(k=seq(1,25,by=1)),trControl=trainControl(method="repeatedcv",number=5))

print(model_knn_tune)

plot(model_knn_tune)

#we see that accuracy is highest when k=5 based on the plot which is what the model has used.

predict_knn_tune <- predict(model_knn_tune, newdata=test_preprocess)
cf_knn<-confusionMatrix(predict_knn_tune,test_preprocess$Gender)
print(cf_knn)

cat("Accuracy for knn tuned Model is ", cf_knn$overall[1])

#SVM
#-------------------------------------------------
#Setup the Train and Test dataset
#This data set is quite large and the time take to train models with caret is very high, so I have decided to use a sample of the original data for my project

BF_ds <- BF_num[sample(nrow(BF_num),50000),]
#Using the Hold out method, I will split the dataset into train which will have 80% of data and test which will have the remaining 20%
split_index <- createDataPartition(BF_ds$Gender, p=0.8, list=F)

training <- BF_ds[split_index, ]
testing <- BF_ds[-split_index, ]

str(training)
nrow(testing)

#Build Model with default parameters
set.seed(489)
model_svm <- train (Gender ~., data=training, method="svmLinear",preProcess=c("center","scale"),scale = FALSE)

#Evaluate the model
print(model_svm)

#Predict
svm_pred <- predict(model_svm,newdata=testing)
cf_svm <- confusionMatrix(svm_pred,testing$Gender)
print(cf_svm)

cat("Accuracy for SVM default Model is ", cf_svm$overall[1])


#SVM constructs a hyperplane such that it has the largest distance to the nearest data points (called support vectors). If the dimensions have different ranges, the dimension with much bigger range of values influences the distance more than other dimensions. So its necessary to scale the features such that all the features have similar influence when calculating the distance to construct a hyperplane. 

#Linear Model
#----------------

model_svm_linear <- suppressWarnings(train(Gender~., data=training, method="svmLinear", preProcess=c("center","scale"),trControl=trainControl(method="boot",number=5),tuneGrid=expand.grid(C=c(0.001, 0.01, 0.1, 1))))

print(model_svm_linear)
plot(model_svm_linear)

#Predict
svm_predict_linear <- predict(model_svm_linear, newdata=testing)
cf_svm_linear <- confusionMatrix(svm_predict_linear,testing$Gender)
print(cf_svm_linear)

cat("Accuracy for SVM Linear Model is ", cf_svm_linear$overall[1])

#Non-Linear Kernel RBF
#----------------------

model_svm_rbf <- suppressWarnings(train(Gender~., data=training, method="svmRadial", preProcess=c("center","scale"),trControl=trainControl(method="boot",number=5),tuneGrid=expand.grid(C=c(0.001, 0.01, 0.1), sigma=c(0.100,0.200,0.300))))

print(model_svm_rbf)

#Predict
svm_predict_rbf <- predict(model_svm_rbf, newdata=testing)
cf_svm_rbf <- confusionMatrix(svm_predict_rbf,testing$Gender)
print(cf_svm_rbf)

cat("Accuracy for SVM RBF Model is ", cf_svm_rbf$overall[1])


