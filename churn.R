library(tibble)
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(tidyverse)
library(randomForest)
library(ipred)
library(gbm)
library(plyr)
#data import
df <- read.csv("churn.csv")
head(df)
#delete columns RowNumber, Surname and CustomerId
df <- df[-c(1:3)]
#check data types
glimpse(df)
#set factor variables
cols <- c("Geography","Tenure","NumOfProducts","Gender","HasCrCard",
          "IsActiveMember","Exited")
df[cols] <- lapply(df[cols], as.factor)
#are there missings?
sum(is.na(df))
#data visualization
groupplot <- function(variable, xlab) {
  ggplot(df, aes(variable, ..count..)) + 
  geom_bar(aes(fill = Exited), position = "dodge") +
  labs(x = xlab)
 }

densitplot <- function(variable, xlab) {
  ggplot(df, aes(x=variable, fill=Exited)) +
    geom_density(alpha=0.4) +
    labs(x = xlab)
}
glimpse(df)

g1 <- groupplot(df$Geography, "Geography")
g2 <- groupplot(df$Gender, "Gender")
g3 <- groupplot(df$Tenure, "Tenure")
g4 <- groupplot(df$NumOfProducts, "NumOfProducts")
g5 <- groupplot(df$HasCrCard, "HasCrCard")

d1 <- densitplot(df$CreditScore, "CreditScore")
d2 <- densitplot(df$Age, "Age")
d3 <- densitplot(df$Balance, "Balance")
d4 <- densitplot(df$EstimatedSalary, "EstimatedSalary")

grid.arrange(g1,g2,g3,g4,g5, nrow = 2)
grid.arrange(d1,d2,d3,d4, nrow = 2)

#split into train, validation and test sets
set.seed(1)
idx <- sample(c(1:3), size = nrow(df), 
              replace = TRUE, prob = c(.7, .15, .15))
train <- (1:nrow(df))[idx == 1]
valid <- (1:nrow(df))[idx == 2]
test <- (1:nrow(df))[idx == 3]
#tree 1
tree.model.1 <- rpart(Exited~.,data = df,subset = train,
                      method = "class",xval = 10)

rpart.plot(tree.model.1, box.palette="RdBu", shadow.col="gray", nn=TRUE)
s.tree.model.1 <- summary(tree.model.1)
tree.pred.1 = predict(tree.model.1, newdata = df[valid,], type = "class")
cm.tree.model.1 <- confusionMatrix(data = tree.pred.1, df[valid,]$Exited,
                                   positive = "1")
cm.tree.model.1
#the most important factor is Sensitivity (we want to maximize it)
cm.tree.model.1$byClass["Sensitivity"]

#tree 2
tree.model.2 <- rpart(Exited~.,data = df,subset = train,
                      method = "class",xval = 10,
                      minbucket = 500)

rpart.plot(tree.model.2, box.palette="RdBu", shadow.col="gray", nn=TRUE)
s.tree.model.2 <- summary(tree.model.2)
tree.pred.2 = predict(tree.model.2, newdata = df[valid,], type = "class")
cm.tree.model.2 <- confusionMatrix(data = tree.pred.2, df[valid,]$Exited,
                                   positive = "1")
cm.tree.model.2
#the most important factor is Sensitivity (we want to maximize it)
cm.tree.model.2$byClass["Sensitivity"]

which.max(c(cm.tree.model.1$byClass["Sensitivity"],
cm.tree.model.2$byClass["Sensitivity"]))
#model 2


#random forest 1
forest.model.1 <- randomForest(Exited~.,data = df,subset = train,
                               mtry = 3, ntree = 500, importance = TRUE)

forest.pred.1 = predict(forest.model.1, newdata = df[valid,], type = "class")
cm.forest.model.1 <- confusionMatrix(data = forest.pred.1, df[valid,]$Exited,
                                   positive = "1")
#Sensitivity
cm.forest.model.1$byClass["Sensitivity"]
importance(forest.model.1)        
varImpPlot(forest.model.1) 

#random forest 2
forest.model.2 <- randomForest(Exited~.,data = df,subset = train,
                               mtry = 3, ntree = 500, importance = TRUE,
                               replace = TRUE, strata = NumOfProducts,
                               norm.votes = TRUE)

forest.pred.2 = predict(forest.model.2, newdata = df[valid,], type = "class")
cm.forest.model.2 <- confusionMatrix(data = forest.pred.2, df[valid,]$Exited,
                                     positive = "1")
#Sensitivity
cm.forest.model.2$byClass["Sensitivity"]
importance(forest.model.2)        
varImpPlot(forest.model.2)  

#random forest 3
forest.model.3 <- randomForest(Exited~.,data = df,subset = train,
                               mtry = 3, ntree = 500, importance = TRUE,
                               replace = FALSE, strata = NumOfProducts,
                               norm.votes = TRUE)

forest.pred.3 = predict(forest.model.3, newdata = df[valid,], type = "class")
cm.forest.model.3 <- confusionMatrix(data = forest.pred.3, df[valid,]$Exited,
                                     positive = "1")
#Sensitivity
cm.forest.model.3$byClass["Sensitivity"]
importance(forest.model.3)        
varImpPlot(forest.model.3)  

which.max(c(cm.forest.model.1$byClass["Sensitivity"],
cm.forest.model.2$byClass["Sensitivity"],
cm.forest.model.3$byClass["Sensitivity"]))

#model 1

#bagging 1
bagging.model.1 <- bagging(Exited~.,data = df,subset = train,
                           nbagg = 25, method = "double")

bagging.pred.1 = predict(bagging.model.1, newdata = df[valid,], type = "class")
cm.bagging.model.1 <- confusionMatrix(data = bagging.pred.1, df[valid,]$Exited,
                                     positive = "1")
#Sensitivity
cm.bagging.model.1$byClass["Sensitivity"]

#bagging 1
bagging.model.2 <- bagging(Exited~.,data = df,subset = train,
                           nbagg = 25, method = "standard",
                           coob = TRUE)

bagging.pred.2 = predict(bagging.model.2, newdata = df[valid,], type = "class")
cm.bagging.model.2 <- confusionMatrix(data = bagging.pred.2, df[valid,]$Exited,
                                      positive = "1")
#Sensitivity
cm.bagging.model.2$byClass["Sensitivity"]

which.max(c(cm.bagging.model.1$byClass["Sensitivity"],
cm.bagging.model.2$byClass["Sensitivity"]))

#model 1

#boosting 1
boosting.model.1 <- gbm(as.character(Exited)~.,data = df[train,],
                        distribution = "bernoulli", n.trees = 5000, verbose = F)
boosting.pred.1 <- predict(boosting.model.1, newdata = df[valid,], 
                           n.trees = 5000, type = "response")
boosting.pred.1 <- ifelse(boosting.pred.1 >= .5, 1, 0)
cm.boosting.model.1 <- confusionMatrix(data = factor(boosting.pred.1), 
                                       factor(df[valid,]$Exited),
                                       positive = "1")
#Sensitivity
cm.boosting.model.1$byClass["Sensitivity"]
summary(boosting.model.1, las = 2)

# reducing the number of iterations:
ntree.opt.oob.1 <- gbm.perf(boosting.model.1, method = "OOB", plot.it = T)

#boosting 2
boosting.model.2 <- gbm(as.character(Exited)~.,data = df[train,],
                        distribution = "bernoulli", n.trees = 150,
                        verbose = F)
boosting.pred.2 <- predict(boosting.model.2, newdata = df[valid,], 
                           n.trees = 150, type = "response")
boosting.pred.2 <- ifelse(boosting.pred.2 >= .5, 1, 0)
cm.boosting.model.2 <- confusionMatrix(data = factor(boosting.pred.2), 
                                       factor(df[valid,]$Exited),
                                       positive = "1")
#Sensitivity
cm.boosting.model.2$byClass["Sensitivity"]
summary(boosting.model.2, las = 2)

# reducing the number of iterations:
ntree.opt.oob.2 <- gbm.perf(boosting.model.2, method = "OOB", plot.it = T)

#boosting 3
boosting.model.3 <- gbm(as.character(Exited)~.,data = df[train,],
                        distribution = "bernoulli", n.trees = 150,
                        verbose = F, interaction.depth = 4,
                        n.minobsinnode = 100)

boosting.pred.3 <- predict(boosting.model.3, newdata = df[valid,], 
                           n.trees = 150, type = "response")
boosting.pred.3 <- ifelse(boosting.pred.3 >= .5, 1, 0)
cm.boosting.model.3 <- confusionMatrix(data = factor(boosting.pred.3), 
                                       factor(df[valid,]$Exited),
                                       positive = "1")
#Sensitivity
cm.boosting.model.3$byClass["Sensitivity"]
summary(boosting.model.3, las = 2)

# reducing the number of iterations:
ntree.opt.oob.3 <- gbm.perf(boosting.model.3, method = "OOB", plot.it = T)

#boosting 4
boosting.model.4 <- gbm(as.character(Exited)~.,data = df[train,],
                        distribution = "bernoulli", n.trees = 60,
                        verbose = F, interaction.depth = 4,
                        n.minobsinnode = 100)

boosting.pred.4 <- predict(boosting.model.4, newdata = df[valid,], 
                           n.trees = 60, type = "response")
boosting.pred.4 <- ifelse(boosting.pred.4 >= .5, 1, 0)
cm.boosting.model.4 <- confusionMatrix(data = factor(boosting.pred.4), 
                                       factor(df[valid,]$Exited),
                                       positive = "1")
#Sensitivity
cm.boosting.model.4$byClass["Sensitivity"]
summary(boosting.model.4, las = 2)

which.max(c(cm.boosting.model.1$byClass["Sensitivity"],
cm.boosting.model.2$byClass["Sensitivity"],
cm.boosting.model.3$byClass["Sensitivity"],
cm.boosting.model.4$byClass["Sensitivity"]))

#model 3
#2,1,1,3
#prediction on a test set
tree.pred.final <- predict(tree.model.2, newdata = df[test,], 
                           type = "class")
forest.pred.final <- predict(forest.model.1, newdata = df[test,], 
                             type = "class")
bagging.pred.final <- predict(bagging.model.1, newdata = df[test,], 
                              type = "class")
boosting.pred.final <- predict(boosting.model.3, newdata = df[test,], 
                           n.trees = 60, type = "response")

cm.tree.model.final <- confusionMatrix(data = tree.pred.final, df[test,]$Exited,
                                      positive = "1")
cm.forest.model.final <- confusionMatrix(data = forest.pred.final, df[test,]$Exited,
                                      positive = "1")
cm.bagging.model.final <- confusionMatrix(data = bagging.pred.final, df[test,]$Exited,
                                      positive = "1")
boosting.pred.final <- ifelse(boosting.pred.final >= .5, 1, 0)
cm.boosting.model.final <- confusionMatrix(data = factor(boosting.pred.final), df[test,]$Exited,
                                      positive = "1")

which.max(c(cm.tree.model.final$byClass["Sensitivity"],
            cm.forest.model.final$byClass["Sensitivity"],
            cm.bagging.model.final$byClass["Sensitivity"],
            cm.boosting.model.final$byClass["Sensitivity"]))
#tree.model.2 chosen for interpretation
cm.tree.model.final$byClass["Sensitivity"]
rpart.plot(tree.model.2, box.palette="RdBu", shadow.col="gray", nn=TRUE)

