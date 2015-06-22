# Machine-Learning-Prediction-Project
Coursera ML Course Project
Machine-Learning-Prediction-Project Coursera ML Course Project

library(dplyr)
library(caret)
library(randomForest)
We start by reading the data for test and training sets and activity labels

harTrainSet <- read.csv("pml-training.csv")
harTestSet <- read.csv("pml-testing.csv")
There is a very long list of predictors and we select only the best predictors in the training and test sets To do so, we weed out poor predictors which exhibit little or no variance (i.e. close to being constants)

nearZero <- nearZeroVar(harTrainSet, saveMetrics = TRUE)
We identify and removepredictors that are flagges as near zero

nearZero$nzv
Finally, we remove predictors which intuitively will not correlate with te outcome such as the name , time of day etc. This leaves us with the following list of better predictors

"roll_belt"            "pitch_belt"           "yaw_belt"          
"total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
"accel_belt_y"         "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
"roll_arm"             "pitch_arm"            "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
"gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
"magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
"yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
"accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
"magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"        "yaw_forearm"          "gyros_forearm_x"     
"gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
"magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"
We can now subset the training and test sets with this more concise list of predictors

harTrainSet_clean <- select(harTrainSet, roll_belt,pitch_belt,yaw_belt,total_accel_belt,gyros_belt_x,gyros_belt_y,gyros_belt_z,accel_belt_x,accel_belt_y,accel_belt_z,magnet_belt_x,magnet_belt_y,magnet_belt_z,roll_arm,pitch_arm,yaw_arm,total_accel_arm,gyros_arm_x,gyros_arm_y,gyros_arm_z,accel_arm_x,accel_arm_y,accel_arm_z,magnet_arm_x,magnet_arm_y,magnet_arm_z,roll_dumbbell,pitch_dumbbell,yaw_dumbbell,total_accel_dumbbell,gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z,accel_dumbbell_x,accel_dumbbell_y,accel_dumbbell_z,magnet_dumbbell_x,magnet_dumbbell_y,magnet_dumbbell_z,roll_forearm,pitch_forearm,yaw_forearm,gyros_forearm_x,gyros_forearm_y,gyros_forearm_z,accel_forearm_x,accel_forearm_y,accel_forearm_z,magnet_forearm_x, magnet_forearm_y, magnet_forearm_z,classe)
harTestSet_clean <- select(harTestSet, roll_belt,pitch_belt,yaw_belt,total_accel_belt,gyros_belt_x,gyros_belt_y,gyros_belt_z,accel_belt_x,accel_belt_y,accel_belt_z,magnet_belt_x,magnet_belt_y,magnet_belt_z,roll_arm,pitch_arm,yaw_arm,total_accel_arm,gyros_arm_x,gyros_arm_y,gyros_arm_z,accel_arm_x,accel_arm_y,accel_arm_z,magnet_arm_x,magnet_arm_y,magnet_arm_z,roll_dumbbell,pitch_dumbbell,yaw_dumbbell,total_accel_dumbbell,gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z,accel_dumbbell_x,accel_dumbbell_y,accel_dumbbell_z,magnet_dumbbell_x,magnet_dumbbell_y,magnet_dumbbell_z,roll_forearm,pitch_forearm,yaw_forearm,gyros_forearm_x,gyros_forearm_y,gyros_forearm_z,accel_forearm_x,accel_forearm_y,accel_forearm_z,magnet_forearm_x, magnet_forearm_y, magnet_forearm_z)
We create a training and test set from the original training set file (70% training, 30% test)

inTrain = createDataPartition(y=harTrainSet_clean$classe, p = 0.7, list=FALSE)
training = harTrainSet_clean[inTrain, ]
testing = harTrainSet_clean[-inTrain, ]
We set the variable y to be a factor variable in both the training set.

training$classe <- factor(training$classe)
We now try different marchine learning algorithms and check their accuracy. Note: I could not make the glm training model work so I tried RPART and RF (Something is wrong; all the Accuracy metric values are missing)

Use the Recursive Partitioning and Regression Trees methodlogy to predict the class outcome (classe)

model<-train(Class ~ ., data = training, method = "rpart")
Test the Recursive Partitioning and Regression Trees model against the trainig set (30% of the orginal training file)

predictions <- predict(model, newdata = testing)
Review the predictions and level or accuracy of the model
c1 <- confusionMatrix(predictions, testing$classe)
c1
Output of the model shows a 49% accuracy - see below: Confusion Matrix and Statistics

 Reference
 Prediction    A    B    C    D    E
          A 1518  449  460  399  159
          B   29  386   34  184  149
          C  122  304  532  381  297
          D    0    0    0    0    0
          E    5    0    0    0  477

 Overall Statistics

 Accuracy : 0.495           
 95% CI : (0.4821, 0.5078)
Cross check: Try now the random forest methodlogy to predict the class outcome (classe)

model <- train(training$classe~., 
               data=training, 
               method = "rf",
               trControl = trainControl(method = "oob"))
Test the Random Fores model against the trainig set (30% of the orginal training file)

predictions <- predict(model, newdata = testing)
Review the predictions and level or accuracy of the model

c1 <- confusionMatrix(predictions, testing$classe)
c1
Output of the model shows a 99% accuracy (likely optimistic due to outfitting) - see below: Confusion Matrix and Statistics

Reference
Prediction    A    B    C    D    E
          A 1673   12    0    0    0
          B    0 1124    3    0    0
          C    0    3 1018   10    1
          D    0    0    5  953    5
          E    1    0    0    1 1076

 Overall Statistics

 Accuracy : 0.993          
 95% CI : (0.9906, 0.995)   
Finally, we use run the highest accuracy model (99%) - the Random Forest model - against the real test data set (20)

predict(model, newdata = harTestSet_clean)
The model returns the following predictions against the 20 test cases

[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
