# Bike Rental Demand Prediction - Final Model Evaluation Script

# Load Required Libraries
library(dplyr)
library(lubridate)
library(rpart)
library(rpart.plot)
library(randomForest)
library(nnet)
library(caret)
library(ggcorrplot)

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
df = read.csv("/Users/hp/Desktop/phase-2/bike.csv")
# Load required libraries
# Drop columns not available at prediction time
df$casual <- NULL
df$registered <- NULL
df$datetime <- NULL

# Convert to appropriate data types
df$season      <- as.factor(df$season)
df$holiday     <- as.factor(as.numeric(df$holiday))
df$workingday  <- as.factor(as.numeric(df$workingday))
df$weather     <- as.factor(df$weather)
df$month       <- factor(df$month, levels = month.name, ordered = TRUE)
df$day         <- as.factor(df$day)
df$year        <- as.factor(df$year)
df$hour        <- as.factor(df$hour)
df$humidity    <- as.numeric(df$humidity)
df$count       <- as.numeric(df$count)

# -----------------------------
# 1.1 Data Cleaning
# -----------------------------
# Check for missing values
missing_values <- colSums(is.na(df))
print("Missing values in each column:")
print(missing_values)

# Drop rows with NA (if any)
df <- na.omit(df)

# Optional: Outlier visualization
boxplot(df$count, main = "Boxplot of Bike Rental Count", col = "lightblue")

# Scale numeric features
numeric_vars <- c("temp", "atemp", "humidity", "windspeed")
df[numeric_vars] <- scale(df[numeric_vars])


# -----------------------------
# 1.3 Data Visualization
# -----------------------------

# GRAPH-1 Histogram of count
hist(df$count,
     breaks = 30,
     col = "skyblue",
     main = "Distribution of Bike Rentals",
     xlab = "Total Rentals",
     ylab = "Frequency")



# GRAPH-2 Rentals by Hour of Day (converted from factor to numeric for plotting)
df$hour_num <- as.numeric(as.character(df$hour))
hourly_avg <- aggregate(count ~ hour_num, data = df, FUN = mean)

plot(hourly_avg$hour_num,
     hourly_avg$count,
     type = "o",
     col = "darkgreen",
     xlab = "Hour of Day",
     ylab = "Average Bike Rentals",
     main = "Average Bike Rentals by Hour of Day",
     xaxt = "n")
axis(1, at = 0:23)



# -----------------------------
# 2. Split Data into Train/Test
# -----------------------------
set.seed(1234)
N <- nrow(df)
trainingSize <- round(N * 0.6)
trainingCases <- sample(N, trainingSize)
training <- df[trainingCases, ]
test     <- df[-trainingCases, ]

# -----------------------------
# 3. Model 1: Linear Regression
# -----------------------------
model_lm <- lm(count ~ ., data = training)
model_lm <- step(model_lm)
predictions_lm <- predict(model_lm, test)
errors_lm <- test$count - predictions_lm
mape_lm <- mean(abs(errors_lm / test$count))
rmse_lm <- sqrt(mean(errors_lm^2))

# -----------------------------
# 4. Model 2: Decision Tree
# -----------------------------
model_tree <- rpart(count ~ ., data = training)
rpart.plot(model_tree)
predictions_tree <- predict(model_tree, test)
errors_tree <- test$count - predictions_tree
mape_tree <- mean(abs(errors_tree / test$count))
rmse_tree <- sqrt(mean(errors_tree^2))

# -----------------------------
# 5. Model 3: Pruned Tree
# -----------------------------
stoppingRules <- rpart.control(minsplit = 2, minbucket = 1, cp = -1)
model_unpruned <- rpart(count ~ ., data = training, control = stoppingRules)
cp_best <- model_unpruned$cptable[which.min(model_unpruned$cptable[, "xerror"]), "CP"]
pruned <- prune(model_unpruned, cp = cp_best)
pred_pruned <- predict(pruned, test)
errors_pruned <- test$count - pred_pruned
mape_pruned <- mean(abs(errors_pruned / test$count))
rmse_pruned <- sqrt(mean(errors_pruned^2))

# -----------------------------
# 6. Model 4: Random Forest
# -----------------------------
model_rf <- randomForest(count ~ ., data = training, ntree = 500)
predictions_rf <- predict(model_rf, test)
errors_rf <- test$count - predictions_rf
mape_rf <- mean(abs(errors_rf / test$count))
rmse_rf <- sqrt(mean(errors_rf^2))

#------------------------------------
# GBM
#------------------------------------
# BUILD-PREDICT-EVALUATE: BOOST
library(gbm)
training$count = as.numeric(training$count)
model_boost = gbm(count ~ ., data=training, n.trees=49,cv.folds=4)
training$count = as.numeric(training$count)

best_size = gbm.perf(model_boost,method="cv")

predictions_boost = predict(model_boost, test, best_size, type="response")
errors_boost <- test$count - predictions_boost
mape_boost <- mean(abs(errors_boost / test$count))
rmse_boost <- sqrt(mean(errors_boost^2))
# We are using Neural nets since the error rate with GBM is significantly higher

# -----------------------------
# 7. Stacked Model (Neural Net)
# -----------------------------
# Create base model predictions
base_preds <- data.frame(
  rf_pred     = predict(model_rf, df),
  tree_pred   = predict(model_tree, df),
  pruned_pred = predict(pruned, df),
  lm_pred     = predict(model_lm, df)
)

# Combine predictions with original count
df_stack <- cbind(count = df$count, base_preds)

# Split stacked data
training_stack <- df_stack[trainingCases, ]
test_stack     <- df_stack[-trainingCases, ]

# Preprocess predictors only
predictor_names <- setdiff(names(training_stack), "count")
preproc <- preProcess(training_stack[, predictor_names], method = c("center", "scale"))
train_scaled <- predict(preproc, training_stack[, predictor_names])
test_scaled  <- predict(preproc, test_stack[, predictor_names])

# Recombine with unscaled target
training_stack_std <- cbind(count = training_stack$count, train_scaled)
test_stack_std     <- cbind(count = test_stack$count, test_scaled)

# Train neural network meta-model
model_stack <- nnet(count ~ ., data = training_stack_std, size = 6, linout = TRUE, trace = FALSE)

# Evaluate stacked model
predictions_stack <- predict(model_stack, test_stack_std)
errors_stack <- test_stack_std$count - predictions_stack
mape_stack <- mean(abs(errors_stack / test_stack_std$count))
rmse_stack <- sqrt(mean(errors_stack^2))

# -----------------------------
# 8. Model Performance Summary
# -----------------------------
cat("\nModel Performance Comparison:\n")
cat("---------------------------------\n")
cat("Linear Regression:\n")
cat("  MAPE:", round(mape_lm, 4), " RMSE:", round(rmse_lm, 2), "\n\n")

cat("Decision Tree:\n")
cat("  MAPE:", round(mape_tree, 4), " RMSE:", round(rmse_tree, 2), "\n\n")

cat("Pruned Decision Tree:\n")
cat("  MAPE:", round(mape_pruned, 4), " RMSE:", round(rmse_pruned, 2), "\n\n")

cat("Random Forest:\n")
cat("  MAPE:", round(mape_rf, 4), " RMSE:", round(rmse_rf, 2), "\n\n")

cat("GBM:\n")
cat("  MAPE:", round(mape_boost, 4), " RMSE:", round(rmse_boost, 2), "\n\n")

cat("Stacked Model (Neural Net):\n")
cat("  MAPE:", round(mape_stack, 4), " RMSE:", round(rmse_stack, 2), "\n")
cat("---------------------------------\n")
