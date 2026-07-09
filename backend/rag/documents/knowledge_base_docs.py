"""
DataStory AI - RAG Knowledge Base Documents
Short reference texts about data science concepts.
"""

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "missing_values",
        "title": "Handling Missing Values",
        "content": (
            "Missing values occur when no data is recorded for a cell. "
            "They can be filled with the mean or median for numeric columns, or mode for categorical columns. "
            "Columns with more than 40-50% missing values are usually dropped. "
            "Never fill missing values using test data — only use training data statistics."
        ),
    },
    {
        "id": "duplicates",
        "title": "Handling Duplicate Rows",
        "content": (
            "Duplicate rows are exact copies of existing rows. "
            "They can skew model training by over-representing certain patterns. "
            "It is usually safe to remove duplicates before training."
        ),
    },
    {
        "id": "train_test_split",
        "title": "Train-Test Split",
        "content": (
            "A train-test split divides a dataset into a training set and a test set. "
            "The model learns from the training set. The test set is used only to evaluate performance. "
            "A common split is 80% training and 20% testing. "
            "Never let the model see test data during training."
        ),
    },
    {
        "id": "cross_validation",
        "title": "Cross-Validation",
        "content": (
            "Cross-validation evaluates a model more reliably by training and testing it on multiple subsets. "
            "K-fold cross-validation splits data into K parts, trains on K-1 parts, and tests on the remaining part. "
            "This is repeated K times. The average score gives a more reliable estimate of model performance."
        ),
    },
    {
        "id": "gridsearchcv",
        "title": "GridSearchCV",
        "content": (
            "GridSearchCV is a technique that tries every combination of hyperparameters you specify. "
            "It uses cross-validation to evaluate each combination. "
            "It returns the best parameters found. "
            "This is better than manually guessing parameters. "
            "It is computationally expensive for large grids."
        ),
    },
    {
        "id": "classification",
        "title": "Classification",
        "content": (
            "Classification is a supervised learning task where the model predicts a category. "
            "Examples: predicting if a customer will churn (yes/no), predicting a flower species. "
            "Common classification models include Logistic Regression, Decision Tree, Random Forest, SVM, and KNN."
        ),
    },
    {
        "id": "regression",
        "title": "Regression",
        "content": (
            "Regression is a supervised learning task where the model predicts a continuous number. "
            "Examples: predicting house prices, predicting student exam scores. "
            "Common regression models include Linear Regression, Ridge Regression, Random Forest Regressor, and Gradient Boosting."
        ),
    },
    {
        "id": "class_imbalance",
        "title": "Class Imbalance",
        "content": (
            "Class imbalance occurs when one class has far more samples than another. "
            "For example, 90% of customers did not churn and only 10% did. "
            "Accuracy alone is misleading for imbalanced datasets. "
            "A model that predicts the majority class for every row would have 90% accuracy but be useless. "
            "F1-score, precision, and recall are better metrics for imbalanced data."
        ),
    },
    {
        "id": "smote",
        "title": "SMOTE - Synthetic Minority Over-sampling Technique",
        "content": (
            "SMOTE creates synthetic samples of the minority class to balance the dataset. "
            "It works by finding nearest neighbors of minority samples and creating new samples along the line between them. "
            "SMOTE must only be applied to training data, never to the test set. "
            "Applying SMOTE before splitting causes data leakage and inflated scores."
        ),
    },
    {
        "id": "class_weights",
        "title": "Class Weights",
        "content": (
            "Using class_weight='balanced' tells the model to penalize mistakes on the minority class more heavily. "
            "This helps the model pay more attention to the smaller class. "
            "It is supported in Logistic Regression, Decision Tree, Random Forest, and SVM. "
            "It does not add new data — it just adjusts how the model learns."
        ),
    },
    {
        "id": "precision",
        "title": "Precision",
        "content": (
            "Precision measures how many of the predicted positives are actually positive. "
            "High precision means fewer false positives. "
            "Precision = True Positives / (True Positives + False Positives). "
            "Precision is important when false positives are costly, like spam detection."
        ),
    },
    {
        "id": "recall",
        "title": "Recall",
        "content": (
            "Recall measures how many of the actual positives were correctly identified. "
            "High recall means fewer false negatives. "
            "Recall = True Positives / (True Positives + False Negatives). "
            "Recall is important when missing a positive is costly, like disease detection."
        ),
    },
    {
        "id": "f1_score",
        "title": "F1-Score",
        "content": (
            "F1-score is the harmonic mean of precision and recall. "
            "It is a single metric that balances both precision and recall. "
            "F1 is especially useful for imbalanced datasets where accuracy alone is misleading. "
            "A high F1-score means the model is both precise and sensitive."
        ),
    },
    {
        "id": "roc_auc",
        "title": "ROC-AUC",
        "content": (
            "ROC-AUC stands for Receiver Operating Characteristic — Area Under Curve. "
            "It measures how well a model distinguishes between classes. "
            "A score of 0.5 means random guessing. A score of 1.0 is perfect. "
            "It is useful for binary classification problems."
        ),
    },
    {
        "id": "mae",
        "title": "Mean Absolute Error (MAE)",
        "content": (
            "MAE measures the average absolute difference between predicted and actual values. "
            "It is easy to interpret — a MAE of 5 means predictions are off by 5 units on average. "
            "Lower MAE is better. It is less sensitive to outliers than RMSE."
        ),
    },
    {
        "id": "rmse",
        "title": "Root Mean Squared Error (RMSE)",
        "content": (
            "RMSE measures the square root of the average squared differences between predictions and actuals. "
            "It penalizes large errors more than MAE. "
            "Lower RMSE is better. "
            "It is in the same unit as the target variable."
        ),
    },
    {
        "id": "r2",
        "title": "R² Score (Coefficient of Determination)",
        "content": (
            "R² measures how well the model explains the variance in the target variable. "
            "An R² of 1.0 means perfect predictions. An R² of 0 means the model is no better than predicting the mean. "
            "Negative R² means the model is worse than predicting the mean."
        ),
    },
    {
        "id": "overfitting",
        "title": "Overfitting",
        "content": (
            "Overfitting occurs when a model learns the training data too well, including noise and outliers. "
            "It performs well on training data but poorly on new, unseen data. "
            "Signs of overfitting include high training accuracy and low test accuracy. "
            "Regularization, simpler models, and cross-validation help prevent overfitting."
        ),
    },
    {
        "id": "underfitting",
        "title": "Underfitting",
        "content": (
            "Underfitting occurs when a model is too simple to capture patterns in the data. "
            "It performs poorly on both training and test data. "
            "Using more complex models, more features, or training longer can help."
        ),
    },
    {
        "id": "data_leakage",
        "title": "Data Leakage",
        "content": (
            "Data leakage occurs when information from outside the training set is used to train the model. "
            "This leads to unrealistically high scores that do not generalize. "
            "Common causes include applying SMOTE before splitting, scaling before splitting, or including target-related columns as features."
        ),
    },
    {
        "id": "feature_importance",
        "title": "Feature Importance",
        "content": (
            "Feature importance shows which columns most influenced the model's predictions. "
            "Tree-based models like Random Forest can calculate importance scores. "
            "High importance means the feature had a big impact on the model's decisions. "
            "It can be used to simplify models by removing low-importance features."
        ),
    },
    {
        "id": "model_comparison",
        "title": "Model Comparison",
        "content": (
            "Model comparison evaluates multiple trained models against each other using the same test set. "
            "For classification, F1-score or accuracy is used to rank models. "
            "For regression, RMSE or R² is used. "
            "The model with the best test score is selected as the final model."
        ),
    },
]
