"""
DataStory AI — RAG Knowledge Base
Plain-text data science concept documents for retrieval-augmented generation.
"""

DOCUMENTS = [
    {
        "id": "missing_values",
        "title": "Handling Missing Values",
        "content": (
            "Missing values occur when data is absent for a cell in your dataset. "
            "They can appear as NaN, None, or blank. "
            "Common strategies: fill numeric columns with mean or median, "
            "fill categorical columns with the most frequent value (mode), "
            "or drop rows/columns if too many values are missing. "
            "Choosing median over mean is safer when the column has outliers."
        ),
    },
    {
        "id": "duplicates",
        "title": "Handling Duplicate Rows",
        "content": (
            "Duplicate rows are identical records that appear more than once. "
            "They can happen due to data collection errors or merging datasets. "
            "Duplicates bias model training because the same example is counted multiple times. "
            "Always remove duplicates before training a model."
        ),
    },
    {
        "id": "train_test_split",
        "title": "Train-Test Split",
        "content": (
            "A train-test split divides your dataset into two parts: "
            "a training set used to teach the model, and a test set used to evaluate it. "
            "A typical split is 80% training and 20% testing. "
            "The test set must never be used during training — it simulates real unseen data."
        ),
    },
    {
        "id": "cross_validation",
        "title": "Cross-Validation",
        "content": (
            "Cross-validation is a technique to estimate model performance more reliably. "
            "In k-fold cross-validation, the training data is split into k equal parts. "
            "The model is trained k times, each time using a different part as a validation set. "
            "The average score across all folds is the cross-validation score."
        ),
    },
    {
        "id": "gridsearchcv",
        "title": "GridSearchCV",
        "content": (
            "GridSearchCV is a method to find the best hyperparameters for a model. "
            "It tries every combination of parameter values you define in a grid. "
            "For each combination it uses cross-validation to score the model. "
            "The combination with the best cross-validation score is selected as the best parameters. "
            "This prevents you from manually guessing hyperparameters."
        ),
    },
    {
        "id": "classification",
        "title": "Classification",
        "content": (
            "Classification is a type of ML task where the model predicts a category label. "
            "Examples: spam or not spam, churn or no churn, survived or not survived. "
            "The target column contains discrete classes, not continuous numbers. "
            "Common metrics are accuracy, precision, recall, F1-score, and ROC-AUC."
        ),
    },
    {
        "id": "regression",
        "title": "Regression",
        "content": (
            "Regression is a type of ML task where the model predicts a continuous number. "
            "Examples: house price, sales revenue, bike rental count. "
            "The target column contains numeric values across a wide range. "
            "Common metrics are MAE, RMSE, and R² score."
        ),
    },
    {
        "id": "class_imbalance",
        "title": "Class Imbalance",
        "content": (
            "Class imbalance occurs when one class has far more examples than another. "
            "Example: 95% of customers do not churn, only 5% do. "
            "In this case accuracy is misleading — a model that always predicts 'no churn' "
            "would be 95% accurate but completely useless. "
            "Better metrics for imbalanced data are F1-score, precision, and recall."
        ),
    },
    {
        "id": "smote",
        "title": "SMOTE — Synthetic Minority Oversampling",
        "content": (
            "SMOTE creates synthetic examples of the minority class by interpolating between "
            "existing minority examples. This balances the class distribution in the training set. "
            "SMOTE must only be applied after the train-test split and only to training data. "
            "Applying it before splitting would leak information from the test set into training."
        ),
    },
    {
        "id": "class_weight",
        "title": "Class Weights",
        "content": (
            "Class weights tell the model to pay more attention to the minority class during training. "
            "Setting class_weight='balanced' in scikit-learn models automatically adjusts weights "
            "inversely proportional to class frequencies. "
            "It is simpler than SMOTE and avoids creating synthetic data."
        ),
    },
    {
        "id": "precision",
        "title": "Precision",
        "content": (
            "Precision answers: of all predictions the model made for a class, how many were correct? "
            "Precision = True Positives / (True Positives + False Positives). "
            "High precision means the model rarely makes false positive predictions. "
            "Use precision when the cost of false positives is high — e.g. spam filters."
        ),
    },
    {
        "id": "recall",
        "title": "Recall (Sensitivity)",
        "content": (
            "Recall answers: of all actual instances of a class, how many did the model find? "
            "Recall = True Positives / (True Positives + False Negatives). "
            "High recall means the model catches most true positives. "
            "Use recall when the cost of missing a positive case is high — e.g. medical diagnosis."
        ),
    },
    {
        "id": "f1_score",
        "title": "F1-Score",
        "content": (
            "F1-score is the harmonic mean of precision and recall. "
            "F1 = 2 × (Precision × Recall) / (Precision + Recall). "
            "It balances the tradeoff between precision and recall. "
            "F1-score is the best single metric for imbalanced classification problems "
            "because it does not let accuracy hide poor minority class performance."
        ),
    },
    {
        "id": "roc_auc",
        "title": "ROC-AUC Score",
        "content": (
            "ROC-AUC stands for Receiver Operating Characteristic — Area Under the Curve. "
            "It measures how well the model separates classes across all threshold values. "
            "AUC = 0.5 means the model is no better than random guessing. "
            "AUC = 1.0 means perfect separation. "
            "AUC > 0.8 is generally considered good for most classification problems."
        ),
    },
    {
        "id": "mae",
        "title": "MAE — Mean Absolute Error",
        "content": (
            "MAE measures the average of absolute differences between predictions and actual values. "
            "It is easy to interpret because it is in the same unit as the target variable. "
            "MAE = mean(|actual - predicted|). "
            "Lower MAE means better predictions. It treats all errors equally."
        ),
    },
    {
        "id": "rmse",
        "title": "RMSE — Root Mean Squared Error",
        "content": (
            "RMSE is the square root of the average squared differences between predictions and actuals. "
            "RMSE penalises large errors more heavily than MAE. "
            "It is in the same unit as the target variable, making it interpretable. "
            "Lower RMSE means better model performance. "
            "RMSE is the most commonly used regression metric."
        ),
    },
    {
        "id": "r2",
        "title": "R² Score (Coefficient of Determination)",
        "content": (
            "R² measures how much of the variance in the target variable the model explains. "
            "R² = 1 means the model perfectly explains all variance. "
            "R² = 0 means the model does no better than predicting the mean. "
            "Negative R² means the model is worse than a naive mean predictor. "
            "Higher R² is better, but it should be interpreted alongside RMSE and MAE."
        ),
    },
    {
        "id": "overfitting",
        "title": "Overfitting",
        "content": (
            "Overfitting happens when a model learns the training data too well — including noise — "
            "and fails to generalise to new data. "
            "Signs: very high training accuracy but much lower test accuracy. "
            "Solutions: regularisation, pruning decision trees, reducing model complexity, "
            "using more data, or cross-validation for honest evaluation."
        ),
    },
    {
        "id": "underfitting",
        "title": "Underfitting",
        "content": (
            "Underfitting occurs when a model is too simple to capture patterns in the data. "
            "Signs: low accuracy on both training and test sets. "
            "Solutions: use a more complex model, add more features, "
            "reduce regularisation, or engineer better features."
        ),
    },
    {
        "id": "data_leakage",
        "title": "Data Leakage",
        "content": (
            "Data leakage happens when information from outside the training set "
            "is used to build the model. "
            "This leads to unrealistically high performance during training but poor results in production. "
            "Common causes: scaling or encoding before splitting, "
            "applying SMOTE before splitting, or including target-correlated features. "
            "Always split data first, then preprocess inside a pipeline."
        ),
    },
    {
        "id": "feature_importance",
        "title": "Feature Importance",
        "content": (
            "Feature importance measures how much each input column contributed to the model's predictions. "
            "Tree-based models (Random Forest, Decision Tree) provide built-in feature importances. "
            "High importance means the feature strongly influences the prediction. "
            "Feature importance helps identify which columns matter most and which can be dropped."
        ),
    },
    {
        "id": "model_comparison",
        "title": "Model Comparison",
        "content": (
            "Comparing multiple models helps find the best one for your data. "
            "Each model makes different assumptions about data distribution. "
            "Always compare models on the same test set using the same metric. "
            "The model with the best balance of performance and interpretability is usually preferred. "
            "GridSearchCV ensures each model is fairly tuned before comparison."
        ),
    },
]
