# User Purchase Classification with PCA and Machine Learning

## 📄 Description  
This project, part of the 365 Data Science program, builds a classification model to predict whether a user will make a purchase based on behavioral data. 
The pipeline includes data cleaning & outlier removal, multicollinearity check with Variance Inflation Factor (VIF), dimensionality reduction using Principal Component Analysis (PCA), encoding categorical variables and scaling, and training and evaluating multiple machine learning models.
Models included: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Random Forest. Model performance is assessed via accuracy, precision, recall, and F1-score.

## 🚀 Features
- Outlier detection and removal with visualization  
- KDE plots for understanding feature distributions  
- VIF analysis to detect and reduce multicollinearity  
- PCA for dimensionality reduction and explained variance visualization  
- Encoding and scaling pipeline for categorical and numerical features  
- Hyperparameter tuning using GridSearchCV  
- Performance evaluation with confusion matrix and classification report  
- Decision tree visualization for interpretability  

## 📊 Machine Learning Models Used

| Model                | Hyperparameters Tuned             |
|----------------------|---------------------------------|
| Logistic Regression   | Statsmodels with constant term   |
| K-Nearest Neighbors   | Number of neighbors, weights     |
| Support Vector Machine| C, kernel type, gamma            |
| Decision Tree        | Cost complexity pruning (ccp_alpha) |
| Random Forest        | ccp_alpha, random_state          |

## 🛠️ Technologies
- Python 3.10+  
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, statsmodels  
- Development environment: Jupyter Notebook / VSCode  

## 🧪 Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  
- Model interpretability via tree plots  
