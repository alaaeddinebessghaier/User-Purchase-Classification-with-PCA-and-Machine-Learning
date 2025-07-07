import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv("C:\\Users\\bessghaier\\365 project\\project-files-machine-learning-for-user-classification\\ml_datasource.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum)
        
        
    ###data exploration


cols = data.select_dtypes("number").columns
n_cols = 3
n_rows = (len(cols)+n_cols-1 )//n_cols
fig ,axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
## Pour faciliter la boucle for, tu veux transformer cette matrice 2D en liste 1D.
axes = axes.flatten()
for ax , col in zip(axes,cols):
    sns.kdeplot(data[col],ax=ax, shade= True,color="blue")
    ax.set_title(col)
for j in range(len(cols),len(axes)):
    fig.delaxes(axes[j])
## Ajuste automatiquement les marges et les espaces entre les sous-graphes pour que tout soit lisible et propre.
plt.tight_layout()
plt.show()

### Removing Outliers
data_filtered = data[
    (data['minutes_watched'] <= 1000) &
    (data['courses_started'] <= 10) &
    (data['practice_exams_started'] <= 10) &
    (data['minutes_spent_on_exams'] <= 40)
]

### Checking for Multicollinearity
data_copy = data_filtered.copy().select_dtypes("number")
data_copy = data_copy.drop('purchased',axis=1)
vif_data = pd.DataFrame()
vif_data['features'] = data_copy.columns
vif_data['VIF'] =  [variance_inflation_factor(data_copy.values, i) for i in range(data_copy.shape[1])]
print(vif_data)

scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(data_copy)
pca = PCA()
x_pca = pca.fit_transform(scaled_data)
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
plt.plot(range(1,len(ratio)+1),cum,marker='o' )
plt.title("explained variance ratio ")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.show()  
pca = PCA(n_components=3)
x_pca = pca.fit_transform(scaled_data)
x_pca_df = pd.DataFrame(x_pca, columns=[f'PC{i+1}' for i in range(x_pca.shape[1])])


pca_vif = pd.DataFrame()
pca_vif['features'] = x_pca_df.columns
pca_vif['VIF'] =  [variance_inflation_factor(x_pca_df.values, i) for i in range(x_pca_df.shape[1])]
print(pca_vif)
print(x_pca_df.columns)




### dealing with NAN 
x_pca_df['purchased'] = data_filtered['purchased'].values
x_pca_df['student_country'] = data_filtered['student_country'].values
x_pca_df['student_country'] = x_pca_df['student_country'].fillna('NAM')

##splitting the data
x= x_pca_df.drop('purchased',axis=1)
y = x_pca_df['purchased']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=365,stratify = y)
## encoding the data 

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=170)
x_train['student_country_enc'] = encoder.fit_transform(x_train[['student_country']])
x_test['student_country_enc'] = encoder.transform(x_test[['student_country']])
x_train.drop(columns=['student_country'], inplace=True)
x_test.drop(columns=['student_country'], inplace=True)

x_train['student_country_enc'] = scaler.fit_transform(x_train[['student_country_enc']])
x_test['student_country_enc'] = scaler.transform(x_test[['student_country_enc']])

x_train_array = np.asarray(x_train, dtype='float')
y_train_array = np.asarray(y_train, dtype='int')
x_test_array = np.asarray(x_test, dtype='float')  # Fix: Should be 'float' for features
y_test_array = np.asarray(y_test, dtype='int')

## logistic regression model
x_train_const = sm.add_constant(x_train_array)
model = sm.Logit(y_train_array, x_train_const)
result = model.fit()
print(result.summary())



## K-nearst Neighbor (KNN)
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors':range(1,51),
    'weights' : ['uniform','distance']
    }
grid = GridSearchCV(knn,param_grid,scoring='accuracy')
grid.fit(x_train_array,y_train)
print(grid.best_estimator_,grid.best_score_)
best_knn = grid.best_estimator_
test_best_knn = best_knn.predict(x_test_array)
#sns.reset_orig()
ConfusionMatrixDisplay.from_predictions(
    y_test_array, test_best_knn,
    labels = best_knn.classes_,
    cmap = 'magma' 
);
plt.show()
print(classification_report(y_test_array, 
                            test_best_knn, 
                           target_names = ['0', '1']))

### Support vector machine
model_svc = SVC()
param_grid_svc = {
    'C': list(range(1, 11)),               # 1 to 10 inclusive
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}
minmax= MinMaxScaler(feature_range=(-1,1))
x_train_svc = minmax.fit_transform(x_train_array)
x_test_svc = minmax.transform(x_test_array)
grid_svc = GridSearchCV(model_svc,param_grid_svc,cv=5,scoring='accuracy')
grid_svc.fit(x_train_svc,y_train_array)
best_svc = grid_svc.best_estimator_
y_pred_svc = best_svc.predict(x_test_svc)

print("Best Parameters svc:", grid_svc.best_params_)
print("\nConfusion Matrix svc:\n", ConfusionMatrixDisplay.from_predictions(y_test_array, y_pred_svc))
print("\nClassification Report svc:\n", classification_report(y_test_array, y_pred_svc)) 


### desicion tree
param_grid_tree = {'ccp_alpha':[0,0.001,0.002,0.003,0.004,0.005]}
tree = DecisionTreeClassifier(random_state=365)
grid_tree = GridSearchCV(tree,param_grid_tree,cv=5,scoring='accuracy')
grid_tree.fit(x_train_array,y_train_array)
best_tree = grid_tree.best_estimator_
y_pred_tree = best_tree.predict(x_test_array)
print("Best ccp_alpha decision tree:", grid_tree.best_params_)
print("\nConfusion Matrix decision tree:\n", ConfusionMatrixDisplay.from_predictions(y_test_array, y_pred_tree))
print("\nClassification Report:decision tree\n", classification_report(y_test_array, y_pred_tree))
plt.figure(figsize=(15,10))

# Plot the decision tree. Feature names and class names are added for better interpretability
plot_tree(best_tree, 
          filled=True, 
          feature_names = list(x.columns), 
          class_names = ['Will not purchase', 
                         'Will purchase'])

# Display the plot
plt.show()


## Random_forest 
random_forest = RandomForestClassifier(ccp_alpha=0.001,random_state=365)
random_forest.fit(x_train_array,y_train_array)
y_pred_rf = random_forest.predict(x_test_array)
print("Random Forest Accuracy:", accuracy_score(y_test_array, y_pred_rf))
print(classification_report(y_test_array, y_pred_rf))
print("\nConfusion Matrix Random Forest:\n", ConfusionMatrixDisplay.from_predictions(y_test_array, y_pred_rf))
