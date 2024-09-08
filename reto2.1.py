import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

column_names = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                'stalk-surface-below-ring', 'stalk-color-above-ring', 
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 
                'ring-type', 'spore-print-color', 'population', 'habitat']

data = pd.read_csv('agaricus-lepiota.data', names=column_names)

print(data.head())

data = data.replace('?', pd.NA)

label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str))

imputer = SimpleImputer(strategy='median')
data_imputed = imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=column_names)

X = data_imputed.drop('target', axis=1)
y = data_imputed['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f'Modelo V2: {best_model}')

y_valid_pred_best = best_model.predict(X_valid)
valid_accuracy_best = accuracy_score(y_valid, y_valid_pred_best)
print(f'Validation Accuracy (Modelo V2): {valid_accuracy_best}')

y_test_pred_best = best_model.predict(X_test)
test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
print(f'Test Accuracy (Modelo V2): {test_accuracy_best}')

cm_best = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de Confusión (Modelo V2)')
plt.show()

from sklearn import tree

plt.figure(figsize=(20,15))
tree.plot_tree(best_model, 
               filled=True, 
               feature_names=X.columns, 
               class_names=['Edible', 'Poisonous'])
plt.show()
