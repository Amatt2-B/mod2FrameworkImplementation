# Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

column_names = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                'stalk-surface-below-ring', 'stalk-color-above-ring', 
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 
                'ring-type', 'spore-print-color', 'population', 'habitat']

data = pd.read_csv('agaricus-lepiota.data', names=column_names)

# Mostrar primeros datos del dataset
print(data.head)

# Filas que contienen un valor "?"
rows_with_missing = data[data.isin(['?']).any(axis=1)]

print(rows_with_missing)



data = data.replace('?', pd.NA)

label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str))

imputer = SimpleImputer(strategy='median')
data_imputed = imputer.fit_transform(data)

data_imputed = pd.DataFrame(data_imputed, columns=column_names)

X = data_imputed.drop('target', axis=1)
y = data_imputed['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn import tree

class_names = ['Edible', 'Poisonous']

plt.figure(figsize=(20,15))
tree.plot_tree(clf, 
               filled=True, 
               feature_names=X.columns, 
               class_names=class_names)
plt.show()
