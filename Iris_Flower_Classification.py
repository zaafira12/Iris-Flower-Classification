import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
columns=['Sepal length','Sepal width','Petal length','Petal width','Species']
df=pd.read_csv('Iris.csv',names=columns)
print(df)

df.describe()
import matplotlib.pyplot as plt

# Extract data
sepal_length = df["Sepal length"]

# Plot histogram and capture Axes object
ax = df['Sepal length'].hist(bins=20, edgecolor='black', figsize=(8,5), grid=False, alpha=1)
plt.xticks(rotation=45) 
# Define designer colors
designer_colors = [
    "#DDBEA9", "#B7B7A4", "#A5A58D", "#6D6875", "#E5989B",
    "#B5838D", "#ADC178", "#FFBCBC", "#A3B18A", "#9A8C98"
]

# Access the patches (bars) from the Axes object
patches = ax.patches

# Repeat colors if there are more bars than colors
designer_colors = designer_colors * (len(patches) // len(designer_colors) + 1)

# Assign each patch (bar) a unique color
for patch, color in zip(patches, designer_colors):
    patch.set_facecolor(color)

# Beautify the plot
plt.xlabel("Sepal length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Sepal length ", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=1)
plt.tight_layout()
plt.show()

sepal_width = df["Sepal width"]
ax = df['Sepal width'].hist(bins=20, edgecolor='black', figsize=(8,5), grid=False, alpha=1)
plt.xticks(rotation=45) 

# Apply designer colors
designer_colors = [
    "#CAD2C5", "#84A98C", "#52796F", "#354F52", "#A4C3B2",
    "#CCE3DE", "#BFD8BD", "#99C1B9", "#C9CBA3", "#8E9A9B"
]
patches = ax.patches
# Repeat if bins exceed colors
designer_colors = designer_colors * (len(patches) // len(designer_colors) + 1)

# Assign each bin a unique, attractive color
for patch, color in zip(patches, designer_colors):
    patch.set_facecolor(color)

# Beautify plot
plt.xlabel("Sepal width", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title(" Histogram of Sepal width", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

petal_length = df["Petal length"]
ax = df['Petal length'].hist(bins=20, edgecolor='black', figsize=(8,5), grid=False, alpha=1)
plt.xticks(rotation=45) 
# Apply designer colors
designer_colors = ["#9A8C98", "#C9ADA7", "#F2E9E4", "#B5B2B4", "#A99985",
 "#D8C3A5", "#EAE0D5", "#B4A69F", "#C2B9B0", "#ADAAA6"]
patches = ax.patches
# Repeat if bins exceed colors
designer_colors = designer_colors * (len(patches) // len(designer_colors) + 1)
# Assign each bin a unique, attractive color
for patch, color in zip(patches, designer_colors):
    patch.set_facecolor(color)

# Beautify plot
plt.xlabel("Petal length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Petal length", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

petal_width = df["Petal width"]
ax = df['Petal width'].hist(bins=20, edgecolor='black', figsize=(8,5), grid=False, alpha=1)
plt.xticks(rotation=45) 
# Apply designer colors
designer_colors = [
    "#A0C4FF", "#BDB2FF", "#CDB4DB", "#9BF6FF", "#B5EAD7",
    "#AFCBFF", "#D0E8F2", "#C3CDE6", "#BFD7EA", "#ACCBE1"

]
patches = ax.patches
# Repeat if bins exceed colors
designer_colors = designer_colors * (len(patches) // len(designer_colors) + 1)

# Assign each bin a unique, attractive color
for patch, color in zip(patches, designer_colors):
    patch.set_facecolor(color)

# Beautify plot
plt.xlabel("Petal width", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Petal width", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

for col in['Sepal length', 'Sepal width', 'Petal length', 'Petal width']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
sns.pairplot(df,hue='Species')
plt.show()
data =df.values
X = data[:,0:4]
Y = data[:,4]
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)


labels = ['Training Set (80%)', 'Testing Set (20%)']
sizes = [80, 20]  # You can also use actual counts if preferred
colors = ['#FFC4C4', '#B2C8BA']  # Soft nude colors
explode = (0.05, 0.05)  # Slight separation between slices

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%',
        startangle=90, shadow=True, textprops={'fontsize': 12})
plt.title('Train-Test Split of Dataset', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

df.columns = df.columns.str.strip()
df_numeric = df.select_dtypes(include=['number'])  # Select only numeric columns
# Compute the correlation matrix
correlation_matrix = df_numeric.corr()
# Print correlation values
print(correlation_matrix)
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8,6))  # Set figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Iris Dataset")
plt.show()

# Using SVM  algorithm
print(df['Species'].isnull().sum())  # Check missing values
df = df.dropna(subset=['Species']) 
print(df)

from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
model_svc = SVC()
model_svc.fit(X_train_imputed,y_train)

prediction1 = model_svc.predict(X_test_imputed)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction1)*100)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction1))
# LOGISTIC REGRESSION
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test) 
model_LR=LogisticRegression()
model_LR.fit(X_train_imputed,y_train)
prediction2 = model_LR.predict(X_test_imputed)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction2)*100)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction2))

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier 
from sklearn.impute import SimpleImputer
from sklearn import metrics
model = DecisionTreeClassifier()
model.fit(X_train_imputed,y_train)
print(model)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test) 
prediction3 = model.predict(X_test_imputed)
print(accuracy_score(y_test,prediction3)*100)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction3))
#KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test) 
knn.fit(X_train_imputed,y_train)
prediction4 = knn.predict(X_test_imputed)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction4)*100)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction4))
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Plot confusion matrices
plt.figure(figsize=(12, 10))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    plt.subplot(2, 2, i)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')
    plt.title(f"Confusion Matrix – {name}")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png")  # Optional: Save the figure
plt.show()

# Example model names and their accuracies
models = [ 'SVM', 'Logistic Regression','Decision Tree','KNN']
accuracies = [0.97, 0.97, 0.93, 0.96]  # Replace these with your actual values
# Choose attractive, balanced colors
colors = ['#F2B5D4', '#A2D5F2', '#D5E1DF', '#F9D5E5']
# Plotting a horizontal bar chart
plt.figure(figsize=(8, 5))
plt.barh(models, accuracies, color=colors)
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1.05)  # Accuracy is between 0 and 1
for i, v in enumerate(accuracies):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)

plt.tight_layout()
plt.show()

X_new = np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.3,4.6,1.9]])
p = model_svc.predict(X_new)ā
print("Prediction of Species:{}".format(p))