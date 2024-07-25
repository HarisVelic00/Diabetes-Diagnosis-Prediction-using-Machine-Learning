import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc

# Učitavanje podataka iz Excel datoteke
excel_file = 'C:/Users/Haris/Desktop/Diabetes-Diagnosis-Prediction-using-Machine-Learning/Data.xlsx'
df = pd.read_excel(excel_file)
print(df.head(300))
# Podjela podataka na ulazne (X) i izlazne (y) varijable
X = df.drop('Dijabetes', axis=1)  # Ulazne varijable (sve osim ciljne varijable 'Dijabetes')
y = df['Dijabetes']  # Izlazna varijabla 'Dijabetes'

# Podjela podataka na skup za treniranje i skup za testiranje (80% za treniranje, 20% za testiranje)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicijalizacija modela (SVM, logistička regresija, KNN)
svm_model = SVC()
logreg_model = LogisticRegression()
knn_model = KNeighborsClassifier()

# Treniranje modela
svm_model.fit(X_train, y_train)
logreg_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Evaluacija modela
models = {
    'SVM': svm_model,
    'Logistic Regression': logreg_model,
    'KNN': knn_model
}

# Izvještaji o evaluaciji modela
for name, model in models.items():
    print(f"Evaluation results for {name}:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print()

# Priprema za vizualizaciju rezultata
plt.figure(figsize=(15, 10))

# 1. Matrice konfuzije kao heatmap
for i, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.subplot(2, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

# 2. Preciznost i odziv (Precision-Recall Curve)
plt.subplot(2, 3, 4)
for name, model in models.items():
    y_scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=name, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)

# 3. ROC krivulja (Receiver Operating Characteristic)
plt.subplot(2, 3, 5)
for name, model in models.items():
    y_scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)

# Spremanje i prikaz grafova
plt.tight_layout()
plt.show()
