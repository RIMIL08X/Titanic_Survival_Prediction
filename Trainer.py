import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("TITANIC DATASET - ML MODEL TRAINING")
print("=" * 80)

# Hard-coded CSV path
CSV_PATH = "/home/rimil0bx/Documents/My Projects/NextGenAI/Dataset/Titanic-Dataset.csv"

# Load dataset
print(f"\n[1/7] Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Data preprocessing
print("\n[2/7] Preprocessing data...")

# Display missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df.drop(['Cabin'], axis=1, inplace=True)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X = df[features]
y = df['Survived']

print(f"\nFeatures used: {features}")
print(f"Target variable: Survived")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create directory structure for models
print("\n[3/7] Creating model directories...")
base_dir = "dumped_models"
os.makedirs(base_dir, exist_ok=True)

models_info = {
    'decision_tree': {
        'name': 'Decision Tree',
        'model': DecisionTreeClassifier(random_state=42, max_depth=5),
        'scaled': False
    },
    'random_forest': {
        'name': 'Random Forest',
        'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
        'scaled': False
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'model': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'scaled': False
    },
    'svc': {
        'name': 'Support Vector Classifier',
        'model': SVC(kernel='rbf', random_state=42, probability=True),
        'scaled': True
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'scaled': True
    }
}

for model_key in models_info.keys():
    model_dir = os.path.join(base_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)

# Train models
print("\n[4/7] Training models...")
results = {}

for model_key, model_info in models_info.items():
    print(f"\nTraining {model_info['name']}...")
    
    # Select appropriate data (scaled or not)
    if model_info['scaled']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train model
    model = model_info['model']
    model.fit(X_train_use, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_use)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[model_key] = {
        'name': model_info['name'],
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }
    
    # Save model and scaler
    model_dir = os.path.join(base_dir, model_key)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    if model_info['scaled']:
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # Save metadata
    metadata = {
        'model_name': model_info['name'],
        'accuracy': accuracy,
        'features': features,
        'scaled': model_info['scaled']
    }
    joblib.dump(metadata, os.path.join(model_dir, 'metadata.pkl'))
    
    print(f"✓ {model_info['name']} - Accuracy: {accuracy:.4f}")
    print(f"  Model saved to: {model_dir}/")

# Display results
print("\n[5/7] Model Comparison")
print("=" * 80)
print(f"{'Model':<30} {'Accuracy':<15}")
print("-" * 80)
for model_key, result in results.items():
    print(f"{result['name']:<30} {result['accuracy']:<15.4f}")
print("=" * 80)

# Visualizations
print("\n[6/7] Generating visualizations...")

# 1. Accuracy Comparison Bar Plot
plt.figure(figsize=(12, 6))
model_names = [results[k]['name'] for k in results.keys()]
accuracies = [results[k]['accuracy'] for k in results.keys()]

colors = sns.color_palette("husl", len(model_names))
bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.ylim([0.7, 1.0])
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Accuracy comparison saved")

# 2. Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (model_key, result) in enumerate(results.items()):
    cm = result['confusion_matrix']
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                square=True, linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.4f}", 
                        fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
    axes[idx].set_xticklabels(['Not Survived', 'Survived'])
    axes[idx].set_yticklabels(['Not Survived', 'Survived'])

# Hide the last subplot
axes[-1].axis('off')

plt.suptitle('Confusion Matrices - All Models', fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved")

# 3. Feature Importance (for tree-based models)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

tree_models = ['decision_tree', 'random_forest', 'gradient_boosting']
for idx, model_key in enumerate(tree_models):
    model = results[model_key]['model']
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    axes[idx].barh(range(len(features)), importances[indices], color=sns.color_palette("viridis", len(features)))
    axes[idx].set_yticks(range(len(features)))
    axes[idx].set_yticklabels([features[i] for i in indices])
    axes[idx].set_xlabel('Importance', fontweight='bold')
    axes[idx].set_title(f"{results[model_key]['name']}\nFeature Importance", fontweight='bold')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved")

# 4. Data Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Survival rate
survival_counts = df['Survived'].value_counts()
axes[0, 0].pie(survival_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%',
               colors=['#ff6b6b', '#51cf66'], startangle=90, explode=[0.05, 0.05])
axes[0, 0].set_title('Overall Survival Rate', fontweight='bold', fontsize=13)

# Gender vs Survival
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=axes[0, 1], 
                                             color=['#ff6b6b', '#51cf66'], alpha=0.8)
axes[0, 1].set_title('Survival by Gender', fontweight='bold', fontsize=13)
axes[0, 1].set_xlabel('Gender (0=Female, 1=Male)', fontweight='bold')
axes[0, 1].set_ylabel('Count', fontweight='bold')
axes[0, 1].set_xticklabels(['Female', 'Male'], rotation=0)
axes[0, 1].legend(['Not Survived', 'Survived'])

# Age distribution
axes[1, 0].hist([df[df['Survived']==0]['Age'], df[df['Survived']==1]['Age']], 
                bins=20, label=['Not Survived', 'Survived'], color=['#ff6b6b', '#51cf66'], alpha=0.7)
axes[1, 0].set_title('Age Distribution by Survival', fontweight='bold', fontsize=13)
axes[1, 0].set_xlabel('Age', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].legend()

# Class vs Survival
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=axes[1, 1],
                                                color=['#ff6b6b', '#51cf66'], alpha=0.8)
axes[1, 1].set_title('Survival by Passenger Class', fontweight='bold', fontsize=13)
axes[1, 1].set_xlabel('Passenger Class', fontweight='bold')
axes[1, 1].set_ylabel('Count', fontweight='bold')
axes[1, 1].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
axes[1, 1].legend(['Not Survived', 'Survived'])

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'data_analysis.png'), dpi=300, bbox_inches='tight')
print("✓ Data analysis plots saved")

# Final Summary
print("\n[7/7] Training Complete!")
print("=" * 80)
print(f"\n✓ All 5 models trained and saved to '{base_dir}/' directory")
print(f"✓ Each model stored in separate folder with:")
print(f"  - model.pkl (trained model)")
print(f"  - metadata.pkl (model information)")
print(f"  - scaler.pkl (for scaled models)")
print(f"\n✓ Visualizations saved:")
print(f"  - accuracy_comparison.png")
print(f"  - confusion_matrices.png")
print(f"  - feature_importance.png")
print(f"  - data_analysis.png")

print("\n" + "=" * 80)
print("BEST MODEL:")
best_model_key = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_model = results[best_model_key]
print(f"  {best_model['name']} with accuracy: {best_model['accuracy']:.4f}")
print("=" * 80)

print("\nAll models are ready for deployment!")
print("To load a model: joblib.load('dumped_models/<model_name>/model.pkl')")
print("\nScript completed successfully! ✓")