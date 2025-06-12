import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Generate mixed synthetic dataset (more realistic)
np.random.seed(42)
n_samples = 200

# Create overlapping clusters
# Class 0: Low hue, high area (with some noise)
hues_0 = np.random.normal(20, 15, n_samples//2)
areas_0 = np.random.normal(450, 80, n_samples//2)

# Class 1: High hue, low area (with some noise) 
hues_1 = np.random.normal(100, 20, n_samples//2)
areas_1 = np.random.normal(350, 100, n_samples//2)

# Add some outliers to make it more challenging
outlier_indices_0 = np.random.choice(n_samples//2, 10, replace=False)
outlier_indices_1 = np.random.choice(n_samples//2, 10, replace=False)

hues_0[outlier_indices_0] = np.random.normal(80, 10, 10)  # Some class 0 with high hue
areas_1[outlier_indices_1] = np.random.normal(500, 20, 10)  # Some class 1 with high area

# Combine data
hues = np.concatenate([hues_0, hues_1])
areas = np.concatenate([areas_0, areas_1])
labels = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

# Create feature matrix
data = np.stack([hues, areas], axis=1)

# Shuffle the data to mix classes
shuffle_idx = np.random.permutation(len(data))
data = data[shuffle_idx]
labels = labels[shuffle_idx]

print("Mixed dataset created with overlapping classes")
print("Sample data (hue, area):\n", np.round(data[:10], 1))
print("Labels:", labels[:10])
print(f"Class distribution: Class 0: {np.sum(labels==0)}, Class 1: {np.sum(labels==1)}")

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)

# Visualize the mixed data
plt.figure(figsize=(12, 8))

# Original data visualization
plt.subplot(2, 2, 1)
class_0_mask = labels == 0
class_1_mask = labels == 1
plt.scatter(data[class_0_mask, 0], data[class_0_mask, 1], c='red', marker='o', alpha=0.6, label='Class 0', s=50)
plt.scatter(data[class_1_mask, 0], data[class_1_mask, 1], c='green', marker='x', alpha=0.6, label='Class 1', s=50)
plt.xlabel('Hue')
plt.ylabel('Area')
plt.title('Mixed Dataset: Overlapping Classes')
plt.legend()
plt.grid(True, alpha=0.3)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Train and evaluate classifiers
results = {}
plot_idx = 2

for name, clf in classifiers.items():
    # Train classifier
    if name == 'Neural Network':
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Test Accuracy: {accuracy:.3f}")
    
    # Plot decision boundary
    plt.subplot(2, 2, plot_idx)
    
    # Create mesh for decision boundary
    h = 2  # step size
    x_min, x_max = data[:, 0].min() - 10, data[:, 0].max() + 10
    y_min, y_max = data[:, 1].min() - 50, data[:, 1].max() + 50
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    if name == 'Neural Network':
        Z = clf.predict(scaler.transform(mesh_points))
    else:
        Z = clf.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['lightcoral', 'lightgreen'])
    
    # Plot training data
    train_0_mask = y_train == 0
    train_1_mask = y_train == 1
    plt.scatter(X_train[train_0_mask, 0], X_train[train_0_mask, 1], 
                c='red', marker='o', alpha=0.7, s=30, edgecolor='black', linewidth=0.5)
    plt.scatter(X_train[train_1_mask, 0], X_train[train_1_mask, 1], 
                c='green', marker='x', alpha=0.7, s=30)
    
    # Plot test data with different markers
    test_0_mask = y_test == 0
    test_1_mask = y_test == 1
    plt.scatter(X_test[test_0_mask, 0], X_test[test_0_mask, 1], 
                c='darkred', marker='s', alpha=0.8, s=25, label='Test Class 0')
    plt.scatter(X_test[test_1_mask, 0], X_test[test_1_mask, 1], 
                c='darkgreen', marker='^', alpha=0.8, s=25, label='Test Class 1')
    
    plt.xlabel('Hue')
    plt.ylabel('Area')
    plt.title(f'{name}\nAccuracy: {accuracy:.3f}')
    plt.grid(True, alpha=0.3)
    if plot_idx == 2:
        plt.legend()
    
    plot_idx += 1

plt.tight_layout()
plt.show()

# Performance comparison
plt.figure(figsize=(12, 8))

# Bar plot of accuracies
plt.subplot(2, 2, 1)
names = list(results.keys())
accuracies = list(results.values())
bars = plt.bar(names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum'])
plt.ylabel('Test Accuracy')
plt.title('Classifier Performance Comparison')
plt.xticks(rotation=45)
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Confusion matrices for best and worst performers
best_clf_name = max(results, key=results.get)
worst_clf_name = min(results, key=results.get)

# Best classifier confusion matrix
plt.subplot(2, 2, 2)
best_clf = classifiers[best_clf_name]
if best_clf_name == 'Neural Network':
    y_pred_best = best_clf.predict(X_test_scaled)
else:
    y_pred_best = best_clf.predict(X_test)
cm_best = confusion_matrix(y_test, y_pred_best)
plt.imshow(cm_best, interpolation='nearest', cmap='Blues')
plt.title(f'Best: {best_clf_name}')
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm_best[i, j]), ha='center', va='center', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Worst classifier confusion matrix
plt.subplot(2, 2, 3)
worst_clf = classifiers[worst_clf_name]
if worst_clf_name == 'Neural Network':
    y_pred_worst = worst_clf.predict(X_test_scaled)
else:
    y_pred_worst = worst_clf.predict(X_test)
cm_worst = confusion_matrix(y_test, y_pred_worst)
plt.imshow(cm_worst, interpolation='nearest', cmap='Reds')
plt.title(f'Worst: {worst_clf_name}')
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm_worst[i, j]), ha='center', va='center', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Feature distribution by class
plt.subplot(2, 2, 4)
plt.hist(data[labels==0, 0], alpha=0.6, color='red', label='Class 0 Hue', bins=20)
plt.hist(data[labels==1, 0], alpha=0.6, color='green', label='Class 1 Hue', bins=20)
plt.xlabel('Hue Value')
plt.ylabel('Frequency')
plt.title('Feature Distribution by Class')
plt.legend()

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*50)
print("CLASSIFICATION RESULTS SUMMARY")
print("="*50)
print(f"Dataset: {len(data)} samples with {data.shape[1]} features")
print(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")
print("\nClassifier Performance:")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {acc:.3f}")

print(f"\nBest performer: {best_clf_name} ({results[best_clf_name]:.3f})")
print(f"Worst performer: {worst_clf_name} ({results[worst_clf_name]:.3f})")
print(f"Performance gap: {results[best_clf_name] - results[worst_clf_name]:.3f}")