import numpy as np
import pandas as pd
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


iris_df = pd.read_csv('/Users/baur/Downloads/Iris.csv', sep=',', header=None)


print("Dataset shape:", iris_df.shape)
print("First few rows:")
print(iris_df.head())


X_all = iris_df.iloc[:, :4].apply(pd.to_numeric, errors='coerce').values
print("Checking for NaN values:")
print("NaN in X_all:", np.any(np.isnan(X_all)))
print("Number of NaN values:", np.sum(np.isnan(X_all)))
print("X_all sample:", X_all[:5])

# Remove rows with NaN values
if np.any(np.isnan(X_all)):
    print("Removing NaN values...")
    valid_rows = ~np.isnan(X_all).any(axis=1)
    X_all = X_all[valid_rows]
    y_all = y_all[valid_rows]
    print("Data shape after removing NaN:", X_all.shape)
y_all = iris_df.iloc[:, 4].values


if isinstance(y_all[0], str):
    unique_species = np.unique(y_all)
    print("Species found:", unique_species)


    species_to_num = {species: i for i, species in enumerate(unique_species)}
    y_all = np.array([species_to_num[species] for species in y_all])

    print("Species mapping:", species_to_num)


class RBPerceptron():
    def __init__(self, number_of_epochs=100, learning_rate=0.1):
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate

    def train(self, X, T):
        num_features = X.shape[1]
        self.w = np.zeros(num_features + 1)

        for i in range(self.number_of_epochs):
            for sample, desired_outcome in zip(X, T):
                prediction = self.predict(sample)
                difference = (desired_outcome - prediction)

                self.w[1:] += self.learning_rate * difference * sample
                self.w[0] += self.learning_rate * difference * 1
        return self

    def predict(self, sample):
        outcome = np.dot(sample, self.w[1:]) + self.w[0]
        return np.where(outcome > 0, 1, 0)


# Part A: Use Sepal Length and Petal Length as features
print("\n" + "=" * 60)
print("Part A: Sepal Length vs Petal Length (Setosa vs Versicolor)")
print("=" * 60)

# Filter for Setosa (class 0) and Versicolor (class 1) only
setosa_versicolor_mask = y_all != 2  # Exclude Virginica (class 2)
X_a = X_all[setosa_versicolor_mask][:, [0, 2]]  # Sepal Length (0), Petal Length (2)
T_a = y_all[setosa_versicolor_mask]

print(f"Dataset shape: {X_a.shape}")
print(f"Classes: Setosa (0): {np.sum(T_a == 0)} samples, Versicolor (1): {np.sum(T_a == 1)} samples")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
setosa_data = X_a[T_a == 0]
versicolor_data = X_a[T_a == 1]
plt.scatter(setosa_data[:, 0], setosa_data[:, 1], color='red', label='Setosa', alpha=0.7)
plt.scatter(versicolor_data[:, 0], versicolor_data[:, 1], color='blue', label='Versicolor', alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Part A: Sepal Length vs Petal Length')
plt.legend()
plt.grid(True, alpha=0.3)

# Train perceptron for Part A
rbp_a = RBPerceptron(500, 0.1)
trained_model_a = rbp_a.train(X_a, T_a)

plt.subplot(1, 2, 2)
plot_decision_regions(X_a, T_a.astype(int), clf=trained_model_a, legend=2)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Part A: Decision Boundary (500 epochs)')
plt.tight_layout()
plt.show()

# Test the model
predictions_a = np.array([trained_model_a.predict(sample) for sample in X_a])
accuracy_a = np.mean(predictions_a == T_a)
print(f"Part A Accuracy: {accuracy_a:.3f} ({accuracy_a * 100:.1f}%)")

print("\n" + "=" * 60)
print("Part B: Sepal Width vs Petal Width (Setosa vs Versicolor)")
print("=" * 60)

# Part B: Use Sepal Width and Petal Width as features
X_b = X_all[setosa_versicolor_mask][:, [1, 3]]  # Sepal Width (1), Petal Width (3)
T_b = y_all[setosa_versicolor_mask]

print(f"Dataset shape: {X_b.shape}")
print(f"Classes: Setosa (0): {np.sum(T_b == 0)} samples, Versicolor (1): {np.sum(T_b == 1)} samples")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
setosa_data_b = X_b[T_b == 0]
versicolor_data_b = X_b[T_b == 1]
plt.scatter(setosa_data_b[:, 0], setosa_data_b[:, 1], color='red', label='Setosa', alpha=0.7)
plt.scatter(versicolor_data_b[:, 0], versicolor_data_b[:, 1], color='blue', label='Versicolor', alpha=0.7)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Part B: Sepal Width vs Petal Width')
plt.legend()
plt.grid(True, alpha=0.3)

# Train perceptron for Part B
rbp_b = RBPerceptron(500, 0.1)
trained_model_b = rbp_b.train(X_b, T_b)

plt.subplot(1, 2, 2)
plot_decision_regions(X_b, T_b.astype(np.integer), clf=trained_model_b, legend=2)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Part B: Decision Boundary (500 epochs)')
plt.tight_layout()
plt.show()


predictions_b = np.array([trained_model_b.predict(sample) for sample in X_b])
accuracy_b = np.mean(predictions_b == T_b)
print(f"Part B Accuracy: {accuracy_b:.3f} ({accuracy_b * 100:.1f}%)")

print("\n" + "=" * 50)
print("COMPARISON AND ANALYSIS")
print("=" * 50)
print(f"Part A (Sepal Length vs Petal Length) Accuracy: {accuracy_a * 100:.1f}%")
print(f"Part B (Sepal Width vs Petal Width) Accuracy: {accuracy_b * 100:.1f}%")

if accuracy_a > accuracy_b:
    print("\nPart A performs better - Length measurements are more discriminative")
    print("than width measurements for separating Setosa from Versicolor.")
else:
    print("\nPart B performs better - Width measurements are more discriminative")
    print("than length measurements for separating Setosa from Versicolor.")


print("CONVERGENCE ANALYSIS")

plt.figure(figsize=(12, 4))

for part_name, X_data, T_data in [("Part A", X_a, T_a), ("Part B", X_b, T_b)]:
    plt.subplot(1, 2, 1 if part_name == "Part A" else 2)

    accuracies = []
    epoch_values = [50, 100, 200, 300, 400, 500]

    for epochs in epoch_values:
        rbp_temp = RBPerceptron(epochs, 0.1)
        model_temp = rbp_temp.train(X_data, T_data)
        predictions_temp = np.array([model_temp.predict(sample) for sample in X_data])
        accuracy_temp = np.mean(predictions_temp == T_data)
        accuracies.append(accuracy_temp)

    plt.plot(epoch_values, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{part_name}: Accuracy vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()