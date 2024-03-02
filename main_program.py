import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc  # Make sure to include these lines
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONFUSION
def plot_confusion_matrix(y_true, y_pred, class_mapping, algorithm):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_mapping.values())

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format=".4g")
    plt.title(f'Confusion Matrix - {algorithm}')
    plt.show()

# Function to load images and labels from the dataset
def load_dataset(root_folder):
    images = []
    labels = []
    class_mapping = {}  # To map class names to numeric labels

    for class_label, class_name in enumerate(os.listdir(root_folder), start=1):
        class_mapping[class_label] = class_name
        class_folder = os.path.join(root_folder, class_name)

        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            try:
                image = imread(image_path)
                image = resize(image, (64, 64), anti_aliasing=True)  # Adjust image size as needed
                images.append(image.flatten())  # Flatten the image
                labels.append(class_label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    return np.array(images), np.array(labels), class_mapping

# Function to save results to a CSV file
def save_results(algorithm, y_true, y_pred, class_mapping):
    results_df = pd.DataFrame({
        'True Label': [class_mapping[label] for label in y_true],
        'Predicted Label': [class_mapping[label] for label in y_pred]
    })

    results_folder = f'./results/{algorithm}'
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, f'{algorithm}_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f'Results saved for {algorithm} at {results_file}')

# Function to save test images with predictions
def save_test_images(algorithm, X_test, y_true, y_pred, class_mapping, probabilities):
    results_folder = f'./results_folder/{algorithm}'
    os.makedirs(results_folder, exist_ok=True)

    for idx, (true_label, pred_label, prob_dist) in enumerate(zip(y_true, y_pred, probabilities)):
        image = X_test[idx].reshape(64, 64, -1)  # Reshape flattened image
        plt.imshow(image)
        plt.title(f'True Label: {class_mapping[true_label]}\nPredicted Label: {class_mapping[pred_label]}')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_test_image_{idx}.png'))
        plt.close()

        # Plot probability distribution
        plt.bar(class_mapping.values(), prob_dist, color='blue')
        plt.title(f'Probability Distribution - Test Image {idx}')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_probability_distribution_{idx}.png'))
        plt.close()

    print(f'Test images saved for {algorithm} in {results_folder}')

# Load your dataset
dataset_folder = "dataset"  # Change this to the path of your dataset folder
X, y, class_mapping = load_dataset(dataset_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example 1: k-Nearest Neighbors (k-NN)
# Initialize k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred_knn = knn_classifier.predict(X_train)

# Save results and training images for k-NN
save_results('knn_train', y_train, y_train_pred_knn, class_mapping)
plot_confusion_matrix(y_train, y_train_pred_knn, class_mapping, 'knn')
# Make predictions on the test set
y_test_pred_knn = knn_classifier.predict(X_test)

# Save results and test images for k-NN
save_results('knn_test', y_test, y_test_pred_knn, class_mapping)
plot_confusion_matrix(y_test, y_test_pred_knn, class_mapping, 'knn')

def plot_roc_curve_overall(y_true, probabilities, algorithm, positive_class=1):
    # Convert multi-class labels to binary for ROC curve calculation
    y_true_binary = (y_true == positive_class).astype(int)

    fpr, tpr, _ = roc_curve(y_true_binary, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, lw=2, label=f'{algorithm} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{algorithm} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


probabilities_knn = knn_classifier.predict_proba(X_test)

plot_roc_curve_overall(y_test, probabilities_knn[:, 1], 'k-NN', positive_class=2)

save_test_images('knn_test', X_test, y_test, y_test_pred_knn, class_mapping, probabilities_knn)

# Example 2: Naive Bayes (Gaussian Naive Bayes)
# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred_nb = nb_classifier.predict(X_train)

# Save results and training images for Naive Bayes
save_results('naive_bayes_train', y_train, y_train_pred_nb, class_mapping)
plot_confusion_matrix(y_train, y_train_pred_nb, class_mapping, 'naive_bayes')
# Make predictions on the test set
y_test_pred_nb = nb_classifier.predict(X_test)

# Save results and test images for Naive Bayes
save_results('naive_bayes_test', y_test, y_test_pred_nb, class_mapping)
plot_confusion_matrix(y_test, y_test_pred_nb, class_mapping, 'naive_bayes')
probabilities_nb = nb_classifier.predict_proba(X_test)
plot_roc_curve_overall(y_test, probabilities_nb[:, 1], 'Naive Bayes', positive_class=2)
(y_test, probabilities_nb[:, 1], 'Naive Bayes')
save_test_images('naive_bayes_test', X_test, y_test, y_test_pred_nb, class_mapping, probabilities_nb)

# Function to plot ROC curves for each class

