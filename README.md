# Pneumonia-Chest-X-Ray-Classification

## **DATASET**
- The dataset used in this project is a collection of chest X-ray images stored in a ZIP file, with a structured folder hierarchy containing three subsets: train, test, and val. Each of these subsets is further divided into two categories: NORMAL (healthy lungs) and PNEUMONIA (lungs infected with pneumonia). The images are in grayscale format and need to be preprocessed before being fed into a deep learning model. Instead of extracting the dataset, we implemented a function to load images directly from the ZIP file, ensuring efficient memory usage. The dataset is then combined into a single training set, which is later split into training, validation, and testing subsets using train_test_split, ensuring an even class distribution.

- The dataset is loaded directly from the ZIP file, converted to RGB format (since CNNs and transfer learning models typically expect 3-channel inputs), and normalized by scaling pixel values to the range [0,1]. 

## **MODEL BUILDING**
- Since deep learning models, especially convolutional neural networks (CNNs), require a large and diverse dataset for optimal performance, data augmentation is applied to artificially expand the dataset. Augmentation techniques such as rotation, shifting and flipping adjustments are used to introduce variability and improve the model’s generalization. These transformations help prevent overfitting by ensuring that the model does not memorize specific image patterns but instead learns to recognize normal and pneumonia features in various orientations and lighting conditions. OpenCV and TensorFlow/Keras libraries are used to apply these transformations dynamically during model training.

In this project, multiple deep learning models were utilized such as:

- 1. Custom Convolutional Neural Network (CNN),
  2. VGG16,
  3. ResNet50,
  4. InceptionV3

## **EVALUATION**
- To assess the model’s performance, key evaluation metrics such as accuracy, precision, recall, and F1-score are computed using the classification_report function from Scikit-learn. A confusion matrix is plotted to visualize the distribution of true positives, true negatives, false positives, and false negatives.
- All models with higher accuracy, better recall, and improved robustness against misclassifications.