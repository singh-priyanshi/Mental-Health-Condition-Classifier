# Mental Health Condition Classifier: Predicting Anxiety, Depression, Insomnia, and OCD

This project implements a machine learning-based approach to predict mental health conditions, such as anxiety, depression, insomnia, and obsessive-compulsive disorder (OCD). Using various machine learning models, the project aims to classify these conditions based on input features from relevant datasets.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Machine Learning Models](#machine-learning-models)
3. [Architecture and Techniques](#architecture-and-techniques)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)

## Project Overview

This project addresses the challenge of predicting mental health conditions using structured data. It evaluates various machine learning models to identify which ones are most effective at classifying these conditions. The primary goal is to provide a robust, high-performing solution to assist in early mental health diagnosis.

The project utilizes both classical machine learning techniques and neural networks to compare and determine the best approach for each mental health condition.

## Data Analysis

<img width="632" alt="image" src="https://github.com/user-attachments/assets/fddffd9e-fb12-41ed-bb23-205abbb2901a">

<img width="583" alt="image" src="https://github.com/user-attachments/assets/44a8226e-1b63-413a-a0b6-b884ae3b87ca">


## Machine Learning Models

The following machine learning algorithms are used:

1. **Random Forest Classifier**: The primary model used for classifying mental health conditions. Random Forest is an ensemble method, known for its high accuracy and resistance to overfitting.
2. **Neural Networks**: A simple feedforward neural network was also tested for classification tasks. While the accuracy was promising, further optimization and tuning are required.
3. **Other Models**: Several other models, including Logistic Regression and SVM, were tested for comparison purposes.

## Architecture and Techniques

1. **Random Forest Classifier**: 
   - Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification. This technique is known for handling classification tasks with high accuracy, especially in structured data problems like this.
   - In this project, the Random Forest model is used to classify anxiety, depression, insomnia, and OCD. The model was tuned for optimal hyperparameters and evaluated for precision, recall, and accuracy on both training and test datasets.

2. **Neural Networks**:
   - A simple feedforward neural network was also employed in this project. While not as accurate as Random Forest, it demonstrated good potential for prediction. The network used ReLU activation functions and was trained using backpropagation with a cross-entropy loss function.
   - The neural network achieved around 55.3% accuracy on the test set, indicating room for improvement through techniques such as hyperparameter tuning and feature engineering.

3. **Cross-validation**:
   - Cross-validation was used to ensure that the model's performance is generalized across unseen data and not overfitted to the training set.

4. **Data Preprocessing**:
   - Data was cleaned and preprocessed, including handling missing values, feature scaling, and encoding categorical variables. Feature selection was applied to reduce dimensionality and improve performance.
   
5. **Evaluation Metrics**:
   - Accuracy, precision, recall, and F1 score were the primary metrics used to evaluate the models. These metrics were used to assess the model’s performance on both training and testing data, with a focus on maximizing recall due to the nature of mental health diagnosis.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/username/mental-health-condition-classifier.git
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. After installing the necessary dependencies, run the notebook or the Python script to train the model and generate predictions.

2. Customize the dataset and model parameters to fine-tune the model for better performance.

## Results
<img width="402" alt="image" src="https://github.com/user-attachments/assets/524094b6-e41c-44e1-aebd-0f17a5499c03">

The Random Forest model emerged as the most reliable model across different conditions, consistently delivering high accuracy. Below is a summary of the results:

- **Anxiety**: 
  - Train accuracy: 47.2%
  - Test accuracy: 42.5%

- **Depression**: 
  - Train accuracy: 54.2%
  - Test accuracy: 48.9%

- **Insomnia**: 
  - Train accuracy: 51.7%
  - Test accuracy: 57.4%

- **OCD**: 
  - Train accuracy: 67.4%
  - Test accuracy: 73.0%

The neural network, while slightly behind Random Forest, shows potential with an accuracy of around 55.3% on the test set.

## Future Work

- **Hyperparameter Optimization**: Further tuning of the neural network and Random Forest hyperparameters to enhance prediction accuracy.
- **Feature Engineering**: Extracting more insightful features to improve model performance.
- **Neural Network Expansion**: Implementing deeper neural networks or trying more sophisticated architectures like CNNs or LSTMs.
- **Explainability**: Adding model explainability techniques like SHAP or LIME to better understand feature importance and model decisions.

--- 

Feel free to contribute to this project by forking the repository and submitting a pull request!


