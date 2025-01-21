# Credit-Card-behaviour-score

This repository contains the implementation of a predictive model for assessing the likelihood of credit card defaults, referred to as the Credit Card Behaviour Score. The project utilizes machine learning techniques to process data, handle imbalances, and develop a robust predictive model.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Development](#model-development)
5. [Model Evaluation](#model-evaluation)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Contributors](#contributors)

---

## Project Overview
The objective of this project is to build a Behaviour Score for predicting credit card defaults based on customer data. The model is designed to help financial institutions better manage credit risk.

## Data Preprocessing
Data preprocessing is a critical step in ensuring the quality and reliability of the model. The following actions were taken:

1. **Handling Missing Values**:
   - Columns with more than 30% missing values were removed.
   - Remaining missing values were imputed as follows:
     - **Bureau Attributes**: Median imputation.
     - **Bureau Enquiry Attributes**: Filled with zeros.
     - **On-us and Transaction Attributes**: Median imputation.

2. **Class Balancing**:
   - The dataset exhibited significant class imbalance (defaults vs. non-defaults).
   - **SMOTE** (Synthetic Minority Oversampling Technique) was used to generate synthetic samples for the minority class.

## Feature Engineering
1. **Feature Selection**:
   - Low-variance features (variance < 0.01) were removed.
   - Highly correlated features (absolute correlation > 0.9) were dropped.

2. **Dimensionality Reduction**:
   - PCA (Principal Component Analysis) reduced the dataset to 10 components, retaining most of the variance while enhancing computational efficiency.

## Model Development

### Algorithm
A Deep Neural Network (DNN) was implemented using TensorFlow/Keras:

1. **Architecture**:
   - **Input Layer**: 64 neurons with ReLU activation.
   - **Hidden Layers**:
     - Layer 1: 32 neurons with ReLU activation.
     - Layer 2: 16 neurons with ReLU activation.
     - Dropout regularization (rate = 0.3) applied after each hidden layer.
   - **Output Layer**: Single neuron with sigmoid activation for binary classification.

2. **Compilation**:
   - **Optimizer**: Adam (learning rate = 0.001).
   - **Loss Function**: Binary cross-entropy.
   - **Metrics**: Accuracy.

3. **Training**:
   - Epochs: 20.
   - Batch size: 32.
   - Validation split: 20%.

## Model Evaluation
The model's performance was evaluated using the following metrics:

- **Accuracy**: Overall prediction correctness.
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve.
- **Precision, Recall, and F1-score**: Metrics from the classification report to assess balance and correctness.

## Usage
To use this repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-behaviour-score.git
   ```
2. Install dependencies (see below).
3. Follow the scripts for preprocessing, training, and evaluation.

## Dependencies
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Contributors
- **Your Name**

Feel free to raise issues or contribute to this repository. Suggestions and pull requests are welcome!

