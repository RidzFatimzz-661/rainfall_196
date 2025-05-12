Rainfall Prediction Model

Overview

This project develops a Rainfall Prediction Model to predict whether it will rain based on weather-related features such as temperature, humidity, wind speed, and pressure. The model implements three machine learning algorithms: Random Forest, Support Vector Machine (SVM), and Logistic Regression. These models are trained and evaluated to provide accurate predictions, making the system useful for applications in agriculture, urban planning, and weather forecasting.

The project includes a Python implementation that preprocesses data, trains the models, compares their performance, and visualizes results. This README provides an overview of the models, instructions for running the code, and details on the dataset and results.

Models Implemented
1. Random Forest Description: Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. It is robust to noisy data and handles non-linear relationships well.
Implementation: The model uses scikit-learn's RandomForestClassifier with 100 trees and tuned hyperparameters (e.g., max depth, minimum samples split). Features are scaled, and the model is trained to classify whether rain will occur (1) or not (0).
Why Used: Random Forest is chosen for its high accuracy, ability to handle feature interactions, and robustness to imbalanced datasets, making it ideal for weather prediction.

2. Support Vector Machine (SVM)
Description: SVM is a supervised learning algorithm that finds the optimal hyperplane to separate classes in a high-dimensional space. For non-linear data, it uses a kernel trick (e.g., RBF kernel) to improve separation.
Implementation: The model uses scikit-learn's SVC with an RBF kernel and tuned parameters (e.g., C for regularization, gamma for kernel scale). Data is standardized to ensure SVM performs well with varying feature scales.
Why Used: SVM is effective for binary classification tasks like rainfall prediction, especially when the decision boundary is complex or non-linear.
3. Logistic Regression
Description: Logistic Regression is a linear model that predicts the probability of a binary outcome (e.g., rain or no rain). It is simple, interpretable, and works well for linearly separable data.
Implementation: The model uses scikit-learn's LogisticRegression with L2 regularization. Features are scaled, and the model outputs probabilities, which are thresholded to classify rain (1) or no rain (0).
Why Used: Logistic Regression is included for its simplicity, interpretability, and as a baseline model to compare against more complex algorithms.
Dataset
Source: The model uses a synthetic or publicly available weather dataset (e.g., inspired by datasets like the Australian Rainfall dataset). For this project, a sample dataset is included (weather_data.csv).
Features: Temperature (°C), Humidity (%), Wind Speed (km/h), Pressure (hPa), Cloud Cover (%).
Target: Binary label (1 for rain, 0 for no rain).
Preprocessing: The data is cleaned (handling missing values), scaled (using StandardScaler), and split into training (80%) and testing (20%) sets.
Project Structure

rainfall_prediction/
│
├── data/
│   └── weather_data.csv       # Sample dataset
├── src/
│   └── rainfall_prediction.py # Main script with model implementation
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies

Installation
Clone the repository:

git clone https://github.com/your-username/rainfall_prediction.git
cd rainfall_prediction
Install dependencies:
pip install -r requirements.txt
Ensure the dataset (weather_data.csv) is in the data/ folder.
Usage
Run the main script to train models and evaluate performance:

python src/rainfall_prediction.py
The script will:
Load and preprocess the dataset.
Train Random Forest, SVM, and Logistic Regression models.
Evaluate models using accuracy, precision, recall, and F1-score.
Display a comparison of model performance and save results.
Generate visualizations (e.g., confusion matrices, feature importance).
Results
Random Forest: Typically achieves the highest accuracy (~85-90%) due to its ability to capture non-linear patterns and feature interactions.

SVM: Performs well (~80-85%) with the RBF kernel, especially for complex decision boundaries.

Logistic Regression: Serves as a baseline (~75-80%), performing adequately for linearly separable data but less effectively for complex patterns.

Visualizations: Confusion matrices and feature importance plots are generated to understand model performance and key predictors (e.g., humidity, pressure).

Requirements
Python 3.8+
Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn
Install via:
pip install numpy pandas scikit-learn matplotlib seaborn
Future Improvements
Incorporate real-time weather API data for dynamic predictions.
Experiment with additional models (e.g., XGBoost, Neural Networks).

Add hyperparameter tuning using GridSearchCV for better performance.

Include time-series features (e.g., previous day's rainfall) for temporal analysis.

Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or additional features.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or feedback, please contact [ridhafatima661@gmail.com] or open an issue on GitHub.
