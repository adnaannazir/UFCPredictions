# Fight Outcome Prediction Project

## Overview

This project involves processing, analyzing, and predicting fight outcomes using machine learning techniques. The goal is to accurately predict the winner between two fighters based on historical and performance data.

## Workflow

### Data Cleaning and Preparation

#### 1. Data Cleaning
**Notebook: `cleaned.ipynb`**
- **Purpose**: Cleans and preprocesses data starting from `NewCleanedDataDuplicated.csv` to produce `FinalCleanedData.csv`.
- **Output**: `FinalCleanedData.csv`, a cleaner and more structured dataset ready for model training.

#### 2. Data Files
- **`NewCleanedDataDuplicated.csv`**: Initial dataset containing duplicated entries that need cleaning.
- **`FinalCleanedData.csv`**: Cleaned and deduplicated dataset used for model training.

### Model Development

#### 1. Main Model Training
**Notebook: `MainModel.ipynb`**
- **Purpose**: Trains the main predictive model using `FinalCleanedData.csv` combined with `Outcome.csv`. Achieves an accuracy of 94.3%.
- **Output**: A trained machine learning model capable of predicting fight outcomes with high accuracy.

#### 2. Data Files
- **`Outcome.csv`**: Contains the outcomes of fights used as labels for training the predictive model.

### Prediction Implementation

#### 1. Prediction Trials
**Notebook: `TrialToPredict.ipynb`**
- **Purpose**: Demonstrates the application of the trained model to predict the outcome of fights between two specified fighters.
- **Functionality**: Utilizes the model to make predictions based on user inputs or predefined criteria.

#### 2. Automated Prediction Logic
**Script: `prediction.py`**
- **Purpose**: Handles the logic for fetching fighters' data, processing it as per model requirements, and feeding it to the model to predict outcomes.
- **Details**: Orchestrates the flow from data retrieval through feature engineering to model application.

#### 3. Prediction Data
- **`Predictions.csv`**: Lists predictions for upcoming fights, which can be used to validate the model's effectiveness or for promotional activities.

### Additional Documentation

#### 1. Model Accuracy and Validation
- The main model (`MainModel.ipynb`) details testing and validation processes showing an accuracy of 94.3%, highlighting the model's reliability and efficiency in real-world scenarios.

#### 2. Data Sources
- **`Copy of History.xlsx`** and **`Info`** files provide background data and additional information relevant to the fighters and fights, although not directly used in the model training.

## How to Use

1. **Prepare the Environment**: Ensure all dependencies are installed as per the `requirements.txt` (if available) or manually install necessary libraries (pandas, numpy, sklearn, xgboost).
2. **Run Notebooks in Order**: Execute the notebooks in the order they are listed to ensure data flows correctly from cleaning to prediction.
3. **Utilize the Prediction Script**: Use `prediction.py` to apply the model in a real-world scenario or to automate fight outcome predictions.

## Dependencies

- Python 3.8+
- Libraries: pandas, numpy, sklearn, xgboost
- Jupyter Notebook for executing IPYNB files

## Future Enhancements

- Integrate real-time data feeds to update fighters' records automatically.
- Enhance the model with new data and advanced machine learning techniques to improve prediction accuracy.
- Expand the model to include more nuanced features such as fight location, referee influence, and more detailed historical performance metrics.

