# UFC Initial Documentation

## Overview

This documentation provides an overview and detailed workflow descriptions for a project focused on processing, analyzing, and modeling fight data to predict outcomes using machine learning techniques.

## Workflow

### Data Processing

#### Step 1: Data Extraction and Initial Cleaning

**Notebook: `FinalData.ipynb`**
- **Purpose**: Extracts raw data and performs the initial cleaning steps to produce a preliminary cleaned dataset.
- **Output**: `CleanData.csv`
- **Execution**: This notebook is located in the `Extract` folder and should be executed first to start the data processing pipeline.

#### Step 2: Additional Data Cleaning

**Notebook: `CleanFeature(CleanData, CleanData2.ipynb)`**
- **Purpose**: Applies additional cleaning and feature engineering on the `CleanData.csv` file. This step may use GPT for automated text and data manipulation to refine the dataset further.
- **Output**: `CleanData2.csv`
- **Details**: Takes `CleanData.csv` as an input and generates `CleanData2.csv`, enhancing the dataset with additional features and cleaning.

#### Step 3: Data Augmentation and Finalization

**Notebook: `Model Combining Data and Cleaning.ipynb`**
- **Purpose**: Matches and combines fighter information from `Results.csv` and `CleanData2.csv`. It augments the dataset by doubling the number of rows to enhance data robustness for training models.
- **Outputs**:
  - `FinalAugmentedResult.csv`
  - `CleanedData2.csv`
- **Details**: This step involves matching fighters across datasets and augmenting the data accordingly to ensure a comprehensive dataset for further applications.

### Model Training and Evaluation

**Notebooks:**

1. **TrialAugment.ipynb**
   - **Purpose**: Implements data augmentation strategies for the prepared datasets to enhance model training.
   - **Outputs**: Augmented datasets ready for machine learning applications.

2. **XGBoost.ipynb**
   - **Purpose**: Trains a predictive model using the XGBoost algorithm. It includes feature scaling, training-test split, model fitting, and performance evaluation.
   - **Outputs**: Trained XGBoost model, performance metrics (e.g., accuracy), hyperparameter tuning results. Obtained an accuracy of 72.72%

### Model Deployment

**Python Script: `main.py`**
- **Purpose**: Deploys the trained model within a Streamlit application, providing an interactive interface for predicting fight outcomes based on user-selected fighters.
- **Key Features**: Interactive selection of fighters, prediction execution, and display of predicted outcomes.

## Data Files

- **Augmented_FinalAugmentedResults.csv**
- **Augmented_Results.csv**
- **CleanData2.csv**
- **FinalAugmentedResults.csv**

These files represent various stages of data cleaning, augmentation, and final datasets used for model training.
