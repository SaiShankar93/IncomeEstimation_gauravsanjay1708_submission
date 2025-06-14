# Income Estimation Model

## Project Overview

This project aims to build a robust and efficient machine learning model for income estimation, designed to replicate the fluency, adaptability, and empathy of a human expert providing financial guidance. The solution incorporates advanced ETL operations, feature engineering, and an ensemble of powerful gradient boosting models (CatBoost, LightGBM, and XGBoost).

## Core Features

-   **Optimized Data Loading**: Efficiently handles large datasets with optimized pandas settings for memory and performance.
-   **Advanced Feature Engineering**: Generates a rich set of features including numeric interactions, polynomial features, and statistical measures (z-scores, ranks) to capture complex relationships in the data.
-   **Smart Cardinality Reduction**: Addresses high cardinality in categorical features by grouping rare categories into an 'other' class.
-   **Robust Preprocessing Pipeline**: Utilizes `PowerTransformer` for handling skewed numeric features, `StandardScaler` for normalization, and `OneHotEncoder` with `sparse_output` for categorical data.
-   **Feature Selection**: Employs `SelectKBest` with `mutual_info_regression` to select the most relevant features, reducing dimensionality and improving model performance and training time.
-   **Ensemble Modeling**: Leverages a `VotingRegressor` combining CatBoost, LightGBM, and XGBoost for superior predictive accuracy and generalization.
-   **GPU Acceleration (Optional)**: Models are configured to utilize GPU for faster training if available in the environment.
-   **Submission Ready**: Provides a `predict` function in `run_inference.py` that adheres to the hackathon's submission guidelines, saving predictions to the specified output folder.

## Setup and Environment

To set up the environment and run the project, follow these steps:

1.  **Clone the Repository (if applicable)**:
    ```bash
    git clone <your-repo-link>
    cd income_estimation_<YourTeamName>_submission
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv env
    ```

3.  **Activate the Virtual Environment**:
    -   **Windows (Command Prompt)**:
        ```bash
        .\env\Scripts\activate
        ```
    -   **Windows (PowerShell)**:
        ```powershell
        .\env\Scripts\Activate.ps1
        ```
    -   **macOS/Linux**:
        ```bash
        source env/bin/activate
        ```

4.  **Install Required Dependencies**:
    Ensure you are in the root directory of the project (where `requirements.txt` is located).
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you plan to use GPU acceleration for CatBoost, LightGBM, or XGBoost, you might need to install their respective GPU-enabled versions. Please refer to their official documentation for specific instructions based on your CUDA version and system configuration. The provided `requirements.txt` lists CPU-compatible versions by default.*

## How to Run

### 1. Train the Model and Save the Pipeline (`main.py`)

This script will load and preprocess the data, engineer features, train the ensemble model, evaluate its performance, and save the trained pipeline as `full_pipeline.joblib` in the root directory. This `full_pipeline.joblib` is crucial for the inference step.

Run the training script from the root directory:

```bash
python main.py
```

Upon successful execution, you should see training progress, evaluation metrics (MAE, R2, RMSE), and a confirmation that the pipeline has been saved.

### 2. Run Inference (`run_inference.py`)

This script is designed to be used for the hackathon's evaluation. It contains a `predict` function that takes a pandas DataFrame as input and returns predictions. When run as a script, it demonstrates how to load the saved pipeline, make predictions on a sample of data (or `test.csv` in the actual submission), and save the results to the `output/` folder.

First, ensure you have run `main.py` to generate `full_pipeline.joblib`.

Then, run the inference script from the root directory:

```bash
python run_inference.py
```

This will generate a `output_predictions.csv` file inside the `output/` directory, containing the `unique_id` and `predicted_income` columns as required.

## API Keys or Environment Variables

This project does not currently require any external API keys or environment variables. All data is loaded locally from the `datasets/` folder.

If future enhancements or external data sources are integrated, relevant environment variables would be listed here and stored in a `.env` file (which should not be committed to Git).

## Evaluation Criteria

As per the hackathon guidelines, the model will be evaluated based on the following metrics:

-   **R2, RMSE, MAE**: Core regression metrics.
-   **Population coverage within 25% deviation**.
-   **Population coverage within absolute difference of 5k**.
-   **System Performance & Latency**.
-   **Practicality, Scalability & Real-World Viability**.

## Contact

For any questions or issues, please refer to the hackathon organizers.