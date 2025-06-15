import pandas as pd
import joblib
import os
import sys


path_to_dataset="./data/Hackathon_bureau_data_400.csv"

def predict(df: pd.DataFrame, true_values=None) -> pd.DataFrame:
    """
    Predicts income based on the input DataFrame using the pre-trained model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features for prediction.
                           Must include a 'id' column.
        true_values (pd.Series, optional): The true target values, if available.

    Returns:
        pd.DataFrame: A DataFrame with 'id' and 'predicted_income' columns.
    """
    print("Loading the trained pipeline...")
    try:
        # Ensure the model path is correct relative to where the script is run
        model_path = 'full_pipeline.joblib'
        if not os.path.exists(model_path):
            # If run_inference.py is in a different directory, adjust path
            # For hackathon structure, it should be in the root alongside main.py
            # If it's in a subfolder like 'src', then it would be '../full_pipeline.joblib'
            # Assuming it's in the same directory for now
            pass
            
        full_pipeline = joblib.load(model_path)
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        raise RuntimeError(f"Failed to load trained model: {e}")

    # Ensure 'id' is present for output
    if 'id' not in df.columns:
        # If id is not present, create a dummy one or handle as per problem spec
        # For this hackathon, test.csv is a replica of Hackathon_bureau_data_400.csv,
        # which has id. So assuming it's always present.
        raise ValueError("Input DataFrame must contain a 'id' column.")

    print("Making predictions...")
    # Fill NA values in the input DataFrame to avoid ambiguous boolean errors
    df = df.fillna(0)
    # Make predictions
    predictions = full_pipeline.predict(df)
    print("Predictions made.",predictions)
    # Create output DataFrame
    result_df = pd.DataFrame({'id': df['id'], 'predicted_income': predictions})
    
    # Ensure predicted_income is non-negative
    result_df['predicted_income'] = result_df['predicted_income'].apply(lambda x: max(0, x))

    # If the true target is provided, calculate and display metrics
    if true_values is not None:
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        import numpy as np
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
        print(f"RMSE: {rmse}")

    print("Predictions completed.")
    return result_df

if __name__ == "__main__":
    print("Running inference script for demonstration...")
    # Allow user to specify a CSV file as a command-line argument
    if len(sys.argv) > 1:
        sample_data_path = sys.argv[1]
    else:
        sample_data_path = path_to_dataset
    try:
        sample_df = pd.read_csv(sample_data_path, dtype_backend='numpy_nullable', low_memory=True)
        # Ensure 'id' is treated as string for consistent output, if it's numeric
        if 'id' in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df['id']):
            sample_df['id'] = sample_df['id'].astype(str)
        # Save true values if present, then drop from features
        true_values = None
        if 'target_income' in sample_df.columns:
            true_values = sample_df['target_income'].copy()
            sample_df = sample_df.drop(columns=['target_income'])
        output_df = predict(sample_df, true_values)
        # Save output to the 'output/' folder
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        # Output file name based on input file name
        input_base = os.path.splitext(os.path.basename(sample_data_path))[0]
        output_file_path = os.path.join(output_dir, f'{input_base}_predictions.csv')
        output_df.to_csv(output_file_path, index=False)
        print(f"Sample predictions saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred during demonstration: {e}")
