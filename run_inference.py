import pandas as pd
import joblib
import os

def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts income based on the input DataFrame using the pre-trained model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features for prediction.
                           Must include a 'unique_id' column.

    Returns:
        pd.DataFrame: A DataFrame with 'unique_id' and 'predicted_income' columns.
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

    # Ensure 'unique_id' is present for output
    if 'unique_id' not in df.columns:
        # If unique_id is not present, create a dummy one or handle as per problem spec
        # For this hackathon, test.csv is a replica of Hackathon_bureau_data_400.csv,
        # which has unique_id. So assuming it's always present.
        raise ValueError("Input DataFrame must contain a 'unique_id' column.")

    print("Making predictions...")
    # Make predictions
    predictions = full_pipeline.predict(df)
    
    # Create output DataFrame
    result_df = pd.DataFrame({'unique_id': df['unique_id'], 'predicted_income': predictions})
    
    # Ensure predicted_income is non-negative
    result_df['predicted_income'] = result_df['predicted_income'].apply(lambda x: max(0, x))

    print("Predictions completed.")
    return result_df

if __name__ == "__main__":
    # This part is for local testing/demonstration. 
    # The actual evaluation will call the 'predict' function directly.
    print("Running inference script for demonstration...")
    # Load a sample of the test data (or part of the training data for testing)
    # In a real scenario, this would be the hidden test.csv
    try:
        sample_data_path = "datasets/Hackathon_bureau_data_50000.csv"
        sample_df = pd.read_csv(sample_data_path, dtype_backend='numpy_nullable', low_memory=True)
        
        # Ensure 'unique_id' is treated as string for consistent output, if it's numeric
        if 'unique_id' in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df['unique_id']):
            sample_df['unique_id'] = sample_df['unique_id'].astype(str)

        # Take a small sample to avoid memory issues during local testing
        sample_df = sample_df.sample(n=100, random_state=42).copy() if len(sample_df) > 100 else sample_df.copy()
        
        # Drop target_income as it's not present in the inference data
        if 'target_income' in sample_df.columns:
            sample_df = sample_df.drop(columns=['target_income'])

        output_df = predict(sample_df)
        
        # Save output to the 'output/' folder
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, 'output_predictions.csv')
        output_df.to_csv(output_file_path, index=False)
        print(f"Sample predictions saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred during demonstration: {e}")
