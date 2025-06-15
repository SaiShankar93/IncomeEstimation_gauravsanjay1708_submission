from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import pandas as pd
import joblib
import os
from tempfile import NamedTemporaryFile
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_files', 'full_pipeline.joblib')
model = joblib.load(MODEL_PATH)

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), 'public')
os.makedirs(PUBLIC_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    with NamedTemporaryFile(delete=False, suffix='.csv') as temp:
        temp.write(await file.read())
        temp_path = temp.name
    try:
        # Read CSV
        df = pd.read_csv(temp_path)
        df_original = df.copy()
        # Prepare true values if present
        true_values = None
        if 'target_income' in df.columns:
            true_values = df['target_income'].copy()
            df = df.drop(columns=['target_income'])
        # Ensure 'id' is string for output
        if 'id' in df.columns and pd.api.types.is_numeric_dtype(df['id']):
            df['id'] = df['id'].astype(str)
        # Fill NA values
        df = df.fillna(0)
        # Make predictions
        preds = model.predict(df)
        # Create output DataFrame
        result_df = pd.DataFrame({'id': df['id'], 'predicted_income': preds})
        result_df['predicted_income'] = result_df['predicted_income'].apply(lambda x: max(0, x))
        # Save output in public directory
        output_filename = f"output_{os.path.basename(temp_path)}"
        output_path = os.path.join(PUBLIC_DIR, output_filename)
        result_df.to_csv(output_path, index=False)
        # Calculate metrics if true values are present
        metrics = {}
        if true_values is not None:
            mae = float(mean_absolute_error(true_values, preds))
            r2 = float(r2_score(true_values, preds))
            rmse = float(np.sqrt(mean_squared_error(true_values, preds)))
            metrics = {"MAE": mae, "R2": r2, "RMSE": rmse}
        # Prepare response with full URL
        server_url = "http://54.190.251.247:8000"  # Change if deploying elsewhere
        download_link = f"{server_url}/public/{output_filename}"
        return {"download_link": download_link, "metrics": metrics}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_path)

from fastapi.staticfiles import StaticFiles
app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(PUBLIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
