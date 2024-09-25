from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import joblib
from src.data_utils import load_and_preprocess_data
from src.model import train_model, evaluate_model
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/train/")
async def train(files: list[UploadFile] = File(...)):
    dataset_models = {}
    for file in files:
        dataset_name = os.path.splitext(file.filename)[0]
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data([file.file])
            model = train_model(X_train, y_train, dataset_name)
            metrics = evaluate_model(model, X_test, y_test)
            dataset_models[dataset_name] = metrics
        except Exception as e:
            return {"error": str(e)}
    
    return {"message": "Models trained successfully", "metrics": dataset_models}

@app.post("/predict/")
async def predict(dataset_name: str = Query(...), file: UploadFile = File(...)):
    try:
        model = joblib.load(f'models/churn_model_{dataset_name}.pkl')
        _, X_test, _, _ = load_and_preprocess_data([file.file])
        predictions = model.predict(X_test)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
