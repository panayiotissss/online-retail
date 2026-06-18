from fastapi import FastAPI , Request
from pydantic import BaseModel
import logging
import mlflow
import pandas as pd
from contextlib import asynccontextmanager
import joblib
from features import log_transform



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerFeatures(BaseModel):
  Recency: float
  Frequency: float
  Monetary: float
  TotalItems: float
  UniqueProducts: float
  Tenure: float
  AvgOrderValue: float
  AvgQuantityPerOrder: float
  AvgDaysBetweenOrders: float


@asynccontextmanager
async def lifespan(app: FastAPI):    
    app.state.model = mlflow.sklearn.load_model("runs:/c07d9d193c0a4bc19372dd80c57eda76/model")
    app.state.scaler = joblib.load('models/scaler.pkl')
    logger.info("Model loaded from model.pkl")
    yield

app = FastAPI(title='Online Retail API', lifespan=lifespan)

@app.post('/predict')
def predict(features: CustomerFeatures, request : Request):
    input_df = log_transform(pd.DataFrame([features.dict()]))
    input_scaled = request.app.state.scaler.transform(input_df)
    prediction = request.app.state.model.predict(input_scaled)[0]

    cluster_names = {0: 'Champions', 1: 'At Risk', 2: 'Lost', 3: 'Loyal but Slow'}
    return {"cluster": int(prediction), "segment": cluster_names[int(prediction)]}
