from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os, io, joblib, boto3
from botocore.config import Config

app = FastAPI(title="CPI Simple Classifier")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")
VEC_PATH = os.getenv("VEC_PATH", "/app/model/vectorizer.pkl")
S3_MODEL_URI = os.getenv("S3_MODEL_URI") or os.getenv("STORAGE_URI")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def _parse_s3_uri(uri: str):
    p = uri[5:]
    bucket, key = p.split("/", 1)
    return bucket, key

def download_s3(uri: str) -> bytes:
    bucket, key = _parse_s3_uri(uri)
    s3 = boto3.client("s3", config=Config(region_name=AWS_REGION))
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def ensure_model():
    global model, vec
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VEC_PATH)
    elif S3_MODEL_URI:
        model = joblib.load(io.BytesIO(download_s3(S3_MODEL_URI)))
        vec_uri = S3_MODEL_URI.rsplit("/", 1)[0] + "/vectorizer.pkl"
        vec = joblib.load(io.BytesIO(download_s3(vec_uri)))
    else:
        raise RuntimeError("Model not found")

ensure_model()

class CPILog(BaseModel):
    ARTIFACT_NAME: Optional[str] = ""
    ORIGIN_COMPONENT_NAME: Optional[str] = ""
    LOG_LEVEL: Optional[str] = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(record: CPILog) -> Dict[str, Any]:
    X = vec.transform([record.dict()])
    pred = model.predict(X)[0]
    conf = float(max(model.predict_proba(X)[0])) if hasattr(model,"predict_proba") else None
    return {"prediction": pred, "confidence": conf}
