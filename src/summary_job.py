import os, io, json, joblib, pandas as pd, requests, boto3
from botocore.config import Config

DATA_PATH = os.getenv("DATA_PATH", "/app/data/cpi_logs.csv")
S3_DATA_URI = os.getenv("S3_DATA_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")
VEC_PATH = os.getenv("VEC_PATH", "/app/model/vectorizer.pkl")
OUT_DIR = os.getenv("OUT_DIR", "/app/out")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

os.makedirs(OUT_DIR, exist_ok=True)

def _parse_s3_uri(uri: str):
    p = uri[5:]
    bucket, key = p.split("/", 1)
    return bucket, key

def read_csv(uri: str) -> pd.DataFrame:
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        s3 = boto3.client("s3", config=Config(region_name=AWS_REGION))
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(uri)

df = read_csv(S3_DATA_URI or DATA_PATH)
model, vec = joblib.load(MODEL_PATH), joblib.load(VEC_PATH)

X = vec.transform(df[["ARTIFACT_NAME","ORIGIN_COMPONENT_NAME","LOG_LEVEL"]].fillna("").to_dict(orient="records"))
df["_pred"] = model.predict(X)
dist = df["_pred"].value_counts().to_dict()

summary = {"total_records": len(df), "predicted_distribution": dist}
with open(os.path.join(OUT_DIR,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
pd.DataFrame(list(dist.items()), columns=["bucket","count"]).to_csv(os.path.join(OUT_DIR,"bucket_summary.csv"), index=False)

if SLACK_WEBHOOK_URL:
    requests.post(SLACK_WEBHOOK_URL, json={"text": str(summary)})

if TEAMS_WEBHOOK_URL:
    requests.post(TEAMS_WEBHOOK_URL, json={"text": str(summary)})
