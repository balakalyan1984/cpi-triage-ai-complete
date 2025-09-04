# CPI-Triage-AI (Complete)

## Build & push images
docker build -f docker/Dockerfile.train -t <registry>/<repo>:cpi-triage-ai-train .
docker build -f docker/Dockerfile.serve -t <registry>/<repo>:cpi-triage-ai-serve .
docker push <registry>/<repo>:cpi-triage-ai-train
docker push <registry>/<repo>:cpi-triage-ai-serve

## Import into Applications
Import all YAMLs under `applications/`.

## Run
- Execute **Train CPI classifier** → produces model artifact `cpimodel`
- Execute **Daily summary job** → produces distribution summary
- Deploy **cpi-model-serving** → exposes `/predict` endpoint
