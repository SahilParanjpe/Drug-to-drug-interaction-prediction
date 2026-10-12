FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenMP runtime is commonly needed by scikit-learn / XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# These files must exist in your project root before building.
COPY main.py /app/main.py
COPY morgan_features.py /app/morgan_features.py
COPY custom32_features.py /app/custom32_features.py
COPY custom32_wl_features.py /app/custom32_wl_features.py
COPY static/ /app/static/

RUN mkdir -p /app/data /app/models/binary_ddi

# All files come from the same public Hugging Face repository.
# The model files are downloaded at image-build time, not per prediction request.
RUN set -eux; \
    HF_BASE="https://huggingface.co/RsRane/ddi-models-and-data/resolve/main"; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/Final_Canonical_Full_DrugBank_SMILES.csv" \
      -o /app/data/Final_Canonical_Full_DrugBank_SMILES.csv; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/drugbank%20vocabulary.csv" \
      -o "/app/data/drugbank vocabulary.csv"; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/morgan_rf.joblib" -o /app/models/binary_ddi/morgan_rf.joblib; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/morgan_xgb.joblib" -o /app/models/binary_ddi/morgan_xgb.joblib; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/custom32_rf.joblib" -o /app/models/binary_ddi/custom32_rf.joblib; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/custom32_xgb.joblib" -o /app/models/binary_ddi/custom32_xgb.joblib; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/custom32_wl_rf.joblib" -o /app/models/binary_ddi/custom32_wl_rf.joblib; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
      "$HF_BASE/custom32_wl_xgb.joblib" -o /app/models/binary_ddi/custom32_wl_xgb.joblib; \
    test -s /app/data/Final_Canonical_Full_DrugBank_SMILES.csv; \
    test -s "/app/data/drugbank vocabulary.csv"; \
    test -s /app/models/binary_ddi/morgan_rf.joblib; \
    test -s /app/models/binary_ddi/morgan_xgb.joblib; \
    test -s /app/models/binary_ddi/custom32_rf.joblib; \
    test -s /app/models/binary_ddi/custom32_xgb.joblib; \
    test -s /app/models/binary_ddi/custom32_wl_rf.joblib; \
    test -s /app/models/binary_ddi/custom32_wl_xgb.joblib

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
