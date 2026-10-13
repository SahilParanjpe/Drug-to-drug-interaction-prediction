from contextlib import asynccontextmanager
from itertools import combinations
from pathlib import Path
from threading import Lock
import importlib
import gc
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "binary_ddi"
STATIC_DIR = BASE_DIR / "static"

CANONICAL_CSV = DATA_DIR / "Final_Canonical_Full_DrugBank_SMILES.csv"
VOCAB_CSV = DATA_DIR / "drugbank vocabulary.csv"

FEATURE_MODULES = {
    "morgan": "morgan_features",
    "custom32": "custom32_features",
    "custom32_wl": "custom32_wl_features",
}

MODEL_FILES = {
    ("rf", "morgan"): MODEL_DIR / "morgan_rf.joblib",
    ("xgb", "morgan"): MODEL_DIR / "morgan_xgb.joblib",
    ("rf", "custom32"): MODEL_DIR / "custom32_rf.joblib",
    ("xgb", "custom32"): MODEL_DIR / "custom32_xgb.joblib",
    ("rf", "custom32_wl"): MODEL_DIR / "custom32_wl_rf.joblib",
    ("xgb", "custom32_wl"): MODEL_DIR / "custom32_wl_xgb.joblib",
}

name_to_info = {}
id_to_info = {}
term_to_id = {}
loaded_feature_modules = {}

# Cache only the most recently requested, usable estimator.
_cached_model_key = None
_cached_model = None
_model_lock = Lock()


def clean_cell(value):
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null"} else value


def load_resources():
    global name_to_info, id_to_info, term_to_id, loaded_feature_modules

    for path in (CANONICAL_CSV, VOCAB_CSV):
        if not path.is_file():
            raise RuntimeError(f"Required data file is missing: {path}")

    canonical = pd.read_csv(
        CANONICAL_CSV, dtype=str, keep_default_na=False, low_memory=False
    )
    canonical.columns = [str(column).strip() for column in canonical.columns]

    required_columns = {"DrugBank_ID", "Name", "SMILES"}
    if not required_columns.issubset(canonical.columns):
        raise RuntimeError(
            f"{CANONICAL_CSV.name} must contain columns: "
            f"{', '.join(sorted(required_columns))}. "
            f"Found: {', '.join(canonical.columns)}"
        )

    name_to_info = {}
    id_to_info = {}

    for _, row in canonical.iterrows():
        drug_id = clean_cell(row.get("DrugBank_ID")).lower()
        name = clean_cell(row.get("Name"))
        smiles = clean_cell(row.get("SMILES"))

        if not drug_id or not smiles:
            continue

        info = {
            "DrugBank_ID": drug_id.upper(),
            "Name": name or drug_id.upper(),
            "SMILES": smiles,
        }
        id_to_info[drug_id] = info
        if name:
            name_to_info[name.casefold()] = info

    vocab = pd.read_csv(
        VOCAB_CSV, dtype=str, keep_default_na=False, low_memory=False
    )
    vocab.columns = [str(column).strip() for column in vocab.columns]

    required_vocab = {"DrugBank ID", "Common name", "Synonyms"}
    if not required_vocab.issubset(vocab.columns):
        raise RuntimeError(
            f"{VOCAB_CSV.name} must contain columns: "
            f"{', '.join(sorted(required_vocab))}. "
            f"Found: {', '.join(vocab.columns)}"
        )

    term_to_id = {}
    for _, row in vocab.iterrows():
        drug_id = clean_cell(row.get("DrugBank ID")).lower()
        if not drug_id:
            continue

        common_name = clean_cell(row.get("Common name"))
        if common_name:
            term_to_id[common_name.casefold()] = drug_id

        synonyms = clean_cell(row.get("Synonyms"))
        if synonyms:
            for synonym in synonyms.split("|"):
                synonym = synonym.strip().casefold()
                if synonym:
                    term_to_id[synonym] = drug_id

    loaded_feature_modules = {}
    for feature_key, module_name in FEATURE_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not import feature module '{module_name}': {exc}"
            ) from exc

        if not callable(getattr(module, "get_pair_features", None)):
            raise RuntimeError(
                f"Feature module '{module_name}' must define "
                "get_pair_features(smiles_a, smiles_b)."
            )
        loaded_feature_modules[feature_key] = module

    missing = [str(path) for path in MODEL_FILES.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing model file(s): " + ", ".join(missing))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="Multi-Model DDI Prediction API",
    version="4.1",
    lifespan=lifespan,
)


def resolve_drug(query: str):
    query_key = query.strip().casefold()
    if not query_key:
        return None

    if query_key in name_to_info:
        return name_to_info[query_key]

    if query_key in id_to_info:
        return id_to_info[query_key]

    drug_id = term_to_id.get(query_key)
    if drug_id:
        return id_to_info.get(drug_id)

    return None


def unwrap_predictor(loaded, model_name="loaded model"):
    """
    Return an object with predict(). Some training scripts save a dictionary
    containing the estimator rather than saving the estimator directly.
    """
    if callable(getattr(loaded, "predict", None)):
        return loaded

    preferred_keys = (
        "model",
        "estimator",
        "classifier",
        "pipeline",
        "best_estimator",
        "best_model",
        "clf",
        "predictor",
    )

    if isinstance(loaded, dict):
        # Prefer conventional keys first.
        for key in preferred_keys:
            if key in loaded:
                try:
                    return unwrap_predictor(loaded[key], f"{model_name}[{key!r}]")
                except TypeError:
                    pass

        # Then inspect other values in case the training script used another key.
        for key, value in loaded.items():
            if key in preferred_keys:
                continue
            try:
                return unwrap_predictor(value, f"{model_name}[{key!r}]")
            except TypeError:
                continue

    elif isinstance(loaded, (list, tuple)):
        for index, value in enumerate(loaded):
            try:
                return unwrap_predictor(value, f"{model_name}[{index}]")
            except TypeError:
                continue

    keys = list(loaded.keys()) if isinstance(loaded, dict) else None
    details = f" Dictionary keys found: {keys}." if keys is not None else ""
    raise TypeError(
        f"{model_name} does not contain an estimator with a predict() method."
        f"{details} Check how this joblib file was saved."
    )


def get_model(model_key):
    global _cached_model_key, _cached_model

    with _model_lock:
        # Return the model if the requested model is already cached.
        if _cached_model_key == model_key and _cached_model is not None:
            return _cached_model

        # Release the previously cached model before loading another.
        _cached_model = None
        _cached_model_key = None
        gc.collect()

        path = MODEL_FILES[model_key]

        try:
            loaded = joblib.load(path)
            model = unwrap_predictor(loaded, path.name)

            # Remove the temporary reference to the loaded object.
            del loaded
            gc.collect()

        except Exception as exc:
            _cached_model = None
            _cached_model_key = None
            gc.collect()

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not load a usable estimator from '{path.name}'. "
                    f"Details: {exc}"
                ),
            ) from exc

        # Cache only the newly selected model.
        _cached_model_key = model_key
        _cached_model = model

        return _cached_model

class PredictionRequest(BaseModel):
    drugs: list[str] = Field(min_length=2, max_length=100)
    feature_type: str = "custom32_wl"
    classifier: str = "rf"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "drug_records": len(id_to_info),
        "models_available": len(MODEL_FILES),
        "feature_pipelines_available": len(loaded_feature_modules),
    }


@app.post("/predict")
def predict_interactions(payload: PredictionRequest):
    classifier = payload.classifier.strip().lower()
    feature_type = payload.feature_type.strip().lower()
    model_key = (classifier, feature_type)

    if model_key not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported model combination.",
                "allowed_classifiers": ["rf", "xgb"],
                "allowed_feature_types": ["morgan", "custom32", "custom32_wl"],
            },
        )

    if feature_type not in loaded_feature_modules:
        raise HTTPException(
            status_code=500,
            detail=f"Feature module '{feature_type}' is unavailable.",
        )

    valid_drugs = []
    missing_drugs = []
    seen_queries = set()
    seen_ids = set()

    for raw_name in payload.drugs:
        input_name = raw_name.strip()
        if not input_name:
            continue

        query_key = input_name.casefold()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)

        resolved = resolve_drug(input_name)
        if resolved is None:
            missing_drugs.append(input_name)
            continue

        drug_id_key = resolved["DrugBank_ID"].casefold()
        if drug_id_key in seen_ids:
            continue
        seen_ids.add(drug_id_key)

        # Keep the exact input label for the dashboard, and the canonical
        # DrugBank ID/SMILES for internal lookup and feature generation.
        valid_drugs.append({
            **resolved,
            "input_name": input_name,
        })

    if len(valid_drugs) < 2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "At least two distinct drugs must resolve to records "
                    "with canonical SMILES."
                ),
                "missing_drugs": missing_drugs,
            },
        )

    model = get_model(model_key)
    get_pair_features = loaded_feature_modules[feature_type].get_pair_features

    classes = list(getattr(model, "classes_", []))
    positive_idx = next(
        (index for index, value in enumerate(classes) if str(value) == "1"),
        None,
    )

    interactions = []
    non_interactions = []
    prediction_errors = []

    for drug_a, drug_b in combinations(valid_drugs, 2):
        try:
            features = get_pair_features(drug_a["SMILES"], drug_b["SMILES"])
            prediction = int(model.predict(features)[0])

            confidence = None
            if callable(getattr(model, "predict_proba", None)):
                probabilities = model.predict_proba(features)[0]
                predicted_idx = next(
                    (
                        index
                        for index, value in enumerate(classes)
                        if str(value) == str(prediction)
                    ),
                    None,
                )

                if predicted_idx is not None and predicted_idx < len(probabilities):
                    confidence = round(float(probabilities[predicted_idx]), 4)
                elif positive_idx is not None and positive_idx < len(probabilities):
                    positive_probability = float(probabilities[positive_idx])
                    confidence = round(
                        positive_probability
                        if prediction == 1
                        else 1.0 - positive_probability,
                        4,
                    )

            # IMPORTANT: frontend pairCard() expects drug_a/drug_b to be
            # objects with .name and .id, not plain strings.
            pair_result = {
                "drug_a": {
                    "name": drug_a["input_name"],
                    "id": drug_a["DrugBank_ID"],
                },
                "drug_b": {
                    "name": drug_b["input_name"],
                    "id": drug_b["DrugBank_ID"],
                },
                "prediction": prediction,
                "confidence_score": confidence,
            }

            if prediction == 1:
                interactions.append(pair_result)
            else:
                non_interactions.append(pair_result)

        except Exception as exc:
            prediction_errors.append({
                "drug_a": {
                    "name": drug_a["input_name"],
                    "id": drug_a["DrugBank_ID"],
                },
                "drug_b": {
                    "name": drug_b["input_name"],
                    "id": drug_b["DrugBank_ID"],
                },
                "error": str(exc),
            })

    sort_key = lambda item: (
        item["confidence_score"]
        if item["confidence_score"] is not None
        else -1
    )

    return {
        "model_used": f"{classifier}_{feature_type}",
        "resolved_drug_count": len(valid_drugs),
        "missing_drugs": missing_drugs,
        "prediction_errors": prediction_errors,
        "interactions": sorted(interactions, key=sort_key, reverse=True),
        "non_interactions": sorted(non_interactions, key=sort_key, reverse=True),
        "notice": (
            "Model predictions are research outputs, not medical advice. "
            "A negative prediction does not prove that a drug combination is safe."
        ),
    }


# Mount static files last so it does not shadow /predict, /health, or /docs.
if not STATIC_DIR.is_dir():
    raise RuntimeError(f"Static frontend directory is missing: {STATIC_DIR}")

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
