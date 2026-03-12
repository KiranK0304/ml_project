from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# NEW: robust paths + better HTTP errors + CORS
from pathlib import Path
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ML Model API", version="1.1.0")



# Allow local frontend/dev clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Model file not found: {path}. "
            "Run training/serialization or place the .pkl into backend/models/."
        )
    with path.open("rb") as f:
        return pickle.load(f)


# Load models (relative to this file, not the process CWD)
knn_model = _load_model("knn_model.pkl")
logistic_model = _load_model("logistic_model.pkl")
svm_model = _load_model("svm_model.pkl")
mlr_model = _load_model("mlr_model.pkl")
polynomial_model = _load_model("polynomial_model.pkl")


# ---------------------------
# Input schemas per model
# ---------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class BreastCancerInput(BaseModel):
    # sklearn.datasets.load_breast_cancer().feature_names (30)
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


class WineInput(BaseModel):
    # sklearn.datasets.load_wine().feature_names (13)
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class CaliforniaHousingInput(BaseModel):
    # sklearn.datasets.fetch_california_housing().feature_names (8)
    medinc: float
    houseage: float
    aveRooms: float
    aveBedrms: float
    population: float
    aveOccup: float
    latitude: float
    longitude: float


class PolynomialInput(BaseModel):
    # Generic 1D polynomial regression input
    x: float


def prepare_iris(data: IrisInput) -> np.ndarray:
    return np.array(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]],
        dtype=float,
    )


def prepare_breast_cancer(data: BreastCancerInput) -> np.ndarray:
    # Keep the exact sklearn feature order
    return np.array(
        [[
            data.mean_radius,
            data.mean_texture,
            data.mean_perimeter,
            data.mean_area,
            data.mean_smoothness,
            data.mean_compactness,
            data.mean_concavity,
            data.mean_concave_points,
            data.mean_symmetry,
            data.mean_fractal_dimension,
            data.radius_error,
            data.texture_error,
            data.perimeter_error,
            data.area_error,
            data.smoothness_error,
            data.compactness_error,
            data.concavity_error,
            data.concave_points_error,
            data.symmetry_error,
            data.fractal_dimension_error,
            data.worst_radius,
            data.worst_texture,
            data.worst_perimeter,
            data.worst_area,
            data.worst_smoothness,
            data.worst_compactness,
            data.worst_concavity,
            data.worst_concave_points,
            data.worst_symmetry,
            data.worst_fractal_dimension,
        ]],
        dtype=float,
    )


def prepare_wine(data: WineInput) -> np.ndarray:
    return np.array(
        [[
            data.alcohol,
            data.malic_acid,
            data.ash,
            data.alcalinity_of_ash,
            data.magnesium,
            data.total_phenols,
            data.flavanoids,
            data.nonflavanoid_phenols,
            data.proanthocyanins,
            data.color_intensity,
            data.hue,
            data.od280_od315_of_diluted_wines,
            data.proline,
        ]],
        dtype=float,
    )


def prepare_california_housing(data: CaliforniaHousingInput) -> np.ndarray:
    return np.array(
        [[
            data.medinc,
            data.houseage,
            data.aveRooms,
            data.aveBedrms,
            data.population,
            data.aveOccup,
            data.latitude,
            data.longitude,
        ]],
        dtype=float,
    )


def prepare_polynomial(data: PolynomialInput) -> np.ndarray:
    # Many polynomial examples are trained on a single feature.
    return np.array([[data.x]], dtype=float)


# Model metadata for frontend
MODEL_SPECS = {
    "knn": {
        "title": "KNN (Iris)",
        "endpoint": "/predict/knn",
        "fields": [
            {"key": "sepal_length", "label": "Sepal Length", "min": 4.3, "max": 7.9, "example": 5.1},
            {"key": "sepal_width", "label": "Sepal Width", "min": 2.0, "max": 4.4, "example": 3.5},
            {"key": "petal_length", "label": "Petal Length", "min": 1.0, "max": 6.9, "example": 1.4},
            {"key": "petal_width", "label": "Petal Width", "min": 0.1, "max": 2.5, "example": 0.2},
        ],
    },
    "logistic": {
        "title": "Logistic Regression (Breast Cancer)",
        "endpoint": "/predict/logistic",
        "fields": [
            {"key": "mean_radius", "label": "mean radius", "min": 6.98, "max": 28.11, "example": 14.13},
            {"key": "mean_texture", "label": "mean texture", "min": 9.71, "max": 39.28, "example": 19.29},
            {"key": "mean_perimeter", "label": "mean perimeter", "min": 43.79, "max": 188.5, "example": 91.97},
            {"key": "mean_area", "label": "mean area", "min": 143.5, "max": 2501.0, "example": 654.9},
            {"key": "mean_smoothness", "label": "mean smoothness", "min": 0.053, "max": 0.163, "example": 0.097},
            {"key": "mean_compactness", "label": "mean compactness", "min": 0.019, "max": 0.345, "example": 0.104},
            {"key": "mean_concavity", "label": "mean concavity", "min": 0.0, "max": 0.427, "example": 0.089},
            {"key": "mean_concave_points", "label": "mean concave points", "min": 0.0, "max": 0.201, "example": 0.048},
            {"key": "mean_symmetry", "label": "mean symmetry", "min": 0.106, "max": 0.304, "example": 0.181},
            {"key": "mean_fractal_dimension", "label": "mean fractal dimension", "min": 0.05, "max": 0.097, "example": 0.063},

            {"key": "radius_error", "label": "radius error", "min": 0.112, "max": 2.873, "example": 0.405},
            {"key": "texture_error", "label": "texture error", "min": 0.36, "max": 4.885, "example": 1.216},
            {"key": "perimeter_error", "label": "perimeter error", "min": 0.757, "max": 21.98, "example": 2.866},
            {"key": "area_error", "label": "area error", "min": 6.802, "max": 542.2, "example": 40.34},
            {"key": "smoothness_error", "label": "smoothness error", "min": 0.0017, "max": 0.031, "example": 0.007},
            {"key": "compactness_error", "label": "compactness error", "min": 0.0023, "max": 0.135, "example": 0.025},
            {"key": "concavity_error", "label": "concavity error", "min": 0.0, "max": 0.396, "example": 0.032},
            {"key": "concave_points_error", "label": "concave points error", "min": 0.0, "max": 0.053, "example": 0.012},
            {"key": "symmetry_error", "label": "symmetry error", "min": 0.0079, "max": 0.079, "example": 0.02},
            {"key": "fractal_dimension_error", "label": "fractal dimension error", "min": 0.0009, "max": 0.03, "example": 0.003},

            {"key": "worst_radius", "label": "worst radius", "min": 7.93, "max": 36.04, "example": 16.27},
            {"key": "worst_texture", "label": "worst texture", "min": 12.02, "max": 49.54, "example": 25.68},
            {"key": "worst_perimeter", "label": "worst perimeter", "min": 50.41, "max": 251.2, "example": 107.26},
            {"key": "worst_area", "label": "worst area", "min": 185.2, "max": 4254.0, "example": 880.6},
            {"key": "worst_smoothness", "label": "worst smoothness", "min": 0.071, "max": 0.223, "example": 0.132},
            {"key": "worst_compactness", "label": "worst compactness", "min": 0.027, "max": 1.058, "example": 0.254},
            {"key": "worst_concavity", "label": "worst concavity", "min": 0.0, "max": 1.252, "example": 0.272},
            {"key": "worst_concave_points", "label": "worst concave points", "min": 0.0, "max": 0.291, "example": 0.115},
            {"key": "worst_symmetry", "label": "worst symmetry", "min": 0.156, "max": 0.664, "example": 0.29},
            {"key": "worst_fractal_dimension", "label": "worst fractal dimension", "min": 0.055, "max": 0.208, "example": 0.084},
        ],
    },
    "svm": {
        "title": "SVM (Wine)",
        "endpoint": "/predict/svm",
        "fields": [
            {"key": "alcohol", "label": "alcohol", "min": 11.03, "max": 14.83, "example": 13.0},
            {"key": "malic_acid", "label": "malic acid", "min": 0.74, "max": 5.8, "example": 2.0},
            {"key": "ash", "label": "ash", "min": 1.36, "max": 3.23, "example": 2.3},
            {"key": "alcalinity_of_ash", "label": "alcalinity of ash", "min": 10.6, "max": 30.0, "example": 19.5},
            {"key": "magnesium", "label": "magnesium", "min": 70.0, "max": 162.0, "example": 99.0},
            {"key": "total_phenols", "label": "total phenols", "min": 0.98, "max": 3.88, "example": 2.3},
            {"key": "flavanoids", "label": "flavanoids", "min": 0.34, "max": 5.08, "example": 2.0},
            {"key": "nonflavanoid_phenols", "label": "nonflavanoid phenols", "min": 0.13, "max": 0.66, "example": 0.3},
            {"key": "proanthocyanins", "label": "proanthocyanins", "min": 0.41, "max": 3.58, "example": 1.6},
            {"key": "color_intensity", "label": "color intensity", "min": 1.28, "max": 13.0, "example": 5.0},
            {"key": "hue", "label": "hue", "min": 0.48, "max": 1.71, "example": 1.04},
            {"key": "od280_od315_of_diluted_wines", "label": "od280/od315", "min": 1.27, "max": 4.0, "example": 2.87},
            {"key": "proline", "label": "proline", "min": 278.0, "max": 1680.0, "example": 745.0},
        ],
    },
    "mlr": {
        "title": "Multiple Linear Regression (California Housing)",
        "endpoint": "/predict/mlr",
        "task": "regression",
        "fields": [
            {"key": "medinc", "label": "Median income (MedInc)", "min": 0.5, "max": 15.0, "example": 3.87},
            {"key": "houseage", "label": "House age (HouseAge)", "min": 1.0, "max": 52.0, "example": 28.0},
            {"key": "aveRooms", "label": "Average rooms (AveRooms)", "min": 0.5, "max": 50.0, "example": 5.43},
            {"key": "aveBedrms", "label": "Average bedrooms (AveBedrms)", "min": 0.1, "max": 10.0, "example": 1.10},
            {"key": "population", "label": "Population", "min": 1.0, "max": 40000.0, "example": 1160.0},
            {"key": "aveOccup", "label": "Average occupancy (AveOccup)", "min": 0.5, "max": 50.0, "example": 2.56},
            {"key": "latitude", "label": "Latitude", "min": 32.0, "max": 42.0, "example": 34.26},
            {"key": "longitude", "label": "Longitude", "min": -124.5, "max": -114.0, "example": -118.25},
        ],
    },
    "polynomial": {
        "title": "Polynomial Regression (1D)",
        "endpoint": "/predict/polynomial",
        "task": "regression",
        "fields": [
            {"key": "x", "label": "x", "min": -10.0, "max": 10.0, "example": 2.0},
        ],
    },
}


@app.get("/models")
def list_models():
    return MODEL_SPECS


@app.get("/")
def home():
    return {
        "message": "ML Model API Running",
        "endpoints": [
            "/predict/knn",
            "/predict/logistic",
            "/predict/svm",
            "/predict/mlr",
            "/predict/polynomial",
        ],
    }


def _predict(model, model_name: str, x: np.ndarray):
    try:
        pred = model.predict(x)[0]
        # Most sklearn classifiers return numpy scalar; make it JSON-serializable
        return {"model": model_name, "prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed for {model_name}: {e}")


@app.post("/predict/knn")
def predict_knn(data: IrisInput):
    return _predict(knn_model, "KNN", prepare_iris(data))


@app.post("/predict/logistic")
def predict_logistic(data: BreastCancerInput):
    # 0 = malignant, 1 = benign (sklearn convention)
    result = _predict(logistic_model, "Logistic Regression", prepare_breast_cancer(data))
    label = "benign" if result["prediction"] == 1 else "malignant"
    return {**result, "label": label}


@app.post("/predict/svm")
def predict_svm(data: WineInput):
    return _predict(svm_model, "SVM", prepare_wine(data))


@app.post("/predict/mlr")
def predict_mlr(data: CaliforniaHousingInput):
    # Regression: return float
    try:
        x = prepare_california_housing(data)
        pred = float(mlr_model.predict(x)[0])
        return {"model": "Multiple Linear Regression", "prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed for MLR: {e}")


@app.post("/predict/polynomial")
def predict_polynomial(data: PolynomialInput):
    try:
        x = prepare_polynomial(data)
        pred = float(polynomial_model.predict(x)[0])
        return {"model": "Polynomial Regression", "prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed for Polynomial Regression: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)