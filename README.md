# 🎬 CineAI — Movie Success Classifier

Can a model predict whether a movie will be a commercial success **before it's released**, using only data available during pre-production?

This project builds and compares three machine learning models — Logistic Regression, Random Forest, and a Neural Network (MLP) — to classify movies as commercial successes or failures, using interpretability (SHAP) to understand *why* the model makes each prediction.

---

## 🎯 Motivation

Movie studios make multi-million dollar decisions before knowing whether a film will succeed. This project asks: how much signal exists in pre-production data alone — budget, genre, director, cast, production company, and release timing — to predict commercial outcomes?

**Target definition:** a movie is labeled a *success* if `revenue > budget` (ROI ≥ 1, before marketing costs). This is an imperfect but transparent and reproducible definition, discussed further in the limitations section.

**A hard constraint guided every decision:** only features that would realistically be known *before release* were used. Post-release signals like audience ratings, popularity scores, and vote counts were explicitly excluded — even though they're available in the raw dataset — to keep the model honest about what it could actually predict in a real-world scenario.

---

## 📊 Dataset

- **Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)
- **Size after cleaning:** 3,229 movies (filtered from 4,803 — removed entries with `budget=0` or `revenue=0`, which represent missing data disguised as zeros)
- **Class balance:** 75.5% success / 24.5% failure

### Features used (28 total, all pre-release)

| Category | Features |
|---|---|
| Numerical | `budget`, `runtime`, `release_year` |
| Historical success rates (leave-one-out, smoothed) | `director_success_rate`, `cast_success_rate`, `company_success_rate` |
| Genre (multi-hot) | 18 binary genre flags |
| Release season (one-hot) | `season_christmas`, `season_summer`, `season_spring`, `season_off_season` |

---

## 🔬 Methodology

The project follows a five-phase pipeline, each documented in its own notebook:

### 1. EDA & Cleaning (`01_eda.ipynb`)
- Merged movie metadata with cast/crew data
- Parsed nested JSON fields (genres, cast, crew, production companies)
- Removed post-release columns (`popularity`, `vote_average`, `vote_count`, `revenue`) to enforce the pre-release constraint
- Univariate analysis of every feature against the success target

### 2. Feature Engineering (`02_features.ipynb`)
- **Historical success rates** for director, cast, and production company, computed with **leave-one-out smoothing** to avoid data leakage and small-sample bias:

  ```
  weighted_rate = (n × own_rate + m × global_rate) / (n + m)
  ```

  Entities with little or no history are pulled toward the global mean (0.755); entities with extensive history retain their true rate.
- **Multi-hot encoding** for genres (a movie can belong to multiple genres simultaneously)
- **Seasonal grouping** of release months into industry-relevant categories (summer, christmas, spring, off-season)
- `original_language` dropped (96% English, near-zero variance)

### 3. Baseline Models (`03_baseline.ipynb`)
- Stratified 80/20 train/test split
- **SMOTE** oversampling applied to training data only (never to test) to address class imbalance
- StandardScaler fit on training data only
- Logistic Regression as the floor baseline
- Random Forest tuned via GridSearchCV (5-fold stratified CV, optimized for ROC-AUC)

### 4. Neural Network (`04_neural_net.ipynb`)
- MLP architecture: 128 → 64 → 32 → 1, with BatchNorm, ReLU, and Dropout
- Trained with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau
- Compared against both baseline models on the same test set

### 5. Interpretability (`05_shap.ipynb`)
- SHAP TreeExplainer applied to the winning model
- Global feature importance via summary plot
- Individual prediction breakdown via waterfall plot, focused on a high-confidence misclassification

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.717 | 0.786 | 0.859 | 0.821 | 0.655 |
| **Random Forest** | **0.765** | **0.799** | **0.920** | **0.855** | **0.768** |
| Neural Network (MLP) | 0.751 | 0.794 | 0.906 | 0.846 | 0.726 |

**Random Forest is the best-performing model across every metric**, and is used as the production model for the demo app.

For a dataset of this size (~3,200 rows) and structure (tabular, mixed continuous/binary features), the tree-based ensemble outperforms the neural network — consistent with widely observed patterns in applied ML: neural networks tend to need substantially more data to outperform ensembles on tabular problems.

---

## 🔍 Key Insights (SHAP)

1. **Production company history is the single strongest predictor.** Movies from companies with a strong historical success rate are pushed clearly toward "success" — and the effect is one of the cleanest, most consistent signals in the model.

2. **Release timing matters more than expected.** In isolation, release season showed only an 11-point spread in success rate during EDA (69%–80%). But SHAP reveals it's one of the most influential features *in combination* with everything else — a Christmas release can shift a prediction by as much as +0.17, more than any other single feature observed.

3. **Budget alone is a weak predictor.** Despite being the most intuitive variable, `budget` has one of the narrowest and most scattered SHAP value ranges — confirming the EDA finding that success and failure movies have nearly identical median budgets.

4. **A fundamental limitation:** the model cannot assess the *quality of execution* — script, directing, performances. An analysis of a high-confidence misclassification (predicted 88.5% success probability for a movie that was actually a failure) showed the model building a convincing case from context alone (Christmas release, strong cast history, solid studio) — but had no way to know the film itself wouldn't deliver. This is not a model failure; it's a structural ceiling of predicting from pre-production data alone.

---

## 🏗️ Architecture & Deployment

Beyond the modeling work, this project is deployed as a real two-service system, not a single notebook-turned-app:

```
┌─────────────────┐        HTTPS         ┌──────────────────────┐
│  Streamlit Cloud │ ───────────────────► │        Render         │
│    (frontend)    │  POST /predict       │   FastAPI (backend)   │
│                   │ ◄─────────────────── │  Random Forest + SHAP │
└─────────────────┘   probability + SHAP  └──────────────────────┘
```

- **Backend — FastAPI**, exposing `POST /predict` and `GET /health`. Loads the trained model and SHAP `TreeExplainer` **once** at startup (not per-request), validates every input with Pydantic, and returns a probability plus the top-3 SHAP features driving that specific prediction. Historical success rates for directors, cast, and production companies are served from a **SQLite** database rather than loose pickles.
- **Frontend — Streamlit**, a thin UI layer with no model logic: it collects form input and calls the backend over HTTP.
- **Containerization — Docker**, with a `Dockerfile` per service and a `docker-compose.yml` for local development, so both services run with a single `docker compose up` instead of manually juggling two virtual environments.
- **Deployment** — backend as a Docker web service on **Render**; frontend on **Streamlit Community Cloud**, which manages its own build from `requirements.txt`.
- **Dependency pinning matters here**: the backend pins `scikit-learn==1.7.2` to match the exact version the model was trained with — a version mismatch can silently change predictions.

---

## 🚀 Live Demo

- **App:** [cineai-pedrosall.streamlit.app](https://cineai-pedrosall.streamlit.app/)
- **API docs (Swagger):** [cineai-api-r3l7.onrender.com/docs](https://cineai-api-r3l7.onrender.com/docs)

Enter raw movie details (budget, genre, director, cast, release date) and get a real-time success prediction with a SHAP-based explanation of the top factors.

> ⚠️ The backend runs on Render's free tier, which sleeps after ~15 minutes of inactivity. The first request after a period of inactivity can take 30–50s to wake it up — this is expected, not a bug.

If a director, actor, or production company isn't found in the historical dataset, the app falls back to the global success rate (0.755) — the same logic used during training for unseen entities.

---

## 🛠️ How to Run Locally

### Option 1 — Notebooks only (model exploration)

```bash
git clone https://github.com/pedrosall/cineai.git
cd cineai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/
```

**Note:** raw data is not included in the repo. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle and place the CSVs in `data/raw/` before running `01_eda.ipynb`.

### Option 2 — Run the full app manually (two services)

```bash
# Terminal 1 — backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2 — frontend
cd ..
python -m venv frontend_venv && source frontend_venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Option 3 — Docker Compose (recommended, mirrors production)

```bash
docker compose up --build
```

Backend at `http://localhost:8000/docs`, frontend at `http://localhost:8501`.

---

## 📁 Repository Structure

```
cineai/
├── README.md
├── requirements.txt          # frontend deps (streamlit, requests)
├── docker-compose.yml
├── .dockerignore
├── data/
│   ├── raw/                  # TMDB CSVs (not tracked)
│   └── processed/            # cleaned pickles (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_neural_net.ipynb
│   └── 05_shap.ipynb
├── scripts/
│   └── build_db.py           # migrates rate lookups from pkl to SQLite
├── models/
│   ├── random_forest.pkl     # tracked (production model)
│   ├── scaler.pkl            # tracked
│   ├── cols_to_scale.pkl     # tracked
│   └── cineai.db             # tracked (director/cast/company rates)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt      # backend deps (fastapi, sklearn, shap...)
│   ├── main.py                # FastAPI app: /health, /predict
│   ├── schemas.py             # Pydantic request/response contracts
│   └── model_service.py       # model loading, prediction, SHAP
├── frontend/
│   └── Dockerfile             # local dev only — Streamlit Cloud ignores this
└── app/
    └── streamlit_app.py       # UI, calls the backend over HTTP
```

---

## ⚠️ Limitations & Future Work

- **Success definition is binary and simplistic.** `revenue > budget` ignores marketing costs (often comparable to production budget) and doesn't capture critical or cultural success.
- **Dataset is anglocentric and dated.** 96% English-language films, with coverage dropping off after ~2013.
- **No NLP yet.** `overview` and `keywords` were preserved during feature engineering but not used — a natural next step would be embedding-based features from plot synopses.
- **Director/cast/company rates are historical averages**, not causal estimates — they capture correlation (e.g., "directors who get hired by major studios tend to have bigger budgets and marketing"), not necessarily individual skill in isolation.

---

## 📜 License

This project is for educational purposes. Dataset © TMDB, distributed via Kaggle under their respective terms.
