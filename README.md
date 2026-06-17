# Wildfire Detector AI

A modular deep learning API designed to predict forest fire risks using a Multilayer Perceptron (MLP) built with **PyTorch** and deployed via **FastAPI**. This system processes geographical, temporal, and historical weather data using the Canadian Forest Fire Weather Index (FWI) system.

## 🛠️ System Architecture

All development code is isolated within the `src/` directory to keep the workspace clean and modular:

```text
wildfire-detector/
├── src/
│   ├── ai/
│   │   ├── data/
│   │   │   ├── data.py              # Data pipeline & preprocessing
│   │   │   └── forestfires.csv      # Dataset source file
│   │   ├── model/
│   │   │   └── network.py           # PyTorch neural network definition
│   │   ├── train.py                 # Training orchestrator script
│   │   └── api.py                   # Main AI router endpoints
│   ├── routers/
│   │   └── health.py                # System health check router
│   └── main.py                      # FastAPI application entry point
├── requirements.txt                 # Project environment dependencies
└── README.md                        # Documentation
```

## 🚀 Getting Started

Follow these steps to set up the environment and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/xcaim04/wildfire-detector.git
cd wildfire-detector
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Pipeline Execution

The system must follow a specific sequence to prepare the data, train the neural network, and serve predictions.

### Step 1: Preprocess Data & Train the Model
Before launching the web server, you can trigger data preparation and training locally via terminal, or use the API endpoints later. To train directly via terminal:
```bash
python -m src.ai.train
```
*This will generate `scaler.pkl` in `src/ai/data/` and `wildfire_model.pth` in `src/ai/model/`.*

### Step 2: Launch the FastAPI Web Server
Run the following command from the root directory of the project to boot up the application:
```bash
uvicorn src.main:app --reload
```

### Step 3: Access Interactive Documentation
Once the server is active, open your web browser and navigate to:
* **Swagger UI Docs:** [http://127.0.0](http://127.0.0)
* **ReDoc:** [http://127.0.0](http://127.0.0)

---

## 📡 Core API Endpoints

All Machine Learning endpoints are grouped under the `/wildfire` prefix route.

### 1. Preprocess Dataset
* **Endpoint:** `POST /wildfire/process-data`
* **Description:** Parses, maps string calendars, balances data slices, and dumps the mathematical feature scaling architecture.

### 2. Model Training
* **Endpoint:** `POST /wildfire/train`
* **Description:** Asynchronously triggers the PyTorch gradient descent optimization loops in a background thread to prevent API blocking.

### 3. Live Prediction Infeference
* **Endpoint:** `POST /wildfire/predict`
* **Payload Format (JSON):**
```json
{
  "X": 7,
  "Y": 5,
  "month": "mar",
  "day": "fri",
  "FFMC": 86.2,
  "DMC": 26.2,
  "DC": 94.3,
  "ISI": 5.1,
  "temp": 8.2,
  "RH": 51.0,
  "wind": 6.7,
  "rain": 0.0
}
```
* **Response Output:**
```json
{
  "status": "success",
  "fire_prediction": true,
  "fire_probability": 0.8421
}
```

---

## 📊 Dataset Features & Parameters

The predictive engine interprets 11 distinct input variables to evaluate environmental hazard thresholds:

* **Spatial Grid (`X`, `Y`):** Coordinates mapping specific sectors of the Montesinho Natural Park (ranging from 1 to 9).
* **Temporal Cyclicity (`month`, `day`):** String dates mapped into linear factors to detect seasonal trends.
* **FFMC (Fine Fuel Moisture Code):** Measures moisture content of surface litter. Highly responsive to immediate climate changes.
* **DMC (Duff Moisture Code):** Evaluates moisture levels in medium-deep organic layers.
* **DC (Drought Code):** Tracks structural underground seasonal deep-drying trends.
* **ISI (Initial Spread Index):** Combines wind speed currents and surface moisture parameters to score the speed of a fire's advance.
* **Weather Metrics (`temp`, `RH`, `wind`, `rain`):** Ambient temperature, relative air humidity percentage, wind velocity metrics, and rain levels.

---

## 👥 Contributors
Ariel David Marin Batista  Github user: ArielDavidXD
Ana Laura Reinoso Fernandez Github user : annarei25


* **Final Project for Numerical Mathematics.**
