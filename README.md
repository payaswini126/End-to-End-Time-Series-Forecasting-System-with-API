# Sales Forecasting System - Production-Ready Time Series API

A comprehensive end-to-end forecasting system that trains multiple algorithms, automatically selects the best model, and serves predictions via REST API.

## 🎯 Features

- **Multiple Forecasting Models**: ARIMA/SARIMA, Facebook Prophet, XGBoost, LSTM
- **Automatic Model Selection**: Compares models and selects best performer based on validation metrics
- **Advanced Feature Engineering**: Lag features, rolling statistics, date features, holiday effects, cyclical encoding
- **Production-Ready API**: FastAPI-based REST API with comprehensive endpoints
- **Time Series Best Practices**: Proper train/validation/test splits with no data leakage
- **Missing Data Handling**: Automatic handling of missing dates and values
- **Seasonality & Trend**: All models handle seasonality and trend components

## 📊 Models Implemented

### 1. SARIMA (Seasonal ARIMA)
- Order: (1,1,1) with seasonal order (1,1,1,52)
- Handles weekly seasonality
- Statistical approach for time series

### 2. Facebook Prophet
- Multiplicative seasonality
- Automatic changepoint detection
- Handles holidays and special events

### 3. XGBoost
- Gradient boosting with lag features
- 200 estimators with optimized hyperparameters
- Feature importance analysis

### 4. LSTM (Deep Learning)
- 2-layer LSTM with dropout
- Sequence length: 12 weeks
- Scaled features for better convergence

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── config.py                 # Configuration settings
├── data_loader.py           # Data loading and preprocessing
├── feature_engineering.py   # Feature creation module
├── models.py                # Model implementations
├── model_trainer.py         # Training and selection logic
├── api.py                   # FastAPI REST API
├── train.py                 # Training pipeline script
├── requirements.txt         # Python dependencies
├── data/                    # Data directory
├── models/                  # Saved models
├── results/                 # Forecast results
└── logs/                    # Training logs
```

## 🚀 Usage

### 1. Train Models

```bash
python train.py
```

This will:
- Load and preprocess the data
- Handle missing dates and values
- Create features for all states
- Train all 4 models for each state
- Compare and select best model
- Save models and results
- Generate sample forecasts

### 2. Start API Server

```bash
python api.py
```

Or with uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Get Available States
```bash
GET http://localhost:8000/states
```

#### Get State Information
```bash
GET http://localhost:8000/state/{state_name}
```

#### Generate Forecast (POST)
```bash
POST http://localhost:8000/forecast
Content-Type: application/json

{
  "state": "California",
  "horizon": 8
}
```

#### Generate Forecast (GET)
```bash
GET http://localhost:8000/forecast/{state_name}?horizon=8
```

### 4. API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📈 Feature Engineering

### Lag Features
- t-1, t-7, t-14, t-30 (previous weeks)

### Rolling Statistics
- Rolling mean, std, min, max for windows: 7, 14, 30 days

### Date Features
- Year, month, week, day of week, day of month
- Quarter, is_month_start, is_month_end
- is_quarter_start, is_quarter_end

### Holiday Features
- US holiday flag
- Days to next holiday
- Days from previous holiday

### Cyclical Features
- Sine/cosine encoding for month, day of week, week of year
- Preserves cyclical nature of time

## 🎯 Model Selection Criteria

Models are compared using validation set performance:
- **Primary Metric**: MAPE (Mean Absolute Percentage Error)
- **Secondary Metrics**: MAE, RMSE
- Best model is automatically selected and used for forecasting

## 📊 Output Files

After training, the following files are generated:

- `data/processed_data.csv` - Cleaned and preprocessed data
- `models/{state}_models.pkl` - Trained models for each state
- `results/model_comparison.csv` - Performance comparison
- `results/forecast_{state}.csv` - Individual state forecasts
- `results/all_forecasts.csv` - Combined forecasts
- `logs/training.log` - Training logs

## 🔧 Configuration

Edit `config.py` to customize:

```python
FORECAST_HORIZON = 8  # Number of weeks to forecast
VALIDATION_WEEKS = 8  # Validation set size
LAG_FEATURES = [1, 7, 14, 30]  # Lag periods
ROLLING_WINDOWS = [7, 14, 30]  # Rolling window sizes
```

## 📝 API Response Example

```json
{
  "state": "California",
  "model_used": "XGBoost",
  "forecast_horizon": 8,
  "predictions": [
    {
      "date": "2024-01-07",
      "predicted_sales": 15234.56,
      "week_number": 1
    },
    ...
  ],
  "metadata": {
    "validation_mae": 1234.56,
    "validation_rmse": 1567.89,
    "validation_mape": 8.45,
    "training_data_points": 156,
    "generated_at": "2024-01-01T12:00:00"
  }
}
```

## 🧪 Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get states
curl http://localhost:8000/states

# Generate forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "horizon": 8}'
```

### Using Python

```python
import requests

# Generate forecast
response = requests.post(
    "http://localhost:8000/forecast",
    json={"state": "California", "horizon": 8}
)
forecast = response.json()
print(forecast)
```

## 🎓 Key Design Decisions

1. **Time Series Split**: No random shuffling - maintains temporal order
2. **No Data Leakage**: Features created separately for train/val/test
3. **Automatic Model Selection**: Based on validation performance
4. **Scalable Architecture**: Easy to add new models or features
5. **Production Ready**: Error handling, logging, API documentation
6. **State-wise Models**: Separate models for each state to capture local patterns

## 📊 Performance Metrics

- **MAE**: Mean Absolute Error (same units as target)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAPE**: Mean Absolute Percentage Error (scale-independent)

## 🚀 Deployment Considerations

For production deployment:

1. **Containerization**: Use Docker for consistent environments
2. **Scaling**: Deploy with Kubernetes or cloud services
3. **Monitoring**: Add logging and monitoring (Prometheus, Grafana)
4. **Caching**: Cache predictions for frequently requested forecasts
5. **Authentication**: Add API key or OAuth authentication
6. **Rate Limiting**: Implement rate limiting for API endpoints
7. **Model Versioning**: Track model versions and performance over time

## 📄 License

This project is designed as a case study for data science interviews and production systems.

## 👨‍💻 Author

**Payaswini**

Built as a high-quality, production-ready data science project demonstrating:

- End-to-end machine learning pipeline
- Multiple forecasting techniques (SARIMA, Prophet, XGBoost, LSTM)
- Automated model comparison and selection
- Production-grade REST API architecture
- Advanced feature engineering for time series
- Scalable deployment and testing workflows
- Industry best practices in forecasting systems
