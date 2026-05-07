# Quick Start Guide

Get the Sales Forecasting System up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Excel file: `Forecasting Case- Study.xlsx` in project directory

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- pandas, numpy, scikit-learn
- statsmodels, prophet, xgboost, tensorflow
- fastapi, uvicorn

### Step 2: Verify Installation

```bash
python -c "import pandas, prophet, xgboost, tensorflow, fastapi; print('All packages installed!')"
```

## Running the System

### Option 1: Complete Pipeline (Recommended)

Run everything with one command:

```bash
python run_pipeline.py
```

This will:
1. ✓ Check dependencies and data file
2. ✓ Train all models for all states
3. ✓ Generate visualizations
4. ✓ Start the API server

### Option 2: Step-by-Step

#### Train Models

```bash
python train.py
```

Expected output:
- Processed data in `data/`
- Trained models in `models/`
- Results and forecasts in `results/`
- Training logs in `logs/`

#### Generate Visualizations

```bash
python visualize_results.py
```

Creates plots in `results/`:
- `model_comparison.png` - Performance comparison
- `forecasts.png` - All state forecasts
- `historical_forecast_{state}.png` - Individual state plots

#### Start API Server

```bash
python api.py
```

Or with uvicorn:

```bash
uvicorn api:app --reload
```

## Using the API

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

### 2. Get Available States

```bash
curl http://localhost:8000/states
```

### 3. Get State Information

```bash
curl http://localhost:8000/state/California
```

### 4. Generate Forecast

**Using curl:**

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "horizon": 8}'
```

**Using Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/forecast",
    json={"state": "California", "horizon": 8}
)

forecast = response.json()
print(f"Model: {forecast['model_used']}")
print(f"MAPE: {forecast['metadata']['validation_mape']:.2f}%")

for pred in forecast['predictions']:
    print(f"Week {pred['week_number']}: ${pred['predicted_sales']:,.2f}")
```

### 5. Interactive API Documentation

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing the API

Run the test suite:

```bash
python test_api.py
```

This tests all endpoints and validates responses.

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t sales-forecasting-api .

# Run container
docker run -p 8000:8000 sales-forecasting-api
```

### Using Docker Compose

```bash
docker-compose up -d
```

## Project Structure

```
.
├── train.py                    # Training pipeline
├── api.py                      # FastAPI server
├── run_pipeline.py             # Complete pipeline runner
├── test_api.py                 # API tests
├── visualize_results.py        # Visualization generator
├── config.py                   # Configuration
├── data_loader.py              # Data preprocessing
├── feature_engineering.py      # Feature creation
├── models.py                   # Model implementations
├── model_trainer.py            # Training logic
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose config
└── README.md                   # Full documentation
```

## Common Issues

### Issue: Import errors

**Solution:** Ensure all packages are installed
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Data file not found

**Solution:** Ensure Excel file is in project root
```bash
ls "Forecasting Case- Study.xlsx"
```

### Issue: Port 8000 already in use

**Solution:** Use a different port
```bash
uvicorn api:app --port 8001
```

### Issue: TensorFlow warnings

**Solution:** These are normal. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

## Next Steps

1. **Explore Results**: Check `results/` directory for forecasts and plots
2. **Review Logs**: Check `logs/training.log` for detailed training info
3. **Customize**: Edit `config.py` to adjust parameters
4. **Integrate**: Use the API in your applications
5. **Deploy**: Use Docker for production deployment

## Performance Tips

- **Faster Training**: Disable LSTM in `config.py` if not needed
- **Better Accuracy**: Increase validation weeks in `config.py`
- **More Features**: Add custom features in `feature_engineering.py`
- **Model Tuning**: Adjust hyperparameters in `models.py`

## Support

For issues or questions:
1. Check the full README.md
2. Review training logs in `logs/`
3. Test with `test_api.py`
4. Check API docs at `/docs`

## Success Checklist

- [ ] Dependencies installed
- [ ] Data file present
- [ ] Training completed successfully
- [ ] Models saved in `models/`
- [ ] API server running
- [ ] Health check returns 200
- [ ] Forecast generated successfully
- [ ] Visualizations created

Congratulations! Your forecasting system is ready! 🚀
