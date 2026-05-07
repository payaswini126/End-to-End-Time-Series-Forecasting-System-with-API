"""
FastAPI REST API for forecasting service
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import config
from model_trainer import ModelTrainer
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Production-ready time series forecasting API for sales prediction"
)

# Global variables
model_trainer = None
available_states = []
data_loaded = False


# Pydantic models
class ForecastRequest(BaseModel):
    state: str = Field(..., description="State name for forecasting")
    horizon: int = Field(default=8, ge=1, le=52, description="Number of weeks to forecast")


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast_horizon: int
    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ModelMetrics(BaseModel):
    model_name: str
    mae: float
    rmse: float
    mape: float


class StateInfo(BaseModel):
    state: str
    best_model: str
    metrics: List[ModelMetrics]
    data_points: int


class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    available_states: List[str]
    models_trained: int


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global model_trainer, available_states, data_loaded
    
    logger.info("Starting Forecasting API...")
    
    try:
        # Load data
        loader = DataLoader(config.DATA_FILE)
        raw_data = loader.load_data()
        processed_data = loader.preprocess()
        processed_data = loader.handle_missing_dates(processed_data)
        processed_data = loader.handle_missing_values(processed_data)
        
        # Get available states
        all_states = sorted(processed_data['State'].unique().tolist())
        
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Load pre-trained models
        logger.info("Loading pre-trained models...")
        import os
        from pathlib import Path
        
        models_dir = Path(config.MODELS_DIR)
        available_states = []
        
        for state in all_states:
            model_file = models_dir / f"{state}_models.pkl"
            if model_file.exists():
                try:
                    model_trainer.load_models(state)
                    available_states.append(state)
                    logger.info(f"Loaded models for {state}")
                except Exception as e:
                    logger.warning(f"Could not load models for {state}: {e}")
        
        logger.info(f"Loaded models for {len(available_states)} states: {available_states}")
        
        data_loaded = True
        logger.info("API initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        data_loaded = False


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sales Forecasting API",
        "version": config.API_VERSION,
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "states": "/states",
            "state_info": "/state/{state}",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if data_loaded else "initializing",
        data_loaded=data_loaded,
        available_states=available_states,
        models_trained=len(model_trainer.results) if model_trainer else 0
    )


@app.get("/states", response_model=List[str])
async def get_states():
    """Get list of available states"""
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Service is initializing")
    
    return available_states


@app.get("/state/{state}", response_model=StateInfo)
async def get_state_info(state: str):
    """Get information about a specific state's models"""
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Service is initializing")
    
    if state not in available_states:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found")
    
    try:
        state_results = model_trainer.results[state]
        
        # Compile metrics for all models
        metrics_list = []
        for model_name, result in state_results['all_results'].items():
            metrics = result['metrics']
            metrics_list.append(ModelMetrics(
                model_name=model_name,
                mae=metrics['MAE'],
                rmse=metrics['RMSE'],
                mape=metrics['MAPE']
            ))
        
        return StateInfo(
            state=state,
            best_model=state_results['best_model'],
            metrics=metrics_list,
            data_points=len(state_results['train_data'])
        )
    
    except Exception as e:
        logger.error(f"Error getting state info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate sales forecast for a state"""
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Service is initializing")
    
    if request.state not in available_states:
        raise HTTPException(
            status_code=404,
            detail=f"State '{request.state}' not found. Available states: {available_states}"
        )
    
    try:
        # Generate forecast
        forecast_df = model_trainer.generate_forecast(request.state, request.horizon)
        
        # Convert to response format
        predictions = []
        for _, row in forecast_df.iterrows():
            predictions.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "predicted_sales": float(row['Predicted_Sales']),
                "week_number": len(predictions) + 1
            })
        
        # Get model info
        state_results = model_trainer.results[request.state]
        best_model_name = state_results['best_model']
        best_model_metrics = state_results['all_results'][best_model_name]['metrics']
        
        return ForecastResponse(
            state=request.state,
            model_used=best_model_name,
            forecast_horizon=request.horizon,
            predictions=predictions,
            metadata={
                "validation_mae": float(best_model_metrics['MAE']),
                "validation_rmse": float(best_model_metrics['RMSE']),
                "validation_mape": float(best_model_metrics['MAPE']),
                "training_data_points": len(state_results['train_data']),
                "generated_at": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{state}", response_model=ForecastResponse)
async def get_forecast(
    state: str,
    horizon: int = Query(default=8, ge=1, le=52, description="Number of weeks to forecast")
):
    """Generate sales forecast for a state (GET method)"""
    request = ForecastRequest(state=state, horizon=horizon)
    return await generate_forecast(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
