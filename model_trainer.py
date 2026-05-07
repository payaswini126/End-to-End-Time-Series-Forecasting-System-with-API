"""
Model training and selection module
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import config
from models import SARIMAForecaster, ProphetForecaster, XGBoostForecaster, LSTMForecaster
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and compares multiple forecasting models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_engineer = FeatureEngineer()
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame, state: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train/validation/test splits"""
        state_df = df[df['State'] == state].copy().reset_index(drop=True)
        
        # Create features
        state_df = self.feature_engineer.create_all_features(state_df)
        
        # Time-based split (no leakage)
        n = len(state_df)
        train_size = n - config.VALIDATION_WEEKS - config.TEST_WEEKS
        val_size = config.VALIDATION_WEEKS
        
        train_df = state_df.iloc[:train_size].copy()
        val_df = state_df.iloc[train_size:train_size + val_size].copy()
        test_df = state_df.iloc[train_size + val_size:].copy()
        
        logger.info(f"State: {state}")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame, state: str) -> Dict:
        """Train all models and compare performance"""
        results = {}
        feature_cols = self.feature_engineer.get_feature_columns(train_df)
        
        # SARIMA
        if config.MODELS['SARIMA']['enabled']:
            try:
                sarima = SARIMAForecaster()
                sarima.fit(train_df)
                metrics = sarima.validate(val_df)
                results['SARIMA'] = {'model': sarima, 'metrics': metrics}
                logger.info(f"SARIMA - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            except Exception as e:
                logger.error(f"SARIMA failed: {e}")
        
        # Prophet
        if config.MODELS['Prophet']['enabled']:
            try:
                prophet = ProphetForecaster()
                prophet.fit(train_df)
                last_train_date = train_df['Date'].iloc[-1]
                metrics = prophet.validate(val_df, last_train_date)
                results['Prophet'] = {'model': prophet, 'metrics': metrics}
                logger.info(f"Prophet - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            except Exception as e:
                logger.error(f"Prophet failed: {e}")
        
        # XGBoost
        if config.MODELS['XGBoost']['enabled']:
            try:
                xgb_model = XGBoostForecaster()
                xgb_model.fit(train_df, feature_cols)
                metrics = xgb_model.validate(val_df)
                results['XGBoost'] = {'model': xgb_model, 'metrics': metrics}
                logger.info(f"XGBoost - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            except Exception as e:
                logger.error(f"XGBoost failed: {e}")
        
        # LSTM
        if config.MODELS['LSTM']['enabled']:
            try:
                lstm = LSTMForecaster(lookback=12)
                lstm.fit(train_df, feature_cols)
                metrics = lstm.validate(val_df, train_df)
                results['LSTM'] = {'model': lstm, 'metrics': metrics}
                logger.info(f"LSTM - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            except Exception as e:
                logger.error(f"LSTM failed: {e}")
        
        return results
    
    def select_best_model(self, results: Dict) -> Tuple[str, any]:
        """Select best model based on validation MAPE"""
        best_mape = float('inf')
        best_name = None
        best_model = None
        
        for name, result in results.items():
            mape = result['metrics']['MAPE']
            if mape < best_mape:
                best_mape = mape
                best_name = name
                best_model = result['model']
        
        logger.info(f"Best model: {best_name} with MAPE: {best_mape:.2f}%")
        return best_name, best_model
    
    def train_for_state(self, df: pd.DataFrame, state: str) -> Dict:
        """Complete training pipeline for a state"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training models for state: {state}")
        logger.info(f"{'='*50}")
        
        # Prepare data
        train_df, val_df, test_df = self.prepare_data(df, state)
        
        # Train all models
        results = self.train_all_models(train_df, val_df, state)
        
        # Select best model
        best_name, best_model = self.select_best_model(results)
        
        # Save results
        state_results = {
            'state': state,
            'best_model': best_name,
            'all_results': results,
            'train_data': train_df,
            'val_data': val_df,
            'test_data': test_df
        }
        
        self.results[state] = state_results
        
        return state_results
    
    def save_models(self, state: str):
        """Save trained models"""
        state_results = self.results[state]
        model_path = config.MODELS_DIR / f"{state}_models.pkl"
        
        joblib.dump(state_results, model_path)
        logger.info(f"Models saved to {model_path}")
    
    def load_models(self, state: str) -> Dict:
        """Load trained models"""
        model_path = config.MODELS_DIR / f"{state}_models.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No saved models found for state: {state}")
        
        state_results = joblib.load(model_path)
        self.results[state] = state_results
        logger.info(f"Models loaded from {model_path}")
        
        return state_results
    
    def generate_forecast(self, state: str, horizon: int = None) -> pd.DataFrame:
        """Generate forecast for specified horizon"""
        if horizon is None:
            horizon = config.FORECAST_HORIZON
        
        state_results = self.results[state]
        best_model_name = state_results['best_model']
        best_model = state_results['all_results'][best_model_name]['model']
        train_df = state_results['train_data']
        
        logger.info(f"Generating {horizon}-week forecast for {state} using {best_model_name}")
        
        # Generate future dates
        last_date = train_df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W')
        
        # Generate predictions based on model type
        if best_model_name == 'SARIMA':
            predictions = best_model.predict(horizon)
        elif best_model_name == 'Prophet':
            predictions = best_model.predict(horizon, last_date)
        elif best_model_name in ['XGBoost', 'LSTM']:
            # Create future dataframe with features
            future_df = pd.DataFrame({'Date': future_dates, 'State': state, 'Sales': 0})
            
            # Combine with history for feature generation
            combined = pd.concat([train_df[['Date', 'State', 'Sales']], future_df], ignore_index=True)
            combined = self.feature_engineer.create_all_features(combined)
            future_with_features = combined.iloc[-horizon:].copy()
            
            if best_model_name == 'XGBoost':
                predictions = best_model.predict(future_with_features)
            else:  # LSTM
                predictions = best_model.predict(future_with_features, train_df)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'State': state,
            'Predicted_Sales': predictions,
            'Model': best_model_name
        })
        
        return forecast_df
