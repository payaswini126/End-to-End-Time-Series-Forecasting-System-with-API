"""
Forecasting models implementation
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None
    layers = None
import warnings
import logging
from typing import Dict, Tuple, Any
import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BaseForecaster:
    """Base class for all forecasting models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate forecasting metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }


class SARIMAForecaster(BaseForecaster):
    """SARIMA model implementation"""
    
    def __init__(self):
        super().__init__("SARIMA")
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 52)  # Weekly seasonality
    
    def fit(self, train_data: pd.DataFrame):
        """Fit SARIMA model"""
        logger.info(f"Training {self.name}...")
        
        try:
            self.model = SARIMAX(
                train_data['Sales'],
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = self.model.fit(disp=False, maxiter=200)
            logger.info(f"{self.name} training completed")
        except Exception as e:
            logger.error(f"Error training {self.name}: {e}")
            # Fallback to simpler model
            self.seasonal_order = (0, 0, 0, 0)
            self.model = SARIMAX(train_data['Sales'], order=self.order, seasonal_order=self.seasonal_order)
            self.model = self.model.fit(disp=False)
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions"""
        forecast = self.model.forecast(steps=steps)
        return forecast.values
    
    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model"""
        predictions = self.predict(len(val_data))
        self.metrics = self.calculate_metrics(val_data['Sales'].values, predictions)
        return self.metrics


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet model implementation"""
    
    def __init__(self):
        super().__init__("Prophet")
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
    
    def fit(self, train_data: pd.DataFrame):
        """Fit Prophet model"""
        logger.info(f"Training {self.name}...")
        
        prophet_df = pd.DataFrame({
            'ds': train_data['Date'],
            'y': train_data['Sales']
        })
        
        self.model.fit(prophet_df)
        logger.info(f"{self.name} training completed")
    
    def predict(self, steps: int, last_date: pd.Timestamp) -> np.ndarray:
        """Generate predictions"""
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future_df)
        return forecast['yhat'].values
    
    def validate(self, val_data: pd.DataFrame, last_train_date: pd.Timestamp) -> Dict[str, float]:
        """Validate model"""
        predictions = self.predict(len(val_data), last_train_date)
        self.metrics = self.calculate_metrics(val_data['Sales'].values, predictions)
        return self.metrics


class XGBoostForecaster(BaseForecaster):
    """XGBoost model with lag features"""
    
    def __init__(self):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        self.feature_columns = None
    
    def fit(self, train_data: pd.DataFrame, feature_columns: list):
        """Fit XGBoost model"""
        logger.info(f"Training {self.name}...")
        
        self.feature_columns = feature_columns
        X_train = train_data[feature_columns].fillna(0)
        y_train = train_data['Sales']
        
        self.model.fit(X_train, y_train)
        logger.info(f"{self.name} training completed")
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X_test = test_data[self.feature_columns].fillna(0)
        return self.model.predict(X_test)
    
    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model"""
        predictions = self.predict(val_data)
        self.metrics = self.calculate_metrics(val_data['Sales'].values, predictions)
        return self.metrics


class LSTMForecaster(BaseForecaster):
    """LSTM deep learning model"""
    
    def __init__(self, lookback: int = 12):
        super().__init__("LSTM")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please install it to use LSTM.")
        self.lookback = lookback
        self.scaler_X = None
        self.scaler_y = None
        self.feature_columns = None
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.lookback):
            X_seq.append(X[i:i + self.lookback])
            y_seq.append(y[i + self.lookback])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, train_data: pd.DataFrame, feature_columns: list):
        """Fit LSTM model"""
        logger.info(f"Training {self.name}...")
        
        from sklearn.preprocessing import StandardScaler
        
        self.feature_columns = feature_columns
        X_train = train_data[feature_columns].fillna(0).values
        y_train = train_data['Sales'].values.reshape(-1, 1)
        
        # Scale features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        
        if len(X_seq) < 10:
            logger.warning(f"Not enough data for LSTM training: {len(X_seq)} sequences")
            return
        
        # Build model
        self.model = keras.Sequential([
            layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(self.lookback, X_train.shape[1])),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train
        self.model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=16,
            verbose=0,
            validation_split=0.1
        )
        
        logger.info(f"{self.name} training completed")
    
    def predict(self, test_data: pd.DataFrame, history_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            return np.zeros(len(test_data))
        
        # Combine history and test for feature generation
        combined = pd.concat([history_data, test_data], ignore_index=True)
        X_combined = combined[self.feature_columns].fillna(0).values
        X_scaled = self.scaler_X.transform(X_combined)
        
        predictions = []
        for i in range(len(history_data), len(combined)):
            if i < self.lookback:
                continue
            X_seq = X_scaled[i - self.lookback:i].reshape(1, self.lookback, -1)
            pred_scaled = self.model.predict(X_seq, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled)
            predictions.append(pred[0, 0])
        
        return np.array(predictions)
    
    def validate(self, val_data: pd.DataFrame, history_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model"""
        predictions = self.predict(val_data, history_data)
        if len(predictions) == 0:
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf')}
        self.metrics = self.calculate_metrics(val_data['Sales'].values[-len(predictions):], predictions)
        return self.metrics
