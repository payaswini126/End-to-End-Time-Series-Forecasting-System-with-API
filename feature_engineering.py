"""
Feature engineering module for time series forecasting
"""
import pandas as pd
import numpy as np
import holidays
from typing import List
import config
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for time series forecasting"""
    
    def __init__(self):
        self.us_holidays = holidays.US()
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lag features"""
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df['Sales'].shift(lag)
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()
        for window in windows:
            df[f'rolling_mean_{window}'] = df['Sales'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['Sales'].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df['Sales'].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df['Sales'].rolling(window=window, min_periods=1).max()
        return df
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-based features"""
        df = df.copy()
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['week'] = df['Date'].dt.isocalendar().week
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['quarter'] = df['Date'].dt.quarter
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday features"""
        df = df.copy()
        df['is_holiday'] = df['Date'].apply(lambda x: int(x in self.us_holidays))
        
        # Days to/from holiday
        df['days_to_holiday'] = 0
        df['days_from_holiday'] = 0
        
        for idx, row in df.iterrows():
            date = row['Date']
            # Check next 30 days for holiday
            for i in range(1, 31):
                if (date + pd.Timedelta(days=i)) in self.us_holidays:
                    df.at[idx, 'days_to_holiday'] = i
                    break
            # Check previous 30 days for holiday
            for i in range(1, 31):
                if (date - pd.Timedelta(days=i)) in self.us_holidays:
                    df.at[idx, 'days_from_holiday'] = i
                    break
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding for periodic features"""
        df = df.copy()
        
        # Month cyclical
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of week cyclical
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Week cyclical
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        logger.info("Creating features...")
        
        df = self.create_date_features(df)
        df = self.create_holiday_features(df)
        df = self.create_cyclical_features(df)
        df = self.create_lag_features(df, config.LAG_FEATURES)
        df = self.create_rolling_features(df, config.ROLLING_WINDOWS)
        
        logger.info(f"Features created. Total columns: {len(df.columns)}")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding Date, State, Sales)"""
        exclude = ['Date', 'State', 'Sales']
        return [col for col in df.columns if col not in exclude]
