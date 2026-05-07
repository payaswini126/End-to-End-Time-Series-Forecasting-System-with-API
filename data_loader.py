"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and initial preprocessing"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        logger.info(f"Loading data from {self.file_path}")
        self.raw_data = pd.read_excel(self.file_path)
        logger.info(f"Data loaded: {self.raw_data.shape}")
        logger.info(f"Columns: {self.raw_data.columns.tolist()}")
        return self.raw_data
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data"""
        df = self.raw_data.copy()
        
        # Identify columns
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        state_col = [col for col in df.columns if 'state' in col.lower()][0]
        sales_col = [col for col in df.columns if 'total' in col.lower() or 'sales' in col.lower() or 'value' in col.lower()][0]
        
        # Drop Category column if it exists (only 1 category in dataset)
        if 'Category' in df.columns:
            df = df.drop(columns=['Category'])
        
        # Rename columns
        df = df.rename(columns={
            date_col: 'Date',
            state_col: 'State',
            sales_col: 'Sales'
        })
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by state and date
        df = df.sort_values(['State', 'Date']).reset_index(drop=True)
        
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"States: {df['State'].unique().tolist()}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        self.processed_data = df
        return df
    
    def handle_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates for each state"""
        logger.info("Handling missing dates...")
        
        all_states = []
        for state in df['State'].unique():
            state_df = df[df['State'] == state].copy()
            
            # Create complete date range
            date_range = pd.date_range(
                start=state_df['Date'].min(),
                end=state_df['Date'].max(),
                freq='W'
            )
            
            # Reindex to fill missing dates
            state_df = state_df.set_index('Date').reindex(date_range)
            state_df['State'] = state
            state_df.index.name = 'Date'
            state_df = state_df.reset_index()
            
            all_states.append(state_df)
        
        df_complete = pd.concat(all_states, ignore_index=True)
        logger.info(f"After filling dates: {df_complete.shape}")
        
        return df_complete
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing sales values"""
        logger.info("Handling missing values...")
        
        for state in df['State'].unique():
            mask = df['State'] == state
            
            # Forward fill then backward fill
            df.loc[mask, 'Sales'] = df.loc[mask, 'Sales'].fillna(method='ffill').fillna(method='bfill')
            
            # If still missing, use mean
            if df.loc[mask, 'Sales'].isnull().any():
                df.loc[mask, 'Sales'] = df.loc[mask, 'Sales'].fillna(df.loc[mask, 'Sales'].mean())
        
        logger.info(f"Missing values after handling: {df['Sales'].isnull().sum()}")
        return df
    
    def get_state_data(self, df: pd.DataFrame, state: str) -> pd.DataFrame:
        """Get data for a specific state"""
        return df[df['State'] == state].copy()
