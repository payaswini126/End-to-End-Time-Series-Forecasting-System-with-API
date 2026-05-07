"""
Training script - trains models for all states
"""
import logging
import pandas as pd
from pathlib import Path
import config
from data_loader import DataLoader
from model_trainer import ModelTrainer
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("SALES FORECASTING - MODEL TRAINING PIPELINE")
    logger.info("="*70)
    
    # Load and preprocess data
    logger.info("\n1. Loading and preprocessing data...")
    loader = DataLoader(config.DATA_FILE)
    raw_data = loader.load_data()
    processed_data = loader.preprocess()
    processed_data = loader.handle_missing_dates(processed_data)
    processed_data = loader.handle_missing_values(processed_data)
    
    # Save processed data
    processed_data.to_csv(config.DATA_DIR / 'processed_data.csv', index=False)
    logger.info(f"Processed data saved to {config.DATA_DIR / 'processed_data.csv'}")
    
    # Get states
    states = processed_data['State'].unique()
    logger.info(f"\nStates to process: {states.tolist()}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models for each state
    logger.info("\n2. Training models for each state...")
    all_results = []
    
    for state in states:
        try:
            state_results = trainer.train_for_state(processed_data, state)
            trainer.save_models(state)
            
            # Collect results
            best_model = state_results['best_model']
            best_metrics = state_results['all_results'][best_model]['metrics']
            
            all_results.append({
                'State': state,
                'Best_Model': best_model,
                'MAE': best_metrics['MAE'],
                'RMSE': best_metrics['RMSE'],
                'MAPE': best_metrics['MAPE']
            })
            
        except Exception as e:
            logger.error(f"Error processing state {state}: {e}")
            continue
    
    # Save summary results
    logger.info("\n3. Saving results summary...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.RESULTS_DIR / 'model_comparison.csv', index=False)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED - RESULTS SUMMARY")
    logger.info("="*70)
    print("\n", results_df.to_string(index=False))
    
    # Generate sample forecasts
    logger.info("\n4. Generating sample forecasts...")
    forecast_results = []
    
    for state in states:
        try:
            forecast_df = trainer.generate_forecast(state, config.FORECAST_HORIZON)
            forecast_df.to_csv(config.RESULTS_DIR / f'forecast_{state}.csv', index=False)
            forecast_results.append(forecast_df)
            logger.info(f"Forecast saved for {state}")
        except Exception as e:
            logger.error(f"Error generating forecast for {state}: {e}")
    
    # Combine all forecasts
    if forecast_results:
        all_forecasts = pd.concat(forecast_results, ignore_index=True)
        all_forecasts.to_csv(config.RESULTS_DIR / 'all_forecasts.csv', index=False)
        logger.info(f"\nAll forecasts saved to {config.RESULTS_DIR / 'all_forecasts.csv'}")
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"\nModels saved in: {config.MODELS_DIR}")
    logger.info(f"Results saved in: {config.RESULTS_DIR}")
    logger.info(f"Logs saved in: {config.LOGS_DIR}")


if __name__ == "__main__":
    main()
