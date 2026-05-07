"""
Example usage script demonstrating the forecasting system
"""
import pandas as pd
import requests
import json
from pathlib import Path


def example_1_load_and_explore_data():
    """Example 1: Load and explore the data"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load and Explore Data")
    print("="*70)
    
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader("Forecasting Case- Study.xlsx")
    raw_data = loader.load_data()
    
    print(f"\nRaw data shape: {raw_data.shape}")
    print(f"\nFirst few rows:")
    print(raw_data.head())
    
    # Preprocess
    processed_data = loader.preprocess()
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"\nDate range: {processed_data['Date'].min()} to {processed_data['Date'].max()}")
    print(f"\nStates: {processed_data['State'].unique().tolist()}")
    
    # Handle missing data
    processed_data = loader.handle_missing_dates(processed_data)
    processed_data = loader.handle_missing_values(processed_data)
    
    print(f"\nFinal data shape: {processed_data.shape}")
    print(f"Missing values: {processed_data.isnull().sum().sum()}")


def example_2_feature_engineering():
    """Example 2: Create features"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Feature Engineering")
    print("="*70)
    
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Load data
    loader = DataLoader("Forecasting Case- Study.xlsx")
    raw_data = loader.load_data()
    processed_data = loader.preprocess()
    processed_data = loader.handle_missing_dates(processed_data)
    processed_data = loader.handle_missing_values(processed_data)
    
    # Get data for one state
    state_data = processed_data[processed_data['State'] == processed_data['State'].unique()[0]].copy()
    
    print(f"\nOriginal columns: {state_data.columns.tolist()}")
    
    # Create features
    engineer = FeatureEngineer()
    state_data_with_features = engineer.create_all_features(state_data)
    
    print(f"\nAfter feature engineering: {len(state_data_with_features.columns)} columns")
    print(f"\nNew features created:")
    feature_cols = engineer.get_feature_columns(state_data_with_features)
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"  {i}. {col}")
    print(f"  ... and {len(feature_cols) - 10} more features")


def example_3_train_single_model():
    """Example 3: Train a single model"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Train Single Model")
    print("="*70)
    
    from data_loader import DataLoader
    from model_trainer import ModelTrainer
    
    # Load data
    loader = DataLoader("Forecasting Case- Study.xlsx")
    raw_data = loader.load_data()
    processed_data = loader.preprocess()
    processed_data = loader.handle_missing_dates(processed_data)
    processed_data = loader.handle_missing_values(processed_data)
    
    # Train for first state
    trainer = ModelTrainer()
    state = processed_data['State'].unique()[0]
    
    print(f"\nTraining models for: {state}")
    results = trainer.train_for_state(processed_data, state)
    
    print(f"\nBest model: {results['best_model']}")
    print(f"\nAll model results:")
    for model_name, result in results['all_results'].items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")


def example_4_generate_forecast():
    """Example 4: Generate forecast"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Generate Forecast")
    print("="*70)
    
    from data_loader import DataLoader
    from model_trainer import ModelTrainer
    
    # Load data
    loader = DataLoader("Forecasting Case- Study.xlsx")
    raw_data = loader.load_data()
    processed_data = loader.preprocess()
    processed_data = loader.handle_missing_dates(processed_data)
    processed_data = loader.handle_missing_values(processed_data)
    
    # Train and forecast
    trainer = ModelTrainer()
    state = processed_data['State'].unique()[0]
    
    print(f"\nTraining models for: {state}")
    results = trainer.train_for_state(processed_data, state)
    
    print(f"\nGenerating 8-week forecast...")
    forecast_df = trainer.generate_forecast(state, horizon=8)
    
    print(f"\nForecast for {state}:")
    print(f"Model used: {forecast_df['Model'].iloc[0]}")
    print(f"\nPredictions:")
    for idx, row in forecast_df.iterrows():
        print(f"  Week {idx+1}: {row['Date'].strftime('%Y-%m-%d')} -> ${row['Predicted_Sales']:,.2f}")


def example_5_api_usage():
    """Example 5: Use the API"""
    print("\n" + "="*70)
    print("EXAMPLE 5: API Usage")
    print("="*70)
    
    base_url = "http://localhost:8000"
    
    print("\nNote: Make sure the API is running (python api.py)")
    print("Testing API endpoints...\n")
    
    try:
        # Health check
        print("1. Health Check:")
        response = requests.get(f"{base_url}/health", timeout=2)
        print(f"   Status: {response.json()['status']}")
        
        # Get states
        print("\n2. Get States:")
        response = requests.get(f"{base_url}/states", timeout=2)
        states = response.json()
        print(f"   Available states: {states}")
        
        if states:
            state = states[0]
            
            # Get state info
            print(f"\n3. Get State Info ({state}):")
            response = requests.get(f"{base_url}/state/{state}", timeout=2)
            info = response.json()
            print(f"   Best model: {info['best_model']}")
            print(f"   Data points: {info['data_points']}")
            
            # Generate forecast
            print(f"\n4. Generate Forecast ({state}):")
            response = requests.post(
                f"{base_url}/forecast",
                json={"state": state, "horizon": 8},
                timeout=10
            )
            forecast = response.json()
            print(f"   Model used: {forecast['model_used']}")
            print(f"   Validation MAPE: {forecast['metadata']['validation_mape']:.2f}%")
            print(f"   First prediction: ${forecast['predictions'][0]['predicted_sales']:,.2f}")
            
    except requests.exceptions.ConnectionError:
        print("   ✗ API is not running. Start it with: python api.py")
    except Exception as e:
        print(f"   ✗ Error: {e}")


def example_6_compare_models():
    """Example 6: Compare all models"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Compare All Models")
    print("="*70)
    
    results_file = Path("results/model_comparison.csv")
    
    if not results_file.exists():
        print("\nResults file not found. Run train.py first.")
        return
    
    df = pd.read_csv(results_file)
    
    print("\nModel Performance Summary:")
    print(df.to_string(index=False))
    
    print("\n\nBest Model by State:")
    for _, row in df.iterrows():
        print(f"  {row['State']}: {row['Best_Model']} (MAPE: {row['MAPE']:.2f}%)")
    
    print("\n\nModel Selection Distribution:")
    model_counts = df['Best_Model'].value_counts()
    for model, count in model_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {model}: {count} states ({percentage:.1f}%)")
    
    print("\n\nAverage Performance Across All States:")
    print(f"  Average MAE:  {df['MAE'].mean():.2f}")
    print(f"  Average RMSE: {df['RMSE'].mean():.2f}")
    print(f"  Average MAPE: {df['MAPE'].mean():.2f}%")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("SALES FORECASTING SYSTEM - USAGE EXAMPLES")
    print("="*70)
    
    examples = [
        ("Load and Explore Data", example_1_load_and_explore_data),
        ("Feature Engineering", example_2_feature_engineering),
        ("Train Single Model", example_3_train_single_model),
        ("Generate Forecast", example_4_generate_forecast),
        ("API Usage", example_5_api_usage),
        ("Compare Models", example_6_compare_models),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nChoose an example (1-6) or 'all' to run all examples:")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
