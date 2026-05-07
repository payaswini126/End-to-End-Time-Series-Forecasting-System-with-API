"""
Visualization script for forecasting results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_model_comparison():
    """Plot model comparison across states"""
    results_file = config.RESULTS_DIR / 'model_comparison.csv'
    
    if not results_file.exists():
        print("No results file found. Run train.py first.")
        return
    
    df = pd.read_csv(results_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Model Performance Comparison Across States', fontsize=16, fontweight='bold')
    
    # MAPE comparison
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('MAPE')
    colors = sns.color_palette("husl", len(df))
    ax1.barh(df_sorted['State'], df_sorted['MAPE'], color=colors)
    ax1.set_xlabel('MAPE (%)', fontweight='bold')
    ax1.set_title('Mean Absolute Percentage Error by State')
    ax1.invert_yaxis()
    
    # MAE comparison
    ax2 = axes[0, 1]
    df_sorted = df.sort_values('MAE')
    ax2.barh(df_sorted['State'], df_sorted['MAE'], color=colors)
    ax2.set_xlabel('MAE', fontweight='bold')
    ax2.set_title('Mean Absolute Error by State')
    ax2.invert_yaxis()
    
    # RMSE comparison
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('RMSE')
    ax3.barh(df_sorted['State'], df_sorted['RMSE'], color=colors)
    ax3.set_xlabel('RMSE', fontweight='bold')
    ax3.set_title('Root Mean Squared Error by State')
    ax3.invert_yaxis()
    
    # Best model distribution
    ax4 = axes[1, 1]
    model_counts = df['Best_Model'].value_counts()
    ax4.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Best Model Distribution')
    
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {config.RESULTS_DIR / 'model_comparison.png'}")
    plt.close()


def plot_forecasts():
    """Plot forecasts for all states"""
    forecast_file = config.RESULTS_DIR / 'all_forecasts.csv'
    
    if not forecast_file.exists():
        print("No forecast file found. Run train.py first.")
        return
    
    df = pd.read_csv(forecast_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    states = df['State'].unique()
    n_states = len(states)
    
    fig, axes = plt.subplots(n_states, 1, figsize=(14, 4*n_states))
    if n_states == 1:
        axes = [axes]
    
    fig.suptitle('8-Week Sales Forecasts by State', fontsize=16, fontweight='bold')
    
    for idx, state in enumerate(states):
        state_df = df[df['State'] == state]
        ax = axes[idx]
        
        ax.plot(state_df['Date'], state_df['Predicted_Sales'], marker='o', linewidth=2, markersize=8)
        ax.fill_between(state_df['Date'], state_df['Predicted_Sales'], alpha=0.3)
        ax.set_title(f"{state} - Model: {state_df['Model'].iloc[0]}", fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Sales')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for x, y in zip(state_df['Date'], state_df['Predicted_Sales']):
            ax.text(x, y, f'${y:,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'forecasts.png', dpi=300, bbox_inches='tight')
    print(f"Forecast plot saved to {config.RESULTS_DIR / 'forecasts.png'}")
    plt.close()


def plot_historical_with_forecast(state):
    """Plot historical data with forecast for a specific state"""
    # Load processed data
    data_file = config.DATA_DIR / 'processed_data.csv'
    forecast_file = config.RESULTS_DIR / f'forecast_{state}.csv'
    
    if not data_file.exists() or not forecast_file.exists():
        print(f"Data files not found for {state}")
        return
    
    historical = pd.read_csv(data_file)
    historical = historical[historical['State'] == state].copy()
    historical['Date'] = pd.to_datetime(historical['Date'])
    
    forecast = pd.read_csv(forecast_file)
    forecast['Date'] = pd.to_datetime(forecast['Date'])
    
    plt.figure(figsize=(16, 6))
    
    # Plot historical
    plt.plot(historical['Date'], historical['Sales'], label='Historical Sales', 
             linewidth=2, marker='o', markersize=4, alpha=0.7)
    
    # Plot forecast
    plt.plot(forecast['Date'], forecast['Predicted_Sales'], label='Forecast', 
             linewidth=2, marker='s', markersize=6, color='red', linestyle='--')
    
    # Add vertical line at forecast start
    plt.axvline(x=historical['Date'].iloc[-1], color='gray', linestyle=':', linewidth=2, 
                label='Forecast Start')
    
    plt.title(f'{state} - Historical Sales and 8-Week Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Sales', fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / f'historical_forecast_{state}.png', dpi=300, bbox_inches='tight')
    print(f"Historical + forecast plot saved for {state}")
    plt.close()


def main():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1. Model comparison plot...")
    plot_model_comparison()
    
    print("\n2. Forecast plots...")
    plot_forecasts()
    
    print("\n3. Historical + forecast plots...")
    # Load states from results
    results_file = config.RESULTS_DIR / 'model_comparison.csv'
    if results_file.exists():
        df = pd.read_csv(results_file)
        for state in df['State'].unique():
            plot_historical_with_forecast(state)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETED")
    print("="*70)
    print(f"\nAll plots saved in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
