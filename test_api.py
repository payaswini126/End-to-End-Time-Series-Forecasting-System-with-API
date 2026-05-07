"""
API testing script
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_get_states():
    """Test get states endpoint"""
    print("\n" + "="*50)
    print("Testing Get States Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/states")
    print(f"Status Code: {response.status_code}")
    states = response.json()
    print(f"Available States: {states}")
    return states if response.status_code == 200 else []


def test_state_info(state):
    """Test state info endpoint"""
    print("\n" + "="*50)
    print(f"Testing State Info Endpoint for: {state}")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/state/{state}")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def test_forecast_post(state, horizon=8):
    """Test forecast endpoint (POST)"""
    print("\n" + "="*50)
    print(f"Testing Forecast Endpoint (POST) for: {state}")
    print("="*50)
    
    payload = {
        "state": state,
        "horizon": horizon
    }
    
    response = requests.post(
        f"{BASE_URL}/forecast",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nState: {result['state']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Forecast Horizon: {result['forecast_horizon']}")
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        print(f"\nFirst 3 Predictions:")
        for pred in result['predictions'][:3]:
            print(f"  Week {pred['week_number']}: {pred['date']} -> ${pred['predicted_sales']:.2f}")
    else:
        print(f"Error: {response.text}")


def test_forecast_get(state, horizon=8):
    """Test forecast endpoint (GET)"""
    print("\n" + "="*50)
    print(f"Testing Forecast Endpoint (GET) for: {state}")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/forecast/{state}?horizon={horizon}")
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model Used: {result['model_used']}")
        print(f"Validation MAPE: {result['metadata']['validation_mape']:.2f}%")
    else:
        print(f"Error: {response.text}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SALES FORECASTING API - TEST SUITE")
    print("="*70)
    
    # Wait for API to be ready
    print("\nWaiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("API is ready!")
                break
        except:
            pass
        time.sleep(2)
        print(f"Retry {i+1}/{max_retries}...")
    
    # Run tests
    if test_health():
        states = test_get_states()
        
        if states:
            # Test first state
            test_state = states[0]
            test_state_info(test_state)
            test_forecast_post(test_state, horizon=8)
            test_forecast_get(test_state, horizon=4)
            
            # Test invalid state
            print("\n" + "="*50)
            print("Testing Invalid State")
            print("="*50)
            response = requests.get(f"{BASE_URL}/forecast/InvalidState")
            print(f"Status Code: {response.status_code}")
            print(f"Expected 404, Got: {response.status_code}")
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
