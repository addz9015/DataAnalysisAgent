import requests
import json
import time
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "dev-key")

BASE_URL = "http://localhost:8000"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def print_result(endpoint, input_data, output):
    print(f"| {endpoint.ljust(20)} | {str(input_data)[:23].ljust(25)} | {str(output).ljust(26)} |")

def test_api():
    print("| Endpoint             | Input                     | Output                     |")
    print("| -------------------- | ------------------------- | -------------------------- |")
    
    # 1. /health/
    try:
        response = requests.get(f"{BASE_URL}/health/", headers=HEADERS)
        print_result("`/health/`", "None", "System status")
    except Exception as e:
         print_result("`/health/`", "None", f"Error: {e}")

    # Dummy Claim Data
    claim_data = {
      "claim_id": "C00001",
      "months_as_customer": 180,
      "age": 65,
      "policy_annual_premium": 2388.88,
      "incident_severity": "Minor Damage",
      "total_claim_amount": 3726.01,
      "injury_claim": 1133.05,
      "property_claim": 1411.01,
      "vehicle_claim": 1181.95,
      "incident_type": "Multi-vehicle Collision",
      "collision_type": "No Collision",
      "authorities_contacted": "Police",
      "witnesses": 0,
      "witness_present": "No",
      "police_report_available": "Yes"
    }

    # 2. /predict/
    try:
        response = requests.post(f"{BASE_URL}/predict/", headers=HEADERS, json=claim_data)
        if response.status_code == 200:
             print_result("`/predict/`", "Single claim JSON", "Decision + explanation")
        else:
             print_result("`/predict/`", "Single claim JSON", f"Error {response.status_code}: {response.text}")
    except Exception as e:
         print_result("`/predict/`", "Single claim JSON", f"Error: {e}")

    # 3. /batch/
    batch_data = {
        "claims": [claim_data, claim_data]
    }
    try:
        response = requests.post(f"{BASE_URL}/batch/", headers=HEADERS, json=batch_data)
        if response.status_code == 200:
             print_result("`/batch/`", "Multiple claims JSON", "Batch results + summary")
        else:
             print_result("`/batch/`", "Multiple claims JSON", f"Error {response.status_code}")
    except Exception as e:
         print_result("`/batch/`", "Multiple claims JSON", f"Error: {e}")

    # 4. /query/ask
    query_data = {
        "question": "Is this claim fraudulent?",
        "claim_data": claim_data
    }
    try:
        response = requests.post(f"{BASE_URL}/query/ask", headers=HEADERS, json=query_data)
        if response.status_code == 200:
             print_result("`/query/ask`", "Natural language query", "Answer + intent")
        else:
             print_result("`/query/ask`", "Natural language query", f"Error {response.status_code}")
    except Exception as e:
         print_result("`/query/ask`", "Natural language query", f"Error: {e}")

    # 5. /query/quick-check
    try:
        response = requests.post(f"{BASE_URL}/query/quick-check", headers=HEADERS, json=claim_data)
        if response.status_code == 200:
             print_result("`/query/quick-check`", "Claim data", "Yes/no fraud + prob")
        else:
             print_result("`/query/quick-check`", "Claim data", f"Error {response.status_code}")
    except Exception as e:
         print_result("`/query/quick-check`", "Claim data", f"Error: {e}")

    # 6. /explain/{id}
    try:
        response = requests.get(f"{BASE_URL}/explain/C00001", headers=HEADERS)
        if response.status_code == 200:
             print_result("`/explain/{id}`", "Claim ID", "Detailed explanation")
        else:
             print_result("`/explain/{id}`", "Claim ID", f"Error {response.status_code}")
    except Exception as e:
         print_result("`/explain/{id}`", "Claim ID", f"Error: {e}")

    # 7. /feedback/
    feedback_data = {
        "claim_id": "C00001",
        "agent_decision": "approve",
        "human_decision": "approve",
        "reason": "Looks good",
        "actual_outcome": "approve"
    }
    try:
        response = requests.post(f"{BASE_URL}/feedback/", headers=HEADERS, json=feedback_data)
        if response.status_code == 200:
             print_result("`/feedback/`", "Correction data", "Confirmation")
        else:
             print_result("`/feedback/`", "Correction data", f"Error {response.status_code}")
    except Exception as e:
         print_result("`/feedback/`", "Correction data", f"Error: {e}")


if __name__ == "__main__":
    test_api()
