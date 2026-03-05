# layer1/scripts/generate_sample.py
#!/usr/bin/env python3
"""
Generate synthetic insurance claim data for testing
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_claims(n_records: int = 100, 
                              fraud_rate: float = 0.1,
                              random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic insurance claim data
    
    Args:
        n_records: Number of records to generate
        fraud_rate: Proportion of fraudulent claims
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Generate base features
    data = {
        'claim_id': [f'C{str(i).zfill(5)}' for i in range(n_records)],
        'months_as_customer': np.random.randint(1, 240, n_records),
        'age': np.random.randint(18, 80, n_records),
        'policy_annual_premium': np.random.uniform(500, 3000, n_records).round(2),
    }
    
    # Incident characteristics
    severities = ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']
    data['incident_severity'] = np.random.choice(severities, n_records, 
                                                 p=[0.2, 0.5, 0.2, 0.1])
    
    incident_types = ['Single Vehicle', 'Multi-Vehicle', 'Vehicle Theft', 'Parked Car']
    data['incident_type'] = np.random.choice(incident_types, n_records)
    
    collision_types = ['Front Collision', 'Rear Collision', 'Side Collision', 'No Collision']
    data['collision_type'] = np.random.choice(collision_types, n_records)
    
    # Authorities and witnesses
    data['authorities_contacted'] = np.random.choice(
        ['Police', 'Fire', 'Ambulance', 'None'], 
        n_records, 
        p=[0.5, 0.1, 0.1, 0.3]
    )
    data['witness_present'] = np.random.choice(['Yes', 'No'], n_records, p=[0.3, 0.7])
    data['police_report_available'] = np.random.choice(
        ['Yes', 'No', 'Unknown'], 
        n_records, 
        p=[0.6, 0.3, 0.1]
    )
    
    # Generate claim amounts (correlated with severity)
    base_amounts = {
        'Trivial Damage': (100, 1000),
        'Minor Damage': (1000, 5000),
        'Major Damage': (5000, 15000),
        'Total Loss': (15000, 50000)
    }
    
    total_claims = []
    injury_claims = []
    property_claims = []
    vehicle_claims = []
    
    for severity in data['incident_severity']:
        low, high = base_amounts[severity]
        total = np.random.uniform(low, high)
        
        # Split into components
        injury_ratio = np.random.uniform(0, 0.4)
        property_ratio = np.random.uniform(0.2, 0.5)
        vehicle_ratio = 1 - injury_ratio - property_ratio
        
        total_claims.append(round(total, 2))
        injury_claims.append(round(total * injury_ratio, 2))
        property_claims.append(round(total * property_ratio, 2))
        vehicle_claims.append(round(total * vehicle_ratio, 2))
    
    data['total_claim_amount'] = total_claims
    data['injury_claim'] = injury_claims
    data['property_claim'] = property_claims
    data['vehicle_claim'] = vehicle_claims
    
    # Generate fraud labels (biased by certain features)
    fraud_labels = []
    for i in range(n_records):
        fraud_prob = fraud_rate
        
        # Increase probability if no police report
        if data['police_report_available'][i] == 'No':
            fraud_prob += 0.1
        
        # Increase if no witness
        if data['witness_present'][i] == 'No':
            fraud_prob += 0.05
        
        # Increase if high claim ratio
        claim_premium_ratio = total_claims[i] / data['policy_annual_premium'][i]
        if claim_premium_ratio > 5:
            fraud_prob += 0.1
        
        fraud_labels.append('Y' if np.random.random() < fraud_prob else 'N')
    
    data['fraud_reported'] = fraud_labels
    
    # Add claim status (most are open, some closed)
    statuses = np.random.choice(
        ['Open', 'Closed'], 
        n_records, 
        p=[0.7, 0.3]
    )
    data['claim_status'] = statuses
    
    # Settlement amount (for closed claims)
    settlements = []
    for i, status in enumerate(statuses):
        if status == 'Closed':
            if fraud_labels[i] == 'Y':
                settlements.append(0)  # Denied
            else:
                settlements.append(round(total_claims[i] * np.random.uniform(0.8, 1.0), 2))
        else:
            settlements.append(0)
    data['settlement_amount'] = settlements
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic insurance claims')
    parser.add_argument('--records', '-n', type=int, default=100,
                       help='Number of records to generate')
    parser.add_argument('--fraud-rate', '-f', type=float, default=0.1,
                       help='Fraud rate (0-1)')
    parser.add_argument('--output', '-o', default='synthetic_claims.csv',
                       help='Output file path')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.records} synthetic claims...")
    print(f"  Fraud rate: {args.fraud_rate*100}%")
    print(f"  Random seed: {args.seed}")
    
    df = generate_synthetic_claims(
        n_records=args.records,
        fraud_rate=args.fraud_rate,
        random_seed=args.seed
    )
    
    # Save
    df.to_csv(args.output, index=False)
    
    print(f"\n✓ Generated {len(df)} records")
    print(f"  Fraudulent: {(df['fraud_reported'] == 'Y').sum()}")
    print(f"  Legitimate: {(df['fraud_reported'] == 'N').sum()}")
    print(f"\nSaved to: {args.output}")

if __name__ == '__main__':
    main()