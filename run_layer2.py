# run_layer2.py
#!/usr/bin/env python3
"""
Main entry point to run Layer 2 Stochastic Analysis
"""

import os
import sys
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
# sys.path.insert(0, str(Path(__file__).parent)) # Already in CWD

from layer2.core import StochasticEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("layer2.runner")

def run_layer2():
    """Execute Layer 2 stochastic pipeline"""
    logger.info("Starting Layer 2 Stochastic Analysis...")
    start_time = datetime.now()
    
    # 1. Load data from Layer 1
    input_path = "data/processed/processed_features.csv"
    if not os.path.exists(input_path):
        logger.error(f"Input data not found at {os.path.abspath(input_path)}")
        return
        
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 2. Extract features for HMM
    # These match the feature groups in processing_report.json
    hmm_features = [
        "age_scaled", 
        "total_claim_amount_scaled", 
        "claim_to_premium_ratio_scaled",
        "injury_ratio_scaled",
        "property_ratio_scaled",
        "vehicle_ratio_scaled",
        "severity_score_scaled",
        "complexity_score_scaled",
        "red_flag_count"
    ]
    
    # 3. Initialize and fit ensemble
    ensemble = StochasticEnsemble()
    
    try:
        # Fit models
        logger.info("Fitting models...")
        ensemble.fit(df, hmm_features=hmm_features)
        
        # Generate predictions
        logger.info("Generating predictions for all claims...")
        results_df = ensemble.predict(df)
        
        # 4. Save results
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = os.path.join(output_dir, "stochastic_predictions.csv")
        results_df.to_csv(results_path, index=False)
        
        # 5. Create report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "batch_id": f"L2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": end_time.isoformat(),
            "processing_time_seconds": duration,
            "record_count": len(results_df),
            "model_summaries": ensemble.summary(),
            "output_files": {
                "results_csv": results_path
            },
            "aggregate_metrics": {
                "avg_fraud_probability": float(results_df['fraud_probability'].mean()),
                "avg_expected_steps": float(results_df['expected_investigation_steps'].mean()),
                "state_counts": results_df['hmm_state'].value_counts().to_dict(),
                "action_counts": results_df['optimal_action'].value_counts().to_dict()
            }
        }
        
        report_path = "layer2_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\n[SUCCESS] Layer 2 completed successfully!")
        print(f"  Processed {len(results_df)} claims in {duration:.2f}s")
        print(f"  Average Fraud Probability: {report['aggregate_metrics']['avg_fraud_probability']:.2%}")
        print(f"  Results saved to: {results_path}")
        print(f"  Report saved to: {report_path}")
        
        return results_df, report
        
    except Exception as e:
        logger.exception(f"Layer 2 execution failed: {str(e)}")
        print(f"\n[FAILED] Layer 2 execution failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    run_layer2()
