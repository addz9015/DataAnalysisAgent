#!/usr/bin/env python3
"""
Main entry point to run Layer 3 Agent & Decision Engine
"""

import os
import sys
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Load env variables from root .env
load_dotenv()

from layer3.core.agent_orchestrator import StochClaimAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("layer3.runner")

def run_layer3():
    """Execute Layer 3 Agent pipeline"""
    logger.info("Starting Layer 3 Agent Engine...")
    start_time = datetime.now()
    
    # 1. Load data from Layer 2
    input_path = "data/processed/stochastic_predictions.csv"
    if not os.path.exists(input_path):
        logger.error(f"Input data not found at {os.path.abspath(input_path)}")
        return
        
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check if GROQ_API_KEY is properly loaded to enable LLM features
    use_llm = bool(os.getenv("GROQ_API_KEY"))
    if use_llm:
        logger.info("GROQ API key found. Enabling LLM explanation generation.")
    else:
        logger.warning("GROQ API key NOT found. Running without LLM generation.")
    
    # 2. Initialize Agent Engine with LLM enabled
    agent = StochClaimAgent(
        risk_tolerance='balanced',
        use_llm=use_llm
    )
    
    try:
        # 3. Process the claims batch
        logger.info("Processing decisions for all claims...")
        results_df = agent.process_batch(df)
        
        # 4. Save results (CSV)
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        decisions_path = os.path.join(output_dir, "agent_decisions.csv")
        results_df.to_csv(decisions_path, index=False)
        
        # 5. Extract and save individual LLM explanations if applicable
        expl_dir = os.path.join(output_dir, "explanations")
        os.makedirs(expl_dir, exist_ok=True)
        
        if use_llm:
            logger.info("Saving detailed natural language explanations...")
            # We need to process claim by claim again, or we could just do it during batch
            # Actually, `agent.process_batch` flattens the output and DOES NOT include 'natural_language'
            # Let's extract it by re-running agent process_claim if need be or just storing it in the loop.
            # Wait, since `process_batch` does not store 'natural_language' into the dataframe,
            # Let's process claims individually to get the full dictionary response
            
            for _, row in df.iterrows():
                res = agent.process_claim(row)
                claim_id = res['claim_id']
                if 'detailed' in res['explanations']:
                    nl_text = res['explanations']['detailed']
                    with open(os.path.join(expl_dir, f"{claim_id}_explanation.txt"), "w", encoding="utf-8") as f:
                        f.write(nl_text)
                    
                    # Store detailed JSON explanation
                    technical_data = res['explanations'].get('technical', {})
                    if technical_data:
                        with open(os.path.join(expl_dir, f"{claim_id}_technical.json"), "w", encoding="utf-8") as f:
                            # If it's already a JSON string, write it, else dump it
                            if isinstance(technical_data, str):
                                f.write(technical_data)
                            else:
                                json.dump(technical_data, f, indent=2)

        # 6. Create report / feedback log initializer
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "batch_id": f"L3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": end_time.isoformat(),
            "processing_time_seconds": duration,
            "record_count": len(results_df),
            "output_files": {
                "agent_decisions_csv": decisions_path,
                "explanations_dir": expl_dir
            },
            "aggregate_metrics": {
                "action_distribution": results_df['agent_action'].value_counts().to_dict(),
                "requires_human_review_count": int(results_df['requires_human_review'].sum()),
                "avg_confidence": float(results_df['agent_confidence'].mean())
            }
        }
        
        report_path = "layer3_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\n[SUCCESS] Layer 3 completed successfully!")
        print(f"  Processed {len(results_df)} claims in {duration:.2f}s")
        print(f"  Requires Human Review: {report['aggregate_metrics']['requires_human_review_count']} claims")
        print(f"  Results saved to: {decisions_path}")
        print(f"  Report saved to: {report_path}")
        
        return results_df, report
        
    except Exception as e:
        logger.exception(f"Layer 3 execution failed: {str(e)}")
        print(f"\n[FAILED] Layer 3 execution failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    run_layer3()
