# layer1/scripts/run_ingestion.py
#!/usr/bin/env python3
"""
Command-line script to run Layer 1 ingestion
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from layer1.core.pipeline import Layer1Pipeline

def main():
    parser = argparse.ArgumentParser(description='Run Layer 1 Data Ingestion')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV path', default='processed_claims.csv')
    parser.add_argument('--strict', action='store_true', help='Enable strict validation mode')
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = {
        'strict_mode': args.strict,
        'storage_path': './data/processed'
    }
    
    pipeline = Layer1Pipeline(config)
    
    try:
        processed_df, report = pipeline.process(args.input)
        
        print(f"\n✓ Processing complete!")
        print(f"  Records processed: {report['records']['output']}")
        print(f"  Processing time: {report['processing_time_seconds']:.2f}s")
        print(f"  Data quality score: {report['data_quality_score']:.1f}%")
        print(f"  Markov states: {report['markov_state_distribution']}")
        print(f"\n  Output saved to: {report['outputs']['csv_export_path']}")
        
        # Save report
        import json
        with open('processing_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved to: processing_report.json")
        
    except Exception as e:
        print(f"\n✗ Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()