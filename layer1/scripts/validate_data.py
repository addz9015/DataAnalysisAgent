# layer1/scripts/validate_data.py
#!/usr/bin/env python3
"""
Standalone script to validate data without full preprocessing
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from layer1.core.intake import DataIntake
from layer1.core.validation import DataValidator

def main():
    parser = argparse.ArgumentParser(description='Validate insurance claim data')
    parser.add_argument('input', help='Input CSV file to validate')
    parser.add_argument('--output', '-o', help='Output report file (JSON)', 
                       default='validation_report.json')
    parser.add_argument('--strict', action='store_true', 
                       help='Enable strict mode (reject all invalid)')
    parser.add_argument('--sample', type=int, 
                       help='Validate only first N rows')
    
    args = parser.parse_args()
    
    print(f"Validating: {args.input}")
    
    try:
        # Intake
        intake = DataIntake()
        raw_df = intake.receive(args.input)
        
        if args.sample:
            raw_df = raw_df.head(args.sample)
            print(f"Sampling first {args.sample} rows")
        
        print(f"Total rows: {len(raw_df)}")
        
        # Validate
        validator = DataValidator(strict_mode=args.strict)
        valid_df, report = validator.validate_batch(raw_df)
        
        # Print summary
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total records: {report['summary']['total_records']}")
        print(f"Valid records: {report['summary']['valid_records']}")
        print(f"Invalid records: {report['summary']['invalid_records']}")
        print(f"Valid percentage: {report['summary']['valid_percentage']:.2f}%")
        
        if report['error_breakdown']:
            print(f"\nTop errors:")
            for error, count in list(report['error_breakdown'].items())[:5]:
                print(f"  - {error}: {count} occurrences")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        # Save report
        validator.save_report(report, args.output)
        print(f"\nDetailed report saved to: {args.output}")
        
        # Exit with error code if validation failed
        if report['summary']['valid_records'] == 0:
            print("\n✗ No valid records found!")
            sys.exit(1)
        
        print("\n✓ Validation complete")
        
    except Exception as e:
        print(f"\n✗ Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()