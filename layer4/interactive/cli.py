"""Interactive command-line interface"""
import sys
sys.path.insert(0, '../..')

def interactive_cli():
    """Run interactive CLI"""
    print("🤖 StochClaim Agent - Interactive Mode")
    print("Type 'quit' to exit\n")
    
    from .quick_check import is_fraud
    
    while True:
        print("-" * 50)
        
        claim_id = input("Claim ID (or 'quit'): ")
        if claim_id.lower() == 'quit':
            break
        
        try:
            # Collect inputs
            data = {
                'claim_id': claim_id,
                'months_as_customer': int(input("Months as customer: ")),
                'age': int(input("Age: ")),
                'policy_annual_premium': float(input("Annual premium: ")),
                'incident_severity': input("Severity (Trivial/Minor/Major/Total): "),
                'total_claim_amount': float(input("Total claim amount: ")),
                'injury_claim': float(input("Injury claim: ")),
                'property_claim': float(input("Property claim: ")),
                'vehicle_claim': float(input("Vehicle claim: ")),
                'incident_type': input("Incident type: "),
                'collision_type': input("Collision type: "),
                'authorities_contacted': input("Authorities (Police/Fire/Ambulance/None): "),
                'witness_present': input("Witness? (Yes/No): "),
                'police_report_available': input("Police report? (Yes/No/Unknown): ")
            }
            
            # Process
            print("\n🔄 Processing...")
            result = is_fraud(**data)
            
            # Output
            print("\n" + "=" * 50)
            print("🎯 AGENT DECISION")
            print("=" * 50)
            print(f"Fraudulent: {'YES' if result['is_fraudulent'] else 'NO'}")
            print(f"Probability: {result['fraud_probability']:.1%}")
            print(f"Action: {result['decision'].upper()}")
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"Risk Score: {result['risk_score']}/100")
            print(f"\nExplanation: {result['explanation']}")
            
            if result['requires_human_review']:
                print("\n⚠️  HUMAN REVIEW REQUIRED")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_cli()