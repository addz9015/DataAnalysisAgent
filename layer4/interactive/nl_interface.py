# layer4/interactive/nl_interface.py
"""
Natural language interface for the agent
"""

import sys
sys.path.insert(0, '../..')

import re
from typing import Dict, List, Optional
import pandas as pd

from .quick_check import is_fraud
from layer3.llm.explainer_llm import LLMExplainer


class NLAgentInterface:
    """
    Ask agent questions in natural language
    """
    
    def __init__(self, provider: str = "groq"):
        self.llm = LLMExplainer(provider=provider)
        self.conversation_history = []
        
    def ask(self, question: str, claim_data: Optional[Dict] = None) -> Dict:
        """
        Process natural language question
        
        Examples:
        - "Is this claim fraudulent?"
        - "Why was this denied?"
        - "What should I investigate?"
        - "Compare this to similar claims"
        """
        # Detect intent
        intent = self._parse_intent(question)
        
        # Route to appropriate handler
        handlers = {
            'fraud_check': self._handle_fraud_check,
            'explain_decision': self._handle_explain,
            'investigation_advice': self._handle_investigation,
            'similar_claims': self._handle_similar,
            'help': self._handle_help,
            'general': self._handle_general
        }
        
        handler = handlers.get(intent, self._handle_general)
        answer = handler(question, claim_data)
        
        # Store conversation
        self.conversation_history.append({
            'question': question,
            'intent': intent,
            'answer': answer
        })
        
        return {
            'question': question,
            'intent': intent,
            'answer': answer,
            'claim_data_used': claim_data is not None
        }
    
    def _parse_intent(self, question: str) -> str:
        """Detect user intent from question"""
        q = question.lower()
        
        # Fraud check patterns
        fraud_patterns = [
            r'fraud', r'fraudulent', r'scam', r'fake', 
            r'risk', r'suspicious', r'legitimate'
        ]
        if any(re.search(p, q) for p in fraud_patterns):
            return 'fraud_check'
        
        # Explanation patterns
        explain_patterns = [
            r'why', r'explain', r'reason', r'how', 
            r'because', r'what happened'
        ]
        if any(re.search(p, q) for p in explain_patterns):
            return 'explain_decision'
        
        # Investigation patterns
        investigation_patterns = [
            r'investigate', r'look into', r'check', 
            r'depth', r'how deep', r'should i'
        ]
        if any(re.search(p, q) for p in investigation_patterns):
            return 'investigation_advice'
        
        # Similarity patterns
        similar_patterns = [
            r'similar', r'like this', r'compare', 
            r'others', r'history'
        ]
        if any(re.search(p, q) for p in similar_patterns):
            return 'similar_claims'
        
        # Help patterns
        help_patterns = [
            r'help', r'what can you do', r'capabilities',
            r'how do you work', r'who are you'
        ]
        if any(re.search(p, q) for p in help_patterns):
            return 'help'
        
        return 'general'
    
    def _handle_fraud_check(self, question: str, claim_data: Optional[Dict]) -> str:
        """Handle fraud check intent"""
        if not claim_data:
            return "I can check for fraud, but I need claim details. Please provide: claim amount, customer age, incident type, etc."
        
        # Run fraud check
        result = is_fraud(**claim_data)
        
        # Format response
        risk_emoji = "🔴" if result['risk_score'] > 70 else "🟡" if result['risk_score'] > 40 else "🟢"
        
        response = (
            f"{risk_emoji} **Fraud Assessment**\n\n"
            f"This claim has a **{result['fraud_probability']:.1%}** probability of fraud.\n"
            f"Risk Score: **{result['risk_score']}/100**\n"
            f"Recommended Action: **{result['decision'].upper()}**\n\n"
            f"{result['explanation']}"
        )
        
        if result['requires_human_review']:
            response += "\n\n⚠️ **Human review recommended**"
        
        return response
    
    def _handle_explain(self, question: str, claim_data: Optional[Dict]) -> str:
        """Handle explanation intent"""
        if claim_data:
            result = is_fraud(**claim_data)
            
            return (
                f"I made this decision based on:\n\n"
                f"1. **Fraud Probability**: {result['fraud_probability']:.1%}\n"
                f"2. **Risk Indicators**: {result.get('risk_factors', 'Multiple red flags')}\n"
                f"3. **Confidence**: {result['confidence']:.0%}\n\n"
                f"The model analyzed claim amount, customer history, incident details, "
                f"and {result.get('red_flag_count', 'several')} red flags to reach this conclusion."
            )
        
        return (
            "I analyze claims using a multi-layer approach:\n\n"
            "1. **Data Processing**: Clean and engineer features\n"
            "2. **Stochastic Models**: Markov chains, HMM, survival analysis\n"
            "3. **Agent Decision**: Optimize action based on cost-benefit\n\n"
            "Provide a claim ID or details for specific explanation."
        )
    
    def _handle_investigation(self, question: str, claim_data: Optional[Dict]) -> str:
        """Handle investigation advice"""
        if not claim_data:
            return "Provide claim details and I'll recommend investigation depth."
        
        result = is_fraud(**claim_data)
        
        depth_guide = {
            'approve': "No investigation needed. Fast approval.",
            'fast_track': "Light review: Verify basic documentation (1-2 days).",
            'standard': "Standard investigation: Check records, contact parties (3-5 days).",
            'deep': "Deep investigation: Forensic analysis, field verification (1-2 weeks).",
            'deny': "Immediate denial with fraud flag."
        }
        
        action = result['decision']
        advice = depth_guide.get(action, "Standard review recommended.")
        
        return (
            f"**Investigation Recommendation: {action.upper()}**\n\n"
            f"{advice}\n\n"
            f"Fraud Probability: {result['fraud_probability']:.1%}\n"
            f"Expected Investigation Cost: ${result.get('investigation_cost', 'N/A')}"
        )
    
    def _handle_similar(self, question: str, claim_data: Optional[Dict]) -> str:
        """Handle similar claims request"""
        return "Similar claims analysis would retrieve historical patterns. Feature coming soon."
    
    def _handle_help(self, question: str, claim_data: Optional[Dict]) -> str:
        """Handle help request"""
        return (
            "**StochClaim Agent - Capabilities**\n\n"
            "I can help you with:\n\n"
            "🕵️ **Fraud Detection**: Check if a claim is fraudulent\n"
            "📊 **Risk Scoring**: 0-100 risk assessment\n"
            "🎯 **Action Recommendations**: Approve, investigate, or deny\n"
            "📝 **Explanations**: Why I made a decision\n"
            "🔍 **Investigation Guidance**: How deep to investigate\n\n"
            "**Example questions:**\n"
            "- \"Is this claim fraudulent?\"\n"
            "- \"Why did you recommend deep investigation?\"\n"
            "- \"What should I check for this claim?\"\n\n"
            "Provide claim details for analysis."
        )
    
    def _handle_general(self, question: str, claim_data: Optional[Dict]) -> str:
        """Handle general questions with LLM"""
        # Use LLM for open-ended questions
        prompt = f"""
        You are StochClaim, an AI insurance fraud detection assistant.
        
        User question: {question}
        
        Claim data available: {'Yes' if claim_data else 'No'}
        
        Provide a helpful, concise response about fraud detection or ask for claim details if needed.
        """
        
        try:
            return self.llm._call_groq(prompt)
        except:
            return "I'm not sure I understand. Try asking about fraud detection, or type 'help' for options."
    
    def chat(self, message: str, claim_data: Optional[Dict] = None) -> str:
        """Simple chat interface"""
        result = self.ask(message, claim_data)
        return result['answer']


# Standalone function for quick use
def ask_agent(question: str, **claim_data) -> str:
    """Quick function to ask agent a question"""
    nl = NLAgentInterface()
    result = nl.ask(question, claim_data if claim_data else None)
    return result['answer']