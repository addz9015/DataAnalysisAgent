# layer3/llm/explainer_llm.py
"""
LLM-powered explanation generation using Groq or Gemini
"""

import os
import json
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger("layer3.llm")

class LLMExplainer:
    """
    Generate natural language explanations using Groq or Gemini API
    """
    
    def __init__(self, 
                 provider: str = "groq",  # or "gemini"
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        
        # Default models
        self.model = model or self._default_model()
        self.client = None
        self.last_source = "template"
        self._init_client()
    
    def _get_api_key(self) -> str:
        """Get API key from environment"""
        if self.provider == "groq":
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("GROQ_API_KEY not found in environment")
            return key
        elif self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found")
            return key
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _default_model(self) -> str:
        """Get default model for provider"""
        if self.provider == "groq":
            return "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768"
        elif self.provider == "gemini":
            return "gemini-1.5-flash"  # or "gemini-1.5-pro"
    
    def _init_client(self):
        """Initialize API client"""
        if self.provider == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                logger.warning("groq package not installed; using template fallback explanations")
                self.client = None
            except Exception as e:
                logger.warning(f"Groq client init failed; using template fallback explanations: {e}")
                self.client = None

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                logger.warning("google-generativeai package not installed; using template fallback explanations")
                self.client = None
            except Exception as e:
                logger.warning(f"Gemini client init failed; using template fallback explanations: {e}")
                self.client = None
    
    def explain_decision(self, 
                        claim_id: str,
                        decision: str,
                        fraud_probability: float,
                        confidence: float,
                        key_evidence: Dict,
                        reasoning: str,
                        tone: str = "professional") -> str:

        """
        Generate natural language explanation using LLM
        """
        if self.client is None:
            self.last_source = "template"
            return self._fallback_explanation(decision, fraud_probability)

        prompt = self._build_prompt(
            claim_id=claim_id,
            decision=decision,
            fraud_probability=fraud_probability,
            confidence=confidence,
            key_evidence=key_evidence,
            reasoning=reasoning,
            tone=tone
        )

        try:
            if self.provider == "groq":
                self.last_source = "llm"
                return self._call_groq(prompt)
            elif self.provider == "gemini":
                self.last_source = "llm"
                return self._call_gemini(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            self.last_source = "template"
            return self._fallback_explanation(decision, fraud_probability)

        self.last_source = "template"
        return self._fallback_explanation(decision, fraud_probability)
    
    def _build_prompt(self, **kwargs) -> str:
        """Build prompt for LLM"""
        
        evidence_str = "\n".join([
            f"- {k.replace('_', ' ').title()}: {v}" 
            for k, v in kwargs['key_evidence'].items()
        ])
        
        return f"""You are an expert insurance fraud analyst explaining a claim decision to a customer service representative.

CLAIM DECISION SUMMARY:
- Claim ID: {kwargs['claim_id']}
- Decision: {kwargs['decision'].upper()}
- Fraud Probability: {kwargs['fraud_probability']:.1%}
- Confidence: {kwargs['confidence']:.0%}

KEY EVIDENCE:
{evidence_str}

TECHNICAL REASONING:
{kwargs['reasoning']}

TASK:
Write a clear, {kwargs['tone']} explanation (3-4 sentences) of why this decision was made. 
Focus on the most important factors. Do not use technical jargon.
Mention what the next steps are for this claim.

EXPLANATION:"""
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        time.sleep(0.5)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful insurance fraud analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 300
            }
        )
        return response.text.strip()
    
    def _fallback_explanation(self, decision: str, fraud_prob: float) -> str:
        """Fallback if LLM fails"""
        explanations = {
            'approve': f"This claim shows low fraud risk ({fraud_prob:.0%}) and has been approved for fast processing.",
            'fast_track': f"This claim has moderate risk ({fraud_prob:.0%}) and will undergo a quick review.",
            'standard': f"This claim requires standard investigation due to {fraud_prob:.0%} fraud probability.",
            'deep': f"This claim shows high fraud risk ({fraud_prob:.0%}) and needs thorough investigation.",
            'deny': f"This claim has been denied due to high fraud indicators ({fraud_prob:.0%})."
        }
        return explanations.get(decision, "Claim processed according to standard procedures.")
    
    def batch_explain(self, decisions: list) -> list:
        """Explain multiple decisions (use carefully due to API costs)"""
        return [
            self.explain_decision(**d)
            for d in decisions
        ]