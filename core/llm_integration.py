"""
LLM Integration Module - GPT-4o integration for the Universal Generic Agent

This module provides GPT-4o integration for actual AI-powered responses in the agent.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai>=1.0.0")

from core.generic_types import (
    GenericGoal, GenericAction, GenericContext, GenericDecision,
    TaskDomain, IntentType, ComponentConfig
)

# -------------------------
# 1. LLM Configuration
# -------------------------

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1500
    timeout: float = 30.0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI_GPT4O = "gpt-4o"
    OPENAI_GPT4_TURBO = "gpt-4-turbo"
    OPENAI_GPT35_TURBO = "gpt-3.5-turbo"

# -------------------------
# 2. Prompt Templates
# -------------------------

class PromptTemplates:
    """GPT-4o prompt templates for different agent components"""
    
    @staticmethod
    def theory_of_mind_prompt(instruction: str, context: GenericContext, config: ComponentConfig) -> str:
        """Prompt for Theory of Mind goal inference"""
        return f"""You are an expert at understanding human intentions and mental states. Analyze this instruction and context to infer the user's true goals.

INSTRUCTION: "{instruction}"

CONTEXT:
- Domain: {context.domain.value}
- Current Focus: {json.dumps(context.current_focus, indent=2) if context.current_focus else "None"}
- User Profile: {json.dumps(context.user_profile, indent=2) if context.user_profile else "None"}
- Recent Activities: {json.dumps(context.recent_activities[:3], indent=2) if context.recent_activities else "None"}
- Constraints: {json.dumps(context.constraints[:3], indent=2) if context.constraints else "None"}

ANALYSIS FRAMEWORK:
1. Mental State Assessment:
   - Detect urgency level (0.0-1.0)
   - Assess frustration indicators
   - Evaluate user confidence level
   - Estimate expertise in domain

2. Goal Inference Levels:
   - Direct: What they explicitly asked for
   - Implied: What they actually need
   - Meta: Higher-level objectives
   - Social: Contextual/collaborative goals

3. Ambiguity Assessment:
   - How clear/vague is the instruction?
   - What clarification might be needed?

Respond with a JSON object containing:
{{
    "mental_state": {{
        "urgency": 0.0-1.0,
        "frustration": 0.0-1.0,
        "confidence": 0.0-1.0,
        "expertise": 0.0-1.0
    }},
    "inferred_goals": [
        {{
            "goal_id": "descriptive_id",
            "description": "clear description",
            "intent_type": "create|modify|analyze|etc",
            "confidence": 0.0-1.0,
            "reasoning": "why you inferred this goal",
            "priority": 1-5
        }}
    ],
    "ambiguity_score": 0.0-1.0,
    "clarification_questions": ["question1", "question2"],
    "reasoning_chain": ["step1", "step2", "step3"]
}}"""

    @staticmethod
    def action_generation_prompt(goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> str:
        """Prompt for generating candidate actions"""
        goals_json = json.dumps([{
            "goal_id": g.goal_id,
            "description": g.description,
            "intent_type": g.intent_type.value,
            "confidence": g.confidence
        } for g in goals], indent=2)
        
        return f"""You are an expert at generating actionable solutions. Given these inferred goals and context, generate specific candidate actions.

INFERRED GOALS:
{goals_json}

CONTEXT:
- Domain: {context.domain.value}
- Available Resources: {json.dumps(context.available_resources[:5], indent=2) if context.available_resources else "None"}
- Constraints: {json.dumps(context.constraints[:5], indent=2) if context.constraints else "None"}
- User Capabilities: {json.dumps(context.user_capabilities, indent=2) if context.user_capabilities else "None"}

DOMAIN-SPECIFIC CONSIDERATIONS:
{PromptTemplates._get_domain_considerations(context.domain)}

For each goal, generate 1-3 candidate actions. Consider:
- Feasibility given available resources
- User capabilities and constraints
- Domain-specific best practices
- Risk vs. benefit trade-offs
- Effort required vs. impact

Respond with a JSON object:
{{
    "candidate_actions": [
        {{
            "action_id": "unique_id",
            "description": "specific actionable description",
            "intent_type": "create|modify|analyze|etc",
            "parameters": {{"key": "value"}},
            "estimated_effort": 0.0-1.0,
            "estimated_success_probability": 0.0-1.0,
            "estimated_duration_minutes": number,
            "prerequisites": ["prereq1", "prereq2"],
            "outcomes": ["expected_outcome1", "outcome2"],
            "side_effects": ["side_effect1", "effect2"],
            "resources_required": ["resource1", "resource2"],
            "risks": ["risk1", "risk2"],
            "reversibility": 0.0-1.0
        }}
    ],
    "reasoning": "explanation of action generation approach"
}}"""

    @staticmethod
    def decision_theory_prompt(actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> str:
        """Prompt for decision theory analysis"""
        actions_json = json.dumps([{
            "action_id": a.action_id,
            "description": a.description,
            "estimated_effort": a.estimated_effort,
            "estimated_success_probability": a.estimated_success_probability,
            "outcomes": a.outcomes,
            "risks": getattr(a, 'risks', []),
            "reversibility": a.reversibility
        } for a in actions], indent=2)
        
        goals_json = json.dumps([{
            "goal_id": g.goal_id,
            "description": g.description,
            "confidence": g.confidence,
            "priority": g.priority
        } for g in goals], indent=2)
        
        return f"""You are an expert in decision theory and utility optimization. Analyze these candidate actions using expected utility theory.

CANDIDATE ACTIONS:
{actions_json}

GOALS (with confidence as probability weights):
{goals_json}

CONTEXT:
- Domain: {context.domain.value}
- Risk Tolerance: {config.dt_risk_tolerance}
- User Profile: {json.dumps(context.user_profile, indent=2) if context.user_profile else "None"}

DECISION ANALYSIS FRAMEWORK:
1. For each action, calculate utility across multiple factors:
   - User satisfaction potential
   - Technical correctness likelihood
   - Efficiency (effort vs. impact)
   - Safety and risk mitigation
   - Long-term maintainability
   - Learning/growth value

2. Calculate expected utility using goal confidence as probability weights:
   E[U(action)] = Σ P(goal) × U(action|goal)

3. Assess uncertainty sources and risk factors

4. Consider domain-specific criteria:
{PromptTemplates._get_domain_decision_criteria(context.domain)}

Respond with a JSON object:
{{
    "decision_analysis": {{
        "selected_action_id": "best_action_id",
        "expected_utility": 0.0-1.0,
        "confidence": 0.0-1.0,
        "reasoning": "detailed decision reasoning",
        "utility_breakdown": {{
            "user_satisfaction": 0.0-1.0,
            "technical_correctness": 0.0-1.0,
            "efficiency": 0.0-1.0,
            "safety": 0.0-1.0,
            "maintainability": 0.0-1.0,
            "learning_value": 0.0-1.0
        }},
        "risk_assessment": {{
            "technical_risk": 0.0-1.0,
            "user_experience_risk": 0.0-1.0,
            "resource_risk": 0.0-1.0,
            "time_risk": 0.0-1.0
        }},
        "uncertainty_sources": ["source1", "source2"],
        "alternative_actions": ["action_id2", "action_id3"],
        "fallback_plan": "fallback description if needed"
    }}
}}"""

    @staticmethod
    def simple_prompt(instruction: str, context: GenericContext) -> str:
        """Simple prompt when ToM and DT are disabled"""
        return f"""You are a helpful AI assistant. Provide a direct, actionable response to this request.

INSTRUCTION: "{instruction}"

CONTEXT:
- Domain: {context.domain.value}
- Current Focus: {json.dumps(context.current_focus, indent=2) if context.current_focus else "None"}
- Constraints: {json.dumps(context.constraints[:3], indent=2) if context.constraints else "None"}

Provide a clear, specific action or response. Be concise but helpful.

Respond with a JSON object:
{{
    "action": {{
        "description": "specific action to take",
        "steps": ["step1", "step2", "step3"],
        "estimated_time": "time estimate",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
    }}
}}"""

    @staticmethod
    def _get_domain_considerations(domain: TaskDomain) -> str:
        """Get domain-specific considerations"""
        considerations = {
            TaskDomain.BUSINESS: """
- Consider stakeholder impact and ROI
- Include compliance and regulatory factors
- Account for budget constraints and timelines
- Think about scalability and business processes
""",
            TaskDomain.RESEARCH: """
- Ensure methodological rigor and reproducibility
- Consider ethical implications and IRB requirements
- Account for peer review and publication standards
- Include proper statistical analysis considerations
""",
            TaskDomain.CREATIVE: """
- Consider audience and brand consistency
- Account for aesthetic and creative principles
- Include technical feasibility for creative tools
- Think about style guidelines and creative brief requirements
""",
            TaskDomain.PERSONAL: """
- Prioritize privacy and personal values
- Consider work-life balance and sustainability
- Account for personal capabilities and time constraints
- Include habit formation and long-term lifestyle impact
""",
            TaskDomain.CODING: """
- Follow software engineering best practices
- Consider code quality, security, and maintainability
- Account for testing, documentation, and deployment
- Include performance and scalability considerations
"""
        }
        return considerations.get(domain, "Consider general best practices and user needs.")

    @staticmethod
    def _get_domain_decision_criteria(domain: TaskDomain) -> str:
        """Get domain-specific decision criteria"""
        criteria = {
            TaskDomain.BUSINESS: "Prioritize ROI, stakeholder satisfaction, compliance, and business impact.",
            TaskDomain.RESEARCH: "Prioritize methodological rigor, ethical compliance, reproducibility, and scientific validity.",
            TaskDomain.CREATIVE: "Prioritize creative impact, brand consistency, audience engagement, and aesthetic quality.",
            TaskDomain.PERSONAL: "Prioritize personal values, privacy, sustainability, and life balance.",
            TaskDomain.CODING: "Prioritize code quality, security, maintainability, and user experience."
        }
        return criteria.get(domain, "Prioritize user satisfaction, efficiency, and safety.")

# -------------------------
# 3. LLM Integration Class
# -------------------------

class GPT4oIntegration:
    """GPT-4o integration for the Universal Generic Agent"""
    
    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai>=1.0.0")
        
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        self.logger = logging.getLogger(__name__)
    
    def infer_goals_with_tom(self, instruction: str, context: GenericContext, component_config: ComponentConfig) -> Dict[str, Any]:
        """Use GPT-4o for Theory of Mind goal inference"""
        prompt = PromptTemplates.theory_of_mind_prompt(instruction, context, component_config)
        
        try:
            response = self._call_gpt4o(prompt, temperature=0.3)  # Lower temperature for analysis
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"ToM inference failed: {e}")
            return self._fallback_goal_inference(instruction, context)
    
    def generate_actions_with_llm(self, goals: List[GenericGoal], context: GenericContext, component_config: ComponentConfig) -> Dict[str, Any]:
        """Use GPT-4o for action generation"""
        prompt = PromptTemplates.action_generation_prompt(goals, context, component_config)
        
        try:
            response = self._call_gpt4o(prompt, temperature=0.7)  # Higher temperature for creativity
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Action generation failed: {e}")
            return self._fallback_action_generation(goals, context)
    
    def make_decision_with_dt(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, component_config: ComponentConfig) -> Dict[str, Any]:
        """Use GPT-4o for decision theory analysis"""
        prompt = PromptTemplates.decision_theory_prompt(actions, goals, context, component_config)
        
        try:
            response = self._call_gpt4o(prompt, temperature=0.2)  # Very low temperature for consistency
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Decision theory analysis failed: {e}")
            return self._fallback_decision_making(actions, goals, context)
    
    def simple_response(self, instruction: str, context: GenericContext) -> Dict[str, Any]:
        """Use GPT-4o for simple response when ToM and DT are disabled"""
        prompt = PromptTemplates.simple_prompt(instruction, context)
        
        try:
            response = self._call_gpt4o(prompt, temperature=0.7)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Simple response failed: {e}")
            return self._fallback_simple_response(instruction, context)
    
    async def async_infer_goals_with_tom(self, instruction: str, context: GenericContext, component_config: ComponentConfig) -> Dict[str, Any]:
        """Async version of ToM goal inference"""
        prompt = PromptTemplates.theory_of_mind_prompt(instruction, context, component_config)
        
        try:
            response = await self._call_gpt4o_async(prompt, temperature=0.3)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Async ToM inference failed: {e}")
            return self._fallback_goal_inference(instruction, context)
    
    def _call_gpt4o(self, prompt: str, temperature: float = None) -> str:
        """Call GPT-4o with the given prompt"""
        if temperature is None:
            temperature = self.config.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant with deep understanding of human psychology, decision theory, and domain expertise. Always respond with valid JSON as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT-4o API call failed: {e}")
            raise
    
    async def _call_gpt4o_async(self, prompt: str, temperature: float = None) -> str:
        """Async call to GPT-4o"""
        if temperature is None:
            temperature = self.config.temperature
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant with deep understanding of human psychology, decision theory, and domain expertise. Always respond with valid JSON as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Async GPT-4o API call failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from GPT-4o"""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from GPT-4o: {e}")
    
    def _fallback_goal_inference(self, instruction: str, context: GenericContext) -> Dict[str, Any]:
        """Fallback goal inference when GPT-4o fails"""
        return {
            "mental_state": {"urgency": 0.5, "frustration": 0.3, "confidence": 0.5, "expertise": 0.5},
            "inferred_goals": [{
                "goal_id": "fallback_goal",
                "description": f"Address the instruction: {instruction}",
                "intent_type": "execute",
                "confidence": 0.6,
                "reasoning": "Fallback goal due to LLM failure",
                "priority": 1
            }],
            "ambiguity_score": 0.5,
            "clarification_questions": [],
            "reasoning_chain": ["LLM call failed, using fallback goal inference"]
        }
    
    def _fallback_action_generation(self, goals: List[GenericGoal], context: GenericContext) -> Dict[str, Any]:
        """Fallback action generation when GPT-4o fails"""
        return {
            "candidate_actions": [{
                "action_id": "fallback_action",
                "description": f"Take action for goal: {goals[0].description if goals else 'unknown goal'}",
                "intent_type": "execute",
                "parameters": {},
                "estimated_effort": 0.5,
                "estimated_success_probability": 0.7,
                "estimated_duration_minutes": 30,
                "prerequisites": [],
                "outcomes": ["Complete the requested task"],
                "side_effects": [],
                "resources_required": [],
                "risks": [],
                "reversibility": 0.8
            }],
            "reasoning": "Fallback action due to LLM failure"
        }
    
    def _fallback_decision_making(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext) -> Dict[str, Any]:
        """Fallback decision making when GPT-4o fails"""
        best_action = max(actions, key=lambda a: a.estimated_success_probability) if actions else None
        
        return {
            "decision_analysis": {
                "selected_action_id": best_action.action_id if best_action else "no_action",
                "expected_utility": 0.6,
                "confidence": 0.5,
                "reasoning": "Fallback decision due to LLM failure - selected action with highest success probability",
                "utility_breakdown": {
                    "user_satisfaction": 0.6,
                    "technical_correctness": 0.7,
                    "efficiency": 0.5,
                    "safety": 0.8,
                    "maintainability": 0.6,
                    "learning_value": 0.4
                },
                "risk_assessment": {
                    "technical_risk": 0.3,
                    "user_experience_risk": 0.2,
                    "resource_risk": 0.2,
                    "time_risk": 0.3
                },
                "uncertainty_sources": ["llm_failure"],
                "alternative_actions": [a.action_id for a in actions[1:3] if len(actions) > 1],
                "fallback_plan": "Manual intervention may be required"
            }
        }
    
    def _fallback_simple_response(self, instruction: str, context: GenericContext) -> Dict[str, Any]:
        """Fallback simple response when GPT-4o fails"""
        return {
            "action": {
                "description": f"I'll help you with: {instruction}",
                "steps": ["Analyze the request", "Determine appropriate action", "Provide solution"],
                "estimated_time": "Variable depending on complexity",
                "confidence": 0.5,
                "reasoning": "Fallback response due to LLM failure"
            }
        }

# -------------------------
# 4. Configuration Helper
# -------------------------

def create_llm_config(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    timeout: float = 30.0
) -> LLMConfig:
    """Create LLM configuration with sensible defaults"""
    return LLMConfig(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

def get_available_models() -> List[str]:
    """Get list of available LLM models"""
    return [provider.value for provider in LLMProvider]