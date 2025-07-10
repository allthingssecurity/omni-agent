"""
Domain-Specific Adapters for Universal Agent

This module provides domain-specific adapters that customize the generic agent
for different domains like business, research, creative work, etc.
"""

from typing import Dict, List, Optional, Any, Set
import re
from dataclasses import dataclass
from enum import Enum

from core.generic_types import (
    GenericGoal, GenericAction, GenericContext, TaskDomain, IntentType,
    DomainAdapter, create_generic_context
)

# -------------------------
# 1. Business Domain Adapter
# -------------------------

class BusinessDomainAdapter(DomainAdapter):
    """Adapter for business and organizational tasks"""
    
    def __init__(self):
        super().__init__(TaskDomain.BUSINESS)
        self.business_vocabulary = {
            "analysis": ["analyze", "assess", "evaluate", "audit", "review", "benchmark"],
            "planning": ["plan", "strategy", "roadmap", "forecast", "budget", "schedule"],
            "communication": ["present", "report", "meeting", "email", "notification", "update"],
            "optimization": ["optimize", "streamline", "improve", "efficiency", "cost-reduction"],
            "decision": ["decide", "choose", "select", "approve", "prioritize", "allocate"],
            "execution": ["implement", "execute", "deploy", "launch", "rollout", "deliver"]
        }
        
        self.business_entities = [
            "client", "customer", "stakeholder", "team", "department", "vendor",
            "revenue", "profit", "cost", "budget", "ROI", "KPI", "metric",
            "project", "initiative", "campaign", "product", "service", "market"
        ]
    
    def adapt_instruction(self, instruction: str) -> str:
        """Adapt instruction for business context"""
        # Expand business abbreviations
        business_expansions = {
            "ROI": "return on investment",
            "KPI": "key performance indicator",
            "P&L": "profit and loss",
            "B2B": "business to business",
            "B2C": "business to consumer",
            "CRM": "customer relationship management",
            "ERP": "enterprise resource planning"
        }
        
        adapted = instruction
        for abbrev, expansion in business_expansions.items():
            adapted = re.sub(rf'\b{abbrev}\b', expansion, adapted, flags=re.IGNORECASE)
        
        return adapted
    
    def adapt_context(self, context: GenericContext) -> GenericContext:
        """Adapt context for business domain"""
        # Add business-specific context enhancement
        if not context.domain_data:
            context.domain_data = {}
        
        # Infer business context from available data
        context.domain_data["business_context"] = {
            "has_financial_data": self._has_financial_context(context),
            "has_team_context": self._has_team_context(context),
            "has_client_context": self._has_client_context(context),
            "urgency_level": self._assess_business_urgency(context),
            "stakeholder_impact": self._assess_stakeholder_impact(context)
        }
        
        return context
    
    def adapt_goals(self, goals: List[GenericGoal]) -> List[GenericGoal]:
        """Adapt goals for business domain"""
        adapted_goals = []
        
        for goal in goals:
            # Add business-specific success criteria
            if goal.intent_type == IntentType.ANALYZE:
                goal.success_criteria.extend([
                    "Provide actionable insights",
                    "Include quantitative metrics",
                    "Consider business impact"
                ])
            elif goal.intent_type == IntentType.PLAN:
                goal.success_criteria.extend([
                    "Define clear timeline",
                    "Identify resource requirements",
                    "Include risk assessment"
                ])
            elif goal.intent_type == IntentType.COMMUNICATE:
                goal.success_criteria.extend([
                    "Use business-appropriate language",
                    "Include executive summary",
                    "Provide clear action items"
                ])
            
            # Add business constraints
            goal.constraints.extend([
                "Consider budget implications",
                "Ensure stakeholder alignment",
                "Maintain compliance requirements"
            ])
            
            adapted_goals.append(goal)
        
        return adapted_goals
    
    def adapt_actions(self, actions: List[GenericAction]) -> List[GenericAction]:
        """Adapt actions for business domain"""
        adapted_actions = []
        
        for action in actions:
            # Add business-specific parameters
            if "analysis" in action.description.lower():
                action.parameters.update({
                    "include_financial_impact": True,
                    "stakeholder_considerations": True,
                    "competitive_analysis": True
                })
            
            # Add business resources
            action.resources_required.extend([
                "business_data_access",
                "stakeholder_time",
                "budget_approval"
            ])
            
            # Add business skills
            action.skills_required.extend([
                "business_analysis",
                "stakeholder_communication",
                "financial_literacy"
            ])
            
            adapted_actions.append(action)
        
        return adapted_actions
    
    def validate_decision(self, decision) -> bool:
        """Validate decision for business constraints"""
        # Check if decision considers business impact
        action = decision.selected_action
        
        # Business decisions should consider stakeholders
        if "stakeholder" not in action.description.lower() and action.intent_type in [IntentType.PLAN, IntentType.COMMUNICATE]:
            return False
        
        # Financial decisions should include impact assessment
        if any(keyword in action.description.lower() for keyword in ["budget", "cost", "revenue", "profit"]):
            if "impact" not in action.description.lower():
                return False
        
        return True
    
    def _has_financial_context(self, context: GenericContext) -> bool:
        """Check if context has financial information"""
        financial_keywords = ["budget", "cost", "revenue", "profit", "financial", "money"]
        context_str = str(context.current_focus) + str(context.user_profile)
        return any(keyword in context_str.lower() for keyword in financial_keywords)
    
    def _has_team_context(self, context: GenericContext) -> bool:
        """Check if context involves team/organizational aspects"""
        team_keywords = ["team", "department", "organization", "colleague", "manager", "employee"]
        context_str = str(context.social_context) + str(context.user_profile)
        return any(keyword in context_str.lower() for keyword in team_keywords)
    
    def _has_client_context(self, context: GenericContext) -> bool:
        """Check if context involves clients/customers"""
        client_keywords = ["client", "customer", "stakeholder", "vendor", "partner"]
        context_str = str(context.current_focus) + str(context.environment)
        return any(keyword in context_str.lower() for keyword in client_keywords)
    
    def _assess_business_urgency(self, context: GenericContext) -> str:
        """Assess urgency level for business context"""
        urgency_indicators = context.temporal_context.get("urgency_indicators", [])
        
        if any("deadline" in str(indicator).lower() for indicator in urgency_indicators):
            return "high"
        elif any("soon" in str(indicator).lower() for indicator in urgency_indicators):
            return "medium"
        else:
            return "low"
    
    def _assess_stakeholder_impact(self, context: GenericContext) -> str:
        """Assess stakeholder impact level"""
        if self._has_client_context(context):
            return "high"
        elif self._has_team_context(context):
            return "medium"
        else:
            return "low"

# -------------------------
# 2. Research Domain Adapter
# -------------------------

class ResearchDomainAdapter(DomainAdapter):
    """Adapter for research and academic tasks"""
    
    def __init__(self):
        super().__init__(TaskDomain.RESEARCH)
        self.research_vocabulary = {
            "methodology": ["study", "experiment", "survey", "interview", "observation", "case study"],
            "analysis": ["statistical", "qualitative", "quantitative", "correlation", "causation"],
            "validation": ["peer review", "replication", "validation", "verification", "hypothesis testing"],
            "documentation": ["paper", "report", "thesis", "publication", "citation", "reference"]
        }
    
    def adapt_instruction(self, instruction: str) -> str:
        """Adapt instruction for research context"""
        # Add research-specific context
        research_indicators = ["study", "research", "investigate", "analyze", "hypothesis"]
        
        if any(indicator in instruction.lower() for indicator in research_indicators):
            # Already research-oriented
            return instruction
        else:
            # Add research context
            return f"Research and {instruction}"
    
    def adapt_context(self, context: GenericContext) -> GenericContext:
        """Adapt context for research domain"""
        if not context.domain_data:
            context.domain_data = {}
        
        context.domain_data["research_context"] = {
            "has_hypothesis": self._has_hypothesis(context),
            "has_data": self._has_research_data(context),
            "methodology_specified": self._has_methodology(context),
            "ethical_considerations": self._needs_ethical_review(context),
            "publication_intent": self._has_publication_intent(context)
        }
        
        return context
    
    def adapt_goals(self, goals: List[GenericGoal]) -> List[GenericGoal]:
        """Adapt goals for research domain"""
        for goal in goals:
            # Add research-specific success criteria
            if goal.intent_type == IntentType.ANALYZE:
                goal.success_criteria.extend([
                    "Use appropriate statistical methods",
                    "Consider confounding variables",
                    "Ensure reproducibility",
                    "Document methodology clearly"
                ])
            elif goal.intent_type == IntentType.CREATE:
                goal.success_criteria.extend([
                    "Follow academic standards",
                    "Include proper citations",
                    "Present original contribution"
                ])
            
            # Add research constraints
            goal.constraints.extend([
                "Maintain ethical standards",
                "Ensure data privacy",
                "Follow institutional guidelines",
                "Consider peer review requirements"
            ])
        
        return goals
    
    def adapt_actions(self, actions: List[GenericAction]) -> List[GenericAction]:
        """Adapt actions for research domain"""
        for action in actions:
            # Add research-specific parameters
            action.parameters.update({
                "methodology": "scientific",
                "peer_review_ready": True,
                "ethical_compliance": True,
                "reproducible": True
            })
            
            # Add research resources
            action.resources_required.extend([
                "academic_databases",
                "statistical_software",
                "research_ethics_approval"
            ])
            
            # Add research skills
            action.skills_required.extend([
                "research_methodology",
                "statistical_analysis",
                "academic_writing",
                "critical_thinking"
            ])
        
        return actions
    
    def validate_decision(self, decision) -> bool:
        """Validate decision for research constraints"""
        action = decision.selected_action
        
        # Research actions should consider methodology
        if action.intent_type == IntentType.ANALYZE and "methodology" not in str(action.parameters):
            return False
        
        # Research should consider ethical implications
        if "human" in action.description.lower() or "data" in action.description.lower():
            if not action.parameters.get("ethical_compliance", False):
                return False
        
        return True
    
    def _has_hypothesis(self, context: GenericContext) -> bool:
        """Check if context includes research hypothesis"""
        hypothesis_keywords = ["hypothesis", "research question", "theory", "predict"]
        context_str = str(context.current_focus)
        return any(keyword in context_str.lower() for keyword in hypothesis_keywords)
    
    def _has_research_data(self, context: GenericContext) -> bool:
        """Check if context includes research data"""
        data_keywords = ["data", "dataset", "sample", "participants", "subjects"]
        return any(keyword in str(context.available_resources).lower() for keyword in data_keywords)
    
    def _has_methodology(self, context: GenericContext) -> bool:
        """Check if methodology is specified"""
        method_keywords = ["method", "approach", "design", "procedure", "protocol"]
        return any(keyword in str(context.current_focus).lower() for keyword in method_keywords)
    
    def _needs_ethical_review(self, context: GenericContext) -> bool:
        """Check if ethical review is needed"""
        ethical_keywords = ["human", "participant", "subject", "personal", "sensitive"]
        context_str = str(context.current_focus) + str(context.available_resources)
        return any(keyword in context_str.lower() for keyword in ethical_keywords)
    
    def _has_publication_intent(self, context: GenericContext) -> bool:
        """Check if there's intent to publish"""
        publication_keywords = ["publish", "journal", "conference", "paper", "article"]
        return any(keyword in str(context.user_preferences).lower() for keyword in publication_keywords)

# -------------------------
# 3. Creative Domain Adapter
# -------------------------

class CreativeDomainAdapter(DomainAdapter):
    """Adapter for creative and artistic tasks"""
    
    def __init__(self):
        super().__init__(TaskDomain.CREATIVE)
        self.creative_vocabulary = {
            "creation": ["design", "compose", "create", "craft", "imagine", "innovate"],
            "modification": ["revise", "enhance", "stylize", "adapt", "remix", "iterate"],
            "inspiration": ["brainstorm", "explore", "experiment", "play", "discover"],
            "evaluation": ["critique", "review", "assess", "refine", "polish"]
        }
    
    def adapt_instruction(self, instruction: str) -> str:
        """Adapt instruction for creative context"""
        # Encourage creative thinking
        creative_enhancers = {
            "make": "creatively design",
            "create": "innovatively create",
            "design": "imaginatively design",
            "write": "creatively compose"
        }
        
        adapted = instruction
        for original, enhanced in creative_enhancers.items():
            adapted = re.sub(rf'\b{original}\b', enhanced, adapted, flags=re.IGNORECASE)
        
        return adapted
    
    def adapt_context(self, context: GenericContext) -> GenericContext:
        """Adapt context for creative domain"""
        if not context.domain_data:
            context.domain_data = {}
        
        context.domain_data["creative_context"] = {
            "creative_medium": self._identify_creative_medium(context),
            "style_preferences": self._extract_style_preferences(context),
            "audience": self._identify_target_audience(context),
            "inspiration_sources": self._identify_inspiration_sources(context),
            "creative_constraints": self._identify_creative_constraints(context)
        }
        
        return context
    
    def adapt_goals(self, goals: List[GenericGoal]) -> List[GenericGoal]:
        """Adapt goals for creative domain"""
        for goal in goals:
            # Add creative-specific success criteria
            if goal.intent_type == IntentType.CREATE:
                goal.success_criteria.extend([
                    "Express original vision",
                    "Engage target audience",
                    "Demonstrate creative innovation",
                    "Maintain aesthetic coherence"
                ])
            elif goal.intent_type == IntentType.MODIFY:
                goal.success_criteria.extend([
                    "Preserve original intent",
                    "Enhance creative impact",
                    "Maintain style consistency"
                ])
            
            # Add creative constraints
            goal.constraints.extend([
                "Respect artistic integrity",
                "Consider audience appropriateness",
                "Maintain brand consistency",
                "Follow creative brief"
            ])
        
        return goals
    
    def adapt_actions(self, actions: List[GenericAction]) -> List[GenericAction]:
        """Adapt actions for creative domain"""
        for action in actions:
            # Add creative-specific parameters
            action.parameters.update({
                "creative_approach": "innovative",
                "style_guide": "follow_brand",
                "audience_appropriate": True,
                "original_content": True
            })
            
            # Add creative resources
            action.resources_required.extend([
                "creative_tools",
                "design_assets",
                "style_guidelines",
                "inspiration_materials"
            ])
            
            # Add creative skills
            action.skills_required.extend([
                "creative_thinking",
                "aesthetic_judgment",
                "technical_craft",
                "audience_awareness"
            ])
        
        return actions
    
    def validate_decision(self, decision) -> bool:
        """Validate decision for creative constraints"""
        action = decision.selected_action
        
        # Creative actions should consider audience
        if action.intent_type == IntentType.CREATE and "audience" not in str(action.parameters):
            return False
        
        # Creative work should maintain originality
        if not action.parameters.get("original_content", False):
            return False
        
        return True
    
    def _identify_creative_medium(self, context: GenericContext) -> str:
        """Identify the creative medium"""
        medium_keywords = {
            "visual": ["design", "image", "graphic", "visual", "art", "illustration"],
            "text": ["write", "text", "copy", "content", "article", "story"],
            "audio": ["music", "sound", "audio", "podcast", "voice"],
            "video": ["video", "film", "animation", "motion"],
            "interactive": ["app", "website", "interface", "experience"]
        }
        
        context_str = str(context.current_focus).lower()
        for medium, keywords in medium_keywords.items():
            if any(keyword in context_str for keyword in keywords):
                return medium
        
        return "general"
    
    def _extract_style_preferences(self, context: GenericContext) -> List[str]:
        """Extract style preferences from context"""
        style_keywords = ["modern", "classic", "minimalist", "bold", "elegant", "playful", "professional"]
        preferences = []
        
        context_str = str(context.user_preferences).lower()
        for style in style_keywords:
            if style in context_str:
                preferences.append(style)
        
        return preferences
    
    def _identify_target_audience(self, context: GenericContext) -> str:
        """Identify target audience"""
        audience_keywords = {
            "children": ["kids", "children", "young"],
            "teens": ["teen", "adolescent", "youth"],
            "adults": ["adult", "professional", "mature"],
            "seniors": ["senior", "elderly", "older"],
            "general": ["everyone", "all", "general"]
        }
        
        context_str = str(context.current_focus).lower()
        for audience, keywords in audience_keywords.items():
            if any(keyword in context_str for keyword in keywords):
                return audience
        
        return "general"
    
    def _identify_inspiration_sources(self, context: GenericContext) -> List[str]:
        """Identify inspiration sources from context"""
        sources = []
        
        if context.recent_activities:
            for activity in context.recent_activities:
                if activity.get("type") == "research" or "inspiration" in str(activity):
                    sources.append(activity.get("source", "unknown"))
        
        return sources
    
    def _identify_creative_constraints(self, context: GenericContext) -> List[str]:
        """Identify creative constraints"""
        constraints = []
        
        # Extract from context constraints
        for constraint in context.constraints:
            if constraint.get("type") in ["brand", "style", "content", "technical"]:
                constraints.append(constraint.get("description", str(constraint)))
        
        return constraints

# -------------------------
# 4. Personal Domain Adapter
# -------------------------

class PersonalDomainAdapter(DomainAdapter):
    """Adapter for personal life management tasks"""
    
    def __init__(self):
        super().__init__(TaskDomain.PERSONAL)
        self.personal_vocabulary = {
            "organization": ["organize", "plan", "schedule", "manage", "track"],
            "health": ["health", "fitness", "wellness", "medical", "exercise"],
            "finance": ["budget", "money", "expense", "saving", "investment"],
            "relationships": ["family", "friends", "social", "relationship", "communication"],
            "learning": ["learn", "study", "skill", "hobby", "development"],
            "productivity": ["productivity", "efficiency", "habit", "routine", "goal"]
        }
    
    def adapt_instruction(self, instruction: str) -> str:
        """Adapt instruction for personal context"""
        # Make instructions more personal
        if not any(pronoun in instruction.lower() for pronoun in ["i", "my", "me"]):
            if instruction.startswith(("create", "make", "plan", "organize")):
                instruction = f"Help me {instruction.lower()}"
        
        return instruction
    
    def adapt_context(self, context: GenericContext) -> GenericContext:
        """Adapt context for personal domain"""
        if not context.domain_data:
            context.domain_data = {}
        
        context.domain_data["personal_context"] = {
            "life_area": self._identify_life_area(context),
            "urgency": self._assess_personal_urgency(context),
            "privacy_level": self._assess_privacy_needs(context),
            "habit_building": self._detect_habit_building(context),
            "goal_setting": self._detect_goal_setting(context)
        }
        
        return context
    
    def adapt_goals(self, goals: List[GenericGoal]) -> List[GenericGoal]:
        """Adapt goals for personal domain"""
        for goal in goals:
            # Add personal-specific success criteria
            goal.success_criteria.extend([
                "Fits personal lifestyle",
                "Sustainable long-term",
                "Respects personal values",
                "Maintains work-life balance"
            ])
            
            # Add personal constraints
            goal.constraints.extend([
                "Respect privacy preferences",
                "Consider time availability",
                "Match personal capabilities",
                "Align with personal values"
            ])
        
        return goals
    
    def adapt_actions(self, actions: List[GenericAction]) -> List[GenericAction]:
        """Adapt actions for personal domain"""
        for action in actions:
            # Add personal-specific parameters
            action.parameters.update({
                "personalized": True,
                "privacy_conscious": True,
                "sustainable": True,
                "user_friendly": True
            })
            
            # Add personal resources
            action.resources_required.extend([
                "personal_time",
                "personal_space",
                "personal_motivation"
            ])
            
            # Add personal skills
            action.skills_required.extend([
                "self_management",
                "personal_reflection",
                "habit_formation"
            ])
        
        return actions
    
    def validate_decision(self, decision) -> bool:
        """Validate decision for personal constraints"""
        action = decision.selected_action
        
        # Personal actions should be sustainable
        if action.estimated_effort > 0.8:  # Very high effort
            return False
        
        # Should respect privacy
        if not action.parameters.get("privacy_conscious", False):
            return False
        
        return True
    
    def _identify_life_area(self, context: GenericContext) -> str:
        """Identify which area of personal life this relates to"""
        life_areas = {
            "health": ["health", "fitness", "wellness", "medical", "exercise", "diet"],
            "finance": ["money", "budget", "expense", "saving", "investment", "financial"],
            "career": ["work", "job", "career", "professional", "skill", "resume"],
            "relationships": ["family", "friend", "social", "relationship", "dating"],
            "home": ["home", "house", "cleaning", "organization", "maintenance"],
            "learning": ["learn", "study", "education", "course", "book", "skill"]
        }
        
        context_str = str(context.current_focus).lower()
        for area, keywords in life_areas.items():
            if any(keyword in context_str for keyword in keywords):
                return area
        
        return "general"
    
    def _assess_personal_urgency(self, context: GenericContext) -> str:
        """Assess urgency for personal tasks"""
        urgent_indicators = ["deadline", "urgent", "asap", "emergency", "critical"]
        context_str = str(context.temporal_context).lower()
        
        if any(indicator in context_str for indicator in urgent_indicators):
            return "high"
        elif "soon" in context_str:
            return "medium"
        else:
            return "low"
    
    def _assess_privacy_needs(self, context: GenericContext) -> str:
        """Assess privacy requirements"""
        private_indicators = ["private", "personal", "sensitive", "confidential"]
        context_str = str(context.user_preferences).lower()
        
        if any(indicator in context_str for indicator in private_indicators):
            return "high"
        else:
            return "medium"
    
    def _detect_habit_building(self, context: GenericContext) -> bool:
        """Detect if this is about building habits"""
        habit_keywords = ["habit", "routine", "daily", "regular", "consistent"]
        context_str = str(context.current_focus).lower()
        return any(keyword in context_str for keyword in habit_keywords)
    
    def _detect_goal_setting(self, context: GenericContext) -> bool:
        """Detect if this involves goal setting"""
        goal_keywords = ["goal", "objective", "target", "achievement", "accomplish"]
        context_str = str(context.current_focus).lower()
        return any(keyword in context_str for keyword in goal_keywords)

# -------------------------
# 5. Domain Adapter Factory
# -------------------------

class DomainAdapterFactory:
    """Factory for creating domain-specific adapters"""
    
    _adapters = {
        TaskDomain.BUSINESS: BusinessDomainAdapter,
        TaskDomain.RESEARCH: ResearchDomainAdapter,
        TaskDomain.CREATIVE: CreativeDomainAdapter,
        TaskDomain.PERSONAL: PersonalDomainAdapter,
    }
    
    @classmethod
    def create_adapter(cls, domain: TaskDomain) -> Optional[DomainAdapter]:
        """Create adapter for specified domain"""
        adapter_class = cls._adapters.get(domain)
        if adapter_class:
            return adapter_class()
        return None
    
    @classmethod
    def get_supported_domains(cls) -> List[TaskDomain]:
        """Get list of supported domains"""
        return list(cls._adapters.keys())
    
    @classmethod
    def register_adapter(cls, domain: TaskDomain, adapter_class: type):
        """Register a new domain adapter"""
        cls._adapters[domain] = adapter_class

# -------------------------
# 6. Utility Functions
# -------------------------

def adapt_for_domain(domain: TaskDomain, instruction: str, context: GenericContext, goals: List[GenericGoal], actions: List[GenericAction]) -> tuple:
    """Apply domain adaptation to all components"""
    adapter = DomainAdapterFactory.create_adapter(domain)
    
    if adapter:
        adapted_instruction = adapter.adapt_instruction(instruction)
        adapted_context = adapter.adapt_context(context)
        adapted_goals = adapter.adapt_goals(goals)
        adapted_actions = adapter.adapt_actions(actions)
        
        return adapted_instruction, adapted_context, adapted_goals, adapted_actions
    else:
        # No adapter available, return original
        return instruction, context, goals, actions

def validate_domain_decision(domain: TaskDomain, decision) -> bool:
    """Validate decision against domain constraints"""
    adapter = DomainAdapterFactory.create_adapter(domain)
    
    if adapter:
        return adapter.validate_decision(decision)
    else:
        return True  # No validation if no adapter

def get_domain_vocabulary(domain: TaskDomain) -> Dict[str, List[str]]:
    """Get vocabulary for a domain"""
    adapter = DomainAdapterFactory.create_adapter(domain)
    
    if adapter and hasattr(adapter, 'business_vocabulary'):
        return adapter.business_vocabulary
    elif adapter and hasattr(adapter, 'research_vocabulary'):
        return adapter.research_vocabulary
    elif adapter and hasattr(adapter, 'creative_vocabulary'):
        return adapter.creative_vocabulary
    elif adapter and hasattr(adapter, 'personal_vocabulary'):
        return adapter.personal_vocabulary
    else:
        return {}

def infer_domain_from_instruction(instruction: str) -> TaskDomain:
    """Infer domain from instruction content"""
    domain_indicators = {
        TaskDomain.BUSINESS: ["business", "company", "revenue", "profit", "client", "stakeholder", "ROI", "KPI"],
        TaskDomain.RESEARCH: ["research", "study", "hypothesis", "data", "analysis", "experiment", "paper"],
        TaskDomain.CREATIVE: ["design", "create", "art", "creative", "compose", "style", "aesthetic"],
        TaskDomain.PERSONAL: ["personal", "my", "i need", "help me", "organize my", "plan my"],
        TaskDomain.CODING: ["code", "program", "debug", "function", "api", "software", "development"]
    }
    
    instruction_lower = instruction.lower()
    domain_scores = {}
    
    for domain, indicators in domain_indicators.items():
        score = sum(1 for indicator in indicators if indicator in instruction_lower)
        if score > 0:
            domain_scores[domain] = score
    
    if domain_scores:
        return max(domain_scores.keys(), key=lambda d: domain_scores[d])
    else:
        return TaskDomain.GENERIC