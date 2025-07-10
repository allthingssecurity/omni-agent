"""
Generic Domain-Agnostic Core Types for Universal Agent System

This module provides completely generic types that can be used across any domain,
not just coding. The agent can handle business, research, creative, personal tasks, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import json

# -------------------------
# 1. Generic Enums
# -------------------------

class TaskDomain(Enum):
    """Different domains the agent can operate in"""
    CODING = "coding"
    BUSINESS = "business"
    RESEARCH = "research"
    CREATIVE = "creative"
    PERSONAL = "personal"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    LEGAL = "legal"
    GENERIC = "generic"

class IntentType(Enum):
    """Generic intent types applicable to any domain"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    ANALYZE = "analyze"
    SEARCH = "search"
    ORGANIZE = "organize"
    COMMUNICATE = "communicate"
    LEARN = "learn"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    OPTIMIZE = "optimize"

class ContextSignalType(Enum):
    """Generic context signal types"""
    USER_ACTION = "user_action"
    ENVIRONMENTAL = "environmental"
    TEMPORAL = "temporal"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    RESOURCE = "resource"
    RELATIONSHIP = "relationship"
    GOAL = "goal"

class UncertaintyType(Enum):
    """Types of uncertainty in decision making"""
    AMBIGUOUS_INTENT = "ambiguous_intent"
    INCOMPLETE_INFORMATION = "incomplete_information"
    CONFLICTING_GOALS = "conflicting_goals"
    UNKNOWN_PREFERENCES = "unknown_preferences"
    ENVIRONMENTAL_VOLATILITY = "environmental_volatility"
    RESOURCE_CONSTRAINTS = "resource_constraints"

# -------------------------
# 2. Generic Core Data Structures
# -------------------------

@dataclass
class GenericGoal:
    """Universal goal representation for any domain"""
    goal_id: str
    description: str
    domain: TaskDomain
    intent_type: IntentType
    confidence: float
    priority: int = 1
    reasoning: str = ""
    context_clues: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: float = 0.5
    estimated_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

@dataclass
class GenericAction:
    """Universal action representation for any domain"""
    action_id: str
    description: str
    domain: TaskDomain
    intent_type: IntentType
    parameters: Dict[str, Any]
    estimated_effort: float
    estimated_success_probability: float
    estimated_duration: Optional[float] = None
    prerequisites: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    reversibility: float = 0.5  # 0.0 = irreversible, 1.0 = fully reversible
    resources_required: List[str] = field(default_factory=list)
    skills_required: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenericContext:
    """Universal context representation for any domain"""
    domain: TaskDomain
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    # Core context information
    current_focus: Optional[Dict[str, Any]] = None
    recent_activities: List[Dict[str, Any]] = field(default_factory=list)
    available_resources: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # User information
    user_profile: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    user_capabilities: Dict[str, Any] = field(default_factory=dict)
    user_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Environmental information
    environment: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    social_context: Dict[str, Any] = field(default_factory=dict)
    
    # Domain-specific data
    domain_data: Dict[str, Any] = field(default_factory=dict)
    
    # Signals and feedback
    signals: List[Dict[str, Any]] = field(default_factory=list)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class GenericDecision:
    """Universal decision representation"""
    decision_id: str
    selected_action: GenericAction
    alternative_actions: List[GenericAction]
    reasoning: str
    confidence: float
    expected_utility: float
    expected_outcomes: List[str]
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    uncertainty_sources: List[UncertaintyType] = field(default_factory=list)
    fallback_plan: Optional[str] = None
    monitoring_plan: Optional[str] = None
    success_metrics: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# 3. Component Configuration
# -------------------------

@dataclass
class ComponentConfig:
    """Configuration for enabling/disabling agent components"""
    # Core capabilities
    enable_theory_of_mind: bool = True
    enable_decision_theory: bool = True
    enable_context_awareness: bool = True
    enable_learning: bool = True
    
    # Theory of Mind settings
    tom_inference_depth: int = 3
    tom_use_few_shot: bool = True
    tom_use_behavioral_signals: bool = True
    tom_confidence_threshold: float = 0.3
    
    # Decision Theory settings
    dt_use_expected_utility: bool = True
    dt_use_risk_assessment: bool = True
    dt_uncertainty_penalty: float = 0.2
    dt_risk_tolerance: float = 0.5
    
    # Context Awareness settings
    context_window_size: int = 10
    context_enhancement: bool = True
    context_quality_threshold: float = 0.5
    
    # Learning settings
    learning_from_feedback: bool = True
    learning_rate: float = 0.1
    memory_size: int = 1000
    
    # Fallback modes
    fallback_to_simple_mode: bool = True
    simple_mode_threshold: float = 0.1

@dataclass
class DomainConfig:
    """Domain-specific configuration"""
    domain: TaskDomain
    domain_vocabulary: Dict[str, List[str]] = field(default_factory=dict)
    domain_patterns: Dict[str, Any] = field(default_factory=dict)
    domain_constraints: List[str] = field(default_factory=list)
    domain_resources: List[str] = field(default_factory=list)
    domain_success_metrics: List[str] = field(default_factory=list)
    domain_specific_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentConfig:
    """Complete agent configuration"""
    agent_id: str
    name: str
    description: str
    primary_domain: TaskDomain
    supported_domains: List[TaskDomain] = field(default_factory=list)
    
    # Component configuration
    components: ComponentConfig = field(default_factory=ComponentConfig)
    
    # Domain configurations
    domain_configs: Dict[TaskDomain, DomainConfig] = field(default_factory=dict)
    
    # General settings
    max_goals: int = 5
    max_actions: int = 10
    max_execution_time: float = 60.0
    logging_level: str = "INFO"
    
    # Behavioral settings
    default_mode: str = "collaborative"  # autonomous, collaborative, conservative
    adaptability: float = 0.7
    proactiveness: float = 0.5
    
    def __post_init__(self):
        if self.primary_domain not in self.supported_domains:
            self.supported_domains.append(self.primary_domain)

# -------------------------
# 4. Generic Interfaces
# -------------------------

class GenericGoalInferenceEngine(ABC):
    """Generic interface for goal inference (Theory of Mind)"""
    
    @abstractmethod
    def infer_goals(self, instruction: str, context: GenericContext, config: ComponentConfig) -> List[GenericGoal]:
        """Infer goals from user instruction and context"""
        pass
    
    @abstractmethod
    def assess_ambiguity(self, instruction: str, context: GenericContext) -> float:
        """Assess ambiguity level of instruction"""
        pass

class GenericActionGenerator(ABC):
    """Generic interface for action generation"""
    
    @abstractmethod
    def generate_actions(self, goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> List[GenericAction]:
        """Generate candidate actions for goals"""
        pass

class GenericDecisionMaker(ABC):
    """Generic interface for decision making"""
    
    @abstractmethod
    def make_decision(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> GenericDecision:
        """Make decision among candidate actions"""
        pass

class GenericContextProcessor(ABC):
    """Generic interface for context processing"""
    
    @abstractmethod
    def process_context(self, raw_context: Dict[str, Any], domain: TaskDomain, config: ComponentConfig) -> GenericContext:
        """Process raw context into structured format"""
        pass

class GenericLearningSystem(ABC):
    """Generic interface for learning from feedback"""
    
    @abstractmethod
    def learn_from_feedback(self, decision: GenericDecision, feedback: Dict[str, Any], config: ComponentConfig) -> None:
        """Learn from user feedback"""
        pass

# -------------------------
# 5. Domain Adapters
# -------------------------

class DomainAdapter(ABC):
    """Abstract base class for domain-specific adapters"""
    
    def __init__(self, domain: TaskDomain):
        self.domain = domain
    
    @abstractmethod
    def adapt_instruction(self, instruction: str) -> str:
        """Adapt instruction to domain-specific format"""
        pass
    
    @abstractmethod
    def adapt_context(self, context: GenericContext) -> GenericContext:
        """Adapt context for domain-specific processing"""
        pass
    
    @abstractmethod
    def adapt_goals(self, goals: List[GenericGoal]) -> List[GenericGoal]:
        """Adapt goals for domain-specific processing"""
        pass
    
    @abstractmethod
    def adapt_actions(self, actions: List[GenericAction]) -> List[GenericAction]:
        """Adapt actions for domain-specific processing"""
        pass
    
    @abstractmethod
    def validate_decision(self, decision: GenericDecision) -> bool:
        """Validate decision for domain-specific constraints"""
        pass

# -------------------------
# 6. Utility Functions
# -------------------------

def create_generic_context(domain: TaskDomain, **kwargs) -> GenericContext:
    """Create a generic context for any domain"""
    defaults = {
        "domain": domain,
        "current_focus": {},
        "user_profile": {"experience_level": "intermediate"},
        "environment": {"type": "interactive"},
        "temporal_context": {"time_of_day": "work_hours"},
        "domain_data": {}
    }
    
    defaults.update(kwargs)
    return GenericContext(**defaults)

def get_domain_vocabulary(domain: TaskDomain) -> Dict[str, List[str]]:
    """Get vocabulary for a specific domain"""
    vocabularies = {
        TaskDomain.CODING: {
            "create": ["implement", "code", "write", "build", "develop"],
            "modify": ["fix", "update", "refactor", "optimize", "debug"],
            "analyze": ["review", "inspect", "profile", "test", "validate"]
        },
        TaskDomain.BUSINESS: {
            "create": ["propose", "design", "plan", "establish", "launch"],
            "modify": ["improve", "optimize", "restructure", "pivot", "scale"],
            "analyze": ["evaluate", "assess", "forecast", "audit", "benchmark"]
        },
        TaskDomain.RESEARCH: {
            "create": ["formulate", "hypothesize", "design", "construct", "synthesize"],
            "modify": ["refine", "adjust", "calibrate", "validate", "iterate"],
            "analyze": ["investigate", "examine", "correlate", "interpret", "conclude"]
        },
        TaskDomain.CREATIVE: {
            "create": ["design", "compose", "craft", "imagine", "innovate"],
            "modify": ["revise", "enhance", "stylize", "adapt", "remix"],
            "analyze": ["critique", "interpret", "deconstruct", "evaluate", "appreciate"]
        },
        TaskDomain.PERSONAL: {
            "create": ["plan", "organize", "establish", "build", "develop"],
            "modify": ["improve", "adjust", "change", "update", "optimize"],
            "analyze": ["reflect", "assess", "evaluate", "consider", "review"]
        }
    }
    
    return vocabularies.get(domain, {})

def get_domain_constraints(domain: TaskDomain) -> List[str]:
    """Get typical constraints for a domain"""
    constraints = {
        TaskDomain.CODING: ["syntax_rules", "performance_requirements", "security_standards"],
        TaskDomain.BUSINESS: ["budget_limits", "regulatory_compliance", "stakeholder_approval"],
        TaskDomain.RESEARCH: ["ethical_guidelines", "methodology_standards", "peer_review"],
        TaskDomain.CREATIVE: ["style_guidelines", "brand_consistency", "audience_appropriateness"],
        TaskDomain.PERSONAL: ["time_constraints", "privacy_concerns", "personal_values"]
    }
    
    return constraints.get(domain, [])

def infer_domain_from_context(context: GenericContext) -> TaskDomain:
    """Infer domain from context if not explicitly specified"""
    if context.domain != TaskDomain.GENERIC:
        return context.domain
    
    # Simple heuristics based on context
    domain_data = context.domain_data
    
    if "code" in str(domain_data).lower() or "programming" in str(domain_data).lower():
        return TaskDomain.CODING
    elif "business" in str(domain_data).lower() or "company" in str(domain_data).lower():
        return TaskDomain.BUSINESS
    elif "research" in str(domain_data).lower() or "study" in str(domain_data).lower():
        return TaskDomain.RESEARCH
    elif "creative" in str(domain_data).lower() or "design" in str(domain_data).lower():
        return TaskDomain.CREATIVE
    elif "personal" in str(domain_data).lower() or "life" in str(domain_data).lower():
        return TaskDomain.PERSONAL
    
    return TaskDomain.GENERIC

# -------------------------
# 7. Configuration Helpers
# -------------------------

def create_minimal_config() -> ComponentConfig:
    """Create minimal configuration with only basic features"""
    return ComponentConfig(
        enable_theory_of_mind=False,
        enable_decision_theory=False,
        enable_context_awareness=True,
        enable_learning=False,
        fallback_to_simple_mode=True
    )

def create_full_config() -> ComponentConfig:
    """Create full configuration with all features enabled"""
    return ComponentConfig(
        enable_theory_of_mind=True,
        enable_decision_theory=True,
        enable_context_awareness=True,
        enable_learning=True,
        tom_inference_depth=4,
        tom_use_few_shot=True,
        tom_use_behavioral_signals=True,
        dt_use_expected_utility=True,
        dt_use_risk_assessment=True,
        context_enhancement=True,
        learning_from_feedback=True
    )

def create_domain_config(domain: TaskDomain) -> DomainConfig:
    """Create domain-specific configuration"""
    return DomainConfig(
        domain=domain,
        domain_vocabulary=get_domain_vocabulary(domain),
        domain_constraints=get_domain_constraints(domain),
        domain_resources=[],
        domain_success_metrics=[]
    )

def create_agent_config(
    agent_id: str,
    name: str,
    primary_domain: TaskDomain,
    components: Optional[ComponentConfig] = None,
    **kwargs
) -> AgentConfig:
    """Create complete agent configuration"""
    if components is None:
        components = create_full_config()
    
    return AgentConfig(
        agent_id=agent_id,
        name=name,
        description=f"Generic agent for {primary_domain.value} domain",
        primary_domain=primary_domain,
        supported_domains=[primary_domain],
        components=components,
        domain_configs={primary_domain: create_domain_config(primary_domain)},
        **kwargs
    )