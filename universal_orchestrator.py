"""
Universal Orchestrator - Main entry point for the Universal Generic Agent

This module provides a unified interface for creating and managing universal agents
across any domain with configurable Theory of Mind and Decision Theory components.
"""

from typing import Dict, List, Optional, Any, Union
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from core.generic_types import (
    TaskDomain, ComponentConfig, DomainConfig, AgentConfig,
    GenericGoal, GenericAction, GenericContext, GenericDecision,
    IntentType, UncertaintyType,
    create_agent_config, create_minimal_config, create_full_config,
    create_generic_context, infer_domain_from_context
)
from core.generic_agent import UniversalGenericAgent
from domains.domain_adapters import (
    DomainAdapterFactory, adapt_for_domain, validate_domain_decision,
    infer_domain_from_instruction
)

# -------------------------
# 1. Agent Factory for Different Use Cases
# -------------------------

class UniversalAgentFactory:
    """Factory for creating specialized universal agents"""
    
    @staticmethod
    def create_minimal_agent(agent_id: str = "minimal_agent", name: str = "Minimal Agent") -> UniversalGenericAgent:
        """Create minimal agent with basic functionality only"""
        config = create_agent_config(
            agent_id=agent_id,
            name=name,
            primary_domain=TaskDomain.GENERIC,
            components=create_minimal_config()
        )
        return UniversalGenericAgent(config)
    
    @staticmethod
    def create_full_agent(agent_id: str = "full_agent", name: str = "Full Agent") -> UniversalGenericAgent:
        """Create agent with all capabilities enabled"""
        config = create_agent_config(
            agent_id=agent_id,
            name=name,
            primary_domain=TaskDomain.GENERIC,
            components=create_full_config()
        )
        return UniversalGenericAgent(config)
    
    @staticmethod
    def create_domain_agent(domain: TaskDomain, agent_id: Optional[str] = None, name: Optional[str] = None) -> UniversalGenericAgent:
        """Create agent specialized for a specific domain"""
        if agent_id is None:
            agent_id = f"{domain.value}_agent"
        if name is None:
            name = f"{domain.value.title()} Agent"
        
        config = create_agent_config(
            agent_id=agent_id,
            name=name,
            primary_domain=domain,
            components=create_full_config()
        )
        return UniversalGenericAgent(config)
    
    @staticmethod
    def create_theory_of_mind_agent(agent_id: str = "tom_agent", name: str = "Theory of Mind Agent") -> UniversalGenericAgent:
        """Create agent with only Theory of Mind enabled"""
        config = ComponentConfig(
            enable_theory_of_mind=True,
            enable_decision_theory=False,
            enable_context_awareness=True,
            enable_learning=False,
            tom_inference_depth=4,
            tom_use_few_shot=True,
            tom_use_behavioral_signals=True
        )
        
        agent_config = create_agent_config(
            agent_id=agent_id,
            name=name,
            primary_domain=TaskDomain.GENERIC,
            components=config
        )
        return UniversalGenericAgent(agent_config)
    
    @staticmethod
    def create_decision_theory_agent(agent_id: str = "dt_agent", name: str = "Decision Theory Agent") -> UniversalGenericAgent:
        """Create agent with only Decision Theory enabled"""
        config = ComponentConfig(
            enable_theory_of_mind=False,
            enable_decision_theory=True,
            enable_context_awareness=True,
            enable_learning=False,
            dt_use_expected_utility=True,
            dt_use_risk_assessment=True,
            dt_uncertainty_penalty=0.2
        )
        
        agent_config = create_agent_config(
            agent_id=agent_id,
            name=name,
            primary_domain=TaskDomain.GENERIC,
            components=config
        )
        return UniversalGenericAgent(agent_config)
    
    @staticmethod
    def create_custom_agent(
        agent_id: str,
        name: str,
        primary_domain: TaskDomain = TaskDomain.GENERIC,
        enable_theory_of_mind: bool = True,
        enable_decision_theory: bool = True,
        enable_context_awareness: bool = True,
        enable_learning: bool = False,
        risk_tolerance: float = 0.5,
        **kwargs
    ) -> UniversalGenericAgent:
        """Create custom agent with specific configuration"""
        
        config = ComponentConfig(
            enable_theory_of_mind=enable_theory_of_mind,
            enable_decision_theory=enable_decision_theory,
            enable_context_awareness=enable_context_awareness,
            enable_learning=enable_learning,
            dt_risk_tolerance=risk_tolerance,
            **kwargs
        )
        
        agent_config = create_agent_config(
            agent_id=agent_id,
            name=name,
            primary_domain=primary_domain,
            components=config
        )
        return UniversalGenericAgent(agent_config)

# -------------------------
# 2. Universal Agent Manager
# -------------------------

class UniversalAgentManager:
    """Manager for multiple universal agents"""
    
    def __init__(self):
        self.agents: Dict[str, UniversalGenericAgent] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent: UniversalGenericAgent):
        """Register an agent with the manager"""
        agent_id = agent.config.agent_id
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[UniversalGenericAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [agent.get_capabilities() for agent in self.agents.values()]
    
    def process_with_best_agent(self, instruction: str, context: Dict[str, Any]) -> GenericDecision:
        """Process instruction with the most suitable agent"""
        # Infer domain from instruction and context
        domain = self._infer_best_domain(instruction, context)
        
        # Find best agent for this domain
        best_agent = self._find_best_agent_for_domain(domain)
        
        if best_agent:
            return best_agent.process_instruction(instruction, context)
        else:
            # Create a temporary agent for this domain
            temp_agent = UniversalAgentFactory.create_domain_agent(domain)
            return temp_agent.process_instruction(instruction, context)
    
    def process_with_multiple_agents(self, instruction: str, context: Dict[str, Any]) -> Dict[str, GenericDecision]:
        """Process instruction with multiple agents and compare results"""
        results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                decision = agent.process_instruction(instruction, context)
                results[agent_id] = decision
            except Exception as e:
                self.logger.error(f"Agent {agent_id} failed: {e}")
        
        return results
    
    def _infer_best_domain(self, instruction: str, context: Dict[str, Any]) -> TaskDomain:
        """Infer the best domain for processing this instruction"""
        # First check if domain is specified in context
        if "domain" in context:
            domain = context["domain"]
            if isinstance(domain, str):
                return TaskDomain(domain)
            return domain
        
        # Try to infer from instruction
        inferred_domain = infer_domain_from_instruction(instruction)
        if inferred_domain != TaskDomain.GENERIC:
            return inferred_domain
        
        # Default to generic
        return TaskDomain.GENERIC
    
    def _find_best_agent_for_domain(self, domain: TaskDomain) -> Optional[UniversalGenericAgent]:
        """Find the best agent for a specific domain"""
        # Look for agents that specialize in this domain
        domain_agents = [
            agent for agent in self.agents.values()
            if agent.config.primary_domain == domain
        ]
        
        if domain_agents:
            # Return agent with most capabilities
            return max(domain_agents, key=lambda a: (
                a.config.components.enable_theory_of_mind +
                a.config.components.enable_decision_theory +
                a.config.components.enable_context_awareness +
                a.config.components.enable_learning
            ))
        
        # Look for generic agents
        generic_agents = [
            agent for agent in self.agents.values()
            if domain in agent.config.supported_domains or agent.config.primary_domain == TaskDomain.GENERIC
        ]
        
        if generic_agents:
            return generic_agents[0]
        
        return None

# -------------------------
# 3. Quick Start Interface
# -------------------------

class QuickStart:
    """Quick start interface for common use cases"""
    
    @staticmethod
    def simple_assistant() -> UniversalGenericAgent:
        """Create simple assistant without advanced features"""
        return UniversalAgentFactory.create_minimal_agent()
    
    @staticmethod
    def smart_assistant() -> UniversalGenericAgent:
        """Create smart assistant with all features"""
        return UniversalAgentFactory.create_full_agent()
    
    @staticmethod
    def business_assistant() -> UniversalGenericAgent:
        """Create business-focused assistant"""
        return UniversalAgentFactory.create_domain_agent(TaskDomain.BUSINESS)
    
    @staticmethod
    def research_assistant() -> UniversalGenericAgent:
        """Create research-focused assistant"""
        return UniversalAgentFactory.create_domain_agent(TaskDomain.RESEARCH)
    
    @staticmethod
    def creative_assistant() -> UniversalGenericAgent:
        """Create creative-focused assistant"""
        return UniversalAgentFactory.create_domain_agent(TaskDomain.CREATIVE)
    
    @staticmethod
    def personal_assistant() -> UniversalGenericAgent:
        """Create personal life assistant"""
        return UniversalAgentFactory.create_domain_agent(TaskDomain.PERSONAL)
    
    @staticmethod
    def custom_assistant(
        theory_of_mind: bool = True,
        decision_theory: bool = True,
        domain: str = "generic"
    ) -> UniversalGenericAgent:
        """Create custom assistant with specific features"""
        domain_enum = TaskDomain(domain) if domain != "generic" else TaskDomain.GENERIC
        
        return UniversalAgentFactory.create_custom_agent(
            agent_id=f"custom_{domain}_agent",
            name=f"Custom {domain.title()} Agent",
            primary_domain=domain_enum,
            enable_theory_of_mind=theory_of_mind,
            enable_decision_theory=decision_theory
        )

# -------------------------
# 4. Convenience Functions
# -------------------------

def quick_process(instruction: str, domain: str = "generic", **context_kwargs) -> GenericDecision:
    """Quick processing without creating persistent agent"""
    domain_enum = TaskDomain(domain) if domain != "generic" else TaskDomain.GENERIC
    agent = UniversalAgentFactory.create_domain_agent(domain_enum)
    
    context = {
        "domain": domain_enum,
        **context_kwargs
    }
    
    return agent.process_instruction(instruction, context)

def compare_approaches(instruction: str, context: Dict[str, Any]) -> Dict[str, GenericDecision]:
    """Compare different agent approaches for the same instruction"""
    approaches = {
        "minimal": UniversalAgentFactory.create_minimal_agent(),
        "theory_of_mind_only": UniversalAgentFactory.create_theory_of_mind_agent(),
        "decision_theory_only": UniversalAgentFactory.create_decision_theory_agent(),
        "full_capabilities": UniversalAgentFactory.create_full_agent()
    }
    
    results = {}
    for approach_name, agent in approaches.items():
        try:
            decision = agent.process_instruction(instruction, context)
            results[approach_name] = decision
        except Exception as e:
            logging.error(f"Approach {approach_name} failed: {e}")
    
    return results

def create_multi_domain_agent() -> UniversalGenericAgent:
    """Create agent that supports multiple domains"""
    config = create_agent_config(
        agent_id="multi_domain_agent",
        name="Multi-Domain Agent",
        primary_domain=TaskDomain.GENERIC,
        supported_domains=[
            TaskDomain.BUSINESS,
            TaskDomain.RESEARCH,
            TaskDomain.CREATIVE,
            TaskDomain.PERSONAL,
            TaskDomain.CODING
        ],
        components=create_full_config()
    )
    return UniversalGenericAgent(config)

# -------------------------
# 5. Configuration Helpers
# -------------------------

def get_recommended_config(use_case: str) -> ComponentConfig:
    """Get recommended configuration for specific use cases"""
    configs = {
        "production": ComponentConfig(
            enable_theory_of_mind=True,
            enable_decision_theory=True,
            enable_context_awareness=True,
            enable_learning=False,  # Disabled for consistency
            dt_risk_tolerance=0.3,  # Conservative
            tom_confidence_threshold=0.4,
            dt_uncertainty_penalty=0.3
        ),
        "research": ComponentConfig(
            enable_theory_of_mind=True,
            enable_decision_theory=True,
            enable_context_awareness=True,
            enable_learning=True,
            tom_inference_depth=4,  # Deep analysis
            dt_use_expected_utility=True,
            dt_use_risk_assessment=True
        ),
        "rapid_prototyping": ComponentConfig(
            enable_theory_of_mind=False,
            enable_decision_theory=False,
            enable_context_awareness=True,
            enable_learning=False,
            fallback_to_simple_mode=True
        ),
        "educational": ComponentConfig(
            enable_theory_of_mind=True,
            enable_decision_theory=True,
            enable_context_awareness=True,
            enable_learning=True,
            tom_inference_depth=2,  # Moderate depth
            dt_uncertainty_penalty=0.1  # More exploratory
        )
    }
    
    return configs.get(use_case, create_full_config())

# -------------------------
# 6. Demo and Testing Functions
# -------------------------

def demo_universal_agent():
    """Demonstration of universal agent capabilities"""
    print("=== Universal Agent Demo ===\n")
    
    # Create different types of agents
    agents = {
        "Minimal": UniversalAgentFactory.create_minimal_agent(),
        "Full": UniversalAgentFactory.create_full_agent(),
        "Business": UniversalAgentFactory.create_domain_agent(TaskDomain.BUSINESS),
        "Theory of Mind Only": UniversalAgentFactory.create_theory_of_mind_agent(),
        "Decision Theory Only": UniversalAgentFactory.create_decision_theory_agent()
    }
    
    # Test instruction
    instruction = "Help me solve this complex problem efficiently"
    context = {
        "domain": TaskDomain.GENERIC,
        "current_focus": {"problem": "resource allocation with multiple constraints"},
        "user_profile": {"experience_level": "intermediate"},
        "constraints": [{"type": "time", "description": "urgent deadline"}]
    }
    
    print(f"Instruction: '{instruction}'\n")
    
    for agent_name, agent in agents.items():
        print(f"{agent_name} Agent:")
        
        # Get capabilities
        caps = agent.get_capabilities()
        print(f"  ToM: {caps['theory_of_mind']}, DT: {caps['decision_theory']}")
        
        # Process instruction
        decision = agent.process_instruction(instruction, context)
        print(f"  Decision: {decision.selected_action.description}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning[:80]}...")
        print()
    
    print("="*50)

def test_component_toggling():
    """Test runtime component toggling"""
    print("=== Component Toggling Test ===\n")
    
    agent = UniversalAgentFactory.create_full_agent()
    instruction = "Analyze this situation and recommend an action"
    context = {"domain": TaskDomain.GENERIC, "current_focus": {"situation": "ambiguous requirements"}}
    
    configs = [
        ("Full Capabilities", ComponentConfig(True, True, True, False)),
        ("ToM Only", ComponentConfig(True, False, True, False)),
        ("DT Only", ComponentConfig(False, True, True, False)),
        ("Minimal", ComponentConfig(False, False, True, False))
    ]
    
    for config_name, config in configs:
        agent.update_config(config)
        decision = agent.process_instruction(instruction, context)
        
        print(f"{config_name}:")
        print(f"  Decision: {decision.selected_action.description}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print()
    
    print("="*50)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demos
    demo_universal_agent()
    test_component_toggling()
    
    # Quick start examples
    print("\n=== Quick Start Examples ===")
    
    # Simple usage
    simple_agent = QuickStart.simple_assistant()
    result = simple_agent.process_instruction(
        "Help me organize my day",
        {"domain": TaskDomain.PERSONAL, "current_focus": {"tasks": ["work", "exercise", "study"]}}
    )
    print(f"Simple Agent: {result.selected_action.description}")
    
    # Smart usage
    smart_agent = QuickStart.smart_assistant()
    result = smart_agent.process_instruction(
        "Analyze market trends and suggest strategy",
        {"domain": TaskDomain.BUSINESS, "current_focus": {"market": "tech_stocks", "timeframe": "Q4"}}
    )
    print(f"Smart Agent: {result.selected_action.description}")
    
    # Compare approaches
    print(f"\n=== Approach Comparison ===")
    comparison = compare_approaches(
        "Design a solution for this problem",
        {"domain": TaskDomain.GENERIC, "current_focus": {"problem": "unclear requirements"}}
    )
    
    for approach, decision in comparison.items():
        print(f"{approach}: {decision.selected_action.description} (confidence: {decision.confidence:.2f})")
    
    print("\nUniversal Agent demonstration completed!")