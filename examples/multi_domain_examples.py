"""
Multi-Domain Examples for Universal Generic Agent

This module demonstrates the agent working across different domains
with configurable Theory of Mind and Decision Theory components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generic_types import (
    TaskDomain, ComponentConfig, create_agent_config, create_minimal_config, create_full_config
)
from core.generic_agent import UniversalGenericAgent
from domains.domain_adapters import DomainAdapterFactory

# -------------------------
# 1. Configuration Examples
# -------------------------

def demonstrate_configurable_components():
    """Demonstrate configurable Theory of Mind and Decision Theory"""
    print("=== Configurable Components Demonstration ===\n")
    
    # Test instruction
    instruction = "Help me analyze this situation and decide what to do"
    context = {
        "domain": TaskDomain.BUSINESS,
        "current_focus": {"situation": "declining sales in Q3"},
        "user_profile": {"role": "sales_manager", "experience_level": "intermediate"},
        "constraints": [{"type": "budget", "description": "limited marketing budget"}]
    }
    
    # Configuration 1: Full capabilities
    full_config = create_full_config()
    agent_config = create_agent_config(
        agent_id="full_agent",
        name="Full Capability Agent",
        primary_domain=TaskDomain.BUSINESS,
        components=full_config
    )
    full_agent = UniversalGenericAgent(agent_config)
    
    print("1. FULL CAPABILITIES (Theory of Mind + Decision Theory)")
    print(f"   ToM: {full_config.enable_theory_of_mind}")
    print(f"   DT:  {full_config.enable_decision_theory}")
    decision1 = full_agent.process_instruction(instruction, context)
    print(f"   Decision: {decision1.selected_action.description}")
    print(f"   Confidence: {decision1.confidence:.2f}")
    print(f"   Reasoning: {decision1.reasoning[:100]}...")
    print()
    
    # Configuration 2: Only Theory of Mind
    tom_only_config = ComponentConfig(
        enable_theory_of_mind=True,
        enable_decision_theory=False,
        enable_context_awareness=True,
        enable_learning=False
    )
    tom_agent_config = create_agent_config(
        agent_id="tom_agent",
        name="Theory of Mind Only Agent",
        primary_domain=TaskDomain.BUSINESS,
        components=tom_only_config
    )
    tom_agent = UniversalGenericAgent(tom_agent_config)
    
    print("2. THEORY OF MIND ONLY")
    print(f"   ToM: {tom_only_config.enable_theory_of_mind}")
    print(f"   DT:  {tom_only_config.enable_decision_theory}")
    decision2 = tom_agent.process_instruction(instruction, context)
    print(f"   Decision: {decision2.selected_action.description}")
    print(f"   Confidence: {decision2.confidence:.2f}")
    print(f"   Reasoning: {decision2.reasoning[:100]}...")
    print()
    
    # Configuration 3: Only Decision Theory
    dt_only_config = ComponentConfig(
        enable_theory_of_mind=False,
        enable_decision_theory=True,
        enable_context_awareness=True,
        enable_learning=False
    )
    dt_agent_config = create_agent_config(
        agent_id="dt_agent",
        name="Decision Theory Only Agent",
        primary_domain=TaskDomain.BUSINESS,
        components=dt_only_config
    )
    dt_agent = UniversalGenericAgent(dt_agent_config)
    
    print("3. DECISION THEORY ONLY")
    print(f"   ToM: {dt_only_config.enable_theory_of_mind}")
    print(f"   DT:  {dt_only_config.enable_decision_theory}")
    decision3 = dt_agent.process_instruction(instruction, context)
    print(f"   Decision: {decision3.selected_action.description}")
    print(f"   Confidence: {decision3.confidence:.2f}")
    print(f"   Reasoning: {decision3.reasoning[:100]}...")
    print()
    
    # Configuration 4: Minimal (simple mode)
    minimal_config = create_minimal_config()
    minimal_agent_config = create_agent_config(
        agent_id="minimal_agent",
        name="Minimal Agent",
        primary_domain=TaskDomain.BUSINESS,
        components=minimal_config
    )
    minimal_agent = UniversalGenericAgent(minimal_agent_config)
    
    print("4. MINIMAL MODE (No ToM, No DT)")
    print(f"   ToM: {minimal_config.enable_theory_of_mind}")
    print(f"   DT:  {minimal_config.enable_decision_theory}")
    decision4 = minimal_agent.process_instruction(instruction, context)
    print(f"   Decision: {decision4.selected_action.description}")
    print(f"   Confidence: {decision4.confidence:.2f}")
    print(f"   Reasoning: {decision4.reasoning[:100]}...")
    print()
    
    print("="*50 + "\n")

# -------------------------
# 2. Business Domain Examples
# -------------------------

def business_domain_examples():
    """Examples in business domain"""
    print("=== Business Domain Examples ===\n")
    
    # Create business agent
    config = create_agent_config(
        agent_id="business_agent",
        name="Business Assistant",
        primary_domain=TaskDomain.BUSINESS
    )
    agent = UniversalGenericAgent(config)
    
    examples = [
        {
            "instruction": "Analyze our Q3 sales performance and recommend improvements",
            "context": {
                "domain": TaskDomain.BUSINESS,
                "current_focus": {
                    "report": "Q3_sales_report.xlsx",
                    "metrics": ["revenue", "conversion_rate", "customer_acquisition"]
                },
                "user_profile": {"role": "sales_director", "experience_level": "advanced"},
                "available_resources": [
                    {"type": "sales_data", "timeframe": "Q1-Q3"},
                    {"type": "budget", "amount": "$50K"},
                    {"type": "team", "size": 8}
                ],
                "constraints": [
                    {"type": "timeline", "description": "Results needed for board meeting next week"},
                    {"type": "budget", "description": "Limited to current allocated funds"}
                ]
            }
        },
        {
            "instruction": "Create a presentation for stakeholders about our new product launch",
            "context": {
                "domain": TaskDomain.BUSINESS,
                "current_focus": {
                    "product": "AI-powered analytics tool",
                    "audience": "C-level executives",
                    "duration": "30 minutes"
                },
                "user_profile": {"role": "product_manager", "experience_level": "intermediate"},
                "available_resources": [
                    {"type": "product_specs", "status": "complete"},
                    {"type": "market_research", "status": "complete"},
                    {"type": "financial_projections", "status": "draft"}
                ],
                "constraints": [
                    {"type": "confidentiality", "description": "Competitive sensitive information"},
                    {"type": "compliance", "description": "Must follow corporate presentation guidelines"}
                ]
            }
        },
        {
            "instruction": "Help me prioritize our feature backlog for next quarter",
            "context": {
                "domain": TaskDomain.BUSINESS,
                "current_focus": {
                    "backlog_size": 47,
                    "team_capacity": "6 developers for 12 weeks",
                    "business_objectives": ["increase_user_retention", "expand_enterprise_features"]
                },
                "user_profile": {"role": "engineering_manager", "experience_level": "advanced"},
                "available_resources": [
                    {"type": "user_feedback", "volume": "high"},
                    {"type": "analytics_data", "coverage": "comprehensive"},
                    {"type": "technical_debt_assessment", "status": "recent"}
                ]
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['instruction']}")
        decision = agent.process_instruction(example["instruction"], example["context"])
        
        print(f"Selected Action: {decision.selected_action.description}")
        print(f"Intent Type: {decision.selected_action.intent_type.value}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Expected Utility: {decision.expected_utility:.3f}")
        
        if decision.risk_assessment:
            print(f"Risk Assessment: {decision.risk_assessment}")
        
        if decision.alternative_actions:
            print(f"Alternatives: {len(decision.alternative_actions)} other options")
        
        print(f"Reasoning: {decision.reasoning}")
        print("-" * 40)
    
    print("="*50 + "\n")

# -------------------------
# 3. Research Domain Examples
# -------------------------

def research_domain_examples():
    """Examples in research domain"""
    print("=== Research Domain Examples ===\n")
    
    config = create_agent_config(
        agent_id="research_agent",
        name="Research Assistant",
        primary_domain=TaskDomain.RESEARCH
    )
    agent = UniversalGenericAgent(config)
    
    examples = [
        {
            "instruction": "Design a study to investigate the effectiveness of remote work on productivity",
            "context": {
                "domain": TaskDomain.RESEARCH,
                "current_focus": {
                    "research_question": "Does remote work increase or decrease employee productivity?",
                    "target_population": "knowledge workers",
                    "study_type": "experimental"
                },
                "user_profile": {"role": "graduate_student", "experience_level": "intermediate"},
                "available_resources": [
                    {"type": "university_ethics_board", "status": "available"},
                    {"type": "survey_platform", "name": "Qualtrics"},
                    {"type": "statistical_software", "name": "R"}
                ],
                "constraints": [
                    {"type": "ethical", "description": "Must get IRB approval"},
                    {"type": "timeline", "description": "6 months for completion"},
                    {"type": "budget", "description": "$5,000 research budget"}
                ]
            }
        },
        {
            "instruction": "Analyze this dataset and help me interpret the results",
            "context": {
                "domain": TaskDomain.RESEARCH,
                "current_focus": {
                    "dataset": "customer_satisfaction_survey.csv",
                    "variables": ["satisfaction_score", "usage_frequency", "demographics"],
                    "sample_size": 1247
                },
                "user_profile": {"role": "market_researcher", "experience_level": "beginner"},
                "available_resources": [
                    {"type": "dataset", "format": "CSV", "quality": "cleaned"},
                    {"type": "analysis_software", "name": "SPSS"},
                    {"type": "statistical_consultant", "availability": "limited"}
                ]
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['instruction']}")
        decision = agent.process_instruction(example["instruction"], example["context"])
        
        print(f"Selected Action: {decision.selected_action.description}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Research Methodology: {decision.selected_action.parameters.get('methodology', 'Not specified')}")
        print(f"Ethical Compliance: {decision.selected_action.parameters.get('ethical_compliance', 'Not specified')}")
        print(f"Reasoning: {decision.reasoning}")
        print("-" * 40)
    
    print("="*50 + "\n")

# -------------------------
# 4. Creative Domain Examples
# -------------------------

def creative_domain_examples():
    """Examples in creative domain"""
    print("=== Creative Domain Examples ===\n")
    
    config = create_agent_config(
        agent_id="creative_agent",
        name="Creative Assistant",
        primary_domain=TaskDomain.CREATIVE
    )
    agent = UniversalGenericAgent(config)
    
    examples = [
        {
            "instruction": "Design a logo for a sustainable fashion startup",
            "context": {
                "domain": TaskDomain.CREATIVE,
                "current_focus": {
                    "brand_name": "EcoThreads",
                    "target_audience": "environmentally conscious millennials",
                    "brand_values": ["sustainability", "style", "affordability"]
                },
                "user_profile": {"role": "graphic_designer", "experience_level": "intermediate"},
                "user_preferences": {
                    "style": ["modern", "minimalist", "organic"],
                    "colors": ["earth_tones", "green", "natural"]
                },
                "available_resources": [
                    {"type": "design_software", "name": "Adobe Creative Suite"},
                    {"type": "stock_imagery", "subscription": "Shutterstock"},
                    {"type": "brand_guidelines", "status": "draft"}
                ],
                "constraints": [
                    {"type": "budget", "description": "Logo should work in single color"},
                    {"type": "scalability", "description": "Must work from business card to billboard"},
                    {"type": "timeline", "description": "3 concepts needed by Friday"}
                ]
            }
        },
        {
            "instruction": "Write a compelling story for our brand's social media campaign",
            "context": {
                "domain": TaskDomain.CREATIVE,
                "current_focus": {
                    "campaign_theme": "Stories of transformation",
                    "platform": "Instagram",
                    "content_type": "carousel_post"
                },
                "user_profile": {"role": "content_creator", "experience_level": "advanced"},
                "available_resources": [
                    {"type": "customer_testimonials", "count": 50},
                    {"type": "brand_voice_guide", "status": "complete"},
                    {"type": "visual_assets", "type": "photography"}
                ],
                "constraints": [
                    {"type": "platform", "description": "Instagram character limits"},
                    {"type": "brand", "description": "Must maintain authentic tone"},
                    {"type": "legal", "description": "Customer permission required"}
                ]
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['instruction']}")
        decision = agent.process_instruction(example["instruction"], example["context"])
        
        print(f"Selected Action: {decision.selected_action.description}")
        print(f"Creative Approach: {decision.selected_action.parameters.get('creative_approach', 'Not specified')}")
        print(f"Audience Appropriate: {decision.selected_action.parameters.get('audience_appropriate', 'Not specified')}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        print("-" * 40)
    
    print("="*50 + "\n")

# -------------------------
# 5. Personal Domain Examples
# -------------------------

def personal_domain_examples():
    """Examples in personal domain"""
    print("=== Personal Domain Examples ===\n")
    
    config = create_agent_config(
        agent_id="personal_agent",
        name="Personal Assistant",
        primary_domain=TaskDomain.PERSONAL
    )
    agent = UniversalGenericAgent(config)
    
    examples = [
        {
            "instruction": "Help me create a morning routine that will improve my productivity",
            "context": {
                "domain": TaskDomain.PERSONAL,
                "current_focus": {
                    "goal": "increase daily productivity",
                    "current_routine": "wake up at 7am, check phone, rush to work",
                    "pain_points": ["feeling rushed", "low energy", "distracted"]
                },
                "user_profile": {
                    "lifestyle": "working_professional",
                    "work_schedule": "9am-6pm",
                    "sleep_schedule": "11pm-7am"
                },
                "user_preferences": {
                    "exercise": "prefer_light_activity",
                    "meditation": "beginner",
                    "nutrition": "quick_healthy_options"
                },
                "constraints": [
                    {"type": "time", "description": "maximum 1 hour for morning routine"},
                    {"type": "space", "description": "small apartment"},
                    {"type": "budget", "description": "minimal investment preferred"}
                ]
            }
        },
        {
            "instruction": "Organize my finances and create a savings plan",
            "context": {
                "domain": TaskDomain.PERSONAL,
                "current_focus": {
                    "goal": "save for house down payment",
                    "target_amount": "$30,000",
                    "timeline": "3 years"
                },
                "user_profile": {
                    "income": "monthly_salary",
                    "current_savings": "$5,000",
                    "debt": "student_loans"
                },
                "available_resources": [
                    {"type": "banking_app", "features": ["budgeting", "savings_goals"]},
                    {"type": "financial_knowledge", "level": "basic"}
                ],
                "constraints": [
                    {"type": "privacy", "description": "prefer not to share detailed financial info"},
                    {"type": "complexity", "description": "need simple, manageable approach"}
                ]
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['instruction']}")
        decision = agent.process_instruction(example["instruction"], example["context"])
        
        print(f"Selected Action: {decision.selected_action.description}")
        print(f"Personalized: {decision.selected_action.parameters.get('personalized', 'Not specified')}")
        print(f"Sustainable: {decision.selected_action.parameters.get('sustainable', 'Not specified')}")
        print(f"Privacy Conscious: {decision.selected_action.parameters.get('privacy_conscious', 'Not specified')}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        print("-" * 40)
    
    print("="*50 + "\n")

# -------------------------
# 6. Cross-Domain Comparison
# -------------------------

def cross_domain_comparison():
    """Compare how the same instruction is handled across domains"""
    print("=== Cross-Domain Comparison ===\n")
    
    instruction = "Help me organize and prioritize my tasks"
    
    domains = [
        TaskDomain.BUSINESS,
        TaskDomain.RESEARCH,
        TaskDomain.CREATIVE,
        TaskDomain.PERSONAL
    ]
    
    for domain in domains:
        config = create_agent_config(
            agent_id=f"{domain.value}_agent",
            name=f"{domain.value.title()} Agent",
            primary_domain=domain
        )
        agent = UniversalGenericAgent(config)
        
        # Create domain-appropriate context
        if domain == TaskDomain.BUSINESS:
            context = {
                "domain": domain,
                "current_focus": {"projects": ["product_launch", "quarterly_review", "team_hiring"]},
                "user_profile": {"role": "project_manager"}
            }
        elif domain == TaskDomain.RESEARCH:
            context = {
                "domain": domain,
                "current_focus": {"projects": ["literature_review", "data_collection", "paper_writing"]},
                "user_profile": {"role": "researcher"}
            }
        elif domain == TaskDomain.CREATIVE:
            context = {
                "domain": domain,
                "current_focus": {"projects": ["logo_design", "website_mockup", "brand_guidelines"]},
                "user_profile": {"role": "designer"}
            }
        else:  # PERSONAL
            context = {
                "domain": domain,
                "current_focus": {"projects": ["home_organization", "fitness_goals", "learning_spanish"]},
                "user_profile": {"lifestyle": "working_professional"}
            }
        
        decision = agent.process_instruction(instruction, context)
        
        print(f"{domain.value.upper()} DOMAIN:")
        print(f"  Action: {decision.selected_action.description}")
        print(f"  Intent: {decision.selected_action.intent_type.value}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Domain-specific approach: {decision.selected_action.parameters}")
        print()
    
    print("="*50 + "\n")

# -------------------------
# 7. Component Toggling Demo
# -------------------------

def component_toggling_demo():
    """Demonstrate toggling components on/off during runtime"""
    print("=== Runtime Component Toggling Demo ===\n")
    
    # Create agent with full config
    config = create_agent_config(
        agent_id="toggleable_agent",
        name="Toggleable Agent",
        primary_domain=TaskDomain.GENERIC
    )
    agent = UniversalGenericAgent(config)
    
    instruction = "Help me solve this complex problem"
    context = {
        "domain": TaskDomain.GENERIC,
        "current_focus": {"problem": "multi-factor decision with uncertain outcomes"},
        "user_profile": {"experience_level": "intermediate"}
    }
    
    print("INITIAL CONFIGURATION:")
    caps = agent.get_capabilities()
    print(f"  Theory of Mind: {caps['theory_of_mind']}")
    print(f"  Decision Theory: {caps['decision_theory']}")
    decision1 = agent.process_instruction(instruction, context)
    print(f"  Decision: {decision1.selected_action.description}")
    print(f"  Confidence: {decision1.confidence:.2f}")
    print()
    
    # Disable Theory of Mind
    new_config = ComponentConfig(
        enable_theory_of_mind=False,
        enable_decision_theory=True,
        enable_context_awareness=True
    )
    agent.update_config(new_config)
    
    print("AFTER DISABLING THEORY OF MIND:")
    caps = agent.get_capabilities()
    print(f"  Theory of Mind: {caps['theory_of_mind']}")
    print(f"  Decision Theory: {caps['decision_theory']}")
    decision2 = agent.process_instruction(instruction, context)
    print(f"  Decision: {decision2.selected_action.description}")
    print(f"  Confidence: {decision2.confidence:.2f}")
    print()
    
    # Disable Decision Theory too
    minimal_config = ComponentConfig(
        enable_theory_of_mind=False,
        enable_decision_theory=False,
        enable_context_awareness=True
    )
    agent.update_config(minimal_config)
    
    print("AFTER DISABLING BOTH ToM AND DT:")
    caps = agent.get_capabilities()
    print(f"  Theory of Mind: {caps['theory_of_mind']}")
    print(f"  Decision Theory: {caps['decision_theory']}")
    decision3 = agent.process_instruction(instruction, context)
    print(f"  Decision: {decision3.selected_action.description}")
    print(f"  Confidence: {decision3.confidence:.2f}")
    print()
    
    print("="*50 + "\n")

# -------------------------
# 8. Main Demo Function
# -------------------------

def run_all_multi_domain_examples():
    """Run all multi-domain examples"""
    print("Universal Generic Agent - Multi-Domain Examples")
    print("=" * 60)
    print()
    
    # Core functionality
    demonstrate_configurable_components()
    component_toggling_demo()
    
    # Domain-specific examples
    business_domain_examples()
    research_domain_examples()
    creative_domain_examples()
    personal_domain_examples()
    
    # Cross-domain comparison
    cross_domain_comparison()
    
    print("All multi-domain examples completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("✓ Configurable Theory of Mind and Decision Theory")
    print("✓ Domain-specific adaptations")
    print("✓ Runtime component toggling")
    print("✓ Cross-domain versatility")
    print("✓ Fallback modes for simple operations")

if __name__ == "__main__":
    run_all_multi_domain_examples()