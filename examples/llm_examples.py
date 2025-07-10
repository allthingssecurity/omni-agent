"""
LLM Integration Examples - Shows GPT-4o powered Universal Generic Agent

This module demonstrates the agent with actual GPT-4o integration for real AI responses.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_orchestrator import UniversalAgentFactory, QuickStart
from core.generic_types import TaskDomain, ComponentConfig, create_agent_config
from core.llm_integration import create_llm_config

def setup_api_key():
    """Setup OpenAI API key for examples"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr pass it directly to the create_*_agent() functions.")
        return None
    return api_key

def llm_vs_non_llm_comparison():
    """Compare LLM-powered vs rule-based agent responses"""
    print("=== LLM vs Non-LLM Comparison ===\n")
    
    api_key = setup_api_key()
    if not api_key:
        print("Skipping LLM examples - no API key provided\n")
        return
    
    instruction = "Help me create a comprehensive marketing strategy for our new AI-powered productivity app"
    context = {
        "domain": TaskDomain.BUSINESS,
        "current_focus": {
            "product": "AI productivity app",
            "target_market": "professionals and teams",
            "competition": "Notion, Todoist, Monday.com"
        },
        "user_profile": {"role": "product_manager", "experience_level": "intermediate"},
        "constraints": [
            {"type": "budget", "description": "$50K marketing budget"},
            {"type": "timeline", "description": "6-month launch timeline"}
        ],
        "available_resources": [
            {"type": "team", "description": "marketing team of 3"},
            {"type": "data", "description": "user research and market analysis"}
        ]
    }
    
    print(f"Instruction: '{instruction}'\n")
    
    # 1. LLM-Powered Agent (GPT-4o)
    print("🤖 GPT-4o POWERED AGENT:")
    try:
        llm_agent = UniversalAgentFactory.create_full_agent(
            agent_id="gpt4o_agent",
            name="GPT-4o Business Agent",
            api_key=api_key
        )
        
        llm_decision = llm_agent.process_instruction(instruction, context)
        
        print(f"✅ Action: {llm_decision.selected_action.description}")
        print(f"📊 Confidence: {llm_decision.confidence:.2f}")
        print(f"💡 Reasoning: {llm_decision.reasoning}")
        
        if llm_decision.selected_action.parameters:
            print(f"🔧 Parameters: {llm_decision.selected_action.parameters}")
        
        if llm_decision.alternative_actions:
            print(f"🔄 Alternatives: {len(llm_decision.alternative_actions)} other options")
        
    except Exception as e:
        print(f"❌ LLM Agent failed: {e}")
    
    print("\n" + "-"*50)
    
    # 2. Rule-Based Agent (No LLM)
    print("\n🔧 RULE-BASED AGENT (No LLM):")
    try:
        rule_agent = UniversalAgentFactory.create_full_agent(
            agent_id="rule_agent",
            name="Rule-Based Business Agent"
            # No API key = falls back to rule-based
        )
        
        rule_decision = rule_agent.process_instruction(instruction, context)
        
        print(f"✅ Action: {rule_decision.selected_action.description}")
        print(f"📊 Confidence: {rule_decision.confidence:.2f}")
        print(f"💡 Reasoning: {rule_decision.reasoning}")
        
    except Exception as e:
        print(f"❌ Rule-based Agent failed: {e}")
    
    print("\n" + "="*60 + "\n")

def domain_specific_llm_examples():
    """Show LLM responses across different domains"""
    print("=== Domain-Specific LLM Examples ===\n")
    
    api_key = setup_api_key()
    if not api_key:
        print("Skipping LLM examples - no API key provided\n")
        return
    
    examples = [
        {
            "domain": TaskDomain.BUSINESS,
            "instruction": "Analyze our competitor's pricing strategy and recommend our pricing approach",
            "context": {
                "current_focus": {
                    "competitors": ["Slack", "Microsoft Teams", "Discord"],
                    "our_product": "team communication platform",
                    "current_pricing": "freemium model"
                },
                "constraints": [{"type": "profitability", "description": "need 40% gross margin"}]
            }
        },
        {
            "domain": TaskDomain.RESEARCH,
            "instruction": "Design a study to measure the impact of remote work on team creativity",
            "context": {
                "current_focus": {
                    "research_question": "Does remote work enhance or hinder creative collaboration?",
                    "target_population": "creative teams in tech companies"
                },
                "constraints": [
                    {"type": "ethical", "description": "IRB approval required"},
                    {"type": "timeline", "description": "12-month study duration"}
                ]
            }
        },
        {
            "domain": TaskDomain.CREATIVE,
            "instruction": "Create a visual identity system for a sustainable tech startup",
            "context": {
                "current_focus": {
                    "brand_name": "GreenTech Innovations",
                    "values": ["sustainability", "innovation", "transparency"],
                    "target_audience": "environmentally conscious tech adopters"
                },
                "user_preferences": {"style": ["modern", "clean", "trustworthy"]}
            }
        },
        {
            "domain": TaskDomain.PERSONAL,
            "instruction": "Help me design a learning routine to master machine learning while working full-time",
            "context": {
                "current_focus": {
                    "goal": "become ML engineer in 18 months",
                    "current_skills": "Python programming, basic statistics",
                    "time_available": "10 hours per week"
                },
                "constraints": [
                    {"type": "time", "description": "full-time job limits study time"},
                    {"type": "budget", "description": "prefer free/low-cost resources"}
                ]
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        domain = example["domain"]
        instruction = example["instruction"]
        context = {"domain": domain, **example["context"]}
        
        print(f"Example {i} - {domain.value.upper()} DOMAIN:")
        print(f"Instruction: '{instruction}'")
        
        try:
            # Create domain-specific agent with LLM
            config = create_agent_config(
                agent_id=f"llm_{domain.value}_agent",
                name=f"LLM {domain.value.title()} Agent",
                primary_domain=domain
            )
            llm_config = create_llm_config(api_key=api_key)
            
            from core.generic_agent import UniversalGenericAgent
            agent = UniversalGenericAgent(config, llm_config)
            
            decision = agent.process_instruction(instruction, context)
            
            print(f"🎯 Action: {decision.selected_action.description}")
            print(f"📊 Confidence: {decision.confidence:.2f}")
            print(f"⚡ Intent: {decision.selected_action.intent_type.value}")
            
            # Show domain-specific considerations
            if decision.selected_action.parameters:
                domain_params = {k: v for k, v in decision.selected_action.parameters.items() 
                               if any(domain_word in k.lower() for domain_word in [domain.value, "domain", "specific"])}
                if domain_params:
                    print(f"🌍 Domain-specific: {domain_params}")
            
            print(f"💭 Reasoning: {decision.reasoning[:150]}...")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        print("-" * 40)
    
    print("="*60 + "\n")

def theory_of_mind_llm_demonstration():
    """Demonstrate Theory of Mind with LLM vs without"""
    print("=== Theory of Mind LLM Demonstration ===\n")
    
    api_key = setup_api_key()
    if not api_key:
        print("Skipping LLM examples - no API key provided\n")
        return
    
    # Ambiguous instruction that requires Theory of Mind
    instruction = "This isn't working"
    context = {
        "domain": TaskDomain.CODING,
        "current_focus": {"file": "user_authentication.py", "line": 47},
        "recent_activities": [
            {"type": "test_run", "result": "failed", "error": "authentication_error"},
            {"type": "code_change", "description": "modified login function"},
            {"type": "search", "query": "oauth token validation"}
        ],
        "user_behavior": {
            "frustration_indicators": ["multiple_test_runs", "repeated_searches"],
            "expertise_level": "intermediate",
            "recent_pattern": "debugging_authentication"
        },
        "error_logs": ["AuthenticationError: Invalid token format"]
    }
    
    print(f"Ambiguous Instruction: '{instruction}'")
    print("Context: User is debugging authentication, multiple failed tests, showing frustration\n")
    
    # 1. With Theory of Mind + LLM
    print("🧠 WITH THEORY OF MIND + GPT-4o:")
    try:
        tom_config = ComponentConfig(
            enable_theory_of_mind=True,
            enable_decision_theory=True,
            tom_inference_depth=4,
            tom_use_behavioral_signals=True
        )
        
        config = create_agent_config(
            agent_id="tom_llm_agent",
            name="ToM + LLM Agent",
            primary_domain=TaskDomain.CODING,
            components=tom_config
        )
        
        llm_config = create_llm_config(api_key=api_key)
        
        from core.generic_agent import UniversalGenericAgent
        tom_agent = UniversalGenericAgent(config, llm_config)
        
        tom_decision = tom_agent.process_instruction(instruction, context)
        
        print(f"🎯 Inferred Goal: {tom_decision.selected_action.description}")
        print(f"🧐 Mental State Analysis: Detected frustration with authentication debugging")
        print(f"💡 Theory of Mind Insight: User likely wants help understanding OAuth token validation")
        print(f"📊 Confidence: {tom_decision.confidence:.2f}")
        print(f"🔍 Reasoning: {tom_decision.reasoning}")
        
    except Exception as e:
        print(f"❌ ToM + LLM failed: {e}")
    
    print("\n" + "-"*50)
    
    # 2. Without Theory of Mind (simple mode)
    print("\n🔧 WITHOUT THEORY OF MIND (Simple Mode):")
    try:
        simple_config = ComponentConfig(
            enable_theory_of_mind=False,
            enable_decision_theory=False,
            fallback_to_simple_mode=True
        )
        
        config = create_agent_config(
            agent_id="simple_agent",
            name="Simple Agent", 
            primary_domain=TaskDomain.CODING,
            components=simple_config
        )
        
        llm_config = create_llm_config(api_key=api_key)
        
        from core.generic_agent import UniversalGenericAgent
        simple_agent = UniversalGenericAgent(config, llm_config)
        
        simple_decision = simple_agent.process_instruction(instruction, context)
        
        print(f"🎯 Action: {simple_decision.selected_action.description}")
        print(f"📊 Confidence: {simple_decision.confidence:.2f}")
        print(f"💭 Simple Reasoning: {simple_decision.reasoning}")
        
    except Exception as e:
        print(f"❌ Simple agent failed: {e}")
    
    print("\n" + "="*60 + "\n")

def performance_comparison():
    """Compare performance with and without LLM"""
    print("=== Performance Comparison ===\n")
    
    api_key = setup_api_key()
    if not api_key:
        print("Skipping LLM examples - no API key provided\n")
        return
    
    instruction = "Help me prioritize these feature requests"
    context = {
        "domain": TaskDomain.BUSINESS,
        "current_focus": {
            "features": ["mobile_app", "api_rate_limiting", "user_analytics", "social_login"],
            "team_capacity": "2 developers, 4 weeks"
        },
        "constraints": [{"type": "deadline", "description": "product launch in 1 month"}]
    }
    
    print(f"Instruction: '{instruction}'\n")
    
    # Test different configurations
    configs = [
        ("LLM + Full Features", ComponentConfig(True, True, True, False)),
        ("LLM + ToM Only", ComponentConfig(True, False, True, False)),
        ("LLM + DT Only", ComponentConfig(False, True, True, False)),
        ("Rule-based Full", ComponentConfig(True, True, True, False)),  # Will fallback without LLM
        ("Minimal Mode", ComponentConfig(False, False, True, False))
    ]
    
    for config_name, component_config in configs:
        print(f"🔧 {config_name}:")
        
        try:
            config = create_agent_config(
                agent_id=f"perf_test_{config_name.lower().replace(' ', '_')}",
                name=f"Performance Test {config_name}",
                primary_domain=TaskDomain.BUSINESS,
                components=component_config
            )
            
            # Only provide LLM config for LLM-enabled tests
            llm_config = None
            if "LLM" in config_name:
                llm_config = create_llm_config(api_key=api_key)
            
            from core.generic_agent import UniversalGenericAgent
            agent = UniversalGenericAgent(config, llm_config)
            
            import time
            start_time = time.time()
            decision = agent.process_instruction(instruction, context)
            end_time = time.time()
            
            print(f"   ⏱️  Response Time: {(end_time - start_time):.2f}s")
            print(f"   🎯 Action: {decision.selected_action.description[:80]}...")
            print(f"   📊 Confidence: {decision.confidence:.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print()
    
    print("="*60 + "\n")

def main():
    """Run all LLM integration examples"""
    print("🤖 Omni-Agent: GPT-4o Integration Examples")
    print("=" * 60)
    print()
    
    # Check if OpenAI is available
    try:
        import openai
        print("✅ OpenAI library detected")
    except ImportError:
        print("❌ OpenAI library not installed")
        print("Install with: pip install openai>=1.0.0")
        return
    
    # Run examples
    llm_vs_non_llm_comparison()
    domain_specific_llm_examples()
    theory_of_mind_llm_demonstration()
    performance_comparison()
    
    print("🎉 All LLM integration examples completed!")
    print("\n💡 Key Takeaways:")
    print("• GPT-4o provides much richer, context-aware responses")
    print("• Theory of Mind + LLM can understand ambiguous instructions")
    print("• Domain adaptation works seamlessly with LLM integration")
    print("• Graceful fallback to rule-based inference when LLM fails")
    print("• Configurable complexity: from minimal (fast) to full (intelligent)")

if __name__ == "__main__":
    main()