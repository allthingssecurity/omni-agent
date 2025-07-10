# Universal Generic Agent with Configurable Theory of Mind & Decision Theory

A truly **universal AI agent** that can operate in **any domain** (not just coding) with **fully configurable** Theory of Mind and Decision Theory components that can be **enabled or disabled** as needed.

## 🌟 Key Features

### 🧠 **Configurable Theory of Mind**
- **Enable/Disable**: Turn Theory of Mind on or off
- **Few-shot learning** from similar patterns
- **Behavioral signal analysis** 
- **Deep intention inference** with adjustable depth
- **Mental state modeling** of user frustration, confidence, expertise

### 🎯 **Configurable Decision Theory**
- **Enable/Disable**: Turn Decision Theory on or off
- **Expected utility calculation** under uncertainty
- **Multi-factor utility scoring** (safety, efficiency, user satisfaction)
- **Risk assessment** across multiple dimensions
- **Uncertainty quantification** with confidence intervals

### 🌍 **Universal Domain Support**
- **Business**: Strategy, analysis, stakeholder management
- **Research**: Methodology, ethics, publication standards  
- **Creative**: Design, content, audience considerations
- **Personal**: Life management, habit building, privacy
- **Coding**: Development, debugging, optimization
- **Generic**: Fallback for any other domain

### ⚙️ **Complete Configurability**
- **Runtime toggling**: Enable/disable components during execution
- **Fallback modes**: Simple operation when advanced features disabled
- **Domain adaptation**: Automatic context and vocabulary adjustment
- **Risk tolerance**: Configurable from conservative to aggressive

## 🏗️ Architecture

```
📁 agent/
├── 📁 core/
│   ├── generic_types.py          # Universal domain-agnostic types
│   └── generic_agent.py          # Universal agent implementation
├── 📁 domains/
│   └── domain_adapters.py        # Domain-specific customizations
├── 📁 examples/
│   └── multi_domain_examples.py  # Examples across all domains
├── universal_orchestrator.py     # Main entry point
└── README_UNIVERSAL.md          # This file
```

## 🚀 Quick Start

### Simple Usage
```python
from universal_orchestrator import QuickStart

# Create simple agent (no ToM, no DT)
agent = QuickStart.simple_assistant()

# Create smart agent (full capabilities)
agent = QuickStart.smart_assistant()

# Domain-specific agents
business_agent = QuickStart.business_assistant()
research_agent = QuickStart.research_assistant()
creative_agent = QuickStart.creative_assistant()
personal_agent = QuickStart.personal_assistant()
```

### Custom Configuration
```python
from universal_orchestrator import UniversalAgentFactory
from core.generic_types import TaskDomain

# Create custom agent
agent = UniversalAgentFactory.create_custom_agent(
    agent_id="my_agent",
    name="My Custom Agent",
    primary_domain=TaskDomain.BUSINESS,
    enable_theory_of_mind=True,    # Enable ToM
    enable_decision_theory=False,  # Disable DT
    risk_tolerance=0.3             # Conservative
)
```

### Runtime Component Toggling
```python
from core.generic_types import ComponentConfig

# Start with full capabilities
agent = QuickStart.smart_assistant()

# Disable Theory of Mind during runtime
new_config = ComponentConfig(
    enable_theory_of_mind=False,
    enable_decision_theory=True,
    enable_context_awareness=True
)
agent.update_config(new_config)

# Now the agent uses simple goal inference instead of ToM
```

## 🎯 Domain Examples

### Business Domain
```python
from universal_orchestrator import quick_process

result = quick_process(
    instruction="Analyze our Q3 sales performance and recommend improvements",
    domain="business",
    current_focus={
        "report": "Q3_sales_report.xlsx",
        "metrics": ["revenue", "conversion_rate", "customer_acquisition"]
    },
    user_profile={"role": "sales_director"},
    constraints=[{"type": "timeline", "description": "Results needed for board meeting"}]
)

print(f"Action: {result.selected_action.description}")
print(f"Business considerations: {result.selected_action.parameters}")
```

### Research Domain
```python
result = quick_process(
    instruction="Design a study to investigate remote work effectiveness",
    domain="research",
    current_focus={
        "research_question": "Does remote work increase productivity?",
        "target_population": "knowledge workers"
    },
    constraints=[
        {"type": "ethical", "description": "Must get IRB approval"},
        {"type": "timeline", "description": "6 months for completion"}
    ]
)

print(f"Research methodology: {result.selected_action.parameters.get('methodology')}")
print(f"Ethical compliance: {result.selected_action.parameters.get('ethical_compliance')}")
```

### Creative Domain
```python
result = quick_process(
    instruction="Design a logo for a sustainable fashion startup",
    domain="creative",
    current_focus={
        "brand_name": "EcoThreads",
        "target_audience": "environmentally conscious millennials"
    },
    user_preferences={"style": ["modern", "minimalist", "organic"]},
    constraints=[{"type": "scalability", "description": "Must work from business card to billboard"}]
)

print(f"Creative approach: {result.selected_action.parameters.get('creative_approach')}")
print(f"Style considerations: {result.selected_action.parameters}")
```

### Personal Domain
```python
result = quick_process(
    instruction="Help me create a morning routine that improves productivity",
    domain="personal",
    current_focus={
        "goal": "increase daily productivity",
        "current_routine": "wake up at 7am, check phone, rush to work"
    },
    user_profile={"lifestyle": "working_professional"},
    constraints=[{"type": "time", "description": "maximum 1 hour for morning routine"}]
)

print(f"Personalized: {result.selected_action.parameters.get('personalized')}")
print(f"Sustainable: {result.selected_action.parameters.get('sustainable')}")
```

## 🔧 Configuration Options

### Component Configuration
```python
from core.generic_types import ComponentConfig

# Full configuration
full_config = ComponentConfig(
    # Core capabilities
    enable_theory_of_mind=True,
    enable_decision_theory=True,
    enable_context_awareness=True,
    enable_learning=True,
    
    # Theory of Mind settings
    tom_inference_depth=4,           # How deep to analyze intentions
    tom_use_few_shot=True,           # Use example-based learning
    tom_use_behavioral_signals=True, # Analyze user behavior patterns
    tom_confidence_threshold=0.3,    # Minimum confidence for goal inference
    
    # Decision Theory settings
    dt_use_expected_utility=True,    # Calculate expected utility
    dt_use_risk_assessment=True,     # Assess multiple risk types
    dt_uncertainty_penalty=0.2,      # Penalty for uncertainty
    dt_risk_tolerance=0.5,           # Risk tolerance (0=conservative, 1=aggressive)
    
    # Fallback settings
    fallback_to_simple_mode=True     # Fall back to simple mode if needed
)

# Minimal configuration
minimal_config = ComponentConfig(
    enable_theory_of_mind=False,
    enable_decision_theory=False,
    enable_context_awareness=True,
    fallback_to_simple_mode=True
)
```

### Domain-Specific Configuration
```python
from core.generic_types import TaskDomain, create_domain_config

# Each domain can have specific vocabulary and constraints
business_config = create_domain_config(TaskDomain.BUSINESS)
research_config = create_domain_config(TaskDomain.RESEARCH)
creative_config = create_domain_config(TaskDomain.CREATIVE)
```

## 🧪 Comparing Approaches

```python
from universal_orchestrator import compare_approaches

# Compare different agent configurations for same instruction
results = compare_approaches(
    instruction="Help me solve this complex problem",
    context={
        "domain": "generic",
        "current_focus": {"problem": "multi-factor decision with uncertainty"}
    }
)

for approach, decision in results.items():
    print(f"{approach}: {decision.selected_action.description}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Reasoning: {decision.reasoning}")
```

## 📊 Component Impact Analysis

| Component | When Enabled | When Disabled |
|-----------|-------------|---------------|
| **Theory of Mind** | Analyzes user mental state, infers hidden goals, considers frustration/confidence levels | Uses simple keyword-based goal inference |
| **Decision Theory** | Calculates expected utility, assesses multiple risk factors, quantifies uncertainty | Uses simple success probability ranking |
| **Both Enabled** | Deep intention understanding + optimal decision making | Fast, simple responses |
| **Both Disabled** | Fallback to basic keyword matching + highest success probability | Minimal processing, fastest response |

## 🎛️ Use Case Recommendations

### When to Enable Theory of Mind
- ✅ **Ambiguous instructions** ("fix this", "make it better")
- ✅ **User seems frustrated** or uncertain
- ✅ **Complex multi-step tasks** requiring goal inference
- ✅ **Collaborative scenarios** where understanding user intent matters
- ❌ **Simple, clear instructions** ("delete file X")
- ❌ **Performance-critical applications** requiring fast responses

### When to Enable Decision Theory
- ✅ **Multiple valid approaches** to choose from
- ✅ **High-stakes decisions** where optimal choice matters
- ✅ **Uncertain environments** with risk considerations
- ✅ **Resource-constrained scenarios** requiring efficiency
- ❌ **Single obvious solution** exists
- ❌ **Rapid prototyping** where "good enough" is sufficient

### Recommended Configurations by Use Case

| Use Case | ToM | DT | Context | Learning | Notes |
|----------|-----|----|---------|------------|-------|
| **Production Assistant** | ✅ | ✅ | ✅ | ❌ | Reliable, consistent decisions |
| **Research Tool** | ✅ | ✅ | ✅ | ✅ | Deep analysis, learning from patterns |
| **Rapid Prototyping** | ❌ | ❌ | ✅ | ❌ | Fast, simple responses |
| **Educational Demo** | ✅ | ✅ | ✅ | ✅ | Show all capabilities |
| **Embedded System** | ❌ | ✅ | ✅ | ❌ | Resource efficient but smart |

## 🔄 Runtime Flexibility

The agent can adapt its behavior during execution:

```python
# Start conservative for production
agent = QuickStart.business_assistant()
agent.update_config(ComponentConfig(dt_risk_tolerance=0.2))

# Switch to exploratory for brainstorming
agent.update_config(ComponentConfig(dt_risk_tolerance=0.8, tom_inference_depth=4))

# Go minimal for batch processing
agent.update_config(ComponentConfig(enable_theory_of_mind=False, enable_decision_theory=False))
```

## 📈 Performance Characteristics

| Configuration | Response Time | Accuracy | Resource Usage | Best For |
|---------------|---------------|----------|----------------|----------|
| **Full (ToM + DT)** | Slowest | Highest | Highest | Complex, ambiguous tasks |
| **ToM Only** | Medium | High | Medium | Understanding user intent |
| **DT Only** | Medium | High | Medium | Clear goals, optimal decisions |
| **Minimal** | Fastest | Good | Lowest | Simple, clear instructions |

## 🔍 Advanced Features

### Multi-Agent Comparison
```python
from universal_orchestrator import UniversalAgentManager

manager = UniversalAgentManager()
manager.register_agent(QuickStart.simple_assistant())
manager.register_agent(QuickStart.smart_assistant())

# Compare multiple agents
results = manager.process_with_multiple_agents(
    "Analyze this situation",
    {"domain": "business", "situation": "declining sales"}
)
```

### Domain Auto-Detection
```python
# Agent automatically detects domain from instruction
decision = quick_process("Create a marketing strategy for our new product")
# Automatically routes to business domain

decision = quick_process("Design an experiment to test this hypothesis")  
# Automatically routes to research domain
```

### Adaptive Risk Tolerance
```python
# Risk tolerance can be adjusted based on context
agent = QuickStart.business_assistant()

# Conservative for financial decisions
financial_config = ComponentConfig(dt_risk_tolerance=0.2)

# Aggressive for innovation projects  
innovation_config = ComponentConfig(dt_risk_tolerance=0.8)
```

## 🧠 How It Works

### Theory of Mind Pipeline
1. **Mental State Analysis**: Detect user frustration, confidence, expertise
2. **Behavioral Pattern Recognition**: Analyze recent activities and preferences  
3. **Few-Shot Inference**: Learn from similar historical examples
4. **Deep Goal Inference**: Multi-level reasoning about user intentions
5. **Ambiguity Assessment**: Quantify instruction clarity

### Decision Theory Pipeline
1. **Utility Calculation**: Multi-factor scoring (safety, efficiency, user satisfaction)
2. **Expected Utility**: Weighted by goal probability distribution
3. **Risk Assessment**: Technical, user experience, security, performance risks
4. **Uncertainty Quantification**: Confidence intervals and uncertainty sources
5. **Optimal Selection**: Choose action with highest expected utility

### Fallback Modes
- **ToM Disabled**: Uses simple keyword-based goal inference
- **DT Disabled**: Uses simple success probability ranking
- **Both Disabled**: Direct keyword-to-action mapping
- **Error Handling**: Graceful degradation to simpler approaches

## 📚 Example Applications

1. **Business Intelligence**: Strategy analysis with stakeholder consideration
2. **Academic Research**: Methodology design with ethical compliance
3. **Creative Projects**: Design work with audience and brand awareness
4. **Personal Productivity**: Life management with privacy and sustainability
5. **Software Development**: Code assistance with best practices
6. **Education**: Adaptive tutoring based on student understanding
7. **Healthcare**: Decision support with safety prioritization
8. **Legal**: Document analysis with compliance considerations

## 🔬 Research Foundation

This implementation combines:
- **Theory of Mind**: From cognitive psychology and the Tomcat paper
- **Decision Theory**: Expected utility theory under uncertainty
- **Multi-Domain Intelligence**: Domain adaptation and transfer learning
- **Configurable AI**: Modular architecture for flexible deployment

## 🌟 Why This Matters

Traditional AI agents are either:
- **Domain-specific** (only work for coding, business, etc.)
- **Fixed capability** (can't turn features on/off)
- **One-size-fits-all** (same behavior regardless of context)

This Universal Generic Agent is:
- **Truly universal** (works in any domain)
- **Fully configurable** (enable/disable any component)
- **Context-adaptive** (behavior changes based on situation)
- **Performance-scalable** (from minimal to full capabilities)

Perfect for applications that need **intelligent assistance across multiple domains** with **configurable complexity** based on **specific use case requirements**.

## 🚀 Getting Started

```bash
# Run multi-domain examples
python examples/multi_domain_examples.py

# Run universal orchestrator demo
python universal_orchestrator.py

# Test component toggling
python -c "from examples.multi_domain_examples import component_toggling_demo; component_toggling_demo()"
```

The future of AI assistance is **universal, configurable, and adaptive**. Welcome to the Universal Generic Agent! 🌟