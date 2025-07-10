"""
Omni-Agent: Universal Generic Agent with Configurable Theory of Mind & Decision Theory

A truly universal AI agent that can operate in any domain (business, research, creative, 
personal, coding) with fully configurable Theory of Mind and Decision Theory components 
that can be enabled or disabled based on your specific needs.

Key Features:
- 🧠 Configurable Theory of Mind (enable/disable)
- 🎯 Configurable Decision Theory (enable/disable) 
- 🌍 Universal domain support (business, research, creative, personal, coding, generic)
- ⚙️ Runtime component toggling
- 🔄 Fallback modes for simple operation
- 📊 Performance scalability (minimal to full capabilities)

Quick Start:
    >>> from universal_orchestrator import QuickStart
    >>> agent = QuickStart.smart_assistant()
    >>> result = agent.process_instruction("Help me analyze this situation", context)
    >>> print(result.selected_action.description)

For more examples and documentation, see:
https://github.com/allthingssecurity/omni-agent
"""

from core.generic_types import (
    TaskDomain,
    ComponentConfig, 
    AgentConfig,
    GenericGoal,
    GenericAction,
    GenericContext,
    GenericDecision,
    create_agent_config,
    create_minimal_config,
    create_full_config,
)

from core.generic_agent import UniversalGenericAgent

from universal_orchestrator import (
    QuickStart,
    UniversalAgentFactory,
    UniversalAgentManager,
    quick_process,
    compare_approaches,
)

__version__ = "0.1.0"
__author__ = "All Things Security"
__email__ = "contact@allthingssecurity.dev"
__license__ = "MIT"

__all__ = [
    # Core types
    "TaskDomain",
    "ComponentConfig",
    "AgentConfig", 
    "GenericGoal",
    "GenericAction",
    "GenericContext",
    "GenericDecision",
    
    # Configuration helpers
    "create_agent_config",
    "create_minimal_config", 
    "create_full_config",
    
    # Core agent
    "UniversalGenericAgent",
    
    # Quick start interface
    "QuickStart",
    "UniversalAgentFactory",
    "UniversalAgentManager",
    
    # Convenience functions
    "quick_process",
    "compare_approaches",
]