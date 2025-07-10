"""
Generic Universal Agent with Configurable Theory of Mind and Decision Theory

This module provides a completely generic agent that can operate in any domain
with configurable Theory of Mind and Decision Theory components.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from core.generic_types import (
    GenericGoal, GenericAction, GenericContext, GenericDecision,
    TaskDomain, IntentType, UncertaintyType,
    ComponentConfig, DomainConfig, AgentConfig,
    GenericGoalInferenceEngine, GenericActionGenerator, GenericDecisionMaker,
    GenericContextProcessor, GenericLearningSystem, DomainAdapter,
    create_generic_context, infer_domain_from_context
)

try:
    from core.llm_integration import GPT4oIntegration, LLMConfig, create_llm_config
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# -------------------------
# 1. Simple Fallback Implementations
# -------------------------

class SimpleGoalInference(GenericGoalInferenceEngine):
    """Simple goal inference without Theory of Mind"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def infer_goals(self, instruction: str, context: GenericContext, config: ComponentConfig) -> List[GenericGoal]:
        """Simple keyword-based goal inference"""
        goals = []
        
        # Basic intent mapping
        intent_keywords = {
            IntentType.CREATE: ["create", "make", "build", "generate", "develop", "write", "design"],
            IntentType.MODIFY: ["change", "modify", "update", "fix", "improve", "edit", "adjust"],
            IntentType.DELETE: ["delete", "remove", "eliminate", "clear", "erase"],
            IntentType.ANALYZE: ["analyze", "examine", "review", "check", "investigate", "study"],
            IntentType.SEARCH: ["find", "search", "look", "locate", "discover"],
            IntentType.ORGANIZE: ["organize", "sort", "arrange", "structure", "categorize"],
            IntentType.COMMUNICATE: ["tell", "inform", "notify", "message", "communicate"],
            IntentType.LEARN: ["learn", "understand", "study", "research", "explore"],
            IntentType.PLAN: ["plan", "schedule", "prepare", "organize", "arrange"],
            IntentType.EXECUTE: ["do", "execute", "perform", "run", "carry out"],
            IntentType.VERIFY: ["verify", "check", "validate", "confirm", "test"],
            IntentType.OPTIMIZE: ["optimize", "improve", "enhance", "streamline", "efficient"]
        }
        
        instruction_lower = instruction.lower()
        
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in instruction_lower for keyword in keywords):
                goal = GenericGoal(
                    goal_id=f"simple_{intent_type.value}",
                    description=f"{intent_type.value.title()} based on instruction: {instruction}",
                    domain=context.domain,
                    intent_type=intent_type,
                    confidence=0.7,  # Default confidence
                    reasoning=f"Keyword match for {intent_type.value}"
                )
                goals.append(goal)
        
        # If no goals found, create a generic one
        if not goals:
            goal = GenericGoal(
                goal_id="simple_generic",
                description=f"Handle instruction: {instruction}",
                domain=context.domain,
                intent_type=IntentType.EXECUTE,
                confidence=0.5,
                reasoning="No specific intent detected, defaulting to execute"
            )
            goals.append(goal)
        
        return goals[:config.dt_risk_tolerance * 5 + 1]  # Limit based on risk tolerance
    
    def assess_ambiguity(self, instruction: str, context: GenericContext) -> float:
        """Simple ambiguity assessment"""
        # Simple heuristics
        ambiguity_indicators = ["this", "that", "it", "something", "somehow", "maybe", "perhaps"]
        ambiguity_score = sum(1 for indicator in ambiguity_indicators if indicator in instruction.lower())
        
        return min(1.0, ambiguity_score / len(ambiguity_indicators))

class LLMActionGenerator(GenericActionGenerator):
    """LLM-powered action generation"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        if LLM_AVAILABLE and llm_config:
            try:
                self.llm = GPT4oIntegration(llm_config)
                self.use_llm = True
                self.logger.info("Initialized Action Generator with GPT-4o integration")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM: {e}. Falling back to simple action generation.")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
            self.use_llm = False
    
    def generate_actions(self, goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> List[GenericAction]:
        """Generate actions using LLM or fallback to simple generation"""
        if self.use_llm and self.llm:
            try:
                return self._generate_actions_with_llm(goals, context, config)
            except Exception as e:
                self.logger.error(f"LLM action generation failed: {e}. Falling back to simple generation.")
                # Fall through to simple generation
        
        return self._generate_actions_simple(goals, context, config)
    
    def _generate_actions_with_llm(self, goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> List[GenericAction]:
        """Generate actions using GPT-4o"""
        llm_response = self.llm.generate_actions_with_llm(goals, context, config)
        
        actions = []
        for action_data in llm_response.get("candidate_actions", []):
            try:
                action = GenericAction(
                    action_id=action_data["action_id"],
                    description=action_data["description"],
                    domain=context.domain,
                    intent_type=IntentType(action_data["intent_type"]),
                    parameters=action_data["parameters"],
                    estimated_effort=action_data["estimated_effort"],
                    estimated_success_probability=action_data["estimated_success_probability"],
                    estimated_duration=action_data.get("estimated_duration_minutes"),
                    prerequisites=action_data.get("prerequisites", []),
                    outcomes=action_data.get("outcomes", []),
                    side_effects=action_data.get("side_effects", []),
                    reversibility=action_data.get("reversibility", 0.5),
                    resources_required=action_data.get("resources_required", [])
                )
                actions.append(action)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid action data from LLM: {action_data}. Error: {e}")
        
        return actions
    
    def _generate_actions_simple(self, goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> List[GenericAction]:
        """Simple action generation (fallback)"""
        return SimpleActionGenerator().generate_actions(goals, context, config)

class SimpleActionGenerator(GenericActionGenerator):
    """Simple action generation without complex simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_actions(self, goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> List[GenericAction]:
        """Generate simple actions based on goals"""
        actions = []
        
        for goal in goals:
            # Generate basic action for each goal
            action = GenericAction(
                action_id=f"action_{goal.goal_id}",
                description=f"Execute {goal.intent_type.value} for: {goal.description}",
                domain=goal.domain,
                intent_type=goal.intent_type,
                parameters={"target": goal.description, "method": "direct"},
                estimated_effort=goal.estimated_effort,
                estimated_success_probability=goal.confidence,
                outcomes=[f"Complete {goal.intent_type.value} task"],
                reversibility=0.5  # Default reversibility
            )
            actions.append(action)
            
            # Add alternative approach if confidence is low
            if goal.confidence < 0.6:
                alt_action = GenericAction(
                    action_id=f"alt_action_{goal.goal_id}",
                    description=f"Ask for clarification about: {goal.description}",
                    domain=goal.domain,
                    intent_type=IntentType.COMMUNICATE,
                    parameters={"question": f"Could you clarify what you mean by '{goal.description}'?"},
                    estimated_effort=0.1,
                    estimated_success_probability=0.9,
                    outcomes=["Get clarification from user"],
                    reversibility=1.0
                )
                actions.append(alt_action)
        
        return actions

class SimpleDecisionMaker(GenericDecisionMaker):
    """Simple decision making without complex utility calculation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def make_decision(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> GenericDecision:
        """Make simple decision based on success probability"""
        if not actions:
            return self._create_no_action_decision()
        
        # Simple scoring: success probability * (1 - effort)
        def simple_score(action):
            return action.estimated_success_probability * (1 - action.estimated_effort)
        
        # Sort by score
        sorted_actions = sorted(actions, key=simple_score, reverse=True)
        selected_action = sorted_actions[0]
        alternatives = sorted_actions[1:3]  # Top 2 alternatives
        
        return GenericDecision(
            decision_id=f"decision_{int(time.time())}",
            selected_action=selected_action,
            alternative_actions=alternatives,
            reasoning=f"Selected action with highest score: {simple_score(selected_action):.2f}",
            confidence=selected_action.estimated_success_probability,
            expected_utility=simple_score(selected_action),
            expected_outcomes=selected_action.outcomes,
            risk_assessment={"simple_risk": 1.0 - selected_action.estimated_success_probability}
        )
    
    def _create_no_action_decision(self) -> GenericDecision:
        """Create decision when no actions are available"""
        no_action = GenericAction(
            action_id="no_action",
            description="No suitable action found",
            domain=TaskDomain.GENERIC,
            intent_type=IntentType.COMMUNICATE,
            parameters={"message": "I'm not sure how to help with that. Could you provide more details?"},
            estimated_effort=0.1,
            estimated_success_probability=0.5,
            outcomes=["Ask user for clarification"]
        )
        
        return GenericDecision(
            decision_id=f"no_decision_{int(time.time())}",
            selected_action=no_action,
            alternative_actions=[],
            reasoning="No suitable actions found, asking for clarification",
            confidence=0.5,
            expected_utility=0.5,
            expected_outcomes=no_action.outcomes
        )

# -------------------------
# 2. Advanced Theory of Mind Implementation
# -------------------------

class TheoryOfMindEngine(GenericGoalInferenceEngine):
    """Advanced Theory of Mind goal inference engine with GPT-4o integration"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.few_shot_examples = {}
        self.behavioral_patterns = {}
        
        # Initialize LLM integration if available
        if LLM_AVAILABLE and llm_config:
            try:
                self.llm = GPT4oIntegration(llm_config)
                self.use_llm = True
                self.logger.info("Initialized Theory of Mind with GPT-4o integration")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based inference.")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
            self.use_llm = False
            if not LLM_AVAILABLE:
                self.logger.info("LLM not available. Using rule-based Theory of Mind inference.")
    
    def infer_goals(self, instruction: str, context: GenericContext, config: ComponentConfig) -> List[GenericGoal]:
        """Advanced goal inference using Theory of Mind"""
        # Use GPT-4o if available, otherwise fall back to rule-based inference
        if self.use_llm and self.llm:
            try:
                return self._infer_goals_with_llm(instruction, context, config)
            except Exception as e:
                self.logger.error(f"LLM goal inference failed: {e}. Falling back to rule-based inference.")
                # Fall through to rule-based inference
        
        # Rule-based inference (original implementation)
        return self._infer_goals_rule_based(instruction, context, config)
    
    def _infer_goals_with_llm(self, instruction: str, context: GenericContext, config: ComponentConfig) -> List[GenericGoal]:
        """Infer goals using GPT-4o"""
        llm_response = self.llm.infer_goals_with_tom(instruction, context, config)
        
        goals = []
        for goal_data in llm_response.get("inferred_goals", []):
            try:
                goal = GenericGoal(
                    goal_id=goal_data["goal_id"],
                    description=goal_data["description"],
                    domain=context.domain,
                    intent_type=IntentType(goal_data["intent_type"]),
                    confidence=goal_data["confidence"],
                    reasoning=goal_data["reasoning"],
                    priority=goal_data.get("priority", 1)
                )
                goals.append(goal)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid goal data from LLM: {goal_data}. Error: {e}")
        
        return goals
    
    def _infer_goals_rule_based(self, instruction: str, context: GenericContext, config: ComponentConfig) -> List[GenericGoal]:
        """Rule-based goal inference (fallback)"""
        goals = []
        
        # Step 1: Analyze user mental state
        mental_state = self._analyze_user_mental_state(instruction, context)
        
        # Step 2: Apply few-shot reasoning if enabled
        if config.tom_use_few_shot:
            few_shot_goals = self._few_shot_inference(instruction, context, mental_state)
            goals.extend(few_shot_goals)
        
        # Step 3: Behavioral signal analysis if enabled
        if config.tom_use_behavioral_signals:
            behavioral_goals = self._behavioral_signal_inference(instruction, context, mental_state)
            goals.extend(behavioral_goals)
        
        # Step 4: Deep inference based on configured depth
        if config.tom_inference_depth > 1:
            deep_goals = self._deep_inference(instruction, context, mental_state, config.tom_inference_depth)
            goals.extend(deep_goals)
        
        # Step 5: Filter and rank goals
        filtered_goals = self._filter_and_rank_goals(goals, config)
        
        return filtered_goals
    
    def _analyze_user_mental_state(self, instruction: str, context: GenericContext) -> Dict[str, Any]:
        """Analyze user's mental state from instruction and context"""
        mental_state = {
            "urgency": self._detect_urgency(instruction),
            "frustration": self._detect_frustration(instruction, context),
            "confidence": self._detect_user_confidence(instruction, context),
            "expertise": self._estimate_expertise(context),
            "goals_clarity": self._assess_goal_clarity(instruction),
            "context_awareness": self._assess_context_awareness(context)
        }
        
        return mental_state
    
    def _detect_urgency(self, instruction: str) -> float:
        """Detect urgency in instruction"""
        urgent_keywords = ["urgent", "asap", "quickly", "immediately", "emergency", "critical", "deadline"]
        return min(1.0, sum(1 for keyword in urgent_keywords if keyword in instruction.lower()) / len(urgent_keywords))
    
    def _detect_frustration(self, instruction: str, context: GenericContext) -> float:
        """Detect user frustration"""
        frustration_keywords = ["frustrated", "stuck", "confused", "problem", "issue", "not working", "failed"]
        frustration_score = sum(1 for keyword in frustration_keywords if keyword in instruction.lower())
        
        # Check context for repeated failures
        if context.user_history:
            recent_failures = sum(1 for event in context.user_history[-5:] if event.get("outcome") == "failure")
            frustration_score += recent_failures * 0.2
        
        return min(1.0, frustration_score / 5)
    
    def _detect_user_confidence(self, instruction: str, context: GenericContext) -> float:
        """Detect user confidence level"""
        confident_keywords = ["I want", "I need", "do this", "make this"]
        uncertain_keywords = ["maybe", "perhaps", "could you", "would you", "not sure", "confused"]
        
        confident_score = sum(1 for keyword in confident_keywords if keyword in instruction.lower())
        uncertain_score = sum(1 for keyword in uncertain_keywords if keyword in instruction.lower())
        
        return max(0.1, min(1.0, (confident_score - uncertain_score + 1) / 2))
    
    def _estimate_expertise(self, context: GenericContext) -> float:
        """Estimate user expertise based on context"""
        if not context.user_profile:
            return 0.5  # Default
        
        expertise_indicators = context.user_profile.get("experience_level", "intermediate")
        expertise_map = {
            "beginner": 0.2,
            "intermediate": 0.5,
            "advanced": 0.8,
            "expert": 0.95
        }
        
        return expertise_map.get(expertise_indicators, 0.5)
    
    def _assess_goal_clarity(self, instruction: str) -> float:
        """Assess how clear the user's goals are"""
        vague_indicators = ["this", "that", "it", "something", "somehow", "stuff", "things"]
        specific_indicators = ["create", "delete", "modify", "analyze", "calculate", "generate"]
        
        vague_score = sum(1 for indicator in vague_indicators if indicator in instruction.lower())
        specific_score = sum(1 for indicator in specific_indicators if indicator in instruction.lower())
        
        return max(0.1, min(1.0, (specific_score - vague_score + 1) / 2))
    
    def _assess_context_awareness(self, context: GenericContext) -> float:
        """Assess how context-aware the user is"""
        context_score = 0.0
        
        if context.current_focus:
            context_score += 0.3
        if context.recent_activities:
            context_score += 0.3
        if context.user_preferences:
            context_score += 0.2
        if context.constraints:
            context_score += 0.2
        
        return context_score
    
    def _few_shot_inference(self, instruction: str, context: GenericContext, mental_state: Dict[str, Any]) -> List[GenericGoal]:
        """Use few-shot examples for goal inference"""
        # This would use a database of similar examples
        # For now, return simple goals
        return []
    
    def _behavioral_signal_inference(self, instruction: str, context: GenericContext, mental_state: Dict[str, Any]) -> List[GenericGoal]:
        """Infer goals from behavioral signals"""
        goals = []
        
        # Analyze recent activities
        if context.recent_activities:
            activity_patterns = self._analyze_activity_patterns(context.recent_activities)
            for pattern in activity_patterns:
                goal = GenericGoal(
                    goal_id=f"behavioral_{pattern['type']}",
                    description=f"Continue {pattern['type']} activity",
                    domain=context.domain,
                    intent_type=IntentType.EXECUTE,
                    confidence=pattern['confidence'],
                    reasoning=f"Behavioral pattern suggests {pattern['type']} activity"
                )
                goals.append(goal)
        
        return goals
    
    def _deep_inference(self, instruction: str, context: GenericContext, mental_state: Dict[str, Any], depth: int) -> List[GenericGoal]:
        """Deep inference with multiple levels of reasoning"""
        goals = []
        
        # Level 1: Direct interpretation
        if depth >= 1:
            direct_goals = self._direct_interpretation(instruction, context)
            goals.extend(direct_goals)
        
        # Level 2: Implied goals
        if depth >= 2:
            implied_goals = self._infer_implied_goals(instruction, context, mental_state)
            goals.extend(implied_goals)
        
        # Level 3: Meta-goals
        if depth >= 3:
            meta_goals = self._infer_meta_goals(instruction, context, mental_state)
            goals.extend(meta_goals)
        
        # Level 4: Social/contextual goals
        if depth >= 4:
            social_goals = self._infer_social_goals(instruction, context, mental_state)
            goals.extend(social_goals)
        
        return goals
    
    def _analyze_activity_patterns(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in recent activities"""
        patterns = []
        
        if not activities:
            return patterns
        
        # Group by activity type
        activity_types = {}
        for activity in activities:
            act_type = activity.get("type", "unknown")
            if act_type not in activity_types:
                activity_types[act_type] = []
            activity_types[act_type].append(activity)
        
        # Find dominant patterns
        total_activities = len(activities)
        for act_type, acts in activity_types.items():
            if len(acts) / total_activities > 0.3:  # More than 30% of activities
                patterns.append({
                    "type": act_type,
                    "confidence": len(acts) / total_activities,
                    "recent_count": len(acts)
                })
        
        return patterns
    
    def _direct_interpretation(self, instruction: str, context: GenericContext) -> List[GenericGoal]:
        """Direct interpretation of instruction"""
        # Use the simple inference as base
        simple_engine = SimpleGoalInference()
        return simple_engine.infer_goals(instruction, context, ComponentConfig())
    
    def _infer_implied_goals(self, instruction: str, context: GenericContext, mental_state: Dict[str, Any]) -> List[GenericGoal]:
        """Infer goals that are implied but not explicitly stated"""
        goals = []
        
        # If user seems frustrated, they might need help understanding
        if mental_state.get("frustration", 0) > 0.5:
            goal = GenericGoal(
                goal_id="help_understand",
                description="Help user understand the problem better",
                domain=context.domain,
                intent_type=IntentType.LEARN,
                confidence=mental_state["frustration"],
                reasoning="High frustration suggests need for understanding"
            )
            goals.append(goal)
        
        # If user has low confidence, they might need guidance
        if mental_state.get("confidence", 0.5) < 0.3:
            goal = GenericGoal(
                goal_id="provide_guidance",
                description="Provide step-by-step guidance",
                domain=context.domain,
                intent_type=IntentType.COMMUNICATE,
                confidence=1.0 - mental_state["confidence"],
                reasoning="Low confidence suggests need for guidance"
            )
            goals.append(goal)
        
        return goals
    
    def _infer_meta_goals(self, instruction: str, context: GenericContext, mental_state: Dict[str, Any]) -> List[GenericGoal]:
        """Infer meta-goals (goals about goals)"""
        goals = []
        
        # Learning goal if user seems to be exploring
        if mental_state.get("expertise", 0.5) < 0.4:
            goal = GenericGoal(
                goal_id="meta_learn",
                description="Learn while completing the task",
                domain=context.domain,
                intent_type=IntentType.LEARN,
                confidence=0.6,
                reasoning="Low expertise suggests learning opportunity"
            )
            goals.append(goal)
        
        return goals
    
    def _infer_social_goals(self, instruction: str, context: GenericContext, mental_state: Dict[str, Any]) -> List[GenericGoal]:
        """Infer social/contextual goals"""
        goals = []
        
        # Check if this is a collaborative context
        if context.social_context and context.social_context.get("collaborative", False):
            goal = GenericGoal(
                goal_id="social_collaborate",
                description="Collaborate effectively with others",
                domain=context.domain,
                intent_type=IntentType.COMMUNICATE,
                confidence=0.7,
                reasoning="Collaborative context suggests social goal"
            )
            goals.append(goal)
        
        return goals
    
    def _filter_and_rank_goals(self, goals: List[GenericGoal], config: ComponentConfig) -> List[GenericGoal]:
        """Filter and rank goals based on confidence and relevance"""
        # Filter by confidence threshold
        filtered = [g for g in goals if g.confidence >= config.tom_confidence_threshold]
        
        # Remove duplicates
        unique_goals = []
        seen_ids = set()
        for goal in filtered:
            if goal.goal_id not in seen_ids:
                unique_goals.append(goal)
                seen_ids.add(goal.goal_id)
        
        # Sort by confidence
        unique_goals.sort(key=lambda g: g.confidence, reverse=True)
        
        return unique_goals
    
    def assess_ambiguity(self, instruction: str, context: GenericContext) -> float:
        """Advanced ambiguity assessment"""
        simple_ambiguity = SimpleGoalInference().assess_ambiguity(instruction, context)
        
        # Enhance with context analysis
        context_clarity = self._assess_context_awareness(context)
        goal_clarity = self._assess_goal_clarity(instruction)
        
        # Combine factors
        overall_ambiguity = (simple_ambiguity * 0.4 + (1 - context_clarity) * 0.3 + (1 - goal_clarity) * 0.3)
        
        return overall_ambiguity

# -------------------------
# 3. Advanced Decision Theory Implementation
# -------------------------

class DecisionTheoryEngine(GenericDecisionMaker):
    """Advanced decision making using decision theory with GPT-4o integration"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        if LLM_AVAILABLE and llm_config:
            try:
                self.llm = GPT4oIntegration(llm_config)
                self.use_llm = True
                self.logger.info("Initialized Decision Theory with GPT-4o integration")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based decision making.")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
            self.use_llm = False
    
    def make_decision(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> GenericDecision:
        """Make decision using decision theory"""
        if not actions:
            return SimpleDecisionMaker()._create_no_action_decision()
        
        # Use GPT-4o if available, otherwise fall back to rule-based decision making
        if self.use_llm and self.llm:
            try:
                return self._make_decision_with_llm(actions, goals, context, config)
            except Exception as e:
                self.logger.error(f"LLM decision making failed: {e}. Falling back to rule-based decision making.")
                # Fall through to rule-based decision making
        
        return self._make_decision_rule_based(actions, goals, context, config)
    
    def _make_decision_with_llm(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> GenericDecision:
        """Make decision using GPT-4o"""
        llm_response = self.llm.make_decision_with_dt(actions, goals, context, config)
        decision_data = llm_response.get("decision_analysis", {})
        
        # Find the selected action
        selected_action_id = decision_data.get("selected_action_id")
        selected_action = next((a for a in actions if a.action_id == selected_action_id), actions[0] if actions else None)
        
        # Find alternative actions
        alt_action_ids = decision_data.get("alternative_actions", [])
        alternative_actions = [a for a in actions if a.action_id in alt_action_ids]
        
        return GenericDecision(
            decision_id=f"llm_decision_{int(time.time())}",
            selected_action=selected_action,
            alternative_actions=alternative_actions,
            reasoning=decision_data.get("reasoning", "LLM-based decision"),
            confidence=decision_data.get("confidence", 0.7),
            expected_utility=decision_data.get("expected_utility", 0.7),
            expected_outcomes=selected_action.outcomes if selected_action else [],
            risk_assessment=decision_data.get("risk_assessment", {}),
            fallback_plan=decision_data.get("fallback_plan")
        )
    
    def _make_decision_rule_based(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> GenericDecision:
        """Make decision using rule-based decision theory (fallback)"""
        
        # Step 1: Calculate expected utility for each action
        utilities = []
        for action in actions:
            utility = self._calculate_expected_utility(action, goals, context, config)
            utilities.append((action, utility))
        
        # Step 2: Apply risk assessment if enabled
        if config.dt_use_risk_assessment:
            utilities = self._apply_risk_assessment(utilities, context, config)
        
        # Step 3: Select best action
        utilities.sort(key=lambda x: x[1], reverse=True)
        selected_action, selected_utility = utilities[0]
        alternatives = [action for action, _ in utilities[1:3]]
        
        # Step 4: Calculate decision confidence
        confidence = self._calculate_decision_confidence(utilities, config)
        
        # Step 5: Assess uncertainty sources
        uncertainty_sources = self._assess_uncertainty_sources(selected_action, goals, context)
        
        # Step 6: Generate reasoning
        reasoning = self._generate_reasoning(selected_action, selected_utility, utilities, goals)
        
        return GenericDecision(
            decision_id=f"dt_decision_{int(time.time())}",
            selected_action=selected_action,
            alternative_actions=alternatives,
            reasoning=reasoning,
            confidence=confidence,
            expected_utility=selected_utility,
            expected_outcomes=selected_action.outcomes,
            uncertainty_sources=uncertainty_sources,
            risk_assessment=self._assess_action_risks(selected_action, context)
        )
    
    def _calculate_expected_utility(self, action: GenericAction, goals: List[GenericGoal], context: GenericContext, config: ComponentConfig) -> float:
        """Calculate expected utility of an action"""
        if not config.dt_use_expected_utility:
            # Fall back to simple scoring
            return action.estimated_success_probability * (1 - action.estimated_effort)
        
        # Goal probability distribution
        total_confidence = sum(goal.confidence for goal in goals)
        if total_confidence == 0:
            return 0.0
        
        expected_utility = 0.0
        for goal in goals:
            goal_probability = goal.confidence / total_confidence
            action_utility = self._calculate_action_utility_for_goal(action, goal, context)
            expected_utility += goal_probability * action_utility
        
        return expected_utility
    
    def _calculate_action_utility_for_goal(self, action: GenericAction, goal: GenericGoal, context: GenericContext) -> float:
        """Calculate utility of an action for a specific goal"""
        utility_factors = {
            "success_probability": action.estimated_success_probability * 0.3,
            "effort_efficiency": (1 - action.estimated_effort) * 0.2,
            "goal_alignment": self._assess_goal_alignment(action, goal) * 0.2,
            "reversibility": action.reversibility * 0.1,
            "resource_efficiency": self._assess_resource_efficiency(action, context) * 0.1,
            "safety": self._assess_safety(action, context) * 0.1
        }
        
        return sum(utility_factors.values())
    
    def _assess_goal_alignment(self, action: GenericAction, goal: GenericGoal) -> float:
        """Assess how well action aligns with goal"""
        if action.intent_type == goal.intent_type:
            return 1.0
        
        # Check for compatible intent types
        compatible_intents = {
            IntentType.CREATE: [IntentType.PLAN, IntentType.EXECUTE],
            IntentType.MODIFY: [IntentType.ANALYZE, IntentType.OPTIMIZE],
            IntentType.LEARN: [IntentType.ANALYZE, IntentType.SEARCH],
            IntentType.COMMUNICATE: [IntentType.VERIFY, IntentType.SEARCH]
        }
        
        if goal.intent_type in compatible_intents:
            if action.intent_type in compatible_intents[goal.intent_type]:
                return 0.7
        
        return 0.3  # Default moderate alignment
    
    def _assess_resource_efficiency(self, action: GenericAction, context: GenericContext) -> float:
        """Assess resource efficiency of action"""
        available_resources = context.available_resources
        required_resources = action.resources_required
        
        if not required_resources:
            return 1.0  # No resources required
        
        if not available_resources:
            return 0.3  # Unknown resource availability
        
        # Simple check: are required resources available?
        available_resource_types = {res.get("type") for res in available_resources}
        required_resource_types = set(required_resources)
        
        if required_resource_types.issubset(available_resource_types):
            return 0.9
        else:
            return 0.3
    
    def _assess_safety(self, action: GenericAction, context: GenericContext) -> float:
        """Assess safety of action"""
        safety_score = action.reversibility  # Higher reversibility = safer
        
        # Check for dangerous side effects
        dangerous_effects = ["permanent", "irreversible", "delete", "remove", "destroy"]
        for effect in action.side_effects:
            if any(danger in effect.lower() for danger in dangerous_effects):
                safety_score *= 0.5
        
        return safety_score
    
    def _apply_risk_assessment(self, utilities: List[Tuple[GenericAction, float]], context: GenericContext, config: ComponentConfig) -> List[Tuple[GenericAction, float]]:
        """Apply risk assessment to utilities"""
        risk_adjusted = []
        
        for action, utility in utilities:
            risks = self._assess_action_risks(action, context)
            total_risk = sum(risks.values()) / len(risks) if risks else 0.0
            
            # Apply uncertainty penalty
            risk_penalty = total_risk * config.dt_uncertainty_penalty
            adjusted_utility = utility * (1 - risk_penalty)
            
            risk_adjusted.append((action, adjusted_utility))
        
        return risk_adjusted
    
    def _assess_action_risks(self, action: GenericAction, context: GenericContext) -> Dict[str, float]:
        """Assess various risks of an action"""
        risks = {
            "execution_risk": 1.0 - action.estimated_success_probability,
            "resource_risk": self._assess_resource_risk(action, context),
            "reversibility_risk": 1.0 - action.reversibility,
            "complexity_risk": action.estimated_effort,
            "constraint_risk": self._assess_constraint_risk(action, context)
        }
        
        return risks
    
    def _assess_resource_risk(self, action: GenericAction, context: GenericContext) -> float:
        """Assess risk related to resource availability"""
        if not action.resources_required:
            return 0.0
        
        available_resources = context.available_resources
        if not available_resources:
            return 0.5  # Unknown = moderate risk
        
        # Check resource availability
        available_types = {res.get("type") for res in available_resources}
        required_types = set(action.resources_required)
        
        missing_resources = required_types - available_types
        return len(missing_resources) / len(required_types) if required_types else 0.0
    
    def _assess_constraint_risk(self, action: GenericAction, context: GenericContext) -> float:
        """Assess risk of violating constraints"""
        constraints = context.constraints
        if not constraints:
            return 0.0
        
        # Simple check: does action violate any constraints?
        violations = 0
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            if constraint_type in action.description.lower():
                violations += 1
        
        return violations / len(constraints) if constraints else 0.0
    
    def _calculate_decision_confidence(self, utilities: List[Tuple[GenericAction, float]], config: ComponentConfig) -> float:
        """Calculate confidence in the decision"""
        if len(utilities) < 2:
            return 0.8
        
        # Calculate margin between best and second best
        best_utility = utilities[0][1]
        second_best_utility = utilities[1][1]
        
        margin = best_utility - second_best_utility
        confidence = min(0.95, 0.5 + margin)
        
        return confidence
    
    def _assess_uncertainty_sources(self, action: GenericAction, goals: List[GenericGoal], context: GenericContext) -> List[UncertaintyType]:
        """Assess sources of uncertainty"""
        uncertainty_sources = []
        
        # Check goal ambiguity
        if any(goal.confidence < 0.5 for goal in goals):
            uncertainty_sources.append(UncertaintyType.AMBIGUOUS_INTENT)
        
        # Check context completeness
        if not context.current_focus or not context.user_preferences:
            uncertainty_sources.append(UncertaintyType.INCOMPLETE_INFORMATION)
        
        # Check conflicting goals
        intent_types = {goal.intent_type for goal in goals}
        if len(intent_types) > 2:
            uncertainty_sources.append(UncertaintyType.CONFLICTING_GOALS)
        
        # Check resource constraints
        if not context.available_resources:
            uncertainty_sources.append(UncertaintyType.RESOURCE_CONSTRAINTS)
        
        return uncertainty_sources
    
    def _generate_reasoning(self, selected_action: GenericAction, selected_utility: float, utilities: List[Tuple[GenericAction, float]], goals: List[GenericGoal]) -> str:
        """Generate reasoning for the decision"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Selected action: {selected_action.description}")
        reasoning_parts.append(f"Expected utility: {selected_utility:.3f}")
        
        if len(utilities) > 1:
            margin = selected_utility - utilities[1][1]
            reasoning_parts.append(f"Margin over next best: {margin:.3f}")
        
        # Top goal
        if goals:
            top_goal = max(goals, key=lambda g: g.confidence)
            reasoning_parts.append(f"Primary goal: {top_goal.description}")
        
        return " | ".join(reasoning_parts)

# -------------------------
# 4. Universal Generic Agent
# -------------------------

class UniversalGenericAgent:
    """Universal agent that can operate in any domain with configurable components"""
    
    def __init__(self, config: AgentConfig, llm_config: Optional[LLMConfig] = None):
        self.config = config
        self.llm_config = llm_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM configuration if not provided
        if llm_config is None and LLM_AVAILABLE:
            try:
                self.llm_config = create_llm_config()
                self.logger.info("Created default LLM configuration")
            except Exception as e:
                self.logger.warning(f"Failed to create LLM config: {e}")
                self.llm_config = None
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components based on configuration"""
        # Goal inference component
        if self.config.components.enable_theory_of_mind:
            self.goal_inference = TheoryOfMindEngine(self.llm_config)
        else:
            self.goal_inference = SimpleGoalInference()
        
        # Action generation component
        self.action_generator = LLMActionGenerator(self.llm_config) if LLM_AVAILABLE and self.llm_config else SimpleActionGenerator()
        
        # Decision making component
        if self.config.components.enable_decision_theory:
            self.decision_maker = DecisionTheoryEngine(self.llm_config)
        else:
            self.decision_maker = SimpleDecisionMaker()
        
        # Context processor
        self.context_processor = None  # Could add sophisticated context processing
        
        # Learning system
        self.learning_system = None  # Could add learning capabilities
    
    def process_instruction(self, instruction: str, raw_context: Dict[str, Any]) -> GenericDecision:
        """Process instruction and return decision"""
        try:
            # Step 1: Process context
            context = self._process_context(raw_context)
            
            # Step 2: Infer goals
            goals = self._infer_goals(instruction, context)
            
            # Step 3: Generate actions
            actions = self._generate_actions(goals, context)
            
            # Step 4: Make decision
            decision = self._make_decision(actions, goals, context)
            
            # Step 5: Post-process decision
            final_decision = self._post_process_decision(decision, instruction, context)
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error processing instruction: {e}")
            return self._create_error_decision(str(e))
    
    def _process_context(self, raw_context: Dict[str, Any]) -> GenericContext:
        """Process raw context into structured format"""
        # Determine domain if not specified
        domain = raw_context.get("domain", TaskDomain.GENERIC)
        if isinstance(domain, str):
            domain = TaskDomain(domain)
        
        # Create generic context
        context = GenericContext(
            domain=domain,
            user_id=raw_context.get("user_id"),
            session_id=raw_context.get("session_id"),
            current_focus=raw_context.get("current_focus"),
            recent_activities=raw_context.get("recent_activities", []),
            available_resources=raw_context.get("available_resources", []),
            constraints=raw_context.get("constraints", []),
            user_profile=raw_context.get("user_profile", {}),
            user_preferences=raw_context.get("user_preferences", {}),
            user_capabilities=raw_context.get("user_capabilities", {}),
            user_history=raw_context.get("user_history", []),
            environment=raw_context.get("environment", {}),
            temporal_context=raw_context.get("temporal_context", {}),
            social_context=raw_context.get("social_context", {}),
            domain_data=raw_context.get("domain_data", {}),
            signals=raw_context.get("signals", []),
            feedback_history=raw_context.get("feedback_history", [])
        )
        
        return context
    
    def _infer_goals(self, instruction: str, context: GenericContext) -> List[GenericGoal]:
        """Infer goals from instruction"""
        goals = self.goal_inference.infer_goals(instruction, context, self.config.components)
        
        # Fallback if no goals inferred
        if not goals and self.config.components.fallback_to_simple_mode:
            simple_engine = SimpleGoalInference()
            goals = simple_engine.infer_goals(instruction, context, self.config.components)
        
        return goals
    
    def _generate_actions(self, goals: List[GenericGoal], context: GenericContext) -> List[GenericAction]:
        """Generate actions for goals"""
        actions = self.action_generator.generate_actions(goals, context, self.config.components)
        
        # Limit actions based on configuration
        return actions[:self.config.max_actions]
    
    def _make_decision(self, actions: List[GenericAction], goals: List[GenericGoal], context: GenericContext) -> GenericDecision:
        """Make decision among actions"""
        decision = self.decision_maker.make_decision(actions, goals, context, self.config.components)
        
        # Apply domain-specific validation if available
        domain_config = self.config.domain_configs.get(context.domain)
        if domain_config:
            # Could add domain-specific validation here
            pass
        
        return decision
    
    def _post_process_decision(self, decision: GenericDecision, instruction: str, context: GenericContext) -> GenericDecision:
        """Post-process decision"""
        # Add metadata
        decision.metadata["original_instruction"] = instruction
        decision.metadata["agent_config"] = {
            "tom_enabled": self.config.components.enable_theory_of_mind,
            "dt_enabled": self.config.components.enable_decision_theory,
            "domain": context.domain.value
        }
        
        return decision
    
    def _create_error_decision(self, error_message: str) -> GenericDecision:
        """Create error decision"""
        error_action = GenericAction(
            action_id="error_action",
            description=f"Error occurred: {error_message}",
            domain=TaskDomain.GENERIC,
            intent_type=IntentType.COMMUNICATE,
            parameters={"error": error_message},
            estimated_effort=0.1,
            estimated_success_probability=0.1,
            outcomes=["Report error to user"]
        )
        
        return GenericDecision(
            decision_id=f"error_{int(time.time())}",
            selected_action=error_action,
            alternative_actions=[],
            reasoning=f"Error occurred: {error_message}",
            confidence=0.1,
            expected_utility=0.0,
            expected_outcomes=error_action.outcomes
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        return {
            "agent_id": self.config.agent_id,
            "name": self.config.name,
            "primary_domain": self.config.primary_domain.value,
            "supported_domains": [d.value for d in self.config.supported_domains],
            "theory_of_mind": self.config.components.enable_theory_of_mind,
            "decision_theory": self.config.components.enable_decision_theory,
            "context_awareness": self.config.components.enable_context_awareness,
            "learning": self.config.components.enable_learning,
            "max_goals": self.config.max_goals,
            "max_actions": self.config.max_actions
        }
    
    def update_config(self, new_config: ComponentConfig):
        """Update component configuration"""
        self.config.components = new_config
        self._initialize_components()  # Reinitialize with new config