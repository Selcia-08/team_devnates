"""
LangGraph node wrappers for Fair Dispatch agents.
Each node wraps an existing agent with minimal changes, preserving the original logic.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import asyncio

from app.schemas.allocation_state import AllocationState
from app.schemas.agent_schemas import (
    FairnessThresholds,
    DriverAssignmentProposal,
    DriverContext,
)
from app.services.ml_effort_agent import MLEffortAgent
from app.services.route_planner_agent import RoutePlannerAgent
from app.services.fairness_manager_agent import FairnessManagerAgent
from app.services.driver_liaison_agent import DriverLiaisonAgent
from app.services.final_resolution import FinalResolutionAgent
from app.services.explainability import ExplainabilityAgent
from app.schemas.explainability import DriverExplanationInput
from app.services.fairness import calculate_fairness_score
from app.core.events import agent_event_bus, make_agent_event


class ModelWrapper:
    """Helper to wrap dicts as objects for agent compatibility."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)
        
    @property
    def is_ev(self) -> bool:
        return self._data.get("vehicle_type") == "EV"


def _publish_event_sync(
    allocation_run_id: Optional[str],
    agent_name: str,
    step_type: str,
    state: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Publish an agent event synchronously (fire-and-forget).
    Used by LangGraph nodes which are synchronous functions.
    """
    if not allocation_run_id:
        return
    
    event = make_agent_event(
        allocation_run_id=allocation_run_id,
        agent_name=agent_name,
        step_type=step_type,
        state=state,
        payload=payload,
    )
    
    # Schedule async publish - get or create event loop
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(agent_event_bus.publish(event))
    except RuntimeError:
        # No running loop, create one for this publish
        asyncio.run(agent_event_bus.publish(event))


def _create_decision_log(
    agent_name: str,
    step_type: str,
    input_snapshot: Dict[str, Any],
    output_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a decision log entry compatible with DecisionLog model."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "agent_name": agent_name,
        "step_type": step_type,
        "input_snapshot": input_snapshot,
        "output_snapshot": output_snapshot,
    }


# =============================================================================
# Node 1: ML Effort Agent
# =============================================================================

def ml_effort_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #1: ML Effort Agent.
    
    Computes effort matrix for all driver-route pairs using MLEffortAgent.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    
    # Publish STARTED event
    _publish_event_sync(run_id, "ML_EFFORT", "MATRIX_GENERATION", "STARTED", {
        "num_drivers": len(state.driver_models),
        "num_routes": len(state.route_models),
    })
    
    # Initialize agent
    ml_agent = MLEffortAgent()
    
    # Get EV config from state
    ev_config = {
        "safety_margin_pct": state.config_used.get("ev_safety_margin_pct", 10.0) if state.config_used else 10.0,
        "charging_penalty_weight": state.config_used.get("ev_charging_penalty_weight", 0.3) if state.config_used else 0.3,
    }
    
    # Wrap dicts as objects for agent compatibility
    drivers = [ModelWrapper(d) for d in state.driver_models]
    routes = [ModelWrapper(r) for r in state.route_models]
    
    # Compute effort matrix (EXISTING CODE - UNCHANGED)
    effort_result = ml_agent.compute_effort_matrix(
        drivers=drivers,
        routes=routes,
        ev_config=ev_config,
    )
    
    # Serialize result for state
    effort_dict = {
        "matrix": effort_result.matrix,
        "driver_ids": effort_result.driver_ids,
        "route_ids": effort_result.route_ids,
        "breakdown": {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                     for k, v in effort_result.breakdown.items()},
        "stats": effort_result.stats,
        "infeasible_pairs": list(effort_result.infeasible_pairs) if effort_result.infeasible_pairs else [],
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="ML_EFFORT",
        step_type="MATRIX_GENERATION",
        input_snapshot=ml_agent.get_input_snapshot(drivers, routes),
        output_snapshot={
            **ml_agent.get_output_snapshot(effort_result),
            "num_infeasible_ev_pairs": len(effort_result.infeasible_pairs) if effort_result.infeasible_pairs else 0,
        },
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "ML_EFFORT", "MATRIX_GENERATION", "COMPLETED", {
        "min_effort": effort_result.stats.get("min", 0),
        "max_effort": effort_result.stats.get("max", 0),
        "avg_effort": effort_result.stats.get("avg", 0),
    })
    
    return {
        "effort_matrix": effort_dict,
        "decision_logs": state.decision_logs + [log_entry],
    }


# =============================================================================
# Node 2: Route Planner Agent (Proposal 1)
# =============================================================================

def route_planner_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #2: Route Planner Agent - Proposal 1.
    
    Generates optimal driver-route assignment using OR-Tools.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    
    # Publish STARTED event
    _publish_event_sync(run_id, "ROUTE_PLANNER", "PROPOSAL_1", "STARTED", {
        "num_drivers": len(state.driver_models),
        "num_routes": len(state.route_models),
    })
    
    planner_agent = RoutePlannerAgent()
    
    # Reconstruct EffortMatrixResult-like object for planner
    from app.schemas.agent_schemas import EffortMatrixResult, EffortBreakdown
    
    # Use stats from serialized state or compute if not available
    matrix = state.effort_matrix["matrix"]
    stats = state.effort_matrix.get("stats")
    if not stats:
        all_values = [v for row in matrix for v in row if v < float('inf')]
        stats = {
            "min": min(all_values) if all_values else 0.0,
            "max": max(all_values) if all_values else 0.0,
            "avg": sum(all_values) / len(all_values) if all_values else 0.0,
        }
    
    effort_result = EffortMatrixResult(
        matrix=matrix,
        driver_ids=state.effort_matrix["driver_ids"],
        route_ids=state.effort_matrix["route_ids"],
        breakdown={},  # Simplified - full breakdown not needed for planning
        stats=stats,
        infeasible_pairs=list(state.effort_matrix.get("infeasible_pairs", [])),
    )
    
    # Get recovery penalty weight
    recovery_penalty_weight = state.config_used.get("recovery_penalty_weight", 3.0) if state.config_used else 3.0
    
    # Wrap dicts as objects for agent compatibility
    drivers = [ModelWrapper(d) for d in state.driver_models]
    routes = [ModelWrapper(r) for r in state.route_models]
    
    # Generate Proposal 1 (EXISTING CODE - UNCHANGED)
    proposal1 = planner_agent.plan(
        effort_result=effort_result,
        drivers=drivers,
        routes=routes,
        recovery_targets=state.recovery_targets or {},
        recovery_penalty_weight=recovery_penalty_weight,
        proposal_number=1,
    )
    
    # Serialize result
    proposal_dict = {
        "allocation": [a.model_dump() if hasattr(a, 'model_dump') else a for a in proposal1.allocation],
        "total_effort": proposal1.total_effort,
        "avg_effort": proposal1.avg_effort,
        "solver_status": proposal1.solver_status,
        "proposal_number": proposal1.proposal_number,
        "per_driver_effort": proposal1.per_driver_effort,
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="ROUTE_PLANNER",
        step_type="PROPOSAL_1",
        input_snapshot=planner_agent.get_input_snapshot(effort_result),
        output_snapshot=planner_agent.get_output_snapshot(proposal1),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "ROUTE_PLANNER", "PROPOSAL_1", "COMPLETED", {
        "total_effort": proposal1.total_effort,
        "num_assignments": len(proposal1.allocation),
        "solver_status": proposal1.solver_status,
    })
    
    return {
        "route_proposal_1": proposal_dict,
        "decision_logs": state.decision_logs + [log_entry],
    }


# =============================================================================
# Node 3: Fairness Check Agent
# =============================================================================

def fairness_check_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #3: Fairness Manager Agent.
    
    Evaluates fairness metrics and decides ACCEPT or REOPTIMIZE.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    proposal_number = 2 if state.route_proposal_2 else 1
    
    # Publish STARTED event
    _publish_event_sync(run_id, "FAIRNESS_MANAGER", f"FAIRNESS_CHECK_{proposal_number}", "STARTED", {
        "proposal_number": proposal_number,
    })
    
    # Get thresholds from config
    thresholds = FairnessThresholds(
        gini_threshold=state.config_used.get("gini_threshold", 0.33) if state.config_used else 0.33,
        stddev_threshold=state.config_used.get("stddev_threshold", 25.0) if state.config_used else 25.0,
        max_gap_threshold=state.config_used.get("max_gap_threshold", 25.0) if state.config_used else 25.0,
    )
    
    fairness_agent = FairnessManagerAgent(thresholds=thresholds)
    
    # Reconstruct RoutePlanResult for fairness check
    from app.schemas.agent_schemas import RoutePlanResult, AllocationItem
    
    # Determine which proposal to check
    proposal_to_check = state.route_proposal_2 or state.route_proposal_1
    
    plan_result = RoutePlanResult(
        allocation=[AllocationItem(**a) for a in proposal_to_check["allocation"]],
        total_effort=proposal_to_check["total_effort"],
        avg_effort=proposal_to_check.get("avg_effort", proposal_to_check["total_effort"] / len(proposal_to_check["allocation"]) if proposal_to_check["allocation"] else 0.0),
        solver_status=proposal_to_check.get("solver_status", "OPTIMAL"),
        proposal_number=proposal_number,
        per_driver_effort=proposal_to_check["per_driver_effort"],
    )
    
    # Check fairness (EXISTING CODE - UNCHANGED)
    fairness_result = fairness_agent.check(plan_result, proposal_number=proposal_number)
    
    # Serialize result
    fairness_dict = {
        "status": fairness_result.status,
        "proposal_number": fairness_result.proposal_number,
        "metrics": fairness_result.metrics.model_dump() if hasattr(fairness_result.metrics, 'model_dump') else {
            "avg_effort": fairness_result.metrics.avg_effort,
            "std_dev": fairness_result.metrics.std_dev,
            "gini_index": fairness_result.metrics.gini_index,
            "max_effort": fairness_result.metrics.max_effort,
            "min_effort": fairness_result.metrics.min_effort,
            "max_gap": fairness_result.metrics.max_gap,
        },
        "recommendations": fairness_result.recommendations.model_dump() if fairness_result.recommendations and hasattr(fairness_result.recommendations, 'model_dump') else None,
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="FAIRNESS_MANAGER",
        step_type=f"FAIRNESS_CHECK_PROPOSAL_{proposal_number}",
        input_snapshot=fairness_agent.get_input_snapshot(plan_result),
        output_snapshot=fairness_agent.get_output_snapshot(fairness_result),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "FAIRNESS_MANAGER", f"FAIRNESS_CHECK_{proposal_number}", "COMPLETED", {
        "status": fairness_result.status,
        "gini_index": fairness_dict["metrics"]["gini_index"],
        "std_dev": fairness_dict["metrics"]["std_dev"],
    })
    
    # Update appropriate check result based on proposal number
    updates = {
        "decision_logs": state.decision_logs + [log_entry],
    }
    
    if proposal_number == 1:
        updates["fairness_check_1"] = fairness_dict
    else:
        updates["fairness_check_2"] = fairness_dict
    
    return updates


def fairness_check_2_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node for second fairness check (dedicated wrapper).
    
    This is a separate function to avoid LangGraph state key conflicts
    when the same function is used for multiple nodes.
    """
    run_id = state.allocation_run_id
    proposal_number = 2  # Always proposal 2 for this node
    
    # Publish STARTED event
    _publish_event_sync(run_id, "FAIRNESS_MANAGER", f"FAIRNESS_CHECK_{proposal_number}", "STARTED", {
        "proposal_number": proposal_number,
    })
    
    # Get thresholds from config
    thresholds = FairnessThresholds(
        gini_threshold=state.config_used.get("gini_threshold", 0.33) if state.config_used else 0.33,
        stddev_threshold=state.config_used.get("stddev_threshold", 25.0) if state.config_used else 25.0,
        max_gap_threshold=state.config_used.get("max_gap_threshold", 25.0) if state.config_used else 25.0,
    )
    
    fairness_agent = FairnessManagerAgent(thresholds=thresholds)
    
    # Reconstruct RoutePlanResult for fairness check
    from app.schemas.agent_schemas import RoutePlanResult, AllocationItem
    
    # Always check proposal 2 for this node
    proposal_to_check = state.route_proposal_2
    
    plan_result = RoutePlanResult(
        allocation=[AllocationItem(**a) for a in proposal_to_check["allocation"]],
        total_effort=proposal_to_check["total_effort"],
        avg_effort=proposal_to_check.get("avg_effort", proposal_to_check["total_effort"] / len(proposal_to_check["allocation"]) if proposal_to_check["allocation"] else 0.0),
        solver_status=proposal_to_check.get("solver_status", "OPTIMAL"),
        proposal_number=proposal_number,
        per_driver_effort=proposal_to_check["per_driver_effort"],
    )
    
    # Check fairness
    fairness_result = fairness_agent.check(plan_result, proposal_number=proposal_number)
    
    # Serialize result
    fairness_dict = {
        "status": fairness_result.status,
        "proposal_number": fairness_result.proposal_number,
        "metrics": fairness_result.metrics.model_dump() if hasattr(fairness_result.metrics, 'model_dump') else {
            "avg_effort": fairness_result.metrics.avg_effort,
            "std_dev": fairness_result.metrics.std_dev,
            "gini_index": fairness_result.metrics.gini_index,
            "max_effort": fairness_result.metrics.max_effort,
            "min_effort": fairness_result.metrics.min_effort,
            "max_gap": fairness_result.metrics.max_gap,
        },
        "recommendations": fairness_result.recommendations.model_dump() if fairness_result.recommendations and hasattr(fairness_result.recommendations, 'model_dump') else None,
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="FAIRNESS_MANAGER",
        step_type=f"FAIRNESS_CHECK_PROPOSAL_{proposal_number}",
        input_snapshot=fairness_agent.get_input_snapshot(plan_result),
        output_snapshot=fairness_agent.get_output_snapshot(fairness_result),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "FAIRNESS_MANAGER", f"FAIRNESS_CHECK_{proposal_number}", "COMPLETED", {
        "status": fairness_result.status,
        "gini_index": fairness_dict["metrics"]["gini_index"],
        "std_dev": fairness_dict["metrics"]["std_dev"],
    })
    
    return {
        "fairness_check_2": fairness_dict,
        "decision_logs": state.decision_logs + [log_entry],
    }


# =============================================================================
# Node 4: Route Planner Agent (Proposal 2 - with fairness penalties)
# =============================================================================


def route_planner_reoptimize_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #4: Route Planner Agent - Proposal 2 (re-optimization).
    
    Re-runs OR-Tools with fairness penalties applied.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    
    # Publish STARTED event
    _publish_event_sync(run_id, "ROUTE_PLANNER", "PROPOSAL_2", "STARTED", {
        "reason": "fairness_reoptimization",
    })
    
    planner_agent = RoutePlannerAgent()
    
    # Reconstruct effort result
    from app.schemas.agent_schemas import EffortMatrixResult, FairnessRecommendations
    
    # Use stats from serialized state or compute if not available
    matrix = state.effort_matrix["matrix"]
    stats = state.effort_matrix.get("stats")
    if not stats:
        all_values = [v for row in matrix for v in row if v < float('inf')]
        stats = {
            "min": min(all_values) if all_values else 0.0,
            "max": max(all_values) if all_values else 0.0,
            "avg": sum(all_values) / len(all_values) if all_values else 0.0,
        }
    
    effort_result = EffortMatrixResult(
        matrix=matrix,
        driver_ids=state.effort_matrix["driver_ids"],
        route_ids=state.effort_matrix["route_ids"],
        breakdown={},
        stats=stats,
        infeasible_pairs=list(state.effort_matrix.get("infeasible_pairs", [])),
    )
    
    # Build penalties from recommendations
    recommendations_dict = state.fairness_check_1.get("recommendations")
    penalties = {}
    
    if recommendations_dict:
        recommendations = FairnessRecommendations(**recommendations_dict)
        penalties = planner_agent.build_penalties_from_recommendations(
            recommendations,
            state.route_proposal_1["per_driver_effort"],
        )
    
    # Get recovery settings
    recovery_penalty_weight = state.config_used.get("recovery_penalty_weight", 3.0) if state.config_used else 3.0
    
    # Wrap dicts as objects for agent compatibility
    drivers = [ModelWrapper(d) for d in state.driver_models]
    routes = [ModelWrapper(r) for r in state.route_models]
    
    # Generate Proposal 2 (EXISTING CODE - UNCHANGED)
    proposal2 = planner_agent.plan(
        effort_result=effort_result,
        drivers=drivers,
        routes=routes,
        fairness_penalties=penalties,
        recovery_targets=state.recovery_targets or {},
        recovery_penalty_weight=recovery_penalty_weight,
        proposal_number=2,
    )
    
    # Serialize result
    proposal_dict = {
        "allocation": [a.model_dump() if hasattr(a, 'model_dump') else a for a in proposal2.allocation],
        "total_effort": proposal2.total_effort,
        "avg_effort": proposal2.avg_effort,
        "solver_status": proposal2.solver_status,
        "proposal_number": proposal2.proposal_number,
        "per_driver_effort": proposal2.per_driver_effort,
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="ROUTE_PLANNER",
        step_type="PROPOSAL_2",
        input_snapshot=planner_agent.get_input_snapshot(effort_result, penalties),
        output_snapshot=planner_agent.get_output_snapshot(proposal2),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "ROUTE_PLANNER", "PROPOSAL_2", "COMPLETED", {
        "total_effort": proposal2.total_effort,
        "num_assignments": len(proposal2.allocation),
        "solver_status": proposal2.solver_status,
    })
    
    return {
        "route_proposal_2": proposal_dict,
        "decision_logs": state.decision_logs + [log_entry],
    }



# =============================================================================
# Node 5: Select Final Proposal
# =============================================================================

def select_final_proposal_node(state: AllocationState) -> Dict[str, Any]:
    """
    Select the final proposal after fairness checks.
    
    If proposal 2 exists and has better fairness, use it.
    Otherwise, use proposal 1.
    """
    final_proposal = state.route_proposal_1
    final_fairness = state.fairness_check_1
    
    if state.route_proposal_2 and state.fairness_check_2:
        # Compare fairness metrics
        check1_metrics = state.fairness_check_1["metrics"]
        check2_metrics = state.fairness_check_2["metrics"]
        
        # Use proposal 2 if it improves fairness
        if (check2_metrics["gini_index"] <= check1_metrics["gini_index"] or
            check2_metrics["max_gap"] < check1_metrics["max_gap"]):
            final_proposal = state.route_proposal_2
            final_fairness = state.fairness_check_2
    
    return {
        "final_proposal": final_proposal,
        "final_fairness": final_fairness,
        "final_per_driver_effort": final_proposal["per_driver_effort"],
    }


# =============================================================================
# Node 6: Driver Liaison Agent
# =============================================================================

def driver_liaison_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #6: Driver Liaison Agent.
    
    Reviews proposed assignments and makes ACCEPT/COUNTER decisions per driver.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    
    # Publish STARTED event
    _publish_event_sync(run_id, "DRIVER_LIAISON", "NEGOTIATION", "STARTED", {
        "num_drivers": len(state.driver_models),
    })
    
    from app.schemas.agent_schemas import AllocationItem
    
    liaison_agent = DriverLiaisonAgent()
    
    final_proposal = state.final_proposal or state.route_proposal_1
    final_fairness = state.final_fairness or state.fairness_check_1
    
    # Build DriverAssignmentProposals with ranking
    sorted_allocations = sorted(
        final_proposal["allocation"],
        key=lambda x: x["effort"],
        reverse=True  # Highest effort = rank 1
    )
    
    driver_proposals: List[DriverAssignmentProposal] = []
    for rank, alloc_item in enumerate(sorted_allocations, start=1):
        driver_proposals.append(DriverAssignmentProposal(
            driver_id=str(alloc_item["driver_id"]),
            route_id=str(alloc_item["route_id"]),
            effort=alloc_item["effort"],
            rank_in_team=rank,
        ))
    
    # Get global metrics
    metrics = final_fairness["metrics"]
    global_avg_effort = metrics["avg_effort"]
    global_std_effort = metrics["std_dev"]
    
    # Build DriverContext objects
    driver_context_objs: Dict[str, DriverContext] = {}
    for driver_id, context_dict in state.driver_contexts.items():
        driver_context_objs[driver_id] = DriverContext(**context_dict)
    
    # Run liaison for all drivers (EXISTING CODE - UNCHANGED)
    negotiation_result = liaison_agent.run_for_all_drivers(
        proposals=driver_proposals,
        driver_contexts=driver_context_objs,
        effort_matrix=state.effort_matrix["matrix"],
        driver_ids=state.effort_matrix["driver_ids"],
        route_ids=state.effort_matrix["route_ids"],
        global_avg_effort=global_avg_effort,
        global_std_effort=global_std_effort,
    )
    
    # Serialize result
    liaison_dict = {
        "decisions": [d.model_dump() if hasattr(d, 'model_dump') else d for d in negotiation_result.decisions],
        "num_accept": negotiation_result.num_accept,
        "num_counter": negotiation_result.num_counter,
        "num_force_accept": negotiation_result.num_force_accept,
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="DRIVER_LIAISON",
        step_type="NEGOTIATION_DECISIONS",
        input_snapshot=liaison_agent.get_input_snapshot(
            driver_proposals,
            global_avg_effort,
            global_std_effort,
        ),
        output_snapshot=liaison_agent.get_output_snapshot(negotiation_result),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "DRIVER_LIAISON", "NEGOTIATION", "COMPLETED", {
        "num_accept": negotiation_result.num_accept,
        "num_counter": negotiation_result.num_counter,
        "num_force_accept": negotiation_result.num_force_accept,
    })
    
    return {
        "liaison_feedback": liaison_dict,
        "decision_logs": state.decision_logs + [log_entry],
    }


# =============================================================================
# Node 7: Final Resolution Agent
# =============================================================================

def final_resolution_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #7: Final Resolution Agent.
    
    Resolves COUNTER decisions through swaps.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    from app.schemas.agent_schemas import RoutePlanResult, AllocationItem, FairnessMetrics, DriverLiaisonDecision
    
    # Check if there are any COUNTER decisions
    counter_decisions = [
        d for d in state.liaison_feedback["decisions"]
        if d["decision"] == "COUNTER"
    ]
    
    if not counter_decisions:
        # Publish SKIPPED event
        _publish_event_sync(run_id, "FINAL_RESOLUTION", "SWAP_RESOLUTION", "COMPLETED", {
            "reason": "no_counters",
            "swaps_applied": 0,
        })
        # No resolution needed
        return {
            "resolution_result": {"swaps_applied": []},
        }
    
    # Publish STARTED event
    _publish_event_sync(run_id, "FINAL_RESOLUTION", "SWAP_RESOLUTION", "STARTED", {
        "num_counters": len(counter_decisions),
    })
    
    resolution_agent = FinalResolutionAgent()
    
    # Reconstruct objects for resolution
    final_proposal = state.final_proposal or state.route_proposal_1
    final_fairness = state.final_fairness or state.fairness_check_1
    
    approved_proposal = RoutePlanResult(
        allocation=[AllocationItem(**a) for a in final_proposal["allocation"]],
        total_effort=final_proposal["total_effort"],
        avg_effort=final_proposal.get("avg_effort", final_proposal["total_effort"] / len(final_proposal["allocation"]) if final_proposal["allocation"] else 0.0),
        solver_status=final_proposal.get("solver_status", "OPTIMAL"),
        proposal_number=final_proposal["proposal_number"],
        per_driver_effort=final_proposal["per_driver_effort"],
    )
    
    decisions = [DriverLiaisonDecision(**d) for d in state.liaison_feedback["decisions"]]
    
    current_metrics = FairnessMetrics(**final_fairness["metrics"])
    
    # Resolve counters (EXISTING CODE - UNCHANGED)
    resolution_result = resolution_agent.resolve_counters(
        approved_proposal=approved_proposal,
        decisions=decisions,
        effort_matrix=state.effort_matrix["matrix"],
        driver_ids=state.effort_matrix["driver_ids"],
        route_ids=state.effort_matrix["route_ids"],
        current_metrics=current_metrics,
    )
    
    # Serialize result
    resolution_dict = {
        "swaps_applied": [s.model_dump() if hasattr(s, 'model_dump') else s for s in resolution_result.swaps_applied],
        "allocation": resolution_result.allocation,  # Already a list of dicts
        "per_driver_effort": resolution_result.per_driver_effort,
        "metrics": resolution_result.metrics,
    }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="ROUTE_PLANNER",
        step_type="FINAL_RESOLUTION",
        input_snapshot=resolution_agent.get_input_snapshot(
            len(counter_decisions),
            current_metrics,
            final_fairness["metrics"]["avg_effort"],
        ),
        output_snapshot=resolution_agent.get_output_snapshot(resolution_result),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "FINAL_RESOLUTION", "SWAP_RESOLUTION", "COMPLETED", {
        "swaps_applied": len(resolution_result.swaps_applied),
    })
    
    # Update per-driver effort if swaps were applied
    updated_effort = state.final_per_driver_effort.copy()
    if resolution_result.swaps_applied:
        updated_effort = resolution_result.per_driver_effort
    
    return {
        "resolution_result": resolution_dict,
        "final_per_driver_effort": updated_effort,
        "decision_logs": state.decision_logs + [log_entry],
    }


# =============================================================================
# Node 8: Explainability Agent
# =============================================================================

def explainability_node(state: AllocationState) -> Dict[str, Any]:
    """
    LangGraph node #8: Explainability Agent.
    
    Generates template-based explanations for each driver.
    WRAPS EXISTING AGENT - no logic changes.
    """
    run_id = state.allocation_run_id
    
    # Publish STARTED event
    _publish_event_sync(run_id, "EXPLAINABILITY", "EXPLANATIONS", "STARTED", {
        "num_drivers": len(state.driver_models),
    })
    
    explain_agent = ExplainabilityAgent()
    
    final_proposal = state.final_proposal or state.route_proposal_1
    final_fairness = state.final_fairness or state.fairness_check_1
    final_per_driver_effort = state.final_per_driver_effort or final_proposal["per_driver_effort"]
    
    metrics = final_fairness["metrics"]
    avg_effort = metrics["avg_effort"]
    
    # Build lookup structures
    route_by_id = {str(r["id"]): r for r in state.route_models}
    driver_by_id = {str(d["id"]): d for d in state.driver_models}
    route_dict_by_id = {str(r["id"]): rd for r, rd in zip(state.route_models, state.route_dicts)}
    
    # Compute per-driver ranks
    sorted_efforts = sorted(
        final_per_driver_effort.items(),
        key=lambda x: x[1],
        reverse=True
    )
    rank_by_driver = {did: idx + 1 for idx, (did, _) in enumerate(sorted_efforts)}
    num_drivers = len(final_per_driver_effort)
    
    # Build liaison decisions lookup
    liaison_by_driver = {}
    if state.liaison_feedback:
        for decision in state.liaison_feedback["decisions"]:
            liaison_by_driver[decision["driver_id"]] = decision
    
    # Build swaps lookup
    swapped_drivers = set()
    if state.resolution_result and state.resolution_result.get("swaps_applied"):
        for swap in state.resolution_result["swaps_applied"]:
            swapped_drivers.add(swap["driver_a"])
            swapped_drivers.add(swap["driver_b"])
    
    explanations: Dict[str, Dict[str, Any]] = {}
    category_counts: Dict[str, int] = {}
    
    for alloc_item in final_proposal["allocation"]:
        driver_id_str = str(alloc_item["driver_id"])
        route_id_str = str(alloc_item["route_id"])
        
        driver = driver_by_id.get(driver_id_str, {})
        route = route_by_id.get(route_id_str, {})
        route_dict = route_dict_by_id.get(route_id_str, {})
        
        # Use resolved effort if available
        effort = final_per_driver_effort.get(driver_id_str, alloc_item["effort"])
        fairness_score = calculate_fairness_score(effort, avg_effort)
        
        # Get driver context
        driver_context = state.driver_contexts.get(driver_id_str, {})
        history_efforts = [driver_context.get("recent_avg_effort", avg_effort)] if driver_context else []
        history_hard_days = driver_context.get("recent_hard_days", 0) if driver_context else 0
        
        # Get effort breakdown
        breakdown_key = f"{driver_id_str}:{route_id_str}"
        effort_breakdown_data = state.effort_matrix.get("breakdown", {}).get(breakdown_key, {})
        effort_breakdown = {
            "physical_effort": effort_breakdown_data.get("physical_effort", 0),
            "route_complexity": effort_breakdown_data.get("route_complexity", 0),
            "time_pressure": effort_breakdown_data.get("time_pressure", 0),
        }
        
        # Get liaison decision
        liaison_decision = liaison_by_driver.get(driver_id_str)
        
        # Determine if recovery day
        is_recovery = (
            history_hard_days >= 3 and
            effort < avg_effort * 0.85
        )
        
        # Build explanation input
        explain_input = DriverExplanationInput(
            driver_id=driver_id_str,
            driver_name=driver.get("name", "Driver"),
            num_drivers=num_drivers,
            today_effort=effort,
            today_rank=rank_by_driver.get(driver_id_str, num_drivers),
            route_id=route_id_str,
            route_summary={
                "num_packages": route.get("num_packages", 0),
                "total_weight_kg": route.get("total_weight_kg", 0),
                "num_stops": route.get("num_stops", 0),
                "difficulty_score": route.get("route_difficulty_score", 0),
                "estimated_time_minutes": route.get("estimated_time_minutes", 0),
            },
            effort_breakdown=effort_breakdown,
            global_avg_effort=avg_effort,
            global_std_effort=metrics["std_dev"],
            global_gini_index=metrics["gini_index"],
            global_max_gap=metrics["max_gap"],
            history_efforts_last_7_days=history_efforts,
            history_hard_days_last_7=history_hard_days,
            is_recovery_day=is_recovery,
            had_manual_override=False,  # TODO: Query DB if needed
            liaison_decision=liaison_decision["decision"] if liaison_decision else None,
            swap_applied=driver_id_str in swapped_drivers,
        )
        
        # Generate explanations (EXISTING CODE - UNCHANGED)
        explain_output = explain_agent.build_explanation_for_driver(explain_input)
        
        # Track category counts
        category_counts[explain_output.category] = category_counts.get(explain_output.category, 0) + 1
        
        explanations[driver_id_str] = {
            "driver_explanation": explain_output.driver_explanation,
            "admin_explanation": explain_output.admin_explanation,
            "category": explain_output.category,
        }
    
    # Create decision log
    log_entry = _create_decision_log(
        agent_name="EXPLAINABILITY",
        step_type="EXPLANATIONS_GENERATED",
        input_snapshot=explain_agent.get_input_snapshot(
            num_drivers=num_drivers,
            avg_effort=avg_effort,
            std_effort=metrics["std_dev"],
            gini_index=metrics["gini_index"],
            category_counts=category_counts,
        ),
        output_snapshot=explain_agent.get_output_snapshot(
            total_explanations=len(explanations),
            category_counts=category_counts,
        ),
    )
    
    # Publish COMPLETED event
    _publish_event_sync(run_id, "EXPLAINABILITY", "EXPLANATIONS", "COMPLETED", {
        "total_explanations": len(explanations),
        "categories": category_counts,
    })
    
    return {
        "explanations": explanations,
        "decision_logs": state.decision_logs + [log_entry],
    }


# =============================================================================
# Conditional Edge Functions
# =============================================================================

def should_reoptimize(state: AllocationState) -> str:
    """
    Conditional edge: decide if re-optimization is needed.
    
    Returns:
        "reoptimize" - if fairness check 1 says REOPTIMIZE and no proposal 2 yet
        "continue" - otherwise
    """
    if state.fairness_check_1 and state.fairness_check_1.get("status") == "REOPTIMIZE":
        if not state.route_proposal_2:
            return "reoptimize"
    return "continue"


def has_counter_decisions(state: AllocationState) -> str:
    """
    Conditional edge: check if any COUNTER decisions need resolution.
    
    Returns:
        "resolve" - if there are COUNTER decisions
        "skip" - otherwise
    """
    if state.liaison_feedback:
        counter_count = sum(
            1 for d in state.liaison_feedback["decisions"]
            if d["decision"] == "COUNTER"
        )
        if counter_count > 0:
            return "resolve"
    return "skip"
