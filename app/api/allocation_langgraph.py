"""
LangGraph-enabled Allocation API endpoint.
Wraps the existing allocation logic with LangGraph orchestration.
"""

import os
import statistics
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Driver, Package, Route, RoutePackage, Assignment
from app.models.driver import PreferredLanguage, VehicleType
from app.models.package import PackagePriority
from app.models.allocation_run import AllocationRun, AllocationRunStatus
from app.models.decision_log import DecisionLog
from app.models.driver import DriverStatsDaily, DriverFeedback
from app.models.fairness_config import FairnessConfig
from app.schemas.allocation import (
    AllocationRequest,
    AllocationResponse,
    AssignmentResponse,
    GlobalFairness,
    RouteSummary,
)
from app.services.clustering import cluster_packages, order_stops_by_nearest_neighbor, haversine_distance
from app.services.workload import calculate_workload, calculate_route_difficulty, estimate_route_time
from app.services.fairness import calculate_fairness_score
from app.services.learning_agent import LearningAgent, hash_config
from app.schemas.allocation_state import AllocationState
from app.services.langgraph_workflow import invoke_allocation_workflow

router = APIRouter(prefix="/allocate", tags=["Allocation"])


@router.post(
    "/langgraph",
    response_model=AllocationResponse,
    status_code=status.HTTP_200_OK,
    summary="Allocate packages to drivers (LangGraph)",
    description="""
    LangGraph-enabled allocation endpoint using multi-agent workflow:
    1. ML Effort Agent builds effort matrix
    2. Route Planner Agent generates optimal assignment
    3. Fairness Manager evaluates; may trigger re-optimization
    4. Driver Liaison Agent negotiates per-driver
    5. Final Resolution resolves counter-proposals
    6. Explainability Agent generates explanations
    7. (Optional) Gemini 1.5 Flash for personalized explanations
    
    Uses LangGraph StateGraph for orchestration with LangSmith tracing.
    """,
)
async def allocate_langgraph(
    request: AllocationRequest,
    db: AsyncSession = Depends(get_db),
    enable_gemini: bool = Query(False, description="Enable Gemini 1.5 Flash explanations"),
) -> AllocationResponse:
    """Perform fair route allocation using LangGraph workflow."""
    
    # Validate input
    if not request.packages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 package is required",
        )
    if not request.drivers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 driver is required",
        )
    
    allocation_date = request.allocation_date
    
    # ========== START ALLOCATION RUN ==========
    allocation_run = AllocationRun(
        date=allocation_date,
        num_drivers=len(request.drivers),
        num_packages=len(request.packages),
        num_routes=0,
        status=AllocationRunStatus.PENDING,
        started_at=datetime.utcnow(),
    )
    db.add(allocation_run)
    await db.flush()
    
    try:
        # ========== PHASE 0: UPSERT DATA & CLUSTERING ==========
        # (Same as original - this is DB-dependent and must stay in endpoint)
        
        # Step 1: Upsert drivers
        driver_map = {}
        driver_models: List[Driver] = []
        
        for driver_input in request.drivers:
            result = await db.execute(
                select(Driver).where(Driver.external_id == driver_input.id)
            )
            driver = result.scalar_one_or_none()
            
            if driver:
                driver.name = driver_input.name
                driver.vehicle_capacity_kg = driver_input.vehicle_capacity_kg
                driver.preferred_language = PreferredLanguage(driver_input.preferred_language)
            else:
                driver = Driver(
                    external_id=driver_input.id,
                    name=driver_input.name,
                    vehicle_capacity_kg=driver_input.vehicle_capacity_kg,
                    preferred_language=PreferredLanguage(driver_input.preferred_language),
                    vehicle_type=VehicleType.ICE,
                )
                db.add(driver)
            
            driver_map[driver_input.id] = driver
        
        await db.flush()
        driver_models = list(driver_map.values())
        
        # Step 2: Upsert packages
        package_map = {}
        package_dicts = []
        
        for pkg_input in request.packages:
            result = await db.execute(
                select(Package).where(Package.external_id == pkg_input.id)
            )
            package = result.scalar_one_or_none()
            
            if package:
                package.weight_kg = pkg_input.weight_kg
                package.fragility_level = pkg_input.fragility_level
                package.address = pkg_input.address
                package.latitude = pkg_input.latitude
                package.longitude = pkg_input.longitude
                package.priority = PackagePriority(pkg_input.priority)
            else:
                package = Package(
                    external_id=pkg_input.id,
                    weight_kg=pkg_input.weight_kg,
                    fragility_level=pkg_input.fragility_level,
                    address=pkg_input.address,
                    latitude=pkg_input.latitude,
                    longitude=pkg_input.longitude,
                    priority=PackagePriority(pkg_input.priority),
                )
                db.add(package)
            
            package_map[pkg_input.id] = package
            package_dicts.append({
                "external_id": pkg_input.id,
                "weight_kg": pkg_input.weight_kg,
                "fragility_level": pkg_input.fragility_level,
                "address": pkg_input.address,
                "latitude": pkg_input.latitude,
                "longitude": pkg_input.longitude,
                "priority": pkg_input.priority,
            })
        
        await db.flush()
        
        # Step 3: Cluster packages into routes
        clusters = cluster_packages(
            packages=package_dicts,
            num_drivers=len(request.drivers),
        )
        
        # Step 4: Create routes
        route_models: List[Route] = []
        route_dicts = []
        
        for cluster in clusters:
            ordered_packages = order_stops_by_nearest_neighbor(
                cluster.packages,
                request.warehouse.lat,
                request.warehouse.lng,
            )
            
            # Calculate total distance
            total_dist = 0.0
            curr_lat, curr_lng = request.warehouse.lat, request.warehouse.lng
            
            for p in ordered_packages:
                dist = haversine_distance(curr_lat, curr_lng, p["latitude"], p["longitude"])
                total_dist += dist
                curr_lat, curr_lng = p["latitude"], p["longitude"]
            
            total_dist += haversine_distance(curr_lat, curr_lng, request.warehouse.lat, request.warehouse.lng)
            
            avg_fragility = sum(p["fragility_level"] for p in cluster.packages) / max(len(cluster.packages), 1)
            
            difficulty = calculate_route_difficulty(
                total_weight_kg=cluster.total_weight_kg,
                num_stops=cluster.num_stops,
                avg_fragility=avg_fragility,
            )
            
            est_time = estimate_route_time(
                num_packages=cluster.num_packages,
                num_stops=cluster.num_stops,
            )
            
            route = Route(
                date=allocation_date,
                cluster_id=cluster.cluster_id,
                total_weight_kg=cluster.total_weight_kg,
                num_packages=cluster.num_packages,
                num_stops=cluster.num_stops,
                route_difficulty_score=difficulty,
                estimated_time_minutes=est_time,
                total_distance_km=total_dist,
                allocation_run_id=allocation_run.id,
            )
            db.add(route)
            route_models.append(route)
            
            workload = calculate_workload({
                "num_packages": cluster.num_packages,
                "total_weight_kg": cluster.total_weight_kg,
                "route_difficulty_score": difficulty,
                "estimated_time_minutes": est_time,
            })
            
            route_dicts.append({
                "cluster_id": cluster.cluster_id,
                "num_packages": cluster.num_packages,
                "total_weight_kg": cluster.total_weight_kg,
                "num_stops": cluster.num_stops,
                "route_difficulty_score": difficulty,
                "estimated_time_minutes": est_time,
                "workload_score": workload,
                "packages": ordered_packages,
            })
        
        await db.flush()
        
        allocation_run.num_routes = len(route_models)
        
        # Create RoutePackage associations
        for i, route in enumerate(route_models):
            for stop_order, pkg_data in enumerate(route_dicts[i]["packages"]):
                package = package_map[pkg_data["external_id"]]
                route_package = RoutePackage(
                    route_id=route.id,
                    package_id=package.id,
                    stop_order=stop_order + 1,
                )
                db.add(route_package)
        
        # ========== GET CONFIG ==========
        config_result = await db.execute(
            select(FairnessConfig).where(FairnessConfig.is_active == True).limit(1)
        )
        active_config = config_result.scalar_one_or_none()
        
        config_used = {}
        if active_config:
            config_used = {
                "gini_threshold": active_config.gini_threshold,
                "stddev_threshold": active_config.stddev_threshold,
                "max_gap_threshold": active_config.max_gap_threshold,
                "ev_safety_margin_pct": active_config.ev_safety_margin_pct,
                "ev_charging_penalty_weight": active_config.ev_charging_penalty_weight,
                "recovery_penalty_weight": active_config.recovery_penalty_weight,
                "recovery_lightening_factor": active_config.recovery_lightening_factor,
            }
        
        # ========== GET RECOVERY TARGETS ==========
        from app.services.recovery_service import get_driver_recovery_targets
        
        driver_ids = [d.id for d in driver_models]
        recovery_targets = await get_driver_recovery_targets(
            db, driver_ids, allocation_date, active_config
        )
        recovery_targets_str = {str(k): v for k, v in recovery_targets.items()}
        
        # ========== BUILD DRIVER CONTEXTS ==========
        driver_contexts: Dict[str, dict] = {}
        cutoff_date = allocation_date - timedelta(days=7)
        
        for driver in driver_models:
            driver_id_str = str(driver.id)
            
            stats_result = await db.execute(
                select(DriverStatsDaily)
                .where(DriverStatsDaily.driver_id == driver.id)
                .where(DriverStatsDaily.date >= cutoff_date)
                .order_by(DriverStatsDaily.date.desc())
            )
            recent_stats = stats_result.scalars().all()
            
            if recent_stats:
                recent_efforts = [s.avg_workload_score for s in recent_stats if s.avg_workload_score]
                if recent_efforts:
                    recent_avg = statistics.mean(recent_efforts)
                    recent_std = statistics.stdev(recent_efforts) if len(recent_efforts) > 1 else 0.0
                else:
                    recent_avg = 60.0
                    recent_std = 15.0
                
                hard_threshold = recent_avg + recent_std
                hard_days = sum(1 for e in recent_efforts if e > hard_threshold)
            else:
                recent_avg = 60.0
                recent_std = 15.0
                hard_days = 0
            
            feedback_result = await db.execute(
                select(DriverFeedback)
                .where(DriverFeedback.driver_id == driver.id)
                .order_by(DriverFeedback.created_at.desc())
                .limit(1)
            )
            recent_feedback = feedback_result.scalar_one_or_none()
            fatigue_score = float(recent_feedback.tiredness_level) if recent_feedback else 3.0
            fatigue_score = max(1.0, min(5.0, fatigue_score))
            
            driver_contexts[driver_id_str] = {
                "driver_id": driver_id_str,
                "recent_avg_effort": recent_avg,
                "recent_std_effort": recent_std,
                "recent_hard_days": hard_days,
                "fatigue_score": fatigue_score,
                "preferences": {},
            }
        
        # ========== SERIALIZE MODELS FOR LANGGRAPH ==========
        driver_model_dicts = []
        for d in driver_models:
            driver_model_dicts.append({
                "id": str(d.id),
                "external_id": d.external_id,
                "name": d.name,
                "vehicle_capacity_kg": d.vehicle_capacity_kg,
                "preferred_language": d.preferred_language.value if hasattr(d.preferred_language, 'value') else d.preferred_language,
                "vehicle_type": d.vehicle_type.value if hasattr(d.vehicle_type, 'value') else str(d.vehicle_type),
                "battery_range_km": getattr(d, 'battery_range_km', None),
                "charging_time_minutes": getattr(d, 'charging_time_minutes', None),
                "is_ev": d.vehicle_type.value == "EV" if hasattr(d.vehicle_type, 'value') else str(d.vehicle_type) == "EV",
                "experience_years": getattr(d, 'experience_years', 2),
            })
        
        route_model_dicts = []
        for r in route_models:
            route_model_dicts.append({
                "id": str(r.id),
                "date": str(r.date),
                "cluster_id": r.cluster_id,
                "total_weight_kg": r.total_weight_kg,
                "num_packages": r.num_packages,
                "num_stops": r.num_stops,
                "route_difficulty_score": r.route_difficulty_score,
                "estimated_time_minutes": r.estimated_time_minutes,
                "total_distance_km": r.total_distance_km,
            })
        
        # Add route IDs to route_dicts
        for i, rd in enumerate(route_dicts):
            rd["id"] = str(route_models[i].id)
        
        # ========== INVOKE LANGGRAPH WORKFLOW ==========
        if enable_gemini:
            os.environ["ENABLE_GEMINI_EXPLAIN"] = "true"
        
        workflow_result = await invoke_allocation_workflow(
            request_dict=request.model_dump(mode="json"),
            config_used=config_used,
            driver_models=driver_model_dicts,
            route_models=route_model_dicts,
            route_dicts=route_dicts,
            driver_contexts=driver_contexts,
            recovery_targets=recovery_targets_str,
            allocation_run_id=str(allocation_run.id),
            thread_id=str(allocation_run.id),
        )
        
        # ========== PERSIST DECISION LOGS ==========
        for log_entry in workflow_result.decision_logs:
            decision_log = DecisionLog(
                allocation_run_id=allocation_run.id,
                agent_name=log_entry["agent_name"],
                step_type=log_entry["step_type"],
                input_snapshot=log_entry.get("input_snapshot", {}),
                output_snapshot=log_entry.get("output_snapshot", {}),
            )
            db.add(decision_log)
        
        # ========== CREATE ASSIGNMENTS ==========
        final_proposal = workflow_result.final_proposal or workflow_result.route_proposal_1
        final_fairness = workflow_result.final_fairness or workflow_result.fairness_check_1
        final_per_driver_effort = workflow_result.final_per_driver_effort or final_proposal["per_driver_effort"]
        
        driver_by_id = {str(d.id): d for d in driver_models}
        route_by_id = {str(r.id): r for r in route_models}
        
        assignments_response = []
        
        for alloc_item in final_proposal["allocation"]:
            driver_id_str = str(alloc_item["driver_id"])
            route_id_str = str(alloc_item["route_id"])
            
            driver = driver_by_id.get(driver_id_str)
            route = route_by_id.get(route_id_str)
            
            if not driver or not route:
                continue
            
            effort = final_per_driver_effort.get(driver_id_str, alloc_item["effort"])
            avg_effort = final_fairness["metrics"]["avg_effort"]
            fairness_score = calculate_fairness_score(effort, avg_effort)
            
            explanation_data = workflow_result.explanations.get(driver_id_str, {})
            driver_explanation = explanation_data.get("driver_explanation", "Route assigned.")
            admin_explanation = explanation_data.get("admin_explanation", "")
            
            assignment = Assignment(
                date=allocation_date,
                driver_id=driver.id,
                route_id=route.id,
                workload_score=effort,
                fairness_score=fairness_score,
                explanation=driver_explanation,
                driver_explanation=driver_explanation,
                admin_explanation=admin_explanation,
                allocation_run_id=allocation_run.id,
            )
            db.add(assignment)
            
            assignments_response.append(AssignmentResponse(
                driver_id=driver.id,
                driver_external_id=driver.external_id,
                driver_name=driver.name,
                route_id=route.id,
                workload_score=effort,
                fairness_score=fairness_score,
                route_summary=RouteSummary(
                    num_packages=route.num_packages,
                    total_weight_kg=route.total_weight_kg,
                    num_stops=route.num_stops,
                    route_difficulty_score=route.route_difficulty_score,
                    estimated_time_minutes=route.estimated_time_minutes,
                ),
                explanation=driver_explanation,
            ))
        
        # ========== UPDATE DAILY STATS ==========
        from app.services.recovery_service import update_daily_stats_for_run
        
        await update_daily_stats_for_run(
            db=db,
            allocation_run_id=allocation_run.id,
            target_date=allocation_date,
            config=active_config,
        )
        
        # ========== CREATE LEARNING EPISODE ==========
        try:
            learning_agent = LearningAgent(db)
            
            import random
            is_experimental = random.random() < 0.10
            
            await learning_agent.create_episode(
                allocation_run_id=allocation_run.id,
                fairness_config=config_used,
                num_drivers=len(driver_models),
                num_routes=len(route_models),
                is_experimental=is_experimental,
            )
        except Exception as learning_error:
            import logging
            logging.warning(f"Failed to create learning episode: {learning_error}")
        
        # ========== FINALIZE ==========
        metrics = final_fairness["metrics"]
        allocation_run.global_gini_index = metrics["gini_index"]
        allocation_run.global_std_dev = metrics["std_dev"]
        allocation_run.global_max_gap = metrics["max_gap"]
        allocation_run.status = AllocationRunStatus.SUCCESS
        allocation_run.finished_at = datetime.utcnow()
        
        await db.commit()
        
        return AllocationResponse(
            allocation_run_id=allocation_run.id,
            allocation_date=allocation_date,
            global_fairness=GlobalFairness(
                avg_workload=metrics["avg_effort"],
                std_dev=metrics["std_dev"],
                gini_index=metrics["gini_index"],
            ),
            assignments=assignments_response,
        )
        
    except Exception as e:
        allocation_run.status = AllocationRunStatus.FAILED
        allocation_run.error_message = str(e)[:500]
        allocation_run.finished_at = datetime.utcnow()
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "LangGraph allocation failed",
                "run_id": str(allocation_run.id),
                "error": str(e)[:200],
            },
        )
