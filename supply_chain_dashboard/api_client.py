"""
Backend API Client for Supply Chain Dashboard.

Provides HTTP-based communication with the FastAPI backend for real-time data.
"""

import requests
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunInfo:
    """Allocation run information."""
    run_id: str
    date: str
    status: str
    num_drivers: int
    num_routes: int
    num_packages: int
    gini_index: float


@dataclass 
class RouteStop:
    """A stop on a route."""
    lat: float
    lng: float
    package_id: str
    stop_order: int
    address: Optional[str] = None


@dataclass
class RouteData:
    """Route data from backend."""
    route_id: str
    driver_id: Optional[str]
    driver_name: Optional[str]
    stops: List[RouteStop]
    total_weight_kg: float
    num_packages: int
    estimated_time_minutes: int


class BackendClient:
    """Client for communicating with the Fair Dispatch Backend API."""
    
    def __init__(self, base_url: str = "http://localhost:8090"):
        """
        Initialize the backend client.
        
        Args:
            base_url: Base URL of the backend API server
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = 10  # seconds
        
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a GET request to the backend."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def _post(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Make a POST request to the backend."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Check if the backend is healthy and connected.
        
        Returns:
            Tuple of (is_connected, status_message)
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return True, f"Connected - {data.get('status', 'OK')}"
            return False, f"Unhealthy: HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to backend"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_latest_runs(self, limit: int = 10) -> List[RunInfo]:
        """
        Get the most recent allocation runs.
        
        Args:
            limit: Maximum number of runs to fetch
            
        Returns:
            List of RunInfo objects
        """
        # Query recent agent events to find run IDs
        data = self._get("/api/v1/agent-events/recent", params={"limit": 200})
        if not data:
            return []
        
        # Extract unique run IDs from events (preserving order - most recent first)
        seen = set()
        run_ids = []
        for event in data.get("events", []):
            run_id = event.get("allocation_run_id")
            if run_id and run_id not in seen:
                seen.add(run_id)
                run_ids.append(run_id)
                if len(run_ids) >= limit:
                    break
        
        # Fetch summary for each run
        runs = []
        for run_id in run_ids:
            summary = self.get_run_summary(run_id)
            if summary:
                runs.append(summary)
        
        return runs
    
    def get_run_summary(self, run_id: str) -> Optional[RunInfo]:
        """
        Get summary for a specific allocation run.
        
        Args:
            run_id: The allocation run UUID
            
        Returns:
            RunInfo object or None if not found
        """
        data = self._get(f"/api/v1/runs/{run_id}/summary")
        if not data:
            return None
        
        return RunInfo(
            run_id=data.get("allocation_run_id", run_id),
            date=data.get("date", ""),
            status=data.get("status", "unknown"),
            num_drivers=data.get("num_drivers", 0),
            num_routes=data.get("num_routes", 0),
            num_packages=data.get("num_packages", 0),
            gini_index=data.get("global_gini_index", 0.0),
        )
    
    def get_routes_for_run(self, run_id: str) -> List[RouteData]:
        """
        Get all routes for an allocation run (for map visualization).
        
        Args:
            run_id: The allocation run UUID
            
        Returns:
            List of RouteData objects
        """
        data = self._get(f"/api/v1/runs/{run_id}/routes-on-map")
        if not data:
            return []
        
        routes = []
        for route_dict in data.get("routes", []):
            stops = [
                RouteStop(
                    lat=s["lat"],
                    lng=s["lng"],
                    package_id=s["package_id"],
                    stop_order=s["stop_order"],
                    address=s.get("address"),
                )
                for s in route_dict.get("stops", [])
            ]
            
            routes.append(RouteData(
                route_id=route_dict.get("route_id", ""),
                driver_id=route_dict.get("driver_id"),
                driver_name=route_dict.get("driver_name"),
                stops=stops,
                total_weight_kg=route_dict.get("total_weight_kg", 0.0),
                num_packages=route_dict.get("num_packages", 0),
                estimated_time_minutes=route_dict.get("estimated_time_minutes", 0),
            ))
        
        return routes
    
    def get_recent_events(self, run_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Get recent agent events.
        
        Args:
            run_id: Optional run ID to filter events
            limit: Maximum number of events
            
        Returns:
            List of event dictionaries
        """
        params = {"limit": limit}
        if run_id:
            params["run_id"] = run_id
        
        data = self._get("/api/v1/agent-events/recent", params=params)
        if not data:
            return []
        
        return data.get("events", [])
    
    def trigger_allocation(
        self,
        drivers: List[Dict],
        packages: List[Dict],
        warehouse_lat: float = 13.0827,
        warehouse_lng: float = 80.2707,
    ) -> Optional[str]:
        """
        Trigger a new allocation via the backend API.
        
        Args:
            drivers: List of driver dicts with id, name, vehicle_capacity_kg
            packages: List of package dicts with id, latitude, longitude, weight_kg
            warehouse_lat: Warehouse latitude
            warehouse_lng: Warehouse longitude
            
        Returns:
            allocation_run_id if successful, None otherwise
        """
        from datetime import date
        
        payload = {
            "allocation_date": date.today().isoformat(),
            "warehouse": {"lat": warehouse_lat, "lng": warehouse_lng},
            "drivers": drivers,
            "packages": packages,
        }
        
        data = self._post("/api/v1/allocate/langgraph", payload)
        if not data:
            return None
        
        return data.get("allocation_run_id")


def convert_routes_to_solution(routes: List[RouteData]) -> Any:
    """
    Convert RouteData list to dashboard Solution format.
    
    Args:
        routes: List of RouteData from the API
        
    Returns:
        Solution object compatible with dashboard visualization
    """
    from solver import Solution
    
    sol = Solution()
    total_dist = 0.0
    
    for route in routes:
        driver_id = route.driver_id or route.route_id
        
        # Build coordinate path from stops
        path_coords = []
        for stop in sorted(route.stops, key=lambda s: s.stop_order):
            path_coords.append((stop.lat, stop.lng))
        
        sol.routes[driver_id] = path_coords
        
        # Estimate distance (rough approximation)
        # In reality, this would use road network distances
        if len(path_coords) >= 2:
            for i in range(len(path_coords) - 1):
                lat1, lon1 = path_coords[i]
                lat2, lon2 = path_coords[i + 1]
                # Haversine approximation (km)
                import math
                R = 6371  # Earth radius in km
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                total_dist += R * c * 1000  # Convert to meters
    
    sol.total_distance = total_dist
    sol.fairness_score = 0.0  # Will be updated with actual fairness data
    
    return sol
