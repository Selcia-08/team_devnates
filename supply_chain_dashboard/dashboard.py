import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import AntPath
try:
    import osmnx as ox
except ImportError:
    pass # Handled in load_graph
import networkx as nx
import random
import numpy as np
import sys
import os
# Add project root to path to import app.services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Tuple
from models import Driver, Package
from solver import SimpleNearestNeighbor, ClusterAndRoute, EfficiencyVRP, Solution
# from mock_agent_service import MockAgentService # REMOVED

try:
    from live_monitor import get_latest_solution_sync
except ImportError:
    get_latest_solution_sync = None

import asyncio
import time
from agent_adapter import DashboardAgentAdapter
from api_client import BackendClient, convert_routes_to_solution

st.set_page_config(layout="wide", page_title="Supply Chain Optimizer")

# --- Constants & State ---
if 'drivers' not in st.session_state:
    st.session_state.drivers = []
if 'packages' not in st.session_state:
    st.session_state.packages = []
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'city_name' not in st.session_state:
    st.session_state.city_name = "Chennai, India"
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://localhost:8090"
if 'selected_run_id' not in st.session_state:
    st.session_state.selected_run_id = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = (False, "Not checked")

# --- Helpers ---
@st.cache_resource
def load_graph(city_name: str, network_type='drive', dist=2000):
    """Downloads and caches the OSM graph."""
    try:
        # Check if city_name looks like a place name or coords?
        # Just assume place name for now, but use graph_from_point if we wanted radius from center
        # Ideally, graph_from_address or graph_from_place accepts 'dist' if using from_point logic, 
        # but from_place downloads the whole boundary.
        # To make "radius" work effectively with a city name, we might geocode first, then get graph from point.
        
        try:
             G = ox.graph_from_place(city_name, network_type=network_type)
        except:
             # Fallback or if place is too large/undefined, maybe just try address
             # But for valid places like "Chennai", it downloads the whole thing which is slow.
             # Let's try to Geocode then use graph_from_point for speed if user wants validation
             # For now, sticking to the requested "radius" feature means we should probably use point.
             # But getting a point from a name requires geocoding.
             
             # Let's trust ox.graph_from_address (which geocodes) with a distance
             G = ox.graph_from_address(city_name, dist=dist, network_type=network_type)
             
        # Add edge speeds and travel times
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        return G
    except Exception as e:
        st.error(f"Error loading map: {e}")
        return None

def create_real_solution(drivers, packages, graph=None):
    """Use REAL Agents to create a solution via LangGraph."""
    try:
        # We need to run the async adapter in a sync wrapper for Streamlit
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        solution = loop.run_until_complete(
            DashboardAgentAdapter.run_simulation(drivers, packages)
        )
        loop.close()
        return solution
    except Exception as e:
        st.error(f"Agent Simulation Failed: {e}")
        return Solution()

def generate_random_scenario(graph, n_drivers, n_packages):
    """Generates random drivers and packages on graph nodes."""
    nodes = list(graph.nodes())
    
    # Sample nodes
    driver_nodes = random.sample(nodes, n_drivers)
    package_nodes = random.sample(nodes, n_packages) # Allow overlap if needed, but sample unique for now
    
    drivers = []
    for i, node in enumerate(driver_nodes):
        drivers.append(Driver(
            id=f"D{i+1}",
            lat=graph.nodes[node]['y'],
            lon=graph.nodes[node]['x'],
            node_id=node
        ))
        
    packages = []
    for i, node in enumerate(package_nodes):
        packages.append(Package(
            id=f"P{i+1}",
            lat=graph.nodes[node]['y'],
            lon=graph.nodes[node]['x'],
            node_id=node
        ))
        
    return drivers, packages

# --- UI Setup ---
st.title("🚛 Intelligent Supply Chain Dispatch")
st.markdown("Optimizing route assignments using real-world road networks.")

# --- Backend Connection Status (Top Bar) ---
backend_client = BackendClient(st.session_state.backend_url)
is_connected, status_msg = backend_client.check_health()
st.session_state.connection_status = (is_connected, status_msg)

if is_connected:
    st.success(f"🟢 Backend: {status_msg}")
else:
    st.warning(f"🔴 Backend: {status_msg}")

# --- Sidebar ---
with st.sidebar:
    st.header("0. Backend Connection")
    backend_url_input = st.text_input(
        "Backend URL", 
        value=st.session_state.backend_url,
        help="URL of the FastAPI backend server"
    )
    if backend_url_input != st.session_state.backend_url:
        st.session_state.backend_url = backend_url_input
        st.rerun()
    
    if st.button("🔄 Test Connection"):
        test_client = BackendClient(backend_url_input)
        connected, msg = test_client.check_health()
        if connected:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")
    
    st.divider()
    
    st.header("1. Configuration")
    city_input = st.text_input("City Scope", value=st.session_state.city_name, key="city_scope_input")
    network_type = st.selectbox("Transport Mode", ["drive", "bike", "walk"])
    
    # New: Radius Slider to control map size and speed
    map_radius = st.slider("Map Radius (meters)", 500, 10000, 2000, help="Smaller radius = Faster download/processing.")
    
    if st.button("Load Map Region"):
        with st.spinner(f"Downloading road network for {city_input} ({map_radius}m radius)..."):
            st.session_state.graph = load_graph(city_input, network_type, dist=map_radius)
            st.session_state.city_name = city_input
            if st.session_state.graph:
                st.success(f"Map loaded! Nodes: {len(st.session_state.graph.nodes)}")
            else:
                st.error("Could not load map. Try a specific address or city.")

    st.divider()
    
    st.header("2. Scenario Generation")
    n_drivers = st.slider("Number of Drivers", 1, 20, 5)
    n_packages = st.slider("Number of Packages", 1, 100, 20)
    
    if st.button("Generate Random Scenario"):
        if st.session_state.graph:
            d, p = generate_random_scenario(st.session_state.graph, n_drivers, n_packages)
            st.session_state.drivers = d
            st.session_state.packages = p
            st.session_state.solution = None # Reset previous solution
            st.success(f"Generated {len(d)} drivers and {len(p)} packages.")
        else:
            st.error("Please load the map region first.")

# --- Main Area ---
col1, col2 = st.columns([3, 1])

with col2:
    st.header("3. Control Panel")
    algo_choice = st.selectbox(
        "Optimization Strategy", 
        ["Live Backend Connection", "Simulation (AI Agents)", "Live Monitor (DB Polling)", "Simple Nearest Neighbor", "Cluster & Route (K-Means + TSP)", "Efficiency Optimized (VRP)"]
    )
    
    # Auto-refresh for Live Backend mode
    if algo_choice == "Live Backend Connection":
        refresh_interval = st.slider("Refresh interval (seconds)", 2, 30, 3, key="refresh_interval")
        st.session_state.auto_refresh = st.checkbox("🔄 Auto-sync (watch for new allocations)", value=st.session_state.auto_refresh)
        if st.session_state.last_refresh:
            st.caption(f"Last sync: {st.session_state.last_refresh}")
        if st.session_state.selected_run_id:
            st.caption(f"Current run: {st.session_state.selected_run_id[:8]}...")
    
    # Auto-refresh timer check (runs before button to avoid loop)
    should_auto_sync = False
    if algo_choice == "Live Backend Connection" and st.session_state.auto_refresh and is_connected:
        import datetime
        if 'last_auto_refresh_time' not in st.session_state:
            st.session_state.last_auto_refresh_time = datetime.datetime.now()
        
        elapsed = (datetime.datetime.now() - st.session_state.last_auto_refresh_time).total_seconds()
        if elapsed >= refresh_interval:
            should_auto_sync = True
            st.session_state.last_auto_refresh_time = datetime.datetime.now()
    
    # Manual sync button
    sync_button_clicked = st.button("📡 Sync Now")
    
    if sync_button_clicked or should_auto_sync:
        
        # Handle Live Backend Connection mode
        if algo_choice == "Live Backend Connection":
            if not is_connected:
                st.error("Backend not connected. Check URL and server status.")
            else:
                with st.spinner("Fetching latest allocation from backend..."):
                    # Get latest runs from the backend
                    runs = backend_client.get_latest_runs(limit=5)
                    
                    if runs:
                        latest_run = runs[0]
                        is_new_run = st.session_state.selected_run_id != latest_run.run_id
                        
                        # Fetch routes for the latest run
                        routes = backend_client.get_routes_for_run(latest_run.run_id)
                        
                        if routes:
                            st.session_state.solution = convert_routes_to_solution(routes)
                            st.session_state.selected_run_id = latest_run.run_id
                            st.session_state.last_refresh = time.strftime("%H:%M:%S")
                            
                            if is_new_run:
                                st.success(f"🆕 New allocation detected! Loaded {len(routes)} routes.")
                            else:
                                st.info(f"Synced {len(routes)} routes (run: {latest_run.run_id[:8]}...)")
                        else:
                            st.warning(f"Run {latest_run.run_id[:8]}... has no routes yet (still processing?).")
                    else:
                        st.info("No allocations found. Trigger a 'Run LangGraph Allocation' from the web UI.")

        
        elif not st.session_state.drivers or not st.session_state.graph:
            st.error("No scenario to solve.")
        else:
            with st.spinner("Calculating optimal routes (Running 5 AI Agents)..."):
                solver = None
                if algo_choice == "Simulation (AI Agents)":
                    st.session_state.solution = create_real_solution(
                        st.session_state.drivers,
                        st.session_state.packages,
                        st.session_state.graph
                    )
                elif algo_choice == "Live Monitor (DB Polling)":
                     if get_latest_solution_sync:
                         status_text = st.empty()
                         st.info("Polling database for new allocations every 5s...")
                         
                         if st.button("Check Now"):
                              run_meta, sol = get_latest_solution_sync()
                              if run_meta:
                                   st.session_state.solution = sol
                                   st.success(f"Loaded Run {run_meta.id} from {run_meta.created_at}")
                                   st.rerun()
                              else:
                                   st.warning("No runs found in DB.")
                     else:
                          st.error("Live Monitor service not available (check logs)")
                          
                elif algo_choice == "Simple Nearest Neighbor":
                    solver = SimpleNearestNeighbor()
                elif algo_choice == "Cluster & Route (K-Means + TSP)":
                    solver = ClusterAndRoute()
                else:
                    solver = EfficiencyVRP()
                
                if solver:
                    st.session_state.solution = solver.solve(
                        st.session_state.drivers, 
                        st.session_state.packages, 
                        st.session_state.graph
                    )
    
    st.markdown("### Metrics")
    if st.session_state.solution:
        st.metric("Total Distance", f"{st.session_state.solution.total_distance/1000:.2f} km")
        st.metric("Fairness (Var)", f"{st.session_state.solution.fairness_score/1000000:.2f}")

with col1:
    # Map Visualization
    
    # Determine if we have something to display
    has_graph = st.session_state.graph is not None
    has_solution = st.session_state.solution is not None and len(st.session_state.solution.routes) > 0
    
    # Calculate center for the map
    map_center = None
    
    if has_graph:
        # Use graph nodes for center
        node_sample = list(st.session_state.graph.nodes())[:10]
        avg_x = np.mean([st.session_state.graph.nodes[n]['x'] for n in node_sample])
        avg_y = np.mean([st.session_state.graph.nodes[n]['y'] for n in node_sample])
        map_center = [avg_y, avg_x]
    elif has_solution:
        # Calculate center from solution routes
        all_coords = []
        for path_coords in st.session_state.solution.routes.values():
            if path_coords:
                all_coords.extend(path_coords)
        if all_coords:
            avg_lat = np.mean([c[0] for c in all_coords])
            avg_lon = np.mean([c[1] for c in all_coords])
            map_center = [avg_lat, avg_lon]
    
    # Default to Chennai if nothing else
    if map_center is None:
        map_center = [13.0827, 80.2707]  # Chennai, India
    
    if has_graph or has_solution:
        m = folium.Map(location=map_center, zoom_start=14)
        
        # Plot Drivers (only if we have local scenario)
        for d in st.session_state.drivers:
            folium.Marker(
                [d.lat, d.lon], 
                popup=f"Driver {d.id} ({d.status})", 
                icon=folium.Icon(color="blue", icon="truck", prefix="fa")
            ).add_to(m)
            
        # Plot Packages (only if we have local scenario)
        for p in st.session_state.packages:
            folium.Marker(
                [p.lat, p.lon], 
                popup=f"Package {p.id}", 
                icon=folium.Icon(color="red", icon="box", prefix="fa")
            ).add_to(m)
            
        # Plot Routes
        if st.session_state.solution:
            colors = ['green', 'purple', 'orange', 'darkblue', 'black', 'pink', 'cadetblue', 'darkred', 'lightblue', 'darkgreen']
            for i, (d_id, path_coords) in enumerate(st.session_state.solution.routes.items()):
                if path_coords and len(path_coords) > 0:
                    color = colors[i % len(colors)]
                    
                    # Add start marker for backend routes
                    folium.Marker(
                        location=path_coords[0],
                        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
                        tooltip=f"Start - Route {d_id[:8] if len(d_id) > 8 else d_id}"
                    ).add_to(m)
                    
                    # 1. Background static line
                    folium.PolyLine(
                        path_coords, 
                        color=color, 
                        weight=3, 
                        opacity=0.4
                    ).add_to(m)
                    
                    # 2. AntPath Flow Animation (Simulating Arrows)
                    AntPath(
                        locations=path_coords,
                        color=color,
                        pulse_color='#FFFFFF',
                        weight=6,
                        opacity=0.9,
                        dash_array=[20, 30],
                        delay=800,
                        tooltip=f"Route {d_id[:8] if len(d_id) > 8 else d_id}"
                    ).add_to(m)
                    
                    # 3. Add End Marker
                    folium.Marker(
                        location=path_coords[-1],
                        icon=folium.Icon(color="green", icon="flag-checkered", prefix="fa"),
                        tooltip=f"Destination - Route {d_id[:8] if len(d_id) > 8 else d_id}"
                    ).add_to(m)
                    
                    # Add intermediate stop markers
                    for j, coord in enumerate(path_coords[1:-1], start=1):
                        folium.CircleMarker(
                            location=coord,
                            radius=6,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.7,
                            tooltip=f"Stop {j}"
                        ).add_to(m)

        st_folium(m, width="100%", height=600)
    else:
        st.info("📍 Load a map region OR select 'Live Backend Connection' and sync to view routes.")

# Non-blocking auto-refresh for Live Backend Connection mode
# Only schedules a rerun if enough time has passed and auto-sync is enabled
if algo_choice == "Live Backend Connection" and st.session_state.auto_refresh and is_connected:
    import datetime
    
    # Initialize timer if needed
    if 'auto_refresh_timer' not in st.session_state:
        st.session_state.auto_refresh_timer = datetime.datetime.now()
    
    elapsed = (datetime.datetime.now() - st.session_state.auto_refresh_timer).total_seconds()
    
    # Only rerun if interval has passed (prevents rapid loops)
    if elapsed >= refresh_interval:
        st.session_state.auto_refresh_timer = datetime.datetime.now()
        st.rerun()
