import logging
import os
import re
import sys
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

# Ensure the project root is in the path so absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.services.framework import DealAgentFramework

load_dotenv(override=True)
st.set_page_config(page_title="The Price is Right AI", layout="wide", page_icon="💰")

AGENT_COLORS = {
    "31": "#FF6B6B",  # Agent.RED
    "32": "#51CF66",  # Agent.GREEN
    "33": "#FCC419",  # Agent.YELLOW
    "34": "#4DABF7",  # Agent.BLUE
    "35": "#DA77F2",  # Agent.MAGENTA
    "36": "#20C997",  # Agent.CYAN
    "37": "#FFFFFF",  # Agent.WHITE
}

def ansi_to_html(text: str) -> str:
    """
    Parses BaseAgent ANSI sequences (\033[40m, \033[3Xm, \033[0m) into HTML spans
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def replace_code(match):
        code_str = match.group(1) or ""
        codes = code_str.split(";")
        
        if not codes or "0" in codes or "" in codes:
            return "</span>"

        for c in codes:
            if c in AGENT_COLORS:
                return f'<span style="color: {AGENT_COLORS[c]}; font-weight: bold;">'
        
        return ""

    formatted = re.sub(r'\x1B\[([0-9;]*)m', replace_code, text)
    open_count = formatted.count("<span")
    close_count = formatted.count("</span>")
    if open_count > close_count:
        formatted += "</span>" * (open_count - close_count)
    return formatted

def extract_product_title(deal) -> str:
    """
    Safely retrieves the product identifier across varying schema iterations
    (product_description, title, name, description, product_name).
    """
    if deal is None:
        return "Unknown Product"
    
    # Priority check for common attributes
    candidate_keys = [
        "title",
        "product_description", 
        "name", 
        "description", 
        "product_name"
    ]
    
    for key in candidate_keys:
        val = getattr(deal, key, None)
        if val is None and isinstance(deal, dict):
            val = deal.get(key)
            
        if val and str(val).strip():
            return str(val).strip()
            
    return "Unknown Item"

# --- Custom Log Handler for Streamlit ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_deque):
        super().__init__()
        self.log_deque = log_deque

    def emit(self, record):
        msg = self.format(record)
        styled_html = ansi_to_html(msg) # Strip colors for UI
        self.log_deque.appendleft(styled_html)

@st.cache_resource
def setup_framework_and_logging():
    """Initializes the backend and log capture only once per session."""
    log_deque = deque(maxlen=40) # Keep last 30 logs in memory
    
    handler = StreamlitLogHandler(log_deque)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    framework = DealAgentFramework(
        model_url=os.getenv("PRICER_MODEL_URL", "http://localhost:11434"),
        api_key=os.getenv("GEMINI_API_KEY")
    )
    return framework, log_deque

framework, log_deque = setup_framework_and_logging()

# --- State Management ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Background pipeline worker
def background_scanner():
    while st.session_state.is_running:
        logging.info("Triggering automated pipeline run...")  # noqa: LOG015
        try:
            framework.run()
        except Exception as e: # noqa: BLE001
            logging.error(f"Pipeline error: {e}") # noqa: LOG015

        # Sleep in short 5-second bursts to check for the stop signal
        for _ in range(6): # 6 * 5s = 30 seconds (0.5 mins)
            try:
                if not getattr(st.session_state, "is_running", False):
                    logging.info("Scanner stopped by user.") # noqa: LOG015
                    break
            except StopException:
                return
            time.sleep(5) # Wait 5 sec between full scans

# --- UI Layout ---
st.markdown("<h1 style='text-align: center;'>💰 The Price is Right AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Autonomous Deal Hunting Agent Framework</p>", unsafe_allow_html=True)

# Controls
col1, col2 = st.columns([1, 6])
with col1:
    if st.session_state.is_running:
        if st.button("🛑 Stop Scanner", width='stretch'):
            st.session_state.is_running = False
            st.rerun()
    else:
        if st.button("▶️ Start Scanner", type="primary", width='stretch'):
            st.session_state.is_running = True
            t = threading.Thread(target=background_scanner, daemon=True)
            add_script_run_ctx(t)
            t.start()
            st.rerun()

st.divider()

# Main Display: 3D Plot & Logs
plot_col, log_col = st.columns([2, 1])

with plot_col:
    st.subheader("Vector Space Topology")
    try:
        documents, vectors, colors = framework.get_plot_data(max_datapoints=800)

        vectors = np.asarray(vectors)
        if vectors.size == 0:
            st.info("Vector database is empty. Run the scanner to generate embeddings.")
        elif vectors.ndim != 2 or vectors.shape[1] != 3:
            st.error(f"Invalid t-SNE output shape: {vectors.shape}. Expected (n_samples, 3).")
        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter3d(
                    x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
                    mode="markers",
                    marker={"size": 4, "color": colors, "opacity": 0.85},
                    text=documents, hovertemplate="%{text}<extra></extra>",
                )
            )

            fig.update_layout(
                margin={"l": 0, "r": 0, "b": 0, "t": 0},
                height=500,
                scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
            )
            
            st.plotly_chart(
                fig, 
                width='stretch', 
                config={"displayModeBar": True, "scrollZoom": True,},
            )

    except Exception as e: # noqa: BLE001
        st.error(f"3D visualization error: {type(e).__name__}: {e}")
        st.exception(e)          
    
with log_col:
    st.subheader("Agent Activity Log")
    log_content = "<br>".join(list(log_deque)) if log_deque else "<span style='color: #6c757d;'>Waiting for agent logs...</span>"
    log_html = (
        f"<div style='height: 400px; overflow-y: auto; font-family: monospace; "
        f"font-size: 12px; background-color: #0E1117; color: #E6EDF3; padding: 12px; "
        f"border-radius: 6px; line-height: 1.5; border: 1px solid #30363D;'>"
        f"{log_content}"
        f"</div>"
    )
    st.markdown(log_html, unsafe_allow_html=True)

# Data Table & Interactions
st.subheader("Discovered Opportunities")
opportunities = framework.read_memory()

if opportunities:
    opp_records = []
    for opp in opportunities:
        deal_obj = getattr(opp, "deal", None)
        product_name = extract_product_title(deal_obj)
        price = getattr(deal_obj, "price", 0.0) if deal_obj else 0.0
        url = getattr(deal_obj, "url", "") if deal_obj else ""
        estimate = getattr(opp, "estimate", 0.0)
        discount = getattr(opp, "discount", 0.0)
        
        opp_records.append({
            "_opp_obj": opp,
            "Product": product_name,
            "Price ($)": float(price),
            "Estimate ($)": float(estimate),
            "Discount ($)": float(discount),
            "URL": str(url)
        })

    # Sort opportunities by highest discount
    opp_records.sort(key=lambda x: x["Discount ($)"], reverse=True)

    # DataFrame for display without internal object references
    display_df = pd.DataFrame([
        {k: v for k, v in item.items() if k != "_opp_obj"} 
        for item in opp_records
    ])
    
    st.markdown("**Select a row to dispatch a manual push notification.**")
    event = st.dataframe(
        display_df, 
        width='stretch', 
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_opp = opp_records[selected_idx]["_opp_obj"]
        selected_title = opp_records[selected_idx]["Product"]
        
        short_title = (selected_title[:45] + "...") if len(selected_title) > 45 else selected_title

        if st.button(f"📲 Send Alert: {short_title}", type="primary"):
            framework.init_agents_as_needed()
            framework.planner.messenger.alert(selected_opp)
            st.success("Push notification dispatched via ntfy!")
else:
    st.info("No active opportunities found. Start the scanner to hunt for deals.")

# Auto-refresh loop to keep logs updating when running
if st.session_state.is_running:
    time.sleep(2)
    st.rerun()