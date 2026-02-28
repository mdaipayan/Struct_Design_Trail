import streamlit as st
import math
import numpy as np
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Building Frame Analysis & Design", page_icon="🏢", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏢 Building Frame App")
st.sidebar.markdown("Structural conceptualization and IS Code checks for multi-story buildings.")
app_mode = st.sidebar.radio(
    "Select Module:",
    [
        "Home", 
        "1. Preliminary Sizing (RCC)", 
        "2. Lateral Load Generator", 
        "3. 3D Frame Torsion & Plan Check",
        "4. 3D FEA Solver Architecture"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Designed for preliminary architectural planning and structural optimization loops.")

# ==========================================
# MODULE: HOME
# ==========================================
if app_mode == "Home":
    st.title("Building Frame Conceptualization & Analysis Tool")
    st.markdown("""
    Welcome to the multi-story building frame application. 
    
    This tool is designed to bridge the gap between architectural grids and full 3D finite element analysis. Navigate through the modules to establish initial sizes, generate lateral loads per Indian Standards, check floor plan irregularities, and set up a custom 3D Direct Stiffness Method solver.
    """)
    # 

# ==========================================
# MODULE 1: PRELIMINARY SIZING
# ==========================================
elif app_mode == "1. Preliminary Sizing (RCC)":
    st.header("Preliminary Sizing (IS 456:2000)")
    tab1, tab2 = st.tabs(["Slab & Beam Sizing", "Column Sizing (Tributary Area)"])
    
    with tab1:
        st.subheader("Slab & Beam Effective Depth")
        st.caption("Based on deflection control criteria ($L/d$ ratios).")
        col1, col2 = st.columns(2)
        with col1:
            span_m = st.number_input("Clear Span (meters)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)
        with col2:
            support = st.selectbox("Support Condition", ["Cantilever", "Simply Supported", "Continuous"])

        span_mm = span_m * 1000
        ratios = {"Cantilever": 7, "Simply Supported": 20, "Continuous": 26}
        base_ratio = ratios.get(support, 20)
        
        mod_factor = 1.2 # Assumed standard modification factor
        allowable_ratio = base_ratio * mod_factor
        req_d = span_mm / allowable_ratio
        
        overall_D = req_d + 30 
        practical_D = ((overall_D // 25) + 1) * 25 

        st.info(f"**Required Effective Depth ($d$):** {req_d:.2f} mm")
        st.success(f"**Recommended Overall Depth ($D$):** {practical_D:.0f} mm")

    with tab2:
        st.subheader("Column Axial Sizing")
        col1, col2 = st.columns(2)
        with col1:
            trib_area = st.number_input("Tributary Area ($m^2$)", min_value=1.0, value=16.0, step=1.0)
            floors = st.number_input("Number of Floors Supported", min_value=1, value=3, step=1)
            load_sqm = st.number_input("Avg. Load per Floor ($kN/m^2$)", value=12.0, step=1.0)
        with col2:
            fck = st.selectbox("Concrete Grade ($f_{ck}$)", [20, 25, 30, 35, 40], index=1)
            fy = st.selectbox("Steel Grade ($f_y$)", [415, 500, 550], index=1)
            steel_percent = st.slider("Assumed Steel % ($p_t$)", 0.8, 4.0, 1.0, 0.1)
        
        total_load = trib_area * floors * load_sqm
        Pu_kN = 1.5 * total_load
        Pu_N = Pu_kN * 1000
        
        pt = steel_percent / 100
        stress_capacity = (0.4 * fck * (1 - pt)) + (0.67 * fy * pt)
        
        Ag_req = Pu_N / stress_capacity
        side_mm = math.sqrt(Ag_req)
        practical_side = math.ceil(side_mm / 25) * 25
        
        st.info(f"**Factored Axial Load ($P_u$):** {Pu_kN:.2f} kN")
        st.success(f"**Preliminary Square Column Size:** {practical_side} mm $\\times$ {practical_side} mm")

# ==========================================
# MODULE 2: LATERAL LOAD GENERATOR
# ==========================================
elif app_mode == "2. Lateral Load Generator":
    st.header("Lateral Load Calculations")
    # 
    tab1, tab2 = st.tabs(["Seismic Base Shear (IS 1893)", "Wind Pressure (IS 875 Pt 3)"])
    
    with tab1:
        st.subheader("Equivalent Static Method (IS 1893:2016)")
        col1, col2, col3 = st.columns(3)
        with col1:
            zone = st.selectbox("Seismic Zone", ["II", "III", "IV", "V"], index=2)
            zone_factors = {"II": 0.10, "III": 0.16, "IV": 0.24, "V": 0.36}
            Z = zone_factors[zone]
            soil_type = st.selectbox("Soil Type", ["I (Hard)", "II (Medium)", "III (Soft)"], index=1)
        with col2:
            I_factor = st.selectbox("Importance Factor (I)", [1.0, 1.2, 1.5], index=2)
            R_factor = st.selectbox("Response Reduction (R)", [3.0, 4.0, 5.0], index=2)
        with col3:
            h = st.number_input("Building Height, h (m)", min_value=3.0, value=15.0, step=1.0)
            W = st.number_input("Total Seismic Weight, W (kN)", min_value=100.0, value=15000.0, step=500.0)

        frame_type = st.radio("Frame Material", ["RC Frame", "Steel Frame"])
        Ta = 0.075 * (h ** 0.75) if frame_type == "RC Frame" else 0.085 * (h ** 0.75)

        def get_sag(Ta, soil_type):
            if "I" in soil_type: return 2.5 if Ta <= 0.40 else 1.0 / Ta
            elif "II" in soil_type: return 2.5 if Ta <= 0.55 else 1.36 / Ta
            else: return 2.5 if Ta <= 0.67 else 1.67 / Ta

        Sa_g = get_sag(Ta, soil_type)
        Ah = max((Z / 2) * (I_factor / R_factor) * Sa_g, Z / 2)
        Vb = Ah * W

        colA, colB = st.columns(2)
        with colA:
            st.info(f"**Fundamental Period ($T_a$):** {Ta:.3f} s \n\n **Spectral Accel ($S_a/g$):** {Sa_g:.3f}")
        with colB:
            st.success(f"**Seismic Coefficient ($A_h$):** {Ah:.4f} \n\n **Design Base Shear ($V_B$):** {Vb:.2f} kN")

    with tab2:
        st.subheader("Design Wind Pressure (IS 875 Part 3:2015)")
        col1, col2 = st.columns(2)
        with col1:
            Vb_wind = st.number_input("Basic Wind Speed, $V_b$ (m/s)", min_value=33, max_value=55, value=44, step=1)
            k1 = st.number_input("Risk Coefficient ($k_1$)", value=1.0, step=0.01)
            k3 = st.number_input("Topography Factor ($k_3$)", value=1.0, step=0.01)
            k4 = st.number_input("Importance Factor ($k_4$)", value=1.0, step=0.01)
        with col2:
            terrain = st.selectbox("Terrain Category", [1, 2, 3, 4], index=1)
            height_z = st.number_input("Height of Structure, $z$ (m)", min_value=1.0, value=10.0, step=1.0)
        
        # Simplified k2 logic
        k2_values = {1: 1.05, 2: 1.00, 3: 0.91, 4: 0.80}
        k2 = k2_values.get(terrain, 1.0)
        
        Vz = Vb_wind * k1 * k2 * k3 * k4
        Pz_kN = (0.6 * (Vz ** 2)) / 1000

        st.info(f"**Design Wind Speed ($V_z$):** {Vz:.2f} m/s")
        st.success(f"**Design Wind Pressure ($P_z$):** {Pz_kN:.3f} $kN/m^2$")

# ==========================================
# MODULE 3: 3D TORSION CHECK
# ==========================================
elif app_mode == "3. 3D Frame Torsion & Plan Check":
    st.header("Floor Plan Torsion Check")
    st.caption("Calculates Center of Mass (CM), Center of Rigidity (CR), and Design Eccentricity per IS 1893.")
    # 

    col1, col2 = st.columns(2)
    with col1:
        L_x = st.number_input("Plan Dimension in X-direction, $b_x$ (m)", value=20.0, step=1.0)
    with col2:
        L_y = st.number_input("Plan Dimension in Y-direction, $b_y$ (m)", value=15.0, step=1.0)

    st.markdown("**Column Locations & Relative Stiffness**")
    default_data = pd.DataFrame({
        "Element": ["C1", "C2", "C3", "C4"],
        "X (m)": [0.0, 20.0, 0.0, 20.0],
        "Y (m)": [0.0, 0.0, 15.0, 15.0],
        "Stiffness kx": [10000, 15000, 10000, 20000],
        "Stiffness ky": [10000, 15000, 10000, 20000]
    })
    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

    if st.button("Calculate Torsional Eccentricity"):
        CM_x, CM_y = L_x / 2.0, L_y / 2.0
        sum_ky = edited_df["Stiffness ky"].sum()
        sum_kx = edited_df["Stiffness kx"].sum()
        
        CR_x = (edited_df["Stiffness ky"] * edited_df["X (m)"]).sum() / sum_ky
        CR_y = (edited_df["Stiffness kx"] * edited_df["Y (m)"]).sum() / sum_kx

        e_x, e_y = abs(CM_x - CR_x), abs(CM_y - CR_y)
        ed_x_1 = 1.5 * e_x + 0.05 * L_x
        ed_y_1 = 1.5 * e_y + 0.05 * L_y

        colA, colB, colC = st.columns(3)
        with colA:
            st.info(f"**CM:** ({CM_x:.2f}, {CM_y:.2f})")
        with colB:
            st.success(f"**CR:** ({CR_x:.2f}, {CR_y:.2f})")
        with colC:
            st.warning(f"**Static Ecc ($e$):** ({e_x:.2f}, {e_y:.2f})")

        st.error(f"**Design Eccentricity ($e_d$):** X = {ed_x_1:.2f} m | Y = {ed_y_1:.2f} m")

# ==========================================
# MODULE 4: 3D FEA SOLVER ARCHITECTURE
# ==========================================
elif app_mode == "4. 3D FEA Solver Architecture":
    st.header("3D Direct Stiffness Method: OOP Architecture")
    st.markdown("""
    When building a custom Python solver for optimization, relying on arrays becomes unmanageable. 
    Below is the foundational Object-Oriented structure required for a 3D frame solver.
    """)
    # 

    code_snippet = '''
class Material:
    def __init__(self, E, G, nu, rho):
        self.E = E
        self.G = G
        self.nu = nu
        self.rho = rho

class Section:
    def __init__(self, A, Iy, Iz, J):
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J

class Node3D:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x, self.y, self.z = x, y, z
        self.dof = [id*6, id*6+1, id*6+2, id*6+3, id*6+4, id*6+5]
        self.fixed = [False] * 6
        self.loads = [0.0] * 6

    def apply_support(self, fix_array):
        self.fixed = fix_array # e.g., [True, True, True, True, True, True] for fully fixed

class Element3D:
    def __init__(self, id, node_i, node_j, material, section, beta_angle=0.0):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.mat = material
        self.sec = section
        self.beta = beta_angle # Web rotation angle
        
    def get_length(self):
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        return (dx**2 + dy**2 + dz**2)**0.5
        
    def get_transformation_matrix(self):
        # Calculates direction cosines and handles vertical column singularity
        # Returns 12x12 block diagonal transformation matrix [T]
        pass
        
    def get_local_stiffness(self):
        # Assembles the 12x12 matrix using E, G, A, Iy, Iz, J, L
        pass

class Model3D:
    def __init__(self):
        self.nodes = []
        self.elements = []
        
    def assemble_global_stiffness(self):
        # Uses scipy.sparse to assemble massive 1200x1200+ matrices efficiently
        pass
        
    def solve(self):
        # Partitions matrix, applies boundary conditions, and solves U = K^-1 * F
        pass
'''
    st.code(code_snippet, language='python')
    st.caption("Copy this blueprint to begin structuring your custom metaheuristic design loops.")
