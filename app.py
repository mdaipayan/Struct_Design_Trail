import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import pandas as pd
import json

# --- PAGE SETUP ---
st.set_page_config(page_title="IS Code Compliant 3D Frame Designer", layout="wide")
st.title("🏢 3D Building Frame Analysis & IS-Code Auto-Design")
st.caption("Strict Compliance: IS 456 (Slenderness, Min Eccentricity, Full Beam Detailing), IS 875, IS 1893")

# --- STATE INITIALIZATION ---
if 'init_done' not in st.session_state:
    st.session_state.floors_df = pd.DataFrame({"Floor": [1, 2, 3], "Height (m)": [3.2, 3.0, 3.0]})
    st.session_state.x_grids_df = pd.DataFrame({"Grid_ID": ["A", "B", "C", "D", "E", "F"], "X_Coord (m)": [0.0, 0.115, 4.112, 4.331, 8.039, 9.449]})
    st.session_state.y_grids_df = pd.DataFrame({"Grid_ID": ["1", "2", "3", "4", "5", "6", "7"], "Y_Coord (m)": [0.0, 2.630, 4.999, 8.343, 9.660, 13.220, 14.326]})
    st.session_state.cols_df = pd.DataFrame({
        "Col_ID": ["C1", "C2", "C3", "C4", "C5"],
        "X_Grid": ["A", "B", "C", "D", "E"], "Y_Grid": ["1", "2", "3", "4", "5"],
        "X_Offset (m)": [0.0]*5, "Y_Offset (m)": [0.0]*5, "Angle (deg)": [0, 90, 90, 0, 90]
    })
    st.session_state.params = {
        "col_dim": "230x450", "beam_dim": "230x400", "fck": 25.0, "fy": 500.0, 
        "sbc": 200.0, "live_load": 2.0, "floor_finish": 1.5, "wall_thickness": 230,
        "slab_thickness": 150, "lateral_coeff": 0.025
    }
    st.session_state.loaded_file = None
    st.session_state.init_done = True

# --- SIDEBAR: PROJECT IMPORT ---
st.sidebar.header("📂 Load Project")
uploaded_file = st.sidebar.file_uploader("Upload Project JSON", type=["json"])
if uploaded_file is not None and st.session_state.loaded_file != uploaded_file.name:
    try:
        data = json.load(uploaded_file)
        st.session_state.floors_df = pd.DataFrame(data.get("floors", []))
        st.session_state.x_grids_df = pd.DataFrame(data.get("x_grids", []))
        st.session_state.y_grids_df = pd.DataFrame(data.get("y_grids", []))
        st.session_state.cols_df = pd.DataFrame(data.get("columns", []))
        
        saved_params = data.get("parameters", {})
        for k, v in saved_params.items():
            st.session_state.params[k] = v
            
        st.session_state.loaded_file = uploaded_file.name
        st.rerun() 
    except Exception as e:
        st.sidebar.error(f"Invalid JSON file: {e}")

# --- SIDEBAR: PARAMETRIC INPUTS ---
st.sidebar.header("1. Floor Elevations")
floor_data = st.sidebar.data_editor(st.session_state.floors_df, num_rows="dynamic", use_container_width=True)

z_elevations = {0: 0.0}
current_z = 0.0
for idx, row in floor_data.iterrows():
    current_z += float(row['Height (m)'])
    z_elevations[int(row['Floor'])] = current_z
num_stories = len(floor_data)

st.sidebar.header("2. Structural Grids (From Plan)")
with st.sidebar.expander("Define X-Grids", expanded=True):
    x_grid_data = st.data_editor(st.session_state.x_grids_df, num_rows="dynamic", use_container_width=True, key="x_grids")

with st.sidebar.expander("Define Y-Grids", expanded=True):
    y_grid_data = st.data_editor(st.session_state.y_grids_df, num_rows="dynamic", use_container_width=True, key="y_grids")

st.sidebar.header("3. Column Placement")
x_map = {str(row['Grid_ID']).strip(): float(row['X_Coord (m)']) for _, row in x_grid_data.iterrows() if pd.notna(row['Grid_ID'])}
y_map = {str(row['Grid_ID']).strip(): float(row['Y_Coord (m)']) for _, row in y_grid_data.iterrows() if pd.notna(row['Grid_ID'])}

with st.sidebar.expander("Column Locations & Orientations", expanded=True):
    col_data = st.data_editor(st.session_state.cols_df, num_rows="dynamic", use_container_width=True)

st.sidebar.header("4. IS Code Design & Load Parameters")
col_dim = st.sidebar.text_input("Init Column Size (mm)", str(st.session_state.params["col_dim"]))
beam_dim = st.sidebar.text_input("Init Beam Size (mm)", str(st.session_state.params["beam_dim"]))

col3, col4 = st.sidebar.columns(2)
fck = col3.number_input("fck (MPa)", value=float(st.session_state.params["fck"]), step=5.0)
fy = col4.number_input("fy (MPa)", value=float(st.session_state.params["fy"]), step=85.0)
sbc = col3.number_input("SBC (kN/m²)", value=float(st.session_state.params["sbc"]), step=10.0)

st.sidebar.subheader("IS 875 Applied Loads")
live_load = st.sidebar.number_input("Live Load (kN/m²)", value=float(st.session_state.params["live_load"]))
floor_finish = st.sidebar.number_input("Floor Finish (kN/m²)", value=float(st.session_state.params["floor_finish"]))
wall_thickness = st.sidebar.number_input("Exterior Wall Thk (mm)", value=int(st.session_state.params["wall_thickness"]))
slab_thickness = st.sidebar.number_input("Slab Thickness (mm)", value=int(st.session_state.params["slab_thickness"]))
lateral_coeff_input = st.sidebar.slider("Seismic Base Shear Ah (%)", 0.0, 20.0, float(st.session_state.params["lateral_coeff"] * 100.0))
lateral_coeff = lateral_coeff_input / 100.0

st.sidebar.header("5. AI Optimization")
auto_optimize = st.sidebar.checkbox("Enable IS 456 Safe Auto-Sizing", value=True)
allow_ai_restructure = st.sidebar.checkbox("Allow AI Restructuring (3-Stage Recovery)", value=True)

def safe_float(val, default=0.0):
    try:
        if pd.isna(val) or val is None or str(val).strip() == "": return default
        return float(val)
    except (ValueError, TypeError): return default

# --- GEOMETRY GENERATOR FUNCTION ---
def build_geometry(primary_pts, secondary_pts, z_dict, n_stories, c_dim, b_dim):
    gen_nodes = []
    gen_elements = []
    nid = 0
    
    for floor_idx in range(n_stories + 1):
        z_val = z_dict.get(floor_idx, 0.0)
        for pt in primary_pts:
            gen_nodes.append({'id': nid, 'x': pt['x'], 'y': pt['y'], 'z': z_val, 'floor': floor_idx, 'angle': pt.get('angle', 0), 'is_primary': True})
            nid += 1
            
    for floor_idx in range(1, n_stories + 1):
        z_val = z_dict.get(floor_idx, 0.0)
        for pt in secondary_pts:
            if pt['floor'] == floor_idx:
                gen_nodes.append({'id': nid, 'x': pt['x'], 'y': pt['y'], 'z': z_val, 'floor': floor_idx, 'angle': 0, 'is_primary': False})
                nid += 1

    eid = 0
    for z in range(n_stories):
        bottom_nodes = [n for n in gen_nodes if n['floor'] == z and n.get('is_primary', True)]
        top_nodes = [n for n in gen_nodes if n['floor'] == z + 1 and n.get('is_primary', True)]
        for bn in bottom_nodes:
            tn = next((n for n in top_nodes if abs(n['x'] - bn['x']) < 0.01 and abs(n['y'] - bn['y']) < 0.01), None)
            if tn:
                gen_elements.append({'id': eid, 'ni': bn['id'], 'nj': tn['id'], 'type': 'Column', 'floor': z, 'size': c_dim, 'angle': bn['angle']})
                eid += 1
                
    tolerance = 0.05 
    for z in range(1, n_stories + 1):
        floor_nodes = [n for n in gen_nodes if n['floor'] == z]
        y_groups = {}
        for n in floor_nodes:
            matched = False
            for y_key in y_groups.keys():
                if abs(n['y'] - y_key) <= tolerance:
                    y_groups[y_key].append(n); matched = True; break
            if not matched: y_groups[n['y']] = [n]
                
        for y_key, group in y_groups.items():
            group = sorted(group, key=lambda k: k['x'])
            for i in range(len(group)-1):
                gen_elements.append({'id': eid, 'ni': group[i]['id'], 'nj': group[i+1]['id'], 'type': 'Beam', 'floor': z, 'size': b_dim, 'angle': 0})
                eid += 1
                
        x_groups = {}
        for n in floor_nodes:
            matched = False
            for x_key in x_groups.keys():
                if abs(n['x'] - x_key) <= tolerance:
                    x_groups[x_key].append(n); matched = True; break
            if not matched: x_groups[n['x']] = [n]
                
        for x_key, group in x_groups.items():
            group = sorted(group, key=lambda k: k['y'])
            for i in range(len(group)-1):
                gen_elements.append({'id': eid, 'ni': group[i]['id'], 'nj': group[i+1]['id'], 'type': 'Beam', 'floor': z, 'size': b_dim, 'angle': 0})
                eid += 1
                
    return gen_nodes, gen_elements

primary_xy = []
for idx, row in col_data.iterrows():
    xg = str(row.get('X_Grid', '')).strip()
    yg = str(row.get('Y_Grid', '')).strip()
    if xg in x_map and yg in y_map:
        calc_x = x_map[xg] + safe_float(row.get('X_Offset (m)'))
        calc_y = y_map[yg] + safe_float(row.get('Y_Offset (m)'))
        primary_xy.append({'x': calc_x, 'y': calc_y, 'angle': safe_float(row.get('Angle (deg)'))})

secondary_xy = [] 
nodes, elements = build_geometry(primary_xy, secondary_xy, z_elevations, num_stories, col_dim, beam_dim)

# --- ANALYSIS ENGINE ---
def get_transformation_matrix(ni, nj):
    dx, dy, dz = nj['x'] - ni['x'], nj['y'] - ni['y'], nj['z'] - ni['z']
    L = math.sqrt(dx**2 + dy**2 + dz**2)
    cx, cy, cz = dx/L, dy/L, dz/L
    lam = np.zeros((3, 3))
    if abs(cx) < 1e-6 and abs(cy) < 1e-6: lam = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) if cz > 0 else np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    else: 
        D = math.sqrt(cx**2 + cy**2)
        lam = np.array([[cx, cy, cz], [-cx*cz/D, -cy*cz/D, D], [-cy/D, cx/D, 0]])
    T = np.zeros((12, 12))
    for i in range(4): T[i*3:(i+1)*3, i*3:(i+1)*3] = lam
    return T

def get_local_stiffness(E, G, A, Iy, Iz, J, L):
    k = np.zeros((12, 12))
    k[0,0]=k[6,6]= E*A/L; k[0,6]=k[6,0]= -E*A/L
    k[3,3]=k[9,9]= G*J/L; k[3,9]=k[9,3]= -G*J/L
    k[2,2]=k[8,8]= 12*E*Iy/L**3; k[2,8]=k[8,2]= -12*E*Iy/L**3
    k[4,4]=k[10,10]= 4*E*Iy/L; k[4,10]=k[10,4]= 2*E*Iy/L
    k[2,4]=k[2,10]=k[4,2]=k[10,2]= -6*E*Iy/L**2; k[8,4]=k[8,10]=k[4,8]=k[10,8]= 6*E*Iy/L**2
    k[1,1]=k[7,7]= 12*E*Iz/L**3; k[1,7]=k[7,1]= -12*E*Iz/L**3
    k[5,5]=k[11,11]= 4*E*Iz/L; k[5,11]=k[11,5]= 2*E*Iz/L
    k[1,5]=k[1,11]=k[5,1]=k[11,1]= 6*E*Iz/L**2; k[7,5]=k[7,11]=k[5,7]=k[11,7]= -6*E*Iz/L**2
    return k

def run_analysis_dynamic(current_elements, current_nodes, optimized_slab_D):
    num_nodes = len(current_nodes)
    if num_nodes == 0: return current_elements, np.zeros(0)
    
    F_global = np.zeros(num_nodes * 6)
    E_conc = 5000 * math.sqrt(fck) * 1e3 
    G_conc = E_conc / 2.4 
    
    X_coords, Y_coords = [n['x'] for n in current_nodes], [n['y'] for n in current_nodes]
    floor_area = (max(X_coords) - min(X_coords)) * (max(Y_coords) - min(Y_coords)) * 0.85 if X_coords else 0
    
    total_dl_per_m2 = ((optimized_slab_D / 1000.0) * 25.0) + floor_finish
    total_floor_dl = floor_area * total_dl_per_m2
    total_floor_ll = floor_area * live_load
    
    total_beam_len = sum([math.sqrt((next(n for n in current_nodes if n['id'] == el['nj'])['x']-next(n for n in current_nodes if n['id'] == el['ni'])['x'])**2 + (next(n for n in current_nodes if n['id'] == el['nj'])['y']-next(n for n in current_nodes if n['id'] == el['ni'])['y'])**2) for el in current_elements if el['type'] == 'Beam'])
    if total_beam_len == 0: total_beam_len = 1.0

    seismic_weight_total = 0.0
    seismic_mass_per_floor = total_floor_dl + ((0.25 if live_load <= 3.0 else 0.50) * total_floor_ll)

    for el in current_elements:
        el['load_kN_m'] = 0.0
        if el['type'] == 'Beam':
            ni = next(n for n in current_nodes if n['id'] == el['ni'])
            nj = next(n for n in current_nodes if n['id'] == el['nj'])
            L = math.sqrt((nj['x']-ni['x'])**2 + (nj['y']-ni['y'])**2)
            el['length'] = L
            
            b, h = map(lambda x: float(x)/1000, el['size'].split('x'))
            if el.get('angle', 0) == 90: b, h = h, b 
            
            is_secondary = not (ni.get('is_primary', True) and nj.get('is_primary', True))
            wall_udl = 0.0
            if not is_secondary: 
                h_story = (z_elevations.get(ni['floor'], 3.0) - z_elevations.get(ni['floor']-1, 0.0)) if ni['floor']>0 else 3.0
                wall_udl = (wall_thickness / 1000.0) * 20.0 * max(0.1, (h_story - h))

            area_dl_udl = total_floor_dl / total_beam_len
            area_ll_udl = total_floor_ll / total_beam_len
            self_wt = b * h * 25.0
            
            el['load_kN_m'] = 1.5 * (area_dl_udl + area_ll_udl + wall_udl + self_wt)
            seismic_weight_total += (wall_udl + self_wt) * L

    seismic_weight_total += (seismic_mass_per_floor * num_stories)
    V_base = lateral_coeff * seismic_weight_total
    
    floor_weights = {z: seismic_weight_total / num_stories for z in range(1, num_stories + 1)}
    sum_wh2 = sum([floor_weights[z] * (z_elevations[z]**2) for z in floor_weights])
    floor_forces = {z: V_base * (floor_weights[z] * (z_elevations[z]**2)) / sum_wh2 if sum_wh2 > 0 else 0 for z in floor_weights}

    for n in current_nodes:
        if n['z'] > 0:
            nodes_this_floor = len([nd for nd in current_nodes if nd['floor'] == n['floor']])
            F_global[n['id'] * 6] += (floor_forces[n['floor']] / nodes_this_floor) if nodes_this_floor > 0 else 0

    for el in current_elements:
        ni_data = next(n for n in current_nodes if n['id'] == el['ni'])
        nj_data = next(n for n in current_nodes if n['id'] == el['nj'])
        L = math.sqrt((nj_data['x']-ni_data['x'])**2 + (nj_data['y']-ni_data['y'])**2 + (nj_data['z']-ni_data['z'])**2)
        el['length'] = L
        
        b, h = map(lambda x: float(x)/1000, el['size'].split('x'))
        if el.get('angle', 0) == 90: b, h = h, b 
            
        A_sec, Iy_sec, Iz_sec = b * h, (b * h**3) / 12.0, (h * b**3) / 12.0
        dim_min, dim_max = min(b, h), max(b, h)
        J_sec = (dim_min**3 * dim_max) * (1/3 - 0.21 * (dim_min/dim_max) * (1 - (dim_min**4) / (12 * dim_max**4)))

        el['E'] = E_conc
        el['Iz'] = Iz_sec

        T_matrix = get_transformation_matrix(ni_data, nj_data)
        k_local = get_local_stiffness(E_conc, G_conc, A_sec, Iy_sec, Iz_sec, J_sec, L)
        el['k_global'] = np.dot(np.dot(T_matrix.T, k_local), T_matrix)

        w = el.get('load_kN_m', 0.0)
        if el['type'] == 'Beam' and w > 0:
            V, M = (w * L) / 2.0, (w * L**2) / 12.0
            F_local_ENL = np.zeros(12)
            F_local_ENL[1], F_local_ENL[5], F_local_ENL[7], F_local_ENL[11] = V, M, V, -M
            P_global = np.dot(T_matrix.T, F_local_ENL)
            F_global[el['ni']*6 : el['ni']*6+6] -= P_global[0:6]
            F_global[el['nj']*6 : el['nj']*6+6] -= P_global[6:12]

    K_global = np.zeros((num_nodes * 6, num_nodes * 6))
    for el in current_elements:
        i_dof, j_dof = el['ni'] * 6, el['nj'] * 6
        k_g = el['k_global']
        K_global[i_dof:i_dof+6, i_dof:i_dof+6] += k_g[0:6, 0:6]
        K_global[i_dof:i_dof+6, j_dof:j_dof+6] += k_g[0:6, 6:12]
        K_global[j_dof:j_dof+6, i_dof:i_dof+6] += k_g[6:12, 0:6]
        K_global[j_dof:j_dof+6, j_dof:j_dof+6] += k_g[6:12, 6:12]

    fixed_dofs = [dof for n in current_nodes if n['z'] == 0 for dof in range(n['id'] * 6, n['id'] * 6 + 6)]
    free_dofs = sorted(list(set(range(num_nodes * 6)) - set(fixed_dofs)))
    
    try:
        U_free = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])
    except np.linalg.LinAlgError:
        raise Exception("Matrix is singular. Frame geometry is unstable.")
        
    U_global = np.zeros(num_nodes * 6)
    U_global[free_dofs] = U_free
    
    for el in current_elements:
        ni_data = next(n for n in current_nodes if n['id'] == el['ni'])
        nj_data = next(n for n in current_nodes if n['id'] == el['nj'])
        T_matrix = get_transformation_matrix(ni_data, nj_data)
        
        b, h = map(lambda x: float(x)/1000, el['size'].split('x'))
        if el.get('angle', 0) == 90: b, h = h, b 
            
        dim_min, dim_max = min(b, h), max(b, h)
        J_sec = (dim_min**3 * dim_max) * (1/3 - 0.21 * (dim_min/dim_max) * (1 - (dim_min**4) / (12 * dim_max**4)))
        k_local = get_local_stiffness(E_conc, G_conc, b*h, (b*h**3)/12.0, (h*b**3)/12.0, J_sec, el['length'])
        u_local = np.dot(T_matrix, np.concatenate((U_global[el['ni']*6:el['ni']*6+6], U_global[el['nj']*6:el['nj']*6+6])))
        
        el['u_local'] = u_local
        
        F_local_ENL = np.zeros(12)
        w = el.get('load_kN_m', 0.0)
        if el['type'] == 'Beam' and w > 0:
            V, M = (w * el['length']) / 2.0, (w * el['length']**2) / 12.0
            F_local_ENL[1], F_local_ENL[5], F_local_ENL[7], F_local_ENL[11] = V, M, V, -M
            
        el['F_internal'] = np.dot(k_local, u_local) - F_local_ENL

    return current_elements, U_global

def perform_design(elements_to_design, U_global, current_nodes, z_elevations):
    design_status = True
    mulim_coeff = 0.133 if fy >= 500 else 0.138 
    tau_c_max = 0.62 * math.sqrt(fck) 
    
    floor_drifts = {}
    for z in range(1, num_stories + 1):
        nodes_z = [n for n in current_nodes if n['floor'] == z]
        nodes_prev = [n for n in current_nodes if n['floor'] == z - 1]
        
        max_x_z = max([abs(U_global[n['id']*6]) for n in nodes_z]) if nodes_z else 0
        max_x_prev = max([abs(U_global[n['id']*6]) for n in nodes_prev]) if nodes_prev else 0
        max_y_z = max([abs(U_global[n['id']*6 + 1]) for n in nodes_z]) if nodes_z else 0
        max_y_prev = max([abs(U_global[n['id']*6 + 1]) for n in nodes_prev]) if nodes_prev else 0
        
        drift_x = abs(max_x_z - max_x_prev)
        drift_y = abs(max_y_z - max_y_prev)
        h_story = z_elevations.get(z, 3.0) - z_elevations.get(z-1, 0.0)
        
        floor_drifts[z] = True if max(drift_x, drift_y) > (0.004 * h_story) else False
            
    for el in elements_to_design:
        b_mm, h_mm = map(lambda x: float(x), el['size'].split('x'))
        if el.get('angle', 0) == 90: b_mm, h_mm = h_mm, b_mm 
        
        el['pass'] = True
        el['design_details'] = {}
        el['failure_mode'] = "" 
        
        if el['type'] == 'Beam':
            Mu_y = max(abs(el['F_internal'][4]), abs(el['F_internal'][10]))
            Mu_z = max(abs(el['F_internal'][5]), abs(el['F_internal'][11]))
            Mu_max = max(Mu_y, Mu_z)
            Vu_y = max(abs(el['F_internal'][1]), abs(el['F_internal'][7]))
            Vu_z = max(abs(el['F_internal'][2]), abs(el['F_internal'][8]))
            Vu_max = max(Vu_y, Vu_z)
            
            d_beam = h_mm - 40 
            Mu_lim = mulim_coeff * fck * b_mm * (d_beam**2) / 1e6
            tau_v = (Vu_max * 1000) / (b_mm * d_beam)
            
            # --- IS 456 BEAM REINFORCEMENT CALCULATION (NEW) ---
            A_quad = (0.87 * fy * fy) / (b_mm * d_beam * fck)
            B_quad = -0.87 * fy * d_beam
            C_quad = Mu_max * 1e6
            
            discriminant = B_quad**2 - 4*A_quad*C_quad
            if discriminant < 0 or Mu_max > Mu_lim:
                el['failure_mode'] += "flexure "
                Ast_req = 0
                num_bars = 0
            else:
                Ast1 = (-B_quad + math.sqrt(discriminant)) / (2*A_quad)
                Ast2 = (-B_quad - math.sqrt(discriminant)) / (2*A_quad)
                Ast_req = min(Ast1, Ast2) 
                
                Ast_min = (0.85 * b_mm * d_beam) / fy
                Ast_req = max(Ast_req, Ast_min)
                
                # Sizing Main Bars (Assuming 16mm typical)
                bar_dia = 16
                a_ast = math.pi * (bar_dia**2) / 4
                num_bars = max(2, math.ceil(Ast_req / a_ast))
            
            Ast_prov = num_bars * math.pi * (16**2) / 4 if num_bars > 0 else 0
            
            # --- IS 456 SHEAR REINFORCEMENT CALCULATION (NEW) ---
            pt = min(100 * Ast_prov / (b_mm * d_beam), 3.0) if (b_mm*d_beam) > 0 else 0
            beta = max(0.8 * fck / (6.89 * pt), 1.0) if pt > 0 else 1.0
            tau_c = 0.85 * math.sqrt(0.8 * fck) * (math.sqrt(1 + 5 * beta) - 1) / (6 * beta) if pt > 0 else 0
            tau_c = min(tau_c, tau_c_max)
            
            Asv = 2 * (math.pi * (8**2) / 4) # 2-legged 8mm stirrups
            sv_min = (0.87 * fy * Asv) / (0.4 * b_mm)
            
            if tau_v <= tau_c:
                sv_req = sv_min
            else:
                Vus = (Vu_max * 1000) - (tau_c * b_mm * d_beam)
                sv_calc = (0.87 * fy * Asv * d_beam) / Vus if Vus > 0 else sv_min
                sv_req = min(sv_calc, sv_min)
                
            sv_provided = min(sv_req, 0.75 * d_beam, 300)
            sv_provided = math.floor(sv_provided / 10) * 10 
            
            if tau_v > tau_c_max: el['failure_mode'] += "shear "
            
            # Deflection Check
            w_load = el.get('load_kN_m', 0.0)
            delta_ss_m = (5 * w_load * (el['length']**4)) / (384 * el['E'] * el['Iz']) if (el.get('E',0)*el.get('Iz',0)) != 0 else 0
            delta_rot_m = (el['length'] / 8) * (el.get('u_local', np.zeros(12))[5] - el.get('u_local', np.zeros(12))[11])
            max_deflection = abs(delta_ss_m * 1000) + abs(delta_rot_m * 1000)
            
            if max_deflection > (el['length'] * 1000 / 250): el['failure_mode'] += "deflection "
                
            if el['failure_mode']:
                el['pass'] = False
                design_status = False
                
            el['design_details'] = {
                'ID': el['id'], 'Floor': el['floor'], 'Size (mm)': el['size'],
                'Mu_max (kN.m)': round(Mu_max, 2), 'Vu_max (kN)': round(Vu_max, 2),
                'Bottom Rebars': f"{num_bars}-16Φ" if num_bars > 0 else "FAIL", 
                'Stirrups (8mm)': f"2L @ {sv_provided} c/c" if tau_v <= tau_c_max else "FAIL",
                'Status': 'Safe' if el['pass'] else el['failure_mode'].strip()
            }
                
        elif el['type'] == 'Column':
            Pu = max(abs(el['F_internal'][0]), abs(el['F_internal'][6]))
            Mu_y = max(abs(el['F_internal'][4]), abs(el['F_internal'][10]))
            Mu_z = max(abs(el['F_internal'][5]), abs(el['F_internal'][11]))
            
            L_eff = el['length'] * 1000
            
            e_min_z = max(L_eff / 500 + b_mm / 30, 20.0)
            e_min_y = max(L_eff / 500 + h_mm / 30, 20.0)
            Mu_z = max(Mu_z, Pu * e_min_z / 1000.0)
            Mu_y = max(Mu_y, Pu * e_min_y / 1000.0)
            
            is_slender_z = (L_eff / b_mm) > 12
            is_slender_y = (L_eff / h_mm) > 12
            
            if is_slender_z:
                Mu_z += (Pu * b_mm / 2000.0) * (L_eff / b_mm)**2 / 1000.0
                el['failure_mode'] += "slender_z "
            if is_slender_y:
                Mu_y += (Pu * h_mm / 2000.0) * (L_eff / h_mm)**2 / 1000.0
                el['failure_mode'] += "slender_y "

            Mu_max = max(Mu_y, Mu_z)
            
            Ag = b_mm * h_mm
            d_col = h_mm - 40
            
            Pu_crushing_limit = (0.4 * fck * Ag + 0.67 * fy * 0.04 * Ag) / 1000.0 
            Asc_req_axial = (Pu * 1000 - 0.4 * fck * Ag) / (0.67 * fy - 0.4 * fck) if (Pu * 1000 > 0.4 * fck * Ag) else 0
            Asc_req_mom = (Mu_max * 1e6) / (0.87 * fy * 0.8 * d_col) if Mu_max > 0 else 0
            Asc_calc = Asc_req_axial + Asc_req_mom
            
            if floor_drifts.get(el['floor'], False):
                el['failure_mode'] += "drift "
                el['pass'] = False; design_status = False
            if Pu > Pu_crushing_limit:
                el['failure_mode'] += "axial_crushing "
                el['pass'] = False; design_status = False
            elif Asc_calc > 0.04 * Ag: 
                el['failure_mode'] += "steel_limit "
                el['pass'] = False; design_status = False
                
            status_str = 'Safe'
            if not el['pass']: status_str = el['failure_mode'].strip()
            elif is_slender_y or is_slender_z: status_str = 'Safe (Slender)'
                    
            el['design_details'] = {
                'ID': el['id'], 'Floor': el['floor'], 'Size (mm)': el['size'],
                'Orientation': f"{el.get('angle', 0)}°", 'Pu_max (kN)': round(Pu, 2), 'Mu_max (kN.m)': round(Mu_max, 2),
                'Req Asc (mm²)': round(max(Asc_calc, 0.008 * Ag), 2),
                'Status': status_str
            }
                    
    return elements_to_design, design_status

def group_elements(elements_list, elem_type):
    df = pd.DataFrame([el['design_details'] for el in elements_list if el['type'] == elem_type])
    if df.empty: return df
    if elem_type == 'Column':
        grouped = df.groupby(['Floor', 'Size (mm)', 'Orientation']).agg(
            Max_Pu=('Pu_max (kN)', 'max'), Max_Mu=('Mu_max (kN.m)', 'max'), Max_Req_Asc=('Req Asc (mm²)', 'max'), Count=('ID', 'count')
        ).reset_index()
        grouped['Group ID'] = [f"C{i+1}" for i in range(len(grouped))]
        return grouped
    elif elem_type == 'Beam':
        grouped = df.groupby(['Floor', 'Size (mm)', 'Bottom Rebars', 'Stirrups (8mm)']).agg(
            Max_Mu=('Mu_max (kN.m)', 'max'), Max_Vu=('Vu_max (kN)', 'max'), Count=('ID', 'count')
        ).reset_index()
        grouped['Group ID'] = [f"B{i+1}" for i in range(len(grouped))]
        return grouped

# --- MAIN EXECUTION BLOCK (3-STAGE AI) ---
if st.button("Run 3-Stage AI Optimization & Design", type="primary", use_container_width=True):
    with st.spinner("Analyzing Frame & Designing Slabs..."):
        if len(nodes) < 2:
            st.error("Not enough valid nodes generated. Please check your grid definitions.")
            st.stop()
            
        x_coords = sorted(list(x_map.values()))
        y_coords = sorted(list(y_map.values()))
        x_spans = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1) if (x_coords[i+1] - x_coords[i]) > 0.5]
        y_spans = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1) if (y_coords[i+1] - y_coords[i]) > 0.5]
        
        panel_lx = max(x_spans) if x_spans else 1.0
        panel_ly = max(y_spans) if y_spans else 1.0
        Lx, Ly = min(panel_lx, panel_ly), max(panel_lx, panel_ly)
        ratio = Ly / Lx
        slab_behavior = "One-Way" if ratio > 2.0 else "Two-Way"
        
        alphas = [0.062, 0.074, 0.084, 0.093, 0.099, 0.104, 0.113, 0.118]
        alpha_x = np.interp(ratio, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0], alphas) if slab_behavior == "Two-Way" else 0.125
        R_max = 0.133 * fck if fy >= 500 else 0.138 * fck
        
        opt_slab_thickness = slab_thickness
        for _ in range(15):
            d_req_def = (Lx * 1000) / 28.0 
            w_u = 1.5 * (live_load + floor_finish + (opt_slab_thickness / 1000.0) * 25.0)
            slab_Mu = alpha_x * w_u * (Lx**2)
            d_req_flex = math.sqrt((slab_Mu * 1e6) / (R_max * 1000))
            D_req = max(d_req_def, d_req_flex) + 20 + 5
            if opt_slab_thickness >= D_req: break
            if auto_optimize: opt_slab_thickness += 10
            else: break

        passed_phase1, passed_phase2, passed_phase3 = False, False, False
        max_iters = 12 
        
        iteration = 1
        while iteration <= max_iters:
            try:
                elements, U_global_res = run_analysis_dynamic(elements, nodes, opt_slab_thickness)
                elements, passed_phase1 = perform_design(elements, U_global_res, nodes, z_elevations)
                
                if passed_phase1 or not auto_optimize:
                    if passed_phase1: st.success(f"✅ Phase 1: 100% Safe Design in {iteration} iteration(s).")
                    break
                else:
                    for el in elements:
                        if not el['pass']:
                            b_str, h_str = el['size'].split('x')
                            b, h = int(b_str), int(h_str)
                            mode = el.get('failure_mode', '')
                            
                            if el['type'] == 'Beam':
                                if 'shear' in mode: b += 50; h += 50
                                elif 'flexure' in mode or 'deflection' in mode: h += 50 
                                else: h += 50
                            else: 
                                if 'axial_crushing' in mode or 'drift' in mode: b += 50; h += 50
                                elif 'steel_limit' in mode:
                                    if 'slender_z' in mode: b += 50
                                    if 'slender_y' in mode: h += 50
                                    if 'slender_z' not in mode and 'slender_y' not in mode:
                                        h += 50 
                                        if h - b > 200: b += 50 
                                else: b += 50; h += 50
                            
                            b, h = min(b, 1000), min(h, 1200) 
                            el['size'] = f"{b}x{h}"
                    iteration += 1
            except Exception as e:
                st.error(f"Solver Error: {e}")
                st.stop()

        if not passed_phase1 and allow_ai_restructure:
            st.warning("⚠️ Phase 1 limits reached. Injecting Secondary Beams...")
            added_sec_nodes = 0
            for el in elements:
                if el['type'] == 'Beam' and not el['pass'] and el['length'] > 4.5:
                    ni = next(n for n in nodes if n['id'] == el['ni'])
                    nj = next(n for n in nodes if n['id'] == el['nj'])
                    mid_x, mid_y = (ni['x'] + nj['x']) / 2.0, (ni['y'] + nj['y']) / 2.0
                    
                    if not any(math.sqrt((p['x']-mid_x)**2 + (p['y']-mid_y)**2) < 1.0 for p in secondary_xy if p['floor'] == ni['floor']):
                        secondary_xy.append({'x': mid_x, 'y': mid_y, 'floor': ni['floor']})
                        added_sec_nodes += 1
                            
            if added_sec_nodes > 0:
                ai_nodes, ai_elements = build_geometry(primary_xy, secondary_xy, z_elevations, num_stories, col_dim, beam_dim)
                ai_iters = 1
                while ai_iters <= 10: 
                    ai_elements, U_global_res = run_analysis_dynamic(ai_elements, ai_nodes, opt_slab_thickness)
                    ai_elements, passed_phase2 = perform_design(ai_elements, U_global_res, ai_nodes, z_elevations)
                    if passed_phase2:
                        st.success(f"✅ Phase 2 Successful! Secondary beams stabilized the structure.")
                        nodes, elements = ai_nodes, ai_elements
                        break
                    else:
                        for el in ai_elements:
                            if not el['pass']:
                                b_str, h_str = el['size'].split('x')
                                b, h = min(int(b_str)+50, 1000), min(int(h_str)+50, 1200)
                                el['size'] = f"{b}x{h}"
                    ai_iters += 1
                if not passed_phase2: nodes, elements = ai_nodes, ai_elements

        if not passed_phase1 and not passed_phase2 and allow_ai_restructure:
            st.error("⚠️ Phase 2 failed. Deep Restructuring...")
            hard_primary_xy = list(primary_xy)
            added_cols = 0
            for pt in secondary_xy:
                if not any(math.sqrt((p['x']-pt['x'])**2 + (p['y']-pt['y'])**2) < 1.0 for p in hard_primary_xy):
                    hard_primary_xy.append({'x': pt['x'], 'y': pt['y'], 'angle': 0.0})
                    added_cols += 1
            for el in elements:
                if el['type'] == 'Beam' and not el['pass'] and el['length'] > 4.0:
                    ni = next(n for n in nodes if n['id'] == el['ni'])
                    nj = next(n for n in nodes if n['id'] == el['nj'])
                    mid_x, mid_y = (ni['x'] + nj['x']) / 2.0, (ni['y'] + nj['y']) / 2.0
                    if not any(math.sqrt((p['x']-mid_x)**2 + (p['y']-mid_y)**2) < 1.0 for p in hard_primary_xy):
                        hard_primary_xy.append({'x': mid_x, 'y': mid_y, 'angle': 0.0})
                        added_cols += 1

            if added_cols > 0:
                hard_nodes, hard_elements = build_geometry(hard_primary_xy, [], z_elevations, num_stories, col_dim, beam_dim)
                hard_iters = 1
                while hard_iters <= 10:
                    hard_elements, U_global_res = run_analysis_dynamic(hard_elements, hard_nodes, opt_slab_thickness)
                    hard_elements, passed_phase3 = perform_design(hard_elements, U_global_res, hard_nodes, z_elevations)
                    if passed_phase3:
                        st.success(f"✅ Phase 3 Deep Restructuring Successful!")
                        break
                    else:
                        for el in hard_elements:
                            if not el['pass']:
                                b_str, h_str = el['size'].split('x')
                                b, h = min(int(b_str)+50, 1000), min(int(h_str)+50, 1200)
                                el['size'] = f"{b}x{h}"
                    hard_iters += 1
                nodes, elements = hard_nodes, hard_elements

        # --- OUTPUT GENERATION ---
        st.divider()
        st.header("Results & Structural Detailing")
        
        col_grp1, col_grp2 = st.columns(2)
        col_grp1.subheader("Beam Reinforcement Schedule")
        col_grp1.dataframe(group_elements(elements, 'Beam'), use_container_width=True)
        col_grp2.subheader("Column Grouping Schedule")
        col_grp2.dataframe(group_elements(elements, 'Column'), use_container_width=True)

        tab1, tab2 = st.tabs(["Raw Beams", "Raw Columns"])
        tab1.dataframe(pd.DataFrame([el['design_details'] for el in elements if el['type'] == 'Beam']), use_container_width=True)
        tab2.dataframe(pd.DataFrame([el['design_details'] for el in elements if el['type'] == 'Column']), use_container_width=True)
