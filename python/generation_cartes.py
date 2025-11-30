import os
import shutil
import numpy as np
import xarray as xr
from datetime import timedelta, datetime
from scipy import ndimage
import folium
from folium.plugins import TimestampedGeoJson
import plotly.graph_objects as go 
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Definition du repertoire de travail
# Utiliser "." pour le repertoire courant ou specifier un chemin absolu.
WORK_DIR = r"."  

# Fichiers d'entree et de sortie
FILENAME = os.path.join(WORK_DIR, "MEDSEA2019.nc")
OUTPUT_ZARR = "Resultat_AMP_Lion.zarr"

# Noms des fichiers de visualisation
FILE_STATIC_MAP = "Carte_Interactive.html"            
FILE_DYNAMIC_MAP = "Carte_Med_Interactive_Full.html" 
FILE_DASH_MAP = "Carte_Dashboard_Interne.html"        
FILE_GRAPH_COMP = "composant_graphique.html"         
FILE_DASHBOARD = "Dashboard_Interactif_Complet.html" 

# Parametres de simulation
AMP_BOX = {'lon_min': 4.2, 'lon_max': 5.2, 'lat_min': 42.5, 'lat_max': 43.2}
NB_PARTICLES = 10000

# Verification de l'existance du fichier source
if not os.path.exists(FILENAME):
    print(f"Error: Input file not found at {os.path.abspath(FILENAME)}")
    print("Check 'WORK_DIR' path in configuration.")
    exit()

# =============================================================================
# 2. SIMULATION
# =============================================================================
def GibraltarWall(particle, fieldset, time):
    """Boundary condition to prevent particles from crossing Gibraltar."""
    if particle.lon < -5.8:
        particle.lon = -5.8

print(f"[1/6] Initializing simulation ({NB_PARTICLES} particles)...")

# Chargement des donnees initiales
ds_init = xr.open_dataset(FILENAME, decode_times=False)
lons_array = ds_init['lon'].values
lats_array = ds_init['lat'].values
depth_val = ds_init['depth'][0].values 
uo_sample = ds_init['uo'][0, 0, :, :].values 
ds_init.close()

# Creation du masque cotier
mask_land = np.isnan(uo_sample) | (uo_sample > 1e10) | (uo_sample == 0)
struct = ndimage.generate_binary_structure(2, 2)
mask_land_dilated = ndimage.binary_dilation(mask_land, structure=struct, iterations=4)
mask_coastal = mask_land_dilated & (~mask_land)
valid_y, valid_x = np.where(mask_coastal)

# Filtrage des points de depart (Exclusion zone Atlantique > -5.5)
safe_indices = [i for i in range(len(valid_x)) if lons_array[valid_x[i]] > -5.5]
valid_y, valid_x = valid_y[safe_indices], valid_x[safe_indices]

indices = np.random.choice(len(valid_y), NB_PARTICLES, replace=False) if len(valid_y) > NB_PARTICLES else np.arange(len(valid_y))

# Configuration du FieldSet et ParticleSet
fieldset = FieldSet.from_netcdf({'U': FILENAME, 'V': FILENAME}, {'U': 'uo', 'V': 'vo'}, 
                                {'lat': 'lat', 'lon': 'lon', 'time': 'time', 'depth': 'depth'}, 
                                allow_time_extrapolation=True)
pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle, 
                             lon=lons_array[valid_x[indices]], lat=lats_array[valid_y[indices]], 
                             depth=np.full(len(indices), depth_val))

# Execution de la simulation
zarr_path = os.path.join(WORK_DIR, OUTPUT_ZARR)
if os.path.exists(zarr_path): shutil.rmtree(zarr_path)

output_file = pset.ParticleFile(name=zarr_path, outputdt=timedelta(hours=12))
pset.execute(AdvectionRK4 + pset.Kernel(GibraltarWall), runtime=timedelta(days=100), dt=timedelta(minutes=30), output_file=output_file)

# =============================================================================
# 3. ANALYSE
# =============================================================================
print("[2/6] Analyzing trajectories...")
ds = xr.open_zarr(zarr_path)
lon_traj = ds['lon'].values
lat_traj = ds['lat'].values
time_vals = ds['time'].values
days_traj = ((time_vals - time_vals[0, 0]) / 1e9) / 86400.0 

n_part, n_steps = lon_traj.shape
captured_ids = set()
curve = []
t_axis = []

# Calcul des captures (Presence dans AMP apres J+30)
for t in range(n_steps):
    day = float(days_traj[0, t])
    if np.isnan(day): break
    
    if day >= 30:
        in_box = ((lon_traj[:, t] >= AMP_BOX['lon_min']) & (lon_traj[:, t] <= AMP_BOX['lon_max']) & 
                  (lat_traj[:, t] >= AMP_BOX['lat_min']) & (lat_traj[:, t] <= AMP_BOX['lat_max']))
        captured_ids.update(np.where(in_box)[0])
            
    curve.append(len(captured_ids))
    t_axis.append(day)

# =============================================================================
# 4. CARTE 1 : STATIQUE
# =============================================================================
print("[3/6] Generating static map...")
m1 = folium.Map(location=[36.0, 15.0], zoom_start=5, tiles='CartoDB positron')

STEP_STATIC = 50 
for p in range(0, n_part, STEP_STATIC): 
    lats, lons = lat_traj[p, :], lon_traj[p, :]
    valid = ~np.isnan(lats)
    if not np.any(valid): continue
    lats, lons = lats[valid], lons[valid]
    
    # Point de depart (Vert)
    folium.CircleMarker([lats[0], lons[0]], radius=2, color='green', fill_color='green', fill=True, fill_opacity=1).add_to(m1)
    
    # Point d'arrivee (Rouge)
    folium.CircleMarker([lats[-1], lons[-1]], radius=3, color='#e74c3c', fill_color='#e74c3c', fill=True, fill_opacity=1).add_to(m1)
    
    # Trajectoire
    folium.PolyLine(list(zip(lats, lons)), color='blue', weight=0.6, opacity=0.4).add_to(m1)

m1.save(os.path.join(WORK_DIR, FILE_STATIC_MAP))

# =============================================================================
# 5. CARTE 2 : DYNAMIQUE
# =============================================================================
print("[4/6] Generating dynamic map (compressed)...")
m2 = folium.Map(location=[36.0, 15.0], zoom_start=5, tiles='CartoDB positron')

features = []
STEP_DYN_PART = 12  
STEP_DYN_TIME = 2   

start_date = datetime(2024, 1, 1)

for p in range(0, n_part, STEP_DYN_PART):
    color = '#e74c3c' if p in captured_ids else '#3498db'
    radius = 1.5 if p in captured_ids else 1.0 
    opacity = 1.0 if p in captured_ids else 0.6
    
    for t in range(0, n_steps, STEP_DYN_TIME): 
        if np.isnan(lon_traj[p, t]): continue
        current_time = start_date + timedelta(days=float(days_traj[0, t]))
        
        lat_opt = round(float(lat_traj[p, t]), 3)
        lon_opt = round(float(lon_traj[p, t]), 3)

        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon_opt, lat_opt]},
            'properties': {
                'time': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'icon': 'circle',
                'iconstyle': {'fillColor': color, 'fillOpacity': opacity, 'stroke': 'false', 'radius': radius}
            }
        })

TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='P1D', duration='P1D', add_last_point=False, 
    auto_play=False, loop=False, max_speed=20, loop_button=True, 
    date_options='YYYY-MM-DD HH:mm', time_slider_drag_update=True,
    transition_time=100  
).add_to(m2)

m2.save(os.path.join(WORK_DIR, FILE_DYNAMIC_MAP))

# =============================================================================
# 6. CARTE 3 : DASHBOARD
# =============================================================================
print("[5/6] Generating dashboard components...")
m3 = folium.Map(location=[42.8, 4.7], zoom_start=7, tiles='CartoDB positron') 

folium.Rectangle(
    bounds=[[AMP_BOX['lat_min'], AMP_BOX['lon_min']], [AMP_BOX['lat_max'], AMP_BOX['lon_max']]],
    color="green", fill=True, fill_color="green", fill_opacity=0.3, weight=2,
    popup="AMP (Aire Marine Protégée)"
).add_to(m3)

features_dash = []
STEP_DASH_PART = 10 
STEP_DASH_TIME = 2

for p in range(0, n_part, STEP_DASH_PART):
    is_captured = p in captured_ids
    color = '#e74c3c' if is_captured else '#3498db'
    radius = 3 if is_captured else 2
    opacity = 1.0 if is_captured else 0.4
    
    for t in range(0, n_steps, STEP_DASH_TIME): 
        if np.isnan(lon_traj[p, t]): continue
        current_time = start_date + timedelta(days=float(days_traj[0, t]))
        
        features_dash.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [round(float(lon_traj[p, t]), 3), round(float(lat_traj[p, t]), 3)]},
            'properties': {
                'time': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'icon': 'circle',
                'iconstyle': {'fillColor': color, 'fillOpacity': opacity, 'stroke': 'false', 'radius': radius}
            }
        })

TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features_dash},
    period='P1D', duration='P1D', add_last_point=False, 
    auto_play=False, loop=False, max_speed=20, loop_button=True, 
    date_options='YYYY-MM-DD HH:mm', time_slider_drag_update=True,
    transition_time=100  
).add_to(m3)

m3.save(os.path.join(WORK_DIR, FILE_DASH_MAP))

# =============================================================================
# 7. DASHBOARD & GRAPHIQUE
# =============================================================================
print("[6/6] Assembling final dashboard...")

fig = go.Figure()
fig.add_vrect(x0=0, x1=30, fillcolor="#bdc3c7", opacity=0.2, layer="below", line_width=0)
fig.add_annotation(x=15, y=0.05, xref="x", yref="paper", text="Phase de<br>dispersion", showarrow=False, font=dict(color="#7f8c8d"))
fig.add_vline(x=30, line_dash="dash", line_color="#2c3e50")

fig.add_annotation(
    x=30, y=0.95, xref="x", yref="paper",
    text="<b>*</b>", showarrow=False,
    font=dict(size=24, color="#e74c3c")
)

fig.add_trace(go.Scatter(x=t_axis, y=curve, mode='lines', name='Capture cumulée', line=dict(color='#e74c3c', width=3)))

fig.update_layout(height=500, title="<b>Dynamique d'accumulation</b>", title_x=0.95,
                  xaxis=dict(title="Temps (Jours)", range=[0, 100]),
                  yaxis=dict(title="Nb particules (cumul)", range=[0, max(curve)*1.2], rangemode='tozero'),
                  plot_bgcolor='white', hovermode="x unified", margin=dict(l=50, r=20, t=50, b=50),
                  legend=dict(orientation="h", y=1.05, x=0.3))
fig.write_html(os.path.join(WORK_DIR, FILE_GRAPH_COMP), config={'displayModeBar': False})

dashboard_html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Dynamique des particules</title>
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; background-color: #f4f4f4; }}
        header {{ background: #2c3e50; color: white; padding: 0 20px; height: 60px; display: flex; align-items: center; justify-content: space-between; }}
        h1 {{ margin: 0; font-size: 1.2rem; }}
        .main-content {{ display: flex; flex: 1; overflow: hidden; }}
        .map-panel {{ width: 55%; border-right: 1px solid #ccc; }}
        .map-frame {{ width: 100%; height: 100%; border: none; }}
        .side-panel {{ width: 45%; padding: 20px; overflow-y: auto; background: white; }}
        .graph-frame {{ width: 100%; height: 520px; border: none; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 8px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #eee; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .card h3 {{ margin-top: 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 1rem; }}
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; }}
        .stat-val {{ font-weight: bold; color: #2c3e50; }}
        .info-note {{ background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; font-size: 0.85rem; border-left: 4px solid #ffeeba; margin-top: 15px; }}
    </style>
</head>
<body>
<header>
    <div>
        <h1>Dynamique des particules</h1>
        <span style="font-size: 0.9rem; opacity: 0.8;">Simulation Golfe du Lion (100 Jours)</span>
    </div>
</header>
<div class="main-content">
    <div class="map-panel">
        <iframe src="{FILE_DASH_MAP}" class="map-frame"></iframe>
    </div>
    <div class="side-panel">
        <iframe src="{FILE_GRAPH_COMP}" class="graph-frame"></iframe>
        
        <div class="card">
            <h3>Statistiques globales</h3>
            <div class="stat-row"><span>Total réel particules :</span> <span class="stat-val">{NB_PARTICLES}</span></div>
            <div class="stat-row"><span>Total capturées :</span> <span class="stat-val">{curve[-1]}</span></div>
            <div class="stat-row"><span>Taux de capture :</span> <span class="stat-val">{curve[-1]/NB_PARTICLES:.1%}</span></div>
        </div>

        <div class="card">
            <h3>Légende et interprétation</h3>
            <div class="stat-row"><div><span style="color:#e74c3c;">●</span> Capturées</div><div style="color:#666;">Dans l'AMP</div></div>
            <div class="stat-row"><div><span style="color:#3498db;">●</span> Libres</div><div style="color:#666;">En mer</div></div>
            
            <div class="info-note">
                <strong><span style="color:#e74c3c; font-size:1.2em;">*</span> Phase de compétence (dès J+30) :</strong><br>
                Ce symbole marque le début de la période où la capture devient possible.
                <br>
                <ul>
                    <li><strong>Organismes vivants :</strong> Stade où les larves sont assez matures pour s'installer dans un habitat (recrutement).</li>
                    <li><strong>Objets inertes :</strong> Moment où les conditions physiques (échouage, densité) favorisent l'accumulation locale.</li>
                </ul>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

with open(os.path.join(WORK_DIR, FILE_DASHBOARD), "w", encoding="utf-8") as f:
    f.write(dashboard_html)

print("-" * 30)
print("Processing completed successfully.")
print(f"Output directory: {os.path.abspath(WORK_DIR)}")
print("-" * 30)