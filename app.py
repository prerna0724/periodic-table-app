import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv("Elements Data Values.csv")

df = load_data()

# --- CLEAN ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['HalfLife'] = df['HalfLife'].fillna(1e30)
df['Lifetime'] = df['Lifetime'].astype('object')
df['Lifetime'] = df['Lifetime'].fillna('Stable')

# --- TITLE ---
st.title("🧪 Baby Chemist's Periodic Table Explorer")
st.sidebar.header("Controls")

# --- SEARCH ---
search = st.sidebar.text_input("Search (Name/Symbol/#)", "")
if search:
    result = df[
        df['Name'].str.contains(search, case=False, na=False) |
        df['Symbol'].str.contains(search, case=False, na=False) |
        df['Atomic_Number'].astype(str).str.contains(search)
    ]
    if not result.empty:
        el = result.iloc[0]
        st.markdown("""
        <style>
            .big-font {font-size:22px!important;font-weight:bold;color:#1E90FF;}
            .element-card {padding:20px;border-left:6px solid #1E90FF;background:#f8f9fa;border-radius:10px;margin:15px 0;box-shadow:0 2px 5px rgba(0,0,0,0.1);}
            .property-label {font-weight:bold;color:#333;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="element-card">', unsafe_allow_html=True)
        st.markdown(f"<p class='big-font'>{el['Symbol']} – {el['Name']}</p>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<span class='property-label'>Atomic #:</span> {int(el['Atomic_Number'])}", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Weight:</span> {el['Atomic_Weight']:.3f} u", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Phase:</span> {el['Phase']}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<span class='property-label'>Melts:</span> {el.get('Melting_Point','N/A')} K", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Boils:</span> {el.get('Boiling_Point','N/A')} K", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Density:</span> {el.get('Density','N/A')} g/cm³", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Not found!")

# --- PROPERTY SELECTOR ---
prop = st.sidebar.selectbox("Color by:", ['Density', 'Electronegativity', 'Melting_Point'])

# --- PERIOD & GROUP ---
def get_period(a): return min((a>2)+(a>10)+(a>18)+(a>36)+(a>54)+(a>86)+1, 7)
def get_group(a): return (a-1)%18 + 1

plot_df = df.copy()
plot_df['Period'] = plot_df['Atomic_Number'].apply(get_period)
plot_df['Group'] = plot_df['Atomic_Number'].apply(get_group)

# --- HEATMAP (FIXED!) ---
full_grid = pd.DataFrame(index=range(1,8), columns=range(1,19))
for _, r in plot_df.iterrows():
    val = r[prop]
    if pd.isna(val):
        full_grid.loc[r['Period'], r['Group']] = np.nan
    else:
        try:
            full_grid.loc[r['Period'], r['Group']] = float(val)
        except:
            full_grid.loc[r['Period'], r['Group']] = np.nan

fig, ax = plt.subplots(figsize=(18,8))
sns.heatmap(
    full_grid,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    ax=ax,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={"label": prop}  # ← This is fine now
)
ax.set_title(f"Periodic Table: {prop}")
ax.set_xlabel("Group"); ax.set_ylabel("Period")
st.pyplot(fig)

# --- PLOTLY ---
st.write("**Density vs Atomic Number (Interactive)**")
density_df = df[['Atomic_Number', 'Density', 'Phase', 'Name']].dropna(subset=['Density'])
density_df['Density'] = pd.to_numeric(density_df['Density'], errors='coerce')
density_df = density_df.dropna(subset=['Density'])

fig_px = px.scatter(
    density_df, x='Atomic_Number', y='Density', color='Phase',
    hover_name='Name', title="Density vs Atomic Number",
    labels={'Density': 'Density (kg/m³)'},
    color_discrete_map={'Gas':'blue','Solid':'red','Liquid':'green'}
)
fig_px.update_layout(height=500)
st.plotly_chart(fig_px, use_container_width=True)

st.caption("Built by Prerna 🧪 | Live & Fixed")
