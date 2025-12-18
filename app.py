import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- LOAD & CLEAN DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("Elements Data Values.csv")
    column_map = {
        'Name': 'Name', 'Symbol': 'Symbol',
        'Atomic_Number': 'Atomic Number', 'Atomic_Weight': 'Atomic Weight',
        'Melting_Point': 'Melting Point', 'Boiling_Point': 'Boiling Point',
        'Density (kg/m³)': 'Density', 'Electronegativity': 'Electronegativity',
        'Block': 'Block', 'Phase': 'Phase',
        'Radioactivity': 'Radioactivity'
    }
    df = df.rename(columns=column_map)
    needed = ['Name', 'Symbol', 'Atomic Number', 'Atomic Weight', 'Phase',
              'Melting Point', 'Boiling Point', 'Density', 'Radioactivity', 'Electronegativity', 'Block']
    df = df[[col for col in needed if col in df.columns]].copy()
    numeric_cols = ['Atomic Number', 'Atomic Weight', 'Melting Point', 'Boiling Point',
                    'Density', 'Electronegativity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# --- TITLE ---
st.title("Prerna's Periodic Table Explorer")

# --- SEARCH + HIGHLIGHT LOGIC (NOW IN MAIN AREA) ---
# Center the search bar a bit for better looks
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search = st.text_input("Search by Name or Atomic Number", placeholder="e.g., Hydrogen or 1", help="Type element name or number to highlight and show details")

highlighted_atomic = None
if search:
    result = df[
        df['Name'].str.contains(search, case=False, na=False) |
        df['Atomic Number'].astype(str).str.contains(search)
    ].head(1)
    if not result.empty:
        el = result.iloc[0]
        highlighted_atomic = int(el['Atomic Number'])
        # === ELEMENT CARD WITH PRIMORDIAL SUPPORT === (your existing code here, unchanged)
        st.markdown("""
        <style>
            .big-font {font-size:28px!important;font-weight:bold;color:#1E90FF;}
            .element-card {
                padding:25px;
                border-left:14px solid transparent;
                background:#f0f8ff;
                border-radius:15px;
                margin:20px 0;
                box-shadow:0 4px 15px rgba(0,0,0,0.1);
            }
            .property-label {font-weight:bold;color:#333;}
            .rad-stable {border-left-color: #00CCFF;}
            .rad-primordial {border-left-color: #00FF99;}
            .rad-synthetic {border-left-color: #FF3333;}
        </style>
        """, unsafe_allow_html=True)
        an = int(el['Atomic Number'])
        rad_value = str(el['Radioactivity']).strip().lower()
        if rad_value == "no":
            rad_class = "rad-stable"
            warning = ""
        elif rad_value == "primordial":
            rad_class = "rad-primordial"
            warning = "☢️ Primordial – occurs naturally (but still radioactive)"
        else:  # "yes"
            rad_class = "rad-synthetic"
            warning = "☢️ Radioactive – synthetic or trace only"
        st.markdown(f'<div class="element-card {rad_class}">', unsafe_allow_html=True)
        if warning:
            st.markdown(f"<p class='big-font'>{el['Symbol']} – {el['Name']} <span style='color:#FF0000;font-weight:bold;'>{warning}</span></p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='big-font'>{el['Symbol']} – {el['Name']}</p>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<span class='property-label'>Atomic Number:</span> {an}", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Atomic Weight:</span> {el.get('Atomic Weight',0):.4f} u", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Phase:</span> {el.get('Phase','N/A')}", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Radioactivity:</span> {el['Radioactivity']}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<span class='property-label'>Melting Point:</span> {el.get('Melting Point','N/A')} K", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Boiling Point:</span> {el.get('Boiling Point','N/A')} K", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Density:</span> {el.get('Density','N/A')} kg/m³", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if not search:
    st.markdown("<p style='font-size:18px; text-align:center; color:#555;'>🔍 Search for an element above to see its details!</p>", unsafe_allow_html=True)

st.markdown(
    "<p style='font-size:14px; color:#000000; text-align:center;'>Built by Prerna Lotlikar.</p>",
    unsafe_allow_html=True
)
