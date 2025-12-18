import streamlit as st
import pandas as pd

# --- LOAD & CLEAN DATA --- (unchanged)
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

# --- NEON CYBERPUNK STYLING ---
st.markdown("""
<style>
    /* Full dark background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }
    
    /* Neon glowing title */
    h1 {
        text-align: center;
        font-size: 3.5rem !important;
        color: #00FFFF !important;
        text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF;
        margin-bottom: 2rem !important;
    }
    
    /* Neon search bar */
    div[data-baseweb="input"] > div {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 2px solid #00FFFF !important;
        border-radius: 50px !important;
        box-shadow: 0 0 15px #00FFFF !important;
        backdrop-filter: blur(10px);
    }
    input {
        color: #00FFFF !important;
        font-size: 1.2rem !important;
        text-align: center;
    }
    ::placeholder {
        color: #00AAAA !important;
        opacity: 1;
    }
    
    /* Prompt text neon */
    .neon-text {
        text-align: center;
        color: #00FFFF;
        text-shadow: 0 0 10px #00FFFF;
        font-size: 1.3rem;
        margin: 1.5rem 0;
    }
    
    /* Footer neon */
    .footer {
        text-align: center;
        color: #AAAAAA;
        font-size: 0.9rem;
        margin-top: 3rem;
        text-shadow: 0 0 5px #00FFFF;
    }
    
    /* Element card - darker with neon glow */
    .element-card {
        background: rgba(20, 20, 40, 0.7) !important;
        border-left: 14px solid transparent !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin: 20px 0 !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
        backdrop-filter: blur(5px);
    }
    .big-font {
        color: #00FFFF !important;
        text-shadow: 0 0 10px #00FFFF;
    }
    .property-label {
        color: #00FFFF !important;
    }
    .rad-stable {border-left-color: #00FFFF !important; box-shadow: 0 0 15px #00FFFF;}
    .rad-primordial {border-left-color: #00FF99 !important; box-shadow: 0 0 15px #00FF99;}
    .rad-synthetic {border-left-color: #FF3333 !important; box-shadow: 0 0 15px #FF3333;}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🧪 Prerna's Periodic Table Explorer")

# --- SEARCH BAR (centered, wider for neon look) ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    search = st.text_input("", placeholder="🔍 Search by Name or Atomic Number (e.g., Neon or 10)")

highlighted_atomic = None
if search:
    result = df[
        df['Name'].str.contains(search, case=False, na=False) |
        df['Atomic Number'].astype(str).str.contains(search)
    ].head(1)
    if not result.empty:
        el = result.iloc[0]
        highlighted_atomic = int(el['Atomic Number'])
        
        an = int(el['Atomic Number'])
        rad_value = str(el['Radioactivity']).strip().lower()
        if rad_value == "no":
            rad_class = "rad-stable"
            warning = ""
        elif rad_value == "primordial":
            rad_class = "rad-primordial"
            warning = "☢️ Primordial – occurs naturally (but still radioactive)"
        else:
            rad_class = "rad-synthetic"
            warning = "☢️ Radioactive – synthetic or trace only"
        
        st.markdown(f'<div class="element-card {rad_class}">', unsafe_allow_html=True)
        if warning:
            st.markdown(f"<p class='big-font'>{el['Symbol']} – {el['Name']} <span style='color:#FF3333;text-shadow:0 0 10px #FF3333;font-weight:bold;'>{warning}</span></p>", unsafe_allow_html=True)
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
    else:
        st.markdown("<p class='neon-text'>😔 No element found – try another search!</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='neon-text'>🔍 Search for an element above to reveal its secrets!</p>", unsafe_allow_html=True)

st.markdown('<p class="footer">Built by Prerna Lotlikar.</p>', unsafe_allow_html=True)
