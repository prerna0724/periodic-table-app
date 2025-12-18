import streamlit as st
import pandas as pd

# --- LOAD & CLEAN DATA --- (same as before)
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

# --- SUBTLE NEON STYLING (toned down version) ---
st.markdown("""
<style>
    /* Dark purple gradient with subtle background image */
    .stApp {
        background: linear-gradient(135deg, #2a1a4f, #120528, #1a0033);
        background-image: url('https://media.istockphoto.com/id/1151082661/vector/retro-sci-fi-background-futuristic-landscape-of-the-80s-digital-cyber-surface-suitable-for.jpg?s=612x612&w=0&k=20&c=4HbMZEmxF08zcS_NgSXDKBJXsWSZTAXRKuC1UNvlOQY=');
        background-size: cover;
        background-attachment: fixed;
        background-blend-mode: overlay;
    }
    
    /* Softer glowing title */
    h1 {
        text-align: center;
        font-size: 3.5rem !important;
        color: #00e6e6 !important;
        text-shadow: 0 0 8px #00ffff, 0 0 15px rgba(0, 255, 255, 0.4);
        margin-bottom: 2rem !important;
    }
    
    /* Subtle search bar */
    div[data-baseweb="input"] > div {
        background: rgba(30, 20, 60, 0.6) !important;
        border: 1px solid #00cccc !important;
        border-radius: 30px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.3) !important;
        backdrop-filter: blur(8px);
    }
    input {
        color: #e0ffff !important;
        font-size: 1.2rem !important;
        text-align: center;
    }
    ::placeholder {
        color: #88dddd !important;
    }
    
    /* Gentle prompt text */
    .neon-text {
        text-align: center;
        color: #00cccc;
        text-shadow: 0 0 6px #00ffff;
        font-size: 1.3rem;
        margin: 1.5rem 0;
    }
    
    /* Footer subtle */
    .footer {
        text-align: center;
        color: #bbbbbb;
        font-size: 0.9rem;
        margin-top: 3rem;
        text-shadow: 0 0 4px #00cccc;
    }
    
    /* Element card - elegant and subdued */
    .element-card {
        background: rgba(20, 15, 50, 0.7) !important;
        border-left: 8px solid transparent !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin: 20px 0 !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2) !important;
        backdrop-filter: blur(6px);
    }
    .big-font {
        color: #00e6e6 !important;
        text-shadow: 0 0 6px #00ffff;
    }
    .property-label {
        color: #00cccc !important;
    }
    .rad-stable {border-left-color: #00cccc !important; box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);}
    .rad-primordial {border-left-color: #00ffaa !important; box-shadow: 0 0 10px rgba(0, 255, 170, 0.3);}
    .rad-synthetic {border-left-color: #ff6666 !important; box-shadow: 0 0 10px rgba(255, 102, 102, 0.3);}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🧪 Prerna's Periodic Table Explorer")

# --- SEARCH BAR (centered) ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    search = st.text_input("", placeholder="🔍 Search by Name or Atomic Number (e.g., Neon or 10)")

# --- REST OF YOUR LOGIC (unchanged from last version) ---
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
            st.markdown(f"<p class='big-font'>{el['Symbol']} – {el['Name']} <span style='color:#ff6666;text-shadow:0 0 6px #ff6666;font-weight:bold;'>{warning}</span></p>", unsafe_allow_html=True)
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
