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

# --- CLEAN FUTURISTIC STYLING ---
st.markdown("""
<style>
    /* Subtle dark space background */
    .stApp {
        background-image: url('https://thumbs.dreamstime.com/b/glowing-particles-floating-space-dark-purple-gradient-background-low-contrast-minimal-glow-389369383.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* Clean white title */
    h1 {
        text-align: center;
        font-size: 3.2rem !important;
        color: #FFFFFF !important;
        margin-bottom: 3rem !important;
        font-weight: 600;
    }
    
    /* Frosted glass search bar with icon and soft glow */
    div[data-baseweb="input"] > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        padding-left: 20px;
    }
    input {
        color: #FFFFFF !important;
        font-size: 1.2rem !important;
        text-align: center;
        background: transparent !important;
    }
    ::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    /* Magnifying glass icon */
    div[data-baseweb="input"]::before {
        content: "🔍 ";
        color: rgba(255, 255, 255, 0.8);
        position: absolute;
        left: 30px;
        top: 50%;
        transform: translateY(-50%);
    }
    
    /* Prompt text */
    .prompt-text {
        text-align: center;
        color: #00FFFF;
        font-size: 1.3rem;
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #AAAAAA;
        font-size: 0.9rem;
        margin-top: 4rem;
    }
    
    /* Element card - clean and readable */
    .element-card {
        background: rgba(20, 20, 50, 0.6) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        margin: 30px 0 !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        border-left: 8px solid transparent;
    }
    .big-font {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: bold;
    }
    .warning {
        color: #FF6666 !important;
        font-weight: bold;
    }
    .property-label {
        color: #00FFFF !important;
        font-weight: bold;
    }
    .property-value {
        color: #FFFFFF !important;
        font-size: 1.1rem;
    }
    .rad-stable {border-left-color: #00FFFF;}
    .rad-primordial {border-left-color: #00FFAA;}
    .rad-synthetic {border-left-color: #FF6666;}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🧪 Prerna's Periodic Table Explorer")

# --- SEARCH BAR ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<div style="position:relative;"><span class="search-icon">🔍</span></div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Search by Name or Atomic Number", label_visibility="collapsed")

# --- ELEMENT DISPLAY LOGIC ---
if search:
    result = df[
        df['Name'].str.contains(search, case=False, na=False) |
        df['Atomic Number'].astype(str).str.contains(search)
    ].head(1)
    if not result.empty:
        el = result.iloc[0]
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
        
        # Header with symbol, name, warning
        warning_html = f'<span class="warning">{warning}</span>' if warning else ""
        st.markdown(f"<p class='big-font'>{el['Symbol']} – {el['Name']} {warning_html}</p>", unsafe_allow_html=True)
        
        # Properties in two columns
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<span class='property-label'>Atomic Number:</span> <span class='property-value'>{an}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Atomic Weight:</span> <span class='property-value'>{el.get('Atomic Weight',0):.4f} u</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Phase:</span> <span class='property-value'>{el.get('Phase','N/A')}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Radioactivity:</span> <span class='property-value'>{el['Radioactivity']}</span>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<span class='property-label'>Melting Point:</span> <span class='property-value'>{el.get('Melting Point','N/A')} K</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Boiling Point:</span> <span class='property-value'>{el.get('Boiling Point','N/A')} K</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Density:</span> <span class='property-value'>{el.get('Density','N/A')} kg/m³</span>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<p class='prompt-text'>😔 No element found – try another search!</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='prompt-text'>🔍 Search for an element above to see its details!</p>", unsafe_allow_html=True)

st.markdown('<p class="footer">Built by Prerna Lotlikar.</p>', unsafe_allow_html=True)
