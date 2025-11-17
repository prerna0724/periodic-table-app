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

    # Map CSV names to clean names
    column_map = {
        'Name': 'Name',
        'Symbol': 'Symbol',
        'Atomic_Number': 'Atomic Number',
        'Atomic_Weight': 'Atomic Weight',
        'Melting_Point': 'Melting Point',
        'Boiling_Point': 'Boiling Point',
        'Density (kg/m³)': 'Density',
        'Electronegativity': 'Electronegativity',
        'Block': 'Block',
        'Phase': 'Phase'
    }
    df = df.rename(columns=column_map)

    # Keep only needed
    needed = ['Name', 'Symbol', 'Atomic Number', 'Atomic Weight', 'Phase',
              'Melting Point', 'Boiling Point', 'Density', 'Electronegativity', 'Block']
    df = df[[col for col in needed if col in df.columns]].copy()

    # Convert numeric
    numeric_cols = ['Atomic Number', 'Atomic Weight', 'Melting Point', 'Boiling Point',
                    'Density', 'Electronegativity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df = load_data()

# --- TITLE ---
st.title("Baby Chemist's Periodic Table Explorer")
st.sidebar.header("Controls")

# --- SEARCH ---
search = st.sidebar.text_input("Search (Name/Symbol/Atomic Number)", "")
if search:
    result = df[
        df['Name'].str.contains(search, case=False, na=False) |
        df['Symbol'].str.contains(search, case=False, na=False) |
        df['Atomic Number'].astype(str).str.contains(search)
    ].head(1)

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
            st.markdown(f"<span class='property-label'>Atomic Number:</span> {int(el.get('Atomic Number', 0))}", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Atomic Weight:</span> {el.get('Atomic Weight', 0):.3f} u", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Phase:</span> {el.get('Phase', 'N/A')}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<span class='property-label'>Melting Point:</span> {el.get('Melting Point','N/A')} K", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Boiling Point:</span> {el.get('Boiling Point','N/A')} K", unsafe_allow_html=True)
            st.markdown(f"<span class='property-label'>Density:</span> {el.get('Density','N/A')} kg/m³", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Not found!")

# --- ONE AND ONLY SELECTBOX ---
prop = st.sidebar.selectbox("Color Heatmap by:", ['Density', 'Melting Point', 'Boiling Point'], key="heatmap_prop")

# --- PERIOD & GROUP ---
def get_period(a): return min((a>2)+(a>10)+(a>18)+(a>36)+(a>54)+(a>86)+1, 7)
def get_group(a): return (a-1)%18 + 1

plot_df = df.copy()
if 'Atomic Number' in plot_df.columns:
    plot_df['Period'] = plot_df['Atomic Number'].apply(get_period)
    plot_df['Group'] = plot_df['Atomic Number'].apply(get_group)

# === 1. DENSITY VS ATOMIC NUMBER ===
st.markdown("---")
st.subheader("1. Density vs Atomic Number")
density_plot_df = df[['Atomic Number', 'Density', 'Phase', 'Atomic Weight', 'Name']].dropna(subset=['Density', 'Atomic Number'])
fig1 = px.scatter(
    density_plot_df,
    x='Atomic Number', y='Density',
    color='Phase',
    size='Atomic Weight',
    hover_name='Name',
    color_discrete_map={'Gas':'lightblue','Solid':'red','Liquid':'green'}
)
fig1.update_layout(height=500, xaxis_title="Atomic Number", yaxis_title="Density (kg/m³)")
st.plotly_chart(fig1, use_container_width=True)

# === 2. ATOMIC WEIGHT VS BLOCK ===
st.subheader("2. Atomic Weight Distribution (by Block)")
weight_plot_df = df[['Atomic Number', 'Atomic Weight', 'Block']].dropna()
fig2 = px.scatter(
    weight_plot_df,
    x='Atomic Number', y='Atomic Weight',
    color='Block',
    hover_data=['Block'],
    color_discrete_map={'s': '#ff9999', 'p': '#66b3ff', 'd': '#99ff99', 'f': '#ffcc99'}
)
fig2.update_traces(marker=dict(size=10))
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# === 3. INTERACTIVE PERIODIC HEATMAP ===
st.subheader("3. Periodic Table Heatmap")

full_grid = pd.DataFrame(index=range(1,8), columns=range(1,19), dtype=float)
for _, r in plot_df.iterrows():
    val = r.get(prop, np.nan)
    try:
        full_grid.loc[r['Period'], r['Group']] = float(val)
    except:
        pass

fig_heatmap = px.imshow(
    full_grid,
    text_auto=True,
    color_continuous_scale="Viridis",
    aspect="auto",
    title=None
)
fig_heatmap.update_layout(
    height=600,
    margin=dict(l=50, r=50, t=80, b=50),
    xaxis_title="Group (1–18)",
    yaxis_title="Period (1–7)",
    xaxis=dict(tickmode='linear', dtick=1),
    yaxis=dict(tickmode='linear', dtick=1)
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# === HEATMAP EXPLANATION ===
with st.expander("How does the Heatmap work? 🤔", expanded=False):
    st.markdown("""
    ### **Periodic Table Heatmap Explained**
    
    - **Rows (Y-axis)** = **Periods** (1–7) → vertical layers  
    - **Columns (X-axis)** = **Groups** (1–18) → left to right  
    - **Each cell** = **one element** (e.g., Carbon = Period 6, Group 14)  
    - **Color** = **value of selected property**  
      - **Yellow/Green** = **High**  
      - **Purple** = **Low**  
    - **Hover** = exact value + element info  
    - **Missing cells** = no data or f-block (Lanthanides/Actinides not shown)
    
    > **Pro Tip:** Change property → see trends pop!
    """)

# === 4. PHASE DISTRIBUTION ===
st.subheader("4. Phase Distribution")
phase_counts = df['Phase'].value_counts().reset_index()
phase_counts.columns = ['Phase', 'Count']
fig4 = px.bar(
    phase_counts, x='Phase', y='Count', color='Phase',
    color_discrete_map={'Solid':'#1f77b4','Gas':'#ff7f0e','Liquid':'#2ca02c'}
)
fig4.update_layout(height=400, showlegend=False)
st.plotly_chart(fig4, use_container_width=True)

# === 5. PCA CLUSTERS ===
st.subheader("5. PCA Clusters (All 118 Elements)")

# --- OPTIONAL INFO EXPANDER ---
with st.expander("PCA Details (Explained Variance & Imputation)", expanded=False):
    st.markdown(f"""
    **Explained Variance:** {pca.explained_variance_ratio_.sum():.1%}  
    → How much of the original data is captured in 2D.  
    *(Higher = better clusters)*
    
    **Imputation:** Missing values filled with **median** (for PCA only).  
    → Original data unchanged.
    """)
    st.caption("Click to hide")

pca_features = ['Atomic Weight', 'Density', 'Melting Point', 'Boiling Point', 'Electronegativity']
available_pca = [f for f in pca_features if f in df.columns]
if len(available_pca) >= 2:
    pca_df = df[available_pca + ['Name', 'Symbol']].copy()
    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(pca_df[available_pca])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    pca = PCA(n_components=2)
    pca_comp = pca.fit_transform(scaled)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_comp)
    result = pca_df.copy()
    result['PCA1'] = pca_comp[:, 0]
    result['PCA2'] = pca_comp[:, 1]
    result['Cluster'] = clusters
    cluster_names = {0: 'Light & Reactive', 1: 'Mid-Weight Metals', 2: 'Dense Transition', 3: 'Heavy & Superheavies'}
    result['Cluster_Name'] = result['Cluster'].map(cluster_names)
    fig_pca = px.scatter(
        result, x='PCA1', y='PCA2', color='Cluster_Name',
        hover_data=['Name', 'Symbol'] + available_pca,
        color_discrete_sequence=px.colors.sequential.Plasma[1:8:2]
    )
    fig_pca.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='white')))
    
# === ANNOTATION ===
    fig_pca.add_annotation(
        text=(
            "<b>Explained Variance:</b> {:.1%}<br>"
            "<b>Imputed with:</b> median (for PCA only)"
        ).format(pca.explained_variance_ratio_.sum()),
        xref="paper", yref="paper",
        x=1.25, y=0.55,  # LOWER, below legend
        xanchor="center",
        showarrow=False,
        font=dict(size=12, color="#1976D2"),
        bgcolor="rgba(227,242,253,0.95)",
        bordercolor="#90CAF9",
        borderwidth=2,
        borderpad=8,
        align="center"
    )
    
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.warning("Not enough data for PCA.")
    
# === 6. FEATURE IMPORTANCE ===
st.subheader("6. Feature Importance for Melting Point")
ml_features = ['Atomic Weight', 'Density', 'Boiling Point', 'Electronegativity']  # ← FIXED
target = 'Melting Point'
ml_df = df[ml_features + [target]].dropna()
if len(ml_df) > 10:
    X = ml_df[ml_features]
    y = ml_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=ml_features).sort_values(ascending=False)
    imp_df = imp.reset_index()
    imp_df.columns = ['Feature', 'Importance']
    fig_imp = px.bar(imp_df, x='Feature', y='Importance', color='Importance', color_continuous_scale='tealgrn')
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption("Built by Prerna Lotlikar.")
