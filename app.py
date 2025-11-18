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
st.title("🧪 Prerna's Periodic Table Explorer")
st.sidebar.header("Controls")

# --- SEARCH + HIGHLIGHT LOGIC ---
search = st.sidebar.text_input("Search (Name/Symbol/Atomic Number)", "")
highlighted_atomic = None

if search:
    result = df[
        df['Name'].str.contains(search, case=False, na=False) |
        df['Symbol'].str.contains(search, case=False, na=False) |
        df['Atomic Number'].astype(str).str.contains(search)
    ].head(1)

    if not result.empty:
        el = result.iloc[0]
        highlighted_atomic = int(el['Atomic Number'])

        # === ELEMENT CARD WITH PRIMORDIAL SUPPORT ===
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
            .rad-stable     {border-left-color: #00CCFF;}
            .rad-primordial {border-left-color: #00FF99;}
            .rad-synthetic  {border-left-color: #FF3333;}
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

# --- HEATMAP PROPERTY ---
prop = st.sidebar.selectbox("Color Heatmap by:", ['Density', 'Melting Point', 'Boiling Point', 'Radioactivity'], key="heatmap_prop")

# --- PERIOD & GROUP ---
def get_period(a): return min((a>2)+(a>10)+(a>18)+(a>36)+(a>54)+(a>86)+1, 7)
def get_group(a): return (a-1)%18 + 1

plot_df = df.copy()
plot_df['Period'] = plot_df['Atomic Number'].apply(get_period)
plot_df['Group'] = plot_df['Atomic Number'].apply(get_group)

# === 1. DENSITY VS ATOMIC NUMBER ===
st.markdown("---")
st.subheader("1. Density vs Atomic Number")
density_df = df[['Atomic Number', 'Density', 'Phase', 'Atomic Weight', 'Name', 'Symbol', 'Radioactivity']].dropna(subset=['Density'])
fig1 = px.scatter(density_df, x='Atomic Number', y='Density', color='Phase', size='Atomic Weight',
                  hover_data=['Radioactivity'], hover_name='Name',
                  color_discrete_map={'Gas':'lightblue','Solid':'#d62728','Liquid':'green'})
if highlighted_atomic:
    h = density_df[density_df['Atomic Number'] == highlighted_atomic]
    if not h.empty:
        fig1.add_scatter(
            x=h['Atomic Number'],
            y=h['Density'],
            mode="markers",  # ← no "+text" = no floating label
            marker=dict(size=50, color="#FFFF00", line=dict(width=5, color="black")),
            showlegend=False
            )
            
fig1.update_layout(height=500)
st.plotly_chart(fig1, use_container_width=True)

# === 2. ATOMIC WEIGHT VS BLOCK ===
st.subheader("2. Atomic Weight Distribution (by Block)")
weight_df = df[['Atomic Number', 'Atomic Weight', 'Block', 'Symbol']].dropna()
fig2 = px.scatter(weight_df, x='Atomic Number', y='Atomic Weight', color='Block',
                  color_discrete_map={'s': '#ff9999', 'p': '#66b3ff', 'd': '#99ff99', 'f': '#ffcc99'})
if highlighted_atomic:
    h = weight_df[weight_df['Atomic Number'] == highlighted_atomic]
    if not h.empty:
        fig2.add_scatter(x=h['Atomic Number'], y=h['Atomic Weight'], mode="markers+text",
                         marker=dict(size=30, color="#FFFF00", line=dict(width=3, color="black")),
                         showlegend=False
                        )
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# === 3. CORRELATION HEATMAP OF ELEMENT PROPERTIES ===
st.subheader("Correlation Heatmap of Numerical Properties")

# Clean numeric columns that might have "Unknown" or garbage
numeric_cols_for_corr = [
    'Atomic Number', 'Atomic Weight', 'Density', 'Melting Point', 'Boiling Point',
    'Electronegativity'  # add any others you have that should be numeric
]

for col in numeric_cols_for_corr:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace inf with NaN and drop useless columns
df.replace([np.inf, -np.inf], np.nan, inplace=True)
corr_df = df[numeric_cols_for_corr].copy()
corr_df.dropna(axis=1, how='all', inplace=True)

if corr_df.shape[1] < 2:
    st.warning("Not enough numeric data for correlation heatmap 😢")
else:
    corr = corr_df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    aspect="auto"
)

fig.update_traces(
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: <b>%{z:.3f}</b><extra></extra>"
)

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    coloraxis_colorbar=dict(
        title="Correlation (r)",
        titleside="top",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1.0", "-0.5", "0.0", "+0.5", "+1.0"],
        len=0.7,
        lenmode='fraction',  # ← THIS LINE IS THE HERO
        thickness=20,
        x=1.02
    ),
    title=dict(
        text="<b>Correlation Heatmap of Element Properties</b><br><sup>Hover for exact values • ★ marks |r| ≥ 0.8 strong correlation</sup>",
        font=dict(size=22),
        x=0.5,
        xanchor="center"
    ),
    margin=dict(t=120, b=100, l=100, r=200),
    height=800
)

fig.add_annotation(
    text="★ = |r| ≥ 0.8 (strong correlation)",
    xref="paper", yref="paper",
    x=0.5, y=-0.22, showarrow=False,
    font=dict(size=11, color="#D32F2F")
)

st.plotly_chart(fig, use_container_width=True)

# Explanation below the plot (perfect order)
with st.expander("How does the correlation heatmap work? 🤔", expanded=False):
    st.markdown("""
    - Shows how strongly different numerical properties are related  
    - Red = strong positive correlation, Blue = strong negative  
    - Diagonal is always 1.0 (property correlated with itself)  
    - ★ marks very strong correlations (|r| ≥ 0.8)
    """)

# === 4. PHASE DISTRIBUTION ===
st.subheader("4. Phase Distribution")
phase_counts = df['Phase'].value_counts().reset_index()
phase_counts.columns = ['Phase', 'Count']
fig4 = px.bar(phase_counts, x='Phase', y='Count', color='Phase',
              color_discrete_map={'Solid':'#1f77b4','Gas':'#ff7f0e','Liquid':'#2ca02c'})
if highlighted_atomic:
    phase = df[df['Atomic Number'] == highlighted_atomic]['Phase'].iloc[0]
    if phase in phase_counts['Phase'].values:
        fig4.add_bar(x=[phase], y=[phase_counts[phase_counts['Phase']==phase]['Count'].iloc[0]],
                     marker_color="#FFFF00", marker_line=dict(color="black", width=8), showlegend=False)
fig4.update_layout(height=400, showlegend=False)
st.plotly_chart(fig4, use_container_width=True)

# === 5. PCA CLUSTERS (now properly colored by No/Primordial/Yes) ===
st.subheader("5. PCA Clusters (All 118 Elements)")
pca_features = ['Atomic Weight', 'Density', 'Melting Point', 'Boiling Point', 'Electronegativity']
available_pca = [f for f in pca_features if f in df.columns]

explained_var = 0.0
imputation_note = "Not computed"

if len(available_pca) >= 2:
    pca_df = df[available_pca + ['Name', 'Symbol', 'Radioactivity']].copy()
    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(pca_df[available_pca])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    pca = PCA(n_components=2)
    pca_comp = pca.fit_transform(scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    imputation_note = "median (for PCA only)"

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_comp)
    result = pca_df.copy()
    result['PCA1'] = pca_comp[:, 0]
    result['PCA2'] = pca_comp[:, 1]
    result['Cluster'] = clusters

    fig_pca = px.scatter(result, x='PCA1', y='PCA2', color='Radioactivity',
                         hover_data=['Name', 'Symbol'] + available_pca,
                         color_discrete_map={
                             'No': '#00FF00',
                             'Primordial': '#FFFF00',
                             'Yes': '#FF0000'
                         },
                         title="Green = Stable | Yellow = Primordial | Red = Synthetic Radioactive")
    fig_pca.update_traces(marker=dict(size=12, opacity=0.9))

    if highlighted_atomic:
        sym = df[df['Atomic Number'] == highlighted_atomic]['Symbol'].iloc[0]
        h = result[result['Symbol'] == sym]
        if not h.empty:
            fig_pca.add_scatter(x=h['PCA1'], y=h['PCA2'], mode="markers+text",
                                marker=dict(size=40, color="#FFFF00", line=dict(width=4, color="black"))
                               )
    st.plotly_chart(fig_pca, use_container_width=True)

with st.expander("PCA Details", expanded=False):
    st.markdown(f"**Explained Variance:** {explained_var:.1%}\n**Imputation:** {imputation_note}")

# === 6. FEATURE IMPORTANCE ===
st.subheader("6. Feature Importance for Melting Point")
ml_features = ['Atomic Weight', 'Density', 'Boiling Point', 'Electronegativity']
target = 'Melting Point'
ml_df = df[ml_features + [target]].dropna()
if len(ml_df) > 10:
    X, y = ml_df[ml_features], ml_df[target]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=ml_features).sort_values(ascending=False)
    fig_imp = px.bar(imp.reset_index(), x='index', y=0, color=0, color_continuous_scale='tealgrn',
                     labels={'index':'Feature', '0':'Importance'})
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption("Built by Prerna Lotlikar")
