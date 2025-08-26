import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import StringIO
import time

# Set page config
st.set_page_config(
    page_title="Soybean ECDF Evolution",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Empirical CDF Evolution: Soybean Patch Areas (2001-2015)")
st.markdown("*Tracking key probability levels using exact quantiles from millions of observations*")
st.markdown("---")

# ==========================================
# 🔗 DATA SOURCE CONFIGURATION  
# ==========================================
# Replace with your actual Google Drive or GitHub link
DATA_URL = "https://drive.google.com/file/d/1D21y5_AuHRDagXM7hZgkBik71N6QPnZJ/view?usp=sharing"

# For GitHub: "https://raw.githubusercontent.com/username/repo/main/soybean_quantiles.csv"

DATA_INFO = {
    "title": "Soybean Quantile Data",
    "description": "Exact quantiles from full dataset using log1p transformation",
    "transformation": "np.log1p(area_hectares)",
    "source": "Extracted from 3.2GB dataset"
}
# ==========================================

@st.cache_data
def load_quantile_data(url):
    """Load the quantile data"""
    try:
        # Handle Google Drive links
        if 'drive.google.com' in url and '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        else:
            download_url = url

        response = requests.get(download_url)
        if response.status_code == 200:
            content = response.text
            if content.strip().startswith('<!DOCTYPE html'):
                st.error("❌ Got HTML response - check if file is publicly accessible")
                return None

            df = pd.read_csv(StringIO(content))
            return df
        else:
            st.error(f"❌ Failed to load data. Status code: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load data
df_quantiles = None

# Try to load from URL
if DATA_URL and "YOUR_FILE_ID" not in DATA_URL:
    with st.spinner("🔄 Loading quantile data..."):
        df_quantiles = load_quantile_data(DATA_URL)

# Fallback to file upload
if df_quantiles is None:
    st.sidebar.header("📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload soybean_quantiles.csv", 
        type=['csv'],
        help="Upload the quantile CSV file created by the extraction script"
    )

    if uploaded_file is not None:
        try:
            df_quantiles = pd.read_csv(uploaded_file)
            st.success("✅ Quantile data uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Main app logic
if df_quantiles is not None:
    # Filter out special records (like mean with percentile = -1)
    df_clean = df_quantiles[df_quantiles['percentile'] > 0].copy()

    # Verify data structure
    required_cols = ['year', 'percentile', 'log1p_area_value']
    missing_cols = [col for col in required_cols if col not in df_clean.columns]

    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.write("Available columns:", list(df_clean.columns))
        st.stop()

    # Filter years
    available_years = sorted(df_clean['year'].unique())
    years_2001_2015 = [y for y in available_years if 2001 <= y <= 2015]

    if len(years_2001_2015) == 0:
        st.error("❌ No data found for years 2001-2015!")
        st.stop()

    st.success(f"✅ Loaded quantile data: {len(years_2001_2015)} years, {len(df_clean):,} quantile points")

    # Sidebar controls
    st.sidebar.header("🎛️ Animation Controls")

    animation_speed = st.sidebar.slider(
        "Animation Speed (seconds per year)", 
        min_value=0.5, max_value=3.0, value=1.0, step=0.1
    )

    # Focus options
    focus_mode = st.sidebar.radio(
        "Focus Level:",
        ["Key Quartiles (25%, 50%, 75%)", "Extended View (5%, 25%, 50%, 75%, 95%)", "Full Detail (All percentiles)"]
    )

    show_values = st.sidebar.checkbox("Show values on markers", value=True)
    show_grid = st.sidebar.checkbox("Show grid", value=False)

    # Year range
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(min(years_2001_2015)),
        max_value=int(max(years_2001_2015)), 
        value=(int(min(years_2001_2015)), int(max(years_2001_2015))),
        step=1
    )

    selected_years = [y for y in years_2001_2015 if year_range[0] <= y <= year_range[1]]

    # Helper functions
    def get_focus_percentiles():
        """Get percentiles based on focus mode"""
        if focus_mode == "Key Quartiles (25%, 50%, 75%)":
            return [25, 50, 75]
        elif focus_mode == "Extended View (5%, 25%, 50%, 75%, 95%)":
            return [5, 25, 50, 75, 95]
        else:
            return sorted(df_clean['percentile'].unique())

    def get_percentile_colors():
        """Get colors for percentiles"""
        focus_percentiles = get_focus_percentiles()

        if len(focus_percentiles) == 3:
            return {25: '#FF6B6B', 50: '#45B7D1', 75: '#96CEB4'}
        elif len(focus_percentiles) == 5:
            return {5: '#FF6B6B', 25: '#4ECDC4', 50: '#45B7D1', 75: '#96CEB4', 95: '#FFEAA7'}
        else:
            # Generate colors for all percentiles
            import plotly.colors as pc
            colors = pc.qualitative.Set3
            color_map = {}
            for i, p in enumerate(focus_percentiles):
                color_map[p] = colors[i % len(colors)]
            return color_map

    def create_smooth_cdf(year_data):
        """Create smooth CDF from quantile data"""
        if len(year_data) == 0:
            return np.array([]), np.array([])

        # Sort by log1p_area_value
        year_data_sorted = year_data.sort_values('log1p_area_value')
        x_values = year_data_sorted['log1p_area_value'].values
        y_values = year_data_sorted['percentile'].values / 100.0  # Convert to probabilities

        return x_values, y_values

    # Layout
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("📊 Dataset Info")

        with st.expander("ℹ️ About this data"):
            st.markdown(f"""
            **{DATA_INFO['title']}**

            {DATA_INFO['description']}

            - **Transformation:** `{DATA_INFO['transformation']}`
            - **Years:** {min(selected_years)} - {max(selected_years)}
            - **Focus:** {focus_mode}
            - **Source:** {DATA_INFO['source']}
            """)

        # Show key statistics for current focus
        st.subheader("📈 Key Values")

        focus_percentiles = get_focus_percentiles()

        # Show values for latest year as reference
        if len(selected_years) > 0:
            latest_year = max(selected_years)
            latest_data = df_clean[
                (df_clean['year'] == latest_year) & 
                (df_clean['percentile'].isin(focus_percentiles))
            ]

            if len(latest_data) > 0:
                st.write(f"**{latest_year} Values:**")
                for _, row in latest_data.iterrows():
                    percentile = int(row['percentile'])
                    value = row['log1p_area_value']
                    original_area = np.exp(value) - 1  # Convert back from log1p

                    if original_area < 1:
                        area_text = f"{original_area:.3f} ha"
                    elif original_area < 10:
                        area_text = f"{original_area:.2f} ha"
                    else:
                        area_text = f"{original_area:.1f} ha"

                    st.write(f"• **P{percentile}%**: {area_text}")

    with col1:
        # Main plot placeholder
        plot_placeholder = st.empty()

    # Animation controls
    col_start, col_stop, col_info = st.columns([1, 1, 2])

    with col_start:
        start_animation = st.button("▶️ Start Animation", type="primary")

    with col_stop:
        stop_animation = st.button("⏹️ Stop")

    with col_info:
        st.markdown("*Watch how patch size thresholds change over time*")

    # Animation logic
    if start_animation and len(selected_years) > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Get focus data
        focus_percentiles = get_focus_percentiles()
        percentile_colors = get_percentile_colors()

        # Calculate global ranges
        focus_data = df_clean[
            (df_clean['year'].isin(selected_years)) &
            (df_clean['percentile'].isin(focus_percentiles))
        ]

        x_min = focus_data['log1p_area_value'].min() - 0.1
        x_max = focus_data['log1p_area_value'].max() + 0.1

        for i, year in enumerate(selected_years):
            if stop_animation:
                break

            # Update progress
            progress = (i + 1) / len(selected_years)
            progress_bar.progress(progress)
            status_text.text(f"📅 Year: {int(year)}")

            # Get data for current year
            year_data = df_clean[df_clean['year'] == year]

            if len(year_data) == 0:
                time.sleep(animation_speed)
                continue

            # Create figure
            fig = go.Figure()

            # Create smooth CDF
            x_cdf, y_cdf = create_smooth_cdf(year_data)

            if len(x_cdf) > 0:
                fig.add_trace(go.Scatter(
                    x=x_cdf,
                    y=y_cdf,
                    mode='lines',
                    name='Empirical CDF',
                    line=dict(color='lightblue', width=2),
                    hovertemplate='<b>Log1p(Area):</b> %{x:.3f}<br><b>Cumulative Prob:</b> %{y:.1%}<extra></extra>',
                    showlegend=True
                ))

            # Add focus percentile points
            focus_year_data = year_data[year_data['percentile'].isin(focus_percentiles)]

            for _, row in focus_year_data.iterrows():
                percentile = int(row['percentile'])
                x_val = row['log1p_area_value']
                y_val = percentile / 100.0
                color = percentile_colors.get(percentile, 'gray')

                # Add horizontal line
                fig.add_shape(
                    type="line",
                    x0=x_min, x1=x_max,
                    y0=y_val, y1=y_val,
                    line=dict(color=color, width=2, dash="dash"),
                )

                # Add marker
                fig.add_trace(go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=12,
                        symbol='diamond',
                        line=dict(color='white', width=2)
                    ),
                    name=f'{percentile}%',
                    hovertemplate=f'<b>{percentile}%:</b> {np.exp(x_val)-1:.2f} hectares<br><b>Log1p value:</b> {x_val:.3f}<extra></extra>',
                    showlegend=True
                ))

                # Add value annotation
                if show_values:
                    original_area = np.exp(x_val) - 1
                    if original_area < 1:
                        area_text = f"{original_area:.3f}ha"
                    else:
                        area_text = f"{original_area:.1f}ha"

                    fig.add_annotation(
                        x=x_val,
                        y=y_val + 0.05,
                        text=f"<b>{percentile}%</b><br>{area_text}",
                        showarrow=False,
                        bgcolor=color,
                        font=dict(color='white', size=10),
                        bordercolor='white',
                        borderwidth=1
                    )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'<b>Empirical CDF: Soybean Patch Areas - Year {int(year)}</b><br>' +
                         f'<span style="font-size:14px">{focus_mode}</span>',
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title="Log1p(Area Hectares)",
                    range=[x_min, x_max],
                    showgrid=show_grid,
                    gridcolor='lightgray' if show_grid else None,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Cumulative Probability",
                    range=[0, 1],
                    showgrid=show_grid,
                    gridcolor='lightgray' if show_grid else None,
                    tickformat='.0%',
                    zeroline=False
                ),
                height=600,
                showlegend=True,
                template='plotly_white',
                font=dict(size=12)
            )

            # Display
            with plot_placeholder.container():
                st.plotly_chart(fig, use_container_width=True)

            time.sleep(animation_speed)

        # Clean up
        progress_bar.empty()
        status_text.empty()
        st.success("🎉 Animation completed!")

    # Summary
    st.markdown("---")
    st.subheader("📊 What This Shows")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🎯 Focus Areas:**
        - **25%**: Quarter of patches smaller
        - **50%**: Median patch size
        - **75%**: Three-quarters smaller
        """)

    with col2:
        st.markdown("""
        **📈 What to Watch:**
        - Lines moving **right** = patches getting larger
        - Lines moving **left** = patches getting smaller
        - Lines **spreading** = more size diversity
        """)

    with col3:
        st.markdown("""
        **🌱 Agricultural Insights:**
        - **Consolidation** trends over time
        - **Policy** impact years
        - **Technology** adoption effects
        """)

else:
    st.info("👆 Upload your quantile CSV file or configure the data URL to start")

    st.markdown("---")
    st.subheader("📋 Setup Instructions")

    st.markdown("""
    **Step 1:** Run the quantile extraction script to create `soybean_quantiles.csv`

    **Step 2:** Either:
    - **Upload the file** using the sidebar uploader
    - **Host online** (Google Drive/GitHub) and update the `DATA_URL`

    **Step 3:** Choose your focus level and start the animation!

    **File format expected:**
    - Columns: `year`, `percentile`, `log1p_area_value`
    - Years: 2001-2015
    - Percentiles: 1-99
    """)

# Footer
st.markdown("---")
st.markdown("🌱 **Soybean Patch Size Evolution** | Built with exact quantiles from millions of observations")
