import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import StringIO
import time
from scipy.stats import gaussian_kde

# Set page config
st.set_page_config(
    page_title="Soybean Area Timelapse",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Empirical CDF Evolution: Log10(Area Hectares) (2001-2015)")
st.markdown("*Track how probability levels correspond to area values over time*")
st.markdown("---")

# ==========================================
# 🔗 DATA SOURCE CONFIGURATION
# ==========================================
# Replace with your actual Google Drive or GitHub link
DATA_URL = "https://drive.google.com/file/d/1Uo3eE_DscZSjqlkuJV8YvNs2a0K1v1WR/view?usp=drive_link"

# For GitHub raw file, use format like:
# DATA_URL = "https://raw.githubusercontent.com/username/repository/main/reduced_soybean_data.csv"

# Data info
DATA_INFO = {
    "title": "Reduced Soybean Patches Dataset",
    "description": "Stratified sample of soybean patch areas (2001-2015)",
    "source": "Agricultural Analysis - Reduced Dataset",
    "total_size": "~20MB"
}
# ==========================================

@st.cache_data
def load_data_from_url(url):
    """Load the reduced dataset"""
    try:
        # Handle Google Drive links
        if 'drive.google.com' in url and '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        else:
            download_url = url

        response = requests.get(download_url)
        if response.status_code == 200:
            # Check if we got HTML (virus warning) instead of CSV
            content = response.text
            if content.strip().startswith('<!DOCTYPE html') or '<html>' in content.lower():
                st.error("❌ Got HTML response - file may be too large or not publicly accessible")
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
df = None

# Try to load from configured URL
if DATA_URL and "YOUR_FILE_ID" not in DATA_URL:
    with st.spinner("🔄 Loading reduced dataset..."):
        df = load_data_from_url(DATA_URL)

# If no data loaded, show file uploader
if df is None:
    st.sidebar.header("📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your reduced_soybean_data.csv", 
        type=['csv'],
        help="Upload the reduced CSV file created by the reduction script"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {str(e)}")

# Main app logic
if df is not None:
    # Verify required columns
    if 'log10_area_hectares' not in df.columns:
        if 'area_hectares' in df.columns:
            # Create log10 column if it doesn't exist
            df = df[df['area_hectares'] > 0].copy()
            df['log10_area_hectares'] = np.log10(df['area_hectares'])
            st.info("ℹ️ Created log10_area_hectares column from area_hectares")
        else:
            st.error("❌ Neither 'log10_area_hectares' nor 'area_hectares' column found!")
            st.stop()

    if 'year' not in df.columns:
        st.error("❌ 'year' column not found!")
        st.stop()

    # Filter data to target years
    df_filtered = df[(df['year'] >= 2001) & (df['year'] <= 2015)].copy()

    if len(df_filtered) == 0:
        st.error("❌ No data found for years 2001-2015!")
        st.stop()

    st.success(f"✅ Data loaded: {len(df_filtered):,} observations ({len(df_filtered['year'].unique())} years)")

    # Sidebar controls
    st.sidebar.header("🎛️ Animation Controls")

    animation_speed = st.sidebar.slider(
        "Animation Speed (seconds per year)", 
        min_value=0.3, max_value=3.0, value=0.8, step=0.1
    )

    n_bins = st.sidebar.slider(
        "Number of Histogram Bins", 
        min_value=20, max_value=100, value=50, step=5
    )

    show_probability_lines = st.sidebar.checkbox("Show Probability Lines", value=True)

    show_values_on_lines = st.sidebar.checkbox("Show Values on Lines", value=True)

    # Year range selector
    available_years = sorted(df_filtered['year'].unique())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(min(available_years)),
        max_value=int(max(available_years)),
        value=(int(min(available_years)), int(max(available_years))),
        step=1
    )

    selected_years = [y for y in available_years if year_range[0] <= y <= year_range[1]]

    # Filter data by selected years
    df_years = df_filtered[df_filtered['year'].isin(selected_years)].copy()

    # Helper functions
    def calculate_empirical_cdf(data):
        """Calculate empirical CDF"""
        if len(data) == 0:
            return np.array([]), np.array([])

        sorted_data = np.sort(data)
        n = len(data)
        # Use (i+1)/n to avoid 0 and include 1
        y_values = np.arange(1, n + 1) / n

        return sorted_data, y_values

    def get_probability_values(data, probabilities=[0.05, 0.25, 0.50, 0.75, 0.90]):
        """Get x-axis values for specific probability levels"""
        if len(data) == 0:
            return {}

        return {prob: np.percentile(data, prob * 100) for prob in probabilities}

    # Layout
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("📊 Dataset Info")

        with st.expander("ℹ️ About this data"):
            st.markdown(f"""
            **{DATA_INFO['title']}**

            {DATA_INFO['description']}

            - **Years:** {min(selected_years)} - {max(selected_years)}
            - **Total observations:** {len(df_years):,}
            - **Variables:** Log10(Area Hectares)
            - **Source:** {DATA_INFO['source']}
            """)

        # Year-by-year stats
        st.subheader("📈 Year Summary")

        stats_data = []
        for year in selected_years:
            year_data = df_years[df_years['year'] == year]['log10_area_hectares']
            if len(year_data) > 0:
                stats_data.append({
                    'Year': int(year),
                    'Count': f"{len(year_data):,}",
                    'Mean': f"{year_data.mean():.2f}",
                    'Median': f"{year_data.median():.2f}",
                    'Std': f"{year_data.std():.2f}"
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

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
        st.markdown("*Click Start Animation to see the distribution evolution over time*")

    # Animation logic
    if start_animation and len(selected_years) > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate global ranges for consistent scaling
        all_log_data = df_years['log10_area_hectares'].values
        x_min, x_max = np.min(all_log_data), np.max(all_log_data)
        x_range = x_max - x_min
        x_min -= x_range * 0.05
        x_max += x_range * 0.05

        # Probability line colors
        prob_colors = {
            0.05: '#FF6B6B',   # Red - 5th percentile
            0.25: '#4ECDC4',   # Teal - 25th percentile  
            0.50: '#45B7D1',   # Blue - 50th percentile (median)
            0.75: '#96CEB4',   # Green - 75th percentile
            0.90: '#FFEAA7'    # Yellow - 90th percentile
        }

        for i, year in enumerate(selected_years):
            if stop_animation:
                break

            # Update progress
            progress = (i + 1) / len(selected_years)
            progress_bar.progress(progress)
            status_text.text(f"📅 Year: {int(year)} | Processing {len(df_years[df_years['year'] == year]):,} observations")

            # Get data for current year
            year_data = df_years[df_years['year'] == year]['log10_area_hectares'].values

            if len(year_data) == 0:
                time.sleep(animation_speed)
                continue

            # Create figure
            fig = go.Figure()

            # Calculate and plot empirical CDF
            x_cdf, y_cdf = calculate_empirical_cdf(year_data)

            if len(x_cdf) > 0:
                fig.add_trace(go.Scatter(
                    x=x_cdf,
                    y=y_cdf,
                    mode='lines',
                    name=f'Empirical CDF - {int(year)}',
                    line=dict(color='blue', width=3),
                    hovertemplate='<b>Log10(Area):</b> %{x:.3f}<br><b>Cumulative Probability:</b> %{y:.3f}<extra></extra>'
                ))

            # Add horizontal probability lines
            if show_probability_lines and len(year_data) > 0:
                prob_values = get_probability_values(year_data)

                for prob, x_value in prob_values.items():
                    if not np.isnan(x_value):
                        color = prob_colors.get(prob, 'gray')

                        # Horizontal line at probability level
                        fig.add_hline(
                            y=prob,
                            line=dict(color=color, width=2, dash="dash"),
                            annotation=dict(
                                text=f"P={prob:.0%}" + (f"<br>x={x_value:.3f}" if show_values_on_lines else ""),
                                showarrow=False,
                                x=0.02,
                                xref="paper",
                                bgcolor="white",
                                bordercolor=color,
                                font=dict(size=10, color=color)
                            )
                        )

                        # Vertical line showing corresponding x-value
                        fig.add_vline(
                            x=x_value,
                            line=dict(color=color, width=1, dash="dot"),
                            opacity=0.6
                        )

                        # Add point at intersection
                        fig.add_trace(go.Scatter(
                            x=[x_value],
                            y=[prob],
                            mode='markers',
                            marker=dict(
                                color=color,
                                size=8,
                                symbol='diamond',
                                line=dict(color='white', width=1)
                            ),
                            name=f'P{prob:.0%}',
                            hovertemplate=f'<b>Probability:</b> {prob:.0%}<br><b>Log10(Area):</b> {x_value:.3f}<br><b>Original Area:</b> {10**x_value:.1f} ha<extra></extra>',
                            showlegend=False
                        ))

            # Update layout for CDF
            fig.update_layout(
                title=dict(
                    text=f'<b>Empirical CDF: Log10(Area Hectares) - Year {int(year)}</b><br>' +
                         f'<span style="font-size:14px">Sample Size: {len(year_data):,} observations</span>',
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title="Log10(Area Hectares)",
                    range=[x_min, x_max],
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title="Cumulative Probability",
                    range=[0, 1],
                    showgrid=True,
                    gridcolor='lightgray',
                    tickformat='.0%'
                ),
                height=600,
                showlegend=True,
                template='plotly_white',
                font=dict(size=12),
                hovermode='closest'
            )

            # Display plot
            with plot_placeholder.container():
                st.plotly_chart(fig, use_container_width=True)

            time.sleep(animation_speed)

        # Clear progress
        progress_bar.empty()
        status_text.empty()
        st.success("🎉 Animation completed!")

    # Show overall summary
    st.markdown("---")
    st.subheader("📊 Overall Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Observations",
            f"{len(df_years):,}",
            f"{len(selected_years)} years"
        )

    with col2:
        log_range = df_years['log10_area_hectares']
        st.metric(
            "Log10 Area Range",
            f"{log_range.min():.2f} - {log_range.max():.2f}",
            f"Span: {log_range.max() - log_range.min():.2f}"
        )

    with col3:
        if 'area_hectares' in df_years.columns:
            original_range = df_years['area_hectares']
            st.metric(
                "Original Area Range",
                f"{original_range.min():.1f} - {original_range.max():,.0f} ha",
                f"Ratio: {original_range.max()/original_range.min():,.0f}:1"
            )
        else:
            # Show range in original hectares from log10 values
            log_range = df_years['log10_area_hectares']
            min_ha = 10**log_range.min()
            max_ha = 10**log_range.max()
            st.metric(
                "Area Range (from log10)",
                f"{min_ha:.1f} - {max_ha:,.0f} ha",
                f"Ratio: {max_ha/min_ha:,.0f}:1"
            )

    with col4:
        median_log = df_years['log10_area_hectares'].median()
        st.metric(
            "Median Log10 Area",
            f"{median_log:.3f}",
            f"= {10**median_log:.1f} hectares"
        )

else:
    # Instructions when no data is loaded
    st.info("👆 Please upload your `reduced_soybean_data.csv` file or configure the data URL")

    st.markdown("---")
    st.subheader("📋 Setup Instructions")

    st.markdown("""
    **Step 1:** Make sure you have the reduced CSV file from the data reduction script

    **Step 2:** Either:
    - **Option A:** Upload the file using the sidebar uploader
    - **Option B:** Upload to Google Drive/GitHub and update the `DATA_URL` in the code

    **Step 3:** Click "Start Animation" to see the empirical CDF evolution!

    **What you'll see:**
    - **Blue CDF curve** showing cumulative probability distribution
    - **Horizontal dashed lines** at 5%, 25%, 50%, 75%, 90% probability levels
    - **Vertical dotted lines** showing corresponding x-axis (area) values
    - **Diamond markers** at intersections for easy tracking

    **Expected file format:**
    - Must contain columns: `log10_area_hectares` and `year`
    - Years should be 2001-2015
    - File size should be manageable (~20MB)
    """)

# Footer
st.markdown("---")
st.markdown("🌱 **Soybean Area Distribution Analysis** | Built with Streamlit & Plotly")
