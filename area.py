import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import time
from scipy import stats
from scipy.stats import gaussian_kde

# Set page config
st.set_page_config(
    page_title="Area Hectares PDF Evolution",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Log10(Area Hectares) Empirical PDF Evolution (2001-2015)")
st.markdown("---")

# Load data function
@st.cache_data
def load_data_from_url(url):
    """Load data from Google Drive or other cloud storage"""
    try:
        # Handle Google Drive links
        if 'drive.google.com' in url and '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        else:
            download_url = url

        response = requests.get(download_url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            st.error(f"Failed to load data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Sidebar for data input
st.sidebar.header("📊 Data Input")
st.sidebar.markdown("**Enter your Google Drive sharing link:**")

# Instructions for Google Drive
with st.sidebar.expander("📋 How to get Google Drive link"):
    st.markdown("""
    1. Upload your CSV to Google Drive
    2. Right-click → Share → "Anyone with the link"
    3. Copy the sharing link
    4. Paste it below (the app will handle the conversion)
    """)

# URL input
default_url = st.sidebar.text_input(
    "Google Drive Link:",
    placeholder="https://drive.google.com/file/d/1CWjIyD_adoGUeiX2_8-r42pwDsyAFcOR/view?usp=sharing",
    help="Paste your Google Drive sharing link here"
)

# Load data
df = None
if default_url:
    with st.spinner("🔄 Loading data from Google Drive..."):
        df = load_data_from_url(default_url)

if df is not None:
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

    # Show data preview
    with st.expander("👀 Data Preview"):
        st.dataframe(df.head(10))
        st.write("**Columns:**", list(df.columns))

    # Check if required columns exist
    if 'area_hectares' not in df.columns:
        st.error("❌ Column 'area_hectares' not found in the dataset!")
        st.write("Available columns:", list(df.columns))
        st.stop()

    if 'year' not in df.columns:
        st.error("❌ Column 'year' not found in the dataset!")
        st.write("Available columns:", list(df.columns))
        st.stop()

    # Data preprocessing
    st.sidebar.header("⚙️ Data Processing")

    # Filter out zero or negative values for log transformation
    original_count = len(df)
    df_filtered = df[df['area_hectares'] > 0].copy()
    filtered_count = len(df_filtered)

    if filtered_count < original_count:
        st.sidebar.warning(f"⚠️ Removed {original_count - filtered_count} rows with area_hectares ≤ 0 for log transformation")

    # Apply log10 transformation
    df_filtered['log10_area'] = np.log10(df_filtered['area_hectares'])

    # Show transformation info
    st.sidebar.info(f"""
    **Transformation Applied:**
    - Original range: {df_filtered['area_hectares'].min():.2f} to {df_filtered['area_hectares'].max():,.0f} hectares
    - Log10 range: {df_filtered['log10_area'].min():.2f} to {df_filtered['log10_area'].max():.2f}
    """)

    # Year selection
    available_years = sorted(df_filtered['year'].unique())
    st.sidebar.write(f"**Available years:** {min(available_years)} - {max(available_years)}")

    year_range = st.sidebar.slider(
        "Select year range:",
        min_value=int(min(available_years)),
        max_value=int(max(available_years)),
        value=(int(min(available_years)), int(max(available_years))),
        step=1
    )

    selected_years = [year for year in available_years if year_range[0] <= year <= year_range[1]]

    # Animation controls
    st.sidebar.header("🎛️ Animation Controls")
    animation_speed = st.sidebar.slider("Animation Speed (seconds per year)", 0.3, 3.0, 0.8, 0.1)
    kde_bandwidth = st.sidebar.slider("PDF Smoothness", 0.05, 0.3, 0.1, 0.01, help="Lower values = more detailed, higher = smoother")

    # Filter data for selected years
    df_years = df_filtered[df_filtered['year'].isin(selected_years)].copy()

    # Function to calculate empirical PDF using KDE
    def calculate_empirical_pdf(data, bandwidth=0.1, n_points=200):
        """Calculate empirical PDF using Kernel Density Estimation"""
        if len(data) < 2:
            return np.array([]), np.array([])

        kde = gaussian_kde(data, bw_method=bandwidth)
        x_min, x_max = data.min(), data.max()
        x_range = x_max - x_min
        x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, n_points)
        pdf_values = kde(x_eval)

        return x_eval, pdf_values

    # Function to calculate quantiles
    def calculate_quantiles(data):
        """Calculate 5th, 25th, 50th, 75th, and 90th percentiles"""
        if len(data) == 0:
            return {}

        percentiles = [5, 25, 50, 75, 90]
        quantiles = np.percentile(data, percentiles)

        return {
            'percentiles': percentiles,
            'values': quantiles,
            'labels': ['5th', '25th', '50th (Median)', '75th', '90th']
        }

    # Main layout
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("📊 Year-by-Year Stats")

        # Calculate stats for all years
        stats_data = []
        for year in selected_years:
            year_data = df_years[df_years['year'] == year]['log10_area']
            if len(year_data) > 0:
                quantiles = calculate_quantiles(year_data)
                stats_data.append({
                    'Year': int(year),
                    'Count': len(year_data),
                    'Mean': f"{year_data.mean():.2f}",
                    'Std': f"{year_data.std():.2f}",
                    'Min': f"{year_data.min():.2f}",
                    'Max': f"{year_data.max():.2f}",
                    'Median': f"{np.median(year_data):.2f}"
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

    with col1:
        # Create placeholder for animated plot
        plot_placeholder = st.empty()

    # Animation controls
    col_start, col_stop, col_download = st.columns([1, 1, 1])

    with col_start:
        start_animation = st.button("▶️ Start Animation", type="primary")

    with col_stop:
        stop_animation = st.button("⏹️ Stop Animation")

    with col_download:
        # Prepare processed data for download
        download_data = df_years[['year', 'area_hectares', 'log10_area']].copy()
        csv_data = download_data.to_csv(index=False)
        st.download_button(
            label="💾 Download Processed Data",
            data=csv_data,
            file_name="log10_area_hectares_data.csv",
            mime="text/csv"
        )

    # Animation logic
    if start_animation and len(selected_years) > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate global ranges for consistent scaling
        all_log_data = df_years['log10_area'].values
        x_min, x_max = np.min(all_log_data), np.max(all_log_data)
        x_range = x_max - x_min
        x_min -= x_range * 0.05
        x_max += x_range * 0.05

        # Calculate global y-axis range by looking at all PDFs
        max_pdf_value = 0
        for year in selected_years[:3]:  # Check first few years to estimate scale
            year_data = df_years[df_years['year'] == year]['log10_area'].values
            if len(year_data) > 1:
                _, pdf_vals = calculate_empirical_pdf(year_data, kde_bandwidth)
                if len(pdf_vals) > 0:
                    max_pdf_value = max(max_pdf_value, np.max(pdf_vals))

        y_max = max_pdf_value * 1.2 if max_pdf_value > 0 else 1

        # Color palette for quantiles
        quantile_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        for i, year in enumerate(selected_years):
            if stop_animation:
                break

            # Update progress
            progress = (i + 1) / len(selected_years)
            progress_bar.progress(progress)
            status_text.text(f"📅 Showing year: {int(year)} | Observations: {len(df_years[df_years['year'] == year]):,}")

            # Get data for current year
            year_data = df_years[df_years['year'] == year]['log10_area'].values

            if len(year_data) > 1:
                # Calculate empirical PDF
                x_pdf, y_pdf = calculate_empirical_pdf(year_data, kde_bandwidth)

                # Calculate quantiles
                quantiles = calculate_quantiles(year_data)

                # Create the plot
                fig = go.Figure()

                # Add PDF curve
                if len(x_pdf) > 0 and len(y_pdf) > 0:
                    fig.add_trace(go.Scatter(
                        x=x_pdf,
                        y=y_pdf,
                        mode='lines',
                        name=f'Empirical PDF - {int(year)}',
                        line=dict(color='blue', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0,100,255,0.1)',
                        hovertemplate='<b>Log10(Area):</b> %{x:.2f}<br><b>Density:</b> %{y:.4f}<extra></extra>'
                    ))

                # Add quantile lines if we have quantiles
                if 'values' in quantiles and len(quantiles['values']) > 0:
                    for j, (percentile, value, label, color) in enumerate(zip(
                        quantiles['percentiles'], 
                        quantiles['values'], 
                        quantiles['labels'],
                        quantile_colors
                    )):
                        # Vertical line to x-axis
                        fig.add_shape(
                            type="line",
                            x0=value, x1=value,
                            y0=0, y1=y_max,
                            line=dict(color=color, width=2, dash="dash"),
                        )

                        # Add quantile value annotation
                        fig.add_annotation(
                            x=value,
                            y=y_max * 0.95,
                            text=f"{label}<br>{value:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor=color,
                            bgcolor="white",
                            bordercolor=color,
                            borderwidth=1,
                            font=dict(size=10, color=color)
                        )

                # Update layout
                fig.update_layout(
                    title=f'<b>Empirical PDF of Log10(Area Hectares) - Year {int(year)}</b><br>' +
                          f'<span style="font-size:14px">Sample Size: {len(year_data):,} observations | ' +
                          f'Bandwidth: {kde_bandwidth}</span>',
                    xaxis_title='Log10(Area Hectares)',
                    yaxis_title='Probability Density',
                    xaxis=dict(range=[x_min, x_max]),
                    yaxis=dict(range=[0, y_max]),
                    height=600,
                    showlegend=True,
                    hovermode='closest',
                    template='plotly_white',
                    font=dict(size=12)
                )

                # Add grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                # Add secondary x-axis labels for original hectares scale
                original_ticks = np.array([0.1, 1, 10, 100, 1000, 10000, 100000])
                log_ticks = np.log10(original_ticks)

                # Filter ticks that are within our range
                valid_ticks = log_ticks[(log_ticks >= x_min) & (log_ticks <= x_max)]
                valid_original = original_ticks[(log_ticks >= x_min) & (log_ticks <= x_max)]

                fig.update_layout(
                    xaxis=dict(
                        range=[x_min, x_max],
                        tickmode='array',
                        tickvals=valid_ticks,
                        ticktext=[f"{val:g}" if val < 1000 else f"{val/1000:.0f}K" for val in valid_original],
                        title="Log10(Area Hectares)<br><span style='font-size:10px'>Bottom labels show original hectares</span>"
                    )
                )

                # Display the plot
                with plot_placeholder.container():
                    st.plotly_chart(fig, use_container_width=True)

            # Wait for specified duration
            time.sleep(animation_speed)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        st.success(f"🎉 Animation completed! Processed {len(selected_years)} years of data.")

    # Summary statistics
    if len(df_years) > 0:
        st.markdown("---")
        st.subheader("📈 Overall Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Observations",
                f"{len(df_years):,}",
                f"{len(selected_years)} years"
            )

        with col2:
            st.metric(
                "Log10 Area Range",
                f"{df_years['log10_area'].min():.2f} to {df_years['log10_area'].max():.2f}",
                f"Span: {df_years['log10_area'].max() - df_years['log10_area'].min():.2f}"
            )

        with col3:
            st.metric(
                "Original Area Range", 
                f"{df_years['area_hectares'].min():.1f} to {df_years['area_hectares'].max():,.0f} ha",
                f"Ratio: {df_years['area_hectares'].max()/df_years['area_hectares'].min():,.0f}:1"
            )

        with col4:
            median_area = df_years['area_hectares'].median()
            st.metric(
                "Median Area",
                f"{median_area:,.1f} hectares",
                f"Log10: {np.log10(median_area):.2f}"
            )

else:
    st.info("👆 Please enter your Google Drive link in the sidebar to get started!")

    # Show example/demo section
    st.markdown("---")
    st.subheader("📖 About This App")

    st.markdown("""
    This app visualizes the evolution of **empirical Probability Density Functions (PDF)** for log10-transformed area data over time.

    **Key Features:**
    - 🔄 **Automatic log10 transformation** of area_hectares (handles wide range from 0.9 to 100K+ hectares)
    - 📊 **Empirical PDF estimation** using Kernel Density Estimation (KDE)
    - 📈 **Quantile tracking** with vertical lines for 5th, 25th, 50th, 75th, and 90th percentiles
    - 🎬 **Time-lapse animation** showing distribution changes from 2001-2015
    - 📏 **Dual x-axis labels** showing both log10 values and original hectares

    **Data Requirements:**
    - CSV file with columns: `area_hectares` and `year`
    - Years should range from 2001-2015 (or subset)
    - Area values > 0 (for log transformation)

    **How to Use:**
    1. Upload your CSV to Google Drive and make it shareable
    2. Paste the sharing link in the sidebar
    3. Adjust animation speed and smoothness
    4. Click "Start Animation" to see the evolution

    The quantile lines help track how the distribution shape and central tendency change over time!
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Plotly, and SciPy | 🌲 Forest Area Analysis")
