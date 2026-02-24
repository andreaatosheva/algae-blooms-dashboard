import gc
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.helper import show_memory_usage



# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, SEASONS, SEASON_COLORS
from utils.data_loader import load_dataset, get_variable_data

# Page config
st.set_page_config(
    page_title="Statistical Analysis",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Statistical Analysis")
st.markdown("Comprehensive statistical analysis and correlations between environmental variables.")

show_memory_usage()

@st.cache_data
def load_variable_data(var_name):
    """Load data for selected variable"""
    if var_name in ['nitrate', 'phosphate', 'ammonia']:
        ds = load_dataset('nutrients')
    else:
        ds = load_dataset(var_name)
    
    if ds is None:
        return None
    
    return get_variable_data(ds, var_name)

st.markdown("### Select Variable for Analysis")

analysis_type = st.selectbox(
    "Statistical Analysis Method",
    options=[
        "Correlation Matrix (All Variables)",
        "Pairwise Correlation Analysis",
        "Multiple Linear Regression",
        "Distribution Comparison",
        "Statistical Tests"
    ],
    index=0
)

st.markdown("---")
if analysis_type == "Correlation Matrix (All Variables)":
    st.markdown("## 🔗 Correlation Matrix - All Variables")
    
    st.info("Explore correlations between all environmental variables")
    
    all_vars = list(VARIABLE_INFO.keys())
    
    selected_vars = st.multiselect(
        "Select Variables for Analysis",
        options=all_vars,
        default=['chlorophyll', 'temperature', 'nitrate', 'phosphate', 'wind_speed'],
        format_func=lambda x: VARIABLE_INFO[x]['name']
    )
    
    if len(selected_vars) < 2:
        st.warning("Please select at least 2 variables for correlation analysis.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            corr_method = st.radio(
                "Correlation Method",
                options=["Pearson", "Spearman"],
                help="Pearson: linear relationships | Spearman: monotonic relationships"
            )
        
        with col2:
            show_pvalues = st.checkbox("Show P-values", value=True)
        
        
        with st.spinner("Loading and calculating correlations..."):
            var_data = {}
            
            for var in selected_vars:
                data = load_variable_data(var)
                if data is not None:
                    ts = data.mean(dim=['latitude', 'longitude'])
                    var_data[VARIABLE_INFO[var]['name']] = ts.values
            
            if len(var_data) < 2:
                st.error("Could not load enough variables.")
            else:
                df_corr = pd.DataFrame(var_data)
                
                if corr_method == "Pearson":
                    corr_matrix = df_corr.corr(method='pearson')
                else:
                    corr_matrix = df_corr.corr(method='spearman')
                
                if show_pvalues:
                    n = len(df_corr)
                    p_values = np.zeros((len(corr_matrix), len(corr_matrix)))
                    
                    for i in range(len(corr_matrix)):
                        for j in range(len(corr_matrix)):
                            if i != j:
                                if corr_method == "Pearson":
                                    _, p = pearsonr(df_corr.iloc[:, i].dropna(), df_corr.iloc[:, j].dropna())
                                else:
                                    _, p = spearmanr(df_corr.iloc[:, i].dropna(), df_corr.iloc[:, j].dropna())
                                p_values[i, j] = p
                            else:
                                p_values[i, j] = 0
                
                fig_corr = go.Figure()
                
                annotations = []
                for i, row in enumerate(corr_matrix.values):
                    for j, value in enumerate(row):
                        if show_pvalues and i != j:
                            if p_values[i, j] < 0.001:
                                sig = "***"
                            elif p_values[i, j] < 0.01:
                                sig = "**"
                            elif p_values[i, j] < 0.05:
                                sig = "*"
                            else:
                                sig = ""
                            text = f"{value:.2f}{sig}"
                        else:
                            text = f"{value:.2f}"
                        
                        annotations.append(
                            dict(
                                x=corr_matrix.columns[j],
                                y=corr_matrix.index[i],
                                text=text,
                                showarrow=False,
                                font=dict(
                                    color='white' if abs(value) > 0.5 else 'black',
                                    size=11,
                                    family='Arial Black'
                                )
                            )
                        )
                
                fig_corr.add_trace(go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Correlation"),
                    hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
                ))
                
                fig_corr.update_layout(
                    annotations=annotations,
                    title=f'{corr_method} Correlation Matrix',
                    width=600,
                    height=600,
                    template='plotly_white',
                    xaxis=dict(side='bottom', tickangle=-45),
                    yaxis=dict(side='left')
                )
                
                st.plotly_chart(fig_corr, width='stretch')
                
                if show_pvalues:
                    st.markdown("**Significance levels:** * p<0.05, ** p<0.01, *** p<0.001")
                
                st.markdown("**Top Correlations**")
                
                corr_pairs = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_val = corr_matrix.iloc[i, j]
                        
                        if abs(corr_val) > 0.7:
                            strength = "🔴 Strong"
                        elif abs(corr_val) > 0.4:
                            strength = "🟡 Moderate"
                        else:
                            strength = "🟢 Weak"
                        
                        corr_pairs.append({
                            'Variable 1': corr_matrix.index[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_val,
                            'Strength': strength
                        })
                
                df_pairs = pd.DataFrame(corr_pairs)
                df_pairs = df_pairs.sort_values('Correlation', key=abs, ascending=False)
                df_pairs['Correlation'] = df_pairs['Correlation'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(df_pairs, hide_index=True, width='stretch', height=500)

elif analysis_type == "Pairwise Correlation Analysis":
    st.markdown("## 🔗 Pairwise Correlation Analysis")

    col1, col2 = st.columns(2)
    
    all_vars = list(VARIABLE_INFO.keys())
    
    with col1:
        var1 = st.selectbox(
            "Variable 1",
            options=all_vars,
            format_func=lambda x: VARIABLE_INFO[x]['name'],
            index=0
        )
    
    with col2:
        var2 = st.selectbox(
            "Variable 2",
            options=[v for v in all_vars if v != var1],
            format_func=lambda x: VARIABLE_INFO[x]['name']
        )
    
    var1_info = VARIABLE_INFO[var1]
    var2_info = VARIABLE_INFO[var2]
    
    with st.spinner("Loading data..."):
        data1 = load_variable_data(var1)
        data2 = load_variable_data(var2)
    
    if data1 is None or data2 is None:
        st.error("Could not load one or both variables.")
    else:
        
        ts1 = data1.mean(dim=['latitude', 'longitude'], skipna=True)
        ts2 = data2.mean(dim=['latitude', 'longitude'], skipna=True)
        
        try:
            df1 = ts1.to_dataframe(name='var1')
            df2 = ts2.to_dataframe(name='var2')
            
            merged = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
                        
            if len(merged) != 0:
                merged_clean = merged.dropna()

                x_data = merged_clean['var1'].values
                y_data = merged_clean['var2'].values
                
                pearson_r, pearson_p = pearsonr(x_data, y_data)
                spearman_r, spearman_p = spearmanr(x_data, y_data)
                
                st.markdown("### 📊 Correlation Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Pearson r", f"{pearson_r:.3f}", help="Pearson R measures the strength and direction of the linear relationship between two variables, ranging from -1 to 1. A value near 1 means strong positive linear correlation, near -1 means strong negative linear correlation, and near 0 means no linear relationship.")
                
                with col2:
                    st.metric("Pearson p-value", f"{pearson_p:.4f}", help="Pearson P-value indicates whether the correlation between two variables is statistically significant. If p < 0.05, the relationship is unlikely due to chance and the correlation can be considered real and meaningful.")
                
                with col3:
                    st.metric("Spearman r", f"{spearman_r:.3f}", help="Spearman R measures the strength and direction of the monotonic relationship between two variables, ranging from -1 to 1. A value near 1 means strong positive correlation, near -1 means strong negative correlation, and near 0 means no relationship.")
                
                with col4:
                    st.metric("Spearman p-value", f"{spearman_p:.4f}", help="Spearman P-value indicates whether the monotonic relationship between two variables is statistically significant. Unlike Pearson, it doesn't assume normality or a linear relationship, making it more robust for skewed data. If p < 0.05, the relationship is significant.")
                
                if pearson_p < 0.05:
                    direction = "positive" if pearson_r > 0 else "negative"
                    strength = "strong" if abs(pearson_r) > 0.7 else "moderate" if abs(pearson_r) > 0.4 else "weak"
                    st.success(f"✅ Statistically significant {strength} {direction} correlation detected!")
                else:
                    st.warning("⚠️ No statistically significant correlation (p > 0.05)")
                
                st.markdown("---")
                    
                col1, col2 = st.columns(2)
                
                with col1:
                    
                    fig_scatter = go.Figure()
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name='Data Points',
                        marker=dict(
                            size=8,
                            color=x_data,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title=var1_info['unit'],
                                len = 0.8,
                                y = 0.05,
                                yanchor = 'bottom',
                                thickness=30),
                            line=dict(width=0.5, color='white'),
                            opacity=0.7
                        ),
                        hovertemplate=f'<b>{var1_info["name"]}</b>: %{{x:.2f}} {var1_info["unit"]}<br>' +
                                        f'<b>{var2_info["name"]}</b>: %{{y:.2f}} {var2_info["unit"]}<br>' +
                                        '<extra></extra>'
                    ))
                    
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_data.min(), x_data.max(), 100)
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode='lines',
                        name=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}',
                        line=dict(color='red', width=3, dash='dash')
                    ))
                    
                    r_squared = pearson_r ** 2
                    
                    fig_scatter.update_layout(
                        title=f'{var1_info["name"]} vs {var2_info["name"]}',
                        xaxis_title=f'{var1_info["name"]} ({var1_info["unit"]})',
                        yaxis_title=f'{var2_info["name"]} ({var2_info["unit"]})',
                        height=500,
                        template='plotly_white',
                        annotations=[
                            dict(
                                x=0.05, y=0.95,
                                xref='paper', yref='paper',
                                text=f"R² = {r_squared:.3f}<br>n = {len(x_data)}",
                                showarrow=False,
                                font=dict(size=12),
                                bgcolor='white',
                                bordercolor='black',
                                borderwidth=1
                            )
                        ]
                    )
                    
                    st.plotly_chart(fig_scatter, width='stretch')
                
                with col2:
                    y_pred = p(x_data)
                    residuals = y_data - y_pred
                    
                    fig_residual = go.Figure()
                    
                    fig_residual.add_trace(go.Scatter(
                        x=y_pred,
                        y=residuals,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='blue',
                            opacity=0.6,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    fig_residual.update_layout(
                        title='Residual Plot',
                        xaxis_title='Predicted Values',
                        yaxis_title='Residuals',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_residual, width='stretch')
                
                st.markdown("### 📊 Regression Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Model Fit**")
                    st.metric("R² (Coefficient of Determination)", f"{r_squared:.3f}", help="R² indicates the proportion of variance in the dependent variable that is predictable from the independent variable. An R² of 0.7 means 70% of the variance is explained by the model.")
                    st.metric("RMSE", f"{np.sqrt(np.mean(residuals**2)):.3f}", help="Root Mean Squared Error (RMSE) measures the average magnitude of the residuals (prediction errors). Lower RMSE indicates a better fit.")
                    st.metric("MAE", f"{np.mean(np.abs(residuals)):.3f}", help="Mean Absolute Error (MAE) measures the average absolute magnitude of the residuals. Like RMSE, lower MAE indicates a better fit, but it is less sensitive to outliers.")
                
                with col2:
                    st.markdown("**Regression Coefficients**")
                    st.metric("Slope", f"{z[0]:.3f}", help="The slope indicates the change in the dependent variable for a one-unit change in the independent variable. A slope of 0.5 means that for every 1 unit increase in Variable 1, Variable 2 increases by 0.5 units on average.")
                    st.metric("Intercept", f"{z[1]:.3f}", help="The intercept is the expected value of the dependent variable when the independent variable is zero. It represents the baseline level of Variable 2 when Variable 1 is zero.")
                    st.metric("Pearson r", f"{pearson_r:.3f}", help="Pearson r measures the linear correlation between the independent and dependent variables. A value of 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.")
                
                with col3:
                    st.markdown("**Data Summary**")
                    st.metric("Sample Size", f"{len(x_data)}")
                    st.metric("Missing Before Merge", f"{len(df1) - len(merged)}")
                    st.metric("NaN Removed", f"{len(merged) - len(merged_clean)}")
    
        except Exception as e:
            st.error(f"Error during alignment: {str(e)}")
            st.exception(e)

# ==================== MULTIPLE LINEAR REGRESSION ====================
elif analysis_type == "Multiple Linear Regression":
    st.markdown("## 📊 Multiple Linear Regression")
    
    st.info("Predict one variable using multiple environmental factors")
    
    target_var = st.selectbox(
        "Target Variable (to predict)",
        options=list(VARIABLE_INFO.keys()),
        format_func=lambda x: VARIABLE_INFO[x]['name'],
        index=0 
    )
    
    target_info = VARIABLE_INFO[target_var]
    
    predictor_vars = st.multiselect(
        "Predictor Variables",
        options=[v for v in VARIABLE_INFO.keys() if v != target_var],
        default=['temperature', 'nitrate', 'phosphate', 'wind_speed'],
        format_func=lambda x: VARIABLE_INFO[x]['name']
    )
    
    if len(predictor_vars) < 1:
        st.warning("Please select at least one predictor variable.")
    else:
        with st.spinner("Loading data and fitting model..."):
            target_data = load_variable_data(target_var)
            if target_data is None:
                st.error(f"Could not load {target_info['name']} data.")
            else:
                target_ts = target_data.mean(dim=['latitude', 'longitude'])
                
                predictor_data = {}
                for var in predictor_vars:
                    data = load_variable_data(var)
                    if data is not None:
                        ts = data.mean(dim=['latitude', 'longitude'])
                        predictor_data[var] = ts.values
                
                if len(predictor_data) < 1:
                    st.error("Could not load predictor variables.")
                else:
                    df_regression = pd.DataFrame(predictor_data)
                    df_regression['target'] = target_ts.values
                    
                    df_regression = df_regression.dropna()
                    
                    if len(df_regression) < 10:
                        st.error("Not enough valid data points for regression.")
                    else:
                        
                        X = df_regression[predictor_vars].values
                        y = df_regression['target'].values
                        
                        # Fit model
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        y_pred = model.predict(X)
                        
                        r2 = r2_score(y, y_pred)
                        rmse = np.sqrt(mean_squared_error(y, y_pred))
                        mae = mean_absolute_error(y, y_pred)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("R²", f"{r2:.3f}")
                        
                        with col2:
                            st.metric("RMSE", f"{rmse:.3f}")
                        
                        with col3:
                            st.metric("MAE", f"{mae:.3f}")
                        
                        with col4:
                            st.metric("Sample Size", len(df_regression))
                        
                        if r2 > 0.7:
                            st.success(f"✅ Excellent model fit (R² = {r2:.3f})! The predictors explain {r2*100:.1f}% of variance.")
                        elif r2 > 0.5:
                            st.info(f"ℹ️ Good model fit (R² = {r2:.3f}). The predictors explain {r2*100:.1f}% of variance.")
                        elif r2 > 0.3:
                            st.warning(f"⚠️ Moderate model fit (R² = {r2:.3f}). The predictors explain {r2*100:.1f}% of variance.")
                        else:
                            st.warning(f"⚠️ Weak model fit (R² = {r2:.3f}). The predictors explain only {r2*100:.1f}% of variance.")
                        
                        st.markdown("---")
                        
                        st.markdown("### 🎯 Feature Importance")
                        
                        coef_df = pd.DataFrame({
                            'Variable': [VARIABLE_INFO[v]['name'] for v in predictor_vars],
                            'Coefficient': model.coef_,
                            'Abs Coefficient': np.abs(model.coef_)
                        })
                        coef_df = coef_df.sort_values('Abs Coefficient', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_coef = go.Figure()
                            
                            colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
                            
                            fig_coef.add_trace(go.Bar(
                                x=coef_df['Variable'],
                                y=coef_df['Coefficient'],
                                marker_color=colors,
                                marker_line_color='black',
                                marker_line_width=1.5,
                                opacity=0.8,
                                text=[f"{c:.3f}" for c in coef_df['Coefficient']],
                                textposition='outside'
                            ))
                            
                            fig_coef.add_hline(y=0, line_color="black", line_width=1)
                            
                            fig_coef.update_layout(
                                title='Regression Coefficients',
                                xaxis_title='Variable',
                                yaxis_title='Coefficient',
                                height=400,
                                template='plotly_white',
                                margin=dict(l=50, r=30, t=50, b=80),
                                )
                            
                            st.plotly_chart(fig_coef, width='stretch')
                        
                        with col2:
                            st.markdown("**Coefficient Values**")
                            
                            coef_display = coef_df[['Variable', 'Coefficient']].copy()
                            coef_display['Coefficient'] = coef_display['Coefficient'].apply(lambda x: f"{x:.4f}")
                            coef_display['Effect'] = ['Negative ↓' if c < 0 else 'Positive ↑' 
                                                      for c in coef_df['Coefficient']]
                            
                            st.dataframe(coef_display, hide_index=True, width='stretch', height=350)
                            
                            st.markdown(f"**Intercept:** {model.intercept_:.4f}")
                        
                        st.markdown("---")
                        st.markdown("### 📈 Actual vs Predicted")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pred = go.Figure()
                            
                            fig_pred.add_trace(go.Scatter(
                                x=y,
                                y=y_pred,
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color='blue',
                                    opacity=0.6,
                                    line=dict(width=0.5, color='white')
                                ),
                                hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
                            ))
                            
                            min_val = min(y.min(), y_pred.min())
                            max_val = max(y.max(), y_pred.max())
                            fig_pred.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash', width=2),
                                name='Perfect Prediction'
                            ))
                            
                            fig_pred.update_layout(
                                title='Actual vs Predicted Values',
                                xaxis_title=f'Actual {target_info["name"]} ({target_info["unit"]})',
                                yaxis_title=f'Predicted {target_info["name"]} ({target_info["unit"]})',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_pred, width='stretch')
                        
                        with col2:
                            fig_ts = go.Figure()
                            
                            fig_ts.add_trace(go.Scatter(
                                x=df_regression.index,
                                y=y,
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color=target_info['color'], width=2),
                                marker=dict(size=4)
                            ))
                            
                            fig_ts.add_trace(go.Scatter(
                                x=df_regression.index,
                                y=y_pred,
                                mode='lines',
                                name='Predicted',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig_ts.update_layout(
                                title='Time Series: Actual vs Predicted',
                                xaxis_title='Date',
                                yaxis_title=f'{target_info["name"]} ({target_info["unit"]})',
                                height=400,
                                template='plotly_white',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_ts, width='stretch')

# ==================== DISTRIBUTION COMPARISON ====================
elif analysis_type == "Distribution Comparison":
    st.markdown("## 📊 Distribution Comparison")
    
    st.info("Compare statistical distributions of different variables")
    
    compare_vars = st.multiselect(
        "Select Variables to Compare",
        options=list(VARIABLE_INFO.keys()),
        default=['chlorophyll', 'temperature', 'nitrate'],
        format_func=lambda x: VARIABLE_INFO[x]['name']
    )
    
    if len(compare_vars) < 2:
        st.warning("Please select at least 2 variables.")
    else:
        with st.spinner("Loading data..."):
            distributions = {}
            
            for var in compare_vars:
                data = load_variable_data(var)
                if data is not None:
                    ts = data.mean(dim=['latitude', 'longitude'])
                    valid_values = ts.values[~np.isnan(ts.values)]
                    distributions[VARIABLE_INFO[var]['name']] = valid_values
        
        if len(distributions) < 2:
            st.error("Could not load enough variables.")
        else:
            normalize_dist = st.checkbox("Normalize distributions (0-1 scale)", value=True)
            
            if normalize_dist:
                distributions_norm = {}
                for name, values in distributions.items():
                    values_norm = (values - np.min(values)) / (np.max(values) - np.min(values))
                    distributions_norm[name] = values_norm
                distributions_plot = distributions_norm
                x_label = "Normalized Value (0-1)"
            else:
                distributions_plot = distributions
                x_label = "Value"
            
            fig_dist = go.Figure()
            
            for name, values in distributions_plot.items():
                fig_dist.add_trace(go.Histogram(
                    x=values,
                    name=name,
                    opacity=0.6,
                    nbinsx=40
                ))
            
            fig_dist.update_layout(
                barmode='overlay',
                title='Distribution Comparison',
                xaxis_title=x_label,
                yaxis_title='Frequency',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_dist, width='stretch')
            
            st.markdown("### 📦 Box Plot Comparison")
            
            fig_box = go.Figure()
            
            for var_name, values in distributions_plot.items():
                fig_box.add_trace(go.Box(
                    y=values,
                    name=var_name,
                    boxmean='sd'
                ))
            
            fig_box.update_layout(
                title='Distribution Statistics',
                yaxis_title=x_label,
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_box, width='stretch')
            
            st.markdown("### 📋 Summary Statistics")
            
            stats_data = []
            for name, values in distributions.items():
                stats_data.append({
                    'Variable': name,
                    'Mean': f"{np.mean(values):.3f}",
                    'Median': f"{np.median(values):.3f}",
                    'Std Dev': f"{np.std(values):.3f}",
                    'Min': f"{np.min(values):.3f}",
                    'Max': f"{np.max(values):.3f}",
                    'Skewness': f"{stats.skew(values):.3f}",
                    'Kurtosis': f"{stats.kurtosis(values):.3f}"
                })
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, hide_index=True, width='stretch')

# ==================== STATISTICAL TESTS ====================
elif analysis_type == "Statistical Tests":
    st.markdown("## 🧪 Statistical Hypothesis Tests")
    
    st.info("Perform statistical tests to compare groups or test hypotheses")
    
    test_type = st.selectbox(
        "Select Test",
        options=[
            "Compare Two Periods (T-test)",
            "Compare Multiple Seasons (ANOVA)",
            "Normality Test",
            "Trend Significance (Mann-Kendall)"
        ]
    )
    
    if test_type == "Compare Two Periods (T-test)":
        st.markdown("### T-Test: Compare Two Time Periods")
        
        test_var = st.selectbox(
            "Select Variable",
            options=list(VARIABLE_INFO.keys()),
            format_func=lambda x: VARIABLE_INFO[x]['name']
        )
        
        test_info = VARIABLE_INFO[test_var]
        
        data = load_variable_data(test_var)
        if data is None:
            st.error("Could not load data.")
        else:
            ts = data.mean(dim=['latitude', 'longitude'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Period 1**")
                period1_year = st.selectbox("Year", options=sorted(range(2014, 2025), reverse=True), 
                                           index=1, key='p1_year')
                period1_season = st.selectbox("Season", 
                                             options=list(SEASONS.keys()),
                                             index=2, key='p1_season')
            
            with col2:
                st.markdown("**Period 2**")
                period2_year = st.selectbox("Year", options=sorted(range(2014, 2025), reverse=True),
                                           index=0, key='p2_year')
                period2_season = st.selectbox("Season",
                                             options=list(SEASONS.keys()),
                                             index=2, key='p2_season')
            
            period1_months = SEASONS[period1_season]
            period2_months = SEASONS[period2_season]
            
            period1_data = ts.where(
                (ts.time.dt.year == period1_year) & (ts.time.dt.month.isin(period1_months)),
                drop=True
            ).values
            period1_data = period1_data[~np.isnan(period1_data)]
            
            period2_data = ts.where(
                (ts.time.dt.year == period2_year) & (ts.time.dt.month.isin(period2_months)),
                drop=True
            ).values
            period2_data = period2_data[~np.isnan(period2_data)]
            
            if len(period1_data) < 2 or len(period2_data) < 2:
                st.error("Not enough data in one or both periods.")
            else:
                t_stat, p_value = stats.ttest_ind(period1_data, period2_data)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Period 1 Mean", f"{np.mean(period1_data):.2f} {test_info['unit']}")
                
                with col2:
                    st.metric("Period 2 Mean", f"{np.mean(period2_data):.2f} {test_info['unit']}")
                
                with col3:
                    st.metric("T-statistic", f"{t_stat:.3f}")
                
                with col4:
                    st.metric("P-value", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    diff = np.mean(period2_data) - np.mean(period1_data)
                    direction = "higher" if diff > 0 else "lower"
                    st.success(f"✅ **Statistically significant difference!** (p = {p_value:.4f})")
                    st.info(f"Period 2 ({period2_season} {period2_year}) is significantly {direction} than Period 1 ({period1_season} {period1_year}).")
                else:
                    st.warning(f"⚠️ No statistically significant difference (p = {p_value:.4f} > 0.05)")
                
                fig_compare = go.Figure()
                
                fig_compare.add_trace(go.Box(
                    y=period1_data,
                    name=f'{period1_season} {period1_year}',
                    marker_color='blue',
                    boxmean='sd'
                ))
                
                fig_compare.add_trace(go.Box(
                    y=period2_data,
                    name=f'{period2_season} {period2_year}',
                    marker_color='red',
                    boxmean='sd'
                ))
                
                fig_compare.update_layout(
                    title=f'{test_info["name"]} Distribution Comparison',
                    yaxis_title=f'{test_info["name"]} ({test_info["unit"]})',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_compare, width='stretch')
    
    elif test_type == "Compare Multiple Seasons (ANOVA)":
        st.markdown("### ANOVA: Compare All Seasons")
        
        test_var = st.selectbox(
            "Select Variable",
            options=list(VARIABLE_INFO.keys()),
            format_func=lambda x: VARIABLE_INFO[x]['name']
        )
        
        test_info = VARIABLE_INFO[test_var]
        
        data = load_variable_data(test_var)
        if data is None:
            st.error("Could not load data.")
        else:
            ts = data.mean(dim=['latitude', 'longitude'])
            
            season_data = {}
            
            for season_name, months in SEASONS.items():
                season_values = ts.where(ts.time.dt.month.isin(months), drop=True).values
                season_values = season_values[~np.isnan(season_values)]
                if len(season_values) > 0:
                    season_data[season_name] = season_values
            
            if len(season_data) < 2:
                st.error("Not enough seasonal data.")
                
            
            else:
                f_stat, p_value = stats.f_oneway(*season_data.values())
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("F-statistic", f"{f_stat:.3f}", help="F-statistic measures how much the seasons differ from each other relative to the natural variation within each season. A higher value means the seasonal differences are stronger and more distinct.")
                
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                
                with col3:
                    result = "Significant ✓" if p_value < 0.05 else "Not Significant"
                    st.metric("Result", result)
                
                if p_value < 0.05:
                    st.success(f"✅ **Statistically significant difference between seasons!**")
                    st.info("At least one season has significantly different values from the others.")
                else:
                    st.warning(f"⚠️ No statistically significant difference between seasons")
                
                fig_seasons = go.Figure()
                
                for season_name, values in season_data.items():
                    fig_seasons.add_trace(go.Box(
                        y=values,
                        name=season_name,
                        marker_color=SEASON_COLORS[season_name],
                        boxmean='sd'
                    ))
                
                fig_seasons.update_layout(
                    title=f'{test_info["name"]} Distribution by Season',
                    yaxis_title=f'{test_info["name"]} ({test_info["unit"]})',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_seasons, width='stretch')
                
                seasonal_stats = pd.DataFrame({
                    'Season': list(season_data.keys()),
                    'Mean': [f"{np.mean(v):.3f}" for v in season_data.values()],
                    'Std Dev': [f"{np.std(v):.3f}" for v in season_data.values()],
                    'N': [len(v) for v in season_data.values()]
                })
                
                st.dataframe(seasonal_stats, hide_index=True, width='stretch')
    
    elif test_type == "Normality Test":
        st.markdown("### Normality Test (Shapiro-Wilk)", help="Normality Test checks whether your data follows a normal (bell-curve) distribution. This matters because many statistical tests assume normality, if the data fails this test, non-parametric alternatives may be more appropriate.")
        
        test_var = st.selectbox(
            "Select Variable",
            options=list(VARIABLE_INFO.keys()),
            format_func=lambda x: VARIABLE_INFO[x]['name']
        )
        
        test_info = VARIABLE_INFO[test_var]
        
        data = load_variable_data(test_var)
        if data is None:
            st.error("Could not load data.")
        else:
            ts = data.mean(dim=['latitude', 'longitude'])
            valid_data = ts.values[~np.isnan(ts.values)]
            
            if len(valid_data) < 3:
                st.error("Not enough data for normality test.")
            else:
                stat, p_value = stats.shapiro(valid_data[:5000]) 
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("W-statistic", f"{stat:.4f}", help="W-statistic measures how closely the data follows a normal distribution. Values close to 1 indicate a good fit to normality, while values significantly less than 1 suggest deviations from normality.")
                
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                
                with col3:
                    result = "Normal ✓" if p_value > 0.05 else "Not Normal"
                    st.metric("Distribution", result)
                
                if p_value > 0.05:
                    st.success(f"✅ Data appears normally distributed (p = {p_value:.4f} > 0.05)")
                else:
                    st.warning(f"⚠️ Data does not appear normally distributed (p = {p_value:.4f} < 0.05)")
                

                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=valid_data,
                        nbinsx=40,
                        name='Data',
                        marker_color=test_info['color'],
                        opacity=0.7,
                        histnorm='probability density'
                    ))
                    
                    mu = np.mean(valid_data)
                    sigma = np.std(valid_data)
                    x_range = np.linspace(valid_data.min(), valid_data.max(), 100)
                    normal_curve = stats.norm.pdf(x_range, mu, sigma)
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=normal_curve,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_hist.update_layout(
                        title='Histogram with Normal Curve',
                        xaxis_title=f'{test_info["name"]} ({test_info["unit"]})',
                        yaxis_title='Density',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_hist, width='stretch')
                
                with col2:
                    fig_qq = go.Figure()
                    
                    (osm, osr), (slope, intercept, r) = stats.probplot(valid_data, dist="norm")
                    
                    fig_qq.add_trace(go.Scatter(
                        x=osm,
                        y=osr,
                        mode='markers',
                        name='Data',
                        marker=dict(color='blue', size=6, opacity=0.6)
                    ))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=osm,
                        y=slope * osm + intercept,
                        mode='lines',
                        name='Normal Reference',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title='Q-Q Plot',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_qq, width='stretch')
    
    elif test_type == "Trend Significance (Mann-Kendall)":
        st.markdown("### Mann-Kendall Trend Test")
        
        st.info("Non-parametric test for monotonic trends (doesn't assume normal distribution)")
        
        test_var = st.selectbox(
            "Select Variable",
            options=list(VARIABLE_INFO.keys()),
            format_func=lambda x: VARIABLE_INFO[x]['name']
        )
        
        test_info = VARIABLE_INFO[test_var]
        
        data = load_variable_data(test_var)
        if data is None:
            st.error("Could not load data.")
        else:
            ts = data.mean(dim=['latitude', 'longitude'])
            valid_data = ts.values[~np.isnan(ts.values)]
            
            n = len(valid_data)
            s = 0
            
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(valid_data[j] - valid_data[i])
            
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("S-statistic", f"{s}", help="S-statistic (Skewness) measures the asymmetry of your data distribution. A value near 0 means symmetric, positive values indicate a longer right tail, and negative values indicate a longer left tail.")
            
            with col2:
                st.metric("Z-score", f"{z:.3f}", help="Z-score (Mann-Kendall test) measures how many standard deviations the trend's S-statistic is from zero. Values beyond ±1.96 indicate a statistically significant trend at the 95% confidence level - positive values suggest an upward trend, negative values a downward trend.")
            
            with col3:
                st.metric("P-value", f"{p_value:.4f}", help="P-value (Mann-Kendall test) indicates the probability of observing the data if there were actually no trend. A common threshold is 0.05; if p < 0.05, we reject the null hypothesis of no trend and conclude that a statistically significant trend exists.")
            
            with col4:
                if p_value < 0.05:
                    trend = "Increasing ↗" if s > 0 else "Decreasing ↘"
                else:
                    trend = "No Trend"
                st.metric("Trend", trend)
            
            if p_value < 0.05:
                direction = "increasing" if s > 0 else "decreasing"
                st.success(f"✅ **Statistically significant {direction} trend detected!** (p = {p_value:.4f})")
            else:
                st.warning(f"⚠️ No statistically significant trend (p = {p_value:.4f} > 0.05)")
            
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=ts.time.values[~np.isnan(ts.values)],
                y=valid_data,
                mode='lines+markers',
                name=test_info['name'],
                line=dict(color=test_info['color'], width=2),
                marker=dict(size=5)
            ))
            
            slopes = []
            time_numeric = np.arange(len(valid_data))
            for i in range(len(valid_data)-1):
                for j in range(i+1, len(valid_data)):
                    slopes.append((valid_data[j] - valid_data[i]) / (j - i))
            
            sen_slope = np.median(slopes)
            sen_intercept = np.median(valid_data - sen_slope * time_numeric)
            
            fig_trend.add_trace(go.Scatter(
                x=ts.time.values[~np.isnan(ts.values)],
                y=sen_slope * time_numeric + sen_intercept,
                mode='lines',
                name="Sen's Slope",
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_trend.update_layout(
                title=f'{test_info["name"]} - Mann-Kendall Trend Analysis',
                xaxis_title='Date',
                yaxis_title=f'{test_info["name"]} ({test_info["unit"]})',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_trend, width='stretch')
            
            st.markdown(f"**Sen's Slope:** {sen_slope:.4f} {test_info['unit']}/month")

# Clear memory
gc.collect()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Use correlation analysis to find relationships, 
    regression to build predictive models, and statistical tests to validate findings.
    </small>
</div>
""", unsafe_allow_html=True)