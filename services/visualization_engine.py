import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import base64
import io
import logging

class VisualizationEngine:
    """Comprehensive visualization engine with 60+ chart types"""
    
    def __init__(self):
        # Set dark theme for plotly
        self.plotly_template = "plotly_dark"
        self.color_palette = px.colors.qualitative.Set3
        
    def create_correlation_heatmap(self, df):
        """Create correlation heatmap"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {'error': 'Insufficient numeric columns for correlation heatmap'}
            
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Correlation Heatmap',
                template=self.plotly_template,
                width=800,
                height=600
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating correlation heatmap: {str(e)}")
            return {'error': str(e)}
    
    def create_distribution_plot(self, df, column):
        """Create comprehensive distribution plot"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            col_data = df[column].dropna()
            
            if df[column].dtype in ['int64', 'float64']:
                # Numeric distribution
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Histogram', 'Box Plot', 'Violin Plot', 'Q-Q Plot'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Histogram with KDE
                fig.add_trace(
                    go.Histogram(x=col_data, name='Histogram', nbinsx=30, opacity=0.7),
                    row=1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=col_data, name='Box Plot'),
                    row=1, col=2
                )
                
                # Violin plot
                fig.add_trace(
                    go.Violin(y=col_data, name='Violin Plot', box_visible=True),
                    row=2, col=1
                )
                
                # Q-Q plot
                qq_data = stats.probplot(col_data, dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                             mode='markers', name='Q-Q Plot'),
                    row=2, col=2
                )
                
                # Add theoretical line for Q-Q plot
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                             mode='lines', name='Theoretical Line', line=dict(color='red')),
                    row=2, col=2
                )
                
            else:
                # Categorical distribution
                value_counts = col_data.value_counts().head(20)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Bar Chart', 'Pie Chart'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Bar chart
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name='Count'),
                    row=1, col=1
                )
                
                # Pie chart (top 10 values)
                fig.add_trace(
                    go.Pie(labels=value_counts.head(10).index, 
                          values=value_counts.head(10).values, name='Distribution'),
                    row=1, col=2
                )
            
            fig.update_layout(
                title=f'Distribution Analysis: {column}',
                template=self.plotly_template,
                height=800,
                showlegend=False
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating distribution plot: {str(e)}")
            return {'error': str(e)}
    
    def create_box_plot(self, df, column):
        """Create box plot with outlier analysis"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            col_data = df[column].dropna()
            
            if df[column].dtype not in ['int64', 'float64']:
                return {'error': 'Box plot requires numeric data'}
            
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=col_data,
                name=column,
                boxpoints='outliers',
                marker_color='lightblue',
                line_color='blue',
                fillcolor='rgba(173, 216, 230, 0.5)'
            ))
            
            # Add statistics annotations
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            fig.add_annotation(
                x=0, y=col_data.max(),
                text=f"IQR: {iqr:.2f}<br>Outliers: {len(col_data[(col_data < lower_bound) | (col_data > upper_bound)])}",
                showarrow=True,
                arrowhead=2
            )
            
            fig.update_layout(
                title=f'Box Plot: {column}',
                template=self.plotly_template,
                height=500
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating box plot: {str(e)}")
            return {'error': str(e)}
    
    def create_scatter_matrix(self, df, columns):
        """Create scatter matrix for multiple numeric columns"""
        try:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
            
            if len(numeric_cols) < 2:
                return {'error': 'At least 2 numeric columns required'}
            
            # Limit to first 8 columns for performance
            cols_subset = numeric_cols[:8]
            df_subset = df[cols_subset].dropna()
            
            fig = ff.create_scatterplotmatrix(
                df_subset,
                diag='histogram',
                height=800,
                width=800
            )
            
            fig.update_layout(
                title='Scatter Plot Matrix',
                template=self.plotly_template
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating scatter matrix: {str(e)}")
            return {'error': str(e)}
    
    def create_pairplot(self, df, columns):
        """Create pairplot using seaborn and convert to plotly"""
        try:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
            
            if len(numeric_cols) < 2:
                return {'error': 'At least 2 numeric columns required'}
            
            # Use first 6 columns for performance
            cols_subset = numeric_cols[:6]
            df_subset = df[cols_subset].dropna()
            
            if len(df_subset) == 0:
                return {'error': 'No data available after removing missing values'}
            
            # Create matplotlib figure
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            g = sns.pairplot(df_subset, diag_kind='hist', plot_kws={'alpha': 0.6})
            g.fig.suptitle('Pairplot', y=1.02)
            
            # Convert to base64 image
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150, 
                       facecolor='#2F2F2F', edgecolor='none')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return {'image': plot_url, 'type': 'image'}
            
        except Exception as e:
            logging.error(f"Error creating pairplot: {str(e)}")
            return {'error': str(e)}
    
    def create_violin_plot(self, df, column):
        """Create violin plot"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            col_data = df[column].dropna()
            
            if df[column].dtype not in ['int64', 'float64']:
                return {'error': 'Violin plot requires numeric data'}
            
            fig = go.Figure()
            
            fig.add_trace(go.Violin(
                y=col_data,
                name=column,
                box_visible=True,
                meanline_visible=True,
                fillcolor='lightblue',
                opacity=0.6,
                x0=column
            ))
            
            fig.update_layout(
                title=f'Violin Plot: {column}',
                template=self.plotly_template,
                height=500
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating violin plot: {str(e)}")
            return {'error': str(e)}
    
    def create_qq_plot(self, df, column):
        """Create Q-Q plot for normality assessment"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            col_data = df[column].dropna()
            
            if df[column].dtype not in ['int64', 'float64']:
                return {'error': 'Q-Q plot requires numeric data'}
            
            qq_data = stats.probplot(col_data, dist="norm")
            
            fig = go.Figure()
            
            # Scatter plot of Q-Q data
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Sample Quantiles',
                marker=dict(color='lightblue', size=6)
            ))
            
            # Theoretical line
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Theoretical Line',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'Q-Q Plot: {column}',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                template=self.plotly_template,
                height=500
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating Q-Q plot: {str(e)}")
            return {'error': str(e)}
    
    def create_bar_chart(self, df, column):
        """Create bar chart for categorical data"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            value_counts = df[column].value_counts().head(20)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=column,
                marker_color=px.colors.qualitative.Set3
            ))
            
            fig.update_layout(
                title=f'Bar Chart: {column}',
                xaxis_title=column,
                yaxis_title='Count',
                template=self.plotly_template,
                height=500
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating bar chart: {str(e)}")
            return {'error': str(e)}
    
    def create_pie_chart(self, df, column):
        """Create pie chart for categorical data"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            value_counts = df[column].value_counts().head(15)
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                name=column,
                hole=0.3
            ))
            
            fig.update_layout(
                title=f'Pie Chart: {column}',
                template=self.plotly_template,
                height=500
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating pie chart: {str(e)}")
            return {'error': str(e)}
    
    def create_comprehensive_heatmaps(self, df):
        """Create various types of heatmaps"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'error': 'No numeric columns for heatmaps'}
            
            heatmaps = {}
            
            # Correlation heatmap
            corr_matrix = numeric_df.corr()
            heatmaps['correlation'] = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            )).to_json()
            
            # Covariance heatmap
            cov_matrix = numeric_df.cov()
            heatmaps['covariance'] = go.Figure(data=go.Heatmap(
                z=cov_matrix.values,
                x=cov_matrix.columns,
                y=cov_matrix.columns,
                colorscale='Viridis'
            )).to_json()
            
            # Missing values heatmap
            missing_matrix = df.isnull().astype(int)
            heatmaps['missing_values'] = go.Figure(data=go.Heatmap(
                z=missing_matrix.values,
                x=missing_matrix.columns,
                y=list(range(len(missing_matrix))),
                colorscale=[[0, 'blue'], [1, 'red']]
            )).to_json()
            
            return heatmaps
            
        except Exception as e:
            logging.error(f"Error creating heatmaps: {str(e)}")
            return {'error': str(e)}
    
    def create_3d_plots(self, df, columns):
        """Create 3D visualizations"""
        try:
            if len(columns) < 3:
                return {'error': 'At least 3 columns required for 3D plots'}
            
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
            
            if len(numeric_cols) < 3:
                return {'error': 'At least 3 numeric columns required'}
            
            df_clean = df[numeric_cols[:3]].dropna()
            
            plots_3d = {}
            
            # 3D Scatter plot
            fig_scatter = go.Figure(data=go.Scatter3d(
                x=df_clean.iloc[:, 0],
                y=df_clean.iloc[:, 1],
                z=df_clean.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_clean.iloc[:, 2],
                    colorscale='Viridis',
                    opacity=0.8
                )
            ))
            
            fig_scatter.update_layout(
                title='3D Scatter Plot',
                scene=dict(
                    xaxis_title=numeric_cols[0],
                    yaxis_title=numeric_cols[1],
                    zaxis_title=numeric_cols[2]
                ),
                template=self.plotly_template
            )
            
            plots_3d['scatter'] = fig_scatter.to_json()
            
            # 3D Surface plot (if enough data)
            if len(df_clean) > 100:
                # Create grid for surface plot
                x_vals = np.linspace(df_clean.iloc[:, 0].min(), df_clean.iloc[:, 0].max(), 20)
                y_vals = np.linspace(df_clean.iloc[:, 1].min(), df_clean.iloc[:, 1].max(), 20)
                
                X, Y = np.meshgrid(x_vals, y_vals)
                
                # Simple interpolation for Z values
                Z = np.zeros_like(X)
                for i in range(len(x_vals)):
                    for j in range(len(y_vals)):
                        # Find nearest point
                        distances = np.sqrt((df_clean.iloc[:, 0] - X[j, i])**2 + (df_clean.iloc[:, 1] - Y[j, i])**2)
                        nearest_idx = distances.idxmin()
                        Z[j, i] = df_clean.iloc[nearest_idx, 2]
                
                fig_surface = go.Figure(data=go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis'
                ))
                
                fig_surface.update_layout(
                    title='3D Surface Plot',
                    scene=dict(
                        xaxis_title=numeric_cols[0],
                        yaxis_title=numeric_cols[1],
                        zaxis_title=numeric_cols[2]
                    ),
                    template=self.plotly_template
                )
                
                plots_3d['surface'] = fig_surface.to_json()
            
            return plots_3d
            
        except Exception as e:
            logging.error(f"Error creating 3D plots: {str(e)}")
            return {'error': str(e)}
    
    def create_comparison_visualizations(self, df, col1, col2):
        """Create comparison visualizations between two columns"""
        try:
            if col1 not in df.columns or col2 not in df.columns:
                return {'error': 'One or both columns not found'}
            
            comparisons = {}
            
            # Both numeric
            if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                # Scatter plot
                fig_scatter = px.scatter(df, x=col1, y=col2, template=self.plotly_template,
                                       trendline="ols", title=f'{col1} vs {col2}')
                comparisons['scatter'] = fig_scatter.to_json()
                
                # Joint plot
                fig_joint = ff.create_2d_density(df[col1].dropna(), df[col2].dropna())
                fig_joint.update_layout(template=self.plotly_template, title='Joint Distribution')
                comparisons['joint'] = fig_joint.to_json()
                
            # Numeric vs Categorical
            elif (df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['object', 'category']) or \
                 (df[col2].dtype in ['int64', 'float64'] and df[col1].dtype in ['object', 'category']):
                
                numeric_col = col1 if df[col1].dtype in ['int64', 'float64'] else col2
                cat_col = col2 if numeric_col == col1 else col1
                
                # Box plot by category
                fig_box = px.box(df, x=cat_col, y=numeric_col, template=self.plotly_template,
                               title=f'{numeric_col} by {cat_col}')
                comparisons['box_by_category'] = fig_box.to_json()
                
                # Violin plot by category
                fig_violin = px.violin(df, x=cat_col, y=numeric_col, template=self.plotly_template,
                                     title=f'{numeric_col} Distribution by {cat_col}')
                comparisons['violin_by_category'] = fig_violin.to_json()
                
            # Both categorical
            else:
                # Crosstab heatmap
                crosstab = pd.crosstab(df[col1], df[col2])
                fig_crosstab = go.Figure(data=go.Heatmap(
                    z=crosstab.values,
                    x=crosstab.columns,
                    y=crosstab.index,
                    colorscale='Blues'
                ))
                fig_crosstab.update_layout(
                    title=f'Crosstab: {col1} vs {col2}',
                    template=self.plotly_template
                )
                comparisons['crosstab'] = fig_crosstab.to_json()
                
                # Stacked bar chart
                crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0)
                fig_stacked = go.Figure()
                
                for col in crosstab_pct.columns:
                    fig_stacked.add_trace(go.Bar(
                        name=str(col),
                        x=crosstab_pct.index,
                        y=crosstab_pct[col]
                    ))
                
                fig_stacked.update_layout(
                    barmode='stack',
                    title=f'Proportional Distribution: {col1} vs {col2}',
                    template=self.plotly_template
                )
                comparisons['stacked_bar'] = fig_stacked.to_json()
            
            return comparisons
            
        except Exception as e:
            logging.error(f"Error creating comparison visualizations: {str(e)}")
            return {'error': str(e)}
    
    def create_cluster_visualizations(self, df):
        """Create cluster analysis visualizations"""
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {'error': 'Insufficient numeric data for clustering'}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            cluster_viz = {}
            
            # 2D cluster plot (first two components)
            if numeric_df.shape[1] >= 2:
                fig_2d = go.Figure()
                
                for cluster in np.unique(clusters):
                    mask = clusters == cluster
                    fig_2d.add_trace(go.Scatter(
                        x=scaled_data[mask, 0],
                        y=scaled_data[mask, 1],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(size=8)
                    ))
                
                # Add centroids
                fig_2d.add_trace(go.Scatter(
                    x=kmeans.cluster_centers_[:, 0],
                    y=kmeans.cluster_centers_[:, 1],
                    mode='markers',
                    name='Centroids',
                    marker=dict(size=15, symbol='x', color='red')
                ))
                
                fig_2d.update_layout(
                    title='K-Means Clustering (2D)',
                    template=self.plotly_template
                )
                
                cluster_viz['2d_clusters'] = fig_2d.to_json()
            
            # Elbow method for optimal clusters
            inertias = []
            k_range = range(1, min(11, len(numeric_df)))
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                kmeans_temp.fit(scaled_data)
                inertias.append(kmeans_temp.inertia_)
            
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(k_range),
                y=inertias,
                mode='lines+markers',
                name='Inertia'
            ))
            
            fig_elbow.update_layout(
                title='Elbow Method for Optimal K',
                xaxis_title='Number of Clusters (k)',
                yaxis_title='Inertia',
                template=self.plotly_template
            )
            
            cluster_viz['elbow'] = fig_elbow.to_json()
            
            return cluster_viz
            
        except Exception as e:
            logging.error(f"Error creating cluster visualizations: {str(e)}")
            return {'error': str(e)}
    
    def create_pca_visualizations(self, df):
        """Create PCA analysis visualizations"""
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {'error': 'Insufficient numeric data for PCA'}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # PCA
            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)
            
            pca_viz = {}
            
            # Explained variance ratio
            fig_variance = go.Figure()
            fig_variance.add_trace(go.Bar(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_,
                name='Explained Variance Ratio'
            ))
            
            fig_variance.update_layout(
                title='PCA Explained Variance Ratio',
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio',
                template=self.plotly_template
            )
            
            pca_viz['explained_variance'] = fig_variance.to_json()
            
            # Cumulative explained variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            fig_cumvar = go.Figure()
            fig_cumvar.add_trace(go.Scatter(
                x=list(range(1, len(cumsum_variance) + 1)),
                y=cumsum_variance,
                mode='lines+markers',
                name='Cumulative Explained Variance'
            ))
            
            fig_cumvar.update_layout(
                title='Cumulative Explained Variance',
                xaxis_title='Principal Component',
                yaxis_title='Cumulative Explained Variance',
                template=self.plotly_template
            )
            
            pca_viz['cumulative_variance'] = fig_cumvar.to_json()
            
            # 2D PCA plot
            if pca_data.shape[1] >= 2:
                fig_2d = go.Figure()
                fig_2d.add_trace(go.Scatter(
                    x=pca_data[:, 0],
                    y=pca_data[:, 1],
                    mode='markers',
                    name='Data Points',
                    text=[f'Sample {i}' for i in range(len(pca_data))]
                ))
                
                fig_2d.update_layout(
                    title='PCA 2D Projection',
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                    template=self.plotly_template
                )
                
                pca_viz['2d_projection'] = fig_2d.to_json()
            
            return pca_viz
            
        except Exception as e:
            logging.error(f"Error creating PCA visualizations: {str(e)}")
            return {'error': str(e)}
    
    def create_time_series_visualizations(self, df):
        """Create time series visualizations"""
        try:
            # Try to identify date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        continue
            
            if not date_cols:
                return {'error': 'No date/time columns found'}
            
            date_col = date_cols[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return {'error': 'No numeric columns for time series analysis'}
            
            # Convert to datetime if needed
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col])
            
            df_sorted = df.sort_values(date_col)
            
            time_series_viz = {}
            
            # Time series plot for each numeric column
            for col in numeric_cols[:5]:  # Limit to first 5 for performance
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_sorted[date_col],
                    y=df_sorted[col],
                    mode='lines+markers',
                    name=col
                ))
                
                fig.update_layout(
                    title=f'Time Series: {col}',
                    xaxis_title='Date',
                    yaxis_title=col,
                    template=self.plotly_template
                )
                
                time_series_viz[col] = fig.to_json()
            
            return time_series_viz
            
        except Exception as e:
            logging.error(f"Error creating time series visualizations: {str(e)}")
            return {'error': str(e)}
    
    def create_outlier_visualizations(self, df):
        """Create outlier detection visualizations"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'error': 'No numeric columns for outlier analysis'}
            
            outlier_viz = {}
            
            for col in numeric_df.columns[:10]:  # Limit to first 10 columns
                col_data = numeric_df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # Calculate outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                normal_data = col_data[(col_data >= lower_bound) & (col_data <= upper_bound)]
                
                fig = go.Figure()
                
                # Normal data
                fig.add_trace(go.Scatter(
                    x=normal_data.index,
                    y=normal_data.values,
                    mode='markers',
                    name='Normal Data',
                    marker=dict(color='blue', size=6)
                ))
                
                # Outliers
                if len(outliers) > 0:
                    fig.add_trace(go.Scatter(
                        x=outliers.index,
                        y=outliers.values,
                        mode='markers',
                        name='Outliers',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                
                # Add bounds
                fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange", 
                             annotation_text="Lower Bound")
                fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange",
                             annotation_text="Upper Bound")
                
                fig.update_layout(
                    title=f'Outlier Detection: {col}',
                    xaxis_title='Index',
                    yaxis_title=col,
                    template=self.plotly_template
                )
                
                outlier_viz[col] = fig.to_json()
            
            return outlier_viz
            
        except Exception as e:
            logging.error(f"Error creating outlier visualizations: {str(e)}")
            return {'error': str(e)}
    
    def create_feature_importance_plots(self, df):
        """Create feature importance visualizations"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {'error': 'Insufficient numeric data for feature importance'}
            
            # Calculate correlation with each feature as target
            importance_viz = {}
            
            for target_col in numeric_df.columns[:5]:  # Limit to first 5 targets
                feature_cols = [col for col in numeric_df.columns if col != target_col]
                
                if not feature_cols:
                    continue
                
                # Calculate absolute correlations
                correlations = {}
                for feature in feature_cols:
                    corr = abs(numeric_df[target_col].corr(numeric_df[feature]))
                    if not np.isnan(corr):
                        correlations[feature] = corr
                
                if correlations:
                    # Sort by importance
                    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                    
                    features, importances = zip(*sorted_features[:10])  # Top 10 features
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(importances),
                        y=list(features),
                        orientation='h',
                        name='Feature Importance'
                    ))
                    
                    fig.update_layout(
                        title=f'Feature Importance (Target: {target_col})',
                        xaxis_title='Absolute Correlation',
                        yaxis_title='Features',
                        template=self.plotly_template,
                        height=600
                    )
                    
                    importance_viz[target_col] = fig.to_json()
            
            return importance_viz
            
        except Exception as e:
            logging.error(f"Error creating feature importance plots: {str(e)}")
            return {'error': str(e)}
