import logging
from flask import Blueprint, render_template, request, jsonify
from models import Dataset
from services.data_processor import DataProcessor
from services.visualization_engine import VisualizationEngine

visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/api/generate/<int:dataset_id>')
def generate_visualizations(dataset_id):
    """Generate visualizations for dataset"""
    try:
        chart_types = request.args.getlist('charts')
        columns = request.args.getlist('columns')
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            viz_engine = VisualizationEngine()
            
            visualizations = {}
            
            # Generate requested visualizations
            for chart_type in chart_types:
                if chart_type == 'correlation_heatmap':
                    visualizations['correlation_heatmap'] = viz_engine.create_correlation_heatmap(df)
                elif chart_type == 'distribution_plots' and columns:
                    visualizations['distribution_plots'] = {}
                    for col in columns:
                        if col in df.columns:
                            visualizations['distribution_plots'][col] = viz_engine.create_distribution_plot(df, col)
                elif chart_type == 'box_plots' and columns:
                    visualizations['box_plots'] = {}
                    for col in columns:
                        if col in df.columns:
                            visualizations['box_plots'][col] = viz_engine.create_box_plot(df, col)
                elif chart_type == 'scatter_matrix':
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) >= 2:
                        visualizations['scatter_matrix'] = viz_engine.create_scatter_matrix(df, numeric_cols[:10])
                elif chart_type == 'pairplot':
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) >= 2:
                        visualizations['pairplot'] = viz_engine.create_pairplot(df, numeric_cols[:8])
                elif chart_type == 'violin_plots' and columns:
                    visualizations['violin_plots'] = {}
                    for col in columns:
                        if col in df.columns:
                            visualizations['violin_plots'][col] = viz_engine.create_violin_plot(df, col)
                elif chart_type == 'qq_plots' and columns:
                    visualizations['qq_plots'] = {}
                    for col in columns:
                        if col in df.columns and df[col].dtype in ['int64', 'float64']:
                            visualizations['qq_plots'][col] = viz_engine.create_qq_plot(df, col)
                elif chart_type == 'bar_charts' and columns:
                    visualizations['bar_charts'] = {}
                    for col in columns:
                        if col in df.columns:
                            visualizations['bar_charts'][col] = viz_engine.create_bar_chart(df, col)
                elif chart_type == 'pie_charts' and columns:
                    visualizations['pie_charts'] = {}
                    for col in columns:
                        if col in df.columns:
                            visualizations['pie_charts'][col] = viz_engine.create_pie_chart(df, col)
                elif chart_type == 'heatmaps':
                    visualizations['heatmaps'] = viz_engine.create_comprehensive_heatmaps(df)
                elif chart_type == '3d_plots':
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) >= 3:
                        visualizations['3d_plots'] = viz_engine.create_3d_plots(df, numeric_cols[:3])
            
            return jsonify(visualizations)
            
    except Exception as e:
        logging.error(f"Visualization generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/comparison/<int:dataset_id>')
def comparison_visualizations(dataset_id):
    """Generate comparison visualizations between columns"""
    try:
        col1 = request.args.get('col1')
        col2 = request.args.get('col2')
        
        if not col1 or not col2:
            return jsonify({'error': 'Two columns must be specified'}), 400
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None and col1 in df.columns and col2 in df.columns:
            viz_engine = VisualizationEngine()
            
            comparison_viz = viz_engine.create_comparison_visualizations(df, col1, col2)
            
            return jsonify(comparison_viz)
            
    except Exception as e:
        logging.error(f"Comparison visualization error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization_bp.route('/api/advanced/<int:dataset_id>')
def advanced_visualizations(dataset_id):
    """Generate advanced visualizations"""
    try:
        viz_type = request.args.get('type')
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            viz_engine = VisualizationEngine()
            
            if viz_type == 'cluster_analysis':
                result = viz_engine.create_cluster_visualizations(df)
            elif viz_type == 'pca_analysis':
                result = viz_engine.create_pca_visualizations(df)
            elif viz_type == 'time_series' and 'date' in str(df.dtypes).lower():
                result = viz_engine.create_time_series_visualizations(df)
            elif viz_type == 'outlier_analysis':
                result = viz_engine.create_outlier_visualizations(df)
            elif viz_type == 'feature_importance':
                result = viz_engine.create_feature_importance_plots(df)
            else:
                return jsonify({'error': 'Invalid visualization type'}), 400
            
            return jsonify(result)
            
    except Exception as e:
        logging.error(f"Advanced visualization error: {str(e)}")
        return jsonify({'error': str(e)}), 500
