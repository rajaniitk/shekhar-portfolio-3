import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from models import Dataset, Analysis
from services.data_processor import DataProcessor
from services.eda_engine import EDAEngine
from services.insights_generator import InsightsGenerator
from app import db

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/dataset/<int:dataset_id>')
def dataset_overview(dataset_id):
    """Dataset overview and basic EDA"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            eda_engine = EDAEngine()
            basic_stats = eda_engine.generate_basic_statistics(df)
            
            return render_template('analysis_dashboard.html', 
                                 dataset=dataset.to_dict(),
                                 basic_stats=basic_stats,
                                 columns=list(df.columns))
    except Exception as e:
        logging.error(f"Dataset overview error: {str(e)}")
        flash(f'Error loading dataset: {str(e)}', 'error')
    
    return redirect(url_for('main.index'))

@analysis_bp.route('/column/<int:dataset_id>')
def column_analysis_page(dataset_id):
    """Column-wise analysis page"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            columns_info = {}
            for col in df.columns:
                columns_info[col] = {
                    'dtype': str(df[col].dtype),
                    'unique_count': df[col].nunique(),
                    'missing_count': df[col].isnull().sum(),
                    'missing_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2)
                }
            
            return render_template('column_analysis.html', 
                                 dataset=dataset.to_dict(),
                                 columns_info=columns_info)
    except Exception as e:
        logging.error(f"Column analysis page error: {str(e)}")
        flash(f'Error loading column analysis: {str(e)}', 'error')
    
    return redirect(url_for('main.index'))

@analysis_bp.route('/api/eda/<int:dataset_id>')
def generate_eda(dataset_id):
    """Generate comprehensive EDA report"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            eda_engine = EDAEngine()
            
            # Generate comprehensive EDA
            eda_results = {
                'basic_stats': eda_engine.generate_basic_statistics(df),
                'missing_values': eda_engine.analyze_missing_values(df),
                'correlation_analysis': eda_engine.correlation_analysis(df),
                'outlier_detection': eda_engine.detect_outliers(df),
                'distribution_analysis': eda_engine.analyze_distributions(df),
                'duplicate_analysis': eda_engine.analyze_duplicates(df)
            }
            
            # Save analysis to database
            analysis = Analysis(
                dataset_id=dataset_id,
                analysis_type='comprehensive_eda'
            )
            analysis.set_results(eda_results)
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify(eda_results)
            
    except Exception as e:
        logging.error(f"EDA generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/api/column_analysis/<int:dataset_id>')
def analyze_column(dataset_id):
    """Analyze specific columns"""
    try:
        columns = request.args.getlist('columns')
        if not columns:
            return jsonify({'error': 'No columns specified'}), 400
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            eda_engine = EDAEngine()
            insights_generator = InsightsGenerator()
            
            results = {}
            for column in columns:
                if column in df.columns:
                    column_analysis = eda_engine.analyze_single_column(df, column)
                    insights = insights_generator.generate_column_insights(df, column)
                    
                    results[column] = {
                        'analysis': column_analysis,
                        'insights': insights
                    }
            
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"Column analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/api/insights/<int:dataset_id>')
def generate_insights(dataset_id):
    """Generate AI-powered insights"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            insights_generator = InsightsGenerator()
            insights = insights_generator.generate_comprehensive_insights(df)
            
            # Save insights as analysis
            analysis = Analysis(
                dataset_id=dataset_id,
                analysis_type='ai_insights'
            )
            analysis.set_results(insights)
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify(insights)
            
    except Exception as e:
        logging.error(f"Insights generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500
