import logging
from flask import Blueprint, render_template, request, jsonify
from models import Dataset, Analysis
from services.data_processor import DataProcessor
from services.statistical_tests import StatisticalTestEngine
from app import db

statistics_bp = Blueprint('statistics', __name__)

@statistics_bp.route('/<int:dataset_id>')
def statistics_page(dataset_id):
    """Statistical tests page"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            # Get column information for test selection
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            return render_template('statistical_tests.html', 
                                 dataset=dataset.to_dict(),
                                 numeric_columns=numeric_cols,
                                 categorical_columns=categorical_cols)
    except Exception as e:
        logging.error(f"Statistics page error: {str(e)}")
    
    return render_template('statistical_tests.html', dataset=dataset.to_dict())

@statistics_bp.route('/api/normality/<int:dataset_id>')
def normality_tests(dataset_id):
    """Perform normality tests"""
    try:
        columns = request.args.getlist('columns')
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            test_engine = StatisticalTestEngine()
            results = {}
            
            for column in columns:
                if column in df.columns and df[column].dtype in ['int64', 'float64']:
                    results[column] = test_engine.normality_tests(df[column])
            
            # Save results
            analysis = Analysis(
                dataset_id=dataset_id,
                analysis_type='normality_tests'
            )
            analysis.set_results(results)
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"Normality tests error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@statistics_bp.route('/api/variance/<int:dataset_id>')
def variance_tests(dataset_id):
    """Perform variance tests"""
    try:
        groups = request.args.getlist('groups')
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None and len(groups) >= 2:
            test_engine = StatisticalTestEngine()
            
            # Prepare data for variance tests
            group_data = []
            for group in groups:
                if group in df.columns and df[group].dtype in ['int64', 'float64']:
                    group_data.append(df[group].dropna())
            
            if len(group_data) >= 2:
                results = test_engine.variance_tests(group_data)
                
                # Save results
                analysis = Analysis(
                    dataset_id=dataset_id,
                    analysis_type='variance_tests'
                )
                analysis.set_results(results)
                db.session.add(analysis)
                db.session.commit()
                
                return jsonify(results)
            
    except Exception as e:
        logging.error(f"Variance tests error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@statistics_bp.route('/api/correlation/<int:dataset_id>')
def correlation_tests(dataset_id):
    """Perform correlation tests"""
    try:
        col1 = request.args.get('col1')
        col2 = request.args.get('col2')
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None and col1 in df.columns and col2 in df.columns:
            test_engine = StatisticalTestEngine()
            results = test_engine.correlation_tests(df[col1], df[col2])
            
            # Save results
            analysis = Analysis(
                dataset_id=dataset_id,
                analysis_type='correlation_tests'
            )
            analysis.set_results(results)
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"Correlation tests error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@statistics_bp.route('/api/hypothesis/<int:dataset_id>')
def hypothesis_tests(dataset_id):
    """Perform hypothesis tests"""
    try:
        test_type = request.args.get('test_type')
        groups = request.args.getlist('groups')
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            test_engine = StatisticalTestEngine()
            results = {}
            
            if test_type == 'ttest':
                if len(groups) >= 2:
                    group1_data = df[groups[0]].dropna() if groups[0] in df.columns else None
                    group2_data = df[groups[1]].dropna() if groups[1] in df.columns else None
                    
                    if group1_data is not None and group2_data is not None:
                        results = test_engine.t_tests(group1_data, group2_data)
                        
            elif test_type == 'anova':
                group_data = []
                for group in groups:
                    if group in df.columns and df[group].dtype in ['int64', 'float64']:
                        group_data.append(df[group].dropna())
                
                if len(group_data) >= 2:
                    results = test_engine.anova_tests(group_data)
                    
            elif test_type == 'chi_square':
                if len(groups) >= 2:
                    col1_data = df[groups[0]] if groups[0] in df.columns else None
                    col2_data = df[groups[1]] if groups[1] in df.columns else None
                    
                    if col1_data is not None and col2_data is not None:
                        results = test_engine.chi_square_test(col1_data, col2_data)
            
            elif test_type == 'non_parametric':
                if len(groups) >= 2:
                    group1_data = df[groups[0]].dropna() if groups[0] in df.columns else None
                    group2_data = df[groups[1]].dropna() if groups[1] in df.columns else None
                    
                    if group1_data is not None and group2_data is not None:
                        results = test_engine.non_parametric_tests(group1_data, group2_data)
            
            # Save results
            if results:
                analysis = Analysis(
                    dataset_id=dataset_id,
                    analysis_type=f'hypothesis_test_{test_type}'
                )
                analysis.set_results(results)
                db.session.add(analysis)
                db.session.commit()
            
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"Hypothesis tests error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@statistics_bp.route('/api/comprehensive/<int:dataset_id>')
def comprehensive_tests(dataset_id):
    """Perform comprehensive statistical analysis"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            test_engine = StatisticalTestEngine()
            results = test_engine.comprehensive_statistical_analysis(df)
            
            # Save results
            analysis = Analysis(
                dataset_id=dataset_id,
                analysis_type='comprehensive_statistical_analysis'
            )
            analysis.set_results(results)
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"Comprehensive tests error: {str(e)}")
        return jsonify({'error': str(e)}), 500
