from flask import Blueprint, render_template, request, redirect, url_for, flash
from models import Dataset, Analysis, ModelTraining
from app import db

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main dashboard showing recent datasets and analyses"""
    recent_datasets = Dataset.query.order_by(Dataset.upload_date.desc()).limit(5).all()
    recent_analyses = Analysis.query.order_by(Analysis.created_date.desc()).limit(10).all()
    
    # Get statistics for dashboard
    total_datasets = Dataset.query.count()
    total_analyses = Analysis.query.count()
    total_models = ModelTraining.query.count()
    
    stats = {
        'total_datasets': total_datasets,
        'total_analyses': total_analyses,
        'total_models': total_models
    }
    
    # Convert datasets to dictionaries for JSON serialization
    datasets_data = [dataset.to_dict() for dataset in recent_datasets]
    
    # Convert analyses to dictionaries for JSON serialization
    analyses_data = [analysis.to_dict() for analysis in recent_analyses]
    
    return render_template('index.html', 
                         recent_datasets=datasets_data,
                         recent_analyses=analyses_data,
                         stats=stats)

@main_bp.route('/dashboard')
def dashboard():
    """Main analysis dashboard"""
    datasets = Dataset.query.order_by(Dataset.upload_date.desc()).all()
    datasets_data = [dataset.to_dict() for dataset in datasets]
    return render_template('analysis_dashboard.html', datasets=datasets_data)

@main_bp.route('/reports')
def reports():
    """Reports and export page"""
    datasets = Dataset.query.order_by(Dataset.upload_date.desc()).all()
    datasets_data = [dataset.to_dict() for dataset in datasets]
    return render_template('reports.html', datasets=datasets_data)

@main_bp.route('/delete_dataset/<int:dataset_id>')
def delete_dataset(dataset_id):
    """Delete a dataset and all associated analyses"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Delete associated analyses and models
    Analysis.query.filter_by(dataset_id=dataset_id).delete()
    ModelTraining.query.filter_by(dataset_id=dataset_id).delete()
    
    # Delete the dataset
    db.session.delete(dataset)
    db.session.commit()
    
    flash('Dataset and all associated analyses have been deleted successfully.', 'success')
    return redirect(url_for('main.index'))

@main_bp.route('/debug_test')
def debug_test():
    """Debug route to test data serialization"""
    from datetime import datetime
    
    # Test with actual database data if available
    try:
        dataset = Dataset.query.first()
        if dataset:
            dataset_dict = dataset.to_dict()
            print("Dataset to_dict() result:")
            print(f"  upload_date type: {type(dataset_dict['upload_date'])}")
            print(f"  upload_date value: {dataset_dict['upload_date']}")
            
            # Test the template expression that's causing the error
            if dataset_dict['upload_date']:
                formatted_date = dataset_dict['upload_date'][:16].replace('T', ' ')
                print(f"  formatted date: {formatted_date}")
            
            return f"<h1>Debug Test Results</h1><p>Dataset ID: {dataset_dict['id']}</p><p>Filename: {dataset_dict['filename']}</p><p>Upload Date: {dataset_dict['upload_date']}</p><p>Date Type: {type(dataset_dict['upload_date'])}</p>"
        else:
            return "<h1>Debug Test</h1><p>No datasets found in database</p>"
            
    except Exception as e:
        return f"<h1>Debug Test Error</h1><p>Error: {str(e)}</p><p>Error Type: {type(e).__name__}</p>"
