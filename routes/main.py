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
    return render_template('analysis_dashboard.html', datasets=datasets)

@main_bp.route('/reports')
def reports():
    """Reports and export page"""
    datasets = Dataset.query.order_by(Dataset.upload_date.desc()).all()
    return render_template('reports.html', datasets=datasets)

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
