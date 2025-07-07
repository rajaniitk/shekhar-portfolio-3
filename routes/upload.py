import os
import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from services.data_processor import DataProcessor
from models import Dataset
from app import db, app

upload_bp = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'parquet'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/')
def upload_page():
    """Upload page for datasets"""
    return render_template('upload.html')

@upload_bp.route('/process', methods=['POST'])
def upload_file():
    """Process uploaded file"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the file
            processor = DataProcessor()
            df, file_info = processor.load_file(file_path)
            
            if df is not None:
                # Save dataset info to database
                dataset = Dataset(
                    filename=filename,
                    file_path=file_path,
                    file_type=file_info['type'],
                    shape_rows=file_info['shape'][0],
                    shape_cols=file_info['shape'][1],
                    memory_usage=file_info['memory_usage']
                )
                dataset.set_column_info(file_info['columns'])
                
                db.session.add(dataset)
                db.session.commit()
                
                flash(f'File uploaded successfully! Dataset contains {file_info["shape"][0]} rows and {file_info["shape"][1]} columns.', 'success')
                return redirect(url_for('analysis.dataset_overview', dataset_id=dataset.id))
            else:
                flash('Error processing file. Please check the file format.', 'error')
                
        except Exception as e:
            logging.error(f"Upload error: {str(e)}")
            flash(f'Error uploading file: {str(e)}', 'error')
    else:
        flash('Invalid file type. Please upload CSV, XLSX, JSON, or Parquet files.', 'error')
    
    return redirect(url_for('upload.upload_page'))

@upload_bp.route('/preview/<int:dataset_id>')
def preview_dataset(dataset_id):
    """Get dataset preview"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            # Get first and last 10 rows
            preview_data = {
                'head': df.head(10).to_html(classes='table table-dark table-striped', escape=False),
                'tail': df.tail(10).to_html(classes='table table-dark table-striped', escape=False),
                'info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict()
                }
            }
            return jsonify(preview_data)
    except Exception as e:
        logging.error(f"Preview error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Unable to load dataset'}), 500
