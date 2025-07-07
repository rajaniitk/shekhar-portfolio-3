import logging
import os
import pickle
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from models import Dataset, ModelTraining
from services.data_processor import DataProcessor
from services.ml_engine import MLEngine
from app import db, app

ml_bp = Blueprint('ml', __name__)

@ml_bp.route('/<int:dataset_id>')
def ml_models_page(dataset_id):
    """Machine learning models page"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            # Get column information for model training
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = list(df.columns)
            
            # Get existing models for this dataset
            existing_models = ModelTraining.query.filter_by(dataset_id=dataset_id).order_by(ModelTraining.created_date.desc()).all()
            
            existing_models_data = [model.to_dict() for model in existing_models]
            
            return render_template('ml_models.html', 
                                 dataset=dataset.to_dict(),
                                 numeric_columns=numeric_cols,
                                 categorical_columns=categorical_cols,
                                 all_columns=all_cols,
                                 existing_models=existing_models_data)
    except Exception as e:
        logging.error(f"ML models page error: {str(e)}")
    
    return render_template('ml_models.html', dataset=dataset.to_dict())

@ml_bp.route('/api/train/<int:dataset_id>', methods=['POST'])
def train_model(dataset_id):
    """Train a machine learning model"""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        problem_type = data.get('problem_type', 'classification')
        hyperparameters = data.get('hyperparameters', {})
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            ml_engine = MLEngine()
            
            # Train the model
            model_results = ml_engine.train_model(
                df=df,
                target_column=target_column,
                feature_columns=feature_columns,
                model_type=model_type,
                problem_type=problem_type,
                hyperparameters=hyperparameters
            )
            
            if model_results['success']:
                # Save model to disk
                models_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                model_filename = f"model_{dataset_id}_{model_type}_{len(ModelTraining.query.filter_by(dataset_id=dataset_id).all()) + 1}.pkl"
                model_path = os.path.join(models_dir, model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_results['model'], f)
                
                # Save model training record
                model_training = ModelTraining(
                    dataset_id=dataset_id,
                    model_type=model_type,
                    target_column=target_column,
                    model_path=model_path
                )
                model_training.set_features(feature_columns)
                model_training.set_hyperparameters(hyperparameters)
                model_training.set_performance_metrics(model_results['metrics'])
                
                db.session.add(model_training)
                db.session.commit()
                
                # Remove the actual model object from results before returning
                model_results.pop('model', None)
                model_results['model_id'] = model_training.id
                
                return jsonify(model_results)
            else:
                return jsonify(model_results), 400
            
    except Exception as e:
        logging.error(f"Model training error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@ml_bp.route('/api/evaluate/<int:model_id>')
def evaluate_model(model_id):
    """Evaluate a trained model"""
    try:
        model_training = ModelTraining.query.get_or_404(model_id)
        dataset = model_training.dataset
        
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            # Load the saved model
            with open(model_training.model_path, 'rb') as f:
                model = pickle.load(f)
            
            ml_engine = MLEngine()
            evaluation_results = ml_engine.evaluate_model(
                model=model,
                df=df,
                target_column=model_training.target_column,
                feature_columns=model_training.get_features()
            )
            
            return jsonify(evaluation_results)
            
    except Exception as e:
        logging.error(f"Model evaluation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ml_bp.route('/api/predict/<int:model_id>', methods=['POST'])
def make_predictions(model_id):
    """Make predictions with a trained model"""
    try:
        data = request.get_json()
        input_data = data.get('input_data', {})
        
        model_training = ModelTraining.query.get_or_404(model_id)
        
        # Load the saved model
        with open(model_training.model_path, 'rb') as f:
            model = pickle.load(f)
        
        ml_engine = MLEngine()
        predictions = ml_engine.make_predictions(
            model=model,
            input_data=input_data,
            feature_columns=model_training.get_features()
        )
        
        return jsonify(predictions)
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ml_bp.route('/api/compare/<int:dataset_id>')
def compare_models(dataset_id):
    """Compare multiple models for a dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        models = ModelTraining.query.filter_by(dataset_id=dataset_id).all()
        
        if not models:
            return jsonify({'error': 'No models found for this dataset'}), 404
        
        comparison_results = []
        for model in models:
            metrics = model.get_performance_metrics()
            comparison_results.append({
                'model_id': model.id,
                'model_type': model.model_type,
                'target_column': model.target_column,
                'metrics': metrics,
                'created_date': model.created_date.isoformat()
            })
        
        return jsonify(comparison_results)
        
    except Exception as e:
        logging.error(f"Model comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ml_bp.route('/api/feature_importance/<int:model_id>')
def get_feature_importance(model_id):
    """Get feature importance for a trained model"""
    try:
        model_training = ModelTraining.query.get_or_404(model_id)
        
        # Load the saved model
        with open(model_training.model_path, 'rb') as f:
            model = pickle.load(f)
        
        ml_engine = MLEngine()
        feature_importance = ml_engine.get_feature_importance(
            model=model,
            feature_columns=model_training.get_features()
        )
        
        return jsonify(feature_importance)
        
    except Exception as e:
        logging.error(f"Feature importance error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ml_bp.route('/api/hyperparameter_tuning/<int:dataset_id>', methods=['POST'])
def hyperparameter_tuning(dataset_id):
    """Perform hyperparameter tuning"""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        problem_type = data.get('problem_type', 'classification')
        param_grid = data.get('param_grid', {})
        
        dataset = Dataset.query.get_or_404(dataset_id)
        processor = DataProcessor()
        df, _ = processor.load_file(dataset.file_path)
        
        if df is not None:
            ml_engine = MLEngine()
            tuning_results = ml_engine.hyperparameter_tuning(
                df=df,
                target_column=target_column,
                feature_columns=feature_columns,
                model_type=model_type,
                problem_type=problem_type,
                param_grid=param_grid
            )
            
            return jsonify(tuning_results)
            
    except Exception as e:
        logging.error(f"Hyperparameter tuning error: {str(e)}")
        return jsonify({'error': str(e)}), 500
