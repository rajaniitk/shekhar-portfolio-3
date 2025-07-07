from app import db
from datetime import datetime
import json

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    shape_rows = db.Column(db.Integer)
    shape_cols = db.Column(db.Integer)
    memory_usage = db.Column(db.Float)
    column_info = db.Column(db.Text)  # JSON string of column metadata
    
    def set_column_info(self, info_dict):
        self.column_info = json.dumps(info_dict)
    
    def get_column_info(self):
        if self.column_info:
            return json.loads(self.column_info)
        return {}

    def to_dict(self):
        """Convert Dataset to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'shape_rows': self.shape_rows,
            'shape_cols': self.shape_cols,
            'memory_usage': self.memory_usage,
            'column_info': self.get_column_info()
        }

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    analysis_type = db.Column(db.String(100), nullable=False)  # 'eda', 'statistical_test', 'ml_model'
    results = db.Column(db.Text)  # JSON string of analysis results
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    dataset = db.relationship('Dataset', backref=db.backref('analyses', lazy=True))
    
    def set_results(self, results_dict):
        self.results = json.dumps(results_dict)
    
    def get_results(self):
        if self.results:
            return json.loads(self.results)
        return {}

    def to_dict(self):
        """Convert Analysis to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'analysis_type': self.analysis_type,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'results': self.get_results(),
            'dataset': {
                'filename': self.dataset.filename if self.dataset else 'Unknown'
            }
        }

class ModelTraining(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)
    target_column = db.Column(db.String(255), nullable=False)
    features = db.Column(db.Text)  # JSON array of feature column names
    hyperparameters = db.Column(db.Text)  # JSON string of hyperparameters
    performance_metrics = db.Column(db.Text)  # JSON string of metrics
    model_path = db.Column(db.String(500))  # Path to saved model
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    dataset = db.relationship('Dataset', backref=db.backref('models', lazy=True))
    
    def set_features(self, features_list):
        self.features = json.dumps(features_list)
    
    def get_features(self):
        if self.features:
            return json.loads(self.features)
        return []
    
    def set_hyperparameters(self, params_dict):
        self.hyperparameters = json.dumps(params_dict)
    
    def get_hyperparameters(self):
        if self.hyperparameters:
            return json.loads(self.hyperparameters)
        return {}
    
    def set_performance_metrics(self, metrics_dict):
        self.performance_metrics = json.dumps(metrics_dict)
    
    def get_performance_metrics(self):
        if self.performance_metrics:
            return json.loads(self.performance_metrics)
        return {}
