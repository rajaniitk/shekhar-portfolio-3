import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
# try:
#     from lightgbm import LGBMClassifier
#     HAS_LGBM = True
# except ImportError:
HAS_LGBM = False

# Regression algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    pass
# try:
#     from lightgbm import LGBMRegressor
# except ImportError:
#     pass

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

class MLEngine:
    """Comprehensive Machine Learning Engine with 10+ algorithms"""
    
    def __init__(self):
        self.plotly_template = "plotly_dark"
        
        # Classification models
        self.classification_models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
        }
        
        if HAS_XGB:
            self.classification_models['xgboost'] = XGBClassifier(random_state=42)
        # if HAS_LGBM:
        #     self.classification_models['lightgbm'] = LGBMClassifier(random_state=42)
        
        # Regression models
        self.regression_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR(),
            'knn': KNeighborsRegressor(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
        }
        
        if HAS_XGB:
            self.regression_models['xgboost'] = XGBRegressor(random_state=42)
        # if HAS_LGBM:
        #     self.regression_models['lightgbm'] = LGBMRegressor(random_state=42)
    
    def preprocess_data(self, df, target_column, feature_columns):
        """Preprocess data for machine learning"""
        try:
            # Prepare features and target
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Remove rows with missing target values
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return None, None, None, {'error': 'No valid data after removing missing target values'}
            
            # Identify column types
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numeric_features),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]), categorical_features)
                ]
            )
            
            # Determine problem type
            if y.dtype in ['object', 'category'] or y.nunique() <= 20:
                problem_type = 'classification'
                # Encode target for classification
                if y.dtype in ['object', 'category']:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y.astype(str))
                    target_encoder = label_encoder
                else:
                    target_encoder = None
            else:
                problem_type = 'regression'
                target_encoder = None
            
            return X, y, preprocessor, {'problem_type': problem_type, 'target_encoder': target_encoder}
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            return None, None, None, {'error': str(e)}
    
    def train_model(self, df, target_column, feature_columns, model_type, problem_type='auto', hyperparameters=None):
        """Train a machine learning model"""
        try:
            # Preprocess data
            X, y, preprocessor, prep_info = self.preprocess_data(df, target_column, feature_columns)
            
            if X is None:
                return {'success': False, 'error': prep_info.get('error', 'Unknown preprocessing error')}
            
            # Determine problem type if auto
            if problem_type == 'auto':
                problem_type = prep_info['problem_type']
            
            # Select model
            if problem_type == 'classification':
                if model_type not in self.classification_models:
                    return {'success': False, 'error': f'Unknown classification model: {model_type}'}
                model = self.classification_models[model_type]
            else:
                if model_type not in self.regression_models:
                    return {'success': False, 'error': f'Unknown regression model: {model_type}'}
                model = self.regression_models[model_type]
            
            # Apply hyperparameters
            if hyperparameters:
                model.set_params(**hyperparameters)
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier' if problem_type == 'classification' else 'regressor', model)
            ])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'classification':
                metrics = self._calculate_classification_metrics(y_test, y_pred, pipeline, X_test)
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(pipeline, X, y, cv=5)
            metrics['cv_mean'] = float(cv_scores.mean())
            metrics['cv_std'] = float(cv_scores.std())
            
            return {
                'success': True,
                'model': pipeline,
                'metrics': metrics,
                'problem_type': problem_type,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'target_encoder': prep_info.get('target_encoder')
            }
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_classification_metrics(self, y_true, y_pred, pipeline, X_test):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
        
        # Additional metrics
        metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else None
        
        return metrics
    
    def evaluate_model(self, model, df, target_column, feature_columns):
        """Evaluate a trained model"""
        try:
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]
            
            # Remove missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return {'error': 'No valid data for evaluation'}
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Determine problem type
            if hasattr(model, 'predict_proba'):
                problem_type = 'classification'
                metrics = self._calculate_classification_metrics(y, y_pred, model, X)
                
                # Generate visualization data
                viz_data = self._create_classification_visualizations(y, y_pred, model, X)
                
            else:
                problem_type = 'regression'
                metrics = self._calculate_regression_metrics(y, y_pred)
                
                # Generate visualization data
                viz_data = self._create_regression_visualizations(y, y_pred)
            
            return {
                'metrics': metrics,
                'visualizations': viz_data,
                'problem_type': problem_type
            }
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            return {'error': str(e)}
    
    def _create_classification_visualizations(self, y_true, y_pred, model, X_test):
        """Create visualizations for classification results"""
        visualizations = {}
        
        try:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f'Predicted {i}' for i in range(len(cm))],
                y=[f'Actual {i}' for i in range(len(cm))],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                template=self.plotly_template
            )
            
            visualizations['confusion_matrix'] = fig_cm.to_json()
            
            # ROC Curve (for binary classification)
            if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc = roc_auc_score(y_true, y_proba)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {auc:.3f})'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Random Classifier'
                ))
                
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template=self.plotly_template
                )
                
                visualizations['roc_curve'] = fig_roc.to_json()
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name='Precision-Recall Curve'
                ))
                
                fig_pr.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    template=self.plotly_template
                )
                
                visualizations['precision_recall'] = fig_pr.to_json()
            
        except Exception as e:
            logging.error(f"Error creating classification visualizations: {str(e)}")
        
        return visualizations
    
    def _create_regression_visualizations(self, y_true, y_pred):
        """Create visualizations for regression results"""
        visualizations = {}
        
        try:
            # Actual vs Predicted
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=6, opacity=0.6)
            ))
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig_scatter.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                template=self.plotly_template
            )
            
            visualizations['actual_vs_predicted'] = fig_scatter.to_json()
            
            # Residuals plot
            residuals = y_true - y_pred
            
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, opacity=0.6)
            ))
            
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig_residuals.update_layout(
                title='Residuals Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                template=self.plotly_template
            )
            
            visualizations['residuals'] = fig_residuals.to_json()
            
            # Residuals histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residuals Distribution'
            ))
            
            fig_hist.update_layout(
                title='Distribution of Residuals',
                xaxis_title='Residuals',
                yaxis_title='Frequency',
                template=self.plotly_template
            )
            
            visualizations['residuals_histogram'] = fig_hist.to_json()
            
        except Exception as e:
            logging.error(f"Error creating regression visualizations: {str(e)}")
        
        return visualizations
    
    def make_predictions(self, model, input_data, feature_columns):
        """Make predictions with a trained model"""
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure all feature columns are present
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = None
            
            # Select only feature columns in correct order
            input_df = input_df[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_df)
            
            result = {
                'prediction': float(prediction[0]) if len(prediction) == 1 else prediction.tolist()
            }
            
            # Add probability for classification
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)
                result['probabilities'] = probabilities[0].tolist()
                result['classes'] = model.classes_.tolist() if hasattr(model, 'classes_') else None
            
            return result
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_importance(self, model, feature_columns):
        """Get feature importance from trained model"""
        try:
            # Extract the actual model from pipeline
            if hasattr(model, 'named_steps'):
                actual_model = model.named_steps.get('classifier') or model.named_steps.get('regressor')
            else:
                actual_model = model
            
            importance_data = {}
            
            # Get feature importance if available
            if hasattr(actual_model, 'feature_importances_'):
                # For tree-based models
                
                # Get feature names after preprocessing
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    try:
                        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                        importances = actual_model.feature_importances_
                        
                        # Map back to original feature names (approximate)
                        feature_importance_dict = {}
                        for orig_feature in feature_columns:
                            # Find all transformed features that came from this original feature
                            matching_indices = [i for i, name in enumerate(feature_names) 
                                              if orig_feature in str(name)]
                            
                            if matching_indices:
                                # Sum importance for all transformations of this feature
                                total_importance = sum(importances[i] for i in matching_indices)
                                feature_importance_dict[orig_feature] = float(total_importance)
                        
                        importance_data['feature_importance'] = feature_importance_dict
                        
                    except:
                        # Fallback to basic importance
                        importance_data['feature_importance'] = {
                            f'feature_{i}': float(imp) 
                            for i, imp in enumerate(actual_model.feature_importances_[:len(feature_columns)])
                        }
                
            elif hasattr(actual_model, 'coef_'):
                # For linear models
                coefficients = actual_model.coef_
                if coefficients.ndim > 1:
                    coefficients = coefficients[0]  # For multiclass, take first class
                
                importance_data['coefficients'] = {
                    feature: float(abs(coef)) 
                    for feature, coef in zip(feature_columns, coefficients[:len(feature_columns)])
                }
            
            # Create visualization
            if 'feature_importance' in importance_data:
                sorted_features = sorted(importance_data['feature_importance'].items(), 
                                       key=lambda x: x[1], reverse=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[imp for _, imp in sorted_features],
                    y=[feat for feat, _ in sorted_features],
                    orientation='h',
                    name='Feature Importance'
                ))
                
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance',
                    yaxis_title='Features',
                    template=self.plotly_template,
                    height=max(400, len(sorted_features) * 25)
                )
                
                importance_data['visualization'] = fig.to_json()
            
            return importance_data
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return {'error': str(e)}
    
    def hyperparameter_tuning(self, df, target_column, feature_columns, model_type, problem_type='auto', param_grid=None):
        """Perform hyperparameter tuning"""
        try:
            # Preprocess data
            X, y, preprocessor, prep_info = self.preprocess_data(df, target_column, feature_columns)
            
            if X is None:
                return {'error': prep_info.get('error', 'Unknown preprocessing error')}
            
            # Determine problem type
            if problem_type == 'auto':
                problem_type = prep_info['problem_type']
            
            # Select model
            if problem_type == 'classification':
                if model_type not in self.classification_models:
                    return {'error': f'Unknown classification model: {model_type}'}
                model = self.classification_models[model_type]
            else:
                if model_type not in self.regression_models:
                    return {'error': f'Unknown regression model: {model_type}'}
                model = self.regression_models[model_type]
            
            # Default parameter grids
            default_param_grids = {
                'random_forest': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                },
                'svm': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['rbf', 'linear'],
                    'classifier__gamma': ['scale', 'auto']
                },
                'logistic_regression': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            }
            
            # Adapt for regression
            if problem_type == 'regression':
                for key in list(default_param_grids.keys()):
                    grid = default_param_grids[key]
                    new_grid = {}
                    for param, values in grid.items():
                        new_param = param.replace('classifier__', 'regressor__')
                        new_grid[new_param] = values
                    default_param_grids[key] = new_grid
            
            # Use provided param_grid or default
            if param_grid is None:
                param_grid = default_param_grids.get(model_type, {})
            
            if not param_grid:
                return {'error': f'No parameter grid available for {model_type}'}
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier' if problem_type == 'classification' else 'regressor', model)
            ])
            
            # Perform grid search
            scoring = 'accuracy' if problem_type == 'classification' else 'r2'
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=3,  # Reduced for performance
                scoring=scoring,
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Return results
            results = {
                'best_parameters': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_model': grid_search.best_estimator_,
                'cv_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'parameters': [dict(params) for params in grid_search.cv_results_['params']]
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in hyperparameter tuning: {str(e)}")
            return {'error': str(e)}
