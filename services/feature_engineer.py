import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, KBinsDiscretizer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
    RFE, RFECV
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from scipy.stats import boxcox
import logging
import json

class FeatureEngineer:
    """Comprehensive feature engineering and transformation engine"""
    
    def __init__(self):
        self.transformations_applied = []
        self.feature_importance_scores = {}
    
    def analyze_features(self, df):
        """Analyze features and suggest transformations"""
        try:
            suggestions = {
                'numeric_transformations': {},
                'categorical_transformations': {},
                'feature_creation': {},
                'feature_selection': {},
                'data_quality': {}
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Analyze numeric features
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                col_suggestions = []
                
                # Check for skewness
                skewness = col_data.skew()
                if abs(skewness) > 1:
                    if skewness > 1:
                        col_suggestions.append({
                            'type': 'log_transform',
                            'reason': f'High positive skewness ({skewness:.2f})',
                            'priority': 'high'
                        })
                    else:
                        col_suggestions.append({
                            'type': 'square_transform',
                            'reason': f'High negative skewness ({skewness:.2f})',
                            'priority': 'medium'
                        })
                
                # Check for outliers
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = len(col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)])
                outlier_pct = (outliers / len(col_data)) * 100
                
                if outlier_pct > 5:
                    col_suggestions.append({
                        'type': 'robust_scaling',
                        'reason': f'Contains {outlier_pct:.1f}% outliers',
                        'priority': 'high'
                    })
                
                # Check scale
                if col_data.std() > 100 or col_data.max() > 1000:
                    col_suggestions.append({
                        'type': 'standard_scaling',
                        'reason': 'Large scale values',
                        'priority': 'medium'
                    })
                
                # Check for zero inflation
                zero_pct = (col_data == 0).sum() / len(col_data) * 100
                if zero_pct > 50:
                    col_suggestions.append({
                        'type': 'zero_inflation_handling',
                        'reason': f'{zero_pct:.1f}% zero values',
                        'priority': 'medium'
                    })
                
                suggestions['numeric_transformations'][col] = col_suggestions
            
            # Analyze categorical features
            for col in categorical_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                col_suggestions = []
                unique_count = col_data.nunique()
                
                # High cardinality
                if unique_count > 50:
                    col_suggestions.append({
                        'type': 'group_rare_categories',
                        'reason': f'High cardinality ({unique_count} unique values)',
                        'priority': 'high'
                    })
                elif unique_count > 10:
                    col_suggestions.append({
                        'type': 'target_encoding',
                        'reason': f'Medium cardinality ({unique_count} unique values)',
                        'priority': 'medium'
                    })
                else:
                    col_suggestions.append({
                        'type': 'one_hot_encoding',
                        'reason': f'Low cardinality ({unique_count} unique values)',
                        'priority': 'low'
                    })
                
                # Check for rare categories
                value_counts = col_data.value_counts()
                rare_categories = (value_counts / len(col_data) < 0.01).sum()
                if rare_categories > 0:
                    col_suggestions.append({
                        'type': 'group_rare_categories',
                        'reason': f'{rare_categories} categories with <1% frequency',
                        'priority': 'medium'
                    })
                
                suggestions['categorical_transformations'][col] = col_suggestions
            
            # Feature creation suggestions
            feature_creation = []
            
            # Polynomial features for numeric data
            if len(numeric_cols) >= 2:
                feature_creation.append({
                    'type': 'polynomial_features',
                    'reason': 'Multiple numeric features available for interaction',
                    'priority': 'medium'
                })
            
            # Binning for numeric features
            for col in numeric_cols:
                if df[col].nunique() > 20:
                    feature_creation.append({
                        'type': 'binning',
                        'column': col,
                        'reason': f'High unique values ({df[col].nunique()})',
                        'priority': 'low'
                    })
            
            suggestions['feature_creation'] = feature_creation
            
            # Feature selection suggestions
            if len(df.columns) > 50:
                suggestions['feature_selection']['dimensionality_reduction'] = {
                    'type': 'pca',
                    'reason': f'High dimensionality ({len(df.columns)} features)',
                    'priority': 'high'
                }
            
            # Data quality issues
            data_quality = []
            
            # Missing values
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            for col, missing_count in missing_cols.items():
                missing_pct = (missing_count / len(df)) * 100
                if missing_pct > 50:
                    data_quality.append({
                        'type': 'drop_column',
                        'column': col,
                        'reason': f'{missing_pct:.1f}% missing values',
                        'priority': 'high'
                    })
                elif missing_pct > 20:
                    data_quality.append({
                        'type': 'advanced_imputation',
                        'column': col,
                        'reason': f'{missing_pct:.1f}% missing values',
                        'priority': 'medium'
                    })
                else:
                    data_quality.append({
                        'type': 'simple_imputation',
                        'column': col,
                        'reason': f'{missing_pct:.1f}% missing values',
                        'priority': 'low'
                    })
            
            suggestions['data_quality'] = data_quality
            
            return suggestions
            
        except Exception as e:
            logging.error(f"Error analyzing features: {str(e)}")
            return {'error': str(e)}
    
    def apply_numeric_transformations(self, df, transformations):
        """Apply numeric transformations to dataframe"""
        try:
            df_transformed = df.copy()
            transformation_log = []
            
            for transformation in transformations:
                transform_type = transformation['type']
                columns = transformation.get('columns', [])
                
                if transform_type == 'standard_scaling':
                    scaler = StandardScaler()
                    for col in columns:
                        if col in df_transformed.columns:
                            col_data = df_transformed[col].values.reshape(-1, 1)
                            df_transformed[col] = scaler.fit_transform(col_data).flatten()
                            transformation_log.append(f"Applied standard scaling to {col}")
                
                elif transform_type == 'min_max_scaling':
                    scaler = MinMaxScaler()
                    for col in columns:
                        if col in df_transformed.columns:
                            col_data = df_transformed[col].values.reshape(-1, 1)
                            df_transformed[col] = scaler.fit_transform(col_data).flatten()
                            transformation_log.append(f"Applied min-max scaling to {col}")
                
                elif transform_type == 'robust_scaling':
                    scaler = RobustScaler()
                    for col in columns:
                        if col in df_transformed.columns:
                            col_data = df_transformed[col].values.reshape(-1, 1)
                            df_transformed[col] = scaler.fit_transform(col_data).flatten()
                            transformation_log.append(f"Applied robust scaling to {col}")
                
                elif transform_type == 'log_transform':
                    for col in columns:
                        if col in df_transformed.columns:
                            # Add small constant to handle zeros
                            df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                            transformation_log.append(f"Applied log transform to {col}")
                
                elif transform_type == 'sqrt_transform':
                    for col in columns:
                        if col in df_transformed.columns:
                            df_transformed[f'{col}_sqrt'] = np.sqrt(np.abs(df_transformed[col]))
                            transformation_log.append(f"Applied square root transform to {col}")
                
                elif transform_type == 'box_cox':
                    for col in columns:
                        if col in df_transformed.columns:
                            col_data = df_transformed[col].dropna()
                            if (col_data > 0).all():
                                transformed_data, _ = boxcox(col_data)
                                df_transformed.loc[col_data.index, f'{col}_boxcox'] = transformed_data
                                transformation_log.append(f"Applied Box-Cox transform to {col}")
                
                elif transform_type == 'binning':
                    n_bins = transformation.get('n_bins', 5)
                    strategy = transformation.get('strategy', 'uniform')
                    
                    for col in columns:
                        if col in df_transformed.columns:
                            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                            col_data = df_transformed[col].values.reshape(-1, 1)
                            df_transformed[f'{col}_binned'] = discretizer.fit_transform(col_data).flatten()
                            transformation_log.append(f"Applied binning to {col} ({n_bins} bins)")
                
                elif transform_type == 'polynomial_features':
                    degree = transformation.get('degree', 2)
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    
                    if columns:
                        col_data = df_transformed[columns]
                        poly_features = poly.fit_transform(col_data)
                        feature_names = poly.get_feature_names_out(columns)
                        
                        # Add new polynomial features
                        for i, name in enumerate(feature_names):
                            if name not in columns:  # Skip original features
                                df_transformed[name] = poly_features[:, i]
                        
                        transformation_log.append(f"Created polynomial features (degree {degree}) for {columns}")
            
            return {
                'transformed_data': df_transformed,
                'transformation_log': transformation_log,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error applying numeric transformations: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def apply_categorical_transformations(self, df, transformations):
        """Apply categorical transformations to dataframe"""
        try:
            df_transformed = df.copy()
            transformation_log = []
            encoders = {}
            
            for transformation in transformations:
                transform_type = transformation['type']
                columns = transformation.get('columns', [])
                
                if transform_type == 'one_hot_encoding':
                    for col in columns:
                        if col in df_transformed.columns:
                            # Create dummy variables
                            dummies = pd.get_dummies(df_transformed[col], prefix=col)
                            df_transformed = pd.concat([df_transformed, dummies], axis=1)
                            df_transformed.drop(col, axis=1, inplace=True)
                            transformation_log.append(f"Applied one-hot encoding to {col}")
                
                elif transform_type == 'label_encoding':
                    for col in columns:
                        if col in df_transformed.columns:
                            encoder = LabelEncoder()
                            df_transformed[f'{col}_encoded'] = encoder.fit_transform(df_transformed[col].astype(str))
                            encoders[col] = encoder
                            transformation_log.append(f"Applied label encoding to {col}")
                
                elif transform_type == 'ordinal_encoding':
                    categories = transformation.get('categories', None)
                    for col in columns:
                        if col in df_transformed.columns:
                            if categories and col in categories:
                                encoder = OrdinalEncoder(categories=[categories[col]])
                            else:
                                encoder = OrdinalEncoder()
                            
                            col_data = df_transformed[col].values.reshape(-1, 1)
                            df_transformed[f'{col}_ordinal'] = encoder.fit_transform(col_data).flatten()
                            encoders[col] = encoder
                            transformation_log.append(f"Applied ordinal encoding to {col}")
                
                elif transform_type == 'frequency_encoding':
                    for col in columns:
                        if col in df_transformed.columns:
                            freq_map = df_transformed[col].value_counts().to_dict()
                            df_transformed[f'{col}_freq'] = df_transformed[col].map(freq_map)
                            transformation_log.append(f"Applied frequency encoding to {col}")
                
                elif transform_type == 'group_rare_categories':
                    threshold = transformation.get('threshold', 0.01)
                    for col in columns:
                        if col in df_transformed.columns:
                            value_counts = df_transformed[col].value_counts()
                            rare_categories = value_counts[value_counts / len(df_transformed) < threshold].index
                            df_transformed[col] = df_transformed[col].replace(rare_categories, 'Other')
                            transformation_log.append(f"Grouped {len(rare_categories)} rare categories in {col}")
                
                elif transform_type == 'target_encoding':
                    target_col = transformation.get('target_column')
                    if target_col and target_col in df_transformed.columns:
                        for col in columns:
                            if col in df_transformed.columns:
                                target_mean = df_transformed.groupby(col)[target_col].mean()
                                df_transformed[f'{col}_target_encoded'] = df_transformed[col].map(target_mean)
                                transformation_log.append(f"Applied target encoding to {col}")
            
            return {
                'transformed_data': df_transformed,
                'transformation_log': transformation_log,
                'encoders': encoders,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error applying categorical transformations: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def handle_missing_values(self, df, strategy='auto'):
        """Handle missing values in the dataset"""
        try:
            df_imputed = df.copy()
            imputation_log = []
            
            missing_summary = df.isnull().sum()
            missing_cols = missing_summary[missing_summary > 0]
            
            if len(missing_cols) == 0:
                return {
                    'imputed_data': df_imputed,
                    'imputation_log': ['No missing values found'],
                    'success': True
                }
            
            for col in missing_cols.index:
                missing_pct = (missing_cols[col] / len(df)) * 100
                
                if strategy == 'auto':
                    # Auto-select strategy based on data type and missing percentage
                    if missing_pct > 50:
                        # Drop columns with >50% missing
                        df_imputed.drop(col, axis=1, inplace=True)
                        imputation_log.append(f"Dropped {col} ({missing_pct:.1f}% missing)")
                    elif df[col].dtype in ['object', 'category']:
                        # Most frequent for categorical
                        imputer = SimpleImputer(strategy='most_frequent')
                        df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                        imputation_log.append(f"Imputed {col} with most frequent value")
                    else:
                        # Median for numeric
                        if missing_pct > 20:
                            # KNN imputation for high missing percentage
                            imputer = KNNImputer(n_neighbors=5)
                            df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                            imputation_log.append(f"Imputed {col} with KNN (5 neighbors)")
                        else:
                            # Simple median imputation
                            imputer = SimpleImputer(strategy='median')
                            df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                            imputation_log.append(f"Imputed {col} with median")
                
                elif strategy == 'drop':
                    df_imputed.drop(col, axis=1, inplace=True)
                    imputation_log.append(f"Dropped {col}")
                
                elif strategy == 'mean':
                    if df[col].dtype in ['int64', 'float64']:
                        imputer = SimpleImputer(strategy='mean')
                        df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                        imputation_log.append(f"Imputed {col} with mean")
                
                elif strategy == 'median':
                    if df[col].dtype in ['int64', 'float64']:
                        imputer = SimpleImputer(strategy='median')
                        df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                        imputation_log.append(f"Imputed {col} with median")
                
                elif strategy == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                    imputation_log.append(f"Imputed {col} with mode")
                
                elif strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                    imputation_log.append(f"Imputed {col} with KNN")
            
            return {
                'imputed_data': df_imputed,
                'imputation_log': imputation_log,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error handling missing values: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def feature_selection(self, df, target_column, method='auto', k=10):
        """Perform feature selection"""
        try:
            if target_column not in df.columns:
                return {'error': f'Target column {target_column} not found'}
            
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            # Remove non-numeric columns for now (can be enhanced later)
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features available for selection'}
            
            results = {}
            
            # Determine problem type
            if y.dtype in ['object', 'category'] or y.nunique() <= 20:
                problem_type = 'classification'
                score_func = f_classif
                mi_func = mutual_info_classif
            else:
                problem_type = 'regression'
                score_func = f_regression
                mi_func = mutual_info_regression
            
            if method == 'auto' or method == 'univariate':
                # Univariate feature selection
                selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_X.columns)))
                X_selected = selector.fit_transform(numeric_X, y)
                selected_features = numeric_X.columns[selector.get_support()].tolist()
                scores = selector.scores_
                
                results['univariate'] = {
                    'selected_features': selected_features,
                    'scores': dict(zip(numeric_X.columns, scores)),
                    'method': 'univariate'
                }
            
            if method == 'auto' or method == 'mutual_info':
                # Mutual information
                mi_scores = mi_func(numeric_X, y)
                mi_results = dict(zip(numeric_X.columns, mi_scores))
                top_mi_features = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)[:k]
                
                results['mutual_info'] = {
                    'selected_features': [feat for feat, _ in top_mi_features],
                    'scores': mi_results,
                    'method': 'mutual_information'
                }
            
            if method == 'auto' or method == 'correlation':
                # Correlation-based selection
                correlations = numeric_X.corrwith(y).abs()
                top_corr_features = correlations.nlargest(k)
                
                results['correlation'] = {
                    'selected_features': top_corr_features.index.tolist(),
                    'scores': top_corr_features.to_dict(),
                    'method': 'correlation'
                }
            
            # Remove highly correlated features
            if method == 'auto' or method == 'remove_correlated':
                corr_matrix = numeric_X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
                remaining_features = [col for col in numeric_X.columns if col not in to_drop]
                
                results['remove_correlated'] = {
                    'selected_features': remaining_features,
                    'dropped_features': to_drop,
                    'method': 'remove_highly_correlated'
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            return {'error': str(e)}
    
    def create_interaction_features(self, df, feature_pairs=None, max_pairs=10):
        """Create interaction features between numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'error': 'At least 2 numeric columns required for interactions'}
            
            df_interactions = df.copy()
            interaction_log = []
            
            if feature_pairs is None:
                # Auto-select top correlated pairs
                corr_matrix = df[numeric_cols].corr().abs()
                
                # Get upper triangle of correlation matrix
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Find pairs with moderate correlation (0.3 < |r| < 0.9)
                pairs = []
                for col in upper_triangle.columns:
                    for row in upper_triangle.index:
                        corr_val = upper_triangle.loc[row, col]
                        if 0.3 < corr_val < 0.9:
                            pairs.append((row, col, corr_val))
                
                # Sort by correlation and take top pairs
                pairs.sort(key=lambda x: x[2], reverse=True)
                feature_pairs = [(pair[0], pair[1]) for pair in pairs[:max_pairs]]
            
            # Create interaction features
            for col1, col2 in feature_pairs:
                if col1 in df.columns and col2 in df.columns:
                    # Multiplication
                    df_interactions[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    interaction_log.append(f"Created multiplication: {col1} × {col2}")
                    
                    # Addition
                    df_interactions[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    interaction_log.append(f"Created addition: {col1} + {col2}")
                    
                    # Ratio (if col2 has no zeros)
                    if (df[col2] != 0).all():
                        df_interactions[f'{col1}_div_{col2}'] = df[col1] / df[col2]
                        interaction_log.append(f"Created ratio: {col1} / {col2}")
                    
                    # Difference
                    df_interactions[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    interaction_log.append(f"Created difference: {col1} - {col2}")
            
            return {
                'data_with_interactions': df_interactions,
                'interaction_log': interaction_log,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error creating interaction features: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def dimensionality_reduction(self, df, method='pca', n_components=None):
        """Apply dimensionality reduction techniques"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'error': 'No numeric columns for dimensionality reduction'}
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            numeric_data = imputer.fit_transform(numeric_df)
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            if method == 'pca':
                if n_components is None:
                    n_components = min(10, scaled_data.shape[1])
                
                pca = PCA(n_components=n_components)
                transformed_data = pca.fit_transform(scaled_data)
                
                # Create DataFrame with PCA components
                pca_df = pd.DataFrame(
                    transformed_data,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=df.index
                )
                
                # Add non-numeric columns back
                non_numeric_df = df.select_dtypes(exclude=[np.number])
                result_df = pd.concat([pca_df, non_numeric_df], axis=1)
                
                return {
                    'transformed_data': result_df,
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': pca.components_.tolist(),
                    'feature_names': numeric_df.columns.tolist(),
                    'method': 'pca',
                    'success': True
                }
            
            else:
                return {'error': f'Unknown dimensionality reduction method: {method}'}
            
        except Exception as e:
            logging.error(f"Error in dimensionality reduction: {str(e)}")
            return {'error': str(e), 'success': False}
