import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

class DataProcessor:
    """Handle data loading and basic preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'json', 'parquet']
    
    def load_file(self, file_path):
        """Load file and return DataFrame with metadata"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
                file_type = 'csv'
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                file_type = 'excel'
            elif file_extension == '.json':
                df = pd.read_json(file_path)
                file_type = 'json'
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
                file_type = 'parquet'
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Generate metadata
            file_info = {
                'type': file_type,
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                'columns': self._get_column_info(df)
            }
            
            return df, file_info
            
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
            return None, None
    
    def _get_column_info(self, df):
        """Get detailed information about each column"""
        column_info = {}
        
        for col in df.columns:
            column_info[col] = {
                'dtype': str(df[col].dtype),
                'unique_count': int(df[col].nunique()),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'memory_usage': float(df[col].memory_usage(deep=True) / 1024),  # KB
            }
            
            # Add type-specific information
            if df[col].dtype in ['int64', 'float64']:
                column_info[col].update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'skewness': float(df[col].skew()) if not df[col].isnull().all() else None,
                    'kurtosis': float(df[col].kurtosis()) if not df[col].isnull().all() else None
                })
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                column_info[col].update({
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                    'average_length': float(df[col].astype(str).str.len().mean()) if not df[col].isnull().all() else None
                })
        
        return column_info
    
    def clean_data(self, df, operations):
        """Apply data cleaning operations"""
        cleaned_df = df.copy()
        
        for operation in operations:
            if operation['type'] == 'drop_nulls':
                if 'columns' in operation:
                    cleaned_df = cleaned_df.dropna(subset=operation['columns'])
                else:
                    cleaned_df = cleaned_df.dropna()
            
            elif operation['type'] == 'fill_nulls':
                if operation['method'] == 'mean':
                    for col in operation['columns']:
                        if col in cleaned_df.columns:
                            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif operation['method'] == 'median':
                    for col in operation['columns']:
                        if col in cleaned_df.columns:
                            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif operation['method'] == 'mode':
                    for col in operation['columns']:
                        if col in cleaned_df.columns:
                            mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                            cleaned_df[col].fillna(mode_value, inplace=True)
                elif operation['method'] == 'constant':
                    for col in operation['columns']:
                        if col in cleaned_df.columns:
                            cleaned_df[col].fillna(operation['value'], inplace=True)
            
            elif operation['type'] == 'remove_duplicates':
                cleaned_df = cleaned_df.drop_duplicates()
            
            elif operation['type'] == 'remove_outliers':
                for col in operation['columns']:
                    if col in cleaned_df.columns and cleaned_df[col].dtype in ['int64', 'float64']:
                        Q1 = cleaned_df[col].quantile(0.25)
                        Q3 = cleaned_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
        
        return cleaned_df
    
    def get_data_quality_report(self, df):
        """Generate a comprehensive data quality report"""
        report = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum(),
            'total_null_values': df.isnull().sum().sum(),
            'columns_with_nulls': df.isnull().any().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'column_details': {}
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'zeros_count': (df[col] == 0).sum(),
                    'negative_count': (df[col] < 0).sum(),
                    'infinite_count': np.isinf(df[col]).sum()
                })
            
            report['column_details'][col] = col_info
        
        return report
