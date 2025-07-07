import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

class EDAEngine:
    """Comprehensive Exploratory Data Analysis Engine"""
    
    def __init__(self):
        pass
    
    def generate_basic_statistics(self, df):
        """Generate basic statistical summary"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(include=['object', 'category'])
            
            basic_stats = {
                'dataset_overview': {
                    'shape': df.shape,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'duplicate_rows': int(df.duplicated().sum()),
                    'total_missing_values': int(df.isnull().sum().sum()),
                    'missing_percentage': float((df.isnull().sum().sum() / df.size) * 100)
                },
                'column_types': {
                    'numeric': len(numeric_df.columns),
                    'categorical': len(categorical_df.columns),
                    'datetime': len(df.select_dtypes(include=['datetime']).columns)
                }
            }
            
            # Numeric statistics
            if not numeric_df.empty:
                basic_stats['numeric_summary'] = numeric_df.describe().to_dict()
                
                # Add skewness and kurtosis
                basic_stats['distribution_metrics'] = {}
                for col in numeric_df.columns:
                    basic_stats['distribution_metrics'][col] = {
                        'skewness': float(numeric_df[col].skew()),
                        'kurtosis': float(numeric_df[col].kurtosis()),
                        'coefficient_of_variation': float(numeric_df[col].std() / numeric_df[col].mean()) if numeric_df[col].mean() != 0 else None
                    }
            
            # Categorical statistics
            if not categorical_df.empty:
                basic_stats['categorical_summary'] = {}
                for col in categorical_df.columns:
                    value_counts = categorical_df[col].value_counts()
                    basic_stats['categorical_summary'][col] = {
                        'unique_count': int(categorical_df[col].nunique()),
                        'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'top_5_values': value_counts.head().to_dict()
                    }
            
            return basic_stats
            
        except Exception as e:
            logging.error(f"Error generating basic statistics: {str(e)}")
            return {'error': str(e)}
    
    def analyze_missing_values(self, df):
        """Comprehensive missing value analysis"""
        try:
            missing_analysis = {
                'summary': {
                    'total_missing': int(df.isnull().sum().sum()),
                    'percentage_missing': float((df.isnull().sum().sum() / df.size) * 100),
                    'columns_with_missing': int(df.isnull().any().sum()),
                    'complete_rows': int(df.dropna().shape[0])
                },
                'by_column': {},
                'missing_patterns': {}
            }
            
            # Column-wise missing analysis
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_analysis['by_column'][col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float((missing_count / len(df)) * 100),
                    'data_type': str(df[col].dtype)
                }
            
            # Missing value patterns
            missing_combinations = df.isnull().value_counts()
            pattern_count = 0
            for pattern, count in missing_combinations.items():
                if pattern_count >= 10:  # Limit to top 10 patterns
                    break
                pattern_key = f"pattern_{pattern_count + 1}"
                missing_analysis['missing_patterns'][pattern_key] = {
                    'pattern': dict(zip(df.columns, pattern)),
                    'count': int(count),
                    'percentage': float((count / len(df)) * 100)
                }
                pattern_count += 1
            
            return missing_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing missing values: {str(e)}")
            return {'error': str(e)}
    
    def correlation_analysis(self, df):
        """Comprehensive correlation analysis"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {'error': 'Insufficient numeric columns for correlation analysis'}
            
            correlation_analysis = {
                'pearson_correlation': numeric_df.corr(method='pearson').to_dict(),
                'spearman_correlation': numeric_df.corr(method='spearman').to_dict(),
                'kendall_correlation': numeric_df.corr(method='kendall').to_dict(),
                'high_correlations': [],
                'multicollinearity_warning': []
            }
            
            # Find high correlations
            pearson_corr = numeric_df.corr(method='pearson')
            high_corr_pairs = []
            
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    corr_value = pearson_corr.iloc[i, j]
                    if abs(corr_value) >= 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'variable_1': pearson_corr.columns[i],
                            'variable_2': pearson_corr.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'Strong' if abs(corr_value) >= 0.8 else 'Moderate'
                        })
            
            correlation_analysis['high_correlations'] = high_corr_pairs
            
            # Multicollinearity warnings
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    corr_value = pearson_corr.iloc[i, j]
                    if abs(corr_value) >= 0.9:
                        correlation_analysis['multicollinearity_warning'].append({
                            'variable_1': pearson_corr.columns[i],
                            'variable_2': pearson_corr.columns[j],
                            'correlation': float(corr_value)
                        })
            
            return correlation_analysis
            
        except Exception as e:
            logging.error(f"Error in correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def detect_outliers(self, df):
        """Comprehensive outlier detection"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'error': 'No numeric columns for outlier detection'}
            
            outlier_analysis = {
                'by_column': {},
                'summary': {
                    'total_outliers': 0,
                    'affected_columns': 0
                }
            }
            
            total_outliers = 0
            affected_columns = 0
            
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # Z-score method (threshold = 3)
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = col_data[z_scores > 3]
                
                # Modified Z-score method
                median = np.median(col_data)
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad if mad != 0 else np.zeros_like(col_data)
                modified_z_outliers = col_data[np.abs(modified_z_scores) > 3.5]
                
                outlier_analysis['by_column'][col] = {
                    'iqr_outliers': {
                        'count': len(iqr_outliers),
                        'percentage': float((len(iqr_outliers) / len(col_data)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    },
                    'z_score_outliers': {
                        'count': len(z_outliers),
                        'percentage': float((len(z_outliers) / len(col_data)) * 100)
                    },
                    'modified_z_outliers': {
                        'count': len(modified_z_outliers),
                        'percentage': float((len(modified_z_outliers) / len(col_data)) * 100)
                    }
                }
                
                if len(iqr_outliers) > 0:
                    total_outliers += len(iqr_outliers)
                    affected_columns += 1
            
            outlier_analysis['summary']['total_outliers'] = total_outliers
            outlier_analysis['summary']['affected_columns'] = affected_columns
            
            return outlier_analysis
            
        except Exception as e:
            logging.error(f"Error detecting outliers: {str(e)}")
            return {'error': str(e)}
    
    def analyze_distributions(self, df):
        """Analyze distributions of numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'error': 'No numeric columns for distribution analysis'}
            
            distribution_analysis = {
                'by_column': {}
            }
            
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # Basic distribution metrics
                distribution_metrics = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'mode': float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    'std': float(col_data.std()),
                    'variance': float(col_data.var()),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis()),
                    'range': float(col_data.max() - col_data.min()),
                    'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
                }
                
                # Normality tests
                normality_tests = {}
                
                if len(col_data) >= 3:
                    # Shapiro-Wilk test (for small samples)
                    if len(col_data) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(col_data)
                        normality_tests['shapiro_wilk'] = {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': bool(shapiro_p > 0.05)
                        }
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                    normality_tests['kolmogorov_smirnov'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'is_normal': bool(ks_p > 0.05)
                    }
                    
                    # Anderson-Darling test
                    ad_result = stats.anderson(col_data, dist='norm')
                    normality_tests['anderson_darling'] = {
                        'statistic': float(ad_result.statistic),
                        'critical_values': ad_result.critical_values.tolist(),
                        'significance_levels': ad_result.significance_level.tolist()
                    }
                
                distribution_analysis['by_column'][col] = {
                    'metrics': distribution_metrics,
                    'normality_tests': normality_tests
                }
            
            return distribution_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing distributions: {str(e)}")
            return {'error': str(e)}
    
    def analyze_duplicates(self, df):
        """Analyze duplicate rows and values"""
        try:
            duplicate_analysis = {
                'summary': {
                    'total_rows': len(df),
                    'duplicate_rows': int(df.duplicated().sum()),
                    'unique_rows': int(len(df) - df.duplicated().sum()),
                    'duplicate_percentage': float((df.duplicated().sum() / len(df)) * 100)
                },
                'by_column': {}
            }
            
            # Column-wise duplicate analysis
            for col in df.columns:
                col_duplicates = df[col].duplicated().sum()
                duplicate_analysis['by_column'][col] = {
                    'duplicate_values': int(col_duplicates),
                    'unique_values': int(df[col].nunique()),
                    'duplicate_percentage': float((col_duplicates / len(df)) * 100)
                }
            
            return duplicate_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing duplicates: {str(e)}")
            return {'error': str(e)}
    
    def analyze_single_column(self, df, column):
        """Comprehensive analysis of a single column"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found in dataset'}
            
            col_data = df[column]
            analysis = {
                'basic_info': {
                    'column_name': column,
                    'data_type': str(col_data.dtype),
                    'total_count': len(col_data),
                    'non_null_count': int(col_data.count()),
                    'null_count': int(col_data.isnull().sum()),
                    'null_percentage': float((col_data.isnull().sum() / len(col_data)) * 100),
                    'unique_count': int(col_data.nunique()),
                    'unique_percentage': float((col_data.nunique() / len(col_data)) * 100)
                }
            }
            
            # Numeric column analysis
            if col_data.dtype in ['int64', 'float64']:
                non_null_data = col_data.dropna()
                
                if len(non_null_data) > 0:
                    analysis['numeric_stats'] = {
                        'mean': float(non_null_data.mean()),
                        'median': float(non_null_data.median()),
                        'mode': float(non_null_data.mode().iloc[0]) if not non_null_data.mode().empty else None,
                        'std': float(non_null_data.std()),
                        'variance': float(non_null_data.var()),
                        'min': float(non_null_data.min()),
                        'max': float(non_null_data.max()),
                        'range': float(non_null_data.max() - non_null_data.min()),
                        'q25': float(non_null_data.quantile(0.25)),
                        'q75': float(non_null_data.quantile(0.75)),
                        'iqr': float(non_null_data.quantile(0.75) - non_null_data.quantile(0.25)),
                        'skewness': float(non_null_data.skew()),
                        'kurtosis': float(non_null_data.kurtosis())
                    }
                    
                    # Outlier detection
                    Q1 = non_null_data.quantile(0.25)
                    Q3 = non_null_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
                    
                    analysis['outliers'] = {
                        'count': len(outliers),
                        'percentage': float((len(outliers) / len(non_null_data)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
            
            # Categorical column analysis
            elif col_data.dtype in ['object', 'category']:
                value_counts = col_data.value_counts()
                
                analysis['categorical_stats'] = {
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'top_10_values': value_counts.head(10).to_dict(),
                    'cardinality': int(col_data.nunique())
                }
                
                # Text analysis for string columns
                if col_data.dtype == 'object':
                    text_data = col_data.dropna().astype(str)
                    if len(text_data) > 0:
                        analysis['text_stats'] = {
                            'avg_length': float(text_data.str.len().mean()),
                            'min_length': int(text_data.str.len().min()),
                            'max_length': int(text_data.str.len().max()),
                            'total_characters': int(text_data.str.len().sum()),
                            'contains_numbers': int(text_data.str.contains(r'\d').sum()),
                            'contains_special_chars': int(text_data.str.contains(r'[^a-zA-Z0-9\s]').sum())
                        }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing column {column}: {str(e)}")
            return {'error': str(e)}
