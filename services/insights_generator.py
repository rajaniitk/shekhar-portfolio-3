import pandas as pd
import numpy as np
from scipy import stats
import logging

class InsightsGenerator:
    """AI-powered insights and recommendations generator"""
    
    def __init__(self):
        self.insights = []
        self.recommendations = []
    
    def generate_comprehensive_insights(self, df):
        """Generate comprehensive insights for the entire dataset"""
        try:
            insights = {
                'dataset_overview': self._analyze_dataset_overview(df),
                'data_quality_insights': self._analyze_data_quality(df),
                'distribution_insights': self._analyze_distributions(df),
                'correlation_insights': self._analyze_correlations(df),
                'feature_insights': self._analyze_features(df),
                'recommendations': self._generate_recommendations(df)
            }
            
            return insights
            
        except Exception as e:
            logging.error(f"Error generating insights: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_dataset_overview(self, df):
        """Analyze overall dataset characteristics"""
        insights = []
        
        # Dataset size insights
        total_cells = df.shape[0] * df.shape[1]
        if df.shape[0] > 100000:
            insights.append({
                'type': 'size',
                'severity': 'info',
                'message': f"Large dataset with {df.shape[0]:,} rows. Consider sampling for faster analysis.",
                'details': f"Total data points: {total_cells:,}"
            })
        elif df.shape[0] < 100:
            insights.append({
                'type': 'size',
                'severity': 'warning',
                'message': f"Small dataset with only {df.shape[0]} rows. Statistical tests may have limited power.",
                'details': "Consider collecting more data for robust analysis."
            })
        
        # Column distribution
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_cols == 0:
            insights.append({
                'type': 'data_types',
                'severity': 'warning',
                'message': "No numeric columns detected. Limited analysis options available.",
                'details': "Consider converting appropriate columns to numeric type."
            })
        elif categorical_cols == 0:
            insights.append({
                'type': 'data_types',
                'severity': 'info',
                'message': "Dataset contains only numeric variables. Good for mathematical analysis.",
                'details': f"{numeric_cols} numeric columns available"
            })
        else:
            insights.append({
                'type': 'data_types',
                'severity': 'info',
                'message': f"Balanced dataset with {numeric_cols} numeric and {categorical_cols} categorical variables.",
                'details': "Good mix for comprehensive analysis"
            })
        
        return insights
    
    def _analyze_data_quality(self, df):
        """Analyze data quality issues"""
        insights = []
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
        
        if missing_percentage > 50:
            insights.append({
                'type': 'missing_data',
                'severity': 'critical',
                'message': f"Severe data quality issue: {missing_percentage:.1f}% of data is missing.",
                'details': "Consider data collection review or imputation strategies."
            })
        elif missing_percentage > 20:
            insights.append({
                'type': 'missing_data',
                'severity': 'warning',
                'message': f"Significant missing data: {missing_percentage:.1f}% of values are missing.",
                'details': f"{len(missing_data[missing_data > 0])} columns affected"
            })
        elif missing_percentage > 5:
            insights.append({
                'type': 'missing_data',
                'severity': 'info',
                'message': f"Moderate missing data: {missing_percentage:.1f}% of values are missing.",
                'details': "Manageable with standard imputation techniques"
            })
        
        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        if duplicate_percentage > 10:
            insights.append({
                'type': 'duplicates',
                'severity': 'warning',
                'message': f"High number of duplicate rows: {duplicate_count} ({duplicate_percentage:.1f}%)",
                'details': "Consider deduplication before analysis"
            })
        elif duplicate_count > 0:
            insights.append({
                'type': 'duplicates',
                'severity': 'info',
                'message': f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)",
                'details': "May want to remove duplicates"
            })
        
        # Constant/quasi-constant features
        constant_features = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_features.append(col)
            elif df[col].nunique() / len(df) < 0.01:  # Less than 1% unique values
                constant_features.append(col)
        
        if constant_features:
            insights.append({
                'type': 'constant_features',
                'severity': 'warning',
                'message': f"Found {len(constant_features)} constant or quasi-constant features",
                'details': f"Features: {', '.join(constant_features[:5])}" + ("..." if len(constant_features) > 5 else "")
            })
        
        return insights
    
    def _analyze_distributions(self, df):
        """Analyze distribution characteristics"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Skewness analysis
        highly_skewed = []
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                highly_skewed.append((col, skewness))
        
        if highly_skewed:
            insights.append({
                'type': 'skewness',
                'severity': 'info',
                'message': f"Found {len(highly_skewed)} highly skewed features",
                'details': f"Most skewed: {highly_skewed[0][0]} (skewness: {highly_skewed[0][1]:.2f})"
            })
        
        # Outlier analysis
        outlier_features = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
            outlier_pct = (outliers / len(df)) * 100
            
            if outlier_pct > 10:
                outlier_features.append((col, outlier_pct))
        
        if outlier_features:
            insights.append({
                'type': 'outliers',
                'severity': 'warning',
                'message': f"High outlier percentage in {len(outlier_features)} features",
                'details': f"Worst: {outlier_features[0][0]} ({outlier_features[0][1]:.1f}% outliers)"
            })
        
        # Normality insights
        non_normal = []
        for col in numeric_cols:
            if len(df[col].dropna()) >= 3:
                try:
                    _, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
                    if p_value < 0.05:
                        non_normal.append(col)
                except:
                    pass
        
        if len(non_normal) > len(numeric_cols) * 0.8:
            insights.append({
                'type': 'normality',
                'severity': 'info',
                'message': f"Most features ({len(non_normal)}/{len(numeric_cols)}) are not normally distributed",
                'details': "Consider non-parametric tests or transformations"
            })
        
        return insights
    
    def _analyze_correlations(self, df):
        """Analyze correlation patterns"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return insights
        
        corr_matrix = df[numeric_cols].corr()
        
        # High correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            insights.append({
                'type': 'multicollinearity',
                'severity': 'warning',
                'message': f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.9)",
                'details': f"Strongest correlation: {high_corr_pairs[0][0]} - {high_corr_pairs[0][1]} ({high_corr_pairs[0][2]:.3f})"
            })
        
        # Correlation with target (if identifiable)
        potential_targets = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['target', 'label', 'class', 'outcome', 'y']):
                potential_targets.append(col)
        
        if potential_targets and potential_targets[0] in numeric_cols:
            target_col = potential_targets[0]
            target_correlations = corr_matrix[target_col].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
            
            if len(target_correlations) > 0:
                insights.append({
                    'type': 'target_correlation',
                    'severity': 'info',
                    'message': f"Feature most correlated with {target_col}: {target_correlations.index[0]}",
                    'details': f"Correlation: {target_correlations.iloc[0]:.3f}"
                })
        
        return insights
    
    def _analyze_features(self, df):
        """Analyze individual feature characteristics"""
        insights = []
        
        # High cardinality categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)
            
            if unique_count > 50 and unique_ratio > 0.9:
                high_cardinality.append((col, unique_count))
        
        if high_cardinality:
            insights.append({
                'type': 'high_cardinality',
                'severity': 'warning',
                'message': f"Found {len(high_cardinality)} high cardinality categorical features",
                'details': f"Highest: {high_cardinality[0][0]} ({high_cardinality[0][1]} unique values)"
            })
        
        # Zero inflation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_inflated = []
        
        for col in numeric_cols:
            zero_pct = (df[col] == 0).sum() / len(df) * 100
            if zero_pct > 50:
                zero_inflated.append((col, zero_pct))
        
        if zero_inflated:
            insights.append({
                'type': 'zero_inflation',
                'severity': 'info',
                'message': f"Found {len(zero_inflated)} zero-inflated features",
                'details': f"Most zero-inflated: {zero_inflated[0][0]} ({zero_inflated[0][1]:.1f}% zeros)"
            })
        
        # Imbalanced categorical features
        imbalanced_categorical = []
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                majority_pct = (value_counts.iloc[0] / len(df)) * 100
                if majority_pct > 95:
                    imbalanced_categorical.append((col, majority_pct))
        
        if imbalanced_categorical:
            insights.append({
                'type': 'class_imbalance',
                'severity': 'warning',
                'message': f"Found {len(imbalanced_categorical)} severely imbalanced categorical features",
                'details': f"Most imbalanced: {imbalanced_categorical[0][0]} ({imbalanced_categorical[0][1]:.1f}% majority class)"
            })
        
        return insights
    
    def _generate_recommendations(self, df):
        """Generate actionable recommendations"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Missing data recommendations
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            recommendations.append({
                'category': 'data_preprocessing',
                'priority': 'high',
                'action': 'Handle missing values',
                'description': f"Address missing data in {len(missing_data[missing_data > 0])} columns using appropriate imputation strategies.",
                'technical_details': "Consider KNN imputation for numeric data and mode imputation for categorical data."
            })
        
        # Feature engineering recommendations
        if len(numeric_cols) >= 2:
            recommendations.append({
                'category': 'feature_engineering',
                'priority': 'medium',
                'action': 'Create interaction features',
                'description': "Generate polynomial and interaction features from existing numeric variables.",
                'technical_details': "Use PolynomialFeatures or manual feature creation for domain-specific interactions."
            })
        
        # Scaling recommendations
        if len(numeric_cols) > 0:
            # Check if scaling is needed
            ranges = []
            for col in numeric_cols:
                col_range = df[col].max() - df[col].min()
                ranges.append(col_range)
            
            if max(ranges) / min(ranges) > 100:  # Large scale differences
                recommendations.append({
                    'category': 'preprocessing',
                    'priority': 'high',
                    'action': 'Scale numeric features',
                    'description': "Features have very different scales. Apply StandardScaler or RobustScaler.",
                    'technical_details': "Use RobustScaler if outliers are present, otherwise StandardScaler."
                })
        
        # Dimensionality reduction recommendations
        if len(df.columns) > 50:
            recommendations.append({
                'category': 'dimensionality_reduction',
                'priority': 'medium',
                'action': 'Apply dimensionality reduction',
                'description': f"High-dimensional dataset ({len(df.columns)} features). Consider PCA or feature selection.",
                'technical_details': "Start with PCA to retain 95% variance or use SelectKBest for feature selection."
            })
        
        # Categorical encoding recommendations
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > 10:
                recommendations.append({
                    'category': 'encoding',
                    'priority': 'medium',
                    'action': f'Encode high-cardinality feature: {col}',
                    'description': f"Feature {col} has {unique_count} unique values. Consider target encoding or frequency encoding.",
                    'technical_details': "Avoid one-hot encoding for high cardinality. Use target encoding with cross-validation."
                })
        
        # Outlier handling recommendations
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
            outlier_pct = (outliers / len(df)) * 100
            
            if outlier_pct > 5:
                outlier_cols.append(col)
        
        if outlier_cols:
            recommendations.append({
                'category': 'outlier_handling',
                'priority': 'medium',
                'action': 'Handle outliers',
                'description': f"Significant outliers detected in {len(outlier_cols)} features.",
                'technical_details': "Consider robust scaling, winsorization, or outlier removal based on domain knowledge."
            })
        
        # Model selection recommendations
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            recommendations.append({
                'category': 'model_selection',
                'priority': 'low',
                'action': 'Consider ensemble methods',
                'description': "Mixed data types suggest ensemble methods (Random Forest, Gradient Boosting) may perform well.",
                'technical_details': "Tree-based models handle mixed data types naturally without extensive preprocessing."
            })
        
        # Statistical testing recommendations
        if len(numeric_cols) >= 2:
            recommendations.append({
                'category': 'analysis',
                'priority': 'low',
                'action': 'Perform correlation analysis',
                'description': "Multiple numeric features available for correlation and causality analysis.",
                'technical_details': "Use Pearson for linear relationships, Spearman for monotonic relationships."
            })
        
        return recommendations
    
    def generate_column_insights(self, df, column):
        """Generate insights for a specific column"""
        try:
            if column not in df.columns:
                return {'error': f'Column {column} not found'}
            
            insights = []
            col_data = df[column]
            
            # Basic insights
            non_null_count = col_data.count()
            null_count = col_data.isnull().sum()
            null_percentage = (null_count / len(col_data)) * 100
            
            if null_percentage > 50:
                insights.append({
                    'type': 'data_quality',
                    'severity': 'critical',
                    'message': f"Column has {null_percentage:.1f}% missing values",
                    'recommendation': "Consider dropping this column or advanced imputation techniques"
                })
            elif null_percentage > 10:
                insights.append({
                    'type': 'data_quality',
                    'severity': 'warning',
                    'message': f"Column has {null_percentage:.1f}% missing values",
                    'recommendation': "Apply appropriate imputation strategy"
                })
            
            # Type-specific insights
            if col_data.dtype in ['int64', 'float64']:
                # Numeric column insights
                insights.extend(self._analyze_numeric_column(col_data, column))
            elif col_data.dtype in ['object', 'category']:
                # Categorical column insights
                insights.extend(self._analyze_categorical_column(col_data, column))
            
            return {'insights': insights}
            
        except Exception as e:
            logging.error(f"Error generating column insights: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_numeric_column(self, col_data, column_name):
        """Analyze numeric column"""
        insights = []
        
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return insights
        
        # Distribution insights
        skewness = clean_data.skew()
        if abs(skewness) > 2:
            direction = "right" if skewness > 0 else "left"
            insights.append({
                'type': 'distribution',
                'severity': 'info',
                'message': f"Highly {direction}-skewed distribution (skewness: {skewness:.2f})",
                'recommendation': "Consider log transformation" if skewness > 0 else "Consider square transformation"
            })
        
        # Outlier insights
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = len(clean_data[(clean_data < Q1 - 1.5 * IQR) | (clean_data > Q3 + 1.5 * IQR)])
        outlier_pct = (outliers / len(clean_data)) * 100
        
        if outlier_pct > 10:
            insights.append({
                'type': 'outliers',
                'severity': 'warning',
                'message': f"High outlier percentage: {outlier_pct:.1f}%",
                'recommendation': "Consider robust scaling or outlier removal"
            })
        
        # Variability insights
        cv = clean_data.std() / clean_data.mean() if clean_data.mean() != 0 else float('inf')
        if cv > 2:
            insights.append({
                'type': 'variability',
                'severity': 'info',
                'message': f"High coefficient of variation: {cv:.2f}",
                'recommendation': "Feature shows high variability - may be important for modeling"
            })
        elif cv < 0.1:
            insights.append({
                'type': 'variability',
                'severity': 'warning',
                'message': f"Low coefficient of variation: {cv:.2f}",
                'recommendation': "Feature shows low variability - may not be useful for modeling"
            })
        
        # Zero inflation
        zero_pct = (clean_data == 0).sum() / len(clean_data) * 100
        if zero_pct > 50:
            insights.append({
                'type': 'zero_inflation',
                'severity': 'info',
                'message': f"Zero-inflated feature: {zero_pct:.1f}% zeros",
                'recommendation': "Consider zero-inflation models or transformation"
            })
        
        return insights
    
    def _analyze_categorical_column(self, col_data, column_name):
        """Analyze categorical column"""
        insights = []
        
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return insights
        
        unique_count = clean_data.nunique()
        
        # Cardinality insights
        if unique_count == 1:
            insights.append({
                'type': 'cardinality',
                'severity': 'critical',
                'message': "Constant feature - only one unique value",
                'recommendation': "Remove this feature as it provides no information"
            })
        elif unique_count == len(clean_data):
            insights.append({
                'type': 'cardinality',
                'severity': 'warning',
                'message': "All values are unique - possible identifier column",
                'recommendation': "Consider removing if this is an ID column"
            })
        elif unique_count > 50:
            insights.append({
                'type': 'cardinality',
                'severity': 'warning',
                'message': f"High cardinality: {unique_count} unique values",
                'recommendation': "Consider target encoding or frequency encoding instead of one-hot encoding"
            })
        
        # Class imbalance insights
        value_counts = clean_data.value_counts()
        majority_pct = (value_counts.iloc[0] / len(clean_data)) * 100
        
        if majority_pct > 95:
            insights.append({
                'type': 'imbalance',
                'severity': 'warning',
                'message': f"Severely imbalanced: {majority_pct:.1f}% in majority class",
                'recommendation': "Consider feature transformation or removal"
            })
        elif majority_pct > 80:
            insights.append({
                'type': 'imbalance',
                'severity': 'info',
                'message': f"Imbalanced distribution: {majority_pct:.1f}% in majority class",
                'recommendation': "Monitor model performance for this feature"
            })
        
        # Rare categories
        rare_categories = (value_counts / len(clean_data) < 0.01).sum()
        if rare_categories > 0:
            insights.append({
                'type': 'rare_categories',
                'severity': 'info',
                'message': f"{rare_categories} categories with <1% frequency",
                'recommendation': "Consider grouping rare categories into 'Other'"
            })
        
        return insights
