# EDA Pro - Intelligent Data Analysis Platform

## Overview

EDA Pro is a comprehensive web-based platform for automated Exploratory Data Analysis (EDA), statistical testing, feature engineering, and machine learning model development. Built with Flask and powered by advanced Python data science libraries, it provides an intuitive interface for data scientists and analysts to perform sophisticated data analysis tasks.

## System Architecture

### Backend Architecture
- **Framework**: Flask web framework with SQLAlchemy ORM
- **Database**: SQLite for development (configurable for PostgreSQL in production)
- **Data Processing**: Pandas, NumPy, SciPy for data manipulation and analysis
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM for model training
- **Visualization**: Plotly, Seaborn, Matplotlib for interactive and static charts
- **Statistical Testing**: Comprehensive suite with 35+ statistical tests using SciPy and Pingouin

### Frontend Architecture
- **Template Engine**: Jinja2 with Bootstrap 5 dark theme
- **JavaScript**: Vanilla JS with Plotly.js for interactive visualizations
- **Styling**: Custom dark theme CSS with neon accents and gradients
- **Responsive Design**: Mobile-first approach with Bootstrap grid system

### Modular Service Architecture
The application follows a service-oriented architecture with specialized engines:
- **DataProcessor**: File loading and preprocessing (CSV, Excel, JSON, Parquet)
- **EDAEngine**: Comprehensive exploratory data analysis
- **VisualizationEngine**: 60+ chart types and interactive visualizations
- **StatisticalTestEngine**: 35+ statistical tests and hypothesis testing
- **MLEngine**: Machine learning model training and evaluation
- **FeatureEngineer**: Automated feature engineering and transformation
- **InsightsGenerator**: AI-powered insights and recommendations

## Key Components

### Data Management
- **Dataset Model**: Stores file metadata, shape, and column information
- **Analysis Model**: Tracks analysis results with JSON storage for flexibility
- **ModelTraining Model**: Manages ML model configurations and performance metrics
- **File Upload**: Secure file handling with validation and preview functionality

### Analysis Modules
1. **EDA Dashboard**: Basic statistics, correlation analysis, distribution metrics
2. **Column Analysis**: Detailed univariate and bivariate analysis
3. **Visualization Suite**: Interactive charts with Plotly integration
4. **Statistical Testing**: Comprehensive hypothesis testing framework
5. **Machine Learning**: Automated model training with hyperparameter optimization
6. **Feature Engineering**: Transformation suggestions and automated preprocessing

### Route Structure
- `/` - Main dashboard with statistics overview
- `/upload/` - File upload and dataset management
- `/analysis/` - EDA and dataset overview
- `/visualization/` - Chart generation and visualization
- `/statistics/` - Statistical testing interface
- `/ml_models/` - Machine learning model training

## Data Flow

### Upload Pipeline
1. File validation and secure storage
2. Automatic format detection and parsing
3. Metadata extraction and database storage
4. Preview generation and column analysis
5. Initial EDA report generation

### Analysis Pipeline
1. Dataset selection and loading
2. Statistical analysis execution
3. Visualization generation
4. Results storage and caching
5. Report generation and export

### ML Pipeline
1. Feature selection and preprocessing
2. Model training with cross-validation
3. Performance evaluation and metrics
4. Model serialization and storage
5. Prediction and inference capabilities

## External Dependencies

### Core Libraries
- **Flask**: Web framework and request handling
- **SQLAlchemy**: Database ORM and migrations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing and statistics
- **Scikit-learn**: Machine learning algorithms

### Visualization
- **Plotly**: Interactive web-based visualizations
- **Seaborn**: Statistical data visualization
- **Matplotlib**: Basic plotting capabilities

### Optional ML Libraries
- **XGBoost**: Gradient boosting (optional)
- **LightGBM**: Fast gradient boosting (optional)
- **Pingouin**: Advanced statistical functions

### Frontend
- **Bootstrap 5**: UI framework and components
- **Font Awesome**: Icons and visual elements
- **Chart.js**: Additional charting capabilities

## Deployment Strategy

### Development Setup
- Flask development server with debug mode
- SQLite database for local development
- File-based uploads with local storage
- Hot reloading for template and code changes

### Production Considerations
- **Database**: Migration to PostgreSQL recommended
- **File Storage**: Cloud storage integration (AWS S3, etc.)
- **Caching**: Redis for session and result caching
- **Security**: CSRF protection, secure file uploads
- **Scaling**: Containerization with Docker
- **Monitoring**: Logging and error tracking

### Configuration
- Environment-based configuration
- Secure secret key management
- Database URL configuration
- Upload directory management
- Maximum file size limits (500MB)

## Changelog

- July 07, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.