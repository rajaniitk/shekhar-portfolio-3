# EDA Pro - Local Setup Guide

## Quick Start

This guide will help you run the EDA Pro application on your local machine.

## Prerequisites

1. **Python 3.11+** - Make sure you have Python 3.11 or later installed
2. **Git** - To clone the repository
3. **UV Package Manager** (recommended) or **pip**

## Installation Steps

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd eda-pro
```

### Step 2: Set Up Python Environment

#### Option A: Using UV (Recommended)
```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

#### Option B: Using Virtual Environment + pip
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install core dependencies
pip install Flask Flask-SQLAlchemy Werkzeug SQLAlchemy Gunicorn
pip install pandas numpy scipy scikit-learn joblib
pip install statsmodels pingouin
pip install plotly matplotlib seaborn
pip install openpyxl pyarrow psycopg2-binary email-validator

# Optional: Install XGBoost (may require additional system dependencies)
# pip install xgboost
```

### Step 3: Set Environment Variables
Create a `.env` file in the root directory:
```bash
# Database configuration
DATABASE_URL=sqlite:///eda_pro.db

# Session secret (generate a random string)
SESSION_SECRET=your-super-secret-key-here

# Optional: Set upload directory
UPLOAD_FOLDER=uploads
```

### Step 4: Initialize Database
```bash
# The database will be created automatically when you first run the app
# No additional setup needed for SQLite
```

### Step 5: Run the Application

#### Option A: Using UV
```bash
uv run gunicorn --bind 0.0.0.0:5000 --reload main:app
```

#### Option B: Using Python directly
```bash
# If using virtual environment, make sure it's activated
python -m gunicorn --bind 0.0.0.0:5000 --reload main:app
```

#### Option C: Development Server (Flask)
```bash
# Set environment variables
export FLASK_APP=main.py
export FLASK_ENV=development

# Run Flask development server
flask run --host=0.0.0.0 --port=5000
```

### Step 6: Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## Features Available

✅ **Data Upload**: Upload CSV, Excel, JSON, and Parquet files  
✅ **Exploratory Data Analysis**: Comprehensive statistical analysis  
✅ **60+ Visualizations**: Interactive charts with Plotly  
✅ **35+ Statistical Tests**: Hypothesis testing and correlation analysis  
✅ **Machine Learning**: 10+ algorithms for classification and regression  
✅ **Feature Engineering**: Automated data preprocessing  
✅ **Dark Theme**: Optimized color scheme for extended use  
✅ **Report Generation**: Export analysis results  

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
If port 5000 is busy, try a different port:
```bash
gunicorn --bind 0.0.0.0:8000 --reload main:app
```

#### 2. Missing Dependencies
Make sure all packages are installed:
```bash
# Check installed packages
pip list

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

#### 3. Database Issues
Delete the database file and restart:
```bash
rm eda_pro.db
# Restart the application
```

#### 4. Import Errors
Some optional packages (like LightGBM) may not work on all systems. The app will continue to work without them.

## File Upload Limits

- **Maximum file size**: 500MB
- **Supported formats**: CSV, Excel (.xlsx, .xls), JSON, Parquet
- **Upload location**: `uploads/` directory (created automatically)

## Development

### Project Structure
```
eda-pro/
├── app.py              # Flask application setup
├── main.py             # Application entry point
├── models.py           # Database models
├── routes/             # Route handlers
│   ├── main.py         # Dashboard routes
│   ├── upload.py       # File upload
│   ├── analysis.py     # EDA routes
│   ├── visualization.py # Chart generation
│   ├── statistics.py   # Statistical tests
│   └── ml_models.py    # Machine learning
├── services/           # Business logic
│   ├── data_processor.py
│   ├── eda_engine.py
│   ├── visualization_engine.py
│   ├── statistical_tests.py
│   ├── ml_engine.py
│   ├── feature_engineer.py
│   └── insights_generator.py
├── templates/          # HTML templates
├── static/            # CSS, JS, images
└── uploads/           # Uploaded files
```

### Adding New Features
1. Create new routes in `routes/`
2. Add business logic in `services/`
3. Create templates in `templates/`
4. Update navigation in `templates/base.html`

## Production Deployment

For production deployment, consider:
- Use PostgreSQL instead of SQLite
- Set up proper secret management
- Configure reverse proxy (nginx)
- Use process manager (systemd, supervisor)
- Set up monitoring and logging

## Support

If you encounter any issues:
1. Check the console logs for error messages
2. Verify all dependencies are installed
3. Ensure Python version compatibility
4. Check file permissions for upload directory

## Performance Tips

- For large datasets (>100MB), processing may take longer
- Close unused browser tabs to save memory
- Use Chrome/Firefox for best compatibility
- Clear browser cache if experiencing issues

---

**Happy analyzing! 🚀**