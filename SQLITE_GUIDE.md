# SQLite Database Guide for EDA Pro

## Overview

EDA Pro uses SQLite as the default database for development. SQLite is a self-contained, serverless database that's perfect for getting started quickly without requiring a separate database server.

## How SQLite is Configured

### 1. Database Configuration (app.py)
```python
# Default SQLite configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///eda_app.db")
```

The database file `eda_app.db` will be automatically created in the `instance/` directory when you first run the application. Flask uses the instance folder to store application-specific files like databases and configuration.

### 2. Environment Variables
You can override the default database by setting the `DATABASE_URL` environment variable:

```bash
# Use a different SQLite file
export DATABASE_URL="sqlite:///my_custom_database.db"

# Or use an absolute path
export DATABASE_URL="sqlite:////full/path/to/database.db"
```

## Database Schema

The application uses three main tables:

### 1. Dataset Table
Stores information about uploaded files:
```sql
CREATE TABLE dataset (
    id INTEGER PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    shape_rows INTEGER,
    shape_cols INTEGER,
    memory_usage FLOAT,
    column_info TEXT  -- JSON string
);
```

### 2. Analysis Table
Stores analysis results:
```sql
CREATE TABLE analysis (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,  -- 'eda', 'statistical_test', etc.
    results TEXT,  -- JSON string
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset (id)
);
```

### 3. ModelTraining Table
Stores machine learning model information:
```sql
CREATE TABLE model_training (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    target_column VARCHAR(255) NOT NULL,
    features TEXT,  -- JSON array
    hyperparameters TEXT,  -- JSON string
    performance_metrics TEXT,  -- JSON string
    model_path VARCHAR(500),
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset (id)
);
```

## Working with the Database

### 1. Database Initialization
The database is automatically created when you first start the application:

```python
with app.app_context():
    import models  # This imports the table definitions
    db.create_all()  # Creates all tables
```

### 2. Viewing the Database
You can use various tools to view your SQLite database:

#### Option A: SQLite Command Line
```bash
# Open the database (note the instance/ directory)
sqlite3 instance/eda_app.db

# View all tables
.tables

# View table structure
.schema dataset

# Query data
SELECT * FROM dataset;

# Exit
.quit
```

#### Option B: DB Browser for SQLite (GUI)
Download from: https://sqlitebrowser.org/
- Open the `eda_app.db` file
- Browse tables and data visually

#### Option C: VS Code Extension
Install "SQLite Viewer" extension in VS Code to browse the database directly in your editor.

### 3. Basic Database Operations

#### Adding a Dataset
```python
from models import Dataset
from app import db

# Create new dataset record
dataset = Dataset(
    filename="sales_data.csv",
    file_path="/uploads/sales_data.csv",
    file_type="csv",
    shape_rows=1000,
    shape_cols=15
)

# Save to database
db.session.add(dataset)
db.session.commit()
```

#### Querying Data
```python
# Get all datasets
datasets = Dataset.query.all()

# Get dataset by ID
dataset = Dataset.query.get(1)

# Filter datasets
csv_datasets = Dataset.query.filter_by(file_type='csv').all()

# Get recent datasets
recent = Dataset.query.order_by(Dataset.upload_date.desc()).limit(5).all()
```

#### Updating Records
```python
# Update a dataset
dataset = Dataset.query.get(1)
dataset.memory_usage = 50.5
db.session.commit()
```

#### Deleting Records
```python
# Delete a dataset
dataset = Dataset.query.get(1)
db.session.delete(dataset)
db.session.commit()
```

## Database Maintenance

### 1. Backup Your Database
```bash
# Create a backup
cp eda_app.db eda_app_backup_$(date +%Y%m%d).db

# Or use SQLite backup command
sqlite3 eda_app.db ".backup eda_app_backup.db"
```

### 2. Reset the Database
```bash
# Delete the database file (will be recreated on next startup)
rm eda_app.db

# Restart the application
python main.py
```

### 3. Database Migrations
If you modify the models, you may need to recreate the database:

```python
# In Python console or script
from app import app, db

with app.app_context():
    db.drop_all()    # Delete all tables
    db.create_all()  # Recreate with new structure
```

## Production Considerations

### 1. Switching to PostgreSQL
For production, consider switching to PostgreSQL:

```bash
# Set environment variable
export DATABASE_URL="postgresql://username:password@localhost:5432/eda_pro"

# Install PostgreSQL driver (already included)
pip install psycopg2-binary
```

### 2. Database Connections
SQLite is single-writer, so it's perfect for development but may have limitations with multiple concurrent users.

### 3. File Location
In production, ensure the SQLite file is in a persistent location and properly backed up.

## Common Database Tasks

### 1. View Upload Statistics
```sql
SELECT 
    file_type,
    COUNT(*) as count,
    AVG(shape_rows) as avg_rows,
    AVG(shape_cols) as avg_cols
FROM dataset 
GROUP BY file_type;
```

### 2. Find Large Datasets
```sql
SELECT filename, shape_rows, shape_cols, memory_usage
FROM dataset 
WHERE shape_rows > 10000 
ORDER BY memory_usage DESC;
```

### 3. Analysis History
```sql
SELECT 
    d.filename,
    a.analysis_type,
    a.created_date
FROM dataset d
JOIN analysis a ON d.id = a.dataset_id
ORDER BY a.created_date DESC;
```

### 4. Model Performance
```sql
SELECT 
    d.filename,
    m.model_type,
    m.performance_metrics,
    m.created_date
FROM dataset d
JOIN model_training m ON d.id = m.dataset_id
ORDER BY m.created_date DESC;
```

## Troubleshooting

### 1. Database Locked Error
```bash
# This happens when multiple processes access SQLite
# Solution: Close all connections and restart the application
```

### 2. Database File Missing
```bash
# If the database file is accidentally deleted, it will be recreated
# but all data will be lost. Always keep backups!
```

### 3. Permission Issues
```bash
# Ensure the application has write permissions to the directory
chmod 755 .
chmod 666 eda_app.db  # If the file exists
```

## Quick Start Commands

```bash
# View database content
sqlite3 eda_app.db "SELECT COUNT(*) FROM dataset;"

# Export data to CSV
sqlite3 eda_app.db -header -csv "SELECT * FROM dataset;" > datasets_export.csv

# Check database size
ls -lh eda_app.db

# Vacuum database (optimize)
sqlite3 eda_app.db "VACUUM;"
```

## Integration with the Application

The database is seamlessly integrated with the Flask application:

1. **File uploads** automatically create Dataset records
2. **Analysis results** are stored in the Analysis table
3. **ML models** are tracked in the ModelTraining table
4. **All operations** use SQLAlchemy ORM for database interactions

The application handles all database operations automatically, so you typically don't need to interact with the database directly unless you're doing advanced analysis or troubleshooting.