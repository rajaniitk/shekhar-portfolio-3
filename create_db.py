#!/usr/bin/env python3
"""
Script to manually create the SQLite database for EDA Pro
"""

from app import app, db

def create_database():
    """Create all database tables"""
    with app.app_context():
        # Import all models to ensure they're registered
        import models
        
        # Create all tables
        db.create_all()
        
        print("Database created successfully!")
        print(f"Database file: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        # Print table information
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"\nCreated tables: {tables}")
        
        for table in tables:
            columns = inspector.get_columns(table)
            print(f"\nTable '{table}' columns:")
            for col in columns:
                print(f"  - {col['name']} ({col['type']})")

if __name__ == "__main__":
    create_database()