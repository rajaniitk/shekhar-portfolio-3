import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///eda_app.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the app with the extension
db.init_app(app)

# Register blueprints
from routes.main import main_bp
from routes.upload import upload_bp
from routes.analysis import analysis_bp
from routes.visualization import visualization_bp
from routes.statistics import statistics_bp
from routes.ml_models import ml_bp

app.register_blueprint(main_bp)
app.register_blueprint(upload_bp, url_prefix='/upload')
app.register_blueprint(analysis_bp, url_prefix='/analysis')
app.register_blueprint(visualization_bp, url_prefix='/visualization')
app.register_blueprint(statistics_bp, url_prefix='/statistics')
app.register_blueprint(ml_bp, url_prefix='/ml')

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
