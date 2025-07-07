// Machine Learning Dashboard JavaScript Functions

class MLManager {
    constructor() {
        this.currentDatasetId = null;
        this.models = {};
        this.currentModel = null;
    }

    setDatasetId(datasetId) {
        this.currentDatasetId = datasetId;
    }

    // Model Training
    async trainModel(modelConfig) {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Training model... This may take a few minutes.');
        
        try {
            const response = await fetch(`/ml/api/train/${this.currentDatasetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(modelConfig)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayTrainingResults(data);
                showAlert('success', 'Model trained successfully!');
            } else {
                showAlert('error', data.error || 'Error training model');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Model Prediction
    async makePrediction(predictionData) {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Making predictions...');
        
        try {
            const response = await fetch(`/ml/api/predict/${this.currentDatasetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictionData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayPredictionResults(data);
            } else {
                showAlert('error', data.error || 'Error making predictions');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Model Evaluation
    async evaluateModel(modelId) {
        if (!this.currentDatasetId || !modelId) {
            showAlert('warning', 'Please select dataset and model.');
            return;
        }

        showLoader('Evaluating model...');
        
        try {
            const response = await fetch(`/ml/api/evaluate/${this.currentDatasetId}/${modelId}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayEvaluationResults(data);
            } else {
                showAlert('error', data.error || 'Error evaluating model');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // AutoML
    async runAutoML(autoMLConfig) {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Running AutoML... This may take several minutes.');
        
        try {
            const response = await fetch(`/ml/api/automl/${this.currentDatasetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(autoMLConfig)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayAutoMLResults(data);
                showAlert('success', 'AutoML completed successfully!');
            } else {
                showAlert('error', data.error || 'Error running AutoML');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Feature Engineering
    async generateFeatures(featureConfig) {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Generating features...');
        
        try {
            const response = await fetch(`/ml/api/features/${this.currentDatasetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(featureConfig)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayFeatureResults(data);
                showAlert('success', 'Features generated successfully!');
            } else {
                showAlert('error', data.error || 'Error generating features');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Display Methods
    displayTrainingResults(data) {
        const container = document.getElementById('training-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-cogs me-2"></i>Training Results</h5>';
        
        // Model Performance
        if (data.performance) {
            html += `
                <div class="card bg-dark border-success mb-3">
                    <div class="card-header bg-success">
                        <h6 class="mb-0">Model Performance</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
            `;
            
            Object.keys(data.performance).forEach(metric => {
                const value = data.performance[metric];
                html += `
                    <div class="col-md-3 mb-2">
                        <div class="text-center">
                            <h4 class="text-primary">${typeof value === 'number' ? value.toFixed(4) : value}</h4>
                            <small class="text-muted">${metric.replace('_', ' ').toUpperCase()}</small>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div></div>';
        }

        // Feature Importance
        if (data.feature_importance) {
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">Feature Importance</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-sm">
                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Importance</th>
                                        <th>Visualization</th>
                                    </tr>
                                </thead>
                                <tbody>
            `;
            
            data.feature_importance.forEach(feature => {
                const barWidth = (feature.importance * 100).toFixed(1);
                html += `
                    <tr>
                        <td>${feature.feature}</td>
                        <td>${feature.importance.toFixed(4)}</td>
                        <td>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar bg-primary" role="progressbar" 
                                     style="width: ${barWidth}%" aria-valuenow="${barWidth}" 
                                     aria-valuemin="0" aria-valuemax="100">
                                    ${barWidth}%
                                </div>
                            </div>
                        </td>
                    </tr>
                `;
            });
            
            html += '</tbody></table></div></div></div>';
        }

        // Model Details
        if (data.model_details) {
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">Model Details</h6>
                    </div>
                    <div class="card-body">
                        <dl class="row">
                            <dt class="col-sm-3">Model Type:</dt>
                            <dd class="col-sm-9">${data.model_details.type}</dd>
                            <dt class="col-sm-3">Training Time:</dt>
                            <dd class="col-sm-9">${data.model_details.training_time || 'N/A'}</dd>
                            <dt class="col-sm-3">Features:</dt>
                            <dd class="col-sm-9">${data.model_details.features || 'N/A'}</dd>
                            <dt class="col-sm-3">Target:</dt>
                            <dd class="col-sm-9">${data.model_details.target || 'N/A'}</dd>
                        </dl>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    displayPredictionResults(data) {
        const container = document.getElementById('prediction-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-brain me-2"></i>Prediction Results</h5>';
        
        if (data.predictions) {
            html += `
                <div class="card bg-dark border-primary mb-3">
                    <div class="card-header bg-primary">
                        <h6 class="mb-0">Predictions</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        <th>Row</th>
                                        <th>Prediction</th>
                                        ${data.predictions[0].confidence ? '<th>Confidence</th>' : ''}
                                    </tr>
                                </thead>
                                <tbody>
            `;
            
            data.predictions.forEach((pred, index) => {
                html += `
                    <tr>
                        <td>${index + 1}</td>
                        <td><span class="badge bg-success">${pred.prediction}</span></td>
                        ${pred.confidence ? `<td>${(pred.confidence * 100).toFixed(1)}%</td>` : ''}
                    </tr>
                `;
            });
            
            html += '</tbody></table></div></div></div>';
        }

        container.innerHTML = html;
    }

    displayEvaluationResults(data) {
        const container = document.getElementById('evaluation-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-chart-line me-2"></i>Model Evaluation</h5>';
        
        // Metrics
        if (data.metrics) {
            html += `
                <div class="card bg-dark border-info mb-3">
                    <div class="card-header bg-info">
                        <h6 class="mb-0">Performance Metrics</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
            `;
            
            Object.keys(data.metrics).forEach(metric => {
                const value = data.metrics[metric];
                html += `
                    <div class="col-md-4 mb-3">
                        <div class="text-center p-3 border rounded">
                            <h3 class="text-info">${typeof value === 'number' ? value.toFixed(4) : value}</h3>
                            <small class="text-muted">${metric.replace('_', ' ').toUpperCase()}</small>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div></div>';
        }

        // Confusion Matrix
        if (data.confusion_matrix) {
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">Confusion Matrix</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-bordered text-center">
                                <thead>
                                    <tr>
                                        <th></th>
                                        ${data.confusion_matrix.labels.map(label => `<th>Predicted ${label}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
            `;
            
            data.confusion_matrix.matrix.forEach((row, i) => {
                html += `<tr><th>Actual ${data.confusion_matrix.labels[i]}</th>`;
                row.forEach(cell => {
                    html += `<td>${cell}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</tbody></table></div></div></div>';
        }

        container.innerHTML = html;
    }

    displayAutoMLResults(data) {
        const container = document.getElementById('automl-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-magic me-2"></i>AutoML Results</h5>';
        
        // Best Model
        if (data.best_model) {
            html += `
                <div class="card bg-dark border-warning mb-3">
                    <div class="card-header bg-warning text-dark">
                        <h6 class="mb-0">Best Model</h6>
                    </div>
                    <div class="card-body">
                        <h5 class="text-warning">${data.best_model.name}</h5>
                        <p class="text-muted">${data.best_model.description || ''}</p>
                        <div class="row">
                            <div class="col-md-6">
                                <strong>Score:</strong> ${data.best_model.score.toFixed(4)}
                            </div>
                            <div class="col-md-6">
                                <strong>CV Score:</strong> ${data.best_model.cv_score.toFixed(4)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Model Comparison
        if (data.model_comparison) {
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">Model Comparison</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        <th>Model</th>
                                        <th>Score</th>
                                        <th>CV Score</th>
                                        <th>Training Time</th>
                                    </tr>
                                </thead>
                                <tbody>
            `;
            
            data.model_comparison.forEach(model => {
                const isSelected = model.name === data.best_model?.name;
                html += `
                    <tr ${isSelected ? 'class="table-warning"' : ''}>
                        <td>
                            ${model.name}
                            ${isSelected ? '<span class="badge bg-warning text-dark ms-2">Best</span>' : ''}
                        </td>
                        <td>${model.score.toFixed(4)}</td>
                        <td>${model.cv_score.toFixed(4)}</td>
                        <td>${model.training_time || 'N/A'}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table></div></div></div>';
        }

        container.innerHTML = html;
    }

    displayFeatureResults(data) {
        const container = document.getElementById('feature-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-wrench me-2"></i>Feature Engineering Results</h5>';
        
        if (data.new_features) {
            html += `
                <div class="card bg-dark border-success mb-3">
                    <div class="card-header bg-success">
                        <h6 class="mb-0">Generated Features</h6>
                    </div>
                    <div class="card-body">
                        <p><strong>Total new features:</strong> ${data.new_features.length}</p>
                        <div class="row">
            `;
            
            data.new_features.forEach(feature => {
                html += `
                    <div class="col-md-6 mb-2">
                        <div class="border rounded p-2">
                            <strong>${feature.name}</strong>
                            <br>
                            <small class="text-muted">${feature.description || 'Auto-generated feature'}</small>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div></div>';
        }

        container.innerHTML = html;
    }

    // UI Helpers
    updateModelOptions() {
        const problemType = document.getElementById('problem-type')?.value;
        const modelSelect = document.getElementById('model-type');
        
        if (!modelSelect) return;

        // Clear existing options
        modelSelect.innerHTML = '<option value="">Select Model</option>';

        let models = [];
        if (problemType === 'classification') {
            models = [
                { value: 'logistic_regression', text: 'Logistic Regression' },
                { value: 'random_forest', text: 'Random Forest' },
                { value: 'gradient_boosting', text: 'Gradient Boosting' },
                { value: 'svm', text: 'Support Vector Machine' },
                { value: 'neural_network', text: 'Neural Network' }
            ];
        } else if (problemType === 'regression') {
            models = [
                { value: 'linear_regression', text: 'Linear Regression' },
                { value: 'random_forest', text: 'Random Forest' },
                { value: 'gradient_boosting', text: 'Gradient Boosting' },
                { value: 'svr', text: 'Support Vector Regression' },
                { value: 'neural_network', text: 'Neural Network' }
            ];
        }

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            modelSelect.appendChild(option);
        });
    }
}

// Initialize ML manager
const mlManager = new MLManager();

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Extract dataset ID from URL if present
    const pathParts = window.location.pathname.split('/');
    const datasetIdIndex = pathParts.indexOf('ml') + 1;
    if (datasetIdIndex > 0 && pathParts[datasetIdIndex]) {
        mlManager.setDatasetId(parseInt(pathParts[datasetIdIndex]));
    }

    // Problem type change handler
    const problemTypeSelect = document.getElementById('problem-type');
    if (problemTypeSelect) {
        problemTypeSelect.addEventListener('change', () => mlManager.updateModelOptions());
    }
});

// Global functions for use in HTML
function trainModel() {
    const problemType = document.getElementById('problem-type')?.value;
    const modelType = document.getElementById('model-type')?.value;
    const targetColumn = document.getElementById('target-column')?.value;
    const features = Array.from(document.querySelectorAll('input[name="features"]:checked'))
                         .map(input => input.value);

    if (!problemType || !modelType || !targetColumn || !features.length) {
        showAlert('warning', 'Please fill in all required fields.');
        return;
    }

    const modelConfig = {
        problem_type: problemType,
        model_type: modelType,
        target_column: targetColumn,
        features: features,
        test_size: parseFloat(document.getElementById('test-size')?.value || 0.2),
        cross_validation: document.getElementById('cross-validation')?.checked || false,
        hyperparameter_tuning: document.getElementById('hyperparameter-tuning')?.checked || false
    };

    mlManager.trainModel(modelConfig);
}

function runAutoML() {
    const problemType = document.getElementById('automl-problem-type')?.value;
    const targetColumn = document.getElementById('automl-target-column')?.value;
    const timeLimit = parseInt(document.getElementById('time-limit')?.value || 300);

    if (!problemType || !targetColumn) {
        showAlert('warning', 'Please select problem type and target column.');
        return;
    }

    const autoMLConfig = {
        problem_type: problemType,
        target_column: targetColumn,
        time_limit: timeLimit,
        metric: document.getElementById('optimization-metric')?.value || 'auto'
    };

    mlManager.runAutoML(autoMLConfig);
}

function generateFeatures() {
    const featureTypes = Array.from(document.querySelectorAll('input[name="feature-types"]:checked'))
                            .map(input => input.value);
    const columns = Array.from(document.querySelectorAll('input[name="feature-columns"]:checked'))
                        .map(input => input.value);

    if (!featureTypes.length || !columns.length) {
        showAlert('warning', 'Please select feature types and columns.');
        return;
    }

    const featureConfig = {
        feature_types: featureTypes,
        columns: columns,
        polynomial_degree: parseInt(document.getElementById('polynomial-degree')?.value || 2),
        interaction_only: document.getElementById('interaction-only')?.checked || false
    };

    mlManager.generateFeatures(featureConfig);
}

function makePrediction() {
    const modelId = document.getElementById('prediction-model')?.value;
    const inputData = {};

    // Collect input values
    const inputs = document.querySelectorAll('.prediction-input');
    inputs.forEach(input => {
        inputData[input.name] = input.value;
    });

    if (!modelId || Object.keys(inputData).length === 0) {
        showAlert('warning', 'Please select model and provide input values.');
        return;
    }

    const predictionData = {
        model_id: modelId,
        input_data: inputData
    };

    mlManager.makePrediction(predictionData);
}

function evaluateModel() {
    const modelId = document.getElementById('evaluation-model')?.value;
    
    if (!modelId) {
        showAlert('warning', 'Please select a model to evaluate.');
        return;
    }

    mlManager.evaluateModel(modelId);
}