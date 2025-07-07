// Statistical Analysis JavaScript Functions

class StatisticsManager {
    constructor() {
        this.currentDatasetId = null;
        this.selectedColumns = [];
    }

    setDatasetId(datasetId) {
        this.currentDatasetId = datasetId;
    }

    // Normality Tests
    async runNormalityTests(columns) {
        if (!this.currentDatasetId || !columns.length) {
            showAlert('warning', 'Please select dataset and columns for normality tests.');
            return;
        }

        showLoader('Running normality tests...');
        
        try {
            const params = new URLSearchParams();
            columns.forEach(col => params.append('columns', col));
            
            const response = await fetch(`/statistics/api/normality/${this.currentDatasetId}?${params}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayNormalityResults(data);
            } else {
                showAlert('error', data.error || 'Error running normality tests');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Variance Tests
    async runVarianceTests(groups) {
        if (!this.currentDatasetId || groups.length < 2) {
            showAlert('warning', 'Please select at least 2 groups for variance tests.');
            return;
        }

        showLoader('Running variance tests...');
        
        try {
            const params = new URLSearchParams();
            groups.forEach(group => params.append('groups', group));
            
            const response = await fetch(`/statistics/api/variance/${this.currentDatasetId}?${params}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayVarianceResults(data);
            } else {
                showAlert('error', data.error || 'Error running variance tests');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Correlation Tests
    async runCorrelationTests(col1, col2) {
        if (!this.currentDatasetId || !col1 || !col2) {
            showAlert('warning', 'Please select two columns for correlation analysis.');
            return;
        }

        showLoader('Running correlation tests...');
        
        try {
            const response = await fetch(`/statistics/api/correlation/${this.currentDatasetId}?col1=${col1}&col2=${col2}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayCorrelationResults(data);
            } else {
                showAlert('error', data.error || 'Error running correlation tests');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Hypothesis Tests
    async runHypothesisTests(testType, groups) {
        if (!this.currentDatasetId || !testType || !groups.length) {
            showAlert('warning', 'Please select test type and groups.');
            return;
        }

        showLoader('Running hypothesis tests...');
        
        try {
            const params = new URLSearchParams();
            params.append('test_type', testType);
            groups.forEach(group => params.append('groups', group));
            
            const response = await fetch(`/statistics/api/hypothesis/${this.currentDatasetId}?${params}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayHypothesisResults(data);
            } else {
                showAlert('error', data.error || 'Error running hypothesis tests');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Comprehensive Analysis
    async runComprehensiveAnalysis() {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Running comprehensive statistical analysis...');
        
        try {
            const response = await fetch(`/statistics/api/comprehensive/${this.currentDatasetId}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayComprehensiveResults(data);
            } else {
                showAlert('error', data.error || 'Error running comprehensive analysis');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Display Methods
    displayNormalityResults(data) {
        const container = document.getElementById('normality-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-chart-line me-2"></i>Normality Test Results</h5>';
        
        Object.keys(data).forEach(column => {
            const tests = data[column];
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">${column}</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
            `;
            
            Object.keys(tests).forEach(testName => {
                const test = tests[testName];
                const badgeClass = test.is_normal ? 'success' : 'danger';
                
                html += `
                    <div class="col-md-6 mb-3">
                        <div class="border rounded p-3">
                            <h6>${testName.replace('_', ' ').toUpperCase()}</h6>
                            <p class="mb-1">Statistic: ${test.statistic.toFixed(4)}</p>
                            <p class="mb-1">P-value: ${test.p_value.toFixed(4)}</p>
                            <span class="badge bg-${badgeClass}">
                                ${test.is_normal ? 'Normal' : 'Not Normal'}
                            </span>
                            <p class="text-muted small mt-2">${test.interpretation}</p>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div></div>';
        });
        
        container.innerHTML = html;
    }

    displayVarianceResults(data) {
        const container = document.getElementById('variance-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-chart-bar me-2"></i>Variance Test Results</h5>';
        
        Object.keys(data).forEach(testName => {
            const test = data[testName];
            const badgeClass = test.equal_variances ? 'success' : 'warning';
            
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-body">
                        <h6>${testName.replace('_', ' ').toUpperCase()}</h6>
                        <p class="mb-1">Statistic: ${test.statistic.toFixed(4)}</p>
                        <p class="mb-1">P-value: ${test.p_value.toFixed(4)}</p>
                        <span class="badge bg-${badgeClass}">
                            ${test.equal_variances ? 'Equal Variances' : 'Unequal Variances'}
                        </span>
                        <p class="text-muted small mt-2">${test.interpretation}</p>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    displayCorrelationResults(data) {
        const container = document.getElementById('correlation-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-link me-2"></i>Correlation Test Results</h5>';
        
        Object.keys(data).forEach(testName => {
            const test = data[testName];
            const badgeClass = test.significant ? 'success' : 'secondary';
            
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-body">
                        <h6>${testName.replace('_', ' ').toUpperCase()}</h6>
                        <p class="mb-1">Correlation: ${test.correlation.toFixed(4)}</p>
                        <p class="mb-1">P-value: ${test.p_value.toFixed(4)}</p>
                        <span class="badge bg-${badgeClass}">
                            ${test.significant ? 'Significant' : 'Not Significant'}
                        </span>
                        <p class="text-muted small mt-2">${test.interpretation}</p>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    displayHypothesisResults(data) {
        const container = document.getElementById('hypothesis-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-flask me-2"></i>Hypothesis Test Results</h5>';
        
        Object.keys(data).forEach(testName => {
            const test = data[testName];
            const badgeClass = test.significant ? 'success' : 'secondary';
            
            html += `
                <div class="card bg-dark border-secondary mb-3">
                    <div class="card-body">
                        <h6>${testName.replace('_', ' ').toUpperCase()}</h6>
                        ${test.statistic ? `<p class="mb-1">Statistic: ${test.statistic.toFixed(4)}</p>` : ''}
                        ${test.p_value ? `<p class="mb-1">P-value: ${test.p_value.toFixed(4)}</p>` : ''}
                        <span class="badge bg-${badgeClass}">
                            ${test.significant ? 'Significant' : 'Not Significant'}
                        </span>
                        <p class="text-muted small mt-2">${test.interpretation}</p>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    displayComprehensiveResults(data) {
        const container = document.getElementById('comprehensive-results');
        if (!container) return;

        let html = '<h5><i class="fas fa-chart-area me-2"></i>Comprehensive Statistical Analysis</h5>';
        
        // Dataset Summary
        if (data.dataset_summary) {
            const summary = data.dataset_summary;
            html += `
                <div class="card bg-dark border-primary mb-3">
                    <div class="card-header bg-primary">
                        <h6 class="mb-0">Dataset Summary</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3"><strong>Total Columns:</strong> ${summary.total_columns}</div>
                            <div class="col-md-3"><strong>Numeric Columns:</strong> ${summary.numeric_columns}</div>
                            <div class="col-md-3"><strong>Categorical Columns:</strong> ${summary.categorical_columns}</div>
                            <div class="col-md-3"><strong>Total Rows:</strong> ${summary.total_rows}</div>
                        </div>
                        <div class="mt-2">
                            <strong>Missing Data:</strong> ${summary.missing_data_percentage.toFixed(2)}%
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Display other analysis sections
        ['normality_analysis', 'correlation_analysis', 'variance_analysis', 'independence_tests'].forEach(section => {
            if (data[section] && Object.keys(data[section]).length > 0) {
                html += `
                    <div class="card bg-dark border-secondary mb-3">
                        <div class="card-header">
                            <h6 class="mb-0">${section.replace('_', ' ').toUpperCase()}</h6>
                        </div>
                        <div class="card-body">
                            <pre class="text-light">${JSON.stringify(data[section], null, 2)}</pre>
                        </div>
                    </div>
                `;
            }
        });
        
        container.innerHTML = html;
    }
}

// Initialize statistics manager
const statisticsManager = new StatisticsManager();

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Extract dataset ID from URL if present
    const pathParts = window.location.pathname.split('/');
    const datasetIdIndex = pathParts.indexOf('statistics') + 1;
    if (datasetIdIndex > 0 && pathParts[datasetIdIndex]) {
        statisticsManager.setDatasetId(parseInt(pathParts[datasetIdIndex]));
    }
});

// Global functions for use in HTML
function runNormalityTest() {
    const selectedColumns = Array.from(document.querySelectorAll('input[name="normality-columns"]:checked'))
                                .map(input => input.value);
    statisticsManager.runNormalityTests(selectedColumns);
}

function runVarianceTest() {
    const selectedGroups = Array.from(document.querySelectorAll('input[name="variance-groups"]:checked'))
                               .map(input => input.value);
    statisticsManager.runVarianceTests(selectedGroups);
}

function runCorrelationTest() {
    const col1 = document.getElementById('correlation-col1')?.value;
    const col2 = document.getElementById('correlation-col2')?.value;
    statisticsManager.runCorrelationTests(col1, col2);
}

function runHypothesisTest() {
    const testType = document.getElementById('hypothesis-test-type')?.value;
    const selectedGroups = Array.from(document.querySelectorAll('input[name="hypothesis-groups"]:checked'))
                               .map(input => input.value);
    statisticsManager.runHypothesisTests(testType, selectedGroups);
}

function runComprehensiveAnalysis() {
    statisticsManager.runComprehensiveAnalysis();
}