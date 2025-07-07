// Visualization JavaScript Functions

class VisualizationManager {
    constructor() {
        this.currentDatasetId = null;
        this.charts = {};
    }

    setDatasetId(datasetId) {
        this.currentDatasetId = datasetId;
    }

    // Generate Charts
    async generateCharts(chartTypes) {
        if (!this.currentDatasetId || !chartTypes.length) {
            showAlert('warning', 'Please select dataset and chart types.');
            return;
        }

        showLoader('Generating visualizations...');
        
        try {
            const params = new URLSearchParams();
            chartTypes.forEach(type => params.append('charts', type));
            
            const response = await fetch(`/visualization/api/generate/${this.currentDatasetId}?${params}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayCharts(data);
            } else {
                showAlert('error', data.error || 'Error generating charts');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Generate Custom Chart
    async generateCustomChart(chartConfig) {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Generating custom chart...');
        
        try {
            const response = await fetch(`/visualization/api/custom/${this.currentDatasetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(chartConfig)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayCustomChart(data);
            } else {
                showAlert('error', data.error || 'Error generating custom chart');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    // Display Charts
    displayCharts(data) {
        const container = document.getElementById('charts-container');
        if (!container) return;

        let html = '<h5><i class="fas fa-chart-pie me-2"></i>Generated Visualizations</h5>';
        
        Object.keys(data).forEach(chartType => {
            const chartData = data[chartType];
            
            if (chartData.error) {
                html += `
                    <div class="card bg-dark border-danger mb-3">
                        <div class="card-header bg-danger">
                            <h6 class="mb-0">${chartType.replace('_', ' ').toUpperCase()}</h6>
                        </div>
                        <div class="card-body">
                            <p class="text-danger">${chartData.error}</p>
                        </div>
                    </div>
                `;
            } else {
                html += `
                    <div class="card bg-dark border-secondary mb-3">
                        <div class="card-header">
                            <div class="d-flex justify-content-between align-items-center">
                                <h6 class="mb-0">${chartType.replace('_', ' ').toUpperCase()}</h6>
                                <div class="btn-group btn-group-sm">
                                    <button class="btn btn-outline-primary" onclick="downloadChart('${chartType}')">
                                        <i class="fas fa-download"></i>
                                    </button>
                                    <button class="btn btn-outline-secondary" onclick="fullscreenChart('${chartType}')">
                                        <i class="fas fa-expand"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <div id="chart-${chartType}" class="chart-container">
                                ${chartData.html || '<div class="text-center">Chart generated successfully</div>'}
                            </div>
                        </div>
                    </div>
                `;
            }
        });
        
        container.innerHTML = html;
        
        // Load chart images or plots if available
        this.loadChartImages(data);
    }

    displayCustomChart(data) {
        const container = document.getElementById('custom-chart-container');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${data.error}
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="card bg-dark border-secondary">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">Custom Chart</h6>
                            <div class="btn-group btn-group-sm">
                                <button class="btn btn-outline-primary" onclick="downloadCustomChart()">
                                    <i class="fas fa-download"></i>
                                </button>
                                <button class="btn btn-outline-secondary" onclick="fullscreenCustomChart()">
                                    <i class="fas fa-expand"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="custom-chart" class="chart-container">
                            ${data.html || '<div class="text-center">Custom chart generated successfully</div>'}
                        </div>
                    </div>
                </div>
            `;
        }
    }

    loadChartImages(data) {
        Object.keys(data).forEach(chartType => {
            const chartData = data[chartType];
            if (chartData.image_url) {
                const container = document.getElementById(`chart-${chartType}`);
                if (container) {
                    container.innerHTML = `<img src="${chartData.image_url}" class="img-fluid" alt="${chartType} chart">`;
                }
            }
        });
    }

    // Chart Controls
    downloadChart(chartType) {
        if (this.currentDatasetId) {
            const link = document.createElement('a');
            link.href = `/visualization/api/download/${this.currentDatasetId}/${chartType}`;
            link.download = `${chartType}_chart.png`;
            link.click();
        }
    }

    fullscreenChart(chartType) {
        const chartContainer = document.getElementById(`chart-${chartType}`);
        if (chartContainer) {
            if (chartContainer.requestFullscreen) {
                chartContainer.requestFullscreen();
            } else if (chartContainer.webkitRequestFullscreen) {
                chartContainer.webkitRequestFullscreen();
            } else if (chartContainer.msRequestFullscreen) {
                chartContainer.msRequestFullscreen();
            }
        }
    }

    // Chart Builder
    buildChart() {
        const chartType = document.getElementById('chart-type')?.value;
        const xColumn = document.getElementById('x-column')?.value;
        const yColumn = document.getElementById('y-column')?.value;
        const colorColumn = document.getElementById('color-column')?.value;
        const sizeColumn = document.getElementById('size-column')?.value;

        if (!chartType || !xColumn) {
            showAlert('warning', 'Please select chart type and X column.');
            return;
        }

        const chartConfig = {
            chart_type: chartType,
            x_column: xColumn,
            y_column: yColumn,
            color_column: colorColumn,
            size_column: sizeColumn,
            title: document.getElementById('chart-title')?.value || '',
            x_label: document.getElementById('x-label')?.value || xColumn,
            y_label: document.getElementById('y-label')?.value || yColumn
        };

        this.generateCustomChart(chartConfig);
    }

    // Interactive Features
    updateColumns() {
        const chartType = document.getElementById('chart-type')?.value;
        const yColumnContainer = document.getElementById('y-column-container');
        const colorColumnContainer = document.getElementById('color-column-container');
        const sizeColumnContainer = document.getElementById('size-column-container');

        if (!yColumnContainer) return;

        // Show/hide Y column based on chart type
        if (['histogram', 'box_plot', 'violin_plot'].includes(chartType)) {
            yColumnContainer.style.display = 'none';
        } else {
            yColumnContainer.style.display = 'block';
        }

        // Show/hide additional options
        if (colorColumnContainer) {
            if (['scatter', 'bubble', 'bar'].includes(chartType)) {
                colorColumnContainer.style.display = 'block';
            } else {
                colorColumnContainer.style.display = 'none';
            }
        }

        if (sizeColumnContainer) {
            if (chartType === 'bubble') {
                sizeColumnContainer.style.display = 'block';
            } else {
                sizeColumnContainer.style.display = 'none';
            }
        }
    }
}

// Data Table Manager
class DataTableManager {
    constructor() {
        this.currentDatasetId = null;
        this.table = null;
    }

    setDatasetId(datasetId) {
        this.currentDatasetId = datasetId;
    }

    async loadData(page = 0, pageSize = 100) {
        if (!this.currentDatasetId) {
            showAlert('warning', 'Please select a dataset.');
            return;
        }

        showLoader('Loading data...');
        
        try {
            const response = await fetch(`/visualization/api/data/${this.currentDatasetId}?page=${page}&size=${pageSize}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayTable(data);
            } else {
                showAlert('error', data.error || 'Error loading data');
            }
        } catch (error) {
            showAlert('error', 'Network error: ' + error.message);
        } finally {
            hideLoader();
        }
    }

    displayTable(data) {
        const container = document.getElementById('data-table-container');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${data.error}
                </div>
            `;
            return;
        }

        // Create table header
        let html = `
            <div class="table-responsive">
                <table class="table table-dark table-striped table-hover">
                    <thead class="table-secondary">
                        <tr>
        `;
        
        data.columns.forEach(column => {
            html += `<th>${column}</th>`;
        });
        
        html += `
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        // Add table rows
        data.data.forEach(row => {
            html += '<tr>';
            row.forEach(cell => {
                html += `<td>${cell !== null ? cell : '<span class="text-muted">null</span>'}</td>`;
            });
            html += '</tr>';
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        // Add pagination info
        if (data.pagination) {
            html += `
                <div class="d-flex justify-content-between align-items-center mt-3">
                    <div class="text-muted">
                        Showing ${data.pagination.start + 1} to ${data.pagination.end} of ${data.pagination.total} rows
                    </div>
                    <nav>
                        <ul class="pagination pagination-sm mb-0">
                            <li class="page-item ${data.pagination.current_page === 0 ? 'disabled' : ''}">
                                <button class="page-link" onclick="dataTableManager.loadData(${data.pagination.current_page - 1})">Previous</button>
                            </li>
                            <li class="page-item active">
                                <span class="page-link">${data.pagination.current_page + 1}</span>
                            </li>
                            <li class="page-item ${data.pagination.current_page >= data.pagination.total_pages - 1 ? 'disabled' : ''}">
                                <button class="page-link" onclick="dataTableManager.loadData(${data.pagination.current_page + 1})">Next</button>
                            </li>
                        </ul>
                    </nav>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }
}

// Initialize managers
const visualizationManager = new VisualizationManager();
const dataTableManager = new DataTableManager();

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Extract dataset ID from URL if present
    const pathParts = window.location.pathname.split('/');
    const datasetIdIndex = pathParts.indexOf('visualization') + 1;
    if (datasetIdIndex > 0 && pathParts[datasetIdIndex]) {
        const datasetId = parseInt(pathParts[datasetIdIndex]);
        visualizationManager.setDatasetId(datasetId);
        dataTableManager.setDatasetId(datasetId);
    }

    // Chart type change handler
    const chartTypeSelect = document.getElementById('chart-type');
    if (chartTypeSelect) {
        chartTypeSelect.addEventListener('change', () => visualizationManager.updateColumns());
    }
});

// Global functions for use in HTML
function generateCharts() {
    const selectedCharts = Array.from(document.querySelectorAll('input[name="chart-types"]:checked'))
                              .map(input => input.value);
    visualizationManager.generateCharts(selectedCharts);
}

function buildChart() {
    visualizationManager.buildChart();
}

function downloadChart(chartType) {
    visualizationManager.downloadChart(chartType);
}

function fullscreenChart(chartType) {
    visualizationManager.fullscreenChart(chartType);
}

function downloadCustomChart() {
    visualizationManager.downloadChart('custom');
}

function fullscreenCustomChart() {
    visualizationManager.fullscreenChart('custom');
}

function loadDataTable() {
    dataTableManager.loadData();
}

function exportData(format) {
    if (dataTableManager.currentDatasetId) {
        window.location.href = `/visualization/api/export/${dataTableManager.currentDatasetId}?format=${format}`;
    }
}