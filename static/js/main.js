// EDA Pro Main JavaScript Functions

// Global variables
let currentDatasetId = null;
let loadingTimeout = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkForUpdates();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize theme
    initializeTheme();
    
    // Setup CSRF token for AJAX requests
    setupCSRFToken();
    
    // Initialize notification system
    initializeNotifications();
    
    console.log('EDA Pro initialized successfully');
}

function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function initializeTheme() {
    // Apply dark theme
    document.documentElement.setAttribute('data-bs-theme', 'dark');
    
    // Apply custom theme preferences if stored
    const savedTheme = localStorage.getItem('eda-pro-theme');
    if (savedTheme) {
        applyThemePreferences(JSON.parse(savedTheme));
    }
}

function applyThemePreferences(preferences) {
    if (preferences.primaryColor) {
        document.documentElement.style.setProperty('--primary-color', preferences.primaryColor);
    }
    if (preferences.accentColor) {
        document.documentElement.style.setProperty('--neon-blue', preferences.accentColor);
    }
}

function setupCSRFToken() {
    // Setup CSRF token for AJAX requests if needed
    const token = document.querySelector('meta[name="csrf-token"]');
    if (token) {
        $.ajaxSetup({
            beforeSend: function(xhr, settings) {
                if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
                    xhr.setRequestHeader("X-CSRFToken", token.getAttribute('content'));
                }
            }
        });
    }
}

function setupEventListeners() {
    // Global keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Handle file drag and drop
    setupDragAndDrop();
    
    // Auto-save forms
    setupAutoSave();
    
    // Progress tracking
    setupProgressTracking();
}

function handleKeyboardShortcuts(event) {
    // Ctrl+U for upload
    if (event.ctrlKey && event.key === 'u') {
        event.preventDefault();
        window.location.href = '/upload';
    }
    
    // Ctrl+D for dashboard
    if (event.ctrlKey && event.key === 'd') {
        event.preventDefault();
        window.location.href = '/dashboard';
    }
    
    // Ctrl+R for reports
    if (event.ctrlKey && event.key === 'r') {
        event.preventDefault();
        window.location.href = '/reports';
    }
    
    // Escape to close modals
    if (event.key === 'Escape') {
        const openModals = document.querySelectorAll('.modal.show');
        openModals.forEach(modal => {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    }
}

function setupDragAndDrop() {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area
    ['dragenter', 'dragover'].forEach(eventName => {
        document.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        const uploadArea = document.querySelector('.upload-area');
        if (uploadArea) {
            uploadArea.classList.add('drag-over');
        }
    }
    
    function unhighlight(e) {
        const uploadArea = document.querySelector('.upload-area');
        if (uploadArea) {
            uploadArea.classList.remove('drag-over');
        }
    }
    
    // Handle dropped files
    document.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0 && window.location.pathname === '/upload') {
            handleFileSelect(files[0]);
        }
    }
}

function setupAutoSave() {
    // Auto-save form data to localStorage
    const forms = document.querySelectorAll('form[data-autosave]');
    
    forms.forEach(form => {
        const formId = form.getAttribute('data-autosave');
        
        // Load saved data
        const savedData = localStorage.getItem(`autosave-${formId}`);
        if (savedData) {
            const data = JSON.parse(savedData);
            restoreFormData(form, data);
        }
        
        // Save on input
        form.addEventListener('input', debounce(() => {
            const formData = new FormData(form);
            const data = Object.fromEntries(formData);
            localStorage.setItem(`autosave-${formId}`, JSON.stringify(data));
        }, 1000));
        
        // Clear on submit
        form.addEventListener('submit', () => {
            localStorage.removeItem(`autosave-${formId}`);
        });
    });
}

function restoreFormData(form, data) {
    Object.keys(data).forEach(key => {
        const input = form.querySelector(`[name="${key}"]`);
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = data[key] === 'on';
            } else {
                input.value = data[key];
            }
        }
    });
}

function setupProgressTracking() {
    // Track user progress through the application
    const milestones = [
        'data_uploaded',
        'first_analysis',
        'first_visualization',
        'first_statistical_test',
        'first_ml_model',
        'first_report'
    ];
    
    function trackMilestone(milestone) {
        const progress = JSON.parse(localStorage.getItem('user-progress') || '{}');
        if (!progress[milestone]) {
            progress[milestone] = new Date().toISOString();
            localStorage.setItem('user-progress', JSON.stringify(progress));
            
            // Show congratulations message
            showMilestoneAchievement(milestone);
        }
    }
    
    // Expose tracking function globally
    window.trackMilestone = trackMilestone;
}

function showMilestoneAchievement(milestone) {
    const messages = {
        'data_uploaded': 'Congratulations! You\'ve uploaded your first dataset!',
        'first_analysis': 'Great! You\'ve completed your first analysis!',
        'first_visualization': 'Awesome! You\'ve created your first visualization!',
        'first_statistical_test': 'Excellent! You\'ve run your first statistical test!',
        'first_ml_model': 'Outstanding! You\'ve trained your first ML model!',
        'first_report': 'Perfect! You\'ve generated your first report!'
    };
    
    showAlert('success', messages[milestone], 5000);
}

// Utility Functions

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    
    if (overlay && text) {
        text.textContent = message;
        overlay.classList.remove('d-none');
        
        // Auto-hide after 30 seconds
        clearTimeout(loadingTimeout);
        loadingTimeout = setTimeout(() => {
            hideLoading();
        }, 30000);
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('d-none');
    }
    clearTimeout(loadingTimeout);
}

function showAlert(type, message, duration = 3000) {
    const alertContainer = createAlertContainer();
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    
    const icon = getAlertIcon(type);
    
    alertDiv.innerHTML = `
        <i class="fas ${icon} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.appendChild(alertDiv);
    
    // Auto-dismiss
    if (duration > 0) {
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.classList.remove('show');
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.parentNode.removeChild(alertDiv);
                    }
                }, 150);
            }
        }, duration);
    }
    
    // Add animation
    alertDiv.classList.add('fade-in');
}

function createAlertContainer() {
    let container = document.getElementById('alert-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'alert-container';
        container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 1050;
            max-width: 400px;
        `;
        document.body.appendChild(container);
    }
    return container;
}

function getAlertIcon(type) {
    const icons = {
        'success': 'fa-check-circle',
        'danger': 'fa-exclamation-triangle',
        'warning': 'fa-exclamation-circle',
        'info': 'fa-info-circle',
        'primary': 'fa-info-circle'
    };
    return icons[type] || 'fa-info-circle';
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return 'N/A';
    
    if (typeof num !== 'number') {
        num = parseFloat(num);
        if (isNaN(num)) return 'N/A';
    }
    
    return num.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: decimals
    });
}

function formatPercentage(value, decimals = 1) {
    if (value === null || value === undefined) return 'N/A';
    
    if (typeof value !== 'number') {
        value = parseFloat(value);
        if (isNaN(value)) return 'N/A';
    }
    
    return (value * 100).toFixed(decimals) + '%';
}

function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showAlert('success', 'Copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            fallbackCopyTextToClipboard(text);
        });
    } else {
        fallbackCopyTextToClipboard(text);
    }
}

function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.top = '0';
    textArea.style.left = '0';
    textArea.style.position = 'fixed';
    
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showAlert('success', 'Copied to clipboard!');
    } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
        showAlert('danger', 'Failed to copy to clipboard');
    }
    
    document.body.removeChild(textArea);
}

function exportData(data, filename, type = 'json') {
    let content, mimeType;
    
    switch (type) {
        case 'json':
            content = JSON.stringify(data, null, 2);
            mimeType = 'application/json';
            break;
        case 'csv':
            content = convertToCSV(data);
            mimeType = 'text/csv';
            break;
        case 'txt':
            content = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
            mimeType = 'text/plain';
            break;
        default:
            content = JSON.stringify(data, null, 2);
            mimeType = 'application/json';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function convertToCSV(data) {
    if (!Array.isArray(data) || data.length === 0) {
        return '';
    }
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
            const value = row[header];
            return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
        }).join(','))
    ].join('\n');
    
    return csvContent;
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validateURL(url) {
    try {
        new URL(url);
        return true;
    } catch {
        return false;
    }
}

function sanitizeHTML(str) {
    const temp = document.createElement('div');
    temp.textContent = str;
    return temp.innerHTML;
}

function createToast(title, message, type = 'info') {
    const toastContainer = getToastContainer();
    
    const toastDiv = document.createElement('div');
    toastDiv.className = 'toast';
    toastDiv.setAttribute('role', 'alert');
    toastDiv.innerHTML = `
        <div class="toast-header">
            <i class="fas ${getAlertIcon(type)} me-2 text-${type}"></i>
            <strong class="me-auto">${sanitizeHTML(title)}</strong>
            <small>now</small>
            <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
        </div>
        <div class="toast-body">
            ${sanitizeHTML(message)}
        </div>
    `;
    
    toastContainer.appendChild(toastDiv);
    
    const toast = new bootstrap.Toast(toastDiv);
    toast.show();
    
    toastDiv.addEventListener('hidden.bs.toast', () => {
        toastDiv.remove();
    });
}

function getToastContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '1055';
        document.body.appendChild(container);
    }
    return container;
}

function initializeNotifications() {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

function showNotification(title, options = {}) {
    if ('Notification' in window && Notification.permission === 'granted') {
        const notification = new Notification(title, {
            icon: '/static/favicon.ico',
            badge: '/static/favicon.ico',
            ...options
        });
        
        notification.onclick = function() {
            window.focus();
            notification.close();
        };
        
        setTimeout(() => {
            notification.close();
        }, 5000);
    }
}

function checkForUpdates() {
    // Check for application updates
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.ready.then(registration => {
            registration.addEventListener('updatefound', () => {
                const newWorker = registration.installing;
                newWorker.addEventListener('statechange', () => {
                    if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                        showAlert('info', 'A new version is available. Refresh to update.', 0);
                    }
                });
            });
        });
    }
}

// Navigation functions
function showColumnAnalysis() {
    if (currentDatasetId) {
        window.location.href = `/analysis/column/${currentDatasetId}`;
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

function showStatisticalTests() {
    if (currentDatasetId) {
        window.location.href = `/statistics/${currentDatasetId}`;
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

function showMLModels() {
    if (currentDatasetId) {
        window.location.href = `/ml/${currentDatasetId}`;
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

// Missing functions that are called from templates
function generateAllVisualizations() {
    if (currentDatasetId) {
        showLoading('Generating visualizations...');
        fetch(`/visualization/generate/${currentDatasetId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.success) {
                showAlert('success', 'Visualizations generated successfully!');
                setTimeout(() => location.reload(), 1000);
            } else {
                showAlert('danger', data.message || 'Failed to generate visualizations');
            }
        })
        .catch(error => {
            hideLoading();
            showAlert('danger', 'Error generating visualizations');
            console.error('Error:', error);
        });
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

function generateCorrelations() {
    if (currentDatasetId) {
        showLoading('Analyzing correlations...');
        fetch(`/analysis/correlations/${currentDatasetId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.success) {
                showAlert('success', 'Correlation analysis completed!');
                setTimeout(() => location.reload(), 1000);
            } else {
                showAlert('danger', data.message || 'Failed to analyze correlations');
            }
        })
        .catch(error => {
            hideLoading();
            showAlert('danger', 'Error analyzing correlations');
            console.error('Error:', error);
        });
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

function generateAIInsights() {
    if (currentDatasetId) {
        showLoading('Generating AI insights...');
        fetch(`/analysis/insights/${currentDatasetId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.success) {
                showAlert('success', 'AI insights generated successfully!');
                setTimeout(() => location.reload(), 1000);
            } else {
                showAlert('danger', data.message || 'Failed to generate insights');
            }
        })
        .catch(error => {
            hideLoading();
            showAlert('danger', 'Error generating insights');
            console.error('Error:', error);
        });
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

function showStatTests() {
    if (currentDatasetId) {
        window.location.href = `/statistics/${currentDatasetId}`;
    } else {
        showAlert('warning', 'Please select a dataset first.');
    }
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
    showAlert('danger', 'An unexpected error occurred. Please refresh the page and try again.');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('warning', 'A background operation failed. Some features may not work correctly.');
});

// Expose global functions
window.EDAProUtils = {
    showLoading,
    hideLoading,
    showAlert,
    formatFileSize,
    formatNumber,
    formatPercentage,
    copyToClipboard,
    exportData,
    createToast,
    showNotification,
    validateEmail,
    validateURL,
    sanitizeHTML,
    debounce,
    throttle
};

console.log('EDA Pro main.js loaded successfully');
