// Assignment 3 JavaScript - Utility functions

document.addEventListener('DOMContentLoaded', function() {
    console.log('Assignment 3 - Hugging Face Model loaded');
});

// Fetch API helper
async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Format percentage
function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

// Format number to 4 decimal places
function formatMetric(value) {
    return parseFloat(value).toFixed(4);
}

// Show notification
function showNotification(message, type = 'info') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    alert.style.position = 'fixed';
    alert.style.top = '20px';
    alert.style.right = '20px';
    alert.style.zIndex = '1000';
    alert.style.minWidth = '300px';
    
    document.body.appendChild(alert);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Get model info
async function getModelInfo() {
    try {
        return await fetchAPI('/model/info');
    } catch (error) {
        console.error('Error getting model info:', error);
        return null;
    }
}

// Load model statistics
async function loadModelStats() {
    try {
        return await fetchAPI('/api/stats');
    } catch (error) {
        console.error('Error loading stats:', error);
        return null;
    }
}

// Single prediction
async function makePrediction(text) {
    try {
        return await fetchAPI('/predict', {
            method: 'POST',
            body: JSON.stringify({text: text})
        });
    } catch (error) {
        console.error('Error making prediction:', error);
        return null;
    }
}

// Batch prediction
async function makeBatchPrediction(texts) {
    try {
        return await fetchAPI('/predict/batch', {
            method: 'POST',
            body: JSON.stringify({texts: texts})
        });
    } catch (error) {
        console.error('Error making batch prediction:', error);
        return null;
    }
}

// Test connection
async function testConnection() {
    try {
        const result = await fetchAPI('/test');
        console.log('Connection test:', result);
        return result;
    } catch (error) {
        console.error('Connection test failed:', error);
        return null;
    }
}
