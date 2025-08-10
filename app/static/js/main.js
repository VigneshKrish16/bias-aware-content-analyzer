document.addEventListener('DOMContentLoaded', function() {
    const analysisForm = document.getElementById('analysis-form');
    const resultsSection = document.getElementById('results');
    let biasChart = null;

    analysisForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = document.getElementById('query').value;
        const domain = document.getElementById('domain').value;
        
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                domain: domain
            })
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
            resultsSection.style.display = 'block';
            window.scrollTo({
                top: resultsSection.offsetTop,
                behavior: 'smooth'
            });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please try again.');
        });
    });

    function displayResults(data) {
        // Update metric values
        document.getElementById('biq-value').textContent = data.biq_score.toFixed(2);
        document.getElementById('cfs-value').textContent = data.cfs_score.toFixed(2);
        document.getElementById('cds-value').textContent = data.cds_score.toFixed(2);
        
        // Animate metric bars
        animateMetric('biq-fill', data.biq_score);
        animateMetric('cfs-fill', data.cfs_score);
        animateMetric('cds-fill', data.cds_score);
        
        // Display documents
        const documentList = document.getElementById('document-list');
        documentList.innerHTML = '';
        
        data.documents.forEach(doc => {
            const docCard = document.createElement('div');
            docCard.className = 'document-card';
            docCard.innerHTML = `
                <h4>${doc.title || 'Document'}</h4>
                <div class="document-meta">
                    <span>Cluster: ${doc.cluster_id}</span>
                    <span>Group: ${doc.demographic_group}</span>
                </div>
                <p>${doc.content || 'No content available'}</p>
                <div class="document-meta">
                    <span>Bias Score: ${doc.bias_score ? doc.bias_score.toFixed(2) : 'N/A'}</span>
                    <span>Relevance: ${doc.relevance ? doc.relevance.toFixed(2) : 'N/A'}</span>
                </div>
            `;
            documentList.appendChild(docCard);
        });
        
        // Create/update chart
        updateBiasChart(data.metrics);
    }
    
    function animateMetric(elementId, value) {
        const element = document.getElementById(elementId);
        // Reset width to 0 before animating
        element.style.width = '0';
        // Force reflow
        void element.offsetWidth;
        // Animate to final width
        element.style.width = `${value * 100}%`;
    }
    
    function updateBiasChart(metrics) {
        const ctx = document.getElementById('bias-chart').getContext('2d');
        
        if (biasChart) {
            biasChart.destroy();
        }
        
        biasChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Inclusivity', 'Diversity', 'Fairness', 'Demographic Parity', 'Intersectional Fairness'],
                datasets: [{
                    label: 'Bias Metrics',
                    data: [
                        metrics.inclusivity,
                        metrics.diversity,
                        metrics.fairness,
                        metrics.demographic_parity || 0.7, // Example value
                        metrics.intersectional_fairness || 0.6 // Example value
                    ],
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(75, 192, 192, 1)'
                }]
            },
            options: {
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                },
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
});