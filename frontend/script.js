document.addEventListener('DOMContentLoaded', function() {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const domainSelect = document.getElementById('domain');
    const biasMitigationSelect = document.getElementById('bias-mitigation');
    const biqScoreElement = document.getElementById('biq-score');
    const inclusivityElement = document.getElementById('inclusivity');
    const diversityElement = document.getElementById('diversity');
    const fairnessElement = document.getElementById('fairness');
    const biqBar = document.getElementById('biq-bar');
    const inclusivityBar = document.getElementById('inclusivity-bar');
    const diversityBar = document.getElementById('diversity-bar');
    const fairnessBar = document.getElementById('fairness-bar');
    const sourcesList = document.getElementById('sources-list');
    const explanationContent = document.getElementById('explanation-content');
    const tabButtons = document.querySelectorAll('.tab-button');
    
    let conversationId = null;
    let currentDocuments = [];
    let currentClusters = [];
    
    // Add a sample welcome message
    addBotMessage("Welcome to the Equitable AI Framework! This system uses Bias Intelligence Quotient (BiQ) metrics and clustering algorithms to mitigate biases in Retrieval-Augmented Generation. How can I assist you today?");
    
    // Handle send button click
    sendButton.addEventListener('click', sendMessage);
    
    // Handle Enter key press
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Handle tab button clicks
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Filter sources based on selected cluster
            filterSources(this.dataset.tab);
        });
    });
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addUserMessage(message);
        userInput.value = '';
        
        // Show typing indicator
        const typingIndicator = showTypingIndicator();
        
        // Prepare request data
        const requestData = {
            text: message,
            domain: domainSelect.value,
            bias_mitigation: biasMitigationSelect.value,
            conversation_id: conversationId
        };
        
        // Simulate API call (in a real implementation, this would be a fetch to your backend)
        simulateApiCall(requestData)
            .then(data => {
                // Remove typing indicator
                chatHistory.removeChild(typingIndicator);
                
                // Update conversation ID
                conversationId = data.conversation_id;
                
                // Add bot response
                addBotMessage(data.response);
                
                // Update metrics
                updateMetrics(data.biq_metrics);
                
                // Update sources
                currentDocuments = data.documents;
                currentClusters = data.clusters;
                updateSources(data.documents, data.clusters);
                
                // Update explanation
                updateExplanation(data.explanation, data.biq_metrics);
            })
            .catch(error => {
                chatHistory.removeChild(typingIndicator);
                addBotMessage("Sorry, I encountered an error. Please try again later.");
                console.error('Error:', error);
            });
    }
    
    function simulateApiCall(requestData) {
        // This is a simulation - in a real app, you'd call your backend API
        return new Promise((resolve) => {
            setTimeout(() => {
                // Simulate different responses based on domain and bias mitigation setting
                const domain = requestData.domain;
                const biasMitigation = requestData.bias_mitigation;
                
                // Sample documents with simulated clusters
                const sampleDocs = generateSampleDocuments(requestData.text, domain);
                const clusters = sampleDocs.map(doc => doc.cluster);
                
                // Calculate simulated BiQ metrics based on mitigation approach
                const biqMetrics = calculateSimulatedBiq(biasMitigation, domain);
                
                // Generate response based on domain
                let responseText = generateResponseText(requestData.text, domain);
                
                // Generate explanation based on mitigation approach
                const explanation = generateExplanation(biasMitigation, biqMetrics);
                
                resolve({
                    conversation_id: conversationId || 'conv_' + Math.random().toString(36).substr(2, 9),
                    response: responseText,
                    biq_metrics: biqMetrics,
                    documents: sampleDocs,
                    clusters: clusters,
                    explanation: explanation
                });
            }, 1500); // Simulate network delay
        });
    }
    
    function generateSampleDocuments(query, domain) {
        // Generate sample documents with different perspectives
        const baseDocs = [];
        
        if (domain === 'healthcare') {
            baseDocs.push(
                {
                    text: "Recent studies show that heart disease affects men more frequently than women, with symptoms often presenting differently between genders.",
                    score: 0.92,
                    cluster: 0,
                    metadata: {
                        source: "Journal of Cardiology",
                        year: "2023",
                        perspective: "Western medicine"
                    }
                },
                {
                    text: "Traditional Chinese medicine approaches to heart health emphasize holistic balance rather than gender-specific treatments.",
                    score: 0.88,
                    cluster: 1,
                    metadata: {
                        source: "TCM Health Review",
                        year: "2022",
                        perspective: "Eastern medicine"
                    }
                },
                {
                    text: "Research indicates socioeconomic factors play a larger role in heart disease outcomes than biological sex differences.",
                    score: 0.85,
                    cluster: 2,
                    metadata: {
                        source: "Social Medicine Journal",
                        year: "2023",
                        perspective: "Social determinants"
                    }
                }
            );
        } else if (domain === 'finance') {
            baseDocs.push(
                {
                    text: "Credit scoring models traditionally favor applicants with long credit histories, which may disadvantage younger borrowers.",
                    score: 0.95,
                    cluster: 0,
                    metadata: {
                        source: "Financial Times",
                        year: "2023",
                        perspective: "Traditional scoring"
                    }
                },
                {
                    text: "Alternative credit scoring using utility payments and rental history can improve access for underserved communities.",
                    score: 0.89,
                    cluster: 1,
                    metadata: {
                        source: "FinTech Journal",
                        year: "2024",
                        perspective: "Alternative data"
                    }
                },
                {
                    text: "Studies show that zip code can be a stronger predictor of loan approval than individual creditworthiness, revealing systemic biases.",
                    score: 0.87,
                    cluster: 2,
                    metadata: {
                        source: "Economic Policy Review",
                        year: "2023",
                        perspective: "Systemic bias"
                    }
                }
            );
        } else {
            // General domain
            baseDocs.push(
                {
                    text: "Standard approaches to " + query + " tend to focus on majority perspectives, which may overlook important alternative viewpoints.",
                    score: 0.91,
                    cluster: 0,
                    metadata: {
                        source: "General Encyclopedia",
                        year: "2023",
                        perspective: "Mainstream"
                    }
                },
                {
                    text: "Alternative research suggests different approaches to " + query + " that challenge conventional wisdom in this field.",
                    score: 0.86,
                    cluster: 1,
                    metadata: {
                        source: "Alternative Perspectives",
                        year: "2023",
                        perspective: "Alternative"
                    }
                },
                {
                    text: "Cultural context plays an important role in understanding " + query + ", with significant variations across different communities.",
                    score: 0.83,
                    cluster: 2,
                    metadata: {
                        source: "Cultural Studies Journal",
                        year: "2022",
                        perspective: "Cultural"
                    }
                }
            );
        }
        
        return baseDocs;
    }
    
    function calculateSimulatedBiq(mitigationApproach, domain) {
        // Base bias levels for different domains
        const domainBias = {
            healthcare: { baseBiq: 0.65, baseInclusivity: 0.6, baseDiversity: 0.55, baseFairness: 0.5 },
            finance: { baseBiq: 0.7, baseInclusivity: 0.55, baseDiversity: 0.5, baseFairness: 0.45 },
            education: { baseBiq: 0.6, baseInclusivity: 0.65, baseDiversity: 0.6, baseFairness: 0.55 },
            general: { baseBiq: 0.55, baseInclusivity: 0.7, baseDiversity: 0.65, baseFairness: 0.6 }
        };
        
        const base = domainBias[domain] || domainBias.general;
        
        // Calculate improvement based on mitigation approach
        let improvementFactor = 0;
        switch (mitigationApproach) {
            case 'full':
                improvementFactor = 0.4;
                break;
            case 'biq-only':
                improvementFactor = 0.25;
                break;
            case 'clustering-only':
                improvementFactor = 0.2;
                break;
            case 'none':
                improvementFactor = 0;
                break;
        }
        
        // Calculate final metrics
        const inclusivity = Math.min(1, base.baseInclusivity + (0.3 * improvementFactor) + (Math.random() * 0.05));
        const diversity = Math.min(1, base.baseDiversity + (0.35 * improvementFactor) + (Math.random() * 0.05));
        const fairness = Math.min(1, base.baseFairness + (0.4 * improvementFactor) + (Math.random() * 0.05));
        
        // BiQ is a weighted sum (0.4*Inclusivity + 0.3*Diversity + 0.3*Fairness)
        const biq = 1 - (0.4 * inclusivity + 0.3 * diversity + 0.3 * fairness);
        
        return {
            biq: biq,
            inclusivity: inclusivity,
            diversity: diversity,
            fairness: fairness
        };
    }
    
    function generateResponseText(query, domain) {
        const domainResponses = {
            healthcare: `Regarding ${query}, our bias-aware retrieval found multiple perspectives. Western medical sources emphasize biological factors, while alternative medicine highlights holistic approaches. Socioeconomic analyses suggest environmental factors may be equally significant. The Equitable AI framework ensures balanced representation of these views.`,
            finance: `For ${query}, our system retrieved diverse financial perspectives. Traditional models focus on credit history, while alternative approaches consider non-traditional data. Research also indicates systemic biases in lending practices. We present these views in proportion to their evidence base while ensuring fair representation.`,
            education: `On the topic of ${query}, educational research varies by cultural context and methodology. Our system retrieved mainstream pedagogical approaches alongside alternative teaching methods and critical analyses of systemic biases in education. The BiQ metric helped balance these perspectives.`,
            general: `About ${query}, the Equitable AI framework retrieved multiple viewpoints. Mainstream perspectives are complemented by alternative analyses and cultural considerations. The clustering algorithm grouped these by thematic similarity, while BiQ scoring ensured balanced representation in the generated response.`
        };
        
        return domainResponses[domain] || domainResponses.general;
    }
    
    function generateExplanation(mitigationApproach, biqMetrics) {
        const approachExplanations = {
            'full': 'The system applied full bias mitigation using both BiQ scoring and clustering. Documents were retrieved based on relevance, then clustered by perspective. BiQ metrics guided final selection to ensure balanced representation across clusters.',
            'biq-only': 'The system used BiQ scoring for bias mitigation without clustering. Documents were scored individually for inclusivity, diversity and fairness, with the highest combined scores selected for generation.',
            'clustering-only': 'The system used clustering for diversity but without BiQ scoring. Documents were grouped by perspective, with representatives selected from each cluster to ensure variety.',
            'none': 'The system performed standard retrieval without specific bias mitigation. Documents were selected purely by relevance score, which may reflect existing biases in the data.'
        };
        
        const metricExplanations = `
            <p><strong>Bias Metrics Analysis:</strong></p>
            <ul>
                <li><strong>Inclusivity (${biqMetrics.inclusivity.toFixed(2)}):</strong> Measures representation balance between different groups in the retrieved content.</li>
                <li><strong>Diversity (${biqMetrics.diversity.toFixed(2)}):</strong> Assesses the variety of perspectives present in the content using information theory.</li>
                <li><strong>Fairness (${biqMetrics.fairness.toFixed(2)}):</strong> Evaluates disparities in representation likelihood across protected attributes.</li>
            </ul>
            <p>The overall <strong>BiQ score (${biqMetrics.biq.toFixed(2)})</strong> combines these metrics to quantify bias in the system's output.</p>
        `;
        
        return approachExplanations[mitigationApproach] + metricExplanations;
    }
    
    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div>${text}</div>
            <div class="message-meta">
                <span><i class="fas fa-user"></i> You</span>
                <span>${new Date().toLocaleTimeString()}</span>
            </div>
        `;
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
    }
    
    function addBotMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.innerHTML = `
            <div>${text}</div>
            <div class="message-meta">
                <span><i class="fas fa-robot"></i> Equitable AI</span>
                <span>${new Date().toLocaleTimeString()}</span>
            </div>
        `;
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
    }
    
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="margin-left: 10px;">Analyzing with BiQ metrics...</span>
        `;
        chatHistory.appendChild(typingDiv);
        scrollToBottom();
        return typingDiv;
    }
    
    function updateMetrics(metrics) {
        // Update metric values
        biqScoreElement.textContent = metrics.biq.toFixed(2);
        inclusivityElement.textContent = metrics.inclusivity.toFixed(2);
        diversityElement.textContent = metrics.diversity.toFixed(2);
        fairnessElement.textContent = metrics.fairness.toFixed(2);
        
        // Update progress bars (inverted for BiQ since lower is better)
        biqBar.style.width = `${(1 - metrics.biq) * 100}%`;
        biqBar.style.backgroundColor = getMetricColor(1 - metrics.biq);
        
        inclusivityBar.style.width = `${metrics.inclusivity * 100}%`;
        inclusivityBar.style.backgroundColor = getMetricColor(metrics.inclusivity);
        
        diversityBar.style.width = `${metrics.diversity * 100}%`;
        diversityBar.style.backgroundColor = getMetricColor(metrics.diversity);
        
        fairnessBar.style.width = `${metrics.fairness * 100}%`;
        fairnessBar.style.backgroundColor = getMetricColor(metrics.fairness);
    }
    
    function getMetricColor(value) {
        // Return color based on metric value (red to green gradient)
        if (value < 0.5) return '#ef4444'; // red
        if (value < 0.7) return '#f59e0b'; // amber
        if (value < 0.85) return '#10b981'; // emerald
        return '#059669'; // green
    }
    
    function updateSources(documents, clusters) {
        // Clear previous sources
        sourcesList.innerHTML = '';
        
        // Add new sources
        documents.forEach((doc, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            // Get cluster color
            const clusterColor = getClusterColor(clusters[index]);
            
            // Format metadata
            let metadataHtml = '';
            if (doc.metadata) {
                metadataHtml = Object.entries(doc.metadata)
                    .map(([key, value]) => 
                        `<span><i class="fas fa-tag"></i> ${key}: ${value}</span>`
                    )
                    .join('');
            }
            
            sourceItem.innerHTML = `
                <div class="source-text">${doc.text}</div>
                <div class="source-meta">
                    ${metadataHtml}
                    <span class="source-score">Score: ${doc.score.toFixed(2)}</span>
                </div>
            `;
            
            // Apply cluster color as border
            sourceItem.style.borderLeftColor = clusterColor;
            sourcesList.appendChild(sourceItem);
        });
    }
    
    function filterSources(clusterTab) {
        if (clusterTab === 'cluster-0') {
            // Show all sources
            const sourceItems = document.querySelectorAll('.source-item');
            sourceItems.forEach(item => item.style.display = '');
        } else {
            const clusterIndex = parseInt(clusterTab.split('-')[1]) - 1;
            const sourceItems = document.querySelectorAll('.source-item');
            
            sourceItems.forEach((item, index) => {
                if (currentClusters[index] === clusterIndex) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        }
    }
    
    function updateExplanation(text, metrics) {
        explanationContent.innerHTML = text;
    }
    
    function getClusterColor(clusterIndex) {
        const colors = ['#5c6bc0', '#26a69a', '#ff7043', '#ab47bc', '#ec407a'];
        return colors[clusterIndex % colors.length];
    }
    
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});