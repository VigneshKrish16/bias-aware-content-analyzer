import os
from flask import Flask, render_template, request, jsonify
from models.retrieval import BiasAwareRetriever
from models.bias_detection import BiasDetector
from models.clustering import ContentClusterer

# Initialize Flask app with correct template and static folders
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'app', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'app', 'static'))

# Initialize components
retriever = BiasAwareRetriever()
bias_detector = BiasDetector()
clusterer = ContentClusterer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    query = data.get('query')
    domain = data.get('domain', 'general')
    
    # Retrieve content with bias awareness
    documents = retriever.retrieve(query, domain)
    
    # Cluster documents for diversity
    clustered_docs = clusterer.cluster(documents)
    
    # Calculate bias metrics
    bias_metrics = bias_detector.calculate_metrics(clustered_docs)
    
    # Prepare results
    results = {
        'documents': clustered_docs,
        'metrics': bias_metrics,
        'biq_score': bias_metrics['biq'],
        'cfs_score': bias_metrics['cfs'],
        'cds_score': bias_metrics['cds']
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)