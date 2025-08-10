import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import spacy
from collections import defaultdict

# Disable sidebar navigation
st.set_page_config(initial_sidebar_state="collapsed")

# Load NLP model for demographic analysis
nlp = spacy.load("en_core_web_sm")

# Custom CSS styling (dark theme from original) with removed sidebar
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .metric-box {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4a6fa5;
    }
    .positive-change {
        color: #2e8b57;
        font-weight: bold;
    }
    .negative-change {
        color: #d62728;
        font-weight: bold;
    }
    .cluster-card {
        border-left: 4px solid #4a6fa5;
        padding: 10px;
        margin: 5px 0;
        background-color: #2d2d2d;
    }
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    /* Center the main content */
    .block-container {
        max-width: 800px;
        padding: 1rem 1rem 10rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Equitable AI Chat")
st.caption("🚀 Bias-Resistant RAG System")

# Initialize models
@st.cache_resource
def load_models():
    llm = ChatOllama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0.3
    )
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, st_model

llm_engine, sentence_model = load_models()

# Knowledge base with bias scores (simulated)
knowledge_base = {
    "healthcare": [
        {"text": "Heart disease affects men more than women according to CDC statistics", 
         "bias": {"gender": 0.8, "race": 0.2, "socioeconomic": 0.3}},
        {"text": "Women often experience atypical heart attack symptoms like fatigue and nausea",
         "bias": {"gender": 0.4, "race": 0.1, "socioeconomic": 0.2}},
        {"text": "African Americans have higher rates of hypertension-related heart disease",
         "bias": {"gender": 0.3, "race": 0.7, "socioeconomic": 0.5}},
        {"text": "Socioeconomic factors significantly impact cardiac care access",
         "bias": {"gender": 0.2, "race": 0.4, "socioeconomic": 0.8}},
    ],
    "finance": [
        {"text": "College graduates get approved for loans 3x more often than non-graduates",
         "bias": {"gender": 0.3, "race": 0.4, "socioeconomic": 0.9}},
        {"text": "Low-income applicants face 40% higher rejection rates for mortgages",
         "bias": {"gender": 0.2, "race": 0.5, "socioeconomic": 0.8}},
        {"text": "Minority neighborhoods receive 25% fewer mortgage approvals according to HUD",
         "bias": {"gender": 0.3, "race": 0.7, "socioeconomic": 0.6}},
        {"text": "Women entrepreneurs get 30% smaller business loans on average",
         "bias": {"gender": 0.7, "race": 0.3, "socioeconomic": 0.4}},
    ],
    "general": [
        {"text": "Educational outcomes vary significantly by socioeconomic status",
         "bias": {"gender": 0.2, "race": 0.3, "socioeconomic": 0.8}},
        {"text": "Urban schools generally have more resources than rural schools",
         "bias": {"gender": 0.1, "race": 0.2, "socioeconomic": 0.7}},
        {"text": "First-generation college students face unique challenges",
         "bias": {"gender": 0.3, "race": 0.4, "socioeconomic": 0.6}},
        {"text": "Gender disparities persist in STEM field participation",
         "bias": {"gender": 0.8, "race": 0.3, "socioeconomic": 0.4}},
    ]
}

# Calculate baseline bias scores for comparison
def calculate_baseline_biases(domain):
    docs = knowledge_base[domain]
    return {
        "gender_bias": np.mean([doc["bias"]["gender"] for doc in docs]),
        "race_bias": np.mean([doc["bias"]["race"] for doc in docs]),
        "socioeconomic_bias": np.mean([doc["bias"]["socioeconomic"] for doc in docs]),
        "overall_bias": np.mean([doc["bias"]["gender"] + doc["bias"]["race"] + doc["bias"]["socioeconomic"] for doc in docs])/3
    }

# Automatic bias weight calculation based on question analysis
def calculate_bias_weights(question):
    doc = nlp(question.lower())
    
    # Initialize default weights (from paper)
    weights = {
        'inclusivity': 0.4,
        'diversity': 0.3,
        'fairness': 0.3
    }
    
    # Adjust weights based on question content
    gender_terms = {'women', 'men', 'gender', 'female', 'male'}
    race_terms = {'black', 'white', 'asian', 'minority', 'race'}
    socio_terms = {'poor', 'rich', 'income', 'wealth', 'socioeconomic'}
    
    if any(token.text in gender_terms for token in doc):
        weights['inclusivity'] = 0.5
        weights['fairness'] = 0.35
    
    if any(token.text in race_terms for token in doc):
        weights['inclusivity'] = 0.5
        weights['diversity'] = 0.4
    
    if any(token.text in socio_terms for token in doc):
        weights['fairness'] = 0.5
        weights['inclusivity'] = 0.3
    
    # Normalize to sum to 1
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

# Calculate BiQ metrics and bias reduction
def calculate_biq_metrics(selected_docs, weights, domain):
    # Get baseline biases for this domain
    baseline = calculate_baseline_biases(domain)
    
    # Calculate current biases in selected docs
    current_gender = np.mean([doc["bias"]["gender"] for doc in selected_docs])
    current_race = np.mean([doc["bias"]["race"] for doc in selected_docs])
    current_socio = np.mean([doc["bias"]["socioeconomic"] for doc in selected_docs])
    current_overall = (current_gender + current_race + current_socio)/3
    
    # Calculate reduction percentages (from paper results)
    reduction = {
        "gender": (baseline["gender_bias"] - current_gender) / baseline["gender_bias"] * 100,
        "race": (baseline["race_bias"] - current_race) / baseline["race_bias"] * 100,
        "socioeconomic": (baseline["socioeconomic_bias"] - current_socio) / baseline["socioeconomic_bias"] * 100,
        "overall": (baseline["overall_bias"] - current_overall) / baseline["overall_bias"] * 100
    }
    
    # Calculate BiQ components (simplified for demo)
    inclusivity = 1 - current_gender
    diversity = 1 - (current_race + current_socio)/2
    fairness = 1 - current_overall
    
    return {
        'inclusivity': max(0, min(1, inclusivity)),
        'diversity': max(0, min(1, diversity)),
        'fairness': max(0, min(1, fairness)),
        'biq': (weights['inclusivity'] * inclusivity + 
               weights['diversity'] * diversity + 
               weights['fairness'] * fairness),
        'bias_reduction': reduction,
        'current_biases': {
            'gender': current_gender,
            'race': current_race,
            'socioeconomic': current_socio,
            'overall': current_overall
        },
        'baseline_biases': baseline
    }

# Define the display_bias_metrics function before it's called
def display_bias_metrics(metrics):
    st.subheader("Bias Reduction Metrics")
    
    # Overall BiQ Score
    st.metric("Overall BiQ Score", f"{metrics['biq']:.2f}/1.0", 
              f"{metrics['bias_reduction']['overall']:.1f}% reduction from baseline")
    
    # Bias reduction columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gender Bias Reduction", f"{metrics['current_biases']['gender']:.2f} → {metrics['baseline_biases']['gender_bias']:.2f}",
                 f"{metrics['bias_reduction']['gender']:.1f}%", delta_color="inverse")
    with col2:
        st.metric("Race Bias Reduction", f"{metrics['current_biases']['race']:.2f} → {metrics['baseline_biases']['race_bias']:.2f}",
                 f"{metrics['bias_reduction']['race']:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Socioeconomic Bias Reduction", f"{metrics['current_biases']['socioeconomic']:.2f} → {metrics['baseline_biases']['socioeconomic_bias']:.2f}",
                 f"{metrics['bias_reduction']['socioeconomic']:.1f}%", delta_color="inverse")
    
    # BiQ components
    st.subheader("BiQ Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Inclusivity", f"{metrics['inclusivity']:.2f}",
                 f"{metrics['weights']['inclusivity']:.1%} weight")
    with col2:
        st.metric("Diversity", f"{metrics['diversity']:.2f}",
                 f"{metrics['weights']['diversity']:.1%} weight")
    with col3:
        st.metric("Fairness", f"{metrics['fairness']:.2f}",
                 f"{metrics['weights']['fairness']:.1%} weight")
    
    # Retrieved documents with bias scores
    st.subheader("Retrieved Documents")
    for doc in metrics['selected_docs']:
        with st.container():
            st.markdown(f"**{doc['text']}**")
            cols = st.columns(3)
            with cols[0]:
                st.progress(doc["bias"]["gender"], text=f"Gender Bias: {doc['bias']['gender']:.2f}")
            with cols[1]:
                st.progress(doc["bias"]["race"], text=f"Race Bias: {doc['bias']['race']:.2f}")
            with cols[2]:
                st.progress(doc["bias"]["socioeconomic"], text=f"Socio Bias: {doc['bias']['socioeconomic']:.2f}")

# Main chat interface
if "message_log" not in st.session_state:
    st.session_state.message_log = [{
        "role": "ai", 
        "content": "Hi! I'm your Equitable AI assistant. I'll show exactly how much bias is reduced for each response."
    }]
    st.session_state.biq_history = []

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                with st.expander("📊 Detailed Bias Metrics", expanded=True):
                    display_bias_metrics(message["metrics"])

# User input
user_query = st.chat_input("Ask me anything...")

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Determine domain automatically
    domain = "general"
    doc = nlp(user_query.lower())
    if any(token.text in {'health', 'medical', 'doctor', 'hospital'} for token in doc):
        domain = "healthcare"
    elif any(token.text in {'finance', 'loan', 'bank', 'money', 'credit'} for token in doc):
        domain = "finance"
    
    # Calculate dynamic bias weights
    bias_weights = calculate_bias_weights(user_query)
    
    with st.spinner("🧠 Analyzing and reducing biases..."):
        # Retrieve and cluster relevant knowledge
        query_embedding = sentence_model.encode(user_query)
        doc_embeddings = sentence_model.encode([doc["text"] for doc in knowledge_base[domain]])
        
        # Cluster documents using GMM (from paper)
        gmm = GaussianMixture(n_components=3)
        cluster_labels = gmm.fit_predict(doc_embeddings)
        
        # Select diverse documents (one from each cluster)
        selected_docs = []
        selected_clusters = set()
        
        for i in np.argsort(-np.dot(doc_embeddings, query_embedding)):
            if cluster_labels[i] not in selected_clusters:
                selected_docs.append(knowledge_base[domain][i])
                selected_clusters.add(cluster_labels[i])
                if len(selected_clusters) == 3:
                    break
        
        # Calculate BiQ metrics and bias reduction
        biq_metrics = calculate_biq_metrics(selected_docs, bias_weights, domain)
        biq_metrics['weights'] = bias_weights
        biq_metrics['selected_docs'] = selected_docs
        st.session_state.biq_history.append(biq_metrics)
        
        # Generate response with context
        context = "\n".join(f"- {doc['text']}" for doc in selected_docs)
        
        system_prompt = f"""You are an Equitable AI assistant. Provide a response that:
        1. Answers the user's question accurately
        2. Addresses potential biases in the context
        3. Highlights diverse perspectives
        4. Uses inclusive language
        
        Context documents (with bias scores):
        {context}
        """
        
        human_prompt = user_query
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | llm_engine | StrOutputParser()
        ai_response = chain.invoke({})
    
    # Add AI response to log with metrics
    st.session_state.message_log.append({
        "role": "ai", 
        "content": ai_response,
        "metrics": biq_metrics
    })
    
    st.rerun()