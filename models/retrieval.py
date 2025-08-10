import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import random

class BiasAwareRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.demographic_groups = ['group_a', 'group_b', 'group_c']
        self.domain_datasets = {
            'healthcare': self._load_healthcare_data(),
            'finance': self._load_finance_data(),
            'education': self._load_education_data(),
            'general': self._load_general_data()
        }
    
    def retrieve(self, query: str, domain: str = 'general') -> List[Dict]:
        # Get relevant domain data
        dataset = self.domain_datasets.get(domain, self.domain_datasets['general'])
        
        # Encode query and documents
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode([doc['content'] for doc in dataset])
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Combine with diversity and fairness scores
        scored_docs = []
        for i, doc in enumerate(dataset):
            scored_doc = {
                **doc,
                'relevance': float(similarities[i]),
                'bias_score': self._calculate_bias_score(doc),
                'demographic_group': random.choice(self.demographic_groups),
                'protected_attribute': random.randint(0, 1)
            }
            scored_docs.append(scored_doc)
        
        # Sort by combined score (relevance - bias_score)
        scored_docs.sort(key=lambda x: x['relevance'] - 0.3 * x['bias_score'], reverse=True)
        
        return scored_docs[:10]  # Return top 10 documents
    
    def _calculate_bias_score(self, doc: Dict) -> float:
        # Simulate bias score calculation
        base_score = random.uniform(0.1, 0.5)  # Base bias level
        if 'gender' in doc.get('content', '').lower():
            base_score += 0.2
        if 'race' in doc.get('content', '').lower():
            base_score += 0.3
        return min(base_score, 1.0)
    
    def _load_healthcare_data(self) -> List[Dict]:
        return [
            {'title': 'Heart Disease in Men', 'content': 'Heart disease affects men differently than women...'},
            {'title': 'Women and Cardiovascular Health', 'content': 'Women often experience different symptoms of heart disease...'},
            {'title': 'Diabetes Management', 'content': 'Effective strategies for managing diabetes across populations...'},
            {'title': 'Mental Health in Urban Areas', 'content': 'Addressing mental health challenges in urban environments...'},
            {'title': 'Rural Healthcare Access', 'content': 'Barriers to healthcare access in rural communities...'},
            {'title': 'Cultural Competence in Healthcare', 'content': 'Importance of cultural understanding in medical practice...'},
            {'title': 'Health Disparities by Race', 'content': 'Examining racial disparities in healthcare outcomes...'},
            {'title': 'Gender Bias in Diagnosis', 'content': 'How gender bias affects medical diagnosis accuracy...'},
            {'title': 'Age-Related Health Concerns', 'content': 'Health issues that disproportionately affect older adults...'},
            {'title': 'Socioeconomic Factors in Health', 'content': 'How income and education impact health outcomes...'}
        ]
    
    def _load_finance_data(self) -> List[Dict]:
        return [
            {'title': 'Personal Finance Basics', 'content': 'Fundamental principles for managing personal finances...'},
            {'title': 'Investing for Beginners', 'content': 'Introduction to investment strategies for new investors...'},
            {'title': 'Credit Access in Underserved Communities', 'content': 'Challenges in accessing credit for minority groups...'},
            {'title': 'Gender Pay Gap and Retirement', 'content': 'How the gender pay gap affects retirement savings...'},
            {'title': 'Microfinance in Developing Countries', 'content': 'The role of microfinance in economic development...'},
            {'title': 'Racial Disparities in Loan Approval', 'content': 'Examining racial bias in loan approval processes...'},
            {'title': 'Financial Literacy Education', 'content': 'The importance of financial education in schools...'},
            {'title': 'Wealth Inequality Trends', 'content': 'Recent trends in wealth distribution across demographics...'},
            {'title': 'Fintech and Financial Inclusion', 'content': 'How technology is expanding access to financial services...'},
            {'title': 'Behavioral Economics Insights', 'content': 'Psychological factors in financial decision making...'}
        ]
    
    def _load_education_data(self) -> List[Dict]:
        return [
            {'title': 'Inclusive Teaching Strategies', 'content': 'Methods for creating inclusive classroom environments...'},
            {'title': 'Achievement Gap Analysis', 'content': 'Factors contributing to educational achievement gaps...'},
            {'title': 'Technology in Education', 'content': 'How technology is transforming learning experiences...'},
            {'title': 'Cultural Bias in Standardized Testing', 'content': 'Examining cultural assumptions in educational assessments...'},
            {'title': 'Special Education Resources', 'content': 'Supporting students with diverse learning needs...'},
            {'title': 'Gender Stereotypes in STEM', 'content': 'Addressing gender bias in science and math education...'},
            {'title': 'Early Childhood Education', 'content': 'The importance of early learning experiences...'},
            {'title': 'Multilingual Education Approaches', 'content': 'Strategies for teaching in linguistically diverse classrooms...'},
            {'title': 'Socioeconomic Factors in Education', 'content': 'How family income affects educational outcomes...'},
            {'title': 'Digital Divide in Education', 'content': 'Addressing unequal access to technology for learning...'}
        ]
    
    def _load_general_data(self) -> List[Dict]:
        return [
            {'title': 'Current Events Overview', 'content': 'Summary of recent global news developments...'},
            {'title': 'Technology Trends', 'content': 'Emerging technologies shaping the future...'},
            {'title': 'Environmental Sustainability', 'content': 'Strategies for addressing climate change...'},
            {'title': 'Social Justice Movements', 'content': 'Recent developments in social justice advocacy...'},
            {'title': 'Economic Policy Analysis', 'content': 'Evaluating different approaches to economic policy...'},
            {'title': 'Cultural Exchange Programs', 'content': 'Benefits of cross-cultural exchange initiatives...'},
            {'title': 'Urban Development Challenges', 'content': 'Addressing growth and infrastructure in cities...'},
            {'title': 'Media Representation Analysis', 'content': 'Examining diversity in media portrayals...'},
            {'title': 'Workplace Diversity Initiatives', 'content': 'Strategies for promoting inclusion in organizations...'},
            {'title': 'Global Migration Patterns', 'content': 'Trends and impacts of international migration...'}
        ]