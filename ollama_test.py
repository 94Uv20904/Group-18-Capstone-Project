import streamlit as st
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
import json
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# Download NLTK data (run once) 
#/Users/urviray/Library/Python/3.9/bin/streamlit run ollama_test.py
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading required language models...")
    nltk.download('punkt')
    nltk.download('stopwords')

class UserContributionManager:
    """Manages user-contributed facts with verification options"""
    
    def __init__(self, contributions_file="user_contributions.csv"):
        self.contributions_file = contributions_file
        self.pending_contributions = self.load_pending_contributions()
        
    def load_pending_contributions(self):
        """Load pending contributions from file"""
        try:
            if os.path.exists(self.contributions_file):
                return pd.read_csv(self.contributions_file)
            else:
                # Create empty dataframe with required columns
                return pd.DataFrame(columns=[
                    'ID', 'Statement', 'IsTrue', 'Category', 'Difficulty', 
                    'Contributor', 'SubmissionDate', 'Status', 'AIVerification',
                    'VerificationConfidence', 'VerificationNotes'
                ])
        except Exception as e:
            st.error(f"Error loading contributions: {str(e)}")
            return pd.DataFrame(columns=[
                'ID', 'Statement', 'IsTrue', 'Category', 'Difficulty', 
                'Contributor', 'SubmissionDate', 'Status', 'AIVerification',
                'VerificationConfidence', 'VerificationNotes'
            ])
    
    def save_contributions(self):
        """Save contributions to file"""
        try:
            self.pending_contributions.to_csv(self.contributions_file, index=False)
        except Exception as e:
            st.error(f"Error saving contributions: {str(e)}")
    
    def add_contribution(self, statement, is_true, category, difficulty, 
                        contributor_name, verify_with_ai=True, fact_checker=None):
        """Add a new contribution"""
        new_id = len(self.pending_contributions) + 1000  # Start IDs from 1000
        
        # AI verification if requested
        ai_verification = None
        verification_confidence = 0.0
        verification_notes = ""
        
        if verify_with_ai and fact_checker:
            try:
                verification_result = fact_checker.analyze_prompt(statement, use_ollama=True)
                
                if verification_result.get('ollama_result'):
                    ollama_data = verification_result['ollama_result']
                    ai_verification = ollama_data.get('is_correct', None)
                    verification_confidence = ollama_data.get('confidence', 0.0)
                    verification_notes = ollama_data.get('explanation', '')
                
                # Check for conflicts between user label and AI assessment
                if ai_verification is not None:
                    user_says_true = (is_true == 1)
                    ai_says_true = ai_verification
                    
                    if user_says_true != ai_says_true:
                        verification_notes += f" [CONFLICT: User says {'True' if user_says_true else 'False'}, AI says {'True' if ai_says_true else 'False'}]"
                        
            except Exception as e:
                verification_notes = f"AI verification failed: {str(e)}"
        
        new_contribution = {
            'ID': new_id,
            'Statement': statement,
            'IsTrue': is_true,
            'Category': category,
            'Difficulty': difficulty,
            'Contributor': contributor_name,
            'SubmissionDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Status': 'Pending',
            'AIVerification': ai_verification,
            'VerificationConfidence': verification_confidence,
            'VerificationNotes': verification_notes
        }
        
        # Add to dataframe
        new_row = pd.DataFrame([new_contribution])
        self.pending_contributions = pd.concat([self.pending_contributions, new_row], ignore_index=True)
        
        # Save to file
        self.save_contributions()
        
        return new_contribution
    
    def approve_contribution(self, contribution_id):
        """Approve a contribution for inclusion in main dataset"""
        mask = self.pending_contributions['ID'] == contribution_id
        self.pending_contributions.loc[mask, 'Status'] = 'Approved'
        self.save_contributions()
    
    def reject_contribution(self, contribution_id, reason=""):
        """Reject a contribution"""
        mask = self.pending_contributions['ID'] == contribution_id
        self.pending_contributions.loc[mask, 'Status'] = 'Rejected'
        if reason:
            current_notes = self.pending_contributions.loc[mask, 'VerificationNotes'].iloc[0]
            self.pending_contributions.loc[mask, 'VerificationNotes'] = f"{current_notes} [REJECTED: {reason}]"
        self.save_contributions()
    
    def get_approved_contributions(self):
        """Get all approved contributions for adding to main dataset"""
        return self.pending_contributions[self.pending_contributions['Status'] == 'Approved'].copy()
    
    def get_pending_contributions(self):
        """Get all pending contributions for review"""
        return self.pending_contributions[self.pending_contributions['Status'] == 'Pending'].copy()

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434", model="llama3.2"):
        self.base_url = base_url
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def generate_response(self, prompt, max_tokens=500, temperature=0.1):
        """Generate response from Ollama model"""
        if not self.available:
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            return None
        except Exception as e:
            st.error(f"Ollama API error: {str(e)}")
            return None

class AdvancedPhysicsFactChecker:
    def __init__(self, dataset, ollama_client=None):
        self.dataset = dataset
        self.ollama_client = ollama_client
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Physics-specific knowledge base
        self.physics_concepts = {
            'temperature': ['celsius', 'fahrenheit', 'kelvin', 'degrees', 'hot', 'cold', 'boil', 'freeze', 'melt'],
            'energy': ['joule', 'calorie', 'kinetic', 'potential', 'thermal', 'nuclear', 'electromagnetic'],
            'waves': ['frequency', 'wavelength', 'amplitude', 'light', 'sound', 'electromagnetic', 'radio'],
            'mechanics': ['force', 'mass', 'weight', 'acceleration', 'velocity', 'momentum', 'gravity'],
            'electricity': ['current', 'voltage', 'resistance', 'charge', 'electron', 'proton', 'field'],
            'quantum': ['photon', 'quantum', 'particle', 'wave', 'uncertainty', 'probability', 'orbital'],
            'thermodynamics': ['entropy', 'heat', 'temperature', 'pressure', 'volume', 'gas', 'liquid', 'solid']
        }
        
        # Common physics misconceptions patterns
        self.misconception_patterns = [
            (r'water boils at (\d+)°?[cf]', self._check_boiling_point),
            (r'sound.*faster.*light', self._check_speed_comparison),
            (r'heavier.*fall.*faster', self._check_falling_objects),
            (r'all.*orbit.*perfect.*circle', self._check_orbital_shape),
            (r'mass.*weight.*same|interchangeable', self._check_mass_weight),
            (r'all.*radiation.*man.?made', self._check_radiation_source),
            (r'electrons.*orbit.*like.*planets', self._check_atomic_model)
        ]
        
        # Precompute TF-IDF vectors for the dataset
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)
        self.dataset_vectors = self._compute_tfidf_vectors()
    
    def _compute_tfidf_vectors(self):
        """Precompute TF-IDF vectors for all statements in dataset"""
        statements = self.dataset['Statement'].tolist()
        return self.vectorizer.fit_transform(statements)
    
    def _preprocess_text(self, text):
        """Advanced text preprocessing for physics content"""
        text = text.lower()
        text = re.sub(r'°c|celsius', 'celsius', text)
        text = re.sub(r'°f|fahrenheit', 'fahrenheit', text)
        text = re.sub(r'°k|kelvin', 'kelvin', text)
        text = re.sub(r'm/s²|m/s2', 'acceleration_unit', text)
        text = re.sub(r'e\s*=\s*mc²|e\s*=\s*mc2', 'mass_energy_equivalence', text)
        
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _semantic_similarity(self, prompt):
        """Calculate semantic similarity using TF-IDF and cosine similarity"""
        prompt_preprocessed = self._preprocess_text(prompt)
        prompt_vector = self.vectorizer.transform([prompt_preprocessed])
        similarities = cosine_similarity(prompt_vector, self.dataset_vectors).flatten()
        return similarities
    
    def _physics_concept_matching(self, prompt):
        """Match physics concepts and calculate domain-specific similarity"""
        prompt_lower = prompt.lower()
        concept_scores = {}
        
        for concept, keywords in self.physics_concepts.items():
            matches = sum(1 for keyword in keywords if keyword in prompt_lower)
            if matches > 0:
                concept_scores[concept] = matches / len(keywords)
        
        return concept_scores
    
    def _check_misconception_patterns(self, prompt):
        """Check for common physics misconceptions using regex patterns"""
        misconceptions = []
        
        for pattern, check_func in self.misconception_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                result = check_func(match, prompt)
                if result:
                    misconceptions.append(result)
        
        return misconceptions
    
    def _ollama_fact_check(self, prompt):
        """Use Ollama to perform AI-powered fact checking"""
        if not self.ollama_client or not self.ollama_client.available:
            return None
        
        fact_check_prompt = f"""
You are a physics expert tasked with fact-checking the following statement. Analyze it carefully and provide your assessment.

Statement to check: "{prompt}"

Please analyze this statement and respond with a JSON object containing:
1. "is_correct": true/false
2. "confidence": a number between 0.0 and 1.0
3. "explanation": a brief explanation of why the statement is correct or incorrect
4. "physics_domain": the main physics domain (e.g., "thermodynamics", "mechanics", "quantum physics")
5. "corrections": if incorrect, provide the correct information

Respond only with valid JSON, no additional text.
"""
        
        response = self.ollama_client.generate_response(fact_check_prompt, max_tokens=300, temperature=0.1)
        
        if response:
            try:
                # Clean the response to extract JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                st.warning("Could not parse Ollama response as JSON")
                return {"raw_response": response}
        
        return None
    
    def _ollama_explain_concepts(self, prompt):
        """Use Ollama to explain physics concepts in the statement"""
        if not self.ollama_client or not self.ollama_client.available:
            return None
        
        explanation_prompt = f"""
You are a physics teacher. Explain the physics concepts mentioned in this statement in simple terms:

Statement: "{prompt}"

Provide a brief, educational explanation of the key physics concepts involved. Keep it concise but informative.
"""
        
        return self.ollama_client.generate_response(explanation_prompt, max_tokens=200, temperature=0.3)
    
    def _check_boiling_point(self, match, prompt):
        temp = int(match.group(1))
        if temp != 100 and 'celsius' in prompt.lower():
            return {'type': 'temperature_misconception', 'description': f'Water boils at 100°C at standard pressure, not {temp}°C', 'severity': 'high'}
        elif temp != 212 and 'fahrenheit' in prompt.lower():
            return {'type': 'temperature_misconception', 'description': f'Water boils at 212°F at standard pressure, not {temp}°F', 'severity': 'high'}
        return None
    
    def _check_speed_comparison(self, match, prompt):
        if 'sound' in prompt.lower() and 'faster' in prompt.lower() and 'light' in prompt.lower():
            return {'type': 'speed_misconception', 'description': 'Light travels much faster than sound (~300,000 km/s vs ~343 m/s)', 'severity': 'high'}
        return None
    
    def _check_falling_objects(self, match, prompt):
        return {'type': 'gravity_misconception', 'description': 'In a vacuum, all objects fall at the same rate regardless of mass (Galileo\'s principle)', 'severity': 'medium'}
    
    def _check_orbital_shape(self, match, prompt):
        return {'type': 'astronomy_misconception', 'description': 'Planetary orbits are elliptical, not perfectly circular (Kepler\'s laws)', 'severity': 'medium'}
    
    def _check_mass_weight(self, match, prompt):
        return {'type': 'mechanics_misconception', 'description': 'Mass is the amount of matter; weight is the gravitational force on that mass', 'severity': 'medium'}
    
    def _check_radiation_source(self, match, prompt):
        return {'type': 'physics_misconception', 'description': 'Natural radiation exists (cosmic rays, radioactive elements, solar radiation)', 'severity': 'medium'}
    
    def _check_atomic_model(self, match, prompt):
        return {'type': 'quantum_misconception', 'description': 'Electrons exist in probability clouds (orbitals), not defined orbital paths', 'severity': 'medium'}
    
    def _calculate_confidence(self, similarities, concept_matches, misconceptions, ollama_result=None):
        """Calculate overall confidence score based on multiple factors"""
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        base_confidence = max_similarity
        concept_boost = sum(concept_matches.values()) * 0.1 if concept_matches else 0
        misconception_boost = len(misconceptions) * 0.3
        
        # Incorporate Ollama confidence if available
        ollama_boost = 0
        if ollama_result and 'confidence' in ollama_result:
            ollama_boost = ollama_result['confidence'] * 0.2
        
        total_confidence = min(0.95, base_confidence + concept_boost + misconception_boost + ollama_boost)
        return total_confidence
    
    def analyze_prompt(self, prompt, similarity_threshold=0.3, use_ollama=True):
        """Comprehensive analysis of physics prompt with Ollama integration"""
        analysis_start = datetime.now()
        
        semantic_similarities = self._semantic_similarity(prompt)
        concept_matches = self._physics_concept_matching(prompt)
        misconceptions = self._check_misconception_patterns(prompt)
        
        # Ollama analysis
        ollama_result = None
        ollama_explanation = None
        if use_ollama and self.ollama_client:
            ollama_result = self._ollama_fact_check(prompt)
            ollama_explanation = self._ollama_explain_concepts(prompt)
        
        best_match_indices = np.argsort(semantic_similarities)[::-1][:5]
        similar_statements = []
        
        for idx in best_match_indices:
            if semantic_similarities[idx] >= similarity_threshold:
                row = self.dataset.iloc[idx]
                similar_statements.append({
                    'id': row['ID'],
                    'statement': row['Statement'],
                    'is_true': row['IsTrue'],
                    'difficulty': row.get('Difficulty', 'Unknown'),
                    'category': row.get('Category', 'Unknown'),
                    'similarity_score': float(semantic_similarities[idx]),
                    'match_type': 'semantic'
                })
        
        confidence = self._calculate_confidence(semantic_similarities, concept_matches, misconceptions, ollama_result)
        
        is_flagged = False
        primary_reason = "No issues detected"
        
        # Check Ollama results first
        if ollama_result and 'is_correct' in ollama_result and not ollama_result['is_correct']:
            is_flagged = True
            primary_reason = f"AI analysis suggests statement is incorrect"
        elif misconceptions:
            is_flagged = True
            primary_reason = f"Detected {len(misconceptions)} physics misconception(s)"
        elif similar_statements and similar_statements[0]['is_true'] == 0:
            is_flagged = True
            primary_reason = "Similar to known incorrect statement"
        elif confidence < 0.2:
            primary_reason = "Insufficient information to verify"
        
        analysis_time = (datetime.now() - analysis_start).total_seconds()
        
        return {
            'flagged': is_flagged,
            'confidence': confidence,
            'primary_reason': primary_reason,
            'misconceptions': misconceptions,
            'physics_concepts': concept_matches,
            'similar_statements': similar_statements,
            'ollama_result': ollama_result,
            'ollama_explanation': ollama_explanation,
            'analysis_time': analysis_time
        }

def load_dataset():
    """Load dataset with hardcoded path for demo CSV"""
    csv_path = "./demo_dataset.csv"
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='cp1252')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.dropna(subset=['Statement'])
            
            if 'IsTrue' in df.columns:
                df['IsTrue'] = df['IsTrue'].astype(int)
            
            if 'Category' in df.columns:
                df['Category'] = df['Category'].fillna('Physics')
            if 'Difficulty' in df.columns:
                df['Difficulty'] = df['Difficulty'].fillna('Medium')
            
            df = df.head(20000)
            return df, True
            
        else:
            st.error(f"Dataset file not found at: {csv_path}")
            st.info("Please place your demo_dataset.csv file in the same directory as this script.")
            return create_fallback_data(), False
            
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_fallback_data(), False

def create_fallback_data():
    """Create fallback sample data"""
    data = [
        {"ID": 1, "Statement": "Water boils at 100°C at standard atmospheric pressure.", "IsTrue": 1, "Difficulty": "Easy", "Category": "Thermodynamics"},
        {"ID": 2, "Statement": "Water boils at 50°C at sea level.", "IsTrue": 0, "Difficulty": "Easy", "Category": "Thermodynamics"},
        {"ID": 3, "Statement": "Sound travels faster than light.", "IsTrue": 0, "Difficulty": "Easy", "Category": "Physics"},
        {"ID": 4, "Statement": "E = mc² relates mass and energy.", "IsTrue": 1, "Difficulty": "Easy", "Category": "Modern Physics"},
        {"ID": 5, "Statement": "Heavier objects always fall faster than lighter objects.", "IsTrue": 0, "Difficulty": "Easy", "Category": "Mechanics"},
    ]
    return pd.DataFrame(data)

def create_user_contribution_interface(contribution_manager, fact_checker):
    """Create interface for user contributions"""
    st.header("Contribute New Physics Facts")
    
    st.write("Help expand our physics knowledge base by contributing new facts!")
    
    with st.form("contribution_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            contributor_name = st.text_input("Your Name/Username", placeholder="Enter your name")
            
            statement = st.text_area(
                "Physics Statement", 
                height=100,
                placeholder="Enter a physics fact or statement to be verified..."
            )
            
            is_true = st.selectbox(
                "Is this statement correct?",
                options=[1, 0],
                format_func=lambda x: "True" if x == 1 else "False"
            )
        
        with col2:
            category = st.selectbox(
                "Physics Category",
                ["Thermodynamics", "Mechanics", "Waves", "Electricity", "Quantum Physics", 
                 "Modern Physics", "Astronomy", "Optics", "Nuclear Physics", "Other"]
            )
            
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard"]
            )
            
            verify_with_ai = st.checkbox(
                "Verify with AI before submission", 
                value=True,
                help="Have the AI check your fact before adding it to the database"
            )
        
        submit_button = st.form_submit_button("Submit Contribution", type="primary")
        
        if submit_button:
            if not contributor_name.strip():
                st.error("Please enter your name")
            elif not statement.strip():
                st.error("Please enter a physics statement")
            else:
                with st.spinner("Processing your contribution..."):
                    contribution = contribution_manager.add_contribution(
                        statement=statement.strip(),
                        is_true=is_true,
                        category=category,
                        difficulty=difficulty,
                        contributor_name=contributor_name.strip(),
                        verify_with_ai=verify_with_ai,
                        fact_checker=fact_checker if verify_with_ai else None
                    )
                    
                st.success("Contribution submitted successfully!")
                
                # Show verification results if AI was used
                if verify_with_ai and contribution.get('AIVerification') is not None:
                    st.subheader("AI Verification Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        ai_verdict = "Correct" if contribution['AIVerification'] else "Incorrect"
                        user_verdict = "Correct" if is_true == 1 else "Incorrect"
                        st.metric("Your Assessment", user_verdict)
                        st.metric("AI Assessment", ai_verdict)
                    
                    with col2:
                        st.metric("AI Confidence", f"{contribution['VerificationConfidence']:.1%}")
                    
                    if contribution['VerificationNotes']:
                        st.write("**AI Explanation:**")
                        st.write(contribution['VerificationNotes'])
                    
                    # Highlight conflicts
                    if "[CONFLICT:" in contribution['VerificationNotes']:
                        st.warning("There's a disagreement between your assessment and the AI's assessment. This contribution will be flagged for manual review.")

def create_contribution_review_interface(contribution_manager):
    """Create interface for reviewing contributions"""
    st.header("Review Pending Contributions")
    
    pending = contribution_manager.get_pending_contributions()
    
    if len(pending) == 0:
        st.info("No pending contributions to review.")
        return
    
    st.write(f"**{len(pending)} contributions awaiting review**")
    
    for idx, contribution in pending.iterrows():
        with st.expander(f"ID {contribution['ID']}: {contribution['Statement'][:60]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Statement:** {contribution['Statement']}")
                st.write(f"**Contributor:** {contribution['Contributor']}")
                st.write(f"**Submitted:** {contribution['SubmissionDate']}")
                st.write(f"**User Says:** {'True' if contribution['IsTrue'] == 1 else 'False'}")
                st.write(f"**Category:** {contribution['Category']}")
                st.write(f"**Difficulty:** {contribution['Difficulty']}")
            
            with col2:
                if pd.notna(contribution['AIVerification']):
                    ai_verdict = "True" if contribution['AIVerification'] else "False"
                    st.write(f"**AI Assessment:** {ai_verdict}")
                    st.write(f"**AI Confidence:** {contribution['VerificationConfidence']:.1%}")
                    
                    if contribution['VerificationNotes']:
                        st.write("**AI Notes:**")
                        st.write(contribution['VerificationNotes'])
                else:
                    st.write("**AI Assessment:** Not performed")
            
            # Action buttons
            col_approve, col_reject = st.columns(2)
            
            with col_approve:
                if st.button(f"Approve {contribution['ID']}", key=f"approve_{contribution['ID']}"):
                    contribution_manager.approve_contribution(contribution['ID'])
                    st.success("Contribution approved!")
                    st.rerun()
            
            with col_reject:
                if st.button(f"Reject {contribution['ID']}", key=f"reject_{contribution['ID']}"):
                    reason = st.text_input(f"Rejection reason for {contribution['ID']}", key=f"reason_{contribution['ID']}")
                    contribution_manager.reject_contribution(contribution['ID'], reason)
                    st.warning("Contribution rejected!")
                    st.rerun()

def create_contribution_stats(contribution_manager):
    """Create statistics dashboard for contributions"""
    st.subheader("Contribution Statistics")
    
    all_contributions = contribution_manager.pending_contributions
    
    if len(all_contributions) == 0:
        st.info("No contributions yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Contributions", len(all_contributions))
    
    with col2:
        pending_count = len(all_contributions[all_contributions['Status'] == 'Pending'])
        st.metric("Pending Review", pending_count)
    
    with col3:
        approved_count = len(all_contributions[all_contributions['Status'] == 'Approved'])
        st.metric("Approved", approved_count)
    
    with col4:
        rejected_count = len(all_contributions[all_contributions['Status'] == 'Rejected'])
        st.metric("Rejected", rejected_count)
    
    # Charts
    if len(all_contributions) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = all_contributions['Category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Contributions by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Status distribution
            status_counts = all_contributions['Status'].value_counts()
            fig = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title="Contribution Status"
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Physics Fact Checker - With User Contributions",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #2E86AB; margin-bottom: 10px;'>Physics Fact Checker</h1>
        <h3 style='color: #A23B72; margin-top: 0;'>Ollama Integration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    ollama_client = OllamaClient()
    dataset, dataset_loaded = load_dataset()
    contribution_manager = UserContributionManager()
    
    if dataset_loaded:
        st.success(f"Successfully loaded {len(dataset):,} physics statements from dataset")
    else:
        st.warning("Using fallback sample data - please add demo_dataset.csv to access full dataset")
    
    # Initialize fact checker
    checker = AdvancedPhysicsFactChecker(dataset, ollama_client)
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Configuration")
        
        # Ollama status
        if ollama_client.available:
            st.success("✅ Ollama Connected")
            
            # Model selection
            available_models = ollama_client.get_available_models()
            if available_models:
                selected_model = st.selectbox(
                    "Select Ollama Model",
                    available_models,
                    index=0 if available_models else None
                )
                ollama_client.model = selected_model
            else:
                st.warning("No models found. Please pull a model first.")
                st.code("ollama pull llama3.2")
        else:
            st.error("❌ Ollama Not Available")
            st.info("Make sure Ollama is running on localhost:11434")
            st.code("ollama serve")
        
        st.markdown("---")
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Single Analysis", "Contribute Facts", "Review Contributions", 
             "Batch Analysis", "Dataset Explorer", "AI Settings"]
        )
        
        st.markdown("---")
        st.subheader("Quick Stats")
        st.info(f"Dataset: {len(dataset):,} statements")
        
        # Show contribution stats in sidebar
        pending_count = len(contribution_manager.get_pending_contributions())
        if pending_count > 0:
            st.warning(f"Pending: {pending_count} contributions")
        
        approved_count = len(contribution_manager.get_approved_contributions())
        if approved_count > 0:
            st.info(f"Approved: {approved_count} contributions")
    
    # Main content based on selected page
    if page == "Dashboard":
        st.header("System Overview")
        
        # AI Status
        col1, col2, col3 = st.columns(3)
        with col1:
            if ollama_client.available:
                st.success("🤖 AI Enhancement: Active")
            else:
                st.warning("🤖 AI Enhancement: Disabled")
        
        with col2:
            st.info(f"📊 Dataset: {len(dataset):,} statements")
        
        with col3:
            if ollama_client.available:
                st.info(f"🔧 Model: {ollama_client.model}")
            else:
                st.info("🔧 Model: None")
        
        # Contribution statistics
        create_contribution_stats(contribution_manager)
        
        # Quick contribution form
        st.subheader("Quick Fact Contribution")
        quick_fact = st.text_input("Add a quick physics fact:", placeholder="e.g., Light travels at 299,792,458 m/s in vacuum")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            quick_true = st.selectbox("Correct?", [1, 0], format_func=lambda x: "True" if x == 1 else "False", key="quick_true")
        with col2:
            quick_category = st.selectbox("Category", ["Physics", "Thermodynamics", "Mechanics", "Other"], key="quick_cat")
        with col3:
            if st.button("Add Fact"):
                if quick_fact.strip():
                    contribution = contribution_manager.add_contribution(
                        statement=quick_fact.strip(),
                        is_true=quick_true,
                        category=quick_category,
                        difficulty="Medium",
                        contributor_name="Dashboard User",
                        verify_with_ai=ollama_client.available,
                        fact_checker=checker if ollama_client.available else None
                    )
                    st.success("Fact added successfully!")
                    st.rerun()
    
    elif page == "Single Analysis":
        st.header("AI-Enhanced Statement Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter physics statement for analysis:",
                height=120,
                placeholder="e.g., Water boils at 50°C at sea level"
            )
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                similarity_threshold = st.slider("Similarity Threshold", 0.1, 0.8, 0.3, 0.05)
            with col_b:
                use_ollama = st.checkbox("Use AI Analysis", value=ollama_client.available, disabled=not ollama_client.available)
            with col_c:
                analyze_btn = st.button("Analyze Statement", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Analysis Settings")
            st.write("**Similarity Threshold:** Controls dataset matching strictness")
            st.write("**AI Analysis:** Uses Ollama for deeper fact-checking")
            if ollama_client.available:
                st.success(f"AI Model: {ollama_client.model}")
            else:
                st.warning("AI analysis unavailable")
        
        if analyze_btn and user_input.strip():
            with st.spinner("Analyzing statement with AI..."):
                result = checker.analyze_prompt(user_input, similarity_threshold, use_ollama)
                
                # Results display
                if result['flagged']:
                    st.error(f"🚨 Statement Flagged: {result['primary_reason']}")
                else:
                    st.success("✅ Statement Approved: No issues detected")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence Score", f"{result['confidence']:.1%}")
                with col2:
                    st.metric("Processing Time", f"{result['analysis_time']*1000:.1f}ms")
                with col3:
                    st.metric("Misconceptions Found", len(result['misconceptions']))
                with col4:
                    ai_status = "✅ Used" if result['ollama_result'] else "❌ Not Used"
                    st.metric("AI Analysis", ai_status)
                
                # AI Results
                if result['ollama_result']:
                    st.subheader("🤖 AI Analysis Results")
                    ollama_data = result['ollama_result']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'is_correct' in ollama_data:
                            status = "✅ Correct" if ollama_data['is_correct'] else "❌ Incorrect"
                            st.metric("AI Verdict", status)
                        
                        if 'confidence' in ollama_data:
                            st.metric("AI Confidence", f"{ollama_data['confidence']:.1%}")
                    
                    with col2:
                        if 'physics_domain' in ollama_data:
                            st.metric("Physics Domain", ollama_data['physics_domain'])
                    
                    if 'explanation' in ollama_data:
                        st.write("**AI Explanation:**")
                        st.write(ollama_data['explanation'])
                    
                    if 'corrections' in ollama_data and ollama_data['corrections']:
                        st.write("**Suggested Corrections:**")
                        st.write(ollama_data['corrections'])
                
                # Option to add statement to dataset
                st.subheader("Add to Dataset")
                st.write("Want to contribute this statement to our knowledge base?")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    contributor_name = st.text_input("Your name:", key="analysis_contributor")
                with col2:
                    statement_correct = st.selectbox("Is statement correct?", [1, 0], format_func=lambda x: "True" if x == 1 else "False", key="analysis_correct")
                with col3:
                    if st.button("Add to Dataset"):
                        if contributor_name.strip():
                            contribution = contribution_manager.add_contribution(
                                statement=user_input.strip(),
                                is_true=statement_correct,
                                category="Physics",
                                difficulty="Medium",
                                contributor_name=contributor_name.strip(),
                                verify_with_ai=False,  # Already analyzed
                                fact_checker=None
                            )
                            st.success("Statement added to contribution queue!")
                        else:
                            st.error("Please enter your name")
    
    elif page == "Contribute Facts":
        create_user_contribution_interface(contribution_manager, checker)
    
    elif page == "Review Contributions":
        create_contribution_review_interface(contribution_manager)
    
    elif page == "Batch Analysis":
        st.header("Batch Statement Analysis with AI")
        
        batch_input = st.text_area(
            "Enter multiple statements (one per line):",
            height=200,
            placeholder="Water boils at 50°C\nSound travels faster than light\nE = mc²"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            use_ai_batch = st.checkbox("Use AI Analysis for Batch", value=ollama_client.available and False, disabled=not ollama_client.available)
        with col2:
            auto_contribute = st.checkbox("Auto-contribute analyzed statements", value=False)
        
        if st.button("Analyze Batch", type="primary"):
            if batch_input.strip():
                statements = [stmt.strip() for stmt in batch_input.split('\n') if stmt.strip()]
                
                results_summary = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, statement in enumerate(statements):
                    status_text.text(f"Analyzing statement {i+1} of {len(statements)}")
                    result = checker.analyze_prompt(statement, 0.3, use_ai_batch)
                    
                    ai_verdict = "N/A"
                    if result['ollama_result'] and 'is_correct' in result['ollama_result']:
                        ai_verdict = "Correct" if result['ollama_result']['is_correct'] else "Incorrect"
                    
                    results_summary.append({
                        'Statement': statement[:60] + "..." if len(statement) > 60 else statement,
                        'Status': 'FLAGGED' if result['flagged'] else 'APPROVED',
                        'Confidence': f"{result['confidence']:.1%}",
                        'AI_Verdict': ai_verdict,
                        'Issues': len(result['misconceptions']),
                        'Processing_Time_ms': f"{result['analysis_time']*1000:.1f}"
                    })
                    
                    # Auto-contribute if enabled
                    if auto_contribute and ai_verdict != "N/A":
                        is_correct = 1 if ai_verdict == "Correct" else 0
                        contribution_manager.add_contribution(
                            statement=statement,
                            is_true=is_correct,
                            category="Physics",
                            difficulty="Medium",
                            contributor_name="Batch Analysis",
                            verify_with_ai=False,
                            fact_checker=None
                        )
                    
                    progress_bar.progress((i + 1) / len(statements))
                
                status_text.text("Analysis complete!")
                
                # Results
                st.subheader("Batch Analysis Results")
                results_df = pd.DataFrame(results_summary)
                st.dataframe(results_df, use_container_width=True)
                
                if auto_contribute:
                    st.success(f"Auto-contributed {len(statements)} statements to the dataset!")
    
    elif page == "Dataset Explorer":
        st.header("Dataset Explorer")
        
        # Include approved contributions in the dataset view
        approved_contributions = contribution_manager.get_approved_contributions()
        
        # Combine main dataset with approved contributions
        combined_dataset = dataset.copy()
        
        if len(approved_contributions) > 0:
            # Ensure approved contributions have the same columns as main dataset
            approved_for_display = approved_contributions[['ID', 'Statement', 'IsTrue', 'Category', 'Difficulty']].copy()
            combined_dataset = pd.concat([combined_dataset, approved_for_display], ignore_index=True)
            
            st.info(f"Showing {len(dataset):,} original statements + {len(approved_contributions):,} approved contributions = {len(combined_dataset):,} total")
        else:
            st.info(f"Showing {len(dataset):,} statements from original dataset")
        
        # Dataset overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Total Records:** {len(combined_dataset):,}")
            st.write(f"**Original Dataset:** {len(dataset):,}")
            st.write(f"**User Contributions:** {len(approved_contributions):,}")
            
            if 'Category' in combined_dataset.columns:
                st.write(f"**Categories:** {combined_dataset['Category'].nunique()}")
        
        with col2:
            st.subheader("Data Quality")
            missing_data = combined_dataset.isnull().sum()
            if missing_data.sum() > 0:
                st.write("**Missing Values:**")
                for col, count in missing_data[missing_data > 0].items():
                    st.write(f"- {col}: {count}")
            else:
                st.success("No missing values detected")
        
        # Display combined dataset
        st.dataframe(combined_dataset.head(100), use_container_width=True)
    
    elif page == "AI Settings":
        st.header("🤖 Ollama Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Connection Status")
            if ollama_client.available:
                st.success("✅ Ollama is connected and ready")
                st.write(f"**Base URL:** {ollama_client.base_url}")
                st.write(f"**Current Model:** {ollama_client.model}")
            else:
                st.error("❌ Cannot connect to Ollama")
                st.write("**Troubleshooting Steps:**")
                st.write("1. Make sure Ollama is installed")
                st.write("2. Start Ollama service: `ollama serve`")
                st.write("3. Check if running on localhost:11434")
        
        with col2:
            st.subheader("Available Models")
            if ollama_client.available:
                models = ollama_client.get_available_models()
                if models:
                    for model in models:
                        st.write(f"• {model}")
                else:
                    st.warning("No models found")
                    st.write("**Install a model:**")
                    st.code("ollama pull llama3.2")
            else:
                st.write("Connect to Ollama to see available models")

if __name__ == "__main__":
    main()