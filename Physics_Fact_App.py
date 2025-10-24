from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
import json
from datetime import datetime, timedelta
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from typing import Dict, List, Optional, Tuple
import hashlib
import zipfile


from io import BytesIO
#streamlit run Physiscs_Fact_App.py
# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

class AppConfig:
    """Application configuration constants"""
    APP_TITLE = "Physics Fact Checker"
    APP_SUBTITLE = "Ollama Powered Verification System"
    VERSION = "2.0.0"
    
    # Paths
    DATASET_PATH = "demo_dataset-2.csv"
    CONTRIBUTIONS_FILE = "user_contributions.csv"
    AUDIT_LOG_FILE = "audit_log.csv"
    
    # Ollama
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL = "llama3.2"
    
    # Analysis
    DEFAULT_SIMILARITY_THRESHOLD = 0.3
    MAX_SIMILAR_STATEMENTS = 10
    
    # UI
    PRIMARY_COLOR = "#2E86AB"
    SUCCESS_COLOR = "#28a745"
    WARNING_COLOR = "#ffc107"
    DANGER_COLOR = "#dc3545"
    INFO_COLOR = "#17a2b8"

# ==============================================================================
# STYLING
# ==============================================================================

def apply_custom_css():
    """Apply professional custom CSS styling"""
    st.markdown("""
    <style>
        /* Main App Styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Header Styling */
        .app-header {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .app-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .app-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Card Styling */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #2E86AB;
            margin-bottom: 1rem;
        }
        
        .metric-card h3 {
            margin: 0 0 0.5rem 0;
            color: #2E86AB;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #333;
        }
        
        /* Status Badges */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-approved {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-pending {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .status-rejected {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .status-flagged {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Tables */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Info Boxes */
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2E86AB;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .error-box {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 5px;
            font-weight: 600;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #2E86AB;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def init_nltk():
    """Initialise NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("Downloading language models..."):
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

def generate_id(text: str) -> str:
    """Generate unique ID from text"""
    return hashlib.md5(text.encode()).hexdigest()[:12]

def format_datetime(dt_str: str) -> str:
    """Format datetime string for display"""
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%b %d, %Y %I:%M %p')
    except:
        return dt_str

def validate_statement(statement: str) -> Tuple[bool, str]:
    """Validate user input statement"""
    if not statement or not statement.strip():
        return False, "Statement cannot be empty"
    if len(statement) < 10:
        return False, "Statement is too short (minimum 10 characters)"
    if len(statement) > 1000:
        return False, "Statement is too long (maximum 1000 characters)"
    return True, ""

def create_backup() -> Tuple[bytes, str, List[str]]:
    """Create a backup of all important data files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"physics_fact_checker_backup_{timestamp}.zip"
    files_to_backup = [
        AppConfig.DATASET_PATH,
        AppConfig.CONTRIBUTIONS_FILE,
        AppConfig.AUDIT_LOG_FILE
    ]
    zip_buffer = BytesIO()
    backed_up_files = []
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                archived_name = f"{timestamp}_{os.path.basename(file_path)}"
                zip_file.write(file_path, archived_name)
                backed_up_files.append(file_path)
        manifest = f"""Physics Fact Checker Backup
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {AppConfig.VERSION}

Files included:
{chr(10).join('- ' + f for f in backed_up_files)}

Total files: {len(backed_up_files)}
"""
        zip_file.writestr(f"{timestamp}_BACKUP_MANIFEST.txt", manifest)
    zip_buffer.seek(0)
    return zip_buffer.getvalue(), backup_filename, backed_up_files

class AuditLogger:
    """Audit logging for enterprise compliance"""
    def __init__(self, log_file: str = AppConfig.AUDIT_LOG_FILE):
        self.log_file = log_file
        self.ensure_log_file()
    def ensure_log_file(self):
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'user', 'action', 'entity_type', 
                'entity_id', 'details', 'ip_address'
            ])
            df.to_csv(self.log_file, index=False)
    def log(self, user: str, action: str, entity_type: str, 
            entity_id: str = "", details: str = ""):
        try:
            log_entry = pd.DataFrame([{
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': user,
                'action': action,
                'entity_type': entity_type,
                'entity_id': entity_id,
                'details': details,
                'ip_address': 'localhost'
            }])
            if os.path.exists(self.log_file):
                existing = pd.read_csv(self.log_file)
                updated = pd.concat([existing, log_entry], ignore_index=True)
            else:
                updated = log_entry
            updated.to_csv(self.log_file, index=False)
        except Exception as e:
            st.error(f"Logging error: {str(e)}")

# ==============================================================================
# OLLAMA CLIENT
# ==============================================================================

class OllamaClient:
    """Enhanced Ollama client with error handling and retry logic"""
    def __init__(self, base_url: str = AppConfig.OLLAMA_BASE_URL, 
                 model: str = AppConfig.OLLAMA_DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.available = self._check_availability()
        self._model_cache = None
    def _check_availability(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    def reconnect(self) -> bool:
        self.available = self._check_availability()
        return self.available
    def get_available_models(self) -> List[str]:
        if self._model_cache is not None:
            return self._model_cache
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                self._model_cache = [m.get('name', '') for m in models if m.get('name')]
                return self._model_cache
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
        return []
    def generate_response(self, prompt: str, max_tokens: int = 500, 
                          temperature: float = 0.1) -> Optional[str]:
        if not self.available:
            return None
        max_retries = 3
        for attempt in range(max_retries):
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
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                st.warning("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Ollama error: {str(e)}")
                break
        return None

# ==============================================================================
# USER CONTRIBUTION MANAGER
# ==============================================================================

class UserContributionManager:
    """Enhanced contribution management with validation and analytics"""
    def __init__(self, contributions_file: str = AppConfig.CONTRIBUTIONS_FILE):
        self.contributions_file = contributions_file
        self.pending_contributions = self.load_pending_contributions()
        self.audit_logger = AuditLogger()
    def load_pending_contributions(self) -> pd.DataFrame:
        try:
            if os.path.exists(self.contributions_file):
                df = pd.read_csv(self.contributions_file)
                required_cols = [
                    'ID', 'Statement', 'IsTrue', 'Category', 'Difficulty',
                    'Contributor', 'SubmissionDate', 'Status', 'AIVerification',
                    'VerificationConfidence', 'VerificationNotes'
                ]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None
                return df
            else:
                return self._create_empty_dataframe()
        except Exception as e:
            st.error(f"Error loading contributions: {str(e)}")
            return self._create_empty_dataframe()
    def _create_empty_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'ID', 'Statement', 'IsTrue', 'Category', 'Difficulty',
            'Contributor', 'SubmissionDate', 'Status', 'AIVerification',
            'VerificationConfidence', 'VerificationNotes'
        ])
    def save_contributions(self) -> bool:
        try:
            if os.path.exists(self.contributions_file):
                backup_file = f"{self.contributions_file}.backup"
                os.replace(self.contributions_file, backup_file)
            self.pending_contributions.to_csv(self.contributions_file, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving contributions: {str(e)}")
            return False
    def add_contribution(self, statement: str, is_true: int, category: str,
                         difficulty: str, contributor_name: str,
                         verify_with_ai: bool = True,
                         fact_checker: Optional['AdvancedPhysicsFactChecker'] = None) -> Dict:
        is_valid, error_msg = validate_statement(statement)
        if not is_valid:
            raise ValueError(error_msg)
        new_id = len(self.pending_contributions) + 1000
        ai_verification = None
        verification_confidence = 0.0
        verification_notes = ""
        if verify_with_ai and fact_checker:
            try:
                result = fact_checker.analyze_prompt(statement, use_ollama=True)
                if result.get('ollama_result'):
                    ollama_data = result['ollama_result']
                    ai_verification = ollama_data.get('is_correct')
                    verification_confidence = ollama_data.get('confidence', 0.0)
                    verification_notes = ollama_data.get('explanation', '')
                if ai_verification is not None:
                    user_says_true = (is_true == 1)
                    if user_says_true != ai_verification:
                        verification_notes += f" [CONFLICT: User says {user_says_true}, AI says {ai_verification}]"
            except Exception as e:
                verification_notes = f"AI verification failed: {str(e)}"
        new_contribution = {
            'ID': new_id,
            'Statement': statement.strip(),
            'IsTrue': is_true,
            'Category': category,
            'Difficulty': difficulty,
            'Contributor': contributor_name.strip(),
            'SubmissionDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Status': 'Pending',
            'AIVerification': ai_verification,
            'VerificationConfidence': verification_confidence,
            'VerificationNotes': verification_notes
        }
        new_row = pd.DataFrame([new_contribution])
        self.pending_contributions = pd.concat(
            [self.pending_contributions, new_row], 
            ignore_index=True
        )
        self.save_contributions()
        self.audit_logger.log(
            user=contributor_name,
            action="CREATE",
            entity_type="contribution",
            entity_id=str(new_id),
            details=f"Added: {statement[:50]}..."
        )
        return new_contribution
    def approve_contribution(self, contribution_id: int, approver: str = "Admin"):
        mask = self.pending_contributions['ID'] == contribution_id
        self.pending_contributions.loc[mask, 'Status'] = 'Approved'
        self.save_contributions()
        self.audit_logger.log(
            user=approver,
            action="APPROVE",
            entity_type="contribution",
            entity_id=str(contribution_id)
        )
    def reject_contribution(self, contribution_id: int, reason: str = "", 
                            rejector: str = "Admin"):
        mask = self.pending_contributions['ID'] == contribution_id
        self.pending_contributions.loc[mask, 'Status'] = 'Rejected'
        if reason:
            current_notes = self.pending_contributions.loc[mask, 'VerificationNotes'].iloc[0]
            updated_notes = f"{current_notes} [REJECTED: {reason}]"
            self.pending_contributions.loc[mask, 'VerificationNotes'] = updated_notes
        self.save_contributions()
        self.audit_logger.log(
            user=rejector,
            action="REJECT",
            entity_type="contribution",
            entity_id=str(contribution_id),
            details=reason
        )
    def get_statistics(self) -> Dict:
        if len(self.pending_contributions) == 0:
            return {
                'total': 0,
                'pending': 0,
                'approved': 0,
                'rejected': 0,
                'approval_rate': 0.0,
                'avg_confidence': 0.0
            }
        total = len(self.pending_contributions)
        pending = len(self.pending_contributions[self.pending_contributions['Status'] == 'Pending'])
        approved = len(self.pending_contributions[self.pending_contributions['Status'] == 'Approved'])
        rejected = len(self.pending_contributions[self.pending_contributions['Status'] == 'Rejected'])
        reviewed = approved + rejected
        approval_rate = (approved / reviewed * 100) if reviewed > 0 else 0.0
        verified = self.pending_contributions[self.pending_contributions['VerificationConfidence'].notna()]
        avg_confidence = verified['VerificationConfidence'].mean() if len(verified) > 0 else 0.0
        return {
            'total': total,
            'pending': pending,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': approval_rate,
            'avg_confidence': avg_confidence
        }
    def get_approved_contributions(self) -> pd.DataFrame:
        return self.pending_contributions[self.pending_contributions['Status'] == 'Approved'].copy()
    def get_pending_contributions(self) -> pd.DataFrame:
        return self.pending_contributions[self.pending_contributions['Status'] == 'Pending'].copy()

# ==============================================================================
# FACT CHECKER
# ==============================================================================

class AdvancedPhysicsFactChecker:
    """Enhanced fact checker with improved analysis and simplified confidence scoring"""

    def __init__(self, dataset: pd.DataFrame, ollama_client: Optional[OllamaClient] = None):
        self.dataset = dataset
        self.ollama_client = ollama_client

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.physics_concepts = {
            'temperature': ['celsius', 'fahrenheit', 'kelvin', 'degrees', 'hot', 'cold', 'boil', 'freeze', 'melt'],
            'energy': ['joule', 'calorie', 'kinetic', 'potential', 'thermal', 'nuclear', 'electromagnetic'],
            'waves': ['frequency', 'wavelength', 'amplitude', 'light', 'sound', 'electromagnetic', 'radio'],
            'mechanics': ['force', 'mass', 'weight', 'acceleration', 'velocity', 'momentum', 'gravity'],
            'electricity': ['current', 'voltage', 'resistance', 'charge', 'electron', 'proton', 'field'],
            'quantum': ['photon', 'quantum', 'particle', 'wave', 'uncertainty', 'probability', 'orbital'],
            'thermodynamics': ['entropy', 'heat', 'temperature', 'pressure', 'volume', 'gas', 'liquid', 'solid'],
        }

        self.misconception_patterns = [
            (r'water boils at (\d+)°?[cf]', self._check_boiling_point),
            (r'sound.*faster.*light', self._check_speed_comparison),
            (r'heavier.*fall.*faster', self._check_falling_objects),
            (r'all.*orbit.*perfect.*circle', self._check_orbital_shape),
            (r'mass.*weight.*same|interchangeable', self._check_mass_weight),
            (r'all.*radiation.*man.?made', self._check_radiation_source),
            (r'electrons.*orbit.*like.*planets', self._check_atomic_model),
            (r'pressure.*decreas.*depth|decreas.*pressure.*depth', self._check_pressure_depth),
        ]

        self._initialize_vectorizer()

    def _initialize_vectorizer(self):
        """Fit TF-IDF on lower-cased dataset; keep vectorizer lowercase=False and lower inputs ourselves."""
        try:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000,
                lowercase=False,  # we control casing explicitly
            )
            statements = self.dataset['Statement'].astype(str).str.lower().tolist()
            self.dataset_vectors = self.vectorizer.fit_transform(statements)
        except Exception as e:
            st.error(f"Error initializing vectorizer: {str(e)}")
            self.vectorizer = None
            self.dataset_vectors = None

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'°c|celsius', 'celsius', text)
        text = re.sub(r'°f|fahrenheit', 'fahrenheit', text)
        text = re.sub(r'°k|kelvin', 'kelvin', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(tokens)

    def _semantic_similarity(self, prompt: str) -> np.ndarray:
        if self.vectorizer is None or self.dataset_vectors is None:
            return np.array([])
        try:
            prompt_vec = self.vectorizer.transform([prompt.lower()])  # match training casing
            return cosine_similarity(prompt_vec, self.dataset_vectors).flatten()
        except Exception as e:
            st.error(f"Similarity calculation error: {str(e)}")
            return np.array([])

    def _physics_concept_matching(self, prompt: str) -> Dict[str, float]:
        prompt_lower = prompt.lower()
        concept_scores = {}
        for concept, keywords in self.physics_concepts.items():
            matches = sum(1 for keyword in keywords if keyword in prompt_lower)
            if matches > 0:
                concept_scores[concept] = min(1.0, matches / len(keywords))
        return concept_scores

    def _check_misconception_patterns(self, prompt: str) -> List[Dict]:
        misconceptions = []
        for pattern, check_func in self.misconception_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                result = check_func(match, prompt)
                if result:
                    misconceptions.append(result)
        return misconceptions

    def _ollama_fact_check(self, prompt: str) -> Optional[Dict]:
        if not self.ollama_client or not self.ollama_client.available:
            return None

        fact_check_prompt = f"""You are a physics expert. Analyze this statement and respond with ONLY a JSON object.

Statement: "{prompt}"

JSON format:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief explanation",
  "physics_domain": "domain name",
  "corrections": "corrections if needed"
}}

Respond ONLY with valid JSON, no other text."""
        response = self.ollama_client.generate_response(
            fact_check_prompt, max_tokens=300, temperature=0.1
        )
        if response:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                return {"raw_response": response}
        return None

    # ---- Misconception check helpers ----
    def _check_boiling_point(self, match, prompt):
        temp = int(match.group(1))
        if temp != 100 and 'celsius' in prompt.lower():
            return {
                'type': 'temperature_misconception',
                'description': f'Water boils at 100°C at standard pressure, not {temp}°C',
                'severity': 'high'
            }
        elif temp != 212 and 'fahrenheit' in prompt.lower():
            return {
                'type': 'temperature_misconception',
                'description': f'Water boils at 212°F at standard pressure, not {temp}°F',
                'severity': 'high'
            }
        return None

    def _check_speed_comparison(self, match, prompt):
        return {
            'type': 'speed_misconception',
            'description': 'Light travels much faster than sound (~300,000 km/s vs ~343 m/s)',
            'severity': 'high'
        }

    def _check_falling_objects(self, match, prompt):
        return {
            'type': 'gravity_misconception',
            'description': 'In a vacuum, all objects fall at the same rate regardless of mass',
            'severity': 'medium'
        }

    def _check_orbital_shape(self, match, prompt):
        return {
            'type': 'astronomy_misconception',
            'description': 'Planetary orbits are elliptical, not perfectly circular',
            'severity': 'medium'
        }

    def _check_mass_weight(self, match, prompt):
        return {
            'type': 'mechanics_misconception',
            'description': 'Mass is the amount of matter; weight is the gravitational force',
            'severity': 'medium'
        }

    def _check_radiation_source(self, match, prompt):
        return {
            'type': 'physics_misconception',
            'description': 'Natural radiation exists (cosmic rays, radioactive elements)',
            'severity': 'medium'
        }

    def _check_atomic_model(self, match, prompt):
        return {
            'type': 'quantum_misconception',
            'description': 'Electrons exist in probability clouds, not defined orbital paths',
            'severity': 'medium'
        }

    def _check_pressure_depth(self, match, prompt):
        return {
            'type': 'fluid_pressure_misconception',
            'description': 'In liquids, pressure increases with depth (p = p0 + ρ g h).',
            'severity': 'high'
        }

    # ---- Confidence scoring ----
    def _calculate_confidence(self, similarities, concept_matches,
                              misconceptions, ollama_result=None) -> Dict:
        # 1) Similarity (0..40)
        similarity_score = float(np.max(similarities)) * 40 if len(similarities) > 0 else 0
        # 2) Concepts (0..20)
        concept_score = ((len(concept_matches) / len(self.physics_concepts)) * 20) if concept_matches else 0
        # 3) AI (0..20)
        if ollama_result and 'confidence' in ollama_result:
            try:
                ai_score = float(ollama_result['confidence']) * 20
            except Exception:
                ai_score = 10
        else:
            ai_score = 10
        # 4) Misconceptions (0..20)
        misconception_score = 20 - min(20, len(misconceptions) * 10) if misconceptions else 20

        total_confidence = max(0, min(100, similarity_score + concept_score + ai_score + misconception_score))
        return {
            'confidence': total_confidence / 100,
            'confidence_percent': total_confidence,
            'breakdown': {
                'similarity': similarity_score,
                'concepts': concept_score,
                'ai_verification': ai_score,
                'misconception_check': misconception_score
            }
        }

    @staticmethod
    def get_confidence_label(confidence_percent: float) -> Tuple[str, str]:
        if confidence_percent >= 80:
            return "Very High", "🟢"
        elif confidence_percent >= 60:
            return "High", "🟡"
        elif confidence_percent >= 40:
            return "Medium", "🟠"
        elif confidence_percent >= 20:
            return "Low", "🔴"
        else:
            return "Very Low", "⚫"

    def analyze_prompt(self, prompt: str, similarity_threshold: float = 0.3,
                       use_ollama: bool = True) -> Dict:
        analysis_start = datetime.now()
        try:
            # --- Compute similarities first ---
            semantic_similarities = self._semantic_similarity(prompt)

            # Build similar_statements BEFORE any logic uses it
            best_match_indices = np.argsort(semantic_similarities)[::-1][:AppConfig.MAX_SIMILAR_STATEMENTS]
            similar_statements: List[Dict] = []
            for idx in best_match_indices:
                if semantic_similarities[idx] >= similarity_threshold:
                    row = self.dataset.iloc[idx]
                    similar_statements.append({
                        'id': row.get('ID', idx),
                        'statement': row.get('Statement', ''),
                        'is_true': int(row.get('IsTrue', 1)),
                        'difficulty': row.get('Difficulty', 'Unknown'),
                        'category': row.get('Category', 'Unknown'),
                        'similarity_score': float(semantic_similarities[idx]),
                        'match_type': 'semantic'
                    })

            # Other signals
            concept_matches = self._physics_concept_matching(prompt)
            misconceptions = self._check_misconception_patterns(prompt)

            ollama_result = self._ollama_fact_check(prompt) if use_ollama and self.ollama_client else None

            # Explanation/statement contradiction check
            contradiction = False
            if ollama_result:
                stmt = prompt.lower()
                exp = str(ollama_result.get('explanation', '')).lower()
                if 'pressure' in stmt and (
                    ('decreas' in stmt and 'increas' in exp) or
                    ('increas' in stmt and 'decreas' in exp)
                ):
                    contradiction = True

            # Confidence (after we have all signals)
            confidence_data = self._calculate_confidence(
                semantic_similarities, concept_matches, misconceptions, ollama_result
            )

            # Flagging logic (now we can safely reference similar_statements & confidence_data)
            is_flagged = False
            primary_reason = "No issues detected"

            if contradiction:
                is_flagged = True
                primary_reason = "AI explanation contradicts the statement"
            elif ollama_result and ollama_result.get('is_correct') is False:
                is_flagged = True
                primary_reason = "AI analysis suggests statement is incorrect"
            elif misconceptions:
                is_flagged = True
                primary_reason = f"Detected {len(misconceptions)} physics misconception(s)"
            elif similar_statements and similar_statements[0].get('is_true', 1) == 0:
                is_flagged = True
                primary_reason = "Similar to known incorrect statement"
            elif confidence_data['confidence_percent'] < 20:
                is_flagged = True
                primary_reason = "Very low confidence - insufficient information to verify"

            analysis_time = (datetime.now() - analysis_start).total_seconds()

            return {
                'flagged': is_flagged,
                'confidence': confidence_data['confidence'],
                'confidence_data': confidence_data,
                'primary_reason': primary_reason,
                'misconceptions': misconceptions,
                'physics_concepts': concept_matches,
                'similar_statements': similar_statements,
                'ollama_result': ollama_result,
                'analysis_time': analysis_time
            }

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return {
                'flagged': True,
                'confidence': 0.0,
                'confidence_data': {
                    'confidence': 0.0,
                    'confidence_percent': 0.0,
                    'breakdown': {
                        'similarity': 0,
                        'concepts': 0,
                        'ai_verification': 0,
                        'misconception_check': 0
                    }
                },
                'primary_reason': f"Analysis failed: {str(e)}",
                'misconceptions': [],
                'physics_concepts': {},
                'similar_statements': [],
                'ollama_result': None,
                'analysis_time': 0.0
            }
# ==============================================================================
# DATA LOADING
# ==============================================================================

@st.cache_data
def load_dataset() -> Tuple[pd.DataFrame, bool]:
    csv_path = AppConfig.DATASET_PATH
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
            if 'ID' not in df.columns:
                df['ID'] = range(1, len(df) + 1)
            return df.reset_index(drop=True), True
        else:
            return create_fallback_data(), False
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_fallback_data(), False
@st.cache_resource
def build_vectorizer(df):
    v = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=1000)
    X = v.fit_transform(df['Statement'].astype(str))
    return v, X
   # then inside AdvancedPhysicsFactChecker._initialize_vectorizer:
    self.vectorizer, self.dataset_vectors = build_vectorizer(self.dataset)
def _initialize_vectorizer(self):
    try:
        self.vectorizer, self.dataset_vectors = build_vectorizer(self.dataset)
    except Exception as e:
        st.error(f"Error initializing vectorizer: {str(e)}")
        self.vectorizer, self.dataset_vectors = None, None
def _semantic_similarity(self, prompt: str) -> np.ndarray:
    if self.vectorizer is None or self.dataset_vectors is None:
        return np.array([])
    try:
        # Vectorizer already lowercases internally (lowercase=True)
        prompt_vec = self.vectorizer.transform([prompt])
        return cosine_similarity(prompt_vec, self.dataset_vectors).flatten()
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return np.array([])
@st.cache_resource
def build_vectorizer(df: pd.DataFrame):
    texts = df['Statement'].astype(str).str.lower()
    v = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=1000)
    X = v.fit_transform(texts)
    return v, X
def _initilize_vectorizer(self):
    try:
        self.vectorizer, self.dataset_vectors = build_vectorizer(self.dataset)
    except Exception as e:
        st.error(f"Error initializing vectorizer: {str(e)}")
        self.vectorizer, self.dataset_vectors = None, None

def create_fallback_data() -> pd.DataFrame:
    data = [
        {"ID": 1, "Statement": "Water boils at 100°C at standard atmospheric pressure", 
         "IsTrue": 1, "Difficulty": "Easy", "Category": "Thermodynamics"},
        {"ID": 2, "Statement": "Sound travels faster than light", 
         "IsTrue": 0, "Difficulty": "Easy", "Category": "Physics"},
        {"ID": 3, "Statement": "E = mc² relates mass and energy", 
         "IsTrue": 1, "Difficulty": "Medium", "Category": "Modern Physics"},
        {"ID": 4, "Statement": "Heavier objects fall faster than lighter objects", 
         "IsTrue": 0, "Difficulty": "Easy", "Category": "Mechanics"},
        {"ID": 5, "Statement": "In a vacuum, all objects fall at the same rate", 
         "IsTrue": 1, "Difficulty": "Easy", "Category": "Mechanics"},
    ]
    return pd.DataFrame(data)

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_header():
    st.markdown(f"""
    <div class="app-header">
        <h1>🧪 {AppConfig.APP_TITLE}</h1>
        <p>{AppConfig.APP_SUBTITLE} • v{AppConfig.VERSION}</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, delta: str = None):
    col = st.container()
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

def render_status_badge(status: str) -> str:
    status_lower = status.lower()
    return f'<span class="status-badge status-{status_lower}">{status}</span>'

def render_sidebar(ollama_client: OllamaClient, dataset: pd.DataFrame,
                  contribution_manager: UserContributionManager) -> str:
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        if ollama_client.available:
            st.success("✅ AI Online")
            models = ollama_client.get_available_models()
            if models:
                selected_model = st.selectbox(
                    "AI Model",
                    models,
                    index=0 if ollama_client.model in models else 0
                )
                ollama_client.model = selected_model
        else:
            st.error("❌ AI Offline")
            if st.button("🔄 Reconnect AI"):
                if ollama_client.reconnect():
                    st.success("Reconnected!")
                    st.rerun()
                else:
                    st.error("Connection failed")
        st.markdown("---")
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "🔍 Single Analysis",
                "➕ Contribute Facts",
                "✅ Review Queue",
                "📊 Batch Analysis",
                "📚 Dataset Explorer",
                "⚙️ Settings"
            ],
            label_visibility="collapsed",
            key="nav_page"
        )
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        stats = contribution_manager.get_statistics()
        st.metric("Dataset Size", f"{len(dataset):,}")
        st.metric("Total Contributions", f"{stats['total']:,}")
        st.metric("Pending Review", f"{stats['pending']:,}")
        if stats['approval_rate'] > 0:
            st.metric("Approval Rate", f"{stats['approval_rate']:.1f}%")
        st.markdown("---")
        st.markdown("### 👤 User")
        st.info("Test User")
        return page

def render_dashboard(dataset: pd.DataFrame, contribution_manager: UserContributionManager,
                     ollama_client: OllamaClient):
    st.markdown("## 📊 System Overview")
    stats = contribution_manager.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Statements", f"{len(dataset):,}", 
                  delta=f"+{stats['approved']}" if stats['approved'] > 0 else None)
    with col2:
        st.metric("Pending Review", f"{stats['pending']:,}",
                  delta="Action Required" if stats['pending'] > 0 else None)
    with col3:
        st.metric("Approval Rate", f"{stats['approval_rate']:.1f}%")
    with col4:
        ai_status = "Online" if ollama_client.available else "Offline"
        st.metric("AI Status", ai_status)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Contribution Trends")
        if len(contribution_manager.pending_contributions) > 0:
            status_counts = contribution_manager.pending_contributions['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Status Distribution",
                color_discrete_map={
                    'Approved': AppConfig.SUCCESS_COLOR,
                    'Pending': AppConfig.WARNING_COLOR,
                    'Rejected': AppConfig.DANGER_COLOR
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No contribution data available")
    with col2:
        st.markdown("### 🎯 Category Distribution")
        if 'Category' in dataset.columns:
            category_counts = dataset['Category'].value_counts().head(10)
            fig = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title="Top Categories",
                labels={'x': 'Count', 'y': 'Category'}
            )
            fig.update_traces(marker_color=AppConfig.PRIMARY_COLOR)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available")
    st.markdown("---")
    st.markdown("### 📰 Recent Activity")
    if len(contribution_manager.pending_contributions) > 0:
        recent = contribution_manager.pending_contributions.sort_values(
            'SubmissionDate', ascending=False
        ).head(5)
        for _, contrib in recent.iterrows():
            status_badge = render_status_badge(contrib['Status'])
            st.markdown(f"""
            **{contrib['Contributor']}** • {format_datetime(contrib['SubmissionDate'])}  
            {status_badge}  
            *{contrib['Statement'][:100]}...*
            """, unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No recent activity")

def render_single_analysis(checker: AdvancedPhysicsFactChecker,
                           ollama_client: OllamaClient,
                           contribution_manager: UserContributionManager):
    st.markdown("## 🔍 AI-Enhanced Statement Analysis")
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Enter a physics statement below and our AI will analyse it 
        against our knowledge base, check for common misconceptions, and provide detailed feedback.
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area(
            "Physics Statement",
            height=120,
            placeholder="Example: Water boils at 100°C at sea level",
            help="Enter any physics statement you'd like to verify",
            key="single_input"
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            similarity_threshold = st.slider(
                "Match Strictness",
                0.1, 0.8, 0.3, 0.05,
                help="Higher = stricter matching",
                key="single_similarity"
            )
        with col_b:
            use_ollama = st.checkbox(
                "Use AI Analysis",
                value=ollama_client.available,
                disabled=not ollama_client.available,
                help="Enable AI-powered deep analysis",
                key="single_use_ai"
            )
        with col_c:
            analyse_btn = st.button("🚀 Analyse", type="primary", use_container_width=True, key="single_analyse")
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific and clear  
        - Use proper units  
        - Include context when needed  
        - Check spelling
        """)
    if analyse_btn:
        if not user_input.strip():
            st.error("Please enter a statement to analyse")
        else:
            is_valid, error_msg = validate_statement(user_input)
            if not is_valid:
                st.error(error_msg)
            else:
                with st.spinner("🔬 Analysing statement..."):
                    result = checker.analyze_prompt(
                        user_input,
                        similarity_threshold,
                        use_ollama
                    )
                    if result['flagged']:
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>🚨 Statement Flagged</h3>
                            <p>{result['primary_reason']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ Statement Approved</h3>
                            <p>No issues detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("### 📊 Analysis Metrics")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if 'confidence_data' in result:
                            conf_percent = result['confidence_data']['confidence_percent']
                            label, emoji = AdvancedPhysicsFactChecker.get_confidence_label(conf_percent)
                            st.metric(
                                "Confidence",
                                f"{emoji} {conf_percent:.0f}%",
                                delta=label
                            )
                        else:
                            confidence_colour = (
                                "🟢" if result['confidence'] > 0.7
                                else "🟡" if result['confidence'] > 0.4
                                else "🔴"
                            )
                            st.metric(
                                "Confidence",
                                f"{confidence_colour} {result['confidence']:.1%}"
                            )
                    with c2:
                        st.metric("Processing Time", f"{result['analysis_time']*1000:.0f}ms")
                    with c3:
                        st.metric("Issues Found", len(result['misconceptions']))
                    with c4:
                        ai_used = "Yes" if result['ollama_result'] else "No"
                        st.metric("AI Analysis", ai_used)
                    if 'confidence_data' in result and 'breakdown' in result['confidence_data']:
                        with st.expander("📊 Confidence Score Breakdown"):
                            breakdown = result['confidence_data']['breakdown']
                            st.markdown("**How we calculated your confidence score:**")
                            st.progress(breakdown['similarity'] / 40)
                            st.caption(f"Statement similarity: {breakdown['similarity']:.0f}/40 points")
                            st.progress(breakdown['concepts'] / 20)
                            st.caption(f"Physics concepts detected: {breakdown['concepts']:.0f}/20 points")
                            st.progress(breakdown['ai_verification'] / 20)
                            st.caption(f"AI verification: {breakdown['ai_verification']:.0f}/20 points")
                            st.progress(breakdown['misconception_check'] / 20)
                            st.caption(f"Misconception check: {breakdown['misconception_check']:.0f}/20 points")
                            st.markdown("---")
                            st.markdown("""
                            **What the scores mean:**
                            - **80–100%**: Very high confidence – statement is likely correct
                            - **60–79%**: High confidence – good reliability
                            - **40–59%**: Medium confidence – some uncertainty
                            - **20–39%**: Low confidence – needs review
                            - **0–19%**: Very low confidence – likely incorrect
                            """)
                    if result['ollama_result']:
                        st.markdown("---")
                        st.markdown("### 🤖 AI Analysis")
                        ollama_data = result['ollama_result']
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if 'is_correct' in ollama_data:
                                verdict = "✅ Correct" if ollama_data['is_correct'] else "❌ Incorrect"
                                st.markdown(f"**Verdict:** {verdict}")
                            if 'confidence' in ollama_data:
                                st.progress(ollama_data['confidence'])
                                st.caption(f"AI Confidence: {ollama_data['confidence']:.1%}")
                        with cc2:
                            if 'physics_domain' in ollama_data:
                                st.markdown(f"**Domain:** {ollama_data['physics_domain']}")
                        if 'explanation' in ollama_data:
                            st.markdown("**Explanation:**")
                            st.info(ollama_data['explanation'])
                        if 'corrections' in ollama_data and ollama_data['corrections']:
                            st.markdown("**Suggested Corrections:**")
                            st.warning(ollama_data['corrections'])
                    if result['misconceptions']:
                        st.markdown("---")
                        st.markdown("### ⚠️ Detected Issues")
                        for i, misc in enumerate(result['misconceptions'], 1):
                            severity_color = {
                                'high': '🔴',
                                'medium': '🟡',
                                'low': '🟢'
                            }.get(misc.get('severity', 'medium'), '🟡')
                            with st.expander(f"{severity_color} Issue {i}: {misc['type'].replace('_', ' ').title()}"):
                                st.markdown(f"**Severity:** {misc['severity'].upper()}")
                                st.markdown(f"**Description:** {misc['description']}")
                    if result['similar_statements']:
                        st.markdown("---")
                        st.markdown("### 📚 Similar Statements")
                        df_similar = pd.DataFrame(result['similar_statements'])
                        df_similar['is_true'] = df_similar['is_true'].map({1: '✅ True', 0: '❌ False'})
                        df_similar['similarity_score'] = df_similar['similarity_score'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(
                            df_similar[['statement', 'is_true', 'similarity_score', 'category', 'difficulty']],
                            use_container_width=True,
                            column_config={
                                "statement": "Statement",
                                "is_true": "Truth Value",
                                "similarity_score": "Match",
                                "category": "Category",
                                "difficulty": "Difficulty"
                            }
                        )
                    st.markdown("---")
                    st.markdown("### ➕ Add to Knowledge Base")
                    with st.form("contribute_from_analysis"):
                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            contributor_name = st.text_input("Your Name", placeholder="Enter your name")
                        with cc2:
                            is_correct = st.selectbox(
                                "Statement is:",
                                [1, 0],
                                format_func=lambda x: "True" if x == 1 else "False"
                            )
                        with cc3:
                            submit_contrib = st.form_submit_button("Add to Database", type="primary", use_container_width=True)
                        if submit_contrib:
                            if not contributor_name.strip():
                                st.error("Please enter your name")
                            else:
                                try:
                                    contribution_manager.add_contribution(
                                        statement=user_input.strip(),
                                        is_true=is_correct,
                                        category="Physics",
                                        difficulty="Medium",
                                        contributor_name=contributor_name.strip(),
                                        verify_with_ai=False,
                                        fact_checker=None
                                    )
                                    st.success("✅ Statement added to review queue!")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

# ---------- Safe callback to clear batch state ----------
def _clear_batch_state():
    st.session_state["batch_text"] = ""
    st.session_state["uploaded_batch_df"] = None
    st.session_state["batch_results"] = []

def render_contribute_page(contribution_manager: UserContributionManager,
                           fact_checker: AdvancedPhysicsFactChecker,
                           ollama_available: bool):
    st.markdown("## ➕ Contribute New Physics Facts")
    st.markdown("""
    <div class="info-box">
        <strong>Help us grow!</strong> Your contributions help build a more comprehensive 
        physics knowledge base. All submissions are reviewed before being added to the main dataset.
    </div>
    """, unsafe_allow_html=True)
    with st.form("contribution_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            contributor_name = st.text_input("Your Name / Username *", placeholder="John Doe")
            statement = st.text_area(
                "Physics Statement *",
                height=120,
                placeholder="Enter a physics fact or statement...",
                help="Be clear and specific. Include units where appropriate."
            )
            is_true = st.selectbox(
                "Is this statement correct? *",
                options=[1, 0],
                format_func=lambda x: "✅ True" if x == 1 else "❌ False"
            )
        with col2:
            category = st.selectbox(
                "Physics Category *",
                [
                    "Thermodynamics", "Mechanics", "Waves", "Electricity",
                    "Quantum Physics", "Modern Physics", "Astronomy",
                    "Optics", "Nuclear Physics", "Other"
                ]
            )
            difficulty = st.selectbox("Difficulty Level *", ["Easy", "Medium", "Hard"])
            verify_with_ai = st.checkbox(
                "🤖 Verify with AI before submission",
                value=ollama_available,
                disabled=not ollama_available,
                help="AI will check your fact and flag any discrepancies"
            )
        st.markdown("---")
        submit_button = st.form_submit_button("🚀 Submit Contribution", type="primary", use_container_width=True)
        if submit_button:
            errors = []
            if not contributor_name.strip():
                errors.append("Name is required")
            if not statement.strip():
                errors.append("Statement is required")
            else:
                is_valid, error_msg = validate_statement(statement)
                if not is_valid:
                    errors.append(error_msg)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                with st.spinner("Processing your contribution..."):
                    try:
                        contribution = contribution_manager.add_contribution(
                            statement=statement.strip(),
                            is_true=is_true,
                            category=category,
                            difficulty=difficulty,
                            contributor_name=contributor_name.strip(),
                            verify_with_ai=verify_with_ai,
                            fact_checker=fact_checker if verify_with_ai else None
                        )
                        st.success("✅ Contribution submitted successfully!")
                        if verify_with_ai and contribution.get('AIVerification') is not None:
                            st.markdown("---")
                            st.markdown("### 🤖 AI Verification Results")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                user_verdict = "True" if is_true == 1 else "False"
                                st.metric("Your Assessment", user_verdict)
                            with c2:
                                ai_verdict = "True" if contribution['AIVerification'] else "False"
                                st.metric("AI Assessment", ai_verdict)
                            with c3:
                                st.metric("AI Confidence", f"{contribution['VerificationConfidence']:.1%}")
                            if contribution['VerificationNotes']:
                                st.info(contribution['VerificationNotes'])
                            if "[CONFLICT:" in contribution['VerificationNotes']:
                                st.warning("""
                                ⚠️ **Attention:** There's a disagreement between your assessment 
                                and the AI's assessment. This will be flagged for manual review.
                                """)
                    except Exception as e:
                        st.error(f"Error submitting contribution: {str(e)}")

def render_review_queue(contribution_manager: UserContributionManager):
    st.markdown("## ✅ Review Queue")
    pending = contribution_manager.get_pending_contributions()
    if len(pending) == 0:
        st.markdown("""
        <div class="success-box">
            <h3>🎉 All caught up!</h3>
            <p>There are no pending contributions to review.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown(f"""
    <div class="warning-box">
        <strong>{len(pending)} contributions</strong> awaiting review
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(pending['Category'].unique().tolist())
        )
    with col2:
        contributor_filter = st.selectbox(
            "Filter by Contributor",
            ["All"] + sorted(pending['Contributor'].unique().tolist())
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First", "High Confidence", "Low Confidence"]
        )
    filtered = pending.copy()
    if category_filter != "All":
        filtered = filtered[filtered['Category'] == category_filter]
    if contributor_filter != "All":
        filtered = filtered[filtered['Contributor'] == contributor_filter]
    if sort_by == "Newest First":
        filtered = filtered.sort_values('SubmissionDate', ascending=False)
    elif sort_by == "Oldest First":
        filtered = filtered.sort_values('SubmissionDate', ascending=True)
    elif sort_by == "High Confidence":
        filtered = filtered.sort_values('VerificationConfidence', ascending=False, na_position='last')
    elif sort_by == "Low Confidence":
        filtered = filtered.sort_values('VerificationConfidence', ascending=True, na_position='last')
    st.markdown(f"Showing **{len(filtered)}** of **{len(pending)}** contributions")
    st.markdown("---")
    for idx, contrib in filtered.iterrows():
        with st.expander(
            f"ID {contrib['ID']}: {contrib['Statement'][:80]}..." +
            f" • {contrib['Category']} • {contrib['Contributor']}"
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("### Statement")
                st.write(contrib['Statement'])
                st.markdown("### Metadata")
                m1, m2 = st.columns(2)
                with m1:
                    st.write(f"**Contributor:** {contrib['Contributor']}")
                    st.write(f"**Submitted:** {format_datetime(contrib['SubmissionDate'])}")
                    st.write(f"**User Says:** {'✅ True' if contrib['IsTrue'] == 1 else '❌ False'}")
                with m2:
                    st.write(f"**Category:** {contrib['Category']}")
                    st.write(f"**Difficulty:** {contrib['Difficulty']}")
            with c2:
                st.markdown("### AI Verification")
                if pd.notna(contrib['AIVerification']):
                    ai_verdict = "✅ True" if contrib['AIVerification'] else "❌ False"
                    st.metric("AI Says", ai_verdict)
                    st.metric("Confidence", f"{contrib['VerificationConfidence']:.1%}")
                    if contrib['VerificationNotes']:
                        st.info(contrib['VerificationNotes'])
                else:
                    st.warning("No AI verification")
            st.markdown("---")
            b1, b2, b3 = st.columns([1, 1, 2])
            with b1:
                if st.button(f"✅ Approve", key=f"approve_{contrib['ID']}", 
                             type="primary", use_container_width=True):
                    contribution_manager.approve_contribution(contrib['ID'])
                    st.success("Approved!")
                    time.sleep(1)
                    st.rerun()
            with b2:
                if st.button(f"❌ Reject", key=f"reject_{contrib['ID']}",
                             use_container_width=True):
                    st.session_state[f'rejecting_{contrib["ID"]}'] = True
            with b3:
                if st.session_state.get(f'rejecting_{contrib["ID"]}', False):
                    reason = st.text_input(
                        "Rejection reason",
                        key=f"reason_{contrib['ID']}",
                        placeholder="Enter reason..."
                    )
                    if st.button(f"Confirm Reject", key=f"confirm_{contrib['ID']}"):
                        contribution_manager.reject_contribution(contrib['ID'], reason)
                        st.warning("Rejected!")
                        del st.session_state[f'rejecting_{contrib["ID"]}']
                        time.sleep(1)
                        st.rerun()

def render_batch_analysis(checker: AdvancedPhysicsFactChecker, ollama_client: OllamaClient):
    """Render batch analysis page for analysing multiple statements"""
    st.markdown("## 📊 Batch Analysis")
    st.markdown("""
    <div class="info-box">
        <strong>Batch Processing:</strong> Analyse multiple physics statements at once. 
        Upload a CSV file or enter multiple statements manually.
    </div>
    """, unsafe_allow_html=True)
    # Persisted input method
    input_method = st.radio(
        "Select input method:",
        ["📝 Manual Entry", "📁 CSV Upload"],
        key="batch_input_method"
    )
    # Top controls
    left, right = st.columns([1, 1])
    with left:
        st.button("🧹 Clear batch", use_container_width=True, key="batch_clear", on_click=_clear_batch_state)
    statements_to_analyze: List[str] = []
    if input_method == "📝 Manual Entry":
        st.markdown("### Enter statements (one per line)")
        st.text_area(
            "Physics Statements",
            height=200,
            placeholder="Enter statements here, one per line...\nExample:\nWater boils at 100°C\nLight travels faster than sound",
            label_visibility="collapsed",
            key="batch_text",
        )
        if st.session_state.get("batch_text"):
            statements_to_analyze = [
                s.strip() for s in st.session_state.batch_text.splitlines() if s.strip()
            ]
            st.info(f"Found {len(statements_to_analyze)} statements to analyse")
    else:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should have a 'Statement' column",
            key="batch_upload"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Statement' in df.columns:
                    st.session_state["uploaded_batch_df"] = df
                    statements_to_analyze = df['Statement'].dropna().tolist()
                    st.success(f"Loaded {len(statements_to_analyze)} statements from CSV")
                    with st.expander("Preview uploaded data"):
                        st.dataframe(df.head(10))
                else:
                    st.error("CSV file must contain a 'Statement' column")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        elif st.session_state.get("uploaded_batch_df") is not None:
            df = st.session_state.uploaded_batch_df
            statements_to_analyze = df['Statement'].dropna().tolist()
            st.info(f"Using previously uploaded CSV with {len(statements_to_analyze)} statements")
    st.markdown("---")
    st.markdown("### Analysis Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox(
            "Use AI analysis",
            value=ollama_client.available,
            disabled=not ollama_client.available,
            key="batch_use_ai",
        )
    with c2:
        st.slider(
            "Similarity Threshold",
            0.1, 0.8, 0.3, 0.05,
            key="batch_similarity_threshold"
        )
    with c3:
        st.checkbox("Export Results", value=True, key="batch_export_results")
    run = st.button("🚀 Start Batch Analysis", type="primary",
                    disabled=len(statements_to_analyze) == 0,
                    key="batch_start", use_container_width=True)
    if run and statements_to_analyze:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        total = len(statements_to_analyze)
        for i, statement in enumerate(statements_to_analyze):
            status_text.text(f"Analysing statement {i+1}/{total}...")
            progress_bar.progress((i + 1) / total)
            result = checker.analyze_prompt(
                statement,
                st.session_state.batch_similarity_threshold,
                st.session_state.batch_use_ai
            )
            confidence_display = (
                f"{result['confidence_data']['confidence_percent']:.0f}%"
                if 'confidence_data' in result else
                f"{result['confidence']:.1%}"
            )
            results.append({
                'Statement': statement,
                'Flagged': '❌' if result['flagged'] else '✅',
                'Confidence': confidence_display,
                'Reason': result['primary_reason'],
                'Issues': len(result['misconceptions']),
                'Processing Time (ms)': f"{result['analysis_time']*1000:.0f}"
            })
        st.session_state["batch_results"] = results
        status_text.text("Analysis complete!")
    if st.session_state.get("batch_results"):
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        results_df = pd.DataFrame(st.session_state.batch_results)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            flagged_count = sum(r["Flagged"] == "❌" for r in st.session_state.batch_results)
            total = len(st.session_state.batch_results)
            st.metric("Flagged Statements", f"{flagged_count}/{total}")
        with m2:
            def parse_conf(v: str) -> float:
                v = v.strip()
                if v.endswith("%"):
                    return float(v[:-1]) / 100.0
                try:
                    return float(v)
                except:
                    return 0.0
            avg_conf = np.mean([parse_conf(r["Confidence"]) for r in st.session_state.batch_results])
            st.metric("Average Confidence", f"{avg_conf:.1%}")
        with m3:
            total_issues = sum(r["Issues"] for r in st.session_state.batch_results)
            st.metric("Total Issues Found", total_issues)
        with m4:
            avg_time = np.mean([float(r["Processing Time (ms)"]) for r in st.session_state.batch_results])
            st.metric("Avg Processing Time", f"{avg_time:.0f}ms")
        st.dataframe(results_df, use_container_width=True)
        if st.session_state.batch_export_results:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def render_dataset_explorer(dataset: pd.DataFrame, contribution_manager: UserContributionManager):
    st.markdown("## 📚 Dataset Explorer")
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search & Filter", "📊 Statistics", "📈 Visualisations", "📥 Export"])
    with tab1:
        st.markdown("### Search and Filter Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔎 Search statements", placeholder="Enter keywords...", key="ds_search")
        with col2:
            category_filter = st.multiselect(
                "Categories",
                options=dataset['Category'].unique() if 'Category' in dataset.columns else [],
                key="ds_cat"
            )
        with col3:
            difficulty_filter = st.multiselect(
                "Difficulty",
                options=dataset['Difficulty'].unique() if 'Difficulty' in dataset.columns else [],
                key="ds_diff"
            )
        truth_filter = st.radio(
            "Truth Value",
            ["All", "True Only", "False Only"],
            horizontal=True,
            key="ds_truth"
        )
        filtered_df = dataset.copy()
        if search_term:
            filtered_df = filtered_df[
                filtered_df['Statement'].str.contains(search_term, case=False, na=False)
            ]
        if category_filter:
            filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
        if difficulty_filter:
            filtered_df = filtered_df[filtered_df['Difficulty'].isin(difficulty_filter)]
        if truth_filter == "True Only":
            filtered_df = filtered_df[filtered_df['IsTrue'] == 1]
        elif truth_filter == "False Only":
            filtered_df = filtered_df[filtered_df['IsTrue'] == 0]
        st.info(f"Showing {len(filtered_df)} of {len(dataset)} statements")
        st.dataframe(
            filtered_df[['ID', 'Statement', 'IsTrue', 'Category', 'Difficulty']],
            use_container_width=True,
            column_config={
                "IsTrue": st.column_config.CheckboxColumn("Is True"),
            }
        )
    with tab2:
        st.markdown("### Dataset Statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Statements", f"{len(dataset):,}")
            st.metric("True Statements", f"{(dataset['IsTrue'] == 1).sum():,}")
            st.metric("False Statements", f"{(dataset['IsTrue'] == 0).sum():,}")
        with c2:
            st.metric("Unique Categories", f"{dataset['Category'].nunique()}")
            st.metric("Average Statement Length", f"{dataset['Statement'].str.len().mean():.0f} chars")
            true_ratio = (dataset['IsTrue'] == 1).sum() / len(dataset) * 100
            st.metric("Truth Ratio", f"{true_ratio:.1f}%")
        st.markdown("---")
        st.markdown("### Category Breakdown")
        category_stats = dataset.groupby('Category').agg({
            'ID': 'count',
            'IsTrue': lambda x: (x == 1).sum()
        }).rename(columns={'ID': 'Total', 'IsTrue': 'True Statements'})
        category_stats['False Statements'] = category_stats['Total'] - category_stats['True Statements']
        category_stats['Truth Rate'] = (category_stats['True Statements'] / category_stats['Total'] * 100).round(1)
        st.dataframe(category_stats, use_container_width=True)
    with tab3:
        st.markdown("### Data Visualisations")
        fig1 = px.pie(dataset, names='Category', title="Statement Distribution by Category")
        st.plotly_chart(fig1, use_container_width=True)
        truth_by_category = dataset.groupby(['Category', 'IsTrue']).size().reset_index(name='Count')
        truth_by_category['IsTrue'] = truth_by_category['IsTrue'].map({0: 'False', 1: 'True'})
        fig2 = px.bar(
            truth_by_category,
            x='Category',
            y='Count',
            color='IsTrue',
            title="Truth Values by Category",
            color_discrete_map={'True': AppConfig.SUCCESS_COLOR, 'False': AppConfig.DANGER_COLOR}
        )
        st.plotly_chart(fig2, use_container_width=True)
        if 'Difficulty' in dataset.columns:
            difficulty_order = ['Easy', 'Medium', 'Hard']
            difficulty_counts = dataset['Difficulty'].value_counts()
            fig3 = px.bar(
                x=difficulty_order,
                y=[difficulty_counts.get(d, 0) for d in difficulty_order],
                title="Statement Distribution by Difficulty",
                labels={'x': 'Difficulty', 'y': 'Count'}
            )
            fig3.update_traces(marker_color=AppConfig.PRIMARY_COLOR)
            st.plotly_chart(fig3, use_container_width=True)
    with tab4:
        st.markdown("### Export Dataset")
        export_format = st.radio(
            "Select export format:",
            ["CSV", "JSON", "Excel"],
            horizontal=True,
            key="ds_export_fmt"
        )
        include_contributions = st.checkbox("Include approved contributions", key="ds_include_contribs")
        export_data = dataset.copy()
        if include_contributions:
            approved = contribution_manager.get_approved_contributions()
            if len(approved) > 0:
                approved_formatted = approved[['ID', 'Statement', 'IsTrue', 'Category', 'Difficulty']]
                export_data = pd.concat([export_data, approved_formatted], ignore_index=True)
                st.info(f"Including {len(approved)} approved contributions")
        if export_format == "CSV":
            csv = export_data.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "physics_dataset.csv",
                "text/csv",
                use_container_width=True
            )
        elif export_format == "JSON":
            json_str = export_data.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download JSON",
                json_str,
                "physics_dataset.json",
                "application/json",
                use_container_width=True
            )
        elif export_format == "Excel":
            try:
                   import openpyxl  # noqa
            except ImportError:
                 st.info("Install openpyxl to enable Excel export: pip install openpyxl")
            else:
                 output = BytesIO()
                 with pd.ExcelWriter(output, engine='openpyxl') as writer:
                          export_data.to_excel(writer, index=False, sheet_name='Dataset')
                 excel_data = output.getvalue()
                 st.download_button(
                       "📥 Download Excel",
                        excel_data,
                        "physics_dataset.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         use_container_width=True
             )


def render_settings(ollama_client: OllamaClient):
    st.markdown("## ⚙️ System Settings")
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Settings", "📊 Analysis Settings", "🎨 UI Preferences", "🔧 System"])
    with tab1:
        st.markdown("### AI Model Configuration")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input(
                "Ollama Base URL",
                value=AppConfig.OLLAMA_BASE_URL,
                key="ollama_url",
                help="URL for the Ollama service"
            )
            if ollama_client.available:
                models = ollama_client.get_available_models()
                if models:
                    st.selectbox(
                        "Default Model",
                        models,
                        index=models.index(ollama_client.model) if ollama_client.model in models else 0,
                        key="default_model"
                    )
            else:
                st.warning("AI service is offline")
        with c2:
            st.number_input("Max Tokens", min_value=100, max_value=2000, value=500, step=100, key="max_tokens")
            st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temperature", help="Lower = more focused, Higher = more creative")
        if st.button("Test AI Connection", type="primary"):
            with st.spinner("Testing connection..."):
                if ollama_client.reconnect():
                    st.success("✅ Connection successful!")
                    test_response = ollama_client.generate_response(
                        "Say 'Hello, Physics!' in one sentence.",
                        max_tokens=50
                    )
                    if test_response:
                        st.info(f"AI Response: {test_response}")
                else:
                    st.error("❌ Connection failed. Please check your settings.")
    with tab2:
        st.markdown("### Analysis Configuration")
        c1, c2 = st.columns(2)
        with c1:
            st.slider(
                "Default Similarity Threshold",
                0.1, 0.8,
                AppConfig.DEFAULT_SIMILARITY_THRESHOLD,
                0.05,
                key="similarity_threshold",
                help="Default threshold for similarity matching"
            )
            st.number_input("Max Similar Statements", 1, 50, AppConfig.MAX_SIMILAR_STATEMENTS, key="max_similar")
        with c2:
            st.checkbox("Enable Misconception Detection", value=True, key="enable_misconceptions")
            st.checkbox("Auto-verify Contributions with AI", value=True, key="auto_verify")
        st.markdown("---")
        st.markdown("### Physics Concept Keywords")
        st.info("Customise the keywords used for physics concept detection")
        concept_categories = ['temperature', 'energy', 'waves', 'mechanics']
        selected_concept = st.selectbox("Select concept category:", concept_categories, key="kw_cat")
        st.text_area(
            f"Keywords for {selected_concept}",
            value="celsius, fahrenheit, kelvin, degrees, hot, cold",
            height=100,
            key=f"keywords_{selected_concept}",
            help="Enter comma-separated keywords"
        )
    with tab3:
        st.markdown("### User Interface Preferences")
        c1, c2 = st.columns(2)
        with c1:
            st.color_picker("Primary Colour", value=AppConfig.PRIMARY_COLOR, key="primary_color")
            st.color_picker("Success Colour", value=AppConfig.SUCCESS_COLOR, key="success_color")
        with c2:
            st.color_picker("Warning Colour", value=AppConfig.WARNING_COLOR, key="warning_color")
            st.color_picker("Danger Colour", value=AppConfig.DANGER_COLOR, key="danger_color")
        st.markdown("---")
        st.markdown("### Display Options")
        st.checkbox("Show tooltips", value=True, key="show_tooltips")
        st.checkbox("Enable animations", value=True, key="enable_animations")
        st.checkbox("Compact mode", value=False, key="compact_mode")
    with tab4:
        st.markdown("### System Information")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Application Version", AppConfig.VERSION)
            st.metric("Python Version", "3.x")
            st.metric("Streamlit Version", st.__version__)
        with c2:
            st.metric("Dataset Path", AppConfig.DATASET_PATH[:30] + "...")
            st.metric("Contributions File", AppConfig.CONTRIBUTIONS_FILE)
            st.metric("Audit Log File", AppConfig.AUDIT_LOG_FILE)
        st.markdown("---")
        st.markdown("### Data Management")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        with b2:
            if st.button("📋 View Audit Log", use_container_width=True):
                if os.path.exists(AppConfig.AUDIT_LOG_FILE):
                    audit_df = pd.read_csv(AppConfig.AUDIT_LOG_FILE)
                    st.dataframe(audit_df.tail(10))
                else:
                    st.info("No audit log found")
        with b3:
            if st.button("💾 Create Backup", use_container_width=True):
                with st.spinner("Creating backup..."):
                    try:
                        backup_data, backup_filename, backed_up_files = create_backup()
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ Backup Created Successfully!</h4>
                            <p>Your data has been backed up and is ready to download.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("**Files included in backup:**")
                        for file in backed_up_files:
                            file_size = os.path.getsize(file) / 1024
                            st.write(f"✓ `{file}` ({file_size:.2f} KB)")
                        st.download_button(
                            label="📥 Download Backup ZIP",
                            data=backup_data,
                            file_name=backup_filename,
                            mime="application/zip",
                            use_container_width=True,
                            type="primary"
                        )
                        st.info(f"💡 **Tip:** Store this backup in a safe location. Backup contains {len(backed_up_files)} file(s).")
                    except Exception as e:
                        st.error(f"❌ Backup failed: {str(e)}")
        st.markdown("---")
        st.warning("⚠️ **Advanced Settings** - Changes here may affect system stability")
        with st.expander("Developer Options"):
            st.checkbox("Enable debug mode", key="debug_mode")
            st.checkbox("Show performance metrics", key="show_metrics")
            st.number_input("Request timeout (seconds)", 5, 60, 30, key="timeout")

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_css()
    init_nltk()
    ollama_client = OllamaClient()
    dataset, dataset_loaded = load_dataset()
    contribution_manager = UserContributionManager()
    checker = AdvancedPhysicsFactChecker(dataset, ollama_client)
    render_header()
    page = render_sidebar(ollama_client, dataset, contribution_manager)
    if page == "🏠 Dashboard":
        render_dashboard(dataset, contribution_manager, ollama_client)
    elif page == "🔍 Single Analysis":
        render_single_analysis(checker, ollama_client, contribution_manager)
    elif page == "➕ Contribute Facts":
        render_contribute_page(contribution_manager, checker, ollama_client.available)
    elif page == "✅ Review Queue":
        render_review_queue(contribution_manager)
    elif page == "📊 Batch Analysis":
        render_batch_analysis(checker, ollama_client)
    elif page == "📚 Dataset Explorer":
        render_dataset_explorer(dataset, contribution_manager)
    elif page == "⚙️ Settings":
        render_settings(ollama_client)
    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.caption(f"Version {AppConfig.VERSION}")
    with f2:
        st.caption(f"Dataset: {len(dataset):,} statements")
    with f3:
        st.caption("© 2025 Capstone Project University of Canberra")

if __name__ == "__main__":
    main()
