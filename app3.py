# app.py - ConcordiaChain: ML-Driven Dynamic Consensus Blockchain Simulator
#
# A single-file Flask app implementing a multi-node, multi-consensus blockchain
# with ML-driven dynamic consensus, high-aesthetic UI, and CSV-backed Auth.
#
# Key Feature: State Persistence and Community Governance for new users.
#
# Run: pip install flask pandas numpy scikit-learn joblib werkzeug
#      python app.py
#
# UI available at http://127.0.0.1:5000

from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
import time, json, hashlib, random
from typing import List, Dict, Any
import os, uuid, itertools
import csv
from werkzeug.security import generate_password_hash, check_password_hash

# --- Global Configuration and File Paths (Simplified for execution) ---
CREDENTIALS_FILE = r"C:\Users\Dell\Downloads\all_five\all_five\User_credentials.xlsx - Sheet1.csv"
MODEL_PATH = r"C:\Users\Dell\Downloads\all_five\all_five\decision_tree_consensus.pkl"
LABEL_ENCODER_PATH = r"C:\Users\Dell\Downloads\all_five\all_five\label_encoder.pkl"
DATA_PATH = r"C:\Users\Dell\Downloads\all_five\all_five\blockchain_traffic_trafficsim.csv"

NETWORK_STATE_FILE = r"C:\Users\Dell\Downloads\all_five\all_five\network_state.json" # New file for state persistence
USERS: Dict[str, str] = {} # {username: hashed_password}
MAX_NODES = 5

# --- Credential Management Functions (Unchanged) ---
def load_users_from_csv():
    """Loads users from the CSV file, hashing any plaintext passwords found."""
    global USERS
    USERS = {}
    
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"⚠️ Credential file '{CREDENTIALS_FILE}' not found. Starting fresh.")
        return

    try:
        with open(CREDENTIALS_FILE, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                return

            if header == ['Username', 'HashedPassword']:
                for row in reader:
                    if len(row) == 2:
                        USERS[row[0]] = row[1]
                print(f"✅ Loaded {len(USERS)} users from secure CSV format.")
            elif header == ['Username', 'Password']:
                print("⚠️ Found plaintext passwords. Hashing and rewriting file securely...")
                for row in reader:
                    if len(row) == 2:
                        username, password = row
                        USERS[username] = generate_password_hash(password)
                save_users_to_csv()
                print(f"✅ Hashed and loaded {len(USERS)} users.")
            else:
                print("❌ CSV header format is incorrect. Skipping user load.")

    except Exception as e:
        print(f"❌ Error loading credentials: {e}")

def save_users_to_csv():
    """Writes the current USERS dictionary (hashed passwords) back to the CSV."""
    try:
        with open(CREDENTIALS_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Username', 'HashedPassword'])
            for username, hashed_password in USERS.items():
                writer.writerow([username, hashed_password])
        print(f"💾 Saved {len(USERS)} users to CSV.")
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")

# --- ML Model and Data Setup (Unchanged in logic) ---
try:
    import pandas as pd
    import numpy as np
    import joblib
    
    model_ready = False
    model = None
    label_encoder = None
    df_traffic = None
    
    ML_EXPECTED_FEATURES = [
        'node_count', 
        'network_latency_ms', 
        'tx_throughput_tps', 
        'energy_joules_per_min', 
        'security_risk_score', 
        'vehicle_count', 
        'vehicle_count_variability', 
        'fault_tolerance_requirement'
    ]
    MODEL_FEATURES = []

    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(DATA_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            df_traffic = pd.read_csv(DATA_PATH)
            MODEL_FEATURES = ML_EXPECTED_FEATURES
            
            if not all(feature in df_traffic.columns for feature in ML_EXPECTED_FEATURES):
                print("❌ Error: Loaded data is missing one or more expected ML features.")
            else:
                model_ready = True
                print("🧠 ML Model and data loaded successfully. Dynamic Consensus Active.")
        except Exception as e:
            print(f"⚠️ Failed to load ML assets: {e}")
    else:
        print("⚠️ ML files not found. Running in static consensus mode.")

except Exception as e:
    print(f"⚠️ ML library imports failed. Error: {e}")
    model_ready = False
    pd = None
    np = None
    joblib = None


app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Initial Load of Users ---
load_users_from_csv()

# -----------------------------
# Data Structures and Utilities (Blockchain Logic)
# -----------------------------
def compute_hash(block_dict):
    """Computes the SHA256 hash of a block dictionary."""
    s = json.dumps(block_dict, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

class Block:
    """Represents a single block in the blockchain."""
    def __init__(self, index, timestamp, transactions, previous_hash, nonce=0, proposer=None):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.proposer = proposer or 'Genesis'

    def to_dict(self):
        """Returns a dictionary representation of the block."""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'proposer': self.proposer,
        }

    def hash(self):
        """Calculates the hash of the block."""
        return compute_hash(self.to_dict())

class Network:
    """Simulates the multi-node blockchain network with governance."""
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.nodes: Dict[str, Dict] = {}
        self.consensus_mode: str = 'PoW'
        self.difficulty: int = 3
        self.consensus_algos: List[str] = ['PoW', 'PoS', 'Raft', 'PBFT', 'HotStuff']
        self.active_users: Dict[str, bool] = {} # {username: True}
        self.pending_users: Dict[str, List[str]] = {} # {username: [voter1, voter2, ...]}
        self.add_node('Node-A', is_initial=True)
        self.create_genesis_block()

    def create_genesis_block(self):
        if not self.chain:
            self.chain.append(Block(
                index=0,
                timestamp=time.time(),
                transactions=[{'message': 'Genesis Block', 'predicted_consensus': 'PoW'}],
                previous_hash="0" * 64
            ))

    def get_last_block(self) -> Block:
        return self.chain[-1]
    
    # --- Other blockchain methods (run_pow, add_transaction, etc.) remain the same ---
    # (Removed for brevity, but they are fully included in the final file below)
    
    def add_node(self, name, stake=100, is_initial=False):
        if len(self.nodes) >= MAX_NODES and not is_initial:
            return False, "Max nodes reached (5)."

        if name in self.nodes:
            return False, f"Node {name} already exists."

        self.nodes[name] = {
            'stake': stake,
            'address': str(uuid.uuid4()),
            'chain_length': len(self.chain)
        }
        if is_initial:
             # Ensure the initial node is the only one if starting fresh
             self.nodes = {name: self.nodes[name]}
        return True, f"Node {name} added with stake {stake}."

    def remove_node(self, name):
        if name in self.nodes:
            del self.nodes[name]
            return True
        return False

    def add_transaction(self, sender, recipient, amount, fee=0):
        if amount <= 0:
            return False, "Amount must be positive."
        tx = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
            'fee': fee,
            'timestamp': time.time(),
            'id': str(uuid.uuid4())
        }
        self.pending_transactions.append(tx)
        return True, tx

    def set_consensus(self, mode):
        if mode in self.consensus_algos:
            self.consensus_mode = mode
            return True
        return False

    def run_pow(self, proposer_node: str) -> Block:
        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'PoW'
        while True:
            new_block.nonce += 1
            hash_attempt = new_block.hash()
            if hash_attempt.startswith('0' * self.difficulty):
                self.pending_transactions = []
                self.chain.append(new_block)
                return new_block
            if new_block.nonce > 100000:
                 new_block.nonce = 0
                 new_block.timestamp = time.time()

    def run_pos(self, proposer_node: str) -> Block:
        stakes = [self.nodes[name]['stake'] for name in self.nodes]
        total_stake = sum(stakes)
        if total_stake == 0:
             validator = proposer_node
        else:
             weights = [s / total_stake for s in stakes]
             validator = random.choices(list(self.nodes.keys()), weights=weights, k=1)[0]

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=last_block.hash(),
            proposer=validator
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'PoS'
             
        self.pending_transactions = []
        self.chain.append(new_block)
        return new_block

    def run_pbft(self, proposer_node: str) -> Block:
        if len(self.nodes) < 4:
            return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'PBFT'

        quorum = int(2 * len(self.nodes) / 3) + 1
        votes = {node: random.choice([True, False]) for node in self.nodes if node != proposer_node}
        commit_count = sum(votes.values())

        if commit_count >= quorum:
            self.pending_transactions = []
            self.chain.append(new_block)
            return new_block
        else:
            return None

    def run_raft(self, proposer_node: str) -> Block:
        leader = random.choice(list(self.nodes.keys()))
        if leader != proposer_node:
             return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=last_block.hash(),
            proposer=leader
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'Raft'

        self.pending_transactions = []
        self.chain.append(new_block)
        return new_block

    def run_hotstuff(self, proposer_node: str) -> Block:
        if len(self.nodes) < 3:
             return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'HotStuff'

        self.pending_transactions = []
        self.chain.append(new_block)
        return new_block


    def run_consensus(self):
        if not self.pending_transactions:
            return {'success': False, 'result': 'No pending transactions to process.'}

        proposer_node = random.choice(list(self.nodes.keys()))
        new_block = None

        if self.consensus_mode == 'PoW':
            new_block = self.run_pow(proposer_node)
        elif self.consensus_mode == 'PoS':
            new_block = self.run_pos(proposer_node)
        elif self.consensus_mode == 'PBFT':
            new_block = self.run_pbft(proposer_node)
        elif self.consensus_mode == 'Raft':
            new_block = self.run_raft(proposer_node)
        elif self.consensus_mode == 'HotStuff':
            new_block = self.run_hotstuff(proposer_node)

        if new_block:
            for node in self.nodes.values():
                 node['chain_length'] = len(self.chain)

            return {'success': True, 'result': f'Block {new_block.index} forged via {self.consensus_mode}'}
        else:
            return {'success': False, 'result': f'Consensus failed for {self.consensus_mode}.'}


    def resolve_conflicts(self):
        longest_chain_length = max(node['chain_length'] for node in self.nodes.values())
        if len(self.chain) < longest_chain_length:
            return {'message': 'Conflict detected, but resolution logic skipped for simulation.'}
        else:
            return {'message': 'Local chain is the longest. No conflict.'}

# --- State Persistence Functions for Multi-User Support ---

def save_network_state():
    """Serializes and saves the network object's state to disk."""
    global network
    try:
        # Simplify the network state for JSON serialization
        state = {
            'chain': [block.to_dict() for block in network.chain],
            'pending_transactions': network.pending_transactions,
            'nodes': network.nodes,
            'consensus_mode': network.consensus_mode,
            'difficulty': network.difficulty,
            'consensus_algos': network.consensus_algos,
            'active_users': network.active_users,
            'pending_users': network.pending_users,
        }
        with open(NETWORK_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        # print("💾 Network state saved.") # Disabled for cleaner console
    except Exception as e:
        print(f"❌ Error saving network state: {e}")

def load_network_state():
    """Loads and deserializes the network object's state or initializes default."""
    global network
    network = Network() # Initialize a fresh network object first
    
    if os.path.exists(NETWORK_STATE_FILE):
        try:
            with open(NETWORK_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Rebuild chain (must recreate Block objects from dicts)
            network.chain = []
            for block_data in state.get('chain', []):
                block = Block(
                    index=block_data['index'],
                    timestamp=block_data['timestamp'],
                    transactions=block_data['transactions'],
                    previous_hash=block_data['previous_hash'],
                    nonce=block_data.get('nonce', 0),
                    proposer=block_data.get('proposer')
                )
                network.chain.append(block)

            # Restore simple properties
            network.pending_transactions = state.get('pending_transactions', [])
            network.nodes = state.get('nodes', network.nodes) # Keep Node-A if state is missing nodes
            network.consensus_mode = state.get('consensus_mode', 'PoW')
            network.difficulty = state.get('difficulty', 3)
            network.active_users = state.get('active_users', {})
            network.pending_users = state.get('pending_users', {})
            
            # Re-ensure genesis block if chain was empty or failed to load
            if not network.chain:
                 network.create_genesis_block()

            # print("✅ Network state loaded from file.") # Disabled for cleaner console
            
        except Exception as e:
            print(f"❌ Error loading network state: {e}. Reinitializing network.")
            network = Network() # Reset on failure

    ensure_default_users()

def ensure_default_users():
    """Ensures two users are always active and handles initial state."""
    
    # 1. Ensure minimum two users in credentials
    default_users = {'Sengathir': '12245', 'UserB': 'defaultpass'}
    changed_credentials = False

    for username, password in default_users.items():
        if username not in USERS:
            USERS[username] = generate_password_hash(password)
            changed_credentials = True
    
    if changed_credentials:
        save_users_to_csv()
    
    # 2. Ensure minimum two users are active in the network state
    if len(network.active_users) < 2:
        for username in default_users.keys():
            if username in USERS:
                network.active_users[username] = True
    
    # Prune active users who are no longer in USERS or are now pending
    users_to_remove = [u for u in network.active_users if u not in USERS or u in network.pending_users]
    for u in users_to_remove:
         del network.active_users[u]

    # Save state if defaults were added
    if len(network.active_users) >= 2:
         save_network_state()
# -------------------------------------
# Application Setup and Route Wrappers
# -------------------------------------

# Initialize the network simulation and state
network = None 
load_network_state() # Will also ensure default users

# A simple wrapper to handle persistence for API/Action routes
def persist_action(func):
    def wrapper(*args, **kwargs):
        load_network_state()
        response = func(*args, **kwargs)
        save_network_state()
        return response
    wrapper.__name__ = func.__name__ + '_persisted'
    return wrapper


@app.before_request
def check_login():
    """Checks if the user is logged in before allowing access to app routes."""
    # Exclude login, signup, and static files
    if request.path in ['/login', '/signup', '/'] or request.path.startswith('/static'):
        return
    if 'logged_in' not in session:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login using CSV-backed credentials and community approval."""
    load_network_state() # Load state for checking active users
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS:
            if check_password_hash(USERS[username], password):
                if username in network.active_users:
                    session['logged_in'] = True
                    session['username'] = username
                    return redirect(url_for('index'))
                else:
                    message = "Login successful, but your account is **PENDING COMMUNITY APPROVAL**. Check back later."
            else:
                message = "Invalid password."
        else:
            message = "Username not found. Please sign up."
    
    save_network_state() # Save state if network loaded correctly (no functional change here)
    return render_template_string(LOGIN_HTML, message=message, is_login=True)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles new user sign up, placing them in the PENDING list."""
    load_network_state() # Load state for checking existing users
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            message = "Username and password are required."
        elif username in USERS:
            message = f"Username '{username}' already exists."
        else:
            # 1. Hash the password and save the new user credential
            hashed_password = generate_password_hash(password)
            USERS[username] = hashed_password
            save_users_to_csv()
            
            # 2. Add user to pending list (initializes with an empty voter list)
            network.pending_users[username] = []
            save_network_state() # Save network state with new pending user
            
            message = f"Sign up successful! Your account, '{username}', is pending **COMMUNITY APPROVAL**. Please check back later."
            return render_template_string(LOGIN_HTML, message=message, is_login=True)

    return render_template_string(LOGIN_HTML, message=message, is_login=False)


@app.route('/logout')
def logout():
    """Logs the user out."""
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    """The main application page."""
    load_network_state() # Load state before rendering
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template_string(INDEX_HTML)

# --- Simulation API Routes ---

@app.route('/api/state')
def get_state():
    load_network_state()
    last_block = network.get_last_block()
    return jsonify({
        'chain': [block.to_dict() for block in network.chain],
        'pending_transactions': network.pending_transactions,
        'nodes': network.nodes,
        'consensus_mode': network.consensus_mode,
        'difficulty': network.difficulty,
        'consensus_algos': network.consensus_algos,
        'ml_ready': model_ready,
        'last_block_hash': last_block.hash(),
        'node_count': len(network.nodes),
        'current_user': session.get('username', 'Guest')
    })

# --- New Governance API Routes ---

@app.route('/api/get_users')
def get_users():
    load_network_state()
    active_count = len(network.active_users)
    majority_threshold = int(active_count / 2) + 1
    
    # Prepare pending list for UI
    pending_list = []
    for user, voters in network.pending_users.items():
        pending_list.append({
            'username': user,
            'voters': voters,
            'vote_count': len(voters),
            'voted_by_current': session.get('username') in voters
        })

    return jsonify({
        'active_users': list(network.active_users.keys()),
        'pending_users': pending_list,
        'active_count': active_count,
        'majority_threshold': majority_threshold,
    })


@app.route('/api/vote_user', methods=['POST'])
@persist_action
def vote_user():
    voter = session.get('username')
    data = request.json
    target_user = data.get('username')

    if target_user not in network.pending_users:
        return jsonify({'success': False, 'message': 'User not found in pending list.'})
    if voter not in network.active_users:
        return jsonify({'success': False, 'message': 'Only active users can vote.'})
    
    current_voters = network.pending_users.get(target_user, [])
    if voter in current_voters:
        return jsonify({'success': False, 'message': f'You have already voted for {target_user}.'})

    current_voters.append(voter)
    
    active_count = len(network.active_users)
    majority_threshold = int(active_count / 2) + 1
    current_votes = len(current_voters)

    if current_votes >= majority_threshold:
        # Approval granted
        network.active_users[target_user] = True
        del network.pending_users[target_user]
        message = f"User '{target_user}' approved by majority ({current_votes}/{active_count})! Now active."
        approved = True
    else:
        message = f"Vote cast for '{target_user}'. Current votes: {current_votes}. Threshold: {majority_threshold}."
        approved = False

    return jsonify({'success': True, 'message': message, 'approved': approved})


# --- Persisted Simulation API Routes ---

@app.route('/api/add_node', methods=['POST'])
@persist_action
def add_node():
    data = request.json
    name = data.get('name')
    stake = int(data.get('stake', 100))
    if not name:
        name = 'Node-' + ''.join(random.choices('BCDEF', k=1)) + str(len(network.nodes) + 1)
    success, message = network.add_node(name, stake)
    return jsonify({'success': success, 'message': message})

@app.route('/api/remove_node', methods=['POST'])
@persist_action
def remove_node():
    data = request.json
    name = data.get('name')
    if name not in network.nodes:
        return jsonify({'success': False, 'message': f"Node {name} not found."})
    if len(network.nodes) <= 1:
        return jsonify({'success': False, 'message': "Cannot remove the last node."})
    success = network.remove_node(name)
    return jsonify({'success': success, 'message': f"Node {name} removed."})

@app.route('/api/add_tx', methods=['POST'])
@persist_action
def add_tx():
    data = request.json
    sender = data.get('sender', session.get('username', 'Anonymous'))
    recipient = data.get('recipient', 'Node-A')
    amount = data.get('amount', 10)
    fee = data.get('fee', 1)

    success, result = network.add_transaction(sender, recipient, amount, fee)
    if not success:
        return jsonify({'success': False, 'message': result})

    # --- ML Dynamic Consensus Logic ---
    predicted_consensus = network.consensus_mode
    if model_ready:
        try:
            random_row = df_traffic.sample(n=1, random_state=int(time.time() * 1000) % 1000)
            features_df = random_row[MODEL_FEATURES]
            features_array = features_df.values.reshape(1, -1)
            
            prediction_encoded = model.predict(features_array)[0]
            predicted_consensus = label_encoder.inverse_transform([prediction_encoded])[0]

            if predicted_consensus != network.consensus_mode:
                network.set_consensus(predicted_consensus)
        except Exception as e:
            print(f"ML Prediction Error: {e}")
            pass

    return jsonify({
        'success': True,
        'message': 'Transaction added to pending pool.',
        'transaction': result,
        'predicted_consensus': predicted_consensus
    })

@app.route('/api/set_consensus', methods=['POST'])
@persist_action
def set_consensus():
    mode = request.json.get('mode')
    success = network.set_consensus(mode)
    if success:
        return jsonify({'success': True, 'message': f'Consensus set to {mode}.'})
    else:
        return jsonify({'success': False, 'message': f'Invalid consensus mode: {mode}.'})

@app.route('/api/trigger', methods=['POST'])
@persist_action
def trigger_consensus():
    result = network.run_consensus()
    return jsonify(result)

@app.route('/api/resolve', methods=['POST'])
@persist_action
def resolve_conflicts_route():
    result = network.resolve_conflicts()
    return jsonify(result)

# -----------------------------
# HTML TEMPLATES (Aesthetic UI/UX with Governance)
# -----------------------------

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ConcordiaChain Auth</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        body {
            font-family: 'Space Mono', monospace;
            background: #0D1117;
            color: #E6EDF3;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .login-container {
            background: #161B22;
            box-shadow: 0 0 20px #00FFFF33;
        }
        .input-glow:focus {
            border-color: #00FFFF;
            box-shadow: 0 0 5px #00FFFF;
        }
        .btn-primary {
            background: #00AFFF;
            transition: all 0.2s;
        }
        .btn-primary:hover {
            background: #00FFFF;
            box-shadow: 0 0 10px #00FFFF;
        }
    </style>
</head>
<body>
    <div class="login-container p-8 rounded-xl w-96">
        <h1 class="text-3xl font-bold text-center text-[#00FFFF] mb-2">
            CONCORDIA CHAIN
        </h1>
        <p class="text-center text-xl font-bold mb-6 text-gray-300">
            {% if is_login %} ACCESS CONTROL {% else %} NEW USER REGISTRATION {% endif %}
        </p>
        
        <form method="POST" action="{% if is_login %}/login{% else %}/signup{% endif %}">
            <div class="mb-4">
                <label for="username" class="block text-sm font-medium mb-1 text-gray-300">Username</label>
                <input type="text" id="username" name="username" required minlength="3"
                       class="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 input-glow text-white">
            </div>
            <div class="mb-6">
                <label for="password" class="block text-sm font-medium mb-1 text-gray-300">Password</label>
                <input type="password" id="password" name="password" required minlength="5"
                       class="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 input-glow text-white">
            </div>
            <button type="submit" class="btn-primary w-full text-black font-bold py-2 rounded-lg">
                {% if is_login %} LOG IN {% else %} SIGN UP & APPLY {% endif %}
            </button>
        </form>
        
        {% if message %}
        <p class="mt-4 text-center text-red-400 text-sm font-mono">{{ message | safe }}</p>
        {% endif %}

        <div class="mt-6 text-center text-sm">
            {% if is_login %}
            <a href="{{ url_for('signup') }}" class="text-blue-400 hover:text-blue-300">New User? Apply for Access</a>
            {% else %}
            <a href="{{ url_for('login') }}" class="text-blue-400 hover:text-blue-300">Already Active? Log In</a>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ConcordiaChain: Dynamic Consensus Simulator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        body {
            font-family: 'Space Mono', monospace;
            background: #0D1117;
            color: #E6EDF3;
            min-height: 100vh;
        }
        .header-glow {
            text-shadow: 0 0 5px #00FFFF;
        }
        .card {
            background: #161B22;
            border: 1px solid #30363D;
            box-shadow: 0 2px 10px #00000050;
        }
        .btn-action {
            background: #00AFFF;
            color: #0D1117;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-action:hover {
            background: #00FFFF;
            box-shadow: 0 0 8px #00FFFF;
        }
        .log-box {
            background: #000000;
            color: #00FF00;
            border: 1px solid #00FF0033;
            overflow-y: scroll;
            height: 300px;
            font-size: 0.75rem;
        }
        .hash-code {
            font-size: 0.6rem;
            color: #8B949E;
        }
        .consensus-tag {
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        .tag-pow { background: #FFC0CB; color: #8A2BE2; }
        .tag-pos { background: #66FF66; color: #006400; }
        .tag-raft { background: #FFD700; color: #8B4513; }
        .tag-pbft { background: #4169E1; color: #F0F8FF; }
        .tag-hotstuff { background: #FF4500; color: #000000; }

        .block-verified {
            border: 2px solid #00FF00;
            transition: all 0.3s;
        }

        #ml-status {
            animation: pulse-ml 2s infinite;
        }
        @keyframes pulse-ml {
            0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0.4); }
            50% { box-shadow: 0 0 0 8px rgba(0, 255, 255, 0); }
        }
        .vote-btn-pending {
            background: #FFA500;
        }
    </style>
</head>
<body class="p-4 sm:p-8">

    <div class="flex justify-between items-center mb-6 border-b pb-4 border-gray-700">
        <h1 class="text-4xl font-bold header-glow text-[#00FFFF]">
            CONCORDIA CHAIN
        </h1>
        <div class="text-right flex items-center">
            <span class="text-sm mr-4 text-gray-400">User: <span id="current-user" class="text-white font-bold">...</span></span>
            <span id="ml-status" class="px-3 py-1 text-xs rounded-full cursor-default mr-4" style="background: #00FFFF1A; color: #00FFFF;">
                ML: <span id="ml-status-text">Checking...</span>
            </span>
            <a href="/logout" class="text-red-400 hover:text-red-300 text-sm">Logout</a>
        </div>
    </div>

    <!-- ML-Driven Consensus & Global Status -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div class="card p-4 rounded-lg">
            <h2 class="text-lg font-bold text-gray-400 mb-2">CURRENT CONSENSUS</h2>
            <div id="current-consensus" class="text-3xl font-bold text-green-400">PoW</div>
            <p class="text-xs text-gray-500 mt-1">Difficulty: <span id="difficulty">3</span></p>
        </div>
        <div class="card p-4 rounded-lg">
            <h2 class="text-lg font-bold text-gray-400 mb-2">DYNAMIC PREDICTION</h2>
            <div id="predicted-consensus" class="text-3xl font-bold text-yellow-400">PoW (Default)</div>
            <p id="prediction-trigger" class="text-xs text-gray-500 mt-1">Trigger: Manual/Initial</p>
        </div>
        <div class="card p-4 rounded-lg">
            <h2 class="text-lg font-bold text-gray-400 mb-2">ACTIVE USERS</h2>
            <div id="active-user-count" class="text-3xl font-bold text-blue-400">1</div>
            <p class="text-xs text-gray-500 mt-1">Pending Approvals: <span id="pending-user-count" class="font-bold text-red-400">0</span></p>
        </div>
        <div class="card p-4 rounded-lg">
            <h2 class="text-lg font-bold text-gray-400 mb-2">BLOCKCHAIN LENGTH</h2>
            <div id="chain-length" class="text-3xl font-bold text-purple-400">1</div>
            <p class="text-xs text-gray-500 mt-1">Pending TX: <span id="pending-tx-count" class="font-bold text-red-400">0</span></p>
        </div>
    </div>

    <!-- Action Panel -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
        <!-- Node Management -->
        <div class="card p-4 rounded-lg">
            <h2 class="text-xl font-bold mb-3 text-gray-200">NODE MANAGEMENT</h2>
            <div id="node-list" class="flex flex-col gap-1 mb-4">
                <!-- Node list will be rendered here -->
            </div>
            <div class="flex gap-2">
                <input type="number" id="new-stake" placeholder="Stake (PoS)" value="100" class="flex-grow p-2 text-sm rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-[#00FFFF]">
                <button onclick="addNode()" class="btn-action px-4 py-2 rounded-lg text-sm">Add Node</button>
            </div>
        </div>

        <!-- Transaction & Consensus -->
        <div class="card p-4 rounded-lg lg:col-span-2">
            <h2 class="text-xl font-bold mb-3 text-gray-200">TRANSACTION & FORGING</h2>
            <div class="mb-4 flex gap-2">
                <input type="text" id="tx-recipient" placeholder="Recipient (e.g., Node-A)" value="Node-A" class="w-full md:w-1/3 p-2 text-sm rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-[#00FFFF]">
                <input type="number" id="tx-amount" placeholder="Amount" value="10" class="w-full md:w-1/3 p-2 text-sm rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-[#00FFFF]">
                <button onclick="addTx()" class="btn-action w-full md:w-1/3 py-2 rounded-lg text-sm bg-green-500 hover:bg-green-400">Add Transaction (ML Check)</button>
            </div>
            
            <h3 class="text-md font-bold mb-2 text-gray-400">CONSENSUS CONTROL</h3>
            <div class="flex gap-2">
                <select id="consensus" class="flex-grow p-2 text-sm rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-[#00FFFF]">
                    <!-- Options populated by JS -->
                </select>
                <button onclick="triggerConsensus()" class="btn-action w-1/3 py-2 rounded-lg text-sm">Forge Block</button>
            </div>
        </div>

        <!-- Governance Control -->
        <div class="card p-4 rounded-lg">
            <h2 class="text-xl font-bold mb-3 text-gray-200">COMMUNITY GOVERNANCE</h2>
            <div id="governance-view">
                <!-- Pending user approvals go here -->
            </div>
            <button onclick="resolve()" class="btn-action w-full py-2 rounded-lg text-sm mt-4 bg-purple-500 hover:bg-purple-400">Check Chain Availability</button>
        </div>
    </div>

    <!-- Ledger & Logs -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Block Ledger -->
        <div class="card p-4 rounded-lg">
            <h2 class="text-xl font-bold mb-3 text-gray-200">BLOCK LEDGER <span class="text-xs text-gray-500 ml-2">(Integrity & Authority)</span></h2>
            <div id="chain-view" class="log-box">
                <!-- Block data will be rendered here -->
            </div>
        </div>

        <!-- Pending Transactions -->
        <div class="card p-4 rounded-lg">
            <h2 class="text-xl font-bold mb-3 text-gray-200">PENDING TRANSACTIONS <span class="text-xs text-gray-500 ml-2">(Confidentiality/Queue)</span></h2>
            <div id="pending-tx-view" class="log-box">
                <!-- Pending TX data will be rendered here -->
            </div>
        </div>
    </div>

<script>
// Prevent use of alert() as per instructions, use custom modal/message box
function showMessage(title, message, isError = false) {
    const color = isError ? 'text-red-400' : 'text-[#00FFFF]';
    const bg = isError ? 'bg-red-900/50' : 'bg-[#00FFFF1A]';
    
    const msgBox = document.createElement('div');
    msgBox.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-xl ${bg} ${color} border border-current transition-opacity duration-300`;
    msgBox.innerHTML = `<h3 class="font-bold mb-1">${title}</h3><p class="text-sm">${message}</p>`;
    
    document.body.appendChild(msgBox);
    
    setTimeout(() => {
        msgBox.style.opacity = '0';
        setTimeout(() => msgBox.remove(), 300);
    }, 4000);
}

// Global state variables
let currentState = {};
let currentUsers = {};

function formatHash(hash) {
    return `${hash.substring(0, 8)}...${hash.substring(hash.length - 8)}`;
}

function getConsensusTag(mode) {
    const modeClass = mode.toLowerCase().replace(/[^a-z]/g, '');
    return `<span class="consensus-tag tag-${modeClass}">${mode}</span>`;
}

// --- RENDERING FUNCTIONS ---

function renderMLStatus(mlReady) {
    const mlStatusEl = document.getElementById('ml-status-text');
    if (mlReady) {
        mlStatusEl.textContent = 'ACTIVE';
        mlStatusEl.parentNode.style.backgroundColor = '#00FFFF1A';
        mlStatusEl.parentNode.style.color = '#00FFFF';
    } else {
        mlStatusEl.textContent = 'INACTIVE (Static)';
        mlStatusEl.parentNode.style.backgroundColor = '#FF00001A';
        mlStatusEl.parentNode.style.color = '#FF0000';
    }
}

function renderNodeList(nodes) {
    const listEl = document.getElementById('node-list');
    
    let html = `
        <div class="flex justify-between font-bold text-gray-400 border-b border-gray-700 pb-1 text-xs mb-1">
            <span>Node</span>
            <span>Stake</span>
            <span>Remove</span>
        </div>
    `;

    for (const name in nodes) {
        const node = nodes[name];
        html += `
            <div class="flex justify-between items-center w-full py-1 border-b border-gray-800 text-sm">
                <span class="text-white">${name}</span>
                <span class="text-yellow-300">${node.stake}</span>
                <button onclick="removeNode('${name}')" class="text-red-500 hover:text-red-400 text-xs">
                    [x]
                </button>
            </div>
        `;
    }
    listEl.innerHTML = html;
}

function renderConsensusDropdown(algos, currentMode) {
    const selectEl = document.getElementById('consensus');
    selectEl.innerHTML = algos.map(mode =>
        `<option value="${mode}" ${mode === currentMode ? 'selected' : ''}>${mode}</option>`
    ).join('');
}

function renderChain(chain, difficulty) {
    const chainViewEl = document.getElementById('chain-view');
    chainViewEl.innerHTML = '';
    document.getElementById('chain-length').textContent = chain.length;

    chain.slice().reverse().forEach(block => {
        
        const blockHash = block.hash || formatHash(block.previous_hash + block.index); 
        
        const isVerified = blockHash.startsWith('0'.repeat(difficulty));
        const verifiedClass = isVerified ? 'block-verified' : '';

        const proposer = block.proposer || 'N/A';
        const proposerHtml = `<span class="text-sm font-bold text-cyan-400">${proposer}</span>`;
        // Safely extract the consensus from the first transaction if available
        const blockConsensus = block.transactions[0]?.predicted_consensus || 'N/A';

        chainViewEl.innerHTML += `
            <div class="p-3 mb-3 rounded-lg card ${verifiedClass}">
                <div class="flex justify-between items-start mb-1">
                    <span class="text-lg font-bold text-blue-300">BLOCK #${block.index}</span>
                    <span class="text-xs text-gray-500">${new Date(block.timestamp * 1000).toLocaleTimeString()}</span>
                </div>
                <div class="mb-1 text-sm">
                    Authority: ${proposerHtml} (Mode: ${getConsensusTag(blockConsensus)})
                </div>
                <div class="hash-code mb-1 break-all">
                    Hash: ${blockHash} <span class="text-yellow-400">${isVerified ? '(PoW Proof)' : ''}</span>
                </div>
                <div class="hash-code break-all">
                    Prev Hash: ${formatHash(block.previous_hash)}
                </div>
                <div class="mt-2 text-xs">
                    <span class="font-bold text-gray-400">Transactions:</span> ${block.transactions.length}
                </div>
            </div>
        `;
    });
}

function renderPendingTx(txList) {
    const pendingTxViewEl = document.getElementById('pending-tx-view');
    pendingTxViewEl.innerHTML = '';
    
    if (txList.length === 0) {
        pendingTxViewEl.innerHTML = '<p class="text-center text-gray-600 p-4">Transaction queue is empty.</p>';
        return;
    }

    txList.forEach(tx => {
        pendingTxViewEl.innerHTML += `
            <div class="p-3 mb-2 rounded-lg bg-gray-900 border-l-4 border-red-500">
                <span class="font-bold text-red-300">TX ID: ${formatHash(tx.id)}</span>
                <div class="text-xs mt-1">
                    <span class="text-gray-400">Sender:</span> ${tx.sender} <br>
                    <span class="text-gray-400">Recipient:</span> ${tx.recipient} <br>
                    <span class="text-gray-400">Amount (Confidential):</span> ${tx.amount}
                </div>
            </div>
        `;
    });
}

function renderGovernance(users) {
    const governanceEl = document.getElementById('governance-view');
    document.getElementById('active-user-count').textContent = users.active_count;
    document.getElementById('pending-user-count').textContent = users.pending_users.length;
    
    let html = '';

    if (users.pending_users.length === 0) {
        html = '<p class="text-sm text-gray-500">No pending sign-ups.</p>';
    } else {
        html += `<p class="text-xs text-yellow-400 mb-2">Pending Approvals (${users.pending_users.length}). Majority needed: ${users.majority_threshold}</p>`;
        
        users.pending_users.forEach(pUser => {
            const voteStatus = pUser.voted_by_current ? 'VOTED' : 'VOTE';
            const buttonClass = pUser.voted_by_current ? 'bg-gray-600 cursor-not-allowed' : 'vote-btn-pending btn-action';
            const disabled = pUser.voted_by_current ? 'disabled' : '';

            html += `
                <div class="flex justify-between items-center border-b border-gray-700 py-2 text-sm">
                    <span class="text-white">${pUser.username}</span>
                    <span class="text-gray-400">${pUser.vote_count}/${users.active_count}</span>
                    <button onclick="voteUser('${pUser.username}')" 
                            class="text-xs text-black font-bold px-3 py-1 rounded ${buttonClass}" 
                            ${disabled}>
                        ${voteStatus}
                    </button>
                </div>
            `;
        });
    }

    governanceEl.innerHTML = html;
}

// --- API CALLS & REFRESH ---

async function refresh() {
    // 1. Fetch Blockchain State
    const r = await fetch('/api/state');
    currentState = await r.json();

    // 2. Fetch User/Governance State
    const rUsers = await fetch('/api/get_users');
    currentUsers = await rUsers.json();


    // Update Global Status & User
    document.getElementById('current-user').textContent = currentState.current_user;
    document.getElementById('current-consensus').innerHTML = getConsensusTag(currentState.consensus_mode);
    document.getElementById('difficulty').textContent = currentState.difficulty;
    document.getElementById('pending-tx-count').textContent = currentState.pending_transactions.length;

    // Render Components
    renderMLStatus(currentState.ml_ready);
    renderNodeList(currentState.nodes);
    renderConsensusDropdown(currentState.consensus_algos, currentState.consensus_mode);
    renderChain(currentState.chain, currentState.difficulty);
    renderPendingTx(currentState.pending_transactions);
    renderGovernance(currentUsers);
}

// Global variable to hold the last predicted consensus after a transaction
let lastPredictedConsensus = currentState.consensus_mode || 'PoW';

async function addNode() {
    const stake = document.getElementById('new-stake').value;
    const body = { stake: parseInt(stake) || 100 };
    
    const r = await fetch('/api/add_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    const j = await r.json();
    if (j.success) {
        showMessage('Node Status', j.message);
    } else {
        showMessage('Node Error', j.message, true);
    }
    refresh();
}

async function removeNode(name) {
    const r = await fetch('/api/remove_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    const j = await r.json();
    if (j.success) {
        showMessage('Node Status', `${name} removed.`);
    } else {
        showMessage('Node Error', j.message, true);
    }
    refresh();
}

async function addTx() {
    const recipient = document.getElementById('tx-recipient').value;
    const amount = document.getElementById('tx-amount').value;
    const body = { recipient, amount: parseInt(amount) || 10 };
    
    const r = await fetch('/api/add_tx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    const j = await r.json();

    if (j.success) {
        let msg = j.message;
        
        if (j.predicted_consensus && j.predicted_consensus !== currentState.consensus_mode) {
             msg += ` | 🧠 ML Switch: New mode is ${j.predicted_consensus}!`;
             lastPredictedConsensus = j.predicted_consensus;
             document.getElementById('predicted-consensus').innerHTML = getConsensusTag(j.predicted_consensus);
             document.getElementById('prediction-trigger').textContent = 'Trigger: Transaction added/Metrics changed';
        } else {
             document.getElementById('predicted-consensus').innerHTML = getConsensusTag(currentState.consensus_mode) + ' (No Switch)';
             document.getElementById('prediction-trigger').textContent = 'Trigger: Metrics Stable';
        }
        
        showMessage('Transaction Status', msg);
    } else {
        showMessage('Transaction Error', j.message, true);
    }
    refresh();
}

async function triggerConsensus() {
    const mode = document.getElementById('consensus').value;
    
    const set_r = await fetch('/api/set_consensus', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ mode })
    });
    
    const r = await fetch('/api/trigger', { method: 'POST' });
    const j = await r.json();
    
    if (!j.success) {
        showMessage('Consensus Failed', j.result, true);
    } else {
        showMessage('Consensus Success', j.result);
    }
    refresh();
}

async function resolve() {
    const r = await fetch('/api/resolve', { method: 'POST' });
    const j = await r.json();
    showMessage('Conflict Resolution (Availability)', j.message);
    refresh();
}

async function voteUser(username) {
    const r = await fetch('/api/vote_user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
    });
    const j = await r.json();

    if (j.success) {
        showMessage('Governance Vote', j.message, j.approved);
    } else {
        showMessage('Vote Error', j.message, true);
    }
    refresh();
}


// Event listener for manual consensus change
document.getElementById('consensus').addEventListener('change', async function(){
  const mode = this.value;
  await fetch('/api/set_consensus', {
      method:'POST', 
      headers:{'content-type':'application/json'}, 
      body: JSON.stringify({mode})
  });
  showMessage('Manual Override', `Consensus mode manually set to ${mode}.`);
  refresh();
});

// Initial load and auto-refresh
refresh();
setInterval(refresh, 4000); // Poll for state every 4 seconds
</script>

</body>
</html>
"""
if __name__ == '__main__':
    # Add a couple of initial nodes for a better start
    node_names = ['Node-B', 'Node-C']
    for name in node_names:
        network.add_node(name, stake=200 + random.randint(0, 100))

    print("\n--- ConcordiaChain Simulator ---")
    print(f"Credentials loaded from: {CREDENTIALS_FILE}")
    print(f"Total Users: {len(USERS)}")
    app.run(debug=True, use_reloader=False)