
# ConcordiaChain: ML-Driven Dynamic Consensus Blockchain Simulator

**ConcordiaChain** is a machine learning–assisted blockchain simulator developed using **Flask**, designed to demonstrate how an adaptive blockchain network can dynamically transition between consensus algorithms based on real-time simulated network metrics.

This single-file simulator is built for **research**, **academic study**, and **rapid experimentation** in blockchain governance, consensus modeling, and hybrid AI-driven decision systems.

---

## Overview

ConcordiaChain simulates a **multi-node blockchain network** capable of switching among multiple consensus mechanisms, including **Proof of Work (PoW)**, **Proof of Stake (PoS)**, **Raft**, **PBFT**, and **HotStuff**. The simulator can operate either in static mode or dynamically through an integrated machine learning model that predicts the most suitable consensus algorithm given the current network conditions.

In addition to consensus adaptability, the simulator includes:

* A **TailwindCSS-based web interface** for real-time network visualization.
* A **governance module** that allows registered users to vote on new member approvals.
* A **REST API layer** for headless operation, testing, and automation.

This project is aimed at developers, students, and researchers exploring **adaptive consensus mechanisms**, **distributed governance frameworks**, and **AI-assisted blockchain architectures**.

---

## Core Features

**1. Multi-Consensus Simulation**
Simulates and transitions between five major consensus mechanisms: PoW, PoS, Raft, PBFT, and HotStuff.

**2. Dynamic Consensus Selection via Machine Learning**
Integrates a Decision Tree Classifier trained on simulated traffic and latency data to predict the optimal consensus mechanism for the current network state.

**3. Web Interface Built with Flask and TailwindCSS**
Provides a real-time, responsive visualization of the blockchain ledger, nodes, transactions, and governance status.

**4. Governance and Voting System**
Implements a decentralized user approval process where existing users vote to validate new accounts.

**5. Persistent State Management**
Stores all blockchain data, node configurations, and user credentials locally using JSON and CSV for continuous simulation state preservation.

**6. REST API for Automation**
Offers complete control over transactions, nodes, and consensus operations through REST endpoints, suitable for integration with external tools or scripts.

---

## Architecture

```
 ┌─────────────────────────────┐
 │          Flask Server       │
 ├──────────────┬──────────────┤
 │   Web UI     │   API Layer  │
 │ (Tailwind)   │   (/api/*)   │
 └──────┬───────┴──────────────┘
        │
 ┌──────▼───────────────────────────────┐
 │        Blockchain Engine             │
 │ - Transaction Pool                   │
 │ - Block Creation and Validation      │
 │ - Node Registry                      │
 │ - Governance and Voting              │
 └──────┬───────────────────────────────┘
        │
 ┌──────▼───────────────────────────────┐
 │     Machine Learning Module          │
 │ - Decision Tree Classifier           │
 │ - Label Encoding for Consensus Modes │
 │ - Dynamic Mode Switching             │
 └──────────────────────────────────────┘
```

---

## System Requirements

* **Python:** Version 3.8 or higher (Recommended: 3.10 or 3.11)
* **Dependencies:**

  * Flask
  * pandas
  * numpy
  * scikit-learn
  * joblib
  * werkzeug

---

## Installation

Install the necessary dependencies using pip:

```bash
python -m pip install --upgrade pip
pip install flask pandas numpy scikit-learn joblib werkzeug
```

You may also use a pre-defined `requirements.txt` for reproducible installations.

---

## Running the Simulator

1. **Navigate to the project directory:**

   ```bash
   cd "C:\Users\YourSystem\Blockchain_DCA-Dynamic-consensus-\"
   ```

2. **Run the Flask application:**

   ```bash
   python app1.py
   ```

3. **Access the web interface:**

   Open your browser and go to:

   ```
   http://127.0.0.1:5000
   ```

   The Flask console displays information about credential file paths, total user count, and model status during startup.

---

## Directory Structure

```
Blockchain_DCA-Dynaminc-consensus-/
│
├── app1.py                         # Main Flask application
├── app3.py, dynamic_algo.py # Experimental versions
├── blockchain_traffic_trafficsim.csv # Dataset for ML training
├── decision_tree_consensus.pkl      # Trained ML model (optional)
├── label_encoder.pkl                # Encoded consensus label mapping
├── User_credentials.xlsx - Sheet1.csv # Credential file (auto-generated if missing)
├── network_state.json               # Blockchain and network state file
└── readme.md                        # Documentation
```

---

## User Interface Screens

![Login Page](https://raw.githubusercontent.com/Sengathir24/Blockchain_DCA-Dynamic-consensus-/main/Screenshots/Login_page.jpg)
*Login interface for user authentication.*

![Ledger and Pending Transactions](https://raw.githubusercontent.com/Sengathir24/Blockchain_DCA-Dynamic-consensus-/main/Screenshots/Ledger_pending.jpg)
*Ledger and pending transactions display.*

![Main Dashboard](https://raw.githubusercontent.com/Sengathir24/Blockchain_DCA-Dynamic-consensus-/main/Screenshots/Main_page.jpg)
*Main network dashboard.*

![Notification Banner](https://raw.githubusercontent.com/Sengathir24/Blockchain_DCA-Dynamic-consensus-/main/Screenshots/Notification_top.jpg)
*Top notification and system alerts.*

---

## Authentication and Governance

* **Credential Management:** User data is securely stored in CSV format with passwords hashed using Werkzeug’s security functions.
* **New User Approval:** New sign-ups enter a pending queue, awaiting validation through user voting.
* **Voting Process:** Each active user can vote to approve pending registrations.
* **Governance Quorum:** The system initializes with at least two active users to enable governance functionality.

---

## API Endpoints

| Method | Endpoint             | Description                                                         |
| ------ | -------------------- | ------------------------------------------------------------------- |
| GET    | `/api/state`         | Returns the full blockchain state including nodes and transactions. |
| GET    | `/api/get_users`     | Retrieves active and pending user lists.                            |
| POST   | `/api/vote_user`     | Approves a pending user. Example: `{ "username": "alice" }`         |
| POST   | `/api/add_tx`        | Adds a transaction and optionally triggers ML prediction.           |
| POST   | `/api/trigger`       | Initiates block creation based on the active consensus mode.        |
| POST   | `/api/add_node`      | Registers a new node. Example: `{ "name": "Node1", "stake": 10 }`   |
| POST   | `/api/remove_node`   | Removes a node by name. Example: `{ "name": "Node1" }`              |
| POST   | `/api/set_consensus` | Manually sets consensus mode. Example: `{ "mode": "PoS" }`          |

---

## Dynamic Consensus with Machine Learning

The integrated machine learning module can automatically adjust the consensus algorithm based on simulated network data such as:

* Transaction throughput
* Node latency
* Stake distribution
* Network activity levels

If the ML model determines that a different consensus mechanism would improve performance, the simulator seamlessly transitions to that mode.

**Required Files:**

* `decision_tree_consensus.pkl` — Trained Decision Tree Classifier
* `label_encoder.pkl` — Encoded label mapping for consensus types
* `blockchain_traffic_trafficsim.csv` — Input dataset used during model training

If these files are not found, the simulator runs in static consensus mode.

---

## Configuration

Edit the following constants in `app3.py` to match your environment:

| Variable             | Description               | Default Path                                                                   |
| -------------------- | ------------------------- | ------------------------------------------------------------------------------ |
| `CREDENTIALS_FILE`   | User credentials CSV file | `C:\Users\Dell\Downloads\all_five\all_five\User_credentials.xlsx - Sheet1.csv` |
| `MODEL_PATH`         | ML model path             | `decision_tree_consensus.pkl`                                                  |
| `LABEL_ENCODER_PATH` | Label encoder path        | `label_encoder.pkl`                                                            |
| `NETWORK_STATE_FILE` | Network persistence file  | `network_state.json`                                                           |

---

## Troubleshooting

* **Module Import Errors:** Verify that all required packages are installed.
* **Missing Files:** Ensure all specified paths in `app3.py` are correct.
* **Port Conflicts:** Change the Flask port using `app.run(port=5001)`.
* **CSV Errors:** Delete malformed CSV files; the simulator recreates them automatically.
* **ML Model Errors:** If the model cannot be loaded, the app automatically switches to static consensus mode.

---

## Security Practices

* All stored passwords are hashed for security.
* Sensitive or personal data should not be committed to public repositories.
* The development server should only be used locally and not exposed to production environments.

---

## Development Notes

* You may rename or relocate the credential file by editing `CREDENTIALS_FILE` in `app3.py`.
* Training a custom ML model can improve prediction accuracy for consensus selection.
* The simulator can be integrated with external dashboards or applications via the provided API endpoints.

---

## Planned Enhancements

Future updates may include:

* Reinforcement Learning–based adaptive consensus mechanisms.
* Real-time dashboards for node health and consensus state visualization.
* Advanced governance mechanisms with weighted voting systems.
* Federated learning integration for decentralized ML model training.
* Automated unit testing and performance benchmarking tools.

---

## License and Attribution

This project is open source and intended solely for educational and demonstration purposes.
You are welcome to use, modify, or redistribute it with appropriate attribution to the original author.

---

