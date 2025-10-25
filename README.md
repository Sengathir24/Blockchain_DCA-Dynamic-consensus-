# 🪙 ConcordiaChain — ML-Driven Dynamic Consensus Blockchain Simulator

**ConcordiaChain** is a **machine learning–assisted blockchain simulator** built using **Flask**, demonstrating how an adaptive network can dynamically switch between consensus algorithms based on real-time simulated metrics.

This compact, single-file simulator is designed for **research**, **education**, and **rapid experimentation** in blockchain governance, node consensus, and hybrid ML-driven decision systems.


## 🧩 Overview

ConcordiaChain simulates a **multi-node blockchain network** capable of switching between multiple consensus mechanisms — such as **Proof of Work (PoW)**, **Proof of Stake (PoS)**, **Raft**, **PBFT**, and **HotStuff** — using an **optional machine learning model** that predicts the best-suited consensus algorithm based on network state.

The simulator also includes:

* A **TailwindCSS-based UI** for real-time visualization.
* A **governance module** where existing users vote to approve new members.
* A **REST API layer** for headless control and automated testing.

This project is intended for blockchain developers, students, and researchers exploring **adaptive consensus models**, **distributed governance**, and **AI-assisted blockchain networks**.

---

## ⚙️ Core Features

✅ **Multi-Consensus Simulation:**
Switch between 5 consensus algorithms — PoW, PoS, Raft, PBFT, and HotStuff.

✅ **Dynamic Consensus via ML:**
An integrated **Decision Tree Classifier** predicts the optimal consensus algorithm based on traffic, latency, and transaction density.

✅ **Flask Web Interface (Tailwind UI):**
Visualize the blockchain state, nodes, and governance activity in real time.

✅ **Governance & Voting System:**
Pending users must be approved via **on-chain-like voting** from existing users.

✅ **Persistent State:**
All blockchain, node, and credential data are persisted locally using JSON and CSV files.

✅ **Headless API Access:**
Full control via REST endpoints — useful for automation or integration into external systems.

---

## 🏗️ Architecture

```
 ┌─────────────────────────────┐
 │      Flask Web Server       │
 ├──────────────┬──────────────┤
 │     UI/UX    │    API Layer │
 │  (Tailwind)  │  (/api/*)    │
 └──────┬───────┴──────────────┘
        │
 ┌──────▼────────────────────────────────┐
 │       Blockchain Engine               │
 │ - Transaction Pool                    │
 │ - Block Forging                       │
 │ - Node Registry                       │
 │ - Governance Voting                   │
 └──────┬────────────────────────────────┘
        │
 ┌──────▼────────────────────────────────┐
 │      ML Decision Module (Optional)    │
 │ - Trained Decision Tree Model         │
 │ - Label Encoding of Consensus Modes   │
 │ - Dynamic Algorithm Switching         │
 └───────────────────────────────────────┘
```

---

## 🧰 System Requirements

* **Python:** 3.8 or higher (Recommended: 3.10 / 3.11)
* **Required Packages:**

  * `Flask`
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `joblib`
  * `werkzeug`

---

## 💻 Installation

Open PowerShell or your terminal and install dependencies:

```bash
python -m pip install --upgrade pip
pip install flask pandas numpy scikit-learn joblib werkzeug
```

(You can also use `requirements.txt` once generated.)

---

## 🚀 Running the Application

1. **Navigate to your project directory:**

   ```bash
   cd "C:\Users\Dell\Downloads\all_five\all_five"
   ```

2. **Run the Flask simulator:**

   ```bash
   python app3.py
   ```

3. **Open your browser** at:

   ```
   http://127.0.0.1:5000
   ```

   The Flask console will display startup logs — including detected credential files, user count, and ML model status.

---

## 📂 File and Directory Structure

```
all_five/
│
├── app3.py                        # Main Flask simulator
├── app.py, app1.py, dynamic_algo.py
│                                  # Experimental/alternate builds
├── blockchain_traffic_trafficsim.csv
│                                  # Example dataset for ML model
├── decision_tree_consensus.pkl    # Trained ML model (optional)
├── label_encoder.pkl              # Encoded consensus label map (optional)
├── User_credentials.xlsx - Sheet1.csv
│                                  # User credentials (auto-generated if missing)
├── network_state.json             # Persisted blockchain/network state
└── readme.md                      # Documentation file (this one)
```

---

## 🔐 Authentication and Governance

* **Credentials:** Stored in a CSV file with hashed passwords (using `werkzeug.security`).
* **New Users:** Are added to a `pending_users` queue.
* **Voting System:** Active users vote to approve pending sign-ups.
* **Minimum Governance Requirement:** At least two default active users are ensured at startup for quorum.

---

## 🔗 API Endpoints

| Method | Endpoint             | Description                                                  |
| ------ | -------------------- | ------------------------------------------------------------ |
| `GET`  | `/api/state`         | Get the current blockchain state (chain, nodes, pending tx). |
| `GET`  | `/api/get_users`     | Retrieve lists of active and pending users.                  |
| `POST` | `/api/vote_user`     | Approve a pending user. Body: `{ "username": "alice" }`      |
| `POST` | `/api/add_tx`        | Add a transaction (optionally triggers ML).                  |
| `POST` | `/api/trigger`       | Mine/forge a new block using active consensus.               |
| `POST` | `/api/add_node`      | Add a node. Body: `{ "name": "Node1", "stake": 10 }`         |
| `POST` | `/api/remove_node`   | Remove a node. Body: `{ "name": "Node1" }`                   |
| `POST` | `/api/set_consensus` | Set consensus manually. Body: `{ "mode": "PoS" }`            |

---

## 🧠 Dynamic ML-Based Consensus

If available, the **ML model** dynamically selects the optimal consensus algorithm using simulated metrics such as:

* Transaction volume
* Node latency
* Network throughput
* Stake distribution

The model predicts the ideal consensus and automatically updates the blockchain’s mode if the prediction differs from the current one.

### Expected Files:

* `decision_tree_consensus.pkl` — serialized DecisionTree model.
* `label_encoder.pkl` — label mapping for consensus modes.
* `blockchain_traffic_trafficsim.csv` — input dataset used during training.

If any of these are missing, the app defaults to **static consensus mode**.

---

## ⚙️ Configuration Details

Inside `app3.py`, update these paths if you move the workspace:

| Variable             | Description                       | Default Path                                                                   |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------ |
| `CREDENTIALS_FILE`   | CSV storing user credentials      | `C:\Users\Dell\Downloads\all_five\all_five\User_credentials.xlsx - Sheet1.csv` |
| `MODEL_PATH`         | ML model file                     | `decision_tree_consensus.pkl`                                                  |
| `LABEL_ENCODER_PATH` | Label encoder file                | `label_encoder.pkl`                                                            |
| `NETWORK_STATE_FILE` | JSON storing runtime network data | `network_state.json`                                                           |

---

## 🧯 Troubleshooting

* **Import Errors:** Verify all dependencies are installed correctly.
* **File Not Found:** Ensure file paths in `app3.py` match your directory structure.
* **Port Conflict:** Modify `app.run(port=5001)` in `app3.py`.
* **CSV Errors:** Delete or fix malformed CSVs — app will recreate defaults.
* **ML Failures:** If model loading fails, app logs the error and switches to static mode automatically.

---

## 🔒 Security Considerations

* All passwords are securely hashed.
* No sensitive data is exposed in plaintext.
* Flask debug server is for **local use only** — do not deploy publicly.
* Avoid committing user or model files to GitHub.

---

## 🧑‍💻 Development Notes

* Change credential filename in `app3.py` for simpler usage.
* Add your own trained ML model to improve consensus prediction accuracy.
* You can connect frontend dashboards or scripts using the provided API routes.

---

## 🧭 Future Enhancements

Planned upgrades include:

* 🧩 Addition of **Reinforcement Learning-based adaptive consensus**
* 🔍 Visual dashboards for node health & consensus transitions
* 🧑‍🤝‍🧑 Advanced governance model (reputation-weighted voting)
* 🧠 Integration with **Federated ML** for decentralized training
* 🧪 Automated test coverage for API and consensus mechanisms

---

## 📜 License & Attribution

This project is open-source and provided **for educational and demonstration purposes only**.
You are free to modify and distribute the simulator with proper attribution.

---

## 📬 Contact

If you'd like additional support or collaboration opportunities, I can:

* Add a `requirements.txt` file for reproducible setup
* Write a `setup_env.py` script for automatic environment configuration
* Implement basic unit tests for critical modules

**Happy experimenting!**
Launch the simulator and explore how machine learning can drive consensus evolution in blockchain systems.

---

