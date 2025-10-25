# Generating realistic synthetic realtime-like data for a blockchain-based traffic system
# Saves CSV to /mnt/data/blockchain_traffic_trafficsim.csv and displays a preview.
# The dataset includes:
# timestamp, consensus, node_count, network_latency_ms, tx_throughput_tps,
# energy_consumption_joules_per_min, security_risk_score (0-1),
# vehicle_count, vehicle_count_variability, fault_tolerance_requirement (0-1),
# notes (events like node add/remove or congestion)
#
# You can change `minutes` to generate longer/shorter sequences.
import math, random, os, datetime
import pandas as pd
import numpy as np
from caas_jupyter_tools import display_dataframe_to_user

random.seed(42)
np.random.seed(42)

# Config
consensus_types = ['PoW', 'PoS', 'Raft', 'PBFT', 'HotStuff']
minutes = 720  # number of minutes to simulate (default 720 = 12 hours)
start_time = datetime.datetime.now()
time_index = [start_time + datetime.timedelta(minutes=i) for i in range(minutes)]

rows = []
# baseline parameters per consensus (used to shape distributions)
params = {
    'PoW':     {'latency': (300,100), 'throughput': (20,80),   'energy': (2.5e6,5e5), 'sec_base': 0.08},
    'PoS':     {'latency': (120,40),  'throughput': (200,500), 'energy': (1.0e5,2.0e4), 'sec_base': 0.12},
    'Raft':    {'latency': (50,20),   'throughput': (300,900), 'energy': (8.0e4,1.5e4), 'sec_base': 0.10},
    'PBFT':    {'latency': (150,60),  'throughput': (150,450), 'energy': (2.0e5,3.0e4), 'sec_base': 0.14},
    'HotStuff':{'latency': (60,30),   'throughput': (400,1500),'energy': (1.2e5,2.5e4), 'sec_base': 0.09},
}

# initial node counts (can drift during simulation)
node_counts = {c: random.randint(4,12) for c in consensus_types}

# helper: smooth random walk for node count with occasional add/remove events
def step_node_count(curr):
    if random.random() < 0.02:  # 2% chance of a join/leave event
        change = random.choice([-2,-1,1,2])
        curr = max(3, curr + change)
    # small jitter
    curr = max(3, curr + random.choice([0,0,0,1,-1]))
    return curr

# vehicle count base: diurnal pattern + random spikes (e.g., morning/evening congestion)
def vehicle_base(t_idx, minutes_total):
    # create two peaks: morning (~hour 8) and evening (~hour 18) relative to start_time local hour
    hour = (start_time + datetime.timedelta(minutes=t_idx)).hour % 24
    # normalized daily pattern using two gaussians
    morning = math.exp(-((hour-8)**2)/(2*2.5**2))
    evening = math.exp(-((hour-18)**2)/(2*2.5**2))
    base = 200 * (0.6*morning + 0.6*evening) + 100  # baseline vehicles
    # add slow trend over simulation length
    trend = 20 * math.sin(2*math.pi*(t_idx/minutes_total))
    return max(20, base + trend)

for i, t in enumerate(time_index):
    # vehicle environment (system-level)
    vehicle_count = int(np.round(vehicle_base(i, minutes) + np.random.normal(0, 30)))
    vehicle_variability = max(1.0, abs(np.random.normal(0, 15)))  # rolling-std like measure
    
    for c in consensus_types:
        # update node count with small random walk / rare events
        node_counts[c] = step_node_count(node_counts[c])
        ncount = node_counts[c]
        
        p = params[c]
        # latency influenced by node count and vehicle congestion (more vehicles might add load)
        # base latency normal with mean and sd from params; then scale by node_count and vehicle load
        base_latency = np.random.normal(p['latency'][0], p['latency'][1])
        latency_scaling = 1.0 + (ncount - 7)/20.0 + (vehicle_count/1000.0)
        network_latency_ms = max(5, base_latency * latency_scaling + np.random.normal(0,10))
        
        # throughput shaped by consensus capacity, node_count, and occasional drops
        base_tput = np.random.normal(p['throughput'][0], p['throughput'][1])
        tput_scaling = 1.0 + (ncount - 7)/15.0 - (network_latency_ms/1000.0)
        tx_throughput_tps = max(1, base_tput * tput_scaling * (1.0 + np.random.normal(0,0.05)))
        
        # energy consumption (per minute) — PoW huge, others smaller. scale with throughput and random noise
        base_energy = np.random.normal(p['energy'][0], p['energy'][1])
        # energy decreases slightly with more efficient consensus and increases with node_count and throughput
        energy_consumption_joules_per_min = max(1.0, base_energy * (1 + (ncount-7)/30.0) * (tx_throughput_tps/300.0) * (1+np.random.normal(0,0.08)))
        
        # security risk score: combine base and susceptibility to vehicle variability (e.g., more vehicles -> higher attack surface)
        sec = p['sec_base'] + 0.4*(vehicle_variability/50.0) + np.random.normal(0,0.02)
        # adjust for consensus-specific traits: PoW slightly lower risk wrt some attacks, PoS/others moderate
        if c == 'PoW':
            sec *= 0.9
        elif c == 'HotStuff':
            sec *= 0.95
        sec = min(0.99, max(0.01, sec))
        
        # fault tolerance requirement: as a fraction (0-1) of nodes that must be available; larger networks may need higher redundancy
        # e.g., required fraction = base_quorum + jitter. base_quorum depends on algorithm
        base_quorum = {'PoW':0.33, 'PoS':0.34, 'Raft':0.5, 'PBFT':0.66, 'HotStuff':0.67}[c]
        # if vehicle_count high or security risk high, increase requirement
        fault_tolerance_requirement = min(0.99, base_quorum + 0.2*(vehicle_count/1000.0) + 0.3*sec + np.random.normal(0,0.02))
        
        # notes generation: occasional events
        notes = ''
        if random.random() < 0.01:
            notes = random.choice(['node_join', 'node_leave', 'network_partition', 'congestion_event', 'attack_detected'])
        
        rows.append({
            'timestamp': t.isoformat(),
            'consensus': c,
            'node_count': ncount,
            'network_latency_ms': round(float(network_latency_ms),2),
            'tx_throughput_tps': round(float(tx_throughput_tps),2),
            'energy_joules_per_min': round(float(energy_consumption_joules_per_min),2),
            'security_risk_score': round(float(sec),4),
            'vehicle_count': int(vehicle_count),
            'vehicle_count_variability': round(float(vehicle_variability),2),
            'fault_tolerance_requirement': round(float(fault_tolerance_requirement),4),
            'notes': notes
        })

df = pd.DataFrame(rows)
# add a composite system id for grouping (e.g., consensus + minute index)
df['row_id'] = df.index + 1
cols = ['row_id','timestamp','consensus','node_count','network_latency_ms','tx_throughput_tps',
        'energy_joules_per_min','security_risk_score','vehicle_count','vehicle_count_variability',
        'fault_tolerance_requirement','notes']
df = df[cols]

# Save CSV
out_path = '/blockchain_traffic_trafficsim.csv'
df.to_csv(out_path, index=False)

# Display a preview to the user and provide download path
display_dataframe_to_user("Blockchain Traffic Simulation (preview)", df.sample(20, random_state=1).reset_index(drop=True))

print(f"CSV saved to: {out_path}")

