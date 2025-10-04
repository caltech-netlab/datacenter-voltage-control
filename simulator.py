import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
import andes
import pickle

import load_data
from visualize import *
from data.circuit_data import *
np.set_printoptions(precision=3)
np.set_printoptions(threshold=10000)
np.set_printoptions(linewidth=200)
DT = 5e-3


def controller_decentralized(v, q_t, p_t, K_p, K_q, p_nominal, q_nominal, deadband=None):
    """
    Decentralized safe reinforcement learning for inverter-based voltage control. 
    Updates per-node real and reactive power control offsets using local voltage feedback.

    :param v: np.ndarray, per-node voltage magnitudes (p.u.), shape=(N,)
    :param q_t: np.ndarray, current reactive power control offsets (same units as q_nominal), shape=(N,)
    :param p_t: np.ndarray, current active power control offsets (same units as p_nominal), shape=(N,)
    :param K_p: np.ndarray or float, proportional gain(s) applied to (v - 1) for active power
    :param K_q: np.ndarray or float, proportional gain(s) applied to (v - 1) for reactive power
    :param p_nominal: np.ndarray, nominal active injections used by the caller for clipping, shape=(N,)
    :param q_nominal: np.ndarray, nominal reactive injections used by the caller for clipping, shape=(N,)
    :param deadband: float, optional voltage deadband around 1.0 p.u.
    :return: p_t1, q_t1 -- updated active and reactive power control offsets (same shape/units as inputs)
    """
    p_t1 = p_t + K_p * (v - 1)
    q_t1 = q_t + K_q * (v - 1)
    if deadband:
        mask = (np.abs(v - 1) > deadband)
        p_t1[~mask] = p_nominal[~mask]
        q_t1[~mask] = q_nominal[~mask]
    return p_t1, q_t1

    
def controller_centralized(
    v0, R, X, p_min, p_max, q_min, q_max, p_nominal, q_nominal, ctrl_cost_p, ctrl_cost_q, v_cost=15e-3,
):
    """
    Centralized optimal power flow for inverter-based voltage control
    Positive current injection direction is defined as out of a node.
    Solves a convex quadratic program trading off voltage deviation and control effort.
    Each optimization is done for 1 time step.
    :param v0: slack bus voltage, float
    :param R: np.ndarray, float, resistance matrix, shape=(N, N)
    :param X: np.ndarray, float, reactance matrix, shape=(N, N)
    :param p_min: np.ndarray, float, minimum active power injection, shape=(N,)
    :param p_max: np.ndarray, float, maximum active power injection, shape=(N,)
    :param q_min: np.ndarray, float, minimum reactive power injection, shape=(N,)
    :param q_max: np.ndarray, float, maximum reactive power injection, shape=(N,)
    :param p_nominal: np.ndarray, float, nominal active power injection, shape=(N,)
    :param q_nominal: np.ndarray, float, nominal reactive power injection, shape=(N,)
    :param ctrl_cost_p: np.ndarray, float, real power control cost weight, shape=(N,)
    :param ctrl_cost_q: np.ndarray, float, reactive power control cost weight, shape=(N,)
    :param v_cost: voltage deviation cost weight, float
    :return: np.ndarray, optimal active and reactive power injections (per-unit normalized), shape=(N), shape=(N)
    """
    # Compute the optimal control action
    N = R.shape[0]
    p_pu = cp.Variable(N)
    q_pu = cp.Variable(N)
    objective = cp.Minimize(
        cp.quad_form(cp.multiply(p_pu, p_max) - p_nominal, np.diag(ctrl_cost_p)) +
        cp.quad_form(cp.multiply(q_pu, q_max) - q_nominal, np.diag(ctrl_cost_q)) +
        cp.quad_form(
            v0**2 - 2 * (R @ cp.multiply(p_pu, p_max) + X @ cp.multiply(q_pu, q_max)) - 1, 
            np.diag(np.ones(N) * v_cost)
        )
    )
    assert (p_min <= p_max).all(), (p_min, p_max)
    assert (q_min <= q_max).all(), (q_min, q_max)
    constraints = [
        cp.multiply(p_pu, p_max) >= p_min,
        cp.multiply(p_pu, p_max) <= p_max,
        cp.multiply(q_pu, q_max) >= q_min,
        cp.multiply(q_pu, q_max) <= q_max
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    assert prob.status == 'optimal', prob.status
    return p_pu.value, q_pu.value


def compute_andes_v0(
    ss,
    p, 
    q,
    load_bus_idx,
    load_idx,
    v0_dt,
    gen_bus_trip_time=1.0,
    gen_trip_name="GENROU_2",
    error_msg_printed=False,
    andes_out_path="temp/andes_out"
):
    """
    Advance the Andes dynamic simulator by one step after updating a PQ load, 
    and return the bus voltage. Updates ss.PQ.Ppf.v and ss.PQ.Qpf.v (per-unit 
    on ss.config.mva), advances the simulator by v0_dt, and optionally trips a named generator.
    :param ss: AndesSimulator object
    :param p: active power injection from data center, float
    :param q: reactive power injection from data center, float
    :param load_bus_idx: int, to which bus the data center is connected in the AndesSimulator
    :param load_idx: int, which load in AndesSimulator should be replaced by time-varying data center load
    :param v0_dt: time step for the simulation, float
    :param gen_bus_trip_time: time (in seconds) at which the generator trips, float
    :param gen_trip_name: name of the generator to trip, str
    :param error_msg_printed: bool, whether the error message has been printed before
    :param andes_out_path: str, path to save the Andes output files (doesn't work)
    :return: voltage at load_bus_idx, float
    """
    # Change the load real and reactive power and continue simulation
    # We are *not* re-running power flow here; we are continuing dynamics with a
    # new scheduled load level going forward (piecewise-constant change).
    # Positive is load, negative is generation
    ss.PQ.Ppf.v[load_idx] = p / ss.config.mva      # Change load power. Ppf is an array of 11 elements
    ss.PQ.Qpf.v[load_idx] = q / ss.config.mva      # Change load power. Qpf is an array of 11 elements
    ss.TDS.config.tf += v0_dt   # Simulation end time (seconds)

    # Trip the generator
    if ss.TDS.config.tf >= gen_bus_trip_time:
        status = ss.SynGen.alter("u", gen_trip_name, 0)
        assert status
    
    # Run one step
    ss.TDS.config.no_tqdm = True
    success = ss.TDS.run(tstep=v0_dt, no_summary=True, progress=False, output_path=andes_out_path)
    if (not error_msg_printed) and (not success):
        print("[Warning] Andes simulation diverged.")
    # ss.dae.ts.y is the global state vector bus voltages, shape = (T, N)
    v0 = ss.dae.ts.y[-1,ss.Bus.v.a][load_bus_idx]

    return v0, success
    

def run_simulation(
    circuit_data, 
    bus_data, 
    controller_type,
    compute_pwr_file='data/compute-power-readings/gpu_anonymous_1.csv',     # This one has very large and fast power fluctuations
    # compute_pwr_file='data/compute-power-readings/gpu_anonymous_2.csv',   # This one has much slower fluctuations
    stiff_grid=False,
    decentralized_ctrl_file='data/DecentralizedGain/trained_kq_kp_microsoft.pckl',
    v0_file='data/record_trajectory_LossGen_t0.001.pckl', 
    v0_dt=DT,
    resample_dt=DT,
    sim_length=3,
    compute_flexibility=0.2,
    cooling_flexibility=0.2,
    CENTRALIZED_CTRL_DELAY=0.05,
    LOAD_BUS_IDX=2,
    LOAD_IDX=1,
    V_TRIP_THRESHOLD=0.80,
    V_TRIP_TIME=0.10,
    COOLING_PF=0.9,
    andes_out_path="temp/andes_out",
    cache_dir="temp/cache",
    verbose=False,
):
    """
    End-to-end simulation driver that runs the datacenter control experiment using either 
    a stiff (pre-recorded) grid or an Andes dynamic model. Loads data, sets up control bounds, 
    runs controllers, computes voltages, and returns results. Results are cached by experiment 
    meta to speed repeated runs.
    :param circuit_data: list of dict, see circuit_data.py for details
    :param bus_data: dict of dict, see circuit_data.py for details
    :param controller_type: str, 'decentralized' or 'centralized' or 'no_control'
    :param compute_pwr_file: str, path to the compute power file
    :param stiff_grid: bool, whether to use stiff grid (pre-recorded v0) or Andes simulation
    :param decentralized_ctrl_file: str, path to file containing decentralized control gains
    :param v0_file: str, path to the v0 file
    :param v0_dt: float, time step of the v0 file
    :param resample_dt: float, time step to resample the data
    :param sim_length: float, length of the simulation in seconds
    :param compute_flexibility: float, fraction of compute power that can be controlled
    :param cooling_flexibility: float, fraction of cooling power that can be controlled
    :param CENTRALIZED_CTRL_DELAY: float, seconds, delay for centralized control
    :param LOAD_BUS_IDX: int, index of the load bus in the AndesSimulator
    :param LOAD_IDX: int, index of the load in the AndesSimulator
    :param V_TRIP_THRESHOLD: float, voltage threshold for load trip
    :param V_TRIP_TIME: float, time duration for which voltage must be below threshold to trip the load
    :param COOLING_PF: float, power factor of the cooling load
    :param andes_out_path: str, path to save the Andes output files (doesn't work)
    :param cache_dir: str, directory to store cache files
    :param verbose: bool, whether to print verbose output
    :return: None
    """
    # --------------Cache: build meta and check if cached results exist-----------------
    os.makedirs(cache_dir, exist_ok=True)
    # NOTE: The spec uses a key name 'compute_flexiblity' (typo) — include it as requested.
    meta = {
        "controller_type": controller_type,
        "compute_pwr_file": compute_pwr_file,
        "stiff_grid": stiff_grid,
        "v0_file": v0_file,
        "sim_length": sim_length,
        "compute_flexiblity": compute_flexibility,  # intentional key spelling per request
        "cooling_flexibility": cooling_flexibility,
        "CENTRALIZED_CTRL_DELAY": CENTRALIZED_CTRL_DELAY,
        "LOAD_BUS_IDX": LOAD_BUS_IDX,
        "LOAD_IDX": LOAD_IDX,
        "V_TRIP_THRESHOLD": V_TRIP_THRESHOLD,
        "V_TRIP_TIME": V_TRIP_TIME,
        "COOLING_PF": COOLING_PF,
        "decentralized_ctrl_file": decentralized_ctrl_file,
    }
    # Stable hash from meta (sorted keys) — this enforces “keys match”.
    meta_str = json.dumps(meta, sort_keys=True, separators=(",", ":"))
    meta_hash = hashlib.sha256(meta_str.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"sim_results_{meta_hash}.pkl")

    # If cache exists and meta matches, load and return
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "meta" in payload and "sim_results" in payload:
                if payload["meta"] == meta:
                    # Cached results correspond to the same experiment keys
                    return payload["sim_results"]
        except Exception:
            # Corrupt or incompatible cache → ignore and recompute
            pass

    # --------------No cache, run computation-----------------
    CENTRALIZED_CTRL_DELAY_STEPS = int(CENTRALIZED_CTRL_DELAY / resample_dt)
    # Load data
    R_matrix, X_matrix, C_matrix, node_list, edge_list = \
        load_data.single_phase_lindistflow(circuit_data)
    N = len(node_list) - 1
    
    if N == 13:
        centralized_ctrl_cost_p = 1e-4 * np.array([1, 1, 0.5, 0.5, 10, 2, 2, 2, 2, 2, 2, 2, 2])
        centralized_ctrl_cost_q = 1e-5 * np.array([1, 1, 0.5, 0.5, 10, 2, 2, 2, 2, 2, 2, 2, 2])
    elif N == 12:
        centralized_ctrl_cost_p = 1e-4 * np.array([1, 1, 0.5, 10, 2, 2, 2, 2, 2, 2, 2, 2])
        centralized_ctrl_cost_q = 1e-5 * np.array([1, 1, 0.5, 10, 2, 2, 2, 2, 2, 2, 2, 2])
    elif N == 11:
        centralized_ctrl_cost_p = 1e-4 * np.array([1, 1, 10, 2, 2, 2, 2, 2, 2, 2, 2])
        centralized_ctrl_cost_q = 1e-5 * np.array([1, 1, 10, 2, 2, 2, 2, 2, 2, 2, 2])

    if controller_type == 'decentralized':
        f = open(decentralized_ctrl_file, 'rb')
        dat = pickle.load(f)
        if len(dat) == 2:
            K_p, K_q = dat
        else:
            K_q = dat[0]
            K_p = np.zeros((N,))
        f.close()
        K_p, K_q = K_p.flatten(), K_q.flatten()
        print('Decentralized control gains')
        print(list(zip(node_list[1:], K_p, K_q)))
        assert (K_p >= 0).all(), K_p
        assert (K_q >= 0).all(), K_q
        assert len(K_p) == len(K_q) == N, (len(K_p), len(K_q), N)

    t0, t1 = '2023/10/18 17:41:36.473', '2023/10/18 17:42:22.703'
    keys = {'t': 'timestamp', 'p': 'power.draw [W]'}
    compute_power = load_data.load_compute_power(compute_pwr_file, keys, t0, t1)
    # This is for the other compute power file
    # keys = {'t': 'time_s', 'p': 'gpu_power_norm'}
    # compute_power = load_data.load_compute_power(compute_pwr_file, keys, 28, 40, datetime=False)

    if stiff_grid:
        keys = {'v0': 'record_trajectory'}
        v0 = load_data.load_v0(v0_file, keys, v0_dt)
        data = {'compute_power': compute_power, 'v0': v0}
        aligned_data = load_data.align_data(data, resample_dt)
        T = len(aligned_data['t'])
        print(f"Loaded data with {C_matrix.shape} nodes and edges and {T} time steps.")
    else:
        assert resample_dt == v0_dt, (resample_dt, v0_dt)
        andes.config_logger(stream_level=100)
        ss = andes.load("data/andes/ieee14.raw", addfile="data/andes/ieee14.dyr", setup=False)
        # Add a Toggle that disconnects generator `GENROU_2` at t=1 s
        assert len(ss.Toggle.u.v) == 0, ss.Toggle.as_df()
        ss.setup()
        # calculate power flow: use constant power model for PQ loads
        # PQ load gives greater voltage deviation than constant impedance load
        ss.PQ.config.p2p = 1       # keep active power (P) as constant power load
        ss.PQ.config.q2q = 1       # keep reactive power (Q) as constant power load
        ss.PQ.config.p2z = 0       # do not convert active power (P) to constant impedance
        ss.PQ.config.q2z = 0       # do not convert reactive power (Q) to constant impedance
        ss.PQ.config.pq2z = 0      # turn off combined P+Q to Z conversion under low voltage
        # Run the power-flow (steady-state) solution to get initial operating point.
        os.makedirs(andes_out_path, exist_ok=True)
        ss.PFlow.run(output_path=andes_out_path)
        ss.TDS.config.criteria = 0  # temporarily turn off stability criteria based on angle separation
        ss.TDS.config.tf = 0

        # Load and align compute_power data
        data = {'compute_power': compute_power}
        aligned_data = load_data.align_data(data, resample_dt, t_end=sim_length)
        T = len(aligned_data['t'])
        if verbose:
            print(
                f"Total system load (excluding data center): "
                f"{(ss.PQ.Ppf.v.sum() - ss.PQ.Ppf.v[LOAD_BUS_IDX]) * ss.config.mva:.3f} MW + "
                f"j{(ss.PQ.Qpf.v.sum() - ss.PQ.Qpf.v[LOAD_BUS_IDX]) * ss.config.mva:.3f} MVAr"
            )
            print("GENERATOR DATA:")
            print(ss.GENROU.as_df())
        
    # Simulation parameters
    p_min, p_max, q_min, q_max = np.zeros((N, T)), np.zeros((N, T)), np.zeros((N, T)), np.zeros((N, T))
    p_nominal, q_nominal = np.zeros((N, T)), np.zeros((N, T))
    cooling_bus_idx = []
    for n, node in enumerate(node_list[1:]):
        if bus_data[node]['type'] == 'compute':
            p_nominal[n] = bus_data[node]['p_rating'] * aligned_data['compute_power_factor']
            p_min[n] = p_nominal[n] * (1 - compute_flexibility) + bus_data[node]['pmin']
            p_max[n] = p_nominal[n] * (1 + compute_flexibility) + bus_data[node]['pmax']
            q_min[n] = bus_data[node]['qmin']
            q_max[n] = bus_data[node]['qmax']
        elif bus_data[node]['type'] == 'cooling':
            p_nominal[n] = bus_data[node]['p_rating'] * np.mean(aligned_data['compute_power_factor'])
            p_max[n] = np.minimum(p_nominal[n] * (1 + cooling_flexibility), bus_data[node]['pmax'])
            p_min[n] = np.maximum(p_nominal[n] * (1 - cooling_flexibility), bus_data[node]['pmin'])
            q_nominal[n] = p_nominal[n] * (1 / COOLING_PF - 1)
            q_max[n] = np.minimum(q_nominal[n] * (1 + cooling_flexibility), bus_data[node]['qmax'])
            q_min[n] = np.maximum(q_nominal[n] * (1 - cooling_flexibility), bus_data[node]['qmin'])
            cooling_bus_idx.append(n)
        else:
            if 'pmin' in bus_data[node]:
                p_min[n] = bus_data[node]['pmin']
            if 'pmax' in bus_data[node]:
                p_max[n] = bus_data[node]['pmax']
            if 'qmin' in bus_data[node]:
                q_min[n] = bus_data[node]['qmin']
            if 'qmax' in bus_data[node]:
                q_max[n] = bus_data[node]['qmax']
    if verbose:
        print(f"Total data center nominal load: {p_nominal.sum(axis=0).mean():.3f} MW + j{q_nominal.sum(axis=0).mean():.3f} MVAr")

    sim_results = {
        't': aligned_data['t'],
        'v0': aligned_data['v0'] if stiff_grid else np.ones((T,)),
        'compute_power_factor': aligned_data['compute_power_factor'],
        'v_solution': np.zeros((N, T)),
        'p_inj': p_nominal.copy(),
        'q_inj': q_nominal.copy(),
        'p_nominal': p_nominal,
        'q_nominal': q_nominal,
        'sim_valid': np.ones((T,), dtype=bool),
        'node_list': node_list,
    }
    if stiff_grid:
        v0 = aligned_data['v0'][0]
    else:
        v0, success = compute_andes_v0(ss, p_nominal[:, 0].sum(), q_nominal[:, 0].sum(), LOAD_BUS_IDX, LOAD_IDX, v0_dt)
    v = np.sqrt(v0**2 - 2 * (R_matrix @ p_nominal[:, 0] + X_matrix @ q_nominal[:, 0]))

    trip_counter = 0
    p_ctrl_t, q_ctrl_t = np.zeros((N,)), np.zeros((N,))
    for t in tqdm(range(T)):
        # Apply data center control
        sim_results['v0'][t] = v0
        if controller_type == 'decentralized':
            p_ctrl_t, q_ctrl_t = controller_decentralized(
                v, p_ctrl_t, q_ctrl_t, K_p, K_q, p_nominal[:, t], q_nominal[:, t]
            )
            # # No control constraints
            # sim_results['p_inj'][:, t] = p_ctrl_t + p_nominal[:, t]
            # sim_results['q_inj'][:, t] = q_ctrl_t + q_nominal[:, t]
            # With constraints
            sim_results['p_inj'][:, t] = np.minimum(np.maximum(p_ctrl_t + p_nominal[:, t], p_min[:, t]), p_max[:, t])
            sim_results['q_inj'][:, t] = np.minimum(np.maximum(q_ctrl_t + q_nominal[:, t], q_min[:, t]), q_max[:, t])
            # Update the control action after thresholding
            p_ctrl_t = sim_results['p_inj'][:, t] - p_nominal[:, t]
            q_ctrl_t = sim_results['q_inj'][:, t] - q_nominal[:, t]
        elif controller_type == 'centralized':
            if t + CENTRALIZED_CTRL_DELAY_STEPS < T:
                p_pu_t, q_pu_t = controller_centralized(v0, R_matrix, X_matrix, p_min[:, t], p_max[:, t], q_min[:, t], q_max[:, t], p_nominal[:, t], q_nominal[:, t], centralized_ctrl_cost_p, centralized_ctrl_cost_q)
                # We use p_max, q_max as the per unit base for control actions
                sim_results['p_inj'][:, t + CENTRALIZED_CTRL_DELAY_STEPS] = p_max[:, t + CENTRALIZED_CTRL_DELAY_STEPS] * p_pu_t
                sim_results['q_inj'][:, t + CENTRALIZED_CTRL_DELAY_STEPS] = q_max[:, t + CENTRALIZED_CTRL_DELAY_STEPS] * q_pu_t
        elif controller_type == 'no_control':
            if v0 < V_TRIP_THRESHOLD:
                trip_counter += resample_dt
            if trip_counter >= V_TRIP_TIME:
                # Trip the load (scale down to close to zero)
                sim_results['p_inj'][:, t] = 0
                sim_results['q_inj'][:, t] = 0
        # Ensure constant power factor
        for n in cooling_bus_idx:
            if cooling_flexibility == 0:
                sim_results['p_inj'][n, t] = p_nominal[n, t]
                sim_results['q_inj'][n, t] = q_nominal[n, t]
            else:
                s = sim_results['p_inj'][n, t] + sim_results['q_inj'][n, t]
                sim_results['p_inj'][n, t], sim_results['q_inj'][n, t] = s * COOLING_PF, s * (1 - COOLING_PF)
        # Compute the voltage at next time step
        v0, success = (aligned_data['v0'][t], True) if stiff_grid else \
            compute_andes_v0(ss, sim_results['p_inj'][:, t].sum(), sim_results['q_inj'][:, t].sum(), LOAD_BUS_IDX, LOAD_IDX, v0_dt, error_msg_printed=(not success))
        v = np.sqrt(v0**2 - 2 * (R_matrix @ sim_results['p_inj'][:, t] + X_matrix @ sim_results['q_inj'][:, t]))
        sim_results['v_solution'][:, t] = v
        sim_results['sim_valid'][t] = success

    # Save to cache and return
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": meta, "sim_results": sim_results}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Warning: failed to write cache {cache_path}: {e}")
    
    return sim_results


def summarize_results(sim_results, title, dt=DT, print_info=True):
    """
    Compute summary metrics from sim_results: average and maximum voltage deviation and 
    total control effort (energy). Optionally prints a human-readable summary.
    :param sim_results: dict returned by run_simulation 
    :param title: str, title used when printing summary
    :param dt: float, time step used to convert sums to energy (seconds), default=DT
    :param print_info: bool, whether to print the computed metrics (default=True)
    :return: 
        avg_v_deviation, float
        max_v_deviation, float
        p_ctrl_sum, float
        q_ctrl_sum, float
    """
    avg_v_deviation = np.abs(sim_results['v_solution'] - 1).mean()
    max_v_deviation = np.abs(sim_results['v_solution'] - 1).max()
    p_ctrl_sum = np.abs(sim_results['p_inj'] - sim_results['p_nominal']).sum() * dt / 3600
    q_ctrl_sum = np.abs(sim_results['q_inj'] - sim_results['q_nominal']).sum() * dt / 3600
    if print_info:
        print(
            f"{title}:\n"
            f"avg voltage deviation = {avg_v_deviation:.4f} p.u.\n"
            f"max voltage deviation = {max_v_deviation:.4f} p.u.\n"
            f"total real power control action = {p_ctrl_sum:.4f} MW\n"
            f"total reactive power control action = {q_ctrl_sum:.4f} MVAr\n"
        )
    return avg_v_deviation, max_v_deviation, p_ctrl_sum, q_ctrl_sum


if __name__ == "__main__":
    table_data = {}
    """High battery"""
    # Simulate without controller (disconnect)
    sim_results = run_simulation(vulcan_circuit_data2, vulcan_bus_data2, 'no_control')
    plot_simulation(sim_results, "No ride-through", plot_ctrl=False, out_path='temp/exp_high_battery')
    sim_summary = summarize_results(sim_results, "No ride-through")
    table_data["No Ride-Through"] = sim_summary
    # Simulate with decentralized controller
    sim_results = run_simulation(vulcan_circuit_data2, vulcan_bus_data2, 'decentralized')
    plot_simulation(sim_results, "With Decentralized Controller", plot_ctrl=True, out_path='temp/exp_high_battery')
    sim_summary = summarize_results(sim_results, "With Decentralized Controller")
    table_data["With Decentralized Controller"] = sim_summary
    # Simulate with centralized controller
    sim_results = run_simulation(vulcan_circuit_data2, vulcan_bus_data2, 'centralized')
    plot_simulation(sim_results, "With Centralized Controller", plot_ctrl=True, out_path='temp/exp_high_battery')
    sim_summary = summarize_results(sim_results, "With Centralized Controller")
    table_data["With Centralized Controller"] = sim_summary

    # Centralized controller with longer delay ablation
    summary = []
    delays = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for delay in delays:
        sim_results = run_simulation(vulcan_circuit_data2, vulcan_bus_data2, 'centralized', CENTRALIZED_CTRL_DELAY=delay)
        plot_simulation(sim_results, f"With Centralized Controller (Delay = {delay}s)", plot_ctrl=True, out_path='temp/exp_delay_ablation')
        sim_summary = summarize_results(sim_results, f"With Centralized Controller (Delay = {delay})")
        summary.append(np.array(sim_summary))
        if delay == 0.2:
            table_data["Centralized Delay 0.2s"] = sim_summary
    summary = np.stack(summary)
    plot_delay_ablation(delays, summary)
    
    # q only
    sim_results = run_simulation(vulcan_circuit_data2, vulcan_bus_data4, 'centralized', compute_flexibility=0, cooling_flexibility=0)
    sim_summary = summarize_results(sim_results, "With Centralized Controller (Q only)")
    table_data["With Centralized Controller (Q only)"] = sim_summary
    sim_results = run_simulation(vulcan_circuit_data2, vulcan_bus_data4, 'decentralized', compute_flexibility=0, cooling_flexibility=0, decentralized_ctrl_file='data/DecentralizedGain/trained_kq_Vulcan.pckl')
    sim_summary = summarize_results(sim_results, "With Decentralized Controller (Q only)")
    table_data["With Decentralized Controller (Q only)"] = sim_summary
    plot_simulation(sim_results, "With Decentralized Controller (Q only)", plot_ctrl=True, out_path='temp/exp_high_battery')
        
    """Summarize table"""
    generate_summary_table(table_data)
