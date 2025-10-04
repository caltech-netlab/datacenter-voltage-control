import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx

import utils
from data.circuit_data import *


def single_phase_lindistflow(circuit_data, node_list=None):
    """
    An implementation of the single-phase LinDistFlow is provided for reference.
    Note that the R, X values are averaged (not summed) across the three phases.
    :param circuit_data: dict, edge list, e.g.
        [
            {
                'element_name': 'line_1',
                'source': 'bus_1',
                'target': 'bus_2',
                'series_impedance': complex,
            },
            ...
        ]
    :param per_unit: bool, if True, convert all quantities to per-unit
        Only p.u. is supported for now.
    :param s_base: float, base power for per-unit conversion
    return: R_matrix, X_matrix, C_matrix, node_list
        R_matrix: np.array, shape=(N, N)
        X_matrix: np.array, shape=(N, N)
        C_matrix: np.array, shape=(N+1, N)
        node_list: list of str, node names
        Usage:
            nodal_voltages = v0 + 2 * (R_matrix @ p_inj[1:] + X_matrix @ q_inj[1:])
            branch_power = np.linalg.inv(C_matrix[1:, :]) @ (p + 1j * q)
    """
    edge_list = utils.to_edge_list(circuit_data)
    graph = nx.DiGraph(edge_list)
    root_node = [n for n in graph.nodes if graph.in_degree(n) == 0]
    assert len(root_node) == 1, f"There should be only one root node: {root_node}"
    assert nx.is_directed_acyclic_graph(graph), "Graph contains a cycle. Only radial networks are supported."
    N = len(graph.nodes) - 1    # Excluding the root node
    assert len(graph.edges) == N, len(graph.edges)
    # Build node_list and assign node id
    node_list = node_list or list(nx.bfs_tree(graph, root_node[0]).nodes())
    assert node_list[0] == root_node[0], f"Root node {node_list[0]} does not match {root_node[0]}"
    node2idx = {node: k for k, node in enumerate(node_list)}
    # Replace node names with integer indicies
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    graph = nx.DiGraph(edge_list)
    # Check that the node_list is in BFS or DFS order
    for i in range(1, N+1):
        assert not (set(graph.predecessors(i)) - set(range(i))), \
            str('\n'.join([n.split('|')[0] for n in node_list]) + '\n' + str(set(graph.predecessors(i))) + ' ' + str(i))
    
    # Careful! The graph needs to be in the down orientation.
    # networkx incidence_matrix has the opposite sign convention
    C_matrix = -nx.incidence_matrix(graph, oriented=True, nodelist=list(range(N+1))).toarray()
    ge = list(graph.edges)
    permutation = [ge.index((i, j)) for i, j in edge_list]
    C_matrix = C_matrix[:, permutation]
    C_inv = np.linalg.inv(C_matrix[1:])
    utils.plot_nx_graph(
        graph, 
        node_info={i: n.split('|')[0] for i, n in enumerate(node_list)}, 
        outpath='temp/network.png',
        edge_info={e: circuit_data[i]['series_impedance'] for i, e in enumerate(edge_list)},
    )

    resistances, reactances = np.zeros(N), np.zeros(N)
    for edge in circuit_data:
        e = edge_list.index((node2idx[edge['source']], node2idx[edge['target']]))
        resistances[e] = edge['series_impedance'].real
        reactances[e] = edge['series_impedance'].imag
    R_matrix = C_inv.T @ np.diag(resistances) @ C_inv
    X_matrix = C_inv.T @ np.diag(reactances) @ C_inv
    return R_matrix, X_matrix, C_matrix, node_list, edge_list


def load_compute_power(file, keys, t0=None, t1=None, datetime=True):
    """
    :param file: str, path to the csv file
    :param t0: str, start time in 'YYYY/MM/DD HH:MM:SS.sss'
    :param t1: str, end time in 'YYYY/MM/DD HH:MM:SS.sss'
    :param keys: dict, e.g. {'t': 'timestamp', 'p': 'power.draw [W]'}
    :return: dict with keys 't' (float seconds) and 'power_factor' (per-unit power)
    """
    # Read CSV with timestamp parsing
    df = pd.read_csv(file, parse_dates=[keys['t']])
    # Rename columns to standardized names
    df.rename(columns={keys['t']: 't', keys['p']: 'power'}, inplace=True)
    if df['t'].dtype == 'O':
        df['t'] = df['t'].astype(float)
    # Convert t0, t1 to datetime
    if t0 and t1:
        if datetime:
            t0 = pd.to_datetime(t0)
            t1 = pd.to_datetime(t1)
        # Filter by time range
        mask = (df['t'] >= t0) & (df['t'] <= t1)
        df = df.loc[mask, ['t', 'power']].copy()
    # Reset time to start from zero (in seconds)
    df['t'] = df['t'] - df['t'].iloc[0]
    if datetime:
        df['t'] = df['t'].dt.total_seconds()
    # Compute per-unit power
    max_p = df['power'].max()
    df['power_factor'] = df['power'] / max_p
    # Return dictionary of arrays
    return {'t': df['t'].to_numpy(), 'compute_power_factor': df['power_factor'].to_numpy()}


def load_v0(file, keys, dt, gen_idx=0, test_idx=0):
    """
    Wenqi's mat/pickle file format:
        array[test_idx,0,:,:]
        time = dt*np.arange(0,np.shape(voltage)[-1])
        the test_idx is different outages, the second column is the index for variable, where 0 is voltage
    
    :param file: str, path to the csv file
    :param t0: str, start time in 'YYYY/MM/DD HH:MM:SS.sss'
    :param t1: str, end time in 'YYYY/MM/DD HH:MM:SS.sss'
    :param keys: dict, e.g. {'t': 'timestamp', 'v0': 'voltage [V]'}
    :return: dict with keys 't' (float seconds) and 'v0' (voltage)
    """
    # Read CSV with timestamp parsing
    if file.endswith('.mat'):
        df = sp.io.loadmat(file)
        df = pd.DataFrame({'v0': df[keys['v0']][test_idx, 0, gen_idx]})
    elif file.endswith('.pckl'):
        array = pd.read_pickle(file)[0]
        df = pd.DataFrame({'v0': array[test_idx, 0, gen_idx]})
    else:
        raise ValueError("Unsupported file format")
    
    # Create time column
    df['t'] = dt * np.arange(0, len(df['v0']))

    # Return dictionary of arrays
    return {'t': df['t'].to_numpy(), 'v0': df['v0'].to_numpy()}


def align_data(data, dt, t_start=0, t_end=None):
    """
    Align multiple datasets by a common time base using linear interpolation.
    :param data: dict of named dicts, each with keys 't' and one other data array
    :param resample_interval: float, time step in seconds for interpolation
    :return: None (data is modified in place)
    """
    aligned_data = {}
    # Determine intersection of time ranges
    starts = []
    ends = []
    for entry in data.values():
        t_arr = np.array(entry['t'], dtype=float)
        starts.append(t_arr[0])
        ends.append(t_arr[-1])
    t_start = t_start or max(starts)
    t_end = t_end or min(ends)
    # Create new common time vector
    aligned_data['t'] = np.arange(t_start, t_end, dt)
    # Interpolate each series onto new time base
    for name, entry in data.items():
        # Identify data field (exclude 't')
        for k, v in entry.items():
            if k == 't':
                continue
            # Linear interpolation
            aligned_data[k] = np.interp(aligned_data['t'], entry['t'], v)
    return aligned_data

    
if __name__ == "__main__":
    R_matrix, X_matrix, C_matrix, node_list, edge_list = single_phase_lindistflow(vulcan_circuit_data)
    print("R_matrix:", R_matrix.shape)
    print("X_matrix:", X_matrix.shape)
    print("C_matrix:", C_matrix.shape)
    print("Edge list:", len(edge_list))
    print("Node list:", len(node_list))
    
    file = 'data/compute-power-readings/llama_b16_i512_o128_tp4/nvidiasmi_monitor.csv'
    t0, t1 = '2023/10/18 17:41:36.473', '2023/10/18 17:42:22.703'
    keys = {'t': 'timestamp', 'p': 'power.draw [W]'}
    compute_power = load_compute_power(file, t0, t1, keys)

    file = 'data/record_trajectory_LossGen.mat'
    keys = {'v0': 'record_trajectory'}
    dt = 1e-2  # Resample interval in seconds
    v0 = load_v0(file, keys, dt)
    
    data = {'compute_power': compute_power, 'v0': v0}
    resample_dt = 1e-2
    aligned_data = align_data(data, resample_dt)
    for k, v in aligned_data.items():
        print(f"{k}: {v.shape}")