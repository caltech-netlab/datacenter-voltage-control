import os
import re
import matplotlib.pyplot as plt
import load_data
ORANGE = [1.00, 0.43, 0.12]
plt.rcParams['font.family'] = 'Times New Roman'


def _labelize(name: str) -> str:
    """
    Return a human-friendly label for a node/bus name.

    :param name: str, original node/bus identifier
    :return: str, human-friendly label (patterned or overridden); falls back to original name
    """
    # Extend or edit as needed
    LABEL_OVERRIDES = {
        # Example explicit mappings (these take precedence over pattern rules)
        # "batt_1_sec": "battery 1",
        # "cooling_2_sec": "cooling 2",
        # "datacenter_3_sec": "compute cluster 3",
    }

    # Pattern-based renaming
    _PATTERNS = [
        (re.compile(r"^batt_(\d+)_sec$"),       r"Battery \1"),
        (re.compile(r"^cooling_(\d+)_sec$"),    r"Cooling \1"),
        (re.compile(r"^datacenter_(\d+)_sec$"), r"IT Cluster \1"),
    ]

    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    for pat, repl in _PATTERNS:
        m = pat.match(name)
        if m:
            return pat.sub(repl, name)
    return name  # fallback: original


def plot_simulation_2x2(
    sim_results, 
    filename, 
    nodes_to_plot=None, 
    plot_ctrl=False,
    out_path='temp', 
    ext=('pdf',),
    title=None,
    figsize=(12, 6),
    v_ylim=(0.7, 1.3),
    EXCLUDED_BUS_NAMES={"interconnection_a", "utility_a_pri", "mv_distribution_bus"},
    CONTROL_Y_LIM=(-10, 10),
    INJ_Y_LIM=(-5, 35)
):
    """
    Create and save a 2x2 summary figure showing slack voltage, nodal voltages, and P/Q injections.

    :param sim_results: dict, simulation results with keys 't', 'v0', 'v_solution', 'p_inj', 'q_inj', 'p_nominal', 'q_nominal', 'node_list'
    :param filename: str, base filename (without extension) used to save the figure
    :param nodes_to_plot: iterable or None, indices or names of nodes to include (default: auto-select)
    :param plot_ctrl: bool, if True plot control actions (injections - nominal) instead of raw injections
    :param out_path: str, directory to save outputs
    :param ext: tuple, file extensions to write (e.g., ('pdf',))
    :param title: str or None, optional figure title
    :param figsize: tuple, figure size in inches
    :param v_ylim: tuple, y-limits for voltage axes
    :param EXCLUDED_BUS_NAMES: set, bus names to exclude from default selection
    :param CONTROL_Y_LIM: tuple, y-limits for control plots
    :param INJ_Y_LIM: tuple, y-limits for injection plots
    :return: None (saves figure files to out_path)
    """

    t = sim_results['t']
    node_list = sim_results['node_list'][1:]  # skip slack bus
    N = sim_results['v_solution'].shape[0]

    if not nodes_to_plot:
        # default: first up-to-10 nodes, excluding excluded names
        nodes_to_plot = [i for i in range(N) if node_list[i] not in EXCLUDED_BUS_NAMES][:12]
    else:
        # filter out excluded names if indices were provided
        nodes_to_plot = [i for i in range(N) if node_list[i] not in EXCLUDED_BUS_NAMES and node_list[i] in nodes_to_plot]

    # 2x2 layout:
    # (0,0) slack bus voltage
    # (0,1) voltages
    # (1,0) P
    # (1,1) Q
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=False)
    if title: fig.suptitle(title)
    ax_slack   = axes[0, 0]
    ax_V       = axes[0, 1]
    ax_P       = axes[1, 0]
    ax_Q       = axes[1, 1]

    # Slack bus voltage (top-left)
    ax_slack.plot(t, sim_results['v0'], label='v0 (slack bus)', color='black')
    ax_slack.set_xlabel("Time (s)")
    ax_slack.set_ylabel("Transmission Interconnection Node Voltage (p.u.)")
    ax_slack.grid(True, alpha=0.3)
    ax_slack.set_ylim(v_ylim)

    # Voltages (top-right)
    handles = []
    labels = []
    for i in nodes_to_plot:
        raw_label = node_list[i]
        lbl = _labelize(raw_label)
        h, = ax_V.plot(t, sim_results['v_solution'][i, :], label=lbl)
        handles.append(h)
        labels.append(lbl)
    ax_V.set_xlabel("Time (s)")
    ax_V.set_ylabel("Data Center Nodal Voltages (p.u.)")
    ax_V.grid(True, alpha=0.3)
    ax_V.set_ylim(v_ylim)

    # P (bottom-left)
    for i in nodes_to_plot:
        lbl = _labelize(node_list[i])
        if plot_ctrl:
            p_control_action = sim_results['p_inj'][i, :] - sim_results['p_nominal'][i, :]
            ax_P.plot(t, p_control_action, label=lbl)
        else:
            ax_P.plot(t, sim_results['p_inj'][i, :], label=lbl)
    ax_P.set_xlabel("Time (s)")
    ax_P.set_ylabel("Real Power Control (MW)" if plot_ctrl else "Real Power Injection (MW)")
    ax_P.grid(True, alpha=0.3)
    ax_P.set_ylim(CONTROL_Y_LIM if plot_ctrl else INJ_Y_LIM)

    # Q (bottom-right)
    for i in nodes_to_plot:
        lbl = _labelize(node_list[i])
        if plot_ctrl:
            q_control_action = sim_results['q_inj'][i, :] - sim_results['q_nominal'][i, :]
            ax_Q.plot(t, q_control_action, label=lbl)
        else:
            ax_Q.plot(t, sim_results['q_inj'][i, :], label=lbl)
    ax_Q.set_xlabel("Time (s)")
    ax_Q.set_ylabel("Reactive Power Control (MVar)" if plot_ctrl else "Reactive Power Injection (MVar)")
    ax_Q.grid(True, alpha=0.3)
    ax_Q.set_ylim(CONTROL_Y_LIM if plot_ctrl else INJ_Y_LIM)

    # Add vertical "Grid Fault" line at t=1s on all subplots
    for ax in [ax_slack, ax_V, ax_P, ax_Q]:
        ax.axvline(1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # Place the label slightly above the axis center
        ylim = ax.get_ylim()
        y_pos = ylim[0] + 0.7 * (ylim[1] - ylim[0])
        ax.text(1.02, y_pos, "Grid Fault", color='red', rotation=90, va='bottom')

    # Single legend outside on the right
    fig.subplots_adjust(right=0.80)  
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.82, 0.5), borderaxespad=0.)

    os.makedirs(out_path, exist_ok=True)
    for fmt in ext:
        plt.savefig(os.path.join(out_path, f"{filename}.{fmt}"), bbox_inches='tight', dpi=300)
    plt.close(fig)

    
def plot_simulation(
    sim_results, 
    filename, 
    nodes_to_plot=None, 
    plot_ctrl=False,
    out_path='temp', 
    ext=('pdf',),
    title=None,
    figsize=(10, 12),
    v_ylim=(0.7, 1.3),
    EXCLUDED_BUS_NAMES={"interconnection_a", "utility_a_pri", "mv_distribution_bus"},
    CONTROL_Y_LIM=(-8, 8),
    INJ_Y_LIM=(-5, 35),
    fontsize=22
):
    """
    Create and save a 3-row time-series figure: (1) slack + nodal voltages, (2) real power injections/control, (3) reactive power injections/control.

    :param sim_results: dict, simulation results with keys 't', 'v0', 'v_solution', 'p_inj', 'q_inj', 'p_nominal', 'q_nominal', 'node_list'
    :param filename: str, base filename (without extension) used to save the figure
    :param nodes_to_plot: iterable or None, indices or names of nodes to include (default: auto-select)
    :param plot_ctrl: bool, if True plot control actions (injections - nominal) instead of raw injections
    :param out_path: str, directory to save outputs
    :param ext: tuple, file extensions to write (e.g., ('pdf',))
    :param title: str or None, optional figure title
    :param figsize: tuple, figure size in inches
    :param v_ylim: tuple, y-limits for voltage axis
    :param EXCLUDED_BUS_NAMES: set, bus names to exclude from default selection
    :param CONTROL_Y_LIM: tuple, y-limits for control plots
    :param INJ_Y_LIM: tuple, y-limits for injection plots
    :param fontsize: int, base font size for the figure
    :return: None (saves figure files to out_path)
    """

    plt.rcParams['font.size'] = fontsize  # sets the default font size

    t = sim_results['t']
    node_list = sim_results['node_list'][1:]  # skip slack bus
    N = sim_results['v_solution'].shape[0]

    if not nodes_to_plot:
        nodes_to_plot = [i for i in range(N) if node_list[i] not in EXCLUDED_BUS_NAMES][:12]
    else:
        nodes_to_plot = [i for i in range(N) if node_list[i] not in EXCLUDED_BUS_NAMES and node_list[i] in nodes_to_plot]

    # 3x1 layout:
    # (0) slack + nodal voltages
    # (1) P
    # (2) Q
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    if title: 
        fig.suptitle(title)
    ax_V_all = axes[0]
    ax_P = axes[1]
    ax_Q = axes[2]

    # Slack bus voltage (black) + Nodal voltages
    h, = ax_V_all.plot(t, sim_results['v0'], label='Grid', color='black', linewidth=2)
    handles = [h]
    labels = ['Grid']
    for i in nodes_to_plot:
        raw_label = node_list[i]
        lbl = _labelize(raw_label)
        h, = ax_V_all.plot(t, sim_results['v_solution'][i, :], label=lbl)
        handles.append(h)
        labels.append(lbl)
    # ax_V_all.set_xlabel("Time (s)")
    ax_V_all.set_xticklabels([])
    ax_V_all.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax_V_all.set_ylabel("Voltage (per unit)")
    ax_V_all.grid(True, alpha=0.3)
    ax_V_all.set_ylim(v_ylim)

    # P subplot
    for i in nodes_to_plot:
        lbl = _labelize(node_list[i])
        if plot_ctrl:
            p_control_action = sim_results['p_inj'][i, :] - sim_results['p_nominal'][i, :]
            ax_P.plot(t, p_control_action, label=lbl)
        else:
            ax_P.plot(t, sim_results['p_inj'][i, :], label=lbl)
    # ax_P.set_xlabel("Time (s)")
    ax_P.set_xticklabels([])
    ax_P.set_ylabel("Real Power Control\n(MW)" if plot_ctrl else "Real Power Injection\n(MW)")
    ax_P.grid(True, alpha=0.3)
    ax_P.set_ylim(CONTROL_Y_LIM if plot_ctrl else INJ_Y_LIM)

    # Q subplot
    for i in nodes_to_plot:
        lbl = _labelize(node_list[i])
        if plot_ctrl:
            q_control_action = sim_results['q_inj'][i, :] - sim_results['q_nominal'][i, :]
            ax_Q.plot(t, q_control_action, label=lbl)
        else:
            ax_Q.plot(t, sim_results['q_inj'][i, :], label=lbl)
    ax_Q.set_xlabel("Time (s)")
    ax_Q.set_ylabel("Reactive Power Control\n(MVar)" if plot_ctrl else "Reactive Power Injection\n(MVar)")
    ax_Q.grid(True, alpha=0.3)
    ax_Q.set_ylim(CONTROL_Y_LIM if plot_ctrl else INJ_Y_LIM)

    # Add vertical "Grid Fault" line at t=1s on all subplots
    for ax in [ax_V_all, ax_P, ax_Q]: # ax_P, ax_Q
        ax.axvline(1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    for ax in [ax_V_all]: # ax_P, ax_Q
        ylim = ax.get_ylim()
        y_pos = ylim[0] + 0.52 * (ylim[1] - ylim[0])
        ax.text(1.02, y_pos, "Grid Fault", color='red', rotation=90, va='bottom')

    # Legend at the bottom
    plt.tight_layout(h_pad=0.5)
    fig.subplots_adjust(bottom=0.25)
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.49, 0.02),
        ncol=3,
        frameon=False,
        columnspacing=1
    )
    # # Single legend outside on the right
    # fig.subplots_adjust(right=0.80)  
    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.82, 0.5), borderaxespad=0.)
    
    os.makedirs(out_path, exist_ok=True)
    for fmt in ext:
        plt.savefig(os.path.join(out_path, f"{filename}.{fmt}"), bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_compute_load(p_rating=20, out_file='temp/compute_load.pdf'):
    """
    Plot a short example of compute load time-series using the bundled compute-power CSV.

    :param p_rating: float, peak compute capacity in MW used to scale the per-unit factor (default=20)
    :param out_file: str, output path for the saved figure (PDF)
    :return: None (saves plot to out_file)
    """

    plt.rcParams['font.size'] = 12  # sets the default font size
    compute_pwr_file='data/compute-power-readings/llama_b16_i512_o128_tp4/nvidiasmi_monitor.csv'
    t0, t1 = '2023/10/18 17:41:36.473', '2023/10/18 17:41:39.473'
    keys = {'t': 'timestamp', 'p': 'power.draw [W]'}
    compute_power = load_data.load_compute_power(compute_pwr_file, keys, t0, t1)
    # compute_pwr_file='data/gpu_compute_power_Choukse_power_stabilization.csv',
    # This is for Choukse's data 
    # keys = {'t': 'time_s', 'p': 'gpu_power_norm'}
    # compute_power = load_data.load_compute_power(compute_pwr_file, keys, 28, 40, datetime=False)

    plt.figure(figsize=(5, 2))
    y_vals = compute_power['compute_power_factor'] * p_rating
    plt.plot(compute_power['t'], y_vals, label='Compute Load', color=ORANGE, linewidth=0.5)

    # compute average utilization
    avg_val = y_vals.mean()
    plt.axhline(avg_val, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Mean')

    plt.xlabel("Time (s)")
    plt.ylabel("Compute Load (MW)")
    plt.ylim(0, p_rating * 1.25)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()


def plot_delay_ablation(delays, summaries, out_file='temp/delay_ablation.pdf'):
    """
    Plot ablation results over varying control delays and save to file.

    :param delays: array-like, control delay values (seconds) evaluated
    :param summaries: np.ndarray, shape (len(delays), >=4). Expected columns: [mean_voltage_dev, max_voltage_dev, real_power_effort, reactive_power_effort]
    :param out_file: str, output path for the saved figure (PDF)
    :return: None (saves plot to out_file)
    """

    plt.rcParams['font.size'] = 12  # sets the default font size
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    axs[0].plot(delays, summaries[:, 0], marker='o', label='Mean Voltage Deviation (p.u.)')
    axs[0].plot(delays, summaries[:, 1], marker='o', label='Maximum Voltage Deviation (p.u.)')
    # axs[0].set_xscale('log')
    axs[0].set_xlabel("Control Delay (s)")
    axs[0].set_ylabel("Voltage Deviation (p.u.)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(delays, summaries[:, 2], marker='o', label='Total Real Power Control Effort (MWh)')
    axs[1].plot(delays, summaries[:, 3], marker='o', label='Total Reactive Power Control Effort (MVarh)')
    # axs[1].set_xscale('log')
    axs[1].set_xlabel("Control Delay (s)")
    axs[1].set_ylabel("Control Effort (MVAh)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    ylim = axs[1].get_ylim()
    axs[1].set_ylim(0, ylim[1]*1.1)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()


def generate_summary_table(table_data, out_file='temp/summary_table.tex'):
    """
    Build a LaTeX table summarizing results from `table_data`,
    using wrapped columns (no resizebox) and 3-decimal formatting.
    """
    import os
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    row_map = [
        ("No Voltage Control", "No Ride-Through"),
        ("Centralized", "With Centralized Controller"),
        ("Decentralized", "With Decentralized Controller"),
        ("Centralized (reactive power)", "With Centralized Controller (Q only)"),
        ("Decentralized (reactive power)", "With Decentralized Controller (Q only)"),
        ("Centralized (200ms delay)", "Centralized Delay 0.2s"),
    ]

    def fmt(x):
        return f"{x:.3f}"

    rows = []
    for pretty, key in row_map:
        if key in table_data and table_data[key] is not None:
            avg, maxv, p_mw, q_mvar = table_data[key]
            rows.append(
                f"{pretty} & {fmt(maxv)} & {fmt(avg)} & {fmt(p_mw)} & {fmt(q_mvar)} \\\\"
            )
        else:
            rows.append(f"{pretty} & -- & -- & -- & -- \\\\")

    latex = (
        "\\begin{table*}[]\n"
        "    \\centering\n"
        "    \\caption{Performance Evaluation of Different Control Schemes}\n"
        "    \\begin{tabular}{p{4.0cm}|p{2.4cm}p{2.2cm}p{3.1cm}p{3.4cm}}\n"
        "         \\textbf{Control Scheme} & "
        "\\textbf{Largest Voltage Deviation (p.u.)} & "
        "\\textbf{Mean Voltage Deviation (p.u.)} & "
        "\\textbf{Average Real Power Control Effort (MW)} & "
        "\\textbf{Average Reactive Power Control Effort (MVAr)} \\\\\n"
        "         \\hline\n"
        f"         " + "\n         ".join(rows) + "\n"
        "    \\end{tabular}\n"
        "    \\label{tab:exp_metrics}\n"
        "\\end{table*}\n"
    )

    with open(out_file, "w") as f:
        f.write(latex)

    return latex


if __name__ == "__main__":
    plot_compute_load()

