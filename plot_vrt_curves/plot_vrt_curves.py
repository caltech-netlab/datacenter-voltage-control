#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12

x_min = 0.1

df_lvrt = pd.read_csv("grid_codes.csv")
# Extend to the y-axis
ext_rows = []
for name, g in df_lvrt.groupby("curve"):
    g2 = g.sort_values("time_ms")
    y0 = float(g2.iloc[0]["voltage_pct"])
    ext_rows.append({"curve": name, "time_ms": x_min, "voltage_pct": y0})
df_lvrt = pd.concat([df_lvrt, pd.DataFrame(ext_rows)], ignore_index=True).sort_values(["curve","time_ms"])
df_lvrt = df_lvrt.groupby("curve")

df_iec = pd.read_csv("iec62040-3_curve3_combined.csv")
df_iti = pd.read_csv("iti_cbema.csv")

styles = {
  "ITI_CBEMA_UV": {"color": (0.7, 0.7, 0.7), "dash": "-", "lw": 2.5, "z": 0, 'name': 'ITI/CBEMA (Standard)'},
  "ENTSO-E_recommendation_for_data_centres": {"color": (1.00, 0.43, 0.12), "dash": (0, (4, 2)), "lw": 2.5, "z": 1, 'name': 'ENTSO-E (Recommendation)'},
  "RTE_France_Proposal": {"color": (0.19, 0.63, 0.29), "dash": (0, (4, 2)),   "lw": 2.5, "z": 1, 'name': 'RTE France (Proposal)'},
  "ERCOT_Proposal": {"color": (0.95, 0.76, 0.06), "dash": (0, (2, 1)),      "lw": 2.5, "z": 10, 'name': 'ERCOT (Proposal)'},
  "EirGrid_Proposal": {"color": (0.55, 0.27, 0.74), "dash": (0, (6, 3)),  "lw": 2.5, "z": 1, 'name': 'EirGrid (Proposal)'},
  "Energinet_Teknisk_forskrift_3.4.3": {"color": (0.80, 0.29, 0.37), "dash": (0, (4, 2)), "lw": 2.5, "z": 1, 'name': 'Energinet (Grid Code)'},
}
sorted_list = [
  "ITI_CBEMA_UV",
  "Energinet_Teknisk_forskrift_3.4.3",
  "ERCOT_Proposal",
  "EirGrid_Proposal",
  "RTE_France_Proposal",
  # "ENTSO-E_recommendation_for_data_centres",
]


fig, ax = plt.subplots(figsize=(10,4), dpi=160)

# IEC combined; NaNs break the polyline automatically
ax.plot(df_iec["time_ms"].values, df_iec["voltage_pct"].values,
        color=(0.0, 0.0, 0.0), linestyle="-", linewidth=2.5, zorder=0, 
        label="IEC 62040-3 Curve 3 (Standard)")
# # ITI/CBEMA combined; NaNs break the polyline automatically
# ax.plot(df_iti["time_ms"].values, df_iti["voltage_pct"].values,
#         color=styles["ITI_CBEMA_UV"]["color"], linestyle=styles["ITI_CBEMA_UV"]["dash"], 
#         linewidth=styles["ITI_CBEMA_UV"]["lw"], zorder=styles["ITI_CBEMA_UV"]["z"],
#         label=styles["ITI_CBEMA_UV"]['name'])

# The rest
for name in sorted_list:
    g = df_lvrt.get_group(name)
    st = styles[name]
    ax.plot(g["time_ms"].values, g["voltage_pct"].values,
            color=tuple(st["color"]), linestyle=st["dash"], 
            linewidth=st["lw"], zorder=st["z"],
            label=st['name'])

ax.set_xscale('log'); ax.set_xlim(0.1, 10000)
ax.set_ylim(-10, 210)
ax.set_yticks(range(0, 201, 20))
ax.set_xlabel('time [ms]'); ax.set_ylabel('Voltage [%]')
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(loc='upper right', frameon=True, edgecolor='none', ncol=2)

# Add text annontations
# Add text annotations with highlight
ax.text(
0.95, 0.05, "Disconnect Possible",
ha='right', va='bottom',
transform=ax.transAxes,
fontsize=12, color='black', fontweight='bold',
bbox=dict(facecolor='red', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3'),
zorder=100
)

ax.text(
0.05, 0.5, "Remain Connected",
ha='left', va='center',
transform=ax.transAxes,
fontsize=12, color='black', fontweight='bold',
bbox=dict(facecolor='green', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3'),
zorder=100
)

ax.text(
0.32, 0.22, "————Performance Gap————",
ha='left', va='center',
transform=ax.transAxes,
fontsize=12, color='black', fontweight='bold',
bbox=dict(facecolor='orange', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3'),
zorder=100
)


plt.tight_layout()
plt.savefig("vrt_standards.png")
plt.savefig("vrt_standards.pdf")
print("Saved figure as PNG and PDF.")
