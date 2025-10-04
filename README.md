
Datacenter Voltage Control
==========================

This directory contains scripts and data for running datacenter voltage-control experiments.

Main entry
----------
The primary script is `simulator.py`. It runs simulations for centralized, decentralized, or no-control experiments
using either pre-recorded slack voltage traces or an Andes dynamic model.

Quick start
-----------
Install Python dependencies first and then run:

    pip install -r requirements.txt

From this directory run:

	python simulator.py

Notes
-----
- The repository expects some data files under `data/` (compute power traces, Andes network files, controller gains).
- See `visualize.py` for plotting helpers and `data/circuit_data.py` for example network definitions.



