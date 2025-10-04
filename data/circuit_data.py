"""
Conventions
- All power values are in MW or MVar
- Positive nodal power injection is defined as out of a node
"""


# System S_base = 1 MVA, V_base = 345kV, 16kV, 240V
# Z_base = V_base^2 / S_base = <not needed>
vulcan_circuit_data1 = [
    {
        'element_name': 'line_grid_utility_a_pri',
        'source': 'interconnection_a',
        'target': 'utility_a_pri',
        'series_impedance': 0.000571+0.006432j,           # 345kV line, 10 mile, in p.u.
    },
    {
        'element_name': 'tr_utility_a_mv',
        'source': 'utility_a_pri',
        'target': 'mv_distribution_bus',
        'series_impedance': 0.00003656+0.0006559j,      # 100MVA rating, in p.u.
    },
    {
        'element_name': 'tr_mv_bat_1',
        'source': 'mv_distribution_bus',
        'target': 'batt_1_sec',
        'series_impedance': 0.0001+0.003326j,           # 50MVA rating
    },
    {
        'element_name': 'tr_mv_bat_2',
        'source': 'mv_distribution_bus',
        'target': 'batt_2_sec',
        'series_impedance': 0.0002494+0.008315j,        # 20MVA rating  
    },
    {
        'element_name': 'tr_mv_cooling_1',
        'source': 'mv_distribution_bus',
        'target': 'cooling_1_sec',
        'series_impedance': 0.0002494+0.008315j,        # 20MVA rating  
    },
    {
        'element_name': 'tr_mv_datacenter_1',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_1_sec',
        'series_impedance': 0.0001995+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_2',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_2_sec',
        'series_impedance': 0.0001995+0.006652j,        # 25MVA rating
    },
]

vulcan_bus_data1 = {
    'interconnection_a': {'qmin': 0, 'qmax': 0, 'type': 'grid'},
    'utility_a_pri': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'mv_distribution_bus': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'batt_1_sec': {'qmin': -25, 'qmax': 25, 'pmin': -50, 'pmax': 50, 'type': 'battery'},
    'batt_2_sec': {'qmin': -25, 'qmax': 25, 'pmin': -50, 'pmax': 20, 'type': 'battery'},
    'cooling_1_sec': {'qmin': 0, 'qmax': 2, 'pmin': 0, 'pmax': 20, 'p_rating': 20, 'type': 'cooling'},
    'datacenter_1_sec': {'qmin': -20, 'qmax': 20, 'pmax': 25, 'type': 'compute'},
    'datacenter_2_sec': {'qmin': -20, 'qmax': 20, 'pmax': 25, 'type': 'compute'},
}

vulcan_circuit_data2 = [
    {
        'element_name': 'line_grid_utility_a_pri',
        'source': 'interconnection_a',
        'target': 'utility_a_pri',
        'series_impedance': 5e-5 + 5e-4j,              # 765kV line, ~0.1 mile, in p.u.
    },
    {
        'element_name': 'tr_utility_a_mv',
        'source': 'utility_a_pri',
        'target': 'mv_distribution_bus',
        'series_impedance': 9.14e-06+0.000163975j,      # 400MVA rating, in p.u.
    },
    {
        'element_name': 'tr_mv_bat_1',
        'source': 'mv_distribution_bus',
        'target': 'batt_1_sec',
        'series_impedance': 5e-05+0.001663j,            # 100MVA rating
    },
    {
        'element_name': 'tr_mv_bat_2',
        'source': 'mv_distribution_bus',
        'target': 'batt_2_sec',
        'series_impedance': 5e-05+0.001663j,            # 100MVA rating  
    },
    {
        'element_name': 'tr_mv_cooling_1',
        'source': 'mv_distribution_bus',
        'target': 'cooling_1_sec',
        'series_impedance': 6.235e-05+0.00207875j,        # 20MVA rating  
    },
    {
        'element_name': 'tr_mv_datacenter_1',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_1_sec',
        'series_impedance': 0.0001195+0.004652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_2',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_2_sec',
        'series_impedance': 0.0001995+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_3',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_3_sec',
        'series_impedance': 0.0001095+0.003652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_4',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_4_sec',
        'series_impedance': 0.0001495+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_5',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_5_sec',
        'series_impedance': 0.0001195+0.004652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_6',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_6_sec',
        'series_impedance': 0.0001295+0.003652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_7',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_7_sec',
        'series_impedance': 0.0001995+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_8',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_8_sec',
        'series_impedance': 0.0001+0.003652j,        # 25MVA rating
    },
]

vulcan_bus_data2 = {
    'interconnection_a': {'qmin': 0, 'qmax': 0, 'type': 'grid'},
    'utility_a_pri': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'mv_distribution_bus': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'batt_1_sec': {'qmin': -20, 'qmax': 20, 'pmin': -20, 'pmax': 20, 'type': 'battery'},
    'batt_2_sec': {'qmin': -20, 'qmax': 20, 'pmin': -20, 'pmax': 20, 'type': 'battery'},
    'cooling_1_sec': {'qmin': 0, 'qmax': 8, 'pmin': 0, 'pmax': 80, 'p_rating': 80, 'type': 'cooling'},
    'datacenter_1_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_2_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_3_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_4_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_5_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_6_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_7_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_8_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
}

vulcan_circuit_data3 = [
    {
        'element_name': 'line_grid_utility_a_pri',
        'source': 'interconnection_a',
        'target': 'utility_a_pri',
        'series_impedance': 5e-5 + 5e-4j,              # 765kV line, ~0.1 mile, in p.u.
    },
    {
        'element_name': 'tr_utility_a_mv',
        'source': 'utility_a_pri',
        'target': 'mv_distribution_bus',
        'series_impedance': 9.14e-06+0.000163975j,      # 400MVA rating, in p.u.
    },
    {
        'element_name': 'tr_mv_cooling_1',
        'source': 'mv_distribution_bus',
        'target': 'cooling_1_sec',
        'series_impedance': 6.235e-05+0.00207875j,      # 20MVA rating  
    },
    {
        'element_name': 'tr_mv_datacenter_1',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_1_sec',
        'series_impedance': 0.0001195+0.004652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_2',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_2_sec',
        'series_impedance': 0.0001995+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_3',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_3_sec',
        'series_impedance': 0.0001095+0.003652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_4',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_4_sec',
        'series_impedance': 0.0001495+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_5',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_5_sec',
        'series_impedance': 0.0001195+0.004652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_6',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_6_sec',
        'series_impedance': 0.0001295+0.003652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_7',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_7_sec',
        'series_impedance': 0.0001995+0.006652j,        # 25MVA rating
    },
    {
        'element_name': 'tr_mv_datacenter_8',
        'source': 'mv_distribution_bus',
        'target': 'datacenter_8_sec',
        'series_impedance': 0.0001+0.003652j,        # 25MVA rating
    },
]

vulcan_bus_data3 = {
    'interconnection_a': {'qmin': 0, 'qmax': 0, 'type': 'grid'},
    'utility_a_pri': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'mv_distribution_bus': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'cooling_1_sec': {'qmin': 0, 'qmax': 8, 'pmin': 0, 'pmax': 80, 'p_rating': 80, 'type': 'cooling'},
    'datacenter_1_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_2_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_3_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_4_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_5_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_6_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_7_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
    'datacenter_8_sec': {'qmin': -5, 'qmax': 5, 'pmin': -5, 'pmax': 5, 'p_rating': 20,'type': 'compute'},
}

"""No real power control"""
vulcan_bus_data4 = {
    'interconnection_a': {'qmin': 0, 'qmax': 0, 'type': 'grid'},
    'utility_a_pri': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'mv_distribution_bus': {'qmin': 0, 'qmax': 0, 'type': 'internal'},
    'batt_1_sec': {'qmin': -20, 'qmax': 20, 'pmin': 0, 'pmax': 0, 'type': 'battery'},
    'batt_2_sec': {'qmin': -20, 'qmax': 20, 'pmin': 0, 'pmax': 0, 'type': 'battery'},
    'cooling_1_sec': {'qmin': 0, 'qmax': 8, 'pmin': 0, 'pmax': 80, 'p_rating': 80, 'type': 'cooling'},
    'datacenter_1_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_2_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_3_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_4_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_5_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_6_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_7_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
    'datacenter_8_sec': {'qmin': -5, 'qmax': 5, 'pmin': 0, 'pmax': 0, 'p_rating': 20, 'type': 'compute'},
}



"""
Notes
Per-unit impedance reference from example openDSS:
1. 34.4kV:4.36kV
New Transformer.D1_1B Phases=3 Windings=2 XHL=6.55936480
~ wdg=1 bus=D1_East.1.2.3 conn=delta kV=34.399998 kva=3750.00000000 %R=0.36562806
~ wdg=2 bus=D1_1B.1.2.3.0 conn=wye kV=4.360000 kva=3750.00000000 %R=0.36562806
Z_pu = 0.003656+j0.06559
2. 2.4kV:240V
New Transformer.V89-0_B Phases=1 Windings=2 XHL=1.99557030
~ wdg=1 bus=UG6.2.0 conn=wye kV=2.401777 kva=100.00000000 %R=0.06651906
~ wdg=2 bus=V89-0.2.0 conn=wye kV=0.240000 kva=100.00000000 %R=0.06651906
Z _pu = 0.000665+j0.0346
3. 16kV:240V
New Transformer.662 Phases=1 Windings=2 XHL=4.98892593
~ wdg=1 bus=101.1.2 conn=delta kV=16.340000 kva=300.00000000 %R=0.16629764
~ wdg=2 bus=662.1.2 conn=delta kV=0.240000 kva=300.00000000 %R=0.16629764
Z_pu = 0.004988+j0.1663

https://www.eng-tips.com/threads/typical-transmission-line-parameters.65375/?utm_source=chatgpt.com
NOMINAL VOLTAGE CLASS (kV)    R (Ω/mi)          X (Ω/mi)    B (μmho/mi)
138                           0.10 to 0.25      0.60        4.5
230                           0.05 to 0.10      0.50        5.5
345                           0.03 to 0.05      0.40        6.0
500                           0.02 to 0.04      0.30        6.5

p.22, IA State, EE456 Transmission Line Design Information: Drive.google.com/drive/u/1/folders/0BzFTyjxorIWAeFo0dWtqVFRrb1k?resourcekey=0-sBzhb8ZPfbgfwJkWnte2Eg
Nominal_Voltage_kV    System_Strength_MVA3φ    Percent_X    Z_percent_per_mi          Y_percent_per_mi    SIL_MW
345                   30000                    0.333        0.00571+j0.06432        j0.6604             320
765                   66000                    0.151        0.00033+j0.00918        j4.689              2250
1100                  95000                    0.105        0.00007+j0.00598        j10.69              5180
1500                  130000                   0.077        0.00003+j0.00207        j20.51              9940

Based on Kersting's "Distribution System Modeling"
Voltage_kV    R_ohm_per_km      X_ohm_per_km      B_uS_per_km    X/R_ratio
23            0.30 to 0.50      0.15 to 0.25      1 to 3         2 to 4
13.8          0.40 to 0.60      0.20 to 0.30      0.8 to 2       2 to 3
12.47         0.45 to 0.65      0.22 to 0.33      0.7 to 1.8     2 to 3
4.16          0.80 to 1.50      0.35 to 0.70      0.2 to 0.5     2 to 3
0.48 (480 V)  3.00 to 5.00      0.50 to 1.20      ~0            1 to 2

"""