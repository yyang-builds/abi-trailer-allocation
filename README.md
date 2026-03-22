# ABI Trailer Allocation

`abi-trailer-allocation` is a research-oriented Python project on trailer loading and shipment planning under uncertainty, with a particular emphasis on reinforcement learning for dynamic trailer assignment.

This public repository reorganizes dissertation-era code into a modular, reproducible research codebase suitable for an academic portfolio and for future methodological extension.

**Related publication:** [ScienceDirect article](https://www.sciencedirect.com/science/article/abs/pii/S0925527323002116)

![Project Overview](outputs/generated/project_overview.svg)

## Problem Statement

This project studies the trailer shipment problem faced by ABI. The key challenge is to determine the proper weight of products loaded on a trailer while meeting the gross weight limit regulation, even though tractor weight is varying and unknown at the time when trailers are pre-loaded. This creates a trade-off between overloading, which leads to scaleback and rework cost, and underloading, which leads to opportunity cost from not fully utilizing trailer capacity.

## Key Results

The main finding is that reinforcement learning-based dynamic trailer assignment can significantly outperform static loading rules. In the dissertation, DTA achieved the best performance among the tested methods by using both diversification of trailer weights and dynamic use of updated tractor information. More broadly, the results highlight the operational value of flexibility in trailer shipment planning and show how dynamic assignment can reduce mismatch cost more effectively than purely static approaches.

![Method Cost Comparison](outputs/generated/method_cost_comparison.png)

![Distance vs Tractor Weight](outputs/generated/distance_vs_weight.png)

![One-Day Assignment Flow](outputs/generated/one_day_assignment_flow.png)

## Methods

This repository implements the main methodologies developed in the dissertation. As a benchmark, the Newsvendor method is used to determine trailer load size under uncertainty. Sample Average Approximation (SAA) and Empirical Risk Minimization (ERM) provide data-driven approaches for optimizing trailer weight using historical observations and shipment features. The central contribution is Dynamic Trailer Assignment (DTA), a reinforcement learning approach based on Q-learning that dynamically assigns pre-loaded trailers using realized tractor weight information and current trailer inventory.

The repository does not include proprietary ABI shipment data, private spreadsheets, or unrelated legacy files from the original working directory.

## Why This Repo Exists

The original dissertation working folder mixed together:

- repeated month-specific scripts such as `Feb_revised.py` and similar copies
- notebooks and checkpoint files
- Excel training and test workbooks
- draft documents and unrelated files
- method prototypes without a reusable package structure

This public repo reorganizes that work into a maintainable research codebase suitable for GitHub.

The core methodological framing is:

- `Newsvendor`: a Gaussian benchmark using historical tractor weight moments
- `SAA`: empirical optimization over observed historical samples
- `ERM`: a linear decision rule `q(d) = q0 + q1 d` trained by minimizing empirical mismatch cost
- `DTA`: a dynamic trailer assignment policy trained with tabular Q-learning

## Public Data Policy

This repo does **not** ship original ABI data.

Included instead:

- a small synthetic shipment dataset for demos and tests
- a data interface that can be pointed at a local private CSV
- documentation for the expected data schema

## Repository Layout

```text
abi-trailer-allocation/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
├── configs/
├── src/
├── scripts/
├── notebooks/
├── outputs/
└── tests/
```

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the bundled public demo:

```bash
python3 scripts/run_saa.py
python3 scripts/run_erm.py
python3 scripts/run_dta.py
python3 scripts/make_figures.py
```

Generated artifacts are written to `outputs/generated/`.

## Demo Outputs

The public demo can generate:

- tractor weight vs. distance scatter plot
- method cost comparison chart
- one-day assignment flow visualization
- inventory sensitivity plot

## Data Schema

Expected columns for local private data:

- `shipment_id`
- `ship_date`
- `distance_miles`
- `tractor_weight_lbs`
- `carrier_group` (optional)

## Notes On Reproducibility

- The bundled synthetic data is for demonstration only.
- Public results will not numerically match the original dissertation because the original operational data is not included.
- The ERM implementation uses SciPy rather than Gurobi to keep the public repo easier to reproduce.

## Next Extensions

Natural future extensions include:

- rolling monthly backtests
- carrier-specific ERM rules
- deeper RL variants
- richer synthetic scenario generation
