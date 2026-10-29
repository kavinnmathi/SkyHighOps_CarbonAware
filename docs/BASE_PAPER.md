# Base paper

**Title:** Carbon-Aware Scheduling in Cloud Computing Operations: A Multi-Objective Optimisation Approach
**First author shown in the search result:** Kassem Danach (kassem.danach@mu.edu.lb). I only saw the first author in the page text I could read; check the full author list on the publisher page.
**Journal:** IET Smart Grid, Volume 9, Issue 1, article number e70056 (Wiley / IET)
**First published:** 14 January 2026 (shown on the publisher page)
**DOI:** 10.1049/stg2.70056 (taken from the article URL `https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/stg2.70056`)
**Type:** original research article (it proposes and evaluates a framework; it is not a review or survey)

## What I verified, and how
I verified the title, journal, volume/issue, article number, "First published" date and DOI from search-result excerpts of the publisher page. **The publisher site blocked automated access, so I could not read the full text.** The methodology and dataset statements below come from the abstract and section excerpts only. Please open the DOI link yourself and check them before citing details in your report.

## Methodology (from the excerpts I could read)
- A dynamic carbon-aware scheduling framework that combines (1) machine-learning based carbon-intensity forecasting, (2) multi-objective mixed-integer optimisation that minimises operating cost and carbon emissions together, and (3) adaptive rolling-horizon control.
- The workflow uses a centralised scheduler that interfaces with a simulated cloud cluster.
- Workload traces: job arrivals, durations and resource footprints derived from the public Google Cluster Workload Traces (2019 release).
- Grid data: UK National Grid data, with regional series for GB, FR and DE discussed in the data-acquisition section.

## How this project relates to the paper
This project is an **adaptation for an edge/cloud setting, not a replication**:

| Paper (per excerpts) | This project |
|---|---|
| Mixed-integer multi-objective optimisation, rolling horizon | Transparent heuristic scheduler (predicted-emissions minimisation + deferral + migration) |
| ML carbon forecasting | scikit-learn HistGradientBoostingRegressor forecaster, compared with persistence and seasonal-naive baselines |
| Google 2019 trace + UK National Grid | Same two source types, with an offline synthetic sample as default |
| Cloud cluster | Heterogeneous edge + cloud nodes with latency constraints and a migration-overhead model |

We do not claim to reproduce the paper's numbers.

## Suggested citation (check against the publisher page)
Danach, K. et al. (2026). Carbon-Aware Scheduling in Cloud Computing Operations: A Multi-Objective Optimisation Approach. *IET Smart Grid*, 9(1), e70056. https://doi.org/10.1049/stg2.70056
