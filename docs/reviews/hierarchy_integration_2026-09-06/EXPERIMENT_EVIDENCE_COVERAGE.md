# Experiment evidence coverage

All 15 groups have methods, results and detailed condition slots. These are evaluation/profiling cells, not counts of additional training runs. Verified ALC cells reuse six FRT runs. Pending values are not zero or favorable predictions.

| Group | Methods | Results | Table | Detailed slots | Status |
|---|---|---|---|---:|---|
| FRT | 3.6.3 | 4.8.1 | 4.6 | 36 | перевірено |
| ALC | 3.6.3 | 4.8.1 | 4.7 | 14 | частково перевірено: три повторно використані політики FRT |
| CEX | 3.6.4 | 4.8.2 | 4.8 | 12 | очікується |
| STS | 3.6.4 | 4.8.2 | 4.9 | 24 | очікується |
| LVL | 3.6.5 | 4.8.3 | 4.10 | 6 | очікується |
| OVR | 3.6.5 | 4.8.3 | 4.11 | 6 | очікується |
| CONF | 3.6.6 | 4.8.4 | 4.12 | 6 | очікується |
| MECH | 3.6.6 | 4.8.4 | 4.14 | 6 | очікується |
| VAL300 | 3.6.6 | 4.8.4 | 4.13 | 6 | очікується |
| ROLL64 | 3.6.6 | 4.8.4 | 4.15 | 16 | умовний дослід |
| PARITY | 3.7.7 | 4.9.4 | 4.18 | 24 | очікується |
| RESOURCE | 3.7.7 | 4.9.4 | 4.19 | 144 | очікується |
| SCREEN | 3.7.7 | 4.9.5 | 4.20 | 4 | очікується |
| QUALITY | 3.7.7 | 4.9.5 | 4.21 | 4 | очікується |
| COMP | 3.7.7 | 4.9.5 | 4.22 | 2 | умовний дослід |

The JSON inventory defines every dimension, dependency, status and verified artifact. Conditional rollout and comparator slots do not authorize additional runs.
