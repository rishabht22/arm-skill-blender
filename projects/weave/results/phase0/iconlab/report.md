# WEAVE Phase 0 measurements

Git `de1f3a492066485ddb1f78c1dee287dc2a5c4722` · config `e20fe68b10c3ee0371650843476f6fcdeec5d86852a101427364b3d744d43570` · implementation `c7ce8eb5fca818d70e04f48b88c795c27db83f09b4298b0541c1f478c7ac100c`

## Recording inventory

| Task | Recorded ID | Arms | Train / validation / test | RGB streams | State snapshot |
| --- | --- | ---: | --- | ---: | --- |
| PickMeat | PickMeat | 1 | 105 / 22 / 23 | 1 | absent |
| StackCube | StackCube | 1 | 105 / 22 / 23 | 1 | absent |
| PassShoe | BrushShoe | 2 | 105 / 22 / 23 | 3 | absent |
| PlaceFood | IngredientsPlace | 2 | 105 / 22 / 23 | 3 | absent |
| TwoRobotsStackCube | TwoRobotStackCube | 2 | 105 / 22 / 23 | 3 | absent |

## Native open-loop replay

| Task | Raw success | Delta success | Repeat agreement | BC precheck | Control Hz |
| --- | --- | --- | --- | --- | ---: |
| PickMeat | 10.0% | 10.0% | 96.0% | fail | 20 |
| StackCube | 0.0% | 0.0% | 96.0% | fail | 20 |
| PassShoe | 2.0% | 0.0% | 90.0% | fail | 20 |
| PlaceFood | 36.0% | 0.0% | 24.0% | fail | 20 |
| TwoRobotsStackCube | 2.0% | 0.0% | 98.0% | fail | 20 |


## Native two-Panda throughput

| Task | Backend | Envs | Env steps/s | Process MiB | Incremental MiB/env | Contact probe |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PassShoe | gpu | 1 | 10 | 1666 | 1666.00 | pass |
| PassShoe | gpu | 64 | 627 | 1666 | 26.03 | pass |
| PassShoe | gpu | 256 | 2349 | 1794 | 7.01 | pass |
| PassShoe | gpu | 1024 | 9386 | 2118 | 2.07 | pass |
| PlaceFood | gpu | 1 | 10 | 1802 | 1802.00 | pass |
| PlaceFood | gpu | 64 | 765 | 1866 | 29.16 | pass |
| PlaceFood | gpu | 256 | 2243 | 1930 | 7.54 | pass |
| PlaceFood | gpu | 1024 | 8922 | 2448 | 2.39 | pass |
| TwoRobotsStackCube | gpu | 1 | 4 | 1826 | 1826.00 | pass |
| TwoRobotsStackCube | gpu | 64 | 228 | 1826 | 28.53 | pass |
| TwoRobotsStackCube | gpu | 256 | 818 | 1954 | 7.63 | pass |
| TwoRobotsStackCube | gpu | 1024 | 3201 | 2284 | 2.23 | pass |

G0 requires a reviewed environment branch decision in the audit.
