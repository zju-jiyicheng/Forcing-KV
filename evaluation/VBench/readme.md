# Quick Evaluation for VBench / VBench-Long
You can use the `vbench.sh` and `vbenchlong.sh` script here. It includes:
- **Inference using the specified config file**
- **VBench Evaluation** (Need VBench Dependencies and a VBench codebose. Please see [VBench](https://github.com/Vchitect/VBench)).
- **Calculate the Final Score**

We use the VBench coefficients to calculate the total score, please see `scripts/cal_final_score.py` and `scripts/cal_long_final_score.py`.

In addition, we also provide the script of `drift` metrics.