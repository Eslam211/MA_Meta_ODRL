
Multi-agent reinforcement learning (MARL) has shown promise in high-performance computing and data-driven decision-making. However, conventional MARL faces significant challenges in real-world deployments, including unsafe online training, vulnerability to environmental uncertainties, and limited adaptability to dynamic configurations. To address these issues, we propose a meta-offline distributional MARL algorithm, termed meta-conservative quantile regression (M-CQR), which integrates conservative Q-learning (CQL), quantile regression deep Q-network (QR-DQN), and model-agnostic meta-learning (MAML). CQL enables safe offline learning from fixed datasets, QR-DQN models return distributions for risk sensitivity, and MAML supports fast adaptation to new environments. We develop two variants: independent training (M-I-CQR) and centralised training with decentralised execution (M-CTDE-CQR). Simulations in a UAV-based risk-aware communication environment demonstrate that M-CTDE-CQR achieves up to $50\%$ faster convergence in dynamic settings, outperforming baseline methods. The proposed framework offers enhanced scalability, robustness, and adaptability for real-world, risk-sensitive decision-making.

Make sure you run the online folder first and save the training / testing env parameters. Then, copy these parameters to the offline training folders.


We kindly ask you to cite our published paper: E. Eldeeb and H. Alves, "Meta-Offline and Distributional Multi-Agent RL for Risk-Aware Decision-Making," ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2026, pp. 20531-20535, doi: 10.1109/ICASSP55912.2026.11463052.

Also, avaialbale at: https://arxiv.org/abs/2501.16098
