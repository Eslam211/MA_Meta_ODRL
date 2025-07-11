Multi-agent reinforcement learning (MARL) has shown promise in high-performance computing and data-driven decision-making. However, conventional MARL faces significant challenges in real-world deployments, including unsafe online training, vulnerability to environmental uncertainties, and limited adaptability to dynamic configurations. To address these issues, we propose a meta-offline distributional MARL algorithm, termed meta-conservative quantile regression (M-CQR), which integrates conservative Q-learning (CQL), quantile regression deep Q-network (QR-DQN), and model-agnostic meta-learning (MAML). CQL enables safe offline learning from fixed datasets, QR-DQN models return distributions for risk sensitivity, and MAML supports fast adaptation to new environments. We develop two variants: independent training (M-I-CQR) and centralised training with decentralised execution (M-CTDE-CQR). Simulations in a UAV-based risk-aware communication environment demonstrate that M-CTDE-CQR achieves up to $50\%$ faster convergence in dynamic settings, outperforming baseline methods. The proposed framework offers enhanced scalability, robustness, and adaptability for real-world, risk-sensitive decision-making.


Make sure you run the online folder first and save the training / testing env parameters. Then, copy these parameters to the offline training folders.



