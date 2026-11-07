# Product Recommendation with Reinforcement Learning

Course project for **Applied Reinforcement Learning and Optimization (ARLO)**, IIT Kharagpur.

The goal is to recommend products on an e-commerce site as a **multi-armed bandit** problem. Each product is an arm, and a click is a Bernoulli reward with an unknown click-through rate (CTR). The agent has to balance exploring products against exploiting the best one it has found so far.

## Part 1: Beta-Bernoulli Thompson Sampling (done)

**Data:** [Open Bandit Dataset](https://research.zozo.com/data.html) (Saito et al., 2021), sample version: 10,000 real impressions of 80 fashion products from ZOZOTOWN, logged under a **uniform-random** policy, so the CTR estimates are unbiased. Overall CTR is 0.38%.

**Methods:** Random, ε-greedy, UCB1, and Beta-Bernoulli Thompson Sampling (implemented from scratch) with two priors.

**Evaluation:**
1. Online simulation in an environment calibrated to the real per-product CTRs (100k rounds, 8 runs)
2. Offline replay evaluation on the real logs (Li et al., 2011)

### Results (simulation)

| Policy | Cumulative regret | % of shows going to the top-5 products |
|---|---|---|
| Random | 2255 | 6% |
| UCB1 | 2200 | 8% |
| ε-greedy (ε = 0.1) | 800 | 84% |
| TS, uniform prior Beta(1,1) | 739 | 69% |
| **TS, base-rate prior Beta(1,262)** | **381** | **89%** |

- Thompson Sampling has the lowest regret. Its regret grows roughly logarithmically, while ε-greedy's grows linearly.
- When rewards are this sparse, a prior centred on the real base CTR (empirical Bayes) roughly **halves** the regret.
- UCB1's confidence bonus is far larger than the CTR gaps between products, so it barely does better than random.
- Replay on the 10k-row sample gives only about 125 matched events per pass, so all policies fall within noise of each other. The full dataset would be needed to separate them.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook ARLO_Thompson_Sampling_Recommendation.ipynb
```

The full run takes about 3 minutes on a laptop.

## Repository structure

```
├── ARLO_Thompson_Sampling_Recommendation.ipynb   # main notebook (with outputs)
├── data/random/all/                              # Open Bandit Dataset sample (uniform-random logs)
├── requirements.txt
└── README.md
```

## Next steps
- Contextual bandits (Linear TS / LinUCB) using the `user_feature_*` and `user-item_affinity_*` columns
- Sequential RL (Q-learning / DQN, policy gradient) for session-level recommendation

## References
1. Chapelle, O. & Li, L. (2011). An Empirical Evaluation of Thompson Sampling. *NeurIPS*.
2. Agrawal, S. & Goyal, N. (2012). Analysis of Thompson Sampling for the Multi-armed Bandit Problem. *COLT*.
3. Li, L., Chu, W., Langford, J., Wang, X. (2011). Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms. *WSDM*.
4. Saito, Y., Aihara, S., Matsutani, M., Narita, Y. (2021). Open Bandit Dataset and Pipeline. *NeurIPS Datasets & Benchmarks*.
5. Russo, D., Van Roy, B., Kazerouni, A., Osband, I., Wen, Z. (2018). A Tutorial on Thompson Sampling. *Foundations and Trends in ML*.
