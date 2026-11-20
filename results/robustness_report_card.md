# CIFAR-10 Robustness Report Card

## Model & Experiment Configuration
- **Architecture:** `CIFAR10CNN` (304,794 trainable parameters)
- **Device:** `cpu` | **Seed:** `42`
- **Dataset Subsets:** Train+Val = `10000`, Test = `2000`
- **Standard Training:** `12` epochs, LR = `0.001`, Optimizer = `adam`
- **Adversarial Training (PGD):** `8` epochs, eps = `4/255`, steps = `3`

## Consolidated Evaluation Table (PGD)

| model       | epsilon_255   |   clean_accuracy |   adversarial_accuracy |   attack_success_rate |   mean_clean_confidence |   mean_adversarial_confidence |
|:------------|:--------------|-----------------:|-----------------------:|----------------------:|------------------------:|------------------------------:|
| Standard    | 0/255         |           0.6870 |                 0.6870 |                0.0000 |                  0.6631 |                        0.6631 |
| Standard    | 1/255         |           0.6870 |                 0.4090 |                0.4047 |                  0.6631 |                        0.6257 |
| Standard    | 2/255         |           0.6870 |                 0.1985 |                0.7111 |                  0.6631 |                        0.6477 |
| Standard    | 4/255         |           0.6870 |                 0.0225 |                0.9672 |                  0.6631 |                        0.7635 |
| Standard    | 8/255         |           0.6870 |                 0.0000 |                1.0000 |                  0.6631 |                        0.8933 |
| Adversarial | 0/255         |           0.5020 |                 0.5020 |                0.0000 |                  0.3546 |                        0.3546 |
| Adversarial | 1/255         |           0.5020 |                 0.4675 |                0.0707 |                  0.3546 |                        0.3493 |
| Adversarial | 2/255         |           0.5020 |                 0.4365 |                0.1335 |                  0.3546 |                        0.3448 |
| Adversarial | 4/255         |           0.5020 |                 0.3620 |                0.2829 |                  0.3546 |                        0.3379 |
| Adversarial | 8/255         |           0.5020 |                 0.2395 |                0.5249 |                  0.3546 |                        0.3337 |

## Automated Results Interpretation
- **Clean Performance:** The **Standard** model achieved higher clean accuracy (Standard: 68.70%, Adversarial: 50.20%).
- **Adversarial Robustness (at eps=8/255):** The **Adversarial** model achieved higher PGD accuracy (Standard: 0.00%, Adversarial: 23.95%) with Attack Success Rates of 100.00% (Standard) vs. 52.49% (Adversarial).
- **Accuracy–Robustness Trade-Off:** Clean accuracy shifted by -18.50%, while PGD accuracy at eps=8/255 shifted by +23.95%.
