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
| Standard    | 0/255         |           0.6965 |                 0.6965 |                0.0000 |                  0.6817 |                        0.6817 |
| Standard    | 1/255         |           0.6965 |                 0.4405 |                0.3676 |                  0.6817 |                        0.6437 |
| Standard    | 2/255         |           0.6965 |                 0.2195 |                0.6849 |                  0.6817 |                        0.6674 |
| Standard    | 4/255         |           0.6965 |                 0.0315 |                0.9548 |                  0.6817 |                        0.7815 |
| Standard    | 8/255         |           0.6965 |                 0.0005 |                0.9993 |                  0.6817 |                        0.9015 |
| Adversarial | 0/255         |           0.5020 |                 0.5020 |                0.0000 |                  0.3546 |                        0.3546 |
| Adversarial | 1/255         |           0.5020 |                 0.4675 |                0.0707 |                  0.3546 |                        0.3493 |
| Adversarial | 2/255         |           0.5020 |                 0.4365 |                0.1335 |                  0.3546 |                        0.3448 |
| Adversarial | 4/255         |           0.5020 |                 0.3620 |                0.2829 |                  0.3546 |                        0.3379 |
| Adversarial | 8/255         |           0.5020 |                 0.2395 |                0.5249 |                  0.3546 |                        0.3337 |

## Automated Results Interpretation
- **Clean Performance:** The **Standard** model achieved higher clean accuracy (Standard: 69.65%, Adversarial: 50.20%).
- **Adversarial Robustness (at eps=8/255):** The **Adversarial** model achieved higher PGD accuracy (Standard: 0.05%, Adversarial: 23.95%) with Attack Success Rates of 99.93% (Standard) vs. 52.49% (Adversarial).
- **Accuracy–Robustness Trade-Off:** Clean accuracy shifted by -19.45%, while PGD accuracy at eps=8/255 shifted by +23.90%.
