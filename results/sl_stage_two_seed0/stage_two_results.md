# SL stage two: first run

Arms: linear_prior, film_prior_dE, film_dE, measured; PARIS pairs from outputs/sl/paris/paris_seed0/pairs.parquet (173809). Training rows 42519 over 29 single-cell SL training lines; held-out rows 16152; left out: training rows gene not decoded 3621, held rows gene not decoded 1555, held rows pair not in training 17. Top K = 20. Epochs chosen on validation: paris_head_context_ablated 23, linear_prior/paris_head 13, film_prior_dE/paris_head 10, film_prior_dE/v_head 10, film_prior_dE/v_head_context_ablated 45, film_dE/paris_head 18, film_dE/v_head 12, film_dE/v_head_context_ablated 35, measured/paris_head 48. Nothing is chosen here; this is not a calibrated SL probability.

| Split | Screen | Line | Arm | Scorer | Pairs | SL | AUPRC | Lift | AUROC | P@K | Varying pairs | Varying SL | Varying AUPRC |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | harle_2025 | A549 | film_dE | a_blind | 340 | 23 | 0.1106 | 1.63 | 0.6615 | 0.10 | 62 | 13 | 0.2044 |
| test | harle_2025 | A549 | film_dE | pair_identity | 340 | 23 | 0.7350 | 10.87 | 0.9606 | 0.80 | 62 | 13 | 0.6629 |
| test | harle_2025 | A549 | film_dE | pan_essentiality | 340 | 23 | 0.1085 | 1.60 | 0.6620 | 0.05 | 62 | 13 | 0.2076 |
| test | harle_2025 | A549 | film_dE | paris | 340 | 23 | 0.2560 | 3.78 | 0.8111 | 0.25 | 62 | 13 | 0.3739 |
| test | harle_2025 | A549 | film_dE | paris_head | 340 | 23 | 0.3331 | 4.92 | 0.8733 | 0.30 | 62 | 13 | 0.3629 |
| test | harle_2025 | A549 | film_dE | paris_head_context_ablated | 340 | 23 | 0.3671 | 5.43 | 0.8867 | 0.35 | 62 | 13 | 0.3480 |
| test | harle_2025 | A549 | film_dE | v_head | 340 | 23 | 0.2614 | 3.86 | 0.8325 | 0.30 | 62 | 13 | 0.3243 |
| test | harle_2025 | A549 | film_dE | v_head_context_ablated | 340 | 23 | 0.7003 | 10.35 | 0.9407 | 0.65 | 62 | 13 | 0.5975 |
| test | harle_2025 | A549 | film_prior_dE | a_blind | 340 | 23 | 0.1057 | 1.56 | 0.6577 | 0.05 | 62 | 13 | 0.2021 |
| test | harle_2025 | A549 | film_prior_dE | pair_identity | 340 | 23 | 0.7350 | 10.87 | 0.9606 | 0.80 | 62 | 13 | 0.6629 |
| test | harle_2025 | A549 | film_prior_dE | pan_essentiality | 340 | 23 | 0.1085 | 1.60 | 0.6620 | 0.05 | 62 | 13 | 0.2076 |
| test | harle_2025 | A549 | film_prior_dE | paris | 340 | 23 | 0.2560 | 3.78 | 0.8111 | 0.25 | 62 | 13 | 0.3739 |
| test | harle_2025 | A549 | film_prior_dE | paris_head | 340 | 23 | 0.3132 | 4.63 | 0.8667 | 0.25 | 62 | 13 | 0.3436 |
| test | harle_2025 | A549 | film_prior_dE | paris_head_context_ablated | 340 | 23 | 0.3671 | 5.43 | 0.8867 | 0.35 | 62 | 13 | 0.3480 |
| test | harle_2025 | A549 | film_prior_dE | v_head | 340 | 23 | 0.2100 | 3.10 | 0.8048 | 0.20 | 62 | 13 | 0.3586 |
| test | harle_2025 | A549 | film_prior_dE | v_head_context_ablated | 340 | 23 | 0.7126 | 10.53 | 0.9569 | 0.80 | 62 | 13 | 0.7096 |
| test | harle_2025 | A549 | linear_prior | a_blind | 340 | 23 | 0.1055 | 1.56 | 0.6570 | 0.05 | 62 | 13 | 0.2060 |
| test | harle_2025 | A549 | linear_prior | pair_identity | 340 | 23 | 0.7350 | 10.87 | 0.9606 | 0.80 | 62 | 13 | 0.6629 |
| test | harle_2025 | A549 | linear_prior | pan_essentiality | 340 | 23 | 0.1085 | 1.60 | 0.6620 | 0.05 | 62 | 13 | 0.2076 |
| test | harle_2025 | A549 | linear_prior | paris | 340 | 23 | 0.2560 | 3.78 | 0.8111 | 0.25 | 62 | 13 | 0.3739 |
| test | harle_2025 | A549 | linear_prior | paris_head | 340 | 23 | 0.3193 | 4.72 | 0.8676 | 0.35 | 62 | 13 | 0.3428 |
| test | harle_2025 | A549 | linear_prior | paris_head_context_ablated | 340 | 23 | 0.3671 | 5.43 | 0.8867 | 0.35 | 62 | 13 | 0.3480 |
| test | harle_2025 | A549 | measured | a_blind | 340 | 23 | 0.1067 | 1.58 | 0.5833 | 0.15 | 62 | 13 | 0.1952 |
| test | harle_2025 | A549 | measured | pair_identity | 340 | 23 | 0.7350 | 10.87 | 0.9606 | 0.80 | 62 | 13 | 0.6629 |
| test | harle_2025 | A549 | measured | pan_essentiality | 340 | 23 | 0.1085 | 1.60 | 0.6620 | 0.05 | 62 | 13 | 0.2076 |
| test | harle_2025 | A549 | measured | paris | 340 | 23 | 0.2560 | 3.78 | 0.8111 | 0.25 | 62 | 13 | 0.3739 |
| test | harle_2025 | A549 | measured | paris_head | 340 | 23 | 0.4038 | 5.97 | 0.8719 | 0.45 | 62 | 13 | 0.3314 |
| test | harle_2025 | A549 | measured | paris_head_context_ablated | 340 | 23 | 0.3671 | 5.43 | 0.8867 | 0.35 | 62 | 13 | 0.3480 |
| test | harle_2025 | SK-MES-1 | film_dE | a_blind | 326 | 40 | 0.2275 | 1.85 | 0.6977 | 0.20 | 59 | 29 | 0.5670 |
| test | harle_2025 | SK-MES-1 | film_dE | pair_identity | 326 | 40 | 0.8441 | 6.88 | 0.9771 | 0.90 | 59 | 29 | 0.6819 |
| test | harle_2025 | SK-MES-1 | film_dE | pan_essentiality | 326 | 40 | 0.2261 | 1.84 | 0.7043 | 0.20 | 59 | 29 | 0.5528 |
| test | harle_2025 | SK-MES-1 | film_dE | paris | 326 | 40 | 0.3652 | 2.98 | 0.7496 | 0.45 | 59 | 29 | 0.5894 |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head | 326 | 40 | 0.5136 | 4.19 | 0.8318 | 0.55 | 59 | 29 | 0.6623 |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head_context_ablated | 326 | 40 | 0.5552 | 4.53 | 0.8438 | 0.65 | 59 | 29 | 0.6685 |
| test | harle_2025 | SK-MES-1 | film_dE | v_head | 326 | 40 | 0.4153 | 3.38 | 0.7946 | 0.60 | 59 | 29 | 0.5337 |
| test | harle_2025 | SK-MES-1 | film_dE | v_head_context_ablated | 326 | 40 | 0.7633 | 6.22 | 0.9547 | 0.80 | 59 | 29 | 0.6544 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | a_blind | 326 | 40 | 0.2242 | 1.83 | 0.7038 | 0.20 | 59 | 29 | 0.5824 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | pair_identity | 326 | 40 | 0.8441 | 6.88 | 0.9771 | 0.90 | 59 | 29 | 0.6819 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | pan_essentiality | 326 | 40 | 0.2261 | 1.84 | 0.7043 | 0.20 | 59 | 29 | 0.5528 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris | 326 | 40 | 0.3652 | 2.98 | 0.7496 | 0.45 | 59 | 29 | 0.5894 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head | 326 | 40 | 0.4749 | 3.87 | 0.8276 | 0.50 | 59 | 29 | 0.6485 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head_context_ablated | 326 | 40 | 0.5552 | 4.53 | 0.8438 | 0.65 | 59 | 29 | 0.6685 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | v_head | 326 | 40 | 0.3546 | 2.89 | 0.8076 | 0.40 | 59 | 29 | 0.5429 |
| test | harle_2025 | SK-MES-1 | film_prior_dE | v_head_context_ablated | 326 | 40 | 0.8290 | 6.76 | 0.9684 | 0.90 | 59 | 29 | 0.6960 |
| test | harle_2025 | SK-MES-1 | linear_prior | a_blind | 326 | 40 | 0.2310 | 1.88 | 0.7079 | 0.20 | 59 | 29 | 0.5741 |
| test | harle_2025 | SK-MES-1 | linear_prior | pair_identity | 326 | 40 | 0.8441 | 6.88 | 0.9771 | 0.90 | 59 | 29 | 0.6819 |
| test | harle_2025 | SK-MES-1 | linear_prior | pan_essentiality | 326 | 40 | 0.2261 | 1.84 | 0.7043 | 0.20 | 59 | 29 | 0.5528 |
| test | harle_2025 | SK-MES-1 | linear_prior | paris | 326 | 40 | 0.3652 | 2.98 | 0.7496 | 0.45 | 59 | 29 | 0.5894 |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head | 326 | 40 | 0.4897 | 3.99 | 0.8351 | 0.55 | 59 | 29 | 0.6287 |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head_context_ablated | 326 | 40 | 0.5552 | 4.53 | 0.8438 | 0.65 | 59 | 29 | 0.6685 |
| test | harle_2025 | SK-MES-1 | measured | a_blind | 326 | 40 | 0.1825 | 1.49 | 0.6134 | 0.05 | 59 | 29 | 0.5293 |
| test | harle_2025 | SK-MES-1 | measured | pair_identity | 326 | 40 | 0.8441 | 6.88 | 0.9771 | 0.90 | 59 | 29 | 0.6819 |
| test | harle_2025 | SK-MES-1 | measured | pan_essentiality | 326 | 40 | 0.2261 | 1.84 | 0.7043 | 0.20 | 59 | 29 | 0.5528 |
| test | harle_2025 | SK-MES-1 | measured | paris | 326 | 40 | 0.3652 | 2.98 | 0.7496 | 0.45 | 59 | 29 | 0.5894 |
| test | harle_2025 | SK-MES-1 | measured | paris_head | 326 | 40 | 0.5511 | 4.49 | 0.8177 | 0.65 | 59 | 29 | 0.6840 |
| test | harle_2025 | SK-MES-1 | measured | paris_head_context_ablated | 326 | 40 | 0.5552 | 4.53 | 0.8438 | 0.65 | 59 | 29 | 0.6685 |
| test | ito_2021 | A549_LUNG | film_dE | a_blind | 4272 | 156 | 0.1830 | 5.01 | 0.7450 | 0.40 | 320 | 59 | 0.2797 |
| test | ito_2021 | A549_LUNG | film_dE | pair_identity | 4272 | 156 | 0.3200 | 8.76 | 0.7627 | 0.75 | 320 | 59 | 0.3096 |
| test | ito_2021 | A549_LUNG | film_dE | pan_essentiality | 4272 | 156 | 0.1934 | 5.29 | 0.7812 | 0.35 | 320 | 59 | 0.2765 |
| test | ito_2021 | A549_LUNG | film_dE | paris | 4272 | 156 | 0.0901 | 2.47 | 0.5506 | 0.45 | 320 | 59 | 0.2410 |
| test | ito_2021 | A549_LUNG | film_dE | paris_head | 4272 | 156 | 0.2404 | 6.58 | 0.7552 | 0.65 | 320 | 59 | 0.2790 |
| test | ito_2021 | A549_LUNG | film_dE | paris_head_context_ablated | 4272 | 156 | 0.2484 | 6.80 | 0.7854 | 0.60 | 320 | 59 | 0.2994 |
| test | ito_2021 | A549_LUNG | film_dE | v_head | 4272 | 156 | 0.1821 | 4.99 | 0.7548 | 0.45 | 320 | 59 | 0.2410 |
| test | ito_2021 | A549_LUNG | film_dE | v_head_context_ablated | 4272 | 156 | 0.1988 | 5.44 | 0.7930 | 0.40 | 320 | 59 | 0.2655 |
| test | ito_2021 | A549_LUNG | film_prior_dE | a_blind | 4272 | 156 | 0.1930 | 5.29 | 0.7749 | 0.35 | 320 | 59 | 0.2894 |
| test | ito_2021 | A549_LUNG | film_prior_dE | pair_identity | 4272 | 156 | 0.3200 | 8.76 | 0.7627 | 0.75 | 320 | 59 | 0.3096 |
| test | ito_2021 | A549_LUNG | film_prior_dE | pan_essentiality | 4272 | 156 | 0.1934 | 5.29 | 0.7812 | 0.35 | 320 | 59 | 0.2765 |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris | 4272 | 156 | 0.0901 | 2.47 | 0.5506 | 0.45 | 320 | 59 | 0.2410 |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head | 4272 | 156 | 0.2415 | 6.61 | 0.7800 | 0.65 | 320 | 59 | 0.2880 |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head_context_ablated | 4272 | 156 | 0.2484 | 6.80 | 0.7854 | 0.60 | 320 | 59 | 0.2994 |
| test | ito_2021 | A549_LUNG | film_prior_dE | v_head | 4272 | 156 | 0.1058 | 2.90 | 0.6956 | 0.25 | 320 | 59 | 0.2012 |
| test | ito_2021 | A549_LUNG | film_prior_dE | v_head_context_ablated | 4272 | 156 | 0.1874 | 5.13 | 0.8019 | 0.40 | 320 | 59 | 0.2506 |
| test | ito_2021 | A549_LUNG | linear_prior | a_blind | 4272 | 156 | 0.1969 | 5.39 | 0.7729 | 0.40 | 320 | 59 | 0.2958 |
| test | ito_2021 | A549_LUNG | linear_prior | pair_identity | 4272 | 156 | 0.3200 | 8.76 | 0.7627 | 0.75 | 320 | 59 | 0.3096 |
| test | ito_2021 | A549_LUNG | linear_prior | pan_essentiality | 4272 | 156 | 0.1934 | 5.29 | 0.7812 | 0.35 | 320 | 59 | 0.2765 |
| test | ito_2021 | A549_LUNG | linear_prior | paris | 4272 | 156 | 0.0901 | 2.47 | 0.5506 | 0.45 | 320 | 59 | 0.2410 |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head | 4272 | 156 | 0.2505 | 6.86 | 0.7786 | 0.65 | 320 | 59 | 0.2870 |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head_context_ablated | 4272 | 156 | 0.2484 | 6.80 | 0.7854 | 0.60 | 320 | 59 | 0.2994 |
| test | ito_2021 | A549_LUNG | measured | a_blind | 4272 | 156 | 0.2143 | 5.87 | 0.7777 | 0.50 | 320 | 59 | 0.3077 |
| test | ito_2021 | A549_LUNG | measured | pair_identity | 4272 | 156 | 0.3200 | 8.76 | 0.7627 | 0.75 | 320 | 59 | 0.3096 |
| test | ito_2021 | A549_LUNG | measured | pan_essentiality | 4272 | 156 | 0.1934 | 5.29 | 0.7812 | 0.35 | 320 | 59 | 0.2765 |
| test | ito_2021 | A549_LUNG | measured | paris | 4272 | 156 | 0.0901 | 2.47 | 0.5506 | 0.45 | 320 | 59 | 0.2410 |
| test | ito_2021 | A549_LUNG | measured | paris_head | 4272 | 156 | 0.2964 | 8.12 | 0.7957 | 0.70 | 320 | 59 | 0.3383 |
| test | ito_2021 | A549_LUNG | measured | paris_head_context_ablated | 4272 | 156 | 0.2484 | 6.80 | 0.7854 | 0.60 | 320 | 59 | 0.2994 |
| test | klingbeil_2024 | A549 | film_dE | a_blind | 2224 | 108 | 0.3122 | 6.43 | 0.8294 | 0.55 | 359 | 84 | 0.4479 |
| test | klingbeil_2024 | A549 | film_dE | pair_identity | 2224 | 108 | 0.5305 | 10.92 | 0.8853 | 0.90 | 359 | 84 | 0.5635 |
| test | klingbeil_2024 | A549 | film_dE | pan_essentiality | 2224 | 108 | 0.3348 | 6.89 | 0.8455 | 0.60 | 359 | 84 | 0.4641 |
| test | klingbeil_2024 | A549 | film_dE | paris | 2224 | 108 | 0.1670 | 3.44 | 0.5748 | 0.65 | 359 | 84 | 0.3451 |
| test | klingbeil_2024 | A549 | film_dE | paris_head | 2224 | 108 | 0.3642 | 7.50 | 0.8563 | 0.70 | 359 | 84 | 0.5001 |
| test | klingbeil_2024 | A549 | film_dE | paris_head_context_ablated | 2224 | 108 | 0.3983 | 8.20 | 0.8637 | 0.75 | 359 | 84 | 0.5285 |
| test | klingbeil_2024 | A549 | film_dE | v_head | 2224 | 108 | 0.3042 | 6.26 | 0.8691 | 0.50 | 359 | 84 | 0.3922 |
| test | klingbeil_2024 | A549 | film_dE | v_head_context_ablated | 2224 | 108 | 0.3888 | 8.01 | 0.8881 | 0.75 | 359 | 84 | 0.4670 |
| test | klingbeil_2024 | A549 | film_prior_dE | a_blind | 2224 | 108 | 0.3284 | 6.76 | 0.8479 | 0.65 | 359 | 84 | 0.4621 |
| test | klingbeil_2024 | A549 | film_prior_dE | pair_identity | 2224 | 108 | 0.5305 | 10.92 | 0.8853 | 0.90 | 359 | 84 | 0.5635 |
| test | klingbeil_2024 | A549 | film_prior_dE | pan_essentiality | 2224 | 108 | 0.3348 | 6.89 | 0.8455 | 0.60 | 359 | 84 | 0.4641 |
| test | klingbeil_2024 | A549 | film_prior_dE | paris | 2224 | 108 | 0.1670 | 3.44 | 0.5748 | 0.65 | 359 | 84 | 0.3451 |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head | 2224 | 108 | 0.3858 | 7.94 | 0.8692 | 0.75 | 359 | 84 | 0.5116 |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head_context_ablated | 2224 | 108 | 0.3983 | 8.20 | 0.8637 | 0.75 | 359 | 84 | 0.5285 |
| test | klingbeil_2024 | A549 | film_prior_dE | v_head | 2224 | 108 | 0.2063 | 4.25 | 0.8002 | 0.45 | 359 | 84 | 0.3162 |
| test | klingbeil_2024 | A549 | film_prior_dE | v_head_context_ablated | 2224 | 108 | 0.3878 | 7.99 | 0.8941 | 0.60 | 359 | 84 | 0.4579 |
| test | klingbeil_2024 | A549 | linear_prior | a_blind | 2224 | 108 | 0.3336 | 6.87 | 0.8491 | 0.60 | 359 | 84 | 0.4655 |
| test | klingbeil_2024 | A549 | linear_prior | pair_identity | 2224 | 108 | 0.5305 | 10.92 | 0.8853 | 0.90 | 359 | 84 | 0.5635 |
| test | klingbeil_2024 | A549 | linear_prior | pan_essentiality | 2224 | 108 | 0.3348 | 6.89 | 0.8455 | 0.60 | 359 | 84 | 0.4641 |
| test | klingbeil_2024 | A549 | linear_prior | paris | 2224 | 108 | 0.1670 | 3.44 | 0.5748 | 0.65 | 359 | 84 | 0.3451 |
| test | klingbeil_2024 | A549 | linear_prior | paris_head | 2224 | 108 | 0.3776 | 7.78 | 0.8688 | 0.70 | 359 | 84 | 0.5123 |
| test | klingbeil_2024 | A549 | linear_prior | paris_head_context_ablated | 2224 | 108 | 0.3983 | 8.20 | 0.8637 | 0.75 | 359 | 84 | 0.5285 |
| test | klingbeil_2024 | A549 | measured | a_blind | 2224 | 108 | 0.3101 | 6.38 | 0.8273 | 0.65 | 359 | 84 | 0.4953 |
| test | klingbeil_2024 | A549 | measured | pair_identity | 2224 | 108 | 0.5305 | 10.92 | 0.8853 | 0.90 | 359 | 84 | 0.5635 |
| test | klingbeil_2024 | A549 | measured | pan_essentiality | 2224 | 108 | 0.3348 | 6.89 | 0.8455 | 0.60 | 359 | 84 | 0.4641 |
| test | klingbeil_2024 | A549 | measured | paris | 2224 | 108 | 0.1670 | 3.44 | 0.5748 | 0.65 | 359 | 84 | 0.3451 |
| test | klingbeil_2024 | A549 | measured | paris_head | 2224 | 108 | 0.3699 | 7.62 | 0.8575 | 0.65 | 359 | 84 | 0.5553 |
| test | klingbeil_2024 | A549 | measured | paris_head_context_ablated | 2224 | 108 | 0.3983 | 8.20 | 0.8637 | 0.75 | 359 | 84 | 0.5285 |
| test | klingbeil_2024 | MDAMB231 | film_dE | a_blind | 2172 | 134 | 0.3148 | 5.10 | 0.8021 | 0.65 | 328 | 97 | 0.4732 |
| test | klingbeil_2024 | MDAMB231 | film_dE | pair_identity | 2172 | 134 | 0.4800 | 7.78 | 0.8444 | 0.80 | 328 | 97 | 0.6014 |
| test | klingbeil_2024 | MDAMB231 | film_dE | pan_essentiality | 2172 | 134 | 0.3229 | 5.23 | 0.8163 | 0.60 | 328 | 97 | 0.4815 |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris | 2172 | 134 | 0.1019 | 1.65 | 0.5333 | 0.45 | 328 | 97 | 0.3326 |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head | 2172 | 134 | 0.3391 | 5.50 | 0.8166 | 0.65 | 328 | 97 | 0.4972 |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head_context_ablated | 2172 | 134 | 0.3579 | 5.80 | 0.8292 | 0.60 | 328 | 97 | 0.5167 |
| test | klingbeil_2024 | MDAMB231 | film_dE | v_head | 2172 | 134 | 0.3631 | 5.89 | 0.8374 | 0.60 | 328 | 97 | 0.5228 |
| test | klingbeil_2024 | MDAMB231 | film_dE | v_head_context_ablated | 2172 | 134 | 0.4797 | 7.77 | 0.8605 | 0.90 | 328 | 97 | 0.5887 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | a_blind | 2172 | 134 | 0.3420 | 5.54 | 0.8263 | 0.60 | 328 | 97 | 0.4976 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | pair_identity | 2172 | 134 | 0.4800 | 7.78 | 0.8444 | 0.80 | 328 | 97 | 0.6014 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | pan_essentiality | 2172 | 134 | 0.3229 | 5.23 | 0.8163 | 0.60 | 328 | 97 | 0.4815 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris | 2172 | 134 | 0.1019 | 1.65 | 0.5333 | 0.45 | 328 | 97 | 0.3326 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head | 2172 | 134 | 0.3675 | 5.96 | 0.8391 | 0.65 | 328 | 97 | 0.5195 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head_context_ablated | 2172 | 134 | 0.3579 | 5.80 | 0.8292 | 0.60 | 328 | 97 | 0.5167 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | v_head | 2172 | 134 | 0.2978 | 4.83 | 0.8165 | 0.60 | 328 | 97 | 0.4278 |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | v_head_context_ablated | 2172 | 134 | 0.3812 | 6.18 | 0.8717 | 0.60 | 328 | 97 | 0.5248 |
| test | klingbeil_2024 | MDAMB231 | linear_prior | a_blind | 2172 | 134 | 0.3370 | 5.46 | 0.8234 | 0.60 | 328 | 97 | 0.4970 |
| test | klingbeil_2024 | MDAMB231 | linear_prior | pair_identity | 2172 | 134 | 0.4800 | 7.78 | 0.8444 | 0.80 | 328 | 97 | 0.6014 |
| test | klingbeil_2024 | MDAMB231 | linear_prior | pan_essentiality | 2172 | 134 | 0.3229 | 5.23 | 0.8163 | 0.60 | 328 | 97 | 0.4815 |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris | 2172 | 134 | 0.1019 | 1.65 | 0.5333 | 0.45 | 328 | 97 | 0.3326 |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head | 2172 | 134 | 0.3614 | 5.86 | 0.8374 | 0.60 | 328 | 97 | 0.5196 |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head_context_ablated | 2172 | 134 | 0.3579 | 5.80 | 0.8292 | 0.60 | 328 | 97 | 0.5167 |
| test | klingbeil_2024 | MDAMB231 | measured | a_blind | 2172 | 134 | 0.3727 | 6.04 | 0.8278 | 0.65 | 328 | 97 | 0.5522 |
| test | klingbeil_2024 | MDAMB231 | measured | pair_identity | 2172 | 134 | 0.4800 | 7.78 | 0.8444 | 0.80 | 328 | 97 | 0.6014 |
| test | klingbeil_2024 | MDAMB231 | measured | pan_essentiality | 2172 | 134 | 0.3229 | 5.23 | 0.8163 | 0.60 | 328 | 97 | 0.4815 |
| test | klingbeil_2024 | MDAMB231 | measured | paris | 2172 | 134 | 0.1019 | 1.65 | 0.5333 | 0.45 | 328 | 97 | 0.3326 |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head | 2172 | 134 | 0.3925 | 6.36 | 0.8489 | 0.65 | 328 | 97 | 0.5699 |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head_context_ablated | 2172 | 134 | 0.3579 | 5.80 | 0.8292 | 0.60 | 328 | 97 | 0.5167 |
| val | harle_2025 | NCI-H1299 | film_dE | a_blind | 312 | 39 | 0.2263 | 1.81 | 0.6518 | 0.15 | 57 | 24 | 0.3377 |
| val | harle_2025 | NCI-H1299 | film_dE | pair_identity | 312 | 39 | 0.8804 | 7.04 | 0.9604 | 0.95 | 57 | 24 | 0.8085 |
| val | harle_2025 | NCI-H1299 | film_dE | pan_essentiality | 312 | 39 | 0.2273 | 1.82 | 0.6738 | 0.15 | 57 | 24 | 0.3441 |
| val | harle_2025 | NCI-H1299 | film_dE | paris | 312 | 39 | 0.3827 | 3.06 | 0.7497 | 0.50 | 57 | 24 | 0.5238 |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head | 312 | 39 | 0.5039 | 4.03 | 0.8193 | 0.55 | 57 | 24 | 0.5270 |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head_context_ablated | 312 | 39 | 0.5375 | 4.30 | 0.8288 | 0.70 | 57 | 24 | 0.5371 |
| val | harle_2025 | NCI-H1299 | film_dE | v_head | 312 | 39 | 0.4374 | 3.50 | 0.8383 | 0.45 | 57 | 24 | 0.5219 |
| val | harle_2025 | NCI-H1299 | film_dE | v_head_context_ablated | 312 | 39 | 0.8568 | 6.85 | 0.9683 | 0.95 | 57 | 24 | 0.8087 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | a_blind | 312 | 39 | 0.2255 | 1.80 | 0.6635 | 0.15 | 57 | 24 | 0.3460 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | pair_identity | 312 | 39 | 0.8804 | 7.04 | 0.9604 | 0.95 | 57 | 24 | 0.8085 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | pan_essentiality | 312 | 39 | 0.2273 | 1.82 | 0.6738 | 0.15 | 57 | 24 | 0.3441 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris | 312 | 39 | 0.3827 | 3.06 | 0.7497 | 0.50 | 57 | 24 | 0.5238 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head | 312 | 39 | 0.5039 | 4.03 | 0.8094 | 0.60 | 57 | 24 | 0.5307 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head_context_ablated | 312 | 39 | 0.5375 | 4.30 | 0.8288 | 0.70 | 57 | 24 | 0.5371 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | v_head | 312 | 39 | 0.3861 | 3.09 | 0.8100 | 0.55 | 57 | 24 | 0.5306 |
| val | harle_2025 | NCI-H1299 | film_prior_dE | v_head_context_ablated | 312 | 39 | 0.8363 | 6.69 | 0.9697 | 0.90 | 57 | 24 | 0.7389 |
| val | harle_2025 | NCI-H1299 | linear_prior | a_blind | 312 | 39 | 0.2281 | 1.82 | 0.6641 | 0.15 | 57 | 24 | 0.3447 |
| val | harle_2025 | NCI-H1299 | linear_prior | pair_identity | 312 | 39 | 0.8804 | 7.04 | 0.9604 | 0.95 | 57 | 24 | 0.8085 |
| val | harle_2025 | NCI-H1299 | linear_prior | pan_essentiality | 312 | 39 | 0.2273 | 1.82 | 0.6738 | 0.15 | 57 | 24 | 0.3441 |
| val | harle_2025 | NCI-H1299 | linear_prior | paris | 312 | 39 | 0.3827 | 3.06 | 0.7497 | 0.50 | 57 | 24 | 0.5238 |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head | 312 | 39 | 0.5088 | 4.07 | 0.8134 | 0.60 | 57 | 24 | 0.5216 |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head_context_ablated | 312 | 39 | 0.5375 | 4.30 | 0.8288 | 0.70 | 57 | 24 | 0.5371 |
| val | harle_2025 | NCI-H1299 | measured | a_blind | 312 | 39 | 0.2084 | 1.67 | 0.6174 | 0.25 | 57 | 24 | 0.3424 |
| val | harle_2025 | NCI-H1299 | measured | pair_identity | 312 | 39 | 0.8804 | 7.04 | 0.9604 | 0.95 | 57 | 24 | 0.8085 |
| val | harle_2025 | NCI-H1299 | measured | pan_essentiality | 312 | 39 | 0.2273 | 1.82 | 0.6738 | 0.15 | 57 | 24 | 0.3441 |
| val | harle_2025 | NCI-H1299 | measured | paris | 312 | 39 | 0.3827 | 3.06 | 0.7497 | 0.50 | 57 | 24 | 0.5238 |
| val | harle_2025 | NCI-H1299 | measured | paris_head | 312 | 39 | 0.4838 | 3.87 | 0.8151 | 0.65 | 57 | 24 | 0.5169 |
| val | harle_2025 | NCI-H1299 | measured | paris_head_context_ablated | 312 | 39 | 0.5375 | 4.30 | 0.8288 | 0.70 | 57 | 24 | 0.5371 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | a_blind | 4281 | 64 | 0.2375 | 15.89 | 0.8970 | 0.45 | 325 | 23 | 0.1686 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | pair_identity | 4281 | 64 | 0.3085 | 20.63 | 0.8382 | 0.75 | 325 | 23 | 0.1061 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | pan_essentiality | 4281 | 64 | 0.2359 | 15.78 | 0.9095 | 0.30 | 325 | 23 | 0.1601 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris | 4281 | 64 | 0.0705 | 4.72 | 0.6014 | 0.20 | 325 | 23 | 0.1398 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head | 4281 | 64 | 0.3288 | 21.99 | 0.9130 | 0.50 | 325 | 23 | 0.2250 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head_context_ablated | 4281 | 64 | 0.2883 | 19.28 | 0.9177 | 0.35 | 325 | 23 | 0.2242 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | v_head | 4281 | 64 | 0.2115 | 14.15 | 0.9204 | 0.40 | 325 | 23 | 0.2391 |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | v_head_context_ablated | 4281 | 64 | 0.1492 | 9.98 | 0.8998 | 0.30 | 325 | 23 | 0.1192 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | a_blind | 4281 | 64 | 0.2467 | 16.50 | 0.9008 | 0.50 | 325 | 23 | 0.1644 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | pair_identity | 4281 | 64 | 0.3085 | 20.63 | 0.8382 | 0.75 | 325 | 23 | 0.1061 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | pan_essentiality | 4281 | 64 | 0.2359 | 15.78 | 0.9095 | 0.30 | 325 | 23 | 0.1601 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris | 4281 | 64 | 0.0705 | 4.72 | 0.6014 | 0.20 | 325 | 23 | 0.1398 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head | 4281 | 64 | 0.3292 | 22.02 | 0.9144 | 0.55 | 325 | 23 | 0.2279 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head_context_ablated | 4281 | 64 | 0.2883 | 19.28 | 0.9177 | 0.35 | 325 | 23 | 0.2242 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | v_head | 4281 | 64 | 0.1370 | 9.16 | 0.8891 | 0.10 | 325 | 23 | 0.2070 |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | v_head_context_ablated | 4281 | 64 | 0.1503 | 10.05 | 0.8934 | 0.25 | 325 | 23 | 0.1199 |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | a_blind | 4281 | 64 | 0.2457 | 16.44 | 0.9095 | 0.40 | 325 | 23 | 0.1658 |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | pair_identity | 4281 | 64 | 0.3085 | 20.63 | 0.8382 | 0.75 | 325 | 23 | 0.1061 |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | pan_essentiality | 4281 | 64 | 0.2359 | 15.78 | 0.9095 | 0.30 | 325 | 23 | 0.1601 |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris | 4281 | 64 | 0.0705 | 4.72 | 0.6014 | 0.20 | 325 | 23 | 0.1398 |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head | 4281 | 64 | 0.3199 | 21.40 | 0.9191 | 0.50 | 325 | 23 | 0.2357 |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head_context_ablated | 4281 | 64 | 0.2883 | 19.28 | 0.9177 | 0.35 | 325 | 23 | 0.2242 |
| val | ito_2021 | PATU8988S_PANCREAS | measured | a_blind | 4281 | 64 | 0.2518 | 16.84 | 0.8852 | 0.40 | 325 | 23 | 0.1846 |
| val | ito_2021 | PATU8988S_PANCREAS | measured | pair_identity | 4281 | 64 | 0.3085 | 20.63 | 0.8382 | 0.75 | 325 | 23 | 0.1061 |
| val | ito_2021 | PATU8988S_PANCREAS | measured | pan_essentiality | 4281 | 64 | 0.2359 | 15.78 | 0.9095 | 0.30 | 325 | 23 | 0.1601 |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris | 4281 | 64 | 0.0705 | 4.72 | 0.6014 | 0.20 | 325 | 23 | 0.1398 |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head | 4281 | 64 | 0.4162 | 27.84 | 0.9269 | 0.60 | 325 | 23 | 0.2322 |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head_context_ablated | 4281 | 64 | 0.2883 | 19.28 | 0.9177 | 0.35 | 325 | 23 | 0.2242 |
| val | klingbeil_2024 | H1299 | film_dE | a_blind | 2225 | 76 | 0.1758 | 5.15 | 0.7891 | 0.35 | 363 | 52 | 0.2520 |
| val | klingbeil_2024 | H1299 | film_dE | pair_identity | 2225 | 76 | 0.4119 | 12.06 | 0.8497 | 0.75 | 363 | 52 | 0.4214 |
| val | klingbeil_2024 | H1299 | film_dE | pan_essentiality | 2225 | 76 | 0.1791 | 5.24 | 0.8026 | 0.25 | 363 | 52 | 0.2503 |
| val | klingbeil_2024 | H1299 | film_dE | paris | 2225 | 76 | 0.0896 | 2.62 | 0.5471 | 0.35 | 363 | 52 | 0.1885 |
| val | klingbeil_2024 | H1299 | film_dE | paris_head | 2225 | 76 | 0.2171 | 6.36 | 0.7979 | 0.30 | 363 | 52 | 0.2974 |
| val | klingbeil_2024 | H1299 | film_dE | paris_head_context_ablated | 2225 | 76 | 0.2224 | 6.51 | 0.8049 | 0.35 | 363 | 52 | 0.2995 |
| val | klingbeil_2024 | H1299 | film_dE | v_head | 2225 | 76 | 0.2019 | 5.91 | 0.8355 | 0.30 | 363 | 52 | 0.2838 |
| val | klingbeil_2024 | H1299 | film_dE | v_head_context_ablated | 2225 | 76 | 0.2904 | 8.50 | 0.8558 | 0.45 | 363 | 52 | 0.3171 |
| val | klingbeil_2024 | H1299 | film_prior_dE | a_blind | 2225 | 76 | 0.1879 | 5.50 | 0.8047 | 0.35 | 363 | 52 | 0.2632 |
| val | klingbeil_2024 | H1299 | film_prior_dE | pair_identity | 2225 | 76 | 0.4119 | 12.06 | 0.8497 | 0.75 | 363 | 52 | 0.4214 |
| val | klingbeil_2024 | H1299 | film_prior_dE | pan_essentiality | 2225 | 76 | 0.1791 | 5.24 | 0.8026 | 0.25 | 363 | 52 | 0.2503 |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris | 2225 | 76 | 0.0896 | 2.62 | 0.5471 | 0.35 | 363 | 52 | 0.1885 |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head | 2225 | 76 | 0.2173 | 6.36 | 0.8074 | 0.40 | 363 | 52 | 0.2956 |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head_context_ablated | 2225 | 76 | 0.2224 | 6.51 | 0.8049 | 0.35 | 363 | 52 | 0.2995 |
| val | klingbeil_2024 | H1299 | film_prior_dE | v_head | 2225 | 76 | 0.1506 | 4.41 | 0.7976 | 0.40 | 363 | 52 | 0.2072 |
| val | klingbeil_2024 | H1299 | film_prior_dE | v_head_context_ablated | 2225 | 76 | 0.2466 | 7.22 | 0.8456 | 0.30 | 363 | 52 | 0.3071 |
| val | klingbeil_2024 | H1299 | linear_prior | a_blind | 2225 | 76 | 0.1875 | 5.49 | 0.8134 | 0.30 | 363 | 52 | 0.2579 |
| val | klingbeil_2024 | H1299 | linear_prior | pair_identity | 2225 | 76 | 0.4119 | 12.06 | 0.8497 | 0.75 | 363 | 52 | 0.4214 |
| val | klingbeil_2024 | H1299 | linear_prior | pan_essentiality | 2225 | 76 | 0.1791 | 5.24 | 0.8026 | 0.25 | 363 | 52 | 0.2503 |
| val | klingbeil_2024 | H1299 | linear_prior | paris | 2225 | 76 | 0.0896 | 2.62 | 0.5471 | 0.35 | 363 | 52 | 0.1885 |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head | 2225 | 76 | 0.2203 | 6.45 | 0.8148 | 0.30 | 363 | 52 | 0.2949 |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head_context_ablated | 2225 | 76 | 0.2224 | 6.51 | 0.8049 | 0.35 | 363 | 52 | 0.2995 |
| val | klingbeil_2024 | H1299 | measured | a_blind | 2225 | 76 | 0.2272 | 6.65 | 0.8531 | 0.25 | 363 | 52 | 0.2992 |
| val | klingbeil_2024 | H1299 | measured | pair_identity | 2225 | 76 | 0.4119 | 12.06 | 0.8497 | 0.75 | 363 | 52 | 0.4214 |
| val | klingbeil_2024 | H1299 | measured | pan_essentiality | 2225 | 76 | 0.1791 | 5.24 | 0.8026 | 0.25 | 363 | 52 | 0.2503 |
| val | klingbeil_2024 | H1299 | measured | paris | 2225 | 76 | 0.0896 | 2.62 | 0.5471 | 0.35 | 363 | 52 | 0.1885 |
| val | klingbeil_2024 | H1299 | measured | paris_head | 2225 | 76 | 0.2623 | 7.68 | 0.8464 | 0.45 | 363 | 52 | 0.3286 |
| val | klingbeil_2024 | H1299 | measured | paris_head_context_ablated | 2225 | 76 | 0.2224 | 6.51 | 0.8049 | 0.35 | 363 | 52 | 0.2995 |

## AUPRC contrasts (95% two-way dyadic bootstrap)

| Split | Screen | Line | Arm | Contrast | Difference |
| --- | --- | --- | --- | --- | --- |
| test | harle_2025 | A549 | film_dE | paris_head - pair_identity | -0.4019 [-0.6638, -0.0812] |
| test | harle_2025 | A549 | film_dE | paris_head - pan_essentiality | 0.2246 [0.0244, 0.5039] |
| test | harle_2025 | A549 | film_dE | paris_head - a_blind | 0.2225 [-0.0090, 0.4992] |
| test | harle_2025 | A549 | film_dE | paris_head - paris_head_context_ablated | -0.0340 [-0.1620, 0.0469] |
| test | harle_2025 | A549 | film_dE | paris_head - paris | 0.0771 [-0.1997, 0.2765] |
| test | harle_2025 | A549 | film_dE | v_head - pair_identity | -0.4736 [-0.7285, -0.1210] |
| test | harle_2025 | A549 | film_dE | v_head - pan_essentiality | 0.1529 [-0.0346, 0.4212] |
| test | harle_2025 | A549 | film_dE | v_head - a_blind | 0.1508 [-0.0554, 0.4552] |
| test | harle_2025 | A549 | film_dE | v_head - v_head_context_ablated | -0.4389 [-0.6306, -0.1617] |
| test | harle_2025 | A549 | film_prior_dE | paris_head - pair_identity | -0.4218 [-0.6988, -0.1065] |
| test | harle_2025 | A549 | film_prior_dE | paris_head - pan_essentiality | 0.2047 [0.0025, 0.4873] |
| test | harle_2025 | A549 | film_prior_dE | paris_head - a_blind | 0.2075 [-0.0085, 0.4753] |
| test | harle_2025 | A549 | film_prior_dE | paris_head - paris_head_context_ablated | -0.0539 [-0.2088, 0.0524] |
| test | harle_2025 | A549 | film_prior_dE | paris_head - paris | 0.0572 [-0.2132, 0.2669] |
| test | harle_2025 | A549 | film_prior_dE | v_head - pair_identity | -0.5250 [-0.7788, -0.1196] |
| test | harle_2025 | A549 | film_prior_dE | v_head - pan_essentiality | 0.1015 [-0.0652, 0.3977] |
| test | harle_2025 | A549 | film_prior_dE | v_head - a_blind | 0.1043 [-0.0565, 0.3720] |
| test | harle_2025 | A549 | film_prior_dE | v_head - v_head_context_ablated | -0.5026 [-0.7626, -0.1622] |
| test | harle_2025 | A549 | linear_prior | paris_head - pair_identity | -0.4157 [-0.7012, -0.1020] |
| test | harle_2025 | A549 | linear_prior | paris_head - pan_essentiality | 0.2108 [0.0121, 0.5051] |
| test | harle_2025 | A549 | linear_prior | paris_head - a_blind | 0.2139 [-0.0092, 0.5087] |
| test | harle_2025 | A549 | linear_prior | paris_head - paris_head_context_ablated | -0.0478 [-0.1744, 0.0678] |
| test | harle_2025 | A549 | linear_prior | paris_head - paris | 0.0633 [-0.2075, 0.2889] |
| test | harle_2025 | A549 | measured | paris_head - pair_identity | -0.3313 [-0.6122, 0.0268] |
| test | harle_2025 | A549 | measured | paris_head - pan_essentiality | 0.2952 [0.0300, 0.6081] |
| test | harle_2025 | A549 | measured | paris_head - a_blind | 0.2971 [-0.0407, 0.6446] |
| test | harle_2025 | A549 | measured | paris_head - paris_head_context_ablated | 0.0366 [-0.1062, 0.2012] |
| test | harle_2025 | A549 | measured | paris_head - paris | 0.1477 [-0.1071, 0.3771] |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head - pair_identity | -0.3305 [-0.5402, -0.0927] |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head - pan_essentiality | 0.2874 [0.0257, 0.4936] |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head - a_blind | 0.2861 [0.0509, 0.5095] |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head - paris_head_context_ablated | -0.0417 [-0.1344, 0.0417] |
| test | harle_2025 | SK-MES-1 | film_dE | paris_head - paris | 0.1484 [-0.0239, 0.3051] |
| test | harle_2025 | SK-MES-1 | film_dE | v_head - pair_identity | -0.4288 [-0.6081, -0.1636] |
| test | harle_2025 | SK-MES-1 | film_dE | v_head - pan_essentiality | 0.1891 [-0.0341, 0.4214] |
| test | harle_2025 | SK-MES-1 | film_dE | v_head - a_blind | 0.1878 [-0.0143, 0.4321] |
| test | harle_2025 | SK-MES-1 | film_dE | v_head - v_head_context_ablated | -0.3480 [-0.5278, -0.1022] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head - pair_identity | -0.3692 [-0.5629, -0.1146] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head - pan_essentiality | 0.2488 [0.0226, 0.4808] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head - a_blind | 0.2507 [0.0223, 0.4770] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head - paris_head_context_ablated | -0.0803 [-0.1858, 0.0267] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | paris_head - paris | 0.1097 [-0.0543, 0.2786] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | v_head - pair_identity | -0.4895 [-0.6494, -0.2158] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | v_head - pan_essentiality | 0.1284 [-0.0543, 0.3422] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | v_head - a_blind | 0.1304 [-0.0456, 0.3687] |
| test | harle_2025 | SK-MES-1 | film_prior_dE | v_head - v_head_context_ablated | -0.4744 [-0.6200, -0.2273] |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head - pair_identity | -0.3544 [-0.5726, -0.1148] |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head - pan_essentiality | 0.2636 [0.0046, 0.4896] |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head - a_blind | 0.2587 [0.0122, 0.4940] |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head - paris_head_context_ablated | -0.0656 [-0.1613, 0.0213] |
| test | harle_2025 | SK-MES-1 | linear_prior | paris_head - paris | 0.1245 [-0.0660, 0.2926] |
| test | harle_2025 | SK-MES-1 | measured | paris_head - pair_identity | -0.2930 [-0.5035, -0.0710] |
| test | harle_2025 | SK-MES-1 | measured | paris_head - pan_essentiality | 0.3250 [0.0367, 0.5400] |
| test | harle_2025 | SK-MES-1 | measured | paris_head - a_blind | 0.3686 [0.0900, 0.5970] |
| test | harle_2025 | SK-MES-1 | measured | paris_head - paris_head_context_ablated | -0.0041 [-0.1213, 0.1170] |
| test | harle_2025 | SK-MES-1 | measured | paris_head - paris | 0.1859 [0.0040, 0.3508] |
| test | ito_2021 | A549_LUNG | film_dE | paris_head - pair_identity | -0.0796 [-0.2165, 0.0517] |
| test | ito_2021 | A549_LUNG | film_dE | paris_head - pan_essentiality | 0.0471 [-0.0165, 0.1049] |
| test | ito_2021 | A549_LUNG | film_dE | paris_head - a_blind | 0.0574 [-0.0060, 0.1204] |
| test | ito_2021 | A549_LUNG | film_dE | paris_head - paris_head_context_ablated | -0.0080 [-0.0367, 0.0216] |
| test | ito_2021 | A549_LUNG | film_dE | paris_head - paris | 0.1504 [0.0452, 0.2776] |
| test | ito_2021 | A549_LUNG | film_dE | v_head - pair_identity | -0.1380 [-0.2692, 0.0110] |
| test | ito_2021 | A549_LUNG | film_dE | v_head - pan_essentiality | -0.0113 [-0.1402, 0.0926] |
| test | ito_2021 | A549_LUNG | film_dE | v_head - a_blind | -0.0009 [-0.1139, 0.1108] |
| test | ito_2021 | A549_LUNG | film_dE | v_head - v_head_context_ablated | -0.0167 [-0.0788, 0.0545] |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head - pair_identity | -0.0785 [-0.2062, 0.0619] |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head - pan_essentiality | 0.0482 [-0.0191, 0.1182] |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head - a_blind | 0.0485 [-0.0251, 0.1219] |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head - paris_head_context_ablated | -0.0069 [-0.0346, 0.0173] |
| test | ito_2021 | A549_LUNG | film_prior_dE | paris_head - paris | 0.1515 [0.0556, 0.2694] |
| test | ito_2021 | A549_LUNG | film_prior_dE | v_head - pair_identity | -0.2142 [-0.3595, -0.0774] |
| test | ito_2021 | A549_LUNG | film_prior_dE | v_head - pan_essentiality | -0.0875 [-0.2117, 0.0175] |
| test | ito_2021 | A549_LUNG | film_prior_dE | v_head - a_blind | -0.0872 [-0.2270, 0.0137] |
| test | ito_2021 | A549_LUNG | film_prior_dE | v_head - v_head_context_ablated | -0.0816 [-0.1810, -0.0025] |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head - pair_identity | -0.0695 [-0.2107, 0.0691] |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head - pan_essentiality | 0.0572 [-0.0126, 0.1318] |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head - a_blind | 0.0536 [-0.0252, 0.1304] |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head - paris_head_context_ablated | 0.0021 [-0.0267, 0.0272] |
| test | ito_2021 | A549_LUNG | linear_prior | paris_head - paris | 0.1604 [0.0568, 0.2857] |
| test | ito_2021 | A549_LUNG | measured | paris_head - pair_identity | -0.0237 [-0.1690, 0.1178] |
| test | ito_2021 | A549_LUNG | measured | paris_head - pan_essentiality | 0.1030 [0.0250, 0.1722] |
| test | ito_2021 | A549_LUNG | measured | paris_head - a_blind | 0.0821 [0.0079, 0.1477] |
| test | ito_2021 | A549_LUNG | measured | paris_head - paris_head_context_ablated | 0.0479 [-0.0145, 0.1028] |
| test | ito_2021 | A549_LUNG | measured | paris_head - paris | 0.2063 [0.0877, 0.3274] |
| test | klingbeil_2024 | A549 | film_dE | paris_head - pair_identity | -0.1662 [-0.3344, 0.0243] |
| test | klingbeil_2024 | A549 | film_dE | paris_head - pan_essentiality | 0.0295 [-0.0647, 0.1268] |
| test | klingbeil_2024 | A549 | film_dE | paris_head - a_blind | 0.0520 [-0.0366, 0.1493] |
| test | klingbeil_2024 | A549 | film_dE | paris_head - paris_head_context_ablated | -0.0341 [-0.0729, 0.0085] |
| test | klingbeil_2024 | A549 | film_dE | paris_head - paris | 0.1972 [0.0598, 0.3578] |
| test | klingbeil_2024 | A549 | film_dE | v_head - pair_identity | -0.2263 [-0.3845, -0.0167] |
| test | klingbeil_2024 | A549 | film_dE | v_head - pan_essentiality | -0.0306 [-0.1839, 0.1458] |
| test | klingbeil_2024 | A549 | film_dE | v_head - a_blind | -0.0080 [-0.1803, 0.1563] |
| test | klingbeil_2024 | A549 | film_dE | v_head - v_head_context_ablated | -0.0846 [-0.2130, 0.0302] |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head - pair_identity | -0.1447 [-0.3049, 0.0416] |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head - pan_essentiality | 0.0510 [-0.0362, 0.1589] |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head - a_blind | 0.0574 [-0.0390, 0.1693] |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head - paris_head_context_ablated | -0.0125 [-0.0385, 0.0224] |
| test | klingbeil_2024 | A549 | film_prior_dE | paris_head - paris | 0.2187 [0.0926, 0.3759] |
| test | klingbeil_2024 | A549 | film_prior_dE | v_head - pair_identity | -0.3242 [-0.4780, -0.1445] |
| test | klingbeil_2024 | A549 | film_prior_dE | v_head - pan_essentiality | -0.1285 [-0.3030, 0.0522] |
| test | klingbeil_2024 | A549 | film_prior_dE | v_head - a_blind | -0.1222 [-0.2991, 0.0441] |
| test | klingbeil_2024 | A549 | film_prior_dE | v_head - v_head_context_ablated | -0.1816 [-0.3252, -0.0272] |
| test | klingbeil_2024 | A549 | linear_prior | paris_head - pair_identity | -0.1529 [-0.3311, 0.0297] |
| test | klingbeil_2024 | A549 | linear_prior | paris_head - pan_essentiality | 0.0428 [-0.0555, 0.1556] |
| test | klingbeil_2024 | A549 | linear_prior | paris_head - a_blind | 0.0440 [-0.0487, 0.1528] |
| test | klingbeil_2024 | A549 | linear_prior | paris_head - paris_head_context_ablated | -0.0207 [-0.0497, 0.0098] |
| test | klingbeil_2024 | A549 | linear_prior | paris_head - paris | 0.2105 [0.0768, 0.3787] |
| test | klingbeil_2024 | A549 | measured | paris_head - pair_identity | -0.1606 [-0.3550, 0.0406] |
| test | klingbeil_2024 | A549 | measured | paris_head - pan_essentiality | 0.0351 [-0.1128, 0.1727] |
| test | klingbeil_2024 | A549 | measured | paris_head - a_blind | 0.0598 [-0.0497, 0.1763] |
| test | klingbeil_2024 | A549 | measured | paris_head - paris_head_context_ablated | -0.0284 [-0.1309, 0.0661] |
| test | klingbeil_2024 | A549 | measured | paris_head - paris | 0.2028 [0.0748, 0.3627] |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head - pair_identity | -0.1409 [-0.2974, 0.0305] |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head - pan_essentiality | 0.0162 [-0.0602, 0.1208] |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head - a_blind | 0.0243 [-0.0543, 0.1158] |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head - paris_head_context_ablated | -0.0188 [-0.0548, 0.0236] |
| test | klingbeil_2024 | MDAMB231 | film_dE | paris_head - paris | 0.2372 [0.1128, 0.3928] |
| test | klingbeil_2024 | MDAMB231 | film_dE | v_head - pair_identity | -0.1170 [-0.2861, 0.0629] |
| test | klingbeil_2024 | MDAMB231 | film_dE | v_head - pan_essentiality | 0.0402 [-0.1031, 0.2204] |
| test | klingbeil_2024 | MDAMB231 | film_dE | v_head - a_blind | 0.0483 [-0.1062, 0.1957] |
| test | klingbeil_2024 | MDAMB231 | film_dE | v_head - v_head_context_ablated | -0.1166 [-0.2297, 0.0304] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head - pair_identity | -0.1125 [-0.2674, 0.0558] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head - pan_essentiality | 0.0446 [-0.0324, 0.1367] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head - a_blind | 0.0255 [-0.0492, 0.1187] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head - paris_head_context_ablated | 0.0096 [-0.0166, 0.0423] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | paris_head - paris | 0.2656 [0.1458, 0.4241] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | v_head - pair_identity | -0.1823 [-0.3566, 0.0067] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | v_head - pan_essentiality | -0.0251 [-0.1808, 0.1500] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | v_head - a_blind | -0.0442 [-0.2021, 0.1208] |
| test | klingbeil_2024 | MDAMB231 | film_prior_dE | v_head - v_head_context_ablated | -0.0834 [-0.2295, 0.0514] |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head - pair_identity | -0.1187 [-0.2742, 0.0364] |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head - pan_essentiality | 0.0385 [-0.0390, 0.1323] |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head - a_blind | 0.0244 [-0.0591, 0.1138] |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head - paris_head_context_ablated | 0.0034 [-0.0247, 0.0344] |
| test | klingbeil_2024 | MDAMB231 | linear_prior | paris_head - paris | 0.2594 [0.1323, 0.4056] |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head - pair_identity | -0.0876 [-0.2492, 0.0840] |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head - pan_essentiality | 0.0696 [-0.0546, 0.1756] |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head - a_blind | 0.0198 [-0.0606, 0.1084] |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head - paris_head_context_ablated | 0.0345 [-0.0663, 0.1133] |
| test | klingbeil_2024 | MDAMB231 | measured | paris_head - paris | 0.2906 [0.1666, 0.4328] |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head - pair_identity | -0.3765 [-0.5895, -0.1174] |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head - pan_essentiality | 0.2766 [0.0339, 0.5015] |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head - a_blind | 0.2776 [0.0511, 0.4988] |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head - paris_head_context_ablated | -0.0336 [-0.1389, 0.0452] |
| val | harle_2025 | NCI-H1299 | film_dE | paris_head - paris | 0.1213 [-0.0801, 0.2874] |
| val | harle_2025 | NCI-H1299 | film_dE | v_head - pair_identity | -0.4431 [-0.6321, -0.1507] |
| val | harle_2025 | NCI-H1299 | film_dE | v_head - pan_essentiality | 0.2101 [0.0155, 0.4530] |
| val | harle_2025 | NCI-H1299 | film_dE | v_head - a_blind | 0.2111 [-0.0038, 0.4487] |
| val | harle_2025 | NCI-H1299 | film_dE | v_head - v_head_context_ablated | -0.4194 [-0.5980, -0.1733] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head - pair_identity | -0.3765 [-0.5908, -0.1133] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head - pan_essentiality | 0.2766 [0.0463, 0.5122] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head - a_blind | 0.2783 [0.0521, 0.5101] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head - paris_head_context_ablated | -0.0336 [-0.1543, 0.0581] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | paris_head - paris | 0.1212 [-0.0619, 0.2808] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | v_head - pair_identity | -0.4943 [-0.7135, -0.2215] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | v_head - pan_essentiality | 0.1588 [-0.0609, 0.4159] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | v_head - a_blind | 0.1606 [-0.0792, 0.4293] |
| val | harle_2025 | NCI-H1299 | film_prior_dE | v_head - v_head_context_ablated | -0.4502 [-0.6324, -0.1964] |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head - pair_identity | -0.3716 [-0.5974, -0.1350] |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head - pan_essentiality | 0.2815 [0.0463, 0.5050] |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head - a_blind | 0.2807 [0.0470, 0.5299] |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head - paris_head_context_ablated | -0.0287 [-0.1325, 0.0597] |
| val | harle_2025 | NCI-H1299 | linear_prior | paris_head - paris | 0.1262 [-0.0755, 0.2956] |
| val | harle_2025 | NCI-H1299 | measured | paris_head - pair_identity | -0.3967 [-0.5995, -0.1386] |
| val | harle_2025 | NCI-H1299 | measured | paris_head - pan_essentiality | 0.2565 [0.0270, 0.5178] |
| val | harle_2025 | NCI-H1299 | measured | paris_head - a_blind | 0.2753 [0.0109, 0.5485] |
| val | harle_2025 | NCI-H1299 | measured | paris_head - paris_head_context_ablated | -0.0538 [-0.1839, 0.0658] |
| val | harle_2025 | NCI-H1299 | measured | paris_head - paris | 0.1011 [-0.0889, 0.3001] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head - pair_identity | 0.0203 [-0.1584, 0.2165] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head - pan_essentiality | 0.0929 [-0.0138, 0.2058] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head - a_blind | 0.0913 [-0.0238, 0.1969] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head - paris_head_context_ablated | 0.0405 [-0.0223, 0.1078] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | paris_head - paris | 0.2583 [0.1022, 0.4522] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | v_head - pair_identity | -0.0970 [-0.3231, 0.1280] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | v_head - pan_essentiality | -0.0244 [-0.2148, 0.2188] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | v_head - a_blind | -0.0260 [-0.2602, 0.1899] |
| val | ito_2021 | PATU8988S_PANCREAS | film_dE | v_head - v_head_context_ablated | 0.0624 [-0.0247, 0.1882] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head - pair_identity | 0.0207 [-0.1675, 0.2278] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head - pan_essentiality | 0.0933 [-0.0192, 0.2160] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head - a_blind | 0.0826 [-0.0327, 0.2069] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head - paris_head_context_ablated | 0.0410 [-0.0164, 0.1098] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | paris_head - paris | 0.2587 [0.1007, 0.4537] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | v_head - pair_identity | -0.1715 [-0.4007, 0.0382] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | v_head - pan_essentiality | -0.0989 [-0.3059, 0.0879] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | v_head - a_blind | -0.1097 [-0.3347, 0.0769] |
| val | ito_2021 | PATU8988S_PANCREAS | film_prior_dE | v_head - v_head_context_ablated | -0.0133 [-0.1675, 0.0932] |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head - pair_identity | 0.0114 [-0.1637, 0.2299] |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head - pan_essentiality | 0.0840 [-0.0395, 0.2122] |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head - a_blind | 0.0742 [-0.0559, 0.2203] |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head - paris_head_context_ablated | 0.0317 [-0.0164, 0.0984] |
| val | ito_2021 | PATU8988S_PANCREAS | linear_prior | paris_head - paris | 0.2494 [0.0971, 0.4432] |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head - pair_identity | 0.1077 [-0.0934, 0.3020] |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head - pan_essentiality | 0.1803 [0.0475, 0.3166] |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head - a_blind | 0.1644 [0.0320, 0.2753] |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head - paris_head_context_ablated | 0.1280 [0.0163, 0.2452] |
| val | ito_2021 | PATU8988S_PANCREAS | measured | paris_head - paris | 0.3457 [0.1294, 0.5505] |
| val | klingbeil_2024 | H1299 | film_dE | paris_head - pair_identity | -0.1947 [-0.3776, -0.0274] |
| val | klingbeil_2024 | H1299 | film_dE | paris_head - pan_essentiality | 0.0380 [-0.0427, 0.1241] |
| val | klingbeil_2024 | H1299 | film_dE | paris_head - a_blind | 0.0413 [-0.0414, 0.1271] |
| val | klingbeil_2024 | H1299 | film_dE | paris_head - paris_head_context_ablated | -0.0053 [-0.0287, 0.0160] |
| val | klingbeil_2024 | H1299 | film_dE | paris_head - paris | 0.1275 [0.0161, 0.3009] |
| val | klingbeil_2024 | H1299 | film_dE | v_head - pair_identity | -0.2100 [-0.3910, -0.0068] |
| val | klingbeil_2024 | H1299 | film_dE | v_head - pan_essentiality | 0.0228 [-0.1228, 0.1688] |
| val | klingbeil_2024 | H1299 | film_dE | v_head - a_blind | 0.0261 [-0.1103, 0.1785] |
| val | klingbeil_2024 | H1299 | film_dE | v_head - v_head_context_ablated | -0.0885 [-0.2557, 0.0552] |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head - pair_identity | -0.1946 [-0.3571, 0.0012] |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head - pan_essentiality | 0.0382 [-0.0406, 0.1418] |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head - a_blind | 0.0293 [-0.0478, 0.1251] |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head - paris_head_context_ablated | -0.0051 [-0.0352, 0.0440] |
| val | klingbeil_2024 | H1299 | film_prior_dE | paris_head - paris | 0.1276 [0.0245, 0.3015] |
| val | klingbeil_2024 | H1299 | film_prior_dE | v_head - pair_identity | -0.2613 [-0.4478, -0.0655] |
| val | klingbeil_2024 | H1299 | film_prior_dE | v_head - pan_essentiality | -0.0285 [-0.1832, 0.1204] |
| val | klingbeil_2024 | H1299 | film_prior_dE | v_head - a_blind | -0.0373 [-0.1986, 0.1207] |
| val | klingbeil_2024 | H1299 | film_prior_dE | v_head - v_head_context_ablated | -0.0960 [-0.2473, 0.0252] |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head - pair_identity | -0.1916 [-0.3664, -0.0164] |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head - pan_essentiality | 0.0412 [-0.0428, 0.1399] |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head - a_blind | 0.0328 [-0.0502, 0.1285] |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head - paris_head_context_ablated | -0.0021 [-0.0260, 0.0308] |
| val | klingbeil_2024 | H1299 | linear_prior | paris_head - paris | 0.1306 [0.0235, 0.2844] |
| val | klingbeil_2024 | H1299 | measured | paris_head - pair_identity | -0.1495 [-0.3203, 0.0576] |
| val | klingbeil_2024 | H1299 | measured | paris_head - pan_essentiality | 0.0832 [-0.0215, 0.1978] |
| val | klingbeil_2024 | H1299 | measured | paris_head - a_blind | 0.0352 [-0.0553, 0.1489] |
| val | klingbeil_2024 | H1299 | measured | paris_head - paris_head_context_ablated | 0.0400 [-0.0264, 0.1314] |
| val | klingbeil_2024 | H1299 | measured | paris_head - paris | 0.1727 [0.0674, 0.3421] |
