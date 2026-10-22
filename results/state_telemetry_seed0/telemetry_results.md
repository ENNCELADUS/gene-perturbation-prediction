# State telemetry (telemetry)

Frozen state vocabularies fitted on the single-cell training lines (at most 500 distinct cells each) and projected on the lines of train_cells, val; the frozen PIE on their state inputs. Telemetry only: nothing here says whether states help predict GeneEffect, and nothing is chosen in code.

## Composition, confound and stability (training lines, first fitting seed)

Deposit η²: the mean over states of the share of each state's mass across lines explained by deposit (null q95), then the largest state's; per state in the identity tables. Seed match: the matched states' mean correlation (the mean unmatched mass per line, when the fits' state counts differ).

| Vocabulary | States | States per line ≥ mass | Deposit η² mean (null q95); max state | QC R² max | Second deposit TVD (between lines) | Seed match (unmatched mass) |
| --- | ---: | ---: | --- | ---: | --- | --- |
| cycle | 3 | 3.0 | 0.364 (0.085); 0.395 | 0.208 | 0.159 (0.231), 99 lines | 1.000 (0.000) |
| programs | 17 | 4.0 | 0.351 (0.094); 0.745 | 0.657 | 0.490 (0.801), 99 lines | 0.742 (0.058) |
| scvi_centred_k4 | 4 | 3.0 | 0.434 (0.075); 0.757 | 0.502 | 0.222 (0.341), 99 lines | 0.648 (0.000) |
| scvi_centred_k6 | 6 | 5.0 | 0.492 (0.079); 0.839 | 0.781 | 0.246 (0.456), 99 lines | 0.510 (0.000) |
| scvi_centred_k8 | 8 | 5.0 | 0.448 (0.080); 0.849 | 0.789 | 0.275 (0.474), 99 lines | 0.547 (0.000) |
| scvi_absolute_k4 | 4 | 1.0 | 0.609 (0.137); 0.847 | 0.752 | 0.187 (0.968), 99 lines | 0.729 (0.000) |
| scvi_absolute_k6 | 6 | 1.0 | 0.659 (0.159); 0.938 | 0.750 | 0.980 (0.991), 99 lines | 0.862 (0.000) |
| scvi_absolute_k8 | 8 | 1.0 | 0.566 (0.133); 0.955 | 0.787 | 0.971 (0.990), 99 lines | 0.730 (0.000) |
| hvg_k4 | 4 | 2.0 | 0.635 (0.084); 0.829 | 0.820 | 0.369 (0.587), 99 lines | 0.841 (0.000) |
| hvg_k6 | 6 | 4.0 | 0.536 (0.094); 0.862 | 0.829 | 0.377 (0.675), 99 lines | 0.710 (0.000) |
| hvg_k8 | 8 | 4.0 | 0.519 (0.107); 0.913 | 0.779 | 0.452 (0.744), 99 lines | 0.800 (0.000) |
| tx1_k4 | 4 | 2.0 | 0.510 (0.100); 0.717 | 0.753 | – | 0.974 (0.000) |
| tx1_k6 | 6 | 2.0 | 0.491 (0.087); 0.769 | 0.667 | – | 0.715 (0.000) |
| tx1_k8 | 8 | 2.0 | 0.414 (0.086); 0.774 | 0.724 | – | 0.978 (0.000) |
| qc_k4 | 4 | 3.0 | 0.280 (0.051); 0.381 | 0.548 | 0.178 (0.169), 99 lines | 0.999 (0.000) |
| qc_k6 | 6 | 4.0 | 0.216 (0.045); 0.297 | 0.594 | 0.223 (0.202), 99 lines | 0.926 (0.000) |
| qc_k8 | 8 | 5.0 | 0.205 (0.043); 0.307 | 0.565 | 0.258 (0.231), 99 lines | 0.807 (0.000) |

## Identity (best reference program per state)

### cycle

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| g2m | 0.324 | 0.321 (0.078) | gavish:MP1 Cell Cycle - G2/M +0.273; kinker:Cell Cycle - G2/M +0.252; tirosh:G2M +0.238 |
| s | 0.448 | 0.395 (0.139) | tirosh:S +0.087; gavish:MP2 Cell Cycle - G1/S +0.077; kinker:Cell Cycle - G1/S +0.070 |
| noncycling | 0.228 | 0.376 (0.061) | kinker:p53-Dependent Senescence +0.034; kinker:Epithelial Senescence +0.021; kinker:EMT III +0.016 |

### programs

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| program0 | 0.100 | 0.149 (0.109) | gavish:MP11 Translation initiation +0.014; kinker:p53-Dependent Senescence +0.009; gavish:MP35 Hemato-related-I +0.007 |
| program1 | 0.082 | 0.251 (0.062) | gavish:MP8 Proteasomal degradation +0.089; kinker:Proteasomal Degradation +0.088; gavish:MP20 MYC +0.069 |
| program2 | 0.152 | 0.629 (0.101) | gavish:MP21 Respiration +0.028; kinker:Cell Cycle - G1/S +0.025; gavish:MP3 Cell Cylce HMG-rich +0.023 |
| program3 | 0.035 | 0.275 (0.071) | gavish:MP4 Chromatin +0.017; hallmarks:HALLMARK_WNT_BETA_CATENIN_SIGNALING +0.016; kinker:Cell Cycle - G1/S +0.016 |
| program4 | 0.013 | 0.092 (0.062) | kinker:IFN Response +0.255; gavish:MP17 Interferon/MHC-II (I) +0.169; hallmarks:HALLMARK_INTERFERON_ALPHA_RESPONSE +0.119 |
| program5 | 0.066 | 0.170 (0.064) | gavish:MP4 Chromatin +0.050; gavish:MP41 Unassigned +0.034; gavish:MP20 MYC +0.033 |
| program6 | 0.047 | 0.482 (0.066) | gavish:MP21 Respiration +0.059; gavish:MP3 Cell Cylce HMG-rich +0.049; gavish:MP8 Proteasomal degradation +0.045 |
| program7 | 0.061 | 0.387 (0.118) | gavish:MP21 Respiration +0.068; gavish:MP3 Cell Cylce HMG-rich +0.044; gavish:MP20 MYC +0.029 |
| program8 | 0.017 | 0.346 (0.465) | gavish:MP1 Cell Cycle - G2/M +0.055; kinker:Cell Cycle - G2/M +0.045; tirosh:G2M +0.040 |
| program9 | 0.047 | 0.310 (0.076) | gavish:MP1 Cell Cycle - G2/M +0.152; kinker:Cell Cycle - G2/M +0.115; tirosh:G2M +0.098 |
| program10 | 0.033 | 0.493 (0.087) | kinker:Stress Response +0.108; gavish:MP7 Stress (in vitro) +0.090; gavish:MP5 Stress +0.062 |
| program11 | 0.045 | 0.240 (0.073) | gavish:MP17 Interferon/MHC-II (I) +0.076; kinker:p53-Dependent Senescence +0.056; kinker:IFN Response +0.048 |
| program12 | 0.035 | 0.544 (0.382) | gavish:MP1 Cell Cycle - G2/M +0.102; tirosh:G2M +0.099; kinker:Cell Cycle - G2/M +0.091 |
| program13 | 0.018 | 0.041 (0.081) | gavish:MP9 Unfolded protein response +0.089; gavish:MP7 Stress (in vitro) +0.077; kinker:Stress Response +0.054 |
| program14 | 0.066 | 0.595 (0.090) | tirosh:G2M +0.034; kinker:Cell Cycle - G2/M +0.034; gavish:MP1 Cell Cycle - G2/M +0.034 |
| program15 | 0.041 | 0.212 (0.125) | kinker:Epithelial Senescence +0.029; gavish:MP19 Epithelial Senescence +0.024; gavish:MP30 PDAC-classical +0.021 |
| background | 0.143 | 0.745 (0.061) | gavish:MP16 MES (glioma) +0.012; gavish:MP29 NPC/OPC +0.012; gavish:MP41 Unassigned +0.011 |

### scvi_centred_k4

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.221 | 0.625 (0.099) | gavish:MP41 Unassigned +0.011; gavish:MP9 Unfolded protein response +0.010; gavish:MP10 Protein maturation +0.010 |
| 1 | 0.371 | 0.757 (0.126) | gavish:MP3 Cell Cylce HMG-rich +0.013; gavish:MP21 Respiration +0.012; gavish:MP1 Cell Cycle - G2/M +0.010 |
| 2 | 0.054 | 0.103 (0.058) | kinker:Stress Response +0.022; gavish:MP30 PDAC-classical +0.020; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.020 |
| 3 | 0.354 | 0.249 (0.062) | gavish:MP3 Cell Cylce HMG-rich +0.026; gavish:MP20 MYC +0.020; gavish:MP2 Cell Cycle - G1/S +0.016 |

### scvi_centred_k6

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.040 | 0.090 (0.063) | gavish:MP30 PDAC-classical +0.027; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.026; kinker:Stress Response +0.024 |
| 1 | 0.279 | 0.317 (0.083) | gavish:MP3 Cell Cylce HMG-rich +0.036; gavish:MP20 MYC +0.032; gavish:MP21 Respiration +0.024 |
| 2 | 0.245 | 0.816 (0.125) | kinker:Cell Cycle - G2/M +0.011; kinker:Cell Cycle - G1/S +0.011; tirosh:G2M +0.010 |
| 3 | 0.158 | 0.340 (0.122) | gavish:MP9 Unfolded protein response +0.011; gavish:MP7 Stress (in vitro) +0.010; gavish:MP10 Protein maturation +0.010 |
| 4 | 0.085 | 0.550 (0.106) | gavish:MP8 Proteasomal degradation +0.082; kinker:Proteasomal Degradation +0.079; kinker:Protein Maturation +0.059 |
| 5 | 0.193 | 0.839 (0.064) | gavish:MP3 Cell Cylce HMG-rich +0.008; gavish:MP21 Respiration +0.006; gavish:MP2 Cell Cycle - G1/S +0.006 |

### scvi_centred_k8

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.035 | 0.117 (0.073) | gavish:MP30 PDAC-classical +0.031; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.030; gavish:MP19 Epithelial Senescence +0.022 |
| 1 | 0.073 | 0.326 (0.070) | gavish:MP41 Unassigned +0.022; gavish:MP4 Chromatin +0.020; kinker:Stress Response +0.015 |
| 2 | 0.259 | 0.843 (0.133) | gavish:MP3 Cell Cylce HMG-rich +0.019; gavish:MP1 Cell Cycle - G2/M +0.015; gavish:MP21 Respiration +0.015 |
| 3 | 0.086 | 0.501 (0.088) | gavish:MP8 Proteasomal degradation +0.055; kinker:Proteasomal Degradation +0.051; kinker:Protein Maturation +0.046 |
| 4 | 0.078 | 0.176 (0.048) | gavish:MP11 Translation initiation +0.022; gavish:MP7 Stress (in vitro) +0.020; kinker:Stress Response +0.014 |
| 5 | 0.155 | 0.849 (0.064) | gavish:MP3 Cell Cylce HMG-rich +0.007; gavish:MP2 Cell Cycle - G1/S +0.006; kinker:Cell Cycle - G1/S +0.006 |
| 6 | 0.233 | 0.281 (0.078) | gavish:MP3 Cell Cylce HMG-rich +0.033; gavish:MP20 MYC +0.029; gavish:MP21 Respiration +0.021 |
| 7 | 0.082 | 0.489 (0.219) | gavish:MP21 Respiration +0.026; gavish:MP3 Cell Cylce HMG-rich +0.024; gavish:MP1 Cell Cycle - G2/M +0.017 |

### scvi_absolute_k4

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.207 | 0.797 (0.068) | hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.004; gavish:MP30 PDAC-classical +0.004; gavish:MP38 Glutathione +0.003 |
| 1 | 0.165 | 0.508 (0.127) | gavish:MP9 Unfolded protein response +0.014; gavish:MP7 Stress (in vitro) +0.013; gavish:MP21 Respiration +0.012 |
| 2 | 0.554 | 0.847 (0.161) | gavish:MP3 Cell Cylce HMG-rich +0.019; gavish:MP21 Respiration +0.019; gavish:MP14 EMT-III +0.018 |
| 3 | 0.075 | 0.285 (0.274) | gavish:MP20 MYC +0.035; gavish:MP3 Cell Cylce HMG-rich +0.031; hallmarks:HALLMARK_MYC_TARGETS_V1 +0.026 |

### scvi_absolute_k6

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.208 | 0.794 (0.070) | hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.004; gavish:MP30 PDAC-classical +0.004; gavish:MP38 Glutathione +0.003 |
| 1 | 0.113 | 0.433 (0.063) | gavish:MP7 Stress (in vitro) +0.016; kinker:Stress Response +0.015; gavish:MP9 Unfolded protein response +0.014 |
| 2 | 0.046 | 0.363 (0.316) | gavish:MP20 MYC +0.040; gavish:MP3 Cell Cylce HMG-rich +0.038; gavish:MP8 Proteasomal degradation +0.035 |
| 3 | 0.243 | 0.858 (0.079) | gavish:MP3 Cell Cylce HMG-rich +0.033; gavish:MP21 Respiration +0.030; gavish:MP1 Cell Cycle - G2/M +0.022 |
| 4 | 0.087 | 0.570 (0.512) | gavish:MP3 Cell Cylce HMG-rich +0.006; gavish:MP21 Respiration +0.006; gavish:MP20 MYC +0.005 |
| 5 | 0.302 | 0.938 (0.074) | gavish:MP4 Chromatin +0.003; kinker:EMT II +0.003; gavish:MP13 EMT-II +0.002 |

### scvi_absolute_k8

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.280 | 0.955 (0.071) | gavish:MP21 Respiration +0.005; gavish:MP3 Cell Cylce HMG-rich +0.005; kinker:Cell Cycle - G1/S +0.005 |
| 1 | 0.050 | 0.342 (0.284) | gavish:MP20 MYC +0.041; gavish:MP3 Cell Cylce HMG-rich +0.040; hallmarks:HALLMARK_MYC_TARGETS_V1 +0.036 |
| 2 | 0.181 | 0.789 (0.059) | hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.005; gavish:MP30 PDAC-classical +0.004; gavish:MP38 Glutathione +0.003 |
| 3 | 0.053 | 0.395 (0.459) | gavish:MP3 Cell Cylce HMG-rich +0.006; gavish:MP20 MYC +0.005; hallmarks:HALLMARK_MYC_TARGETS_V1 +0.004 |
| 4 | 0.248 | 0.878 (0.081) | gavish:MP1 Cell Cycle - G2/M +0.022; gavish:MP14 EMT-III +0.020; kinker:Cell Cycle - G2/M +0.018 |
| 5 | 0.067 | 0.172 (0.117) | gavish:MP9 Unfolded protein response +0.029; gavish:MP10 Protein maturation +0.029; kinker:Protein Maturation +0.025 |
| 6 | 0.031 | 0.563 (0.223) | gavish:MP21 Respiration +0.007; gavish:MP3 Cell Cylce HMG-rich +0.007; gavish:MP8 Proteasomal degradation +0.007 |
| 7 | 0.090 | 0.437 (0.080) | gavish:MP21 Respiration +0.036; gavish:MP3 Cell Cylce HMG-rich +0.026; gavish:MP20 MYC +0.026 |

### hvg_k4

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.147 | 0.666 (0.176) | gavish:MP1 Cell Cycle - G2/M +0.081; tirosh:G2M +0.072; kinker:Cell Cycle - G2/M +0.072 |
| 1 | 0.205 | 0.829 (0.061) | gavish:MP30 PDAC-classical +0.007; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.007; gavish:MP38 Glutathione +0.005 |
| 2 | 0.349 | 0.519 (0.056) | gavish:MP3 Cell Cylce HMG-rich +0.067; gavish:MP20 MYC +0.056; gavish:MP2 Cell Cycle - G1/S +0.052 |
| 3 | 0.300 | 0.526 (0.084) | kinker:p53-Dependent Senescence +0.017; gavish:MP14 EMT-III +0.017; kinker:Epithelial Senescence +0.012 |

### hvg_k6

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.133 | 0.507 (0.267) | tirosh:S +0.069; gavish:MP2 Cell Cycle - G1/S +0.061; kinker:Cell Cycle - G1/S +0.053 |
| 1 | 0.154 | 0.352 (0.049) | gavish:MP1 Cell Cycle - G2/M +0.272; kinker:Cell Cycle - G2/M +0.242; tirosh:G2M +0.218 |
| 2 | 0.082 | 0.161 (0.043) | kinker:Stress Response +0.037; gavish:MP7 Stress (in vitro) +0.037; gavish:MP14 EMT-III +0.027 |
| 3 | 0.166 | 0.862 (0.065) | gavish:MP30 PDAC-classical +0.008; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.008; gavish:MP38 Glutathione +0.007 |
| 4 | 0.137 | 0.560 (0.107) | gavish:MP1 Cell Cycle - G2/M +0.039; tirosh:G2M +0.037; kinker:Cell Cycle - G2/M +0.036 |
| 5 | 0.327 | 0.777 (0.126) | gavish:MP21 Respiration +0.017; tirosh:S +0.006; gavish:MP3 Cell Cylce HMG-rich +0.006 |

### hvg_k8

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.085 | 0.577 (0.098) | gavish:MP1 Cell Cycle - G2/M +0.093; tirosh:G2M +0.084; kinker:Cell Cycle - G2/M +0.082 |
| 1 | 0.058 | 0.543 (0.257) | gavish:MP1 Cell Cycle - G2/M +0.309; kinker:Cell Cycle - G2/M +0.273; tirosh:G2M +0.253 |
| 2 | 0.216 | 0.533 (0.110) | gavish:MP3 Cell Cylce HMG-rich +0.059; gavish:MP20 MYC +0.046; gavish:MP21 Respiration +0.045 |
| 3 | 0.111 | 0.234 (0.065) | kinker:p53-Dependent Senescence +0.059; kinker:Epithelial Senescence +0.037; gavish:MP10 Protein maturation +0.031 |
| 4 | 0.056 | 0.180 (0.039) | hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.021; gavish:MP30 PDAC-classical +0.020; kinker:Stress Response +0.015 |
| 5 | 0.143 | 0.913 (0.065) | gavish:MP30 PDAC-classical +0.004; gavish:MP38 Glutathione +0.004; gavish:MP19 Epithelial Senescence +0.003 |
| 6 | 0.114 | 0.520 (0.261) | tirosh:S +0.080; gavish:MP2 Cell Cycle - G1/S +0.073; kinker:Cell Cycle - G1/S +0.063 |
| 7 | 0.216 | 0.655 (0.102) | gavish:MP1 Cell Cycle - G2/M +0.055; gavish:MP3 Cell Cylce HMG-rich +0.044; kinker:Cell Cycle - G2/M +0.038 |

### tx1_k4

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.117 | 0.717 (0.060) | gavish:MP30 PDAC-classical +0.011; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.010; gavish:MP19 Epithelial Senescence +0.010 |
| 1 | 0.320 | 0.417 (0.184) | gavish:MP3 Cell Cylce HMG-rich +0.028; gavish:MP20 MYC +0.023; gavish:MP21 Respiration +0.018 |
| 2 | 0.112 | 0.279 (0.061) | gavish:MP4 Chromatin +0.030; gavish:MP41 Unassigned +0.030; kinker:Protein Maturation +0.019 |
| 3 | 0.451 | 0.625 (0.150) | gavish:MP7 Stress (in vitro) +0.016; kinker:Stress Response +0.014; gavish:MP14 EMT-III +0.011 |

### tx1_k6

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.119 | 0.184 (0.062) | gavish:MP9 Unfolded protein response +0.047; kinker:Protein Maturation +0.046; gavish:MP7 Stress (in vitro) +0.045 |
| 1 | 0.438 | 0.769 (0.153) | gavish:MP21 Respiration +0.020; gavish:MP3 Cell Cylce HMG-rich +0.017; gavish:MP1 Cell Cycle - G2/M +0.009 |
| 2 | 0.049 | 0.318 (0.061) | gavish:MP29 NPC/OPC +0.018; gavish:MP4 Chromatin +0.013; gavish:MP41 Unassigned +0.012 |
| 3 | 0.082 | 0.539 (0.060) | gavish:MP30 PDAC-classical +0.014; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.013; gavish:MP38 Glutathione +0.012 |
| 4 | 0.060 | 0.605 (0.050) | gavish:MP41 Unassigned +0.031; gavish:MP4 Chromatin +0.022; gavish:MP29 NPC/OPC +0.013 |
| 5 | 0.251 | 0.528 (0.229) | gavish:MP3 Cell Cylce HMG-rich +0.029; gavish:MP20 MYC +0.023; gavish:MP8 Proteasomal degradation +0.018 |

### tx1_k8

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.012 | 0.096 (0.069) | gavish:MP30 PDAC-classical +0.037; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.034; gavish:MP38 Glutathione +0.026 |
| 1 | 0.127 | 0.322 (0.107) | gavish:MP7 Stress (in vitro) +0.062; kinker:Stress Response +0.057; gavish:MP10 Protein maturation +0.042 |
| 2 | 0.040 | 0.073 (0.044) | gavish:MP14 EMT-III +0.010; kinker:EMT III +0.008; gavish:MP21 Respiration +0.008 |
| 3 | 0.393 | 0.774 (0.146) | gavish:MP3 Cell Cylce HMG-rich +0.022; gavish:MP21 Respiration +0.022; gavish:MP20 MYC +0.012 |
| 4 | 0.042 | 0.278 (0.130) | gavish:MP3 Cell Cylce HMG-rich +0.058; gavish:MP20 MYC +0.056; gavish:MP8 Proteasomal degradation +0.055 |
| 5 | 0.222 | 0.528 (0.237) | gavish:MP3 Cell Cylce HMG-rich +0.029; gavish:MP20 MYC +0.023; hallmarks:HALLMARK_MYC_TARGETS_V1 +0.018 |
| 6 | 0.059 | 0.513 (0.056) | gavish:MP41 Unassigned +0.065; gavish:MP4 Chromatin +0.064; kinker:Protein Maturation +0.034 |
| 7 | 0.105 | 0.728 (0.064) | gavish:MP30 PDAC-classical +0.009; gavish:MP38 Glutathione +0.008; gavish:MP19 Epithelial Senescence +0.008 |

### qc_k4

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.182 | 0.381 (0.074) | gavish:MP1 Cell Cycle - G2/M +0.094; kinker:Cell Cycle - G2/M +0.084; tirosh:G2M +0.082 |
| 1 | 0.127 | 0.305 (0.058) | gavish:MP30 PDAC-classical +0.018; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.015; gavish:MP38 Glutathione +0.013 |
| 2 | 0.631 | 0.269 (0.057) | gavish:MP3 Cell Cylce HMG-rich +0.014; gavish:MP21 Respiration +0.012; gavish:MP20 MYC +0.009 |
| 3 | 0.061 | 0.164 (0.042) | gavish:MP41 Unassigned +0.042; gavish:MP4 Chromatin +0.037; gavish:MP10 Protein maturation +0.032 |

### qc_k6

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.183 | 0.179 (0.051) | gavish:MP30 PDAC-classical +0.016; gavish:MP19 Epithelial Senescence +0.013; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.013 |
| 1 | 0.414 | 0.198 (0.050) | gavish:MP3 Cell Cylce HMG-rich +0.028; gavish:MP20 MYC +0.021; gavish:MP21 Respiration +0.020 |
| 2 | 0.040 | 0.225 (0.062) | gavish:MP30 PDAC-classical +0.030; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.025; gavish:MP38 Glutathione +0.022 |
| 3 | 0.033 | 0.144 (0.041) | gavish:MP41 Unassigned +0.077; gavish:MP4 Chromatin +0.072; kinker:Protein Maturation +0.058 |
| 4 | 0.174 | 0.253 (0.064) | gavish:MP1 Cell Cycle - G2/M +0.107; kinker:Cell Cycle - G2/M +0.096; tirosh:G2M +0.095 |
| 5 | 0.157 | 0.297 (0.059) | gavish:MP41 Unassigned +0.006; gavish:MP10 Protein maturation +0.005; kinker:Protein Maturation +0.004 |

### qc_k8

| State | Mass (mean) | Deposit η² (null q95) | Top programs |
| --- | ---: | --- | --- |
| 0 | 0.125 | 0.198 (0.047) | gavish:MP30 PDAC-classical +0.021; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.016; gavish:MP19 Epithelial Senescence +0.016 |
| 1 | 0.204 | 0.215 (0.067) | gavish:MP1 Cell Cycle - G2/M +0.098; tirosh:G2M +0.086; kinker:Cell Cycle - G2/M +0.086 |
| 2 | 0.356 | 0.186 (0.050) | gavish:MP3 Cell Cylce HMG-rich +0.018; gavish:MP20 MYC +0.012; gavish:MP21 Respiration +0.011 |
| 3 | 0.029 | 0.149 (0.039) | gavish:MP41 Unassigned +0.060; gavish:MP4 Chromatin +0.051; gavish:MP10 Protein maturation +0.044 |
| 4 | 0.093 | 0.190 (0.046) | kinker:Proteasomal Degradation +0.029; gavish:MP8 Proteasomal degradation +0.029; gavish:MP20 MYC +0.022 |
| 5 | 0.103 | 0.307 (0.057) | kinker:Epithelial Senescence +0.007; gavish:MP31 Alveolar +0.005; hallmarks:HALLMARK_HEDGEHOG_SIGNALING +0.004 |
| 6 | 0.025 | 0.213 (0.067) | gavish:MP30 PDAC-classical +0.047; hallmarks:HALLMARK_KRAS_SIGNALING_DN +0.040; gavish:MP38 Glutathione +0.035 |
| 7 | 0.065 | 0.184 (0.066) | gavish:MP1 Cell Cycle - G2/M +0.111; tirosh:G2M +0.099; kinker:Cell Cycle - G2/M +0.099 |

## The frozen PIE's gap

Pooled over the genes and the role's lines: the gap's energy, and its spread across lines within a gene, over the spread across lines of the whole-line delta within the gene; the noise floor is the set's `|whole` context against the main pass on the same scale.

| Set | Delta | Gap / spread (train) | Gap spread (train) | Noise (train) | Gap / spread (val) | Noise (val) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cycle | dH | 0.0003 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| cycle | dE | 0.0094 | 0.0064 | 0.0000 | 0.0073 | 0.0000 |
| cycle | own_h | 0.0124 | 0.0114 | 0.0000 | 0.0067 | 0.0000 |
| cycle | own_dp | 0.0103 | 0.0092 | 0.0000 | 0.0077 | 0.0000 |
| programs | dH | 0.0002 | 0.0001 | 0.0000 | 0.0002 | 0.0000 |
| programs | dE | 0.0073 | 0.0043 | 0.0000 | 0.0127 | 0.0000 |
| programs | own_h | 0.0378 | 0.0335 | 0.0000 | 0.0462 | 0.0000 |
| programs | own_dp | 0.0065 | 0.0056 | 0.0000 | 0.0091 | 0.0000 |
| scvi_centred_k4 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| scvi_centred_k4 | dE | 0.0055 | 0.0035 | 0.0000 | 0.0074 | 0.0000 |
| scvi_centred_k4 | own_h | 0.0075 | 0.0069 | 0.0000 | 0.0073 | 0.0000 |
| scvi_centred_k4 | own_dp | 0.0065 | 0.0058 | 0.0000 | 0.0078 | 0.0000 |
| scvi_centred_k6 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| scvi_centred_k6 | dE | 0.0085 | 0.0051 | 0.0000 | 0.0140 | 0.0000 |
| scvi_centred_k6 | own_h | 0.0096 | 0.0084 | 0.0000 | 0.0102 | 0.0000 |
| scvi_centred_k6 | own_dp | 0.0078 | 0.0065 | 0.0000 | 0.0119 | 0.0000 |
| scvi_centred_k8 | dH | 0.0004 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| scvi_centred_k8 | dE | 0.0100 | 0.0059 | 0.0000 | 0.0168 | 0.0000 |
| scvi_centred_k8 | own_h | 0.0112 | 0.0097 | 0.0000 | 0.0119 | 0.0000 |
| scvi_centred_k8 | own_dp | 0.0091 | 0.0076 | 0.0000 | 0.0136 | 0.0000 |
| scvi_absolute_k4 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4 | dE | 0.0006 | 0.0005 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4 | own_h | 0.0010 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4 | own_dp | 0.0009 | 0.0009 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6 | dE | 0.0006 | 0.0006 | 0.0000 | 0.0002 | 0.0000 |
| scvi_absolute_k6 | own_h | 0.0009 | 0.0009 | 0.0000 | 0.0002 | 0.0000 |
| scvi_absolute_k6 | own_dp | 0.0010 | 0.0010 | 0.0000 | 0.0002 | 0.0000 |
| scvi_absolute_k8 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k8 | dE | 0.0008 | 0.0007 | 0.0000 | 0.0015 | 0.0000 |
| scvi_absolute_k8 | own_h | 0.0011 | 0.0011 | 0.0000 | 0.0019 | 0.0000 |
| scvi_absolute_k8 | own_dp | 0.0012 | 0.0011 | 0.0000 | 0.0024 | 0.0000 |
| hvg_k4 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0002 | 0.0000 |
| hvg_k4 | dE | 0.0071 | 0.0050 | 0.0000 | 0.0089 | 0.0000 |
| hvg_k4 | own_h | 0.0096 | 0.0087 | 0.0000 | 0.0116 | 0.0000 |
| hvg_k4 | own_dp | 0.0065 | 0.0058 | 0.0000 | 0.0104 | 0.0000 |
| hvg_k6 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0004 | 0.0000 |
| hvg_k6 | dE | 0.0109 | 0.0071 | 0.0000 | 0.0170 | 0.0000 |
| hvg_k6 | own_h | 0.0114 | 0.0100 | 0.0000 | 0.0131 | 0.0000 |
| hvg_k6 | own_dp | 0.0084 | 0.0071 | 0.0000 | 0.0131 | 0.0000 |
| hvg_k8 | dH | 0.0004 | 0.0002 | 0.0000 | 0.0005 | 0.0000 |
| hvg_k8 | dE | 0.0135 | 0.0087 | 0.0000 | 0.0230 | 0.0000 |
| hvg_k8 | own_h | 0.0135 | 0.0117 | 0.0000 | 0.0179 | 0.0000 |
| hvg_k8 | own_dp | 0.0101 | 0.0084 | 0.0000 | 0.0183 | 0.0000 |
| tx1_k4 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| tx1_k4 | dE | 0.0103 | 0.0079 | 0.0000 | 0.0060 | 0.0000 |
| tx1_k4 | own_h | 0.0306 | 0.0294 | 0.0000 | 0.0069 | 0.0000 |
| tx1_k4 | own_dp | 0.0119 | 0.0111 | 0.0000 | 0.0071 | 0.0000 |
| tx1_k6 | dH | 0.0006 | 0.0005 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k6 | dE | 0.0188 | 0.0140 | 0.0000 | 0.0107 | 0.0000 |
| tx1_k6 | own_h | 0.0380 | 0.0358 | 0.0000 | 0.0075 | 0.0000 |
| tx1_k6 | own_dp | 0.0180 | 0.0162 | 0.0000 | 0.0098 | 0.0000 |
| tx1_k8 | dH | 0.0004 | 0.0003 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k8 | dE | 0.0147 | 0.0100 | 0.0000 | 0.0125 | 0.0000 |
| tx1_k8 | own_h | 0.0369 | 0.0346 | 0.0000 | 0.0106 | 0.0000 |
| tx1_k8 | own_dp | 0.0155 | 0.0138 | 0.0000 | 0.0117 | 0.0000 |
| qc_k4 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| qc_k4 | dE | 0.0038 | 0.0026 | 0.0000 | 0.0034 | 0.0000 |
| qc_k4 | own_h | 0.0053 | 0.0048 | 0.0000 | 0.0032 | 0.0000 |
| qc_k4 | own_dp | 0.0054 | 0.0049 | 0.0000 | 0.0037 | 0.0000 |
| qc_k6 | dH | 0.0006 | 0.0004 | 0.0000 | 0.0001 | 0.0000 |
| qc_k6 | dE | 0.0104 | 0.0066 | 0.0000 | 0.0086 | 0.0000 |
| qc_k6 | own_h | 0.0108 | 0.0095 | 0.0000 | 0.0065 | 0.0000 |
| qc_k6 | own_dp | 0.0114 | 0.0098 | 0.0000 | 0.0081 | 0.0000 |
| qc_k8 | dH | 0.0011 | 0.0008 | 0.0000 | 0.0003 | 0.0000 |
| qc_k8 | dE | 0.0192 | 0.0119 | 0.0000 | 0.0176 | 0.0000 |
| qc_k8 | own_h | 0.0164 | 0.0138 | 0.0000 | 0.0109 | 0.0000 |
| qc_k8 | own_dp | 0.0172 | 0.0143 | 0.0000 | 0.0143 | 0.0000 |
| cycling_weighted | dH | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| cycling_weighted | dE | 0.0037 | 0.0029 | 0.0000 | 0.0036 | 0.0000 |
| cycling_weighted | own_h | 0.0098 | 0.0096 | 0.0000 | 0.0057 | 0.0000 |
| cycling_weighted | own_dp | 0.0203 | 0.0190 | 0.0000 | 0.0137 | 0.0000 |
| cycle__perm0 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| cycle__perm0 | dE | 0.0077 | 0.0054 | 0.0000 | 0.0062 | 0.0000 |
| cycle__perm0 | own_h | 0.0104 | 0.0096 | 0.0000 | 0.0064 | 0.0000 |
| cycle__perm0 | own_dp | 0.0085 | 0.0077 | 0.0000 | 0.0064 | 0.0000 |
| programs__perm0 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| programs__perm0 | dE | 0.0063 | 0.0039 | 0.0000 | 0.0107 | 0.0000 |
| programs__perm0 | own_h | 0.0335 | 0.0297 | 0.0000 | 0.0442 | 0.0000 |
| programs__perm0 | own_dp | 0.0051 | 0.0044 | 0.0000 | 0.0087 | 0.0000 |
| scvi_centred_k4__perm0 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| scvi_centred_k4__perm0 | dE | 0.0046 | 0.0030 | 0.0000 | 0.0061 | 0.0000 |
| scvi_centred_k4__perm0 | own_h | 0.0058 | 0.0053 | 0.0000 | 0.0065 | 0.0000 |
| scvi_centred_k4__perm0 | own_dp | 0.0057 | 0.0051 | 0.0000 | 0.0061 | 0.0000 |
| scvi_centred_k6__perm0 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| scvi_centred_k6__perm0 | dE | 0.0072 | 0.0044 | 0.0000 | 0.0122 | 0.0000 |
| scvi_centred_k6__perm0 | own_h | 0.0072 | 0.0063 | 0.0000 | 0.0088 | 0.0000 |
| scvi_centred_k6__perm0 | own_dp | 0.0065 | 0.0055 | 0.0000 | 0.0110 | 0.0000 |
| scvi_centred_k8__perm0 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| scvi_centred_k8__perm0 | dE | 0.0085 | 0.0052 | 0.0000 | 0.0146 | 0.0000 |
| scvi_centred_k8__perm0 | own_h | 0.0085 | 0.0073 | 0.0000 | 0.0104 | 0.0000 |
| scvi_centred_k8__perm0 | own_dp | 0.0076 | 0.0064 | 0.0000 | 0.0126 | 0.0000 |
| scvi_absolute_k4__perm0 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm0 | dE | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm0 | own_h | 0.0004 | 0.0004 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm0 | own_dp | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6__perm0 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6__perm0 | dE | 0.0003 | 0.0003 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k6__perm0 | own_h | 0.0005 | 0.0005 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k6__perm0 | own_dp | 0.0004 | 0.0004 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k8__perm0 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k8__perm0 | dE | 0.0004 | 0.0003 | 0.0000 | 0.0011 | 0.0000 |
| scvi_absolute_k8__perm0 | own_h | 0.0005 | 0.0005 | 0.0000 | 0.0013 | 0.0000 |
| scvi_absolute_k8__perm0 | own_dp | 0.0004 | 0.0004 | 0.0000 | 0.0021 | 0.0000 |
| hvg_k4__perm0 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0002 | 0.0000 |
| hvg_k4__perm0 | dE | 0.0061 | 0.0045 | 0.0000 | 0.0073 | 0.0000 |
| hvg_k4__perm0 | own_h | 0.0072 | 0.0065 | 0.0000 | 0.0149 | 0.0000 |
| hvg_k4__perm0 | own_dp | 0.0051 | 0.0046 | 0.0000 | 0.0077 | 0.0000 |
| hvg_k6__perm0 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| hvg_k6__perm0 | dE | 0.0100 | 0.0068 | 0.0000 | 0.0156 | 0.0000 |
| hvg_k6__perm0 | own_h | 0.0094 | 0.0082 | 0.0000 | 0.0105 | 0.0000 |
| hvg_k6__perm0 | own_dp | 0.0074 | 0.0063 | 0.0000 | 0.0124 | 0.0000 |
| hvg_k8__perm0 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0004 | 0.0000 |
| hvg_k8__perm0 | dE | 0.0119 | 0.0080 | 0.0000 | 0.0195 | 0.0000 |
| hvg_k8__perm0 | own_h | 0.0113 | 0.0098 | 0.0000 | 0.0177 | 0.0000 |
| hvg_k8__perm0 | own_dp | 0.0082 | 0.0070 | 0.0000 | 0.0157 | 0.0000 |
| tx1_k4__perm0 | dH | 0.0002 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| tx1_k4__perm0 | dE | 0.0076 | 0.0060 | 0.0000 | 0.0055 | 0.0000 |
| tx1_k4__perm0 | own_h | 0.0157 | 0.0150 | 0.0000 | 0.0054 | 0.0000 |
| tx1_k4__perm0 | own_dp | 0.0081 | 0.0075 | 0.0000 | 0.0064 | 0.0000 |
| tx1_k6__perm0 | dH | 0.0005 | 0.0004 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k6__perm0 | dE | 0.0138 | 0.0105 | 0.0000 | 0.0093 | 0.0000 |
| tx1_k6__perm0 | own_h | 0.0217 | 0.0204 | 0.0000 | 0.0099 | 0.0000 |
| tx1_k6__perm0 | own_dp | 0.0128 | 0.0117 | 0.0000 | 0.0080 | 0.0000 |
| tx1_k8__perm0 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k8__perm0 | dE | 0.0111 | 0.0078 | 0.0000 | 0.0109 | 0.0000 |
| tx1_k8__perm0 | own_h | 0.0191 | 0.0179 | 0.0000 | 0.0082 | 0.0000 |
| tx1_k8__perm0 | own_dp | 0.0099 | 0.0089 | 0.0000 | 0.0104 | 0.0000 |
| cycle__perm1 | dH | 0.0002 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| cycle__perm1 | dE | 0.0076 | 0.0053 | 0.0000 | 0.0063 | 0.0000 |
| cycle__perm1 | own_h | 0.0104 | 0.0096 | 0.0000 | 0.0058 | 0.0000 |
| cycle__perm1 | own_dp | 0.0085 | 0.0077 | 0.0000 | 0.0063 | 0.0000 |
| programs__perm1 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| programs__perm1 | dE | 0.0063 | 0.0039 | 0.0000 | 0.0106 | 0.0000 |
| programs__perm1 | own_h | 0.0325 | 0.0287 | 0.0000 | 0.0453 | 0.0000 |
| programs__perm1 | own_dp | 0.0050 | 0.0044 | 0.0000 | 0.0085 | 0.0000 |
| scvi_centred_k4__perm1 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| scvi_centred_k4__perm1 | dE | 0.0046 | 0.0030 | 0.0000 | 0.0060 | 0.0000 |
| scvi_centred_k4__perm1 | own_h | 0.0055 | 0.0050 | 0.0000 | 0.0044 | 0.0000 |
| scvi_centred_k4__perm1 | own_dp | 0.0054 | 0.0048 | 0.0000 | 0.0056 | 0.0000 |
| scvi_centred_k6__perm1 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| scvi_centred_k6__perm1 | dE | 0.0072 | 0.0044 | 0.0000 | 0.0122 | 0.0000 |
| scvi_centred_k6__perm1 | own_h | 0.0073 | 0.0064 | 0.0000 | 0.0104 | 0.0000 |
| scvi_centred_k6__perm1 | own_dp | 0.0066 | 0.0056 | 0.0000 | 0.0111 | 0.0000 |
| scvi_centred_k8__perm1 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| scvi_centred_k8__perm1 | dE | 0.0085 | 0.0052 | 0.0000 | 0.0146 | 0.0000 |
| scvi_centred_k8__perm1 | own_h | 0.0082 | 0.0070 | 0.0000 | 0.0098 | 0.0000 |
| scvi_centred_k8__perm1 | own_dp | 0.0076 | 0.0064 | 0.0000 | 0.0128 | 0.0000 |
| scvi_absolute_k4__perm1 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm1 | dE | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm1 | own_h | 0.0004 | 0.0004 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm1 | own_dp | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6__perm1 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6__perm1 | dE | 0.0003 | 0.0003 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k6__perm1 | own_h | 0.0004 | 0.0004 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k6__perm1 | own_dp | 0.0004 | 0.0004 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k8__perm1 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k8__perm1 | dE | 0.0004 | 0.0003 | 0.0000 | 0.0011 | 0.0000 |
| scvi_absolute_k8__perm1 | own_h | 0.0005 | 0.0005 | 0.0000 | 0.0008 | 0.0000 |
| scvi_absolute_k8__perm1 | own_dp | 0.0004 | 0.0004 | 0.0000 | 0.0012 | 0.0000 |
| hvg_k4__perm1 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0002 | 0.0000 |
| hvg_k4__perm1 | dE | 0.0060 | 0.0044 | 0.0000 | 0.0074 | 0.0000 |
| hvg_k4__perm1 | own_h | 0.0069 | 0.0063 | 0.0000 | 0.0119 | 0.0000 |
| hvg_k4__perm1 | own_dp | 0.0052 | 0.0047 | 0.0000 | 0.0095 | 0.0000 |
| hvg_k6__perm1 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| hvg_k6__perm1 | dE | 0.0100 | 0.0068 | 0.0000 | 0.0155 | 0.0000 |
| hvg_k6__perm1 | own_h | 0.0093 | 0.0082 | 0.0000 | 0.0133 | 0.0000 |
| hvg_k6__perm1 | own_dp | 0.0073 | 0.0063 | 0.0000 | 0.0135 | 0.0000 |
| hvg_k8__perm1 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0004 | 0.0000 |
| hvg_k8__perm1 | dE | 0.0120 | 0.0080 | 0.0000 | 0.0194 | 0.0000 |
| hvg_k8__perm1 | own_h | 0.0102 | 0.0087 | 0.0000 | 0.0174 | 0.0000 |
| hvg_k8__perm1 | own_dp | 0.0086 | 0.0073 | 0.0000 | 0.0160 | 0.0000 |
| tx1_k4__perm1 | dH | 0.0002 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| tx1_k4__perm1 | dE | 0.0076 | 0.0060 | 0.0000 | 0.0056 | 0.0000 |
| tx1_k4__perm1 | own_h | 0.0154 | 0.0147 | 0.0000 | 0.0059 | 0.0000 |
| tx1_k4__perm1 | own_dp | 0.0076 | 0.0071 | 0.0000 | 0.0075 | 0.0000 |
| tx1_k6__perm1 | dH | 0.0005 | 0.0004 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k6__perm1 | dE | 0.0139 | 0.0105 | 0.0000 | 0.0092 | 0.0000 |
| tx1_k6__perm1 | own_h | 0.0221 | 0.0208 | 0.0000 | 0.0063 | 0.0000 |
| tx1_k6__perm1 | own_dp | 0.0125 | 0.0114 | 0.0000 | 0.0075 | 0.0000 |
| tx1_k8__perm1 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k8__perm1 | dE | 0.0109 | 0.0077 | 0.0000 | 0.0109 | 0.0000 |
| tx1_k8__perm1 | own_h | 0.0198 | 0.0185 | 0.0000 | 0.0095 | 0.0000 |
| tx1_k8__perm1 | own_dp | 0.0104 | 0.0093 | 0.0000 | 0.0101 | 0.0000 |
| cycle__perm2 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| cycle__perm2 | dE | 0.0077 | 0.0054 | 0.0000 | 0.0063 | 0.0000 |
| cycle__perm2 | own_h | 0.0103 | 0.0095 | 0.0000 | 0.0051 | 0.0000 |
| cycle__perm2 | own_dp | 0.0083 | 0.0075 | 0.0000 | 0.0057 | 0.0000 |
| programs__perm2 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| programs__perm2 | dE | 0.0063 | 0.0038 | 0.0000 | 0.0105 | 0.0000 |
| programs__perm2 | own_h | 0.0322 | 0.0285 | 0.0000 | 0.0424 | 0.0000 |
| programs__perm2 | own_dp | 0.0051 | 0.0044 | 0.0000 | 0.0080 | 0.0000 |
| scvi_centred_k4__perm2 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| scvi_centred_k4__perm2 | dE | 0.0045 | 0.0030 | 0.0000 | 0.0060 | 0.0000 |
| scvi_centred_k4__perm2 | own_h | 0.0054 | 0.0049 | 0.0000 | 0.0061 | 0.0000 |
| scvi_centred_k4__perm2 | own_dp | 0.0052 | 0.0047 | 0.0000 | 0.0060 | 0.0000 |
| scvi_centred_k6__perm2 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| scvi_centred_k6__perm2 | dE | 0.0073 | 0.0045 | 0.0000 | 0.0123 | 0.0000 |
| scvi_centred_k6__perm2 | own_h | 0.0072 | 0.0063 | 0.0000 | 0.0111 | 0.0000 |
| scvi_centred_k6__perm2 | own_dp | 0.0067 | 0.0058 | 0.0000 | 0.0108 | 0.0000 |
| scvi_centred_k8__perm2 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| scvi_centred_k8__perm2 | dE | 0.0085 | 0.0051 | 0.0000 | 0.0148 | 0.0000 |
| scvi_centred_k8__perm2 | own_h | 0.0085 | 0.0073 | 0.0000 | 0.0115 | 0.0000 |
| scvi_centred_k8__perm2 | own_dp | 0.0076 | 0.0065 | 0.0000 | 0.0127 | 0.0000 |
| scvi_absolute_k4__perm2 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm2 | dE | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm2 | own_h | 0.0004 | 0.0004 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k4__perm2 | own_dp | 0.0003 | 0.0003 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6__perm2 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k6__perm2 | dE | 0.0003 | 0.0003 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k6__perm2 | own_h | 0.0004 | 0.0004 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k6__perm2 | own_dp | 0.0004 | 0.0004 | 0.0000 | 0.0001 | 0.0000 |
| scvi_absolute_k8__perm2 | dH | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scvi_absolute_k8__perm2 | dE | 0.0004 | 0.0003 | 0.0000 | 0.0011 | 0.0000 |
| scvi_absolute_k8__perm2 | own_h | 0.0005 | 0.0005 | 0.0000 | 0.0011 | 0.0000 |
| scvi_absolute_k8__perm2 | own_dp | 0.0004 | 0.0004 | 0.0000 | 0.0016 | 0.0000 |
| hvg_k4__perm2 | dH | 0.0002 | 0.0001 | 0.0000 | 0.0002 | 0.0000 |
| hvg_k4__perm2 | dE | 0.0060 | 0.0044 | 0.0000 | 0.0074 | 0.0000 |
| hvg_k4__perm2 | own_h | 0.0074 | 0.0067 | 0.0000 | 0.0094 | 0.0000 |
| hvg_k4__perm2 | own_dp | 0.0049 | 0.0044 | 0.0000 | 0.0088 | 0.0000 |
| hvg_k6__perm2 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0003 | 0.0000 |
| hvg_k6__perm2 | dE | 0.0100 | 0.0068 | 0.0000 | 0.0155 | 0.0000 |
| hvg_k6__perm2 | own_h | 0.0093 | 0.0080 | 0.0000 | 0.0117 | 0.0000 |
| hvg_k6__perm2 | own_dp | 0.0074 | 0.0063 | 0.0000 | 0.0132 | 0.0000 |
| hvg_k8__perm2 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0004 | 0.0000 |
| hvg_k8__perm2 | dE | 0.0119 | 0.0080 | 0.0000 | 0.0196 | 0.0000 |
| hvg_k8__perm2 | own_h | 0.0103 | 0.0088 | 0.0000 | 0.0163 | 0.0000 |
| hvg_k8__perm2 | own_dp | 0.0082 | 0.0070 | 0.0000 | 0.0143 | 0.0000 |
| tx1_k4__perm2 | dH | 0.0002 | 0.0002 | 0.0000 | 0.0001 | 0.0000 |
| tx1_k4__perm2 | dE | 0.0077 | 0.0061 | 0.0000 | 0.0053 | 0.0000 |
| tx1_k4__perm2 | own_h | 0.0164 | 0.0157 | 0.0000 | 0.0059 | 0.0000 |
| tx1_k4__perm2 | own_dp | 0.0077 | 0.0072 | 0.0000 | 0.0066 | 0.0000 |
| tx1_k6__perm2 | dH | 0.0005 | 0.0004 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k6__perm2 | dE | 0.0137 | 0.0104 | 0.0000 | 0.0090 | 0.0000 |
| tx1_k6__perm2 | own_h | 0.0208 | 0.0195 | 0.0000 | 0.0071 | 0.0000 |
| tx1_k6__perm2 | own_dp | 0.0123 | 0.0112 | 0.0000 | 0.0094 | 0.0000 |
| tx1_k8__perm2 | dH | 0.0003 | 0.0002 | 0.0000 | 0.0002 | 0.0000 |
| tx1_k8__perm2 | dE | 0.0109 | 0.0077 | 0.0000 | 0.0110 | 0.0000 |
| tx1_k8__perm2 | own_h | 0.0201 | 0.0188 | 0.0000 | 0.0084 | 0.0000 |
| tx1_k8__perm2 | own_dp | 0.0105 | 0.0095 | 0.0000 | 0.0103 | 0.0000 |

## Projection of lines from deposits no training line comes from

0 of 0 such lines project identically when projected again.
