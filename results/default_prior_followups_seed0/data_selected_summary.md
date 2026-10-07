# Data-selected genes of the reference prior, run default_prior_followups_20261007

Gains are selective Spearman minus the reference row's (the full prior), with 95% paired line-bootstrap intervals. Features are ranked by summed absolute weight over the selective targets, on the training side.

| Variant | Val selective Spearman | Test selective Spearman | Val gain | Test gain |
| --- | ---: | ---: | --- | --- |
| reference | 0.2300 | 0.2352 | — | — |
| without the stage | 0.2234 | 0.2237 | -0.0065 [-0.0133, -0.0005] | -0.0115 [-0.0184, -0.0029] |
| top 10 only | 0.2231 | 0.2245 | -0.0069 [-0.0138, -0.0009] | -0.0106 [-0.0171, -0.0028] |
| without top 10 | 0.2307 | 0.2348 | 0.0008 [-0.0003, 0.0018] | -0.0003 [-0.0012, 0.0005] |
| top 50 only | 0.2238 | 0.2264 | -0.0062 [-0.0111, -0.0017] | -0.0088 [-0.0143, -0.0021] |
| without top 50 | 0.2303 | 0.2336 | 0.0003 [-0.0014, 0.0019] | -0.0016 [-0.0029, -0.0000] |
| top 200 only | 0.2243 | 0.2285 | -0.0057 [-0.0102, -0.0015] | -0.0067 [-0.0111, -0.0013] |
| without top 200 | 0.2306 | 0.2327 | 0.0006 [-0.0024, 0.0031] | -0.0024 [-0.0048, 0.0001] |
| top 1000 only | 0.2262 | 0.2303 | -0.0038 [-0.0062, -0.0013] | -0.0049 [-0.0073, -0.0018] |
| without top 1000 | 0.2296 | 0.2311 | -0.0004 [-0.0053, 0.0037] | -0.0040 [-0.0080, 0.0008] |

Expression genes used by the stage: 9636 of 9711; selections that are a paralog or complex partner of their target: 367 of 152850.

Median share of variance over the fit lines explained by lineage: 0.198 for the top 50 features, 0.138 for every feature used, 0.137 for every space gene; by the expression components: 0.743, 0.749 and 0.750.

Per-target gain of the stage (selective Spearman with minus without): mean 0.0065 validation, 0.0115 test; positive for 1657 and 1766 of 3057 targets; Spearman of the validation and test gains across targets 0.041.

| Decile by validation gain | Val mean gain | Test mean gain |
| ---: | ---: | ---: |
| 1 | 0.1188 | 0.0175 |
| 2 | 0.0677 | 0.0134 |
| 3 | 0.0472 | 0.0140 |
| 4 | 0.0303 | 0.0148 |
| 5 | 0.0142 | 0.0135 |
| 6 | -0.0009 | 0.0157 |
| 7 | -0.0162 | 0.0081 |
| 8 | -0.0332 | 0.0046 |
| 9 | -0.0563 | 0.0083 |
| 10 | -0.1069 | 0.0047 |

Top 30 features:

| Gene | Times selected | Summed abs weight | Partner of target | Selective | Lineage R2 | Components R2 | Heaviest targets |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| FBN1 | 310 | 1.226 | 0 | no | 0.354 | 0.808 | JUN, CAND1, KEAP1, FERMT2, NF2 |
| CDKN1A | 144 | 1.046 | 1 | no | 0.156 | 0.696 | UBE2Q1, MDM2, TP53BP1, PPM1G, PPM1D |
| VASN | 188 | 0.957 | 0 | no | 0.217 | 0.776 | PMVK, TWNK, COX19, DLST, SDHA |
| COL1A1 | 209 | 0.953 | 0 | no | 0.286 | 0.778 | SURF4, YRDC, SLC4A7, EEFSEC, PGM3 |
| ZMAT3 | 156 | 0.930 | 0 | no | 0.273 | 0.747 | MDM2, TP53BP1, ACD, PPM1D, CNOT2 |
| PSAT1 | 114 | 0.912 | 0 | yes | 0.114 | 0.528 | GATB, MRPS2, COA7, GATC, NDUFA10 |
| SGCE | 156 | 0.883 | 0 | yes | 0.290 | 0.631 | DNAJC2, MRPS15, MRPS33, MRPL12, MRPS14 |
| FN1 | 212 | 0.875 | 1 | no | 0.223 | 0.712 | UQCRC1, ITGB3, DOCK7, PPRC1, NDUFC1 |
| HPCAL1 | 140 | 0.776 | 0 | no | 0.173 | 0.715 | MRPL18, MRPL39, TARS2, GTPBP10, MRPL21 |
| COL8A1 | 196 | 0.753 | 0 | no | 0.198 | 0.680 | VCL, ATP5F1C, PTTG1, DHX29, MED12 |
| SLC25A5 | 111 | 0.721 | 0 | yes | 0.111 | 0.786 | MRPS6, RARS2, MRPL45, MRPL16, MRPL24 |
| CCDC80 | 202 | 0.715 | 0 | no | 0.199 | 0.786 | TSC1, ALG5, LMNB1, TSC2, JUN |
| PEA15 | 182 | 0.709 | 0 | yes | 0.241 | 0.868 | FERMT2, TWNK, LARS2, SLC25A26, MRPL33 |
| GLIPR1 | 204 | 0.691 | 0 | no | 0.191 | 0.774 | PRKAR1A, TRIO, JUN, COX5B, MRM2 |
| CALD1 | 168 | 0.675 | 0 | no | 0.380 | 0.784 | SLC4A7, MTHFD1, GMPS, ADSL, PFAS |
| ACTA2 | 122 | 0.662 | 2 | no | 0.286 | 0.630 | MDM2, PPM1G, GLRX3, FDPS, WDR89 |
| SERPINE2 | 132 | 0.662 | 0 | no | 0.323 | 0.724 | SOX10, MRPL45, NDUFA11, DUSP4, PDE12 |
| SYTL3 | 92 | 0.656 | 0 | no | 0.170 | 0.568 | MRPL52, NDUFA3, FASTKD5, ELAVL1, TRUB2 |
| PGAM5 | 152 | 0.645 | 0 | no | 0.196 | 0.820 | ISCA2, GFM1, MRPS16, MRPL18, MRPS10 |
| THBS1 | 175 | 0.637 | 0 | no | 0.132 | 0.767 | UBA5, UFM1, TLN1, COX5B, MICOS10 |
| RGS3 | 121 | 0.617 | 0 | no | 0.222 | 0.615 | MRPL58, NDUFAF4, RPUSD3, MARCHF9, MRPL17 |
| BAX | 90 | 0.614 | 0 | no | 0.140 | 0.739 | ACD, MDM2, TP53BP1, CCDC6, ZC3H13 |
| LRRC26 | 110 | 0.614 | 0 | no | 0.238 | 0.654 | RING1, RNASE13, SUZ12, PCGF1, PPP1R15B |
| SLC39A13 | 161 | 0.614 | 0 | no | 0.207 | 0.840 | IQGAP1, MRPL55, TLN1, MRM2, MRPL39 |
| BEND6 | 145 | 0.609 | 0 | no | 0.281 | 0.737 | MED7, TRMT10C, MED12, FBXO11, MRPL39 |
| ITGA1 | 110 | 0.609 | 0 | no | 0.181 | 0.580 | SUPV3L1, MRPL55, NF2, XPR1, TFB2M |
| RPS27L | 111 | 0.608 | 0 | no | 0.265 | 0.778 | MDM2, TP53BP1, PPM1D, USP7, PPM1G |
| GREM1 | 102 | 0.602 | 0 | no | 0.254 | 0.539 | RTF1, ZNHIT1, COX11, MRPL33, TRIT1 |
| ZNF747 | 83 | 0.592 | 0 | no | 0.050 | 0.590 | OXSM, ATRX, MECR, ALG13, GDI2 |
| PSD3 | 98 | 0.588 | 0 | no | 0.220 | 0.665 | MRPL12, COA7, NDUFB4, NDUFB11, MRPL15 |
