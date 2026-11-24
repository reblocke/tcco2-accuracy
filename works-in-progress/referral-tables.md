# TcCO₂ referral policies in an ABG-tested population

Private aggregate review copy. Each eligible patient has total weight one, split over arterial result records on the first eligible testing day. Each policy assumes a usable, contemporaneous TcCO₂ and a timely, definitive ABG after referral. Other indications for arterial sampling include pH and PaO₂. Their frequency, sensor failure frequency, and actual punctures avoided are unknown.

The primary policies limit the modeled probability of a wrong call at each displayed integer. The class-denominator alternatives limit FNR among all reference-positive and FPR among all reference-negative occasions. All arterial thresholds are inclusive at 45 mmHg. Parentheses give pointwise 95% credible intervals for fixed policies. Operating characteristics are posterior medians. Figure bars and totals are posterior means.

## All settings

### Cutoffs and efficiency

| Policy | Last negative | ABG displays | First positive | Classified without ABG (%) | Referrals /100 | Residual FN /100 | Residual FP /100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 39 | 40–50 | 51 | 63.05 (62.00–64.08) | 36.95 (35.92–38.00) | 1.03 (0.77–1.37) | 1.79 (1.28–2.50) |
| cap 1% | 30 | 31–71 | 72 | 17.61 (16.57–18.75) | 82.39 (81.25–83.43) | 0.08 (0.04–0.15) | 0.01 (0.00–0.04) |
| cap 5% | 38 | 39–60 | 61 | 48.63 (46.99–50.28) | 51.37 (49.72–53.01) | 0.77 (0.56–1.07) | 0.12 (0.05–0.26) |
| cap 10% | 40 | 41–56 | 57 | 59.35 (57.88–60.80) | 40.65 (39.20–42.12) | 1.37 (1.05–1.77) | 0.33 (0.18–0.60) |
| class 1% | 34 | 35–54 | 55 | 38.45 (37.21–39.72) | 61.55 (60.28–62.79) | 0.25 (0.15–0.40) | 0.57 (0.34–0.95) |
| class 5% | 40 | 41–48 | 49 | 70.92 (70.15–71.68) | 29.08 (28.32–29.85) | 1.37 (1.05–1.77) | 3.21 (2.46–4.18) |
| class 10% | 42 | 43–46 | 47 | 83.32 (82.92–83.70) | 16.68 (16.30–17.08) | 2.38 (1.94–2.89) | 5.64 (4.59–6.90) |

### Class and call denominators

| Policy | Negative /100 | Positive /100 | Off-grid referrals /100 | Wrong among negative (%) | Wrong among positive (%) | Whole-strategy FNR (%) | Whole-strategy FPR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 44.71 (42.76–46.66) | 18.31 (17.30–19.47) | 3.98 (3.62–4.51) | 2.30 (1.74–3.03) | 9.77 (7.29–13.00) | 3.51 (2.61–4.67) | 2.53 (1.81–3.55) |
| cap 1% | 13.95 (12.83–15.16) | 3.66 (3.54–3.80) | 3.98 (3.62–4.51) | 0.59 (0.32–1.06) | 0.28 (0.07–1.00) | 0.28 (0.15–0.52) | 0.01 (0.00–0.05) |
| cap 5% | 40.76 (38.84–42.71) | 7.86 (7.53–8.27) | 3.98 (3.62–4.51) | 1.90 (1.38–2.58) | 1.51 (0.69–3.22) | 2.63 (1.89–3.63) | 0.17 (0.07–0.37) |
| cap 10% | 48.63 (46.68–50.56) | 10.71 (10.20–11.34) | 3.98 (3.62–4.51) | 2.81 (2.19–3.58) | 3.07 (1.73–5.39) | 4.66 (3.58–6.01) | 0.47 (0.25–0.86) |
| class 1% | 25.75 (24.11–27.44) | 12.68 (12.04–13.46) | 3.98 (3.62–4.51) | 0.96 (0.60–1.51) | 4.47 (2.78–7.13) | 0.84 (0.52–1.35) | 0.80 (0.48–1.35) |
| class 5% | 48.63 (46.68–50.56) | 22.27 (21.03–23.64) | 3.98 (3.62–4.51) | 2.81 (2.19–3.58) | 14.43 (11.60–17.78) | 4.66 (3.58–6.01) | 4.55 (3.48–5.92) |
| class 10% | 56.15 (54.26–57.96) | 27.16 (25.67–28.77) | 3.98 (3.62–4.51) | 4.23 (3.49–5.08) | 20.77 (17.74–24.10) | 8.09 (6.59–9.83) | 7.98 (6.50–9.77) |

### Reading-specific design risks

Values are posterior-integrated percentages. The numerical upper bounds incorporate chain-batch Monte Carlo and quadrature error. Off-grid displays always go to ABG.

| Policy | Definitive call | Boundary risk (%) | Worst risk (%) | Numerical upper bound (%) |
| --- | --- | --- | --- | --- |
| fixed | negative | 6.5283 | 6.5283 | 6.5285 |
| fixed | positive | 27.3837 | 27.3837 | 27.3843 |
| cap 1% | negative | 0.8874 | 0.8874 | 0.8875 |
| cap 1% | positive | 0.9041 | 0.9041 | 0.9042 |
| cap 5% | negative | 4.9402 | 4.9402 | 4.9403 |
| cap 5% | positive | 4.8249 | 4.8249 | 4.8253 |
| cap 10% | negative | 8.6806 | 8.6806 | 8.6808 |
| cap 10% | positive | 9.4005 | 9.4005 | 9.4010 |
| class 1% | negative | 1.8421 | 1.8421 | 1.8422 |
| class 1% | positive | 13.3001 | 13.3001 | 13.3007 |
| class 5% | negative | 8.6806 | 8.6806 | 8.6808 |
| class 5% | positive | 38.9227 | 38.9227 | 38.9234 |
| class 10% | negative | 15.2922 | 15.2922 | 15.2926 |
| class 10% | positive | 53.1084 | 53.1084 | 53.1090 |

### Paired changes per 100 eligible occasions

| Policy contrast | Referrals | Classified without ABG | False negatives | False positives |
| --- | --- | --- | --- | --- |
| cap 1% minus fixed | 45.44 (44.98–45.80) | -45.44 (-45.80–-44.98) | -0.95 (-1.23–-0.72) | -1.78 (-2.47–-1.27) |
| cap 5% minus fixed | 14.41 (13.75–15.11) | -14.41 (-15.11–-13.75) | -0.26 (-0.31–-0.21) | -1.67 (-2.27–-1.22) |
| cap 10% minus fixed | 3.68 (3.16–4.28) | -3.68 (-4.28–-3.16) | 0.34 (0.28–0.40) | -1.46 (-1.94–-1.08) |
| cap 1% minus cap 5% | 31.01 (30.18–31.84) | -31.01 (-31.84–-30.18) | -0.69 (-0.92–-0.51) | -0.11 (-0.23–-0.05) |
| cap 5% minus cap 10% | 10.73 (10.46–10.96) | -10.73 (-10.96–-10.46) | -0.59 (-0.72–-0.49) | -0.21 (-0.35–-0.13) |

No unresolved named display boundary.
The two reference-class constraints gave a unique numerical optimum at each cap.

## ICU billing proxy

### Cutoffs and efficiency

| Policy | Last negative | ABG displays | First positive | Classified without ABG (%) | Referrals /100 | Residual FN /100 | Residual FP /100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 39 | 40–50 | 51 | 64.86 (63.07–66.54) | 35.14 (33.46–36.93) | 0.92 (0.50–1.69) | 1.43 (0.76–2.86) |
| cap 1% | 31 | 32–65 | 66 | 25.48 (23.30–28.01) | 74.52 (71.99–76.70) | 0.07 (0.02–0.25) | 0.01 (0.00–0.09) |
| cap 5% | 37 | 38–58 | 59 | 48.89 (45.97–51.65) | 51.11 (48.35–54.03) | 0.50 (0.23–1.06) | 0.09 (0.02–0.40) |
| cap 10% | 40 | 41–55 | 56 | 62.00 (59.38–64.26) | 38.00 (35.74–40.62) | 1.23 (0.72–2.13) | 0.25 (0.08–0.81) |
| class 1% | 35 | 36–53 | 54 | 46.55 (44.36–48.87) | 53.45 (51.13–55.64) | 0.26 (0.10–0.66) | 0.50 (0.21–1.33) |
| class 5% | 40 | 41–48 | 49 | 71.97 (70.70–73.14) | 28.03 (26.86–29.30) | 1.23 (0.72–2.13) | 2.74 (1.67–4.73) |
| class 10% | 43 | 44–45 | 46 | 88.69 (87.71–89.10) | 11.31 (10.90–12.29) | 2.78 (1.91–4.02) | 6.51 (4.60–9.41) |

### Class and call denominators

| Policy | Negative /100 | Positive /100 | Off-grid referrals /100 | Wrong among negative (%) | Wrong among positive (%) | Whole-strategy FNR (%) | Whole-strategy FPR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 44.74 (40.84–48.08) | 20.09 (18.39–22.44) | 5.67 (5.00–7.02) | 2.06 (1.16–3.68) | 7.12 (4.03–12.94) | 2.91 (1.59–5.36) | 2.09 (1.11–4.18) |
| cap 1% | 18.13 (15.56–21.03) | 7.34 (6.94–7.86) | 5.67 (5.00–7.02) | 0.39 (0.11–1.29) | 0.11 (0.01–1.21) | 0.22 (0.06–0.79) | 0.01 (0.00–0.14) |
| cap 5% | 37.63 (33.86–41.09) | 11.25 (10.52–12.26) | 5.67 (5.00–7.02) | 1.32 (0.64–2.73) | 0.76 (0.18–3.35) | 1.57 (0.74–3.34) | 0.12 (0.03–0.58) |
| cap 10% | 48.23 (44.33–51.46) | 13.75 (12.77–15.16) | 5.67 (5.00–7.02) | 2.57 (1.54–4.30) | 1.80 (0.62–5.46) | 3.90 (2.28–6.73) | 0.36 (0.12–1.18) |
| class 1% | 30.61 (27.12–34.05) | 15.90 (14.67–17.62) | 5.67 (5.00–7.02) | 0.86 (0.35–2.06) | 3.17 (1.37–7.71) | 0.83 (0.33–2.07) | 0.74 (0.30–1.94) |
| class 5% | 48.23 (44.33–51.46) | 23.72 (21.62–26.54) | 5.67 (5.00–7.02) | 2.57 (1.54–4.30) | 11.57 (7.56–18.01) | 3.90 (2.28–6.73) | 4.01 (2.44–6.92) |
| class 10% | 57.98 (54.32–60.75) | 30.66 (27.86–34.15) | 5.67 (5.00–7.02) | 4.80 (3.40–6.81) | 21.24 (16.35–27.80) | 8.78 (6.05–12.72) | 9.52 (6.73–13.76) |

### Reading-specific design risks

Values are posterior-integrated percentages. The numerical upper bounds incorporate chain-batch Monte Carlo and quadrature error. Off-grid displays always go to ABG.

| Policy | Definitive call | Boundary risk (%) | Worst risk (%) | Numerical upper bound (%) |
| --- | --- | --- | --- | --- |
| fixed | negative | 6.9076 | 6.9076 | 6.9083 |
| fixed | positive | 27.1787 | 27.1787 | 27.1808 |
| cap 1% | negative | 0.8634 | 0.8634 | 0.8636 |
| cap 1% | positive | 0.8638 | 0.8638 | 0.8643 |
| cap 5% | negative | 3.9405 | 3.9405 | 3.9410 |
| cap 5% | positive | 4.2556 | 4.2556 | 4.2568 |
| cap 10% | negative | 9.1383 | 9.1383 | 9.1391 |
| cap 10% | positive | 8.7695 | 8.7695 | 8.7710 |
| class 1% | negative | 2.2868 | 2.2868 | 2.2871 |
| class 1% | positive | 14.0672 | 14.0672 | 14.0691 |
| class 5% | negative | 9.1383 | 9.1383 | 9.1391 |
| class 5% | positive | 39.6910 | 39.6910 | 39.6931 |
| class 10% | negative | 20.2061 | 20.2061 | 20.2074 |
| class 10% | positive | 61.3401 | 61.3401 | 61.3419 |

### Paired changes per 100 eligible occasions

| Policy contrast | Referrals | Classified without ABG | False negatives | False positives |
| --- | --- | --- | --- | --- |
| cap 1% minus fixed | 39.39 (38.16–40.05) | -39.39 (-40.05–-38.16) | -0.85 (-1.47–-0.48) | -1.42 (-2.78–-0.76) |
| cap 5% minus fixed | 15.97 (14.84–17.16) | -15.97 (-17.16–-14.84) | -0.42 (-0.66–-0.26) | -1.34 (-2.51–-0.73) |
| cap 10% minus fixed | 2.85 (2.18–3.86) | -2.85 (-3.86–-2.18) | 0.31 (0.21–0.45) | -1.17 (-2.11–-0.66) |
| cap 1% minus cap 5% | 23.37 (22.30–24.22) | -23.37 (-24.22–-22.30) | -0.42 (-0.82–-0.21) | -0.08 (-0.31–-0.02) |
| cap 5% minus cap 10% | 13.12 (12.37–13.63) | -13.12 (-13.63–-12.37) | -0.73 (-1.11–-0.47) | -0.16 (-0.42–-0.06) |

No unresolved named display boundary.
The two reference-class constraints gave a unique numerical optimum at each cap.

## ED/noncritical inpatient

### Cutoffs and efficiency

| Policy | Last negative | ABG displays | First positive | Classified without ABG (%) | Referrals /100 | Residual FN /100 | Residual FP /100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 39 | 40–50 | 51 | 64.99 (62.15–67.58) | 35.01 (32.42–37.85) | 2.59 (1.54–3.98) | 1.80 (0.79–3.92) |
| cap 1% | None | 20–100 | None | 0.00 (0.00–0.00) | 100.00 (100.00–100.00) | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| cap 5% | 34 | 35–70 | 71 | 33.48 (29.18–37.66) | 66.52 (62.34–70.82) | 0.91 (0.43–1.71) | 0.05 (0.01–0.37) |
| cap 10% | 37 | 38–60 | 61 | 48.28 (43.87–52.29) | 51.72 (47.71–56.13) | 1.71 (0.93–2.85) | 0.25 (0.05–1.04) |
| class 1% | 28 | 29–56 | 57 | 21.05 (18.72–23.72) | 78.95 (76.28–81.28) | 0.25 (0.09–0.59) | 0.52 (0.14–1.67) |
| class 5% | 35 | 36–48 | 49 | 52.90 (50.25–55.34) | 47.10 (44.66–49.75) | 1.13 (0.55–2.03) | 2.90 (1.48–5.52) |
| class 10% | 39 | 40–45 | 46 | 75.12 (73.61–76.46) | 24.88 (23.54–26.39) | 2.59 (1.54–3.98) | 6.03 (3.74–9.60) |

### Class and call denominators

| Policy | Negative /100 | Positive /100 | Off-grid referrals /100 | Wrong among negative (%) | Wrong among positive (%) | Whole-strategy FNR (%) | Whole-strategy FPR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 49.98 (44.67–54.57) | 14.94 (12.85–18.02) | 5.01 (3.37–8.14) | 5.18 (3.19–7.84) | 12.09 (5.85–22.43) | 9.30 (5.54–14.30) | 2.49 (1.09–5.43) |
| cap 1% | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 5.01 (3.37–8.14) | Undefined | Undefined | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| cap 5% | 30.68 (26.15–35.14) | 2.75 (2.46–3.36) | 5.01 (3.37–8.14) | 2.98 (1.48–5.37) | 2.01 (0.20–11.09) | 3.29 (1.53–6.16) | 0.08 (0.01–0.51) |
| cap 10% | 42.18 (37.05–46.88) | 6.02 (5.29–7.38) | 5.01 (3.37–8.14) | 4.06 (2.29–6.61) | 4.22 (0.90–14.36) | 6.15 (3.33–10.24) | 0.35 (0.07–1.43) |
| class 1% | 12.51 (9.95–15.37) | 8.45 (7.35–10.33) | 5.01 (3.37–8.14) | 2.02 (0.79–4.27) | 6.13 (1.80–16.55) | 0.91 (0.31–2.11) | 0.71 (0.19–2.31) |
| class 5% | 34.41 (29.64–39.00) | 18.34 (15.76–21.96) | 5.01 (3.37–8.14) | 3.28 (1.70–5.71) | 15.82 (8.98–25.84) | 4.05 (1.99–7.29) | 4.01 (2.05–7.64) |
| class 10% | 49.98 (44.67–54.57) | 25.08 (21.63–29.59) | 5.01 (3.37–8.14) | 5.18 (3.19–7.84) | 24.08 (16.80–33.12) | 9.30 (5.54–14.30) | 8.35 (5.18–13.30) |

### Reading-specific design risks

Values are posterior-integrated percentages. The numerical upper bounds incorporate chain-batch Monte Carlo and quadrature error. Off-grid displays always go to ABG.

| Policy | Definitive call | Boundary risk (%) | Worst risk (%) | Numerical upper bound (%) |
| --- | --- | --- | --- | --- |
| fixed | negative | 12.4364 | 12.4364 | 12.4370 |
| fixed | positive | 26.1364 | 26.1364 | 26.1377 |
| cap 1% | negative | Undefined | Undefined | Undefined |
| cap 1% | positive | Undefined | Undefined | Undefined |
| cap 5% | negative | 4.9232 | 4.9232 | 4.9235 |
| cap 5% | positive | 4.7443 | 4.7443 | 4.7452 |
| cap 10% | negative | 8.2924 | 8.2924 | 8.2928 |
| cap 10% | positive | 9.3914 | 9.3914 | 9.3924 |
| class 1% | negative | 2.5228 | 2.5228 | 2.5230 |
| class 1% | positive | 13.2163 | 13.2163 | 13.2174 |
| class 5% | negative | 5.7784 | 5.7784 | 5.7788 |
| class 5% | positive | 34.6934 | 34.6934 | 34.6947 |
| class 10% | negative | 12.4364 | 12.4364 | 12.4370 |
| class 10% | positive | 52.4674 | 52.4674 | 52.4686 |

### Paired changes per 100 eligible occasions

| Policy contrast | Referrals | Classified without ABG | False negatives | False positives |
| --- | --- | --- | --- | --- |
| cap 1% minus fixed | 64.99 (62.15–67.58) | -64.99 (-67.58–-62.15) | -2.59 (-3.98–-1.54) | -1.80 (-3.92–-0.79) |
| cap 5% minus fixed | 31.56 (29.34–33.26) | -31.56 (-33.26–-29.34) | -1.66 (-2.37–-1.08) | -1.74 (-3.60–-0.78) |
| cap 10% minus fixed | 16.72 (15.11–18.43) | -16.72 (-18.43–-15.11) | -0.87 (-1.20–-0.59) | -1.53 (-2.99–-0.72) |
| cap 1% minus cap 5% | 33.48 (29.18–37.66) | -33.48 (-37.66–-29.18) | -0.91 (-1.71–-0.43) | -0.05 (-0.37–-0.01) |
| cap 5% minus cap 10% | 14.78 (13.88–15.56) | -14.78 (-15.56–-13.88) | -0.79 (-1.18–-0.48) | -0.20 (-0.67–-0.04) |

No unresolved named display boundary.
The two reference-class constraints gave a unique numerical optimum at each cap.

## Ambulatory

### Cutoffs and efficiency

| Policy | Last negative | ABG displays | First positive | Classified without ABG (%) | Referrals /100 | Residual FN /100 | Residual FP /100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 39 | 40–50 | 51 | 60.64 (55.85–66.04) | 39.36 (33.96–44.15) | 0.65 (0.14–2.54) | 0.67 (0.11–4.00) |
| cap 1% | 33 | 34–100 | None | 20.31 (14.66–28.24) | 79.69 (71.76–85.34) | 0.08 (0.00–0.67) | 0.00 (0.00–0.00) |
| cap 5% | 39 | 40–57 | 58 | 52.61 (45.32–59.63) | 47.39 (40.37–54.68) | 0.65 (0.14–2.54) | 0.05 (0.00–1.01) |
| cap 10% | 41 | 42–53 | 54 | 65.68 (59.65–70.98) | 34.32 (29.02–40.35) | 1.37 (0.42–3.98) | 0.21 (0.02–2.10) |
| class 1% | 35 | 36–52 | 53 | 38.80 (34.16–45.86) | 61.20 (54.14–65.84) | 0.15 (0.01–1.04) | 0.30 (0.03–2.58) |
| class 5% | 40 | 41–47 | 48 | 72.13 (69.06–75.66) | 27.87 (24.34–30.94) | 0.95 (0.24–3.18) | 2.32 (0.76–8.03) |
| class 10% | 42 | 43–45 | 46 | 87.31 (85.98–88.61) | 12.69 (11.39–14.02) | 1.97 (0.71–4.95) | 5.16 (2.38–12.64) |

### Class and call denominators

| Policy | Negative /100 | Positive /100 | Off-grid referrals /100 | Wrong among negative (%) | Wrong among positive (%) | Whole-strategy FNR (%) | Whole-strategy FPR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 46.93 (37.80–54.98) | 13.52 (10.81–19.26) | 0.88 (0.45–3.14) | 1.41 (0.31–5.07) | 5.01 (0.91–21.82) | 2.47 (0.51–9.56) | 0.91 (0.15–5.45) |
| cap 1% | 20.31 (14.66–28.24) | 0.00 (0.00–0.00) | 0.88 (0.45–3.14) | 0.38 (0.02–2.80) | Undefined | 0.29 (0.02–2.53) | 0.00 (0.00–0.00) |
| cap 5% | 46.93 (37.80–54.98) | 5.57 (4.56–8.20) | 0.88 (0.45–3.14) | 1.41 (0.31–5.07) | 0.92 (0.02–13.24) | 2.47 (0.51–9.56) | 0.07 (0.00–1.38) |
| cap 10% | 56.45 (46.94–63.55) | 9.14 (7.36–13.30) | 0.88 (0.45–3.14) | 2.45 (0.79–6.72) | 2.30 (0.18–16.87) | 5.18 (1.58–15.01) | 0.28 (0.02–2.85) |
| class 1% | 28.19 (21.20–36.72) | 10.40 (8.35–15.04) | 0.88 (0.45–3.14) | 0.55 (0.05–3.23) | 2.96 (0.31–18.25) | 0.57 (0.05–3.90) | 0.41 (0.04–3.52) |
| class 5% | 51.75 (42.36–59.36) | 20.22 (16.10–27.71) | 0.88 (0.45–3.14) | 1.85 (0.49–5.83) | 11.60 (4.33–29.88) | 3.58 (0.90–12.00) | 3.16 (1.04–10.93) |
| class 10% | 60.93 (51.47–67.49) | 26.33 (21.03–34.87) | 0.88 (0.45–3.14) | 3.26 (1.24–7.78) | 19.72 (10.66–37.01) | 7.44 (2.68–18.65) | 7.03 (3.24–17.20) |

### Reading-specific design risks

Values are posterior-integrated percentages. The numerical upper bounds incorporate chain-batch Monte Carlo and quadrature error. Off-grid displays always go to ABG.

| Policy | Definitive call | Boundary risk (%) | Worst risk (%) | Numerical upper bound (%) |
| --- | --- | --- | --- | --- |
| fixed | negative | 4.6784 | 4.6784 | 4.6790 |
| fixed | positive | 16.1032 | 16.1032 | 16.1058 |
| cap 1% | negative | 0.9852 | 0.9852 | 0.9855 |
| cap 1% | positive | Undefined | Undefined | Undefined |
| cap 5% | negative | 4.6784 | 4.6784 | 4.6790 |
| cap 5% | positive | 4.7140 | 4.7140 | 4.7156 |
| cap 10% | negative | 9.4821 | 9.4821 | 9.4831 |
| cap 10% | positive | 8.6550 | 8.6550 | 8.6571 |
| class 1% | negative | 1.4874 | 1.4874 | 1.4878 |
| class 1% | positive | 10.4873 | 10.4873 | 10.4896 |
| class 5% | negative | 6.6136 | 6.6136 | 6.6144 |
| class 5% | positive | 33.2458 | 33.2458 | 33.2486 |
| class 10% | negative | 13.6612 | 13.6612 | 13.6624 |
| class 10% | positive | 52.5291 | 52.5291 | 52.5313 |

### Paired changes per 100 eligible occasions

| Policy contrast | Referrals | Classified without ABG | False negatives | False positives |
| --- | --- | --- | --- | --- |
| cap 1% minus fixed | 40.31 (37.55–41.52) | -40.31 (-41.52–-37.55) | -0.57 (-1.95–-0.13) | -0.67 (-4.00–-0.11) |
| cap 5% minus fixed | 7.94 (6.24–11.21) | -7.94 (-11.21–-6.24) | 0.00 (0.00–0.00) | -0.61 (-3.07–-0.11) |
| cap 10% minus fixed | -5.10 (-6.10–-2.48) | 5.10 (2.48–6.10) | 0.70 (0.27–1.55) | -0.45 (-2.01–-0.09) |
| cap 1% minus cap 5% | 32.12 (29.52–33.73) | -32.12 (-33.73–-29.52) | -0.57 (-1.95–-0.13) | -0.05 (-1.01–-0.00) |
| cap 5% minus cap 10% | 13.10 (11.20–14.38) | -13.10 (-14.38–-11.20) | -0.70 (-1.55–-0.27) | -0.15 (-1.12–-0.01) |

Display-risk holds: cap 1% [100]
The two reference-class constraints gave a unique numerical optimum at each cap.

## Does rounding the display change the ABG band?

The first column of bands uses exact-reading point estimates at integer values. The second applies their numerical bounds. The final column integrates the entire rounded display bin and applies its bounds. Every band is inclusive and confined to the 20–100 grid.

| Setting | Cap | Exact reading, point estimate | Exact reading, qualified | Rounded display, qualified |
| --- | --- | --- | --- | --- |
| All settings | cap 1% | 31–71 | 31–71 | 31–71 |
| All settings | cap 5% | 39–60 | 39–60 | 39–60 |
| All settings | cap 10% | 41–56 | 41–56 | 41–56 |
| ICU billing proxy | cap 1% | 32–65 | 32–65 | 32–65 |
| ICU billing proxy | cap 5% | 38–58 | 38–58 | 38–58 |
| ICU billing proxy | cap 10% | 41–55 | 41–55 | 41–55 |
| ED/noncritical inpatient | cap 1% | 20–100 | 20–100 | 20–100 |
| ED/noncritical inpatient | cap 5% | 35–70 | 35–70 | 35–70 |
| ED/noncritical inpatient | cap 10% | 38–60 | 38–60 | 38–60 |
| Ambulatory | cap 1% | 34–99 | 34–100 | 34–100 |
| Ambulatory | cap 5% | 40–57 | 40–57 | 40–57 |
| Ambulatory | cap 10% | 42–53 | 42–53 | 42–53 |

All settings had a maximum numerical radius of 0.0007 percentage points and a maximum quadrature change of 0.0000 percentage points. Downward risk steps occurred at the following displays in mmHg: none.
ICU billing proxy had a maximum numerical radius of 0.0022 percentage points and a maximum quadrature change of 0.0000 percentage points. Downward risk steps occurred at the following displays in mmHg: none.
ED/noncritical inpatient had a maximum numerical radius of 0.0013 percentage points and a maximum quadrature change of 0.0000 percentage points. Downward risk steps occurred at the following displays in mmHg: none.
Ambulatory had a maximum numerical radius of 0.0028 percentage points and a maximum quadrature change of 0.0000 percentage points. Downward risk steps occurred at the following displays in mmHg: 21, 22, 23, 24, 25, 85, 86, 87, 88, 89, 90.
