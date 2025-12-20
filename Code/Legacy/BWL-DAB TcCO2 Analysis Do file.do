
*======================================
*===1) Gases are statistically different between different encounter types
* Descriptives/ANOVA/histograms for gas variables by encounter type; density plots with truncated ranges; CC vs non‑CC comparisons within inpatients.
*======================================

*Inpatient v Ambulatory v Emergency
*PaCO2
table encounter_granular, statistic(mean paco2) statistic(median paco2) statistic(iqr paco2) 
oneway paco2 encounter_ty
twoway ///
    (histogram paco2 if encounter_granular==0 & is_cc!=1, color(blue%50) width(2)) ///
    (histogram paco2 if encounter_granular==1, color(red%50) width(2)) ///
    (histogram paco2 if encounter_granular==2, color(green%50) width(2)), ///
    legend(order(0 "Ambulatory" 1 "Inpatient/Emergency" 2 "ICU"))
preserve
recode paco2 (min/20=.) (70/max=.) //truncate the range ; Purpose: trim tails for nicer densities without altering the main dataset.
twoway (kdensity paco2 if encounter_granular == 1, lcolor(blue)) ///
       (kdensity paco2 if encounter_granular == 2, lcolor(red)) ///
	   (kdensity paco2 if encounter_granular == 3, lcolor(green)), ///
       legend (label(1 "Ambulatory ") label(2 "Inpatient/Emergency") label(3 "ICU")) ///
       xtitle("Arterial Partial Pressure of Carbon Dioxide (mmHg)", size(large)) ///
	   ytitle("Relative Frequency", size(large)) ///
	   title("Kernel Density of PaCO₂ by Encounter Type") ///
       xlabel(, grid) ylabel(, grid)
restore
*sHCO3-
table encounter_granular, statistic(mean serum_hco3) statistic(median serum_hco3) statistic(iqr serum_hco3) 
*Art pH
table encounter_granular, statistic(mean abg_ph) statistic(median abg_ph) statistic(iqr abg_ph)
oneway abg_ph encounter_ty
twoway ///
    (histogram abg_ph if is_amb==1 & is_cc!=1, color(blue%50) width(.01)) ///
    (histogram abg_ph if is_emer==1, color(red%50) width(.01)) ///
    (histogram abg_ph if is_inp==1, color(green%50) width(.01)), ///
    legend(order(1 "Ambulatory" 2 "Emergency" 3 "Inpatient"))
*Ven pH
table encounter_granular, statistic(mean vbg_ph)
oneway vbg_ph encounter_ty
twoway ///
    (histogram vbg_ph if is_amb==1 & is_cc!=1, color(blue%50) width(.01)) ///
    (histogram vbg_ph if is_emer==1, color(red%50) width(.01)) ///
    (histogram vbg_ph if is_inp==1, color(green%50) width(.01)), ///
    legend(order(1 "Ambulatory" 2 "Emergency" 3 "Inpatient"))
	   
*PvCO2
table _granular, statistic(mean vbg_co2)
oneway vbg_co2 encounter_ty
twoway ///
    (histogram vbg_co2 if is_amb==1 & is_cc!=1, color(blue%50) width(2)) ///
    (histogram vbg_co2 if is_emer==1, color(red%50) width(2)) ///
    (histogram vbg_co2 if is_inp==1, color(green%50) width(2)), ///
    legend(order(1 "Ambulatory" 2 "Emergency" 3 "Inpatient"))

*Inpatient; Critical Care vs not critical care
*Art pH
table is_cc if is_inp == 1 & is_amb==0 & is_emerg==0, statistic(mean abg_ph)
ttest abg_ph if is_inp == 1 & is_amb==0 & is_emerg==0, by(is_cc)
twoway ///
    (histogram abg_ph if is_inp==1 & is_cc !=1 & is_amb==0 & is_emerg==0, color(green%50) width(.01)) ///
    (histogram abg_ph if is_inp==1 & is_cc ==1 & is_amb==0 & is_emerg==0, color(red%50) width(.01)), ///
    legend(order(1 "ICU" 2 "Inpatient Non-ICU"))
*Ven pH
table is_cc if is_inp == 1 & is_amb==0 & is_emerg==0, statistic(mean vbg_ph)
ttest vbg_ph if is_inp == 1 & is_amb==0 & is_emerg==0, by(is_cc)
twoway ///
    (histogram vbg_ph if is_inp==1 & is_cc !=1 & is_amb==0 & is_emerg==0, color(green%50) width(.01)) ///
    (histogram vbg_ph if is_inp==1 & is_cc ==1 & is_amb==0 & is_emerg==0, color(red%50) width(.01)), ///
    legend(order(1 "ICU" 2 "Inpatient Non-ICU"))
*PaCO2
table is_cc if is_inp == 1 & is_amb==0 & is_emerg==0, statistic(mean paco2)
ttest paco2 if is_inp == 1 & is_amb==0 & is_emerg==0, by(is_cc)
twoway ///
    (histogram paco2 if is_inp==1 & is_cc !=1 & is_amb==0 & is_emerg==0, color(green%50) width(2)) ///
    (histogram paco2 if is_inp==1 & is_cc ==1 & is_amb==0 & is_emerg==0, color(red%50) width(2)), ///
    legend(order(1 "ICU" 2 "Inpatient Non-ICU"))
*PvCO2:
table is_cc if is_inp == 1 & is_amb==0 & is_emerg==0, statistic(mean vbg_co2)
ttest vbg_co2 if is_inp == 1 & is_amb==0 & is_emerg==0, by(is_cc)
twoway ///
    (histogram vbg_co2 if is_inp==1 & is_cc !=1 & is_amb==0 & is_emerg==0, color(green%50) width(2)) ///
    (histogram vbg_co2 if is_inp==1 & is_cc ==1 & is_amb==0 & is_emerg==0, color(red%50) width(2)), ///
    legend(order(1 "ICU" 2 "Inpatient Non-ICU"))

	
*======================================
*===2) Error in different settings

*Define setting‑specific error distributions for TcCO₂ − PaCO₂ using meta‑analytic parameters (bias, σ, τ²), then simulate TcCO₂ from observed PaCO₂.


*Abbreviations used
*	•	PaCO₂ = arterial partial pressure of carbon dioxide
*	•	TcCO₂ = transcutaneous carbon dioxide
*	•	LoA = limits of agreement. LoA is computed as δ ± 2√(σ² + τ²)
*	•	σ aka corr_sd = within‑study SD of (PaCO₂ − TcCO₂) differences
*	•	τ² = between‑study variance in bias across studies. tau = between-study SD (not squared)
*	•	Bias (δ) = PaCO₂ − TcCO₂ (this is the sign convention used in the Conway meta‑analysis).

*======================================

/* WISH LIST *

Parameter uncertainty
	•	Conway's estimates (δ, σ, τ²) are treated as fixed. Inference about downstream operating characteristics should propagate uncertainty (e.g., draw δ, σ, τ² from their sampling distributions). The paper provides outer CIs for the pooled LoA; you can use those to calibrate a simple parametric bootstrap around SD_total, or pull SEs directly from the provided code/data if you integrate the R workflow... one practical approach is to back out an approximate SE for SD_total from the reported outer 95% CI for LoA and use that in a bootstrap; or consume the authors' R data and robust-variance code to sample directly.￼  ￼
	
*/ 



/*==Overall Error in PaCO2 and TcCO2 in all settings
Bias = 0.0852 //mean bias from raw data
Tau = 2.97 //variance in the between-study estimates of mean bias
SD 1.9 // variance in the intra-study agreement between measurements 
LOAL = -7.1
LOAU = 6.9*/
set seed 12345
local mean_bias_all = 0.0852 //mean bias from raw data
local tau_all = sqrt(8.85) //variance in the between-study estimates of mean bias
local corr_sd_all = 1.9 // variance in the intra-study agreement between measurements
gen error_all = rnormal(0, sqrt(`corr_sd_all'^2 + `tau_all'^2)) - `mean_bias_all' 
summ error_all, detail
centile error_all, centile(2.5 97.5)
di as txt "TcCO2−PaCO2 empirical LoA: " %5.2f r(c_1) " to " %5.2f r(c_2)

/*==Error between PaCO2 and TcCO2 in Ambulatory (PFTs)
Bias = -0.1 
SD= 1.6
Tau^2 = 1.4
LOAL = -4.0
LOAU = 3.9*/
set seed 12345
local mean_bias_amb = -0.1 //mean bias from raw data
local tau_amb = sqrt(1.4) //variance in the between-study estimates of mean bias
local corr_sd_amb 1.6 // variance in the intra-study agreement between measurements
gen error_amb = (rnormal(0, sqrt(`corr_sd_amb'^2 + `tau_amb'^2)) - `mean_bias_amb') if is_amb==1 & is_cc!=1
summ error_amb, detail //check
centile error_amb, centile(2.5 97.5)
di as txt "TcCO2−PaCO2 empirical LoA: " %5.2f r(c_1) " to " %5.2f r(c_2)

/*==Error between PaCO2 and TcCO2 in Emergency + Inpatient (Acute Resp Failure)
Bias = 1.7 
SD= 2.0
Tau^2 = 3.2
LOAL = -3.7
LOAU = 7.1*/
set seed 12345
local mean_bias_inp_emerg = 1.7 //mean bias from raw data
local tau_inp_emerg = sqrt(3.2) //variance in the between-study estimates of mean bias
local corr_sd_inp_emerg 2.0 // variance in the intra-study agreement between measurements
gen error_inp_emerg = (rnormal(0, sqrt(`corr_sd_inp_emerg'^2 + `tau_inp_emerg'^2)) - `mean_bias_inp_emerg') if ((is_cc~=1) & (is_inp==1 | is_emerg==1))
summ error_inp_emerg, detail //check
centile error_inp_emerg, centile(2.5 97.5)
di as txt "TcCO2−PaCO2 empirical LoA: " %5.2f r(c_1) " to " %5.2f r(c_2)

/*==Error between PaCO2 and TcCO2 in ICU
Bias = -0.6 
SD= 2.0
Tau^2 = 1.9
LOAL = -5.4
LOAU = 4.2*/
set seed 12345
local mean_bias_cc = -0.6 //mean bias from raw data
local tau_cc = sqrt(1.9) //variance in the between-study estimates of mean bias
local corr_sd_cc 2.0 // variance in the intra-study agreement between measurements
gen error_cc = (rnormal(0, sqrt(`corr_sd_cc'^2 + `tau_cc'^2)) - `mean_bias_cc') if (is_cc==1 & is_emerg!=1 & is_amb!=1)
summ error_cc, detail //check
centile error_cc, centile(2.5 97.5)
di as txt "TcCO2−PaCO2 empirical LoA: " %5.2f r(c_1) " to " %5.2f r(c_2)

*======================================
*==3) Simulate a TcCO2 for each setting

* Construct tcco2_* from PaCO₂ plus simulated error draws.

*======================================

*All
gen tcco2_all = paco2 + error_all if !missing(paco2)
*Inpatient+Emergency:
gen tcco2_inp_emerg = paco2 + error_inp_emerg if  & ((is_inp==1 & is_cc!=1) | is_emerg ==1))
*ICU: 
gen tcco2_cc = paco2 + error_cc if (is_cc==1 & is_emerg!=1 & is_amb!=1 & !missing(paco2)) 
**Note that there are ambulatory encounters w/ cc_time, these were considered amb but not ICU**
*Outpatient: 
gen tcco2_amb = paco2 + error_amb if (is_amb==1 & !missing(paco2) & is_cc!=1) 

**Note steps 2 and 3 were done with alternate code than is shown here to account for all uncertainty. See do file "Parameter uncertainty simulation do file" for those code.

*======================================
*==4) Generate a variable for hypercapnic v not for both PaCO2 and TcCO2 for each setting
* Create binary indicators for PaCO₂ ≥45 mmHg and TcCO₂ ≥45 mmHg (overall and by setting).
*======================================

*All Settings
gen paco2_all_hypercap = .
label variable paco2_all_hypercap "PaCO2 ≥ 45 mmHg within 24h of admission for all settings"
replace paco2_all_hypercap = 1 if !missing(paco2) & paco2 >= 45
replace paco2_all_hypercap = 0 if !missing(paco2) & paco2 < 45
label define paco2_all_hypercap_lab2 0 "PaCO2 < 45 mmHg" 1 "PaCO2 ≥ 45 mmHg" 
label values paco2_all_hypercap paco2_all_hypercap_lab2

gen tcco2_all_hypercap = .
label variable tcco2_all_hypercap "TcCO2 ≥ 45 mmHg within 24h of admission?"
replace tcco2_all_hypercap =  1 if !missing(tcco2_all) & tcco2_all >= 45
replace tcco2_all_hypercap =  0 if !missing(tcco2_all) & tcco2_all < 45
label define tcco2_all_hypercap_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_all_hypercap tcco2_all_hypercap_lab2

*TcCO2 for all settings from Brian's uncertainty
gen tcco2_all_hypercap_2 = .
label variable tcco2_all_hypercap_2 "TcCO2 ≥ 45 mmHg within 24h of admission?"
replace tcco2_all_hypercap_2 =  1 if !missing(tcco2_sim) & tcco2_sim >= 45
replace tcco2_all_hypercap_2 =  0 if !missing(tcco2_sim) & tcco2_sim < 45
label define tcco2_all_hypercap_2_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_all_hypercap tcco2_all_hypercap_2_lab2

*Inpatient+Emergency
gen paco2_inp_emerg_hypercap = .
label variable paco2_inp_emerg_hypercap "PaCO2 ≥ 45 mmHg within 24h of admission for Inpatient + Emergency"
replace paco2_inp_emerg_hypercap = 1 if !missing(paco2) & paco2 >= 45 & ((is_inp==1 & is_cc==0) | is_emerg==1)
replace paco2_inp_emerg_hypercap = 0 if !missing(paco2) & paco2 < 45 & ((is_inp==1 & is_cc==0) | is_emerg==1)
label define paco2_inp_emerg_hypercap_lab2 0 "PaCO2 < 45 mmHg" 1 "PaCO2 ≥ 45 mmHg" 
label values paco2_inp_emerg_hypercap paco2_inp_emerg_hypercap_lab2

gen tcco2_inp_emerg_hypercap = .
label variable tcco2_inp_emerg_hypercap "TcCO2 ≥ 45 mmHg within 24h of admission for Inpatient + Emergency"
replace tcco2_inp_emerg_hypercap =  1 if !missing(tcco2_inp_emerg) & tcco2_inp_emerg >= 45
replace tcco2_inp_emerg_hypercap =  0 if !missing(tcco2_inp_emerg) & tcco2_inp_emerg < 45
label define tcco2_inp_emerg_hypercap_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_inp_emerg_hypercap tcco2_inp_emerg_hypercap_lab2

*TcCO2 for Inpatient+Emergency from Brian's uncertainty
gen tcco2_inp_emerg_hypercap_2 = .
label variable tcco2_inp_emerg_hypercap_2 "TcCO2 ≥ 45 mmHg within 24h of admission for Inpatient + Emergency"
replace tcco2_inp_emerg_hypercap_2 =  1 if !missing(tcco2_sim_inp_emerg) & tcco2_sim_inp_emerg >= 45
replace tcco2_inp_emerg_hypercap_2 =  0 if !missing(tcco2_sim_inp_emerg) & tcco2_sim_inp_emerg < 45
label define tcco2_inp_emerg_hypercap_2_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_inp_emerg_hypercap_2 tcco2_inp_emerg_hypercap_2_lab2

*ICU
gen paco2_cc_hypercap = .
label variable paco2_cc_hypercap "PaCO2 ≥ 45 mmHg within 24h of admission for ICU"
replace paco2_cc_hypercap = 1 if !missing(paco2) & paco2 >= 45 & is_cc==1 & is_emerg!=1 & is_amb!=1
replace paco2_cc_hypercap = 0 if !missing(paco2) & paco2 < 45 & is_cc==1 & is_emerg!=1 & is_amb!=1
label define paco2_cc_hypercap_lab2 0 "PaCO2 < 45 mmHg" 1 "PaCO2 ≥ 45 mmHg" 
label values paco2_cc_hypercap paco2_cc_hypercap_lab2

gen tcco2_cc_hypercap = .
label variable tcco2_cc_hypercap "TcCO2 ≥ 45 mmHg within 24h of admission?"
replace tcco2_cc_hypercap =  1 if !missing(tcco2_cc) & tcco2_cc >= 45
replace tcco2_cc_hypercap =  0 if !missing(tcco2_cc) & tcco2_cc < 45
label define tcco2_cc_hypercap_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_cc_hypercap tcco2_cc_hypercap_lab2

*TcCO2 for ICU from Brian's uncertainty
gen tcco2_cc_hypercap_2 = .
label variable tcco2_cc_hypercap_2 "TcCO2 ≥ 45 mmHg within 24h of admission?"
replace tcco2_cc_hypercap_2 =  1 if !missing(tcco2_sim_cc) & tcco2_sim_cc >= 45
replace tcco2_cc_hypercap_2 =  0 if !missing(tcco2_sim_cc) & tcco2_sim_cc < 45
label define tcco2_cc_hypercap_2_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_cc_hypercap_2 tcco2_cc_hypercap_2_lab2

*Outpatient
gen paco2_amb_hypercap = .
label variable paco2_amb_hypercap "PaCO2 ≥ 45 mmHg in ambulatory setting"
replace paco2_amb_hypercap = 1 if !missing(paco2) & paco2 >= 45 & is_amb==1 & is_cc!=1
replace paco2_amb_hypercap = 0 if !missing(paco2) & paco2 < 45 & is_amb==1 & is_cc!=1
label define paco2_amb_hypercap_lab2 0 "PaCO2 < 45 mmHg" 1 "PaCO2 ≥ 45 mmHg" 
label values paco2_amb_hypercap paco2_amb_hypercap_lab2

gen tcco2_amb_hypercap = .
label variable tcco2_amb_hypercap "TcCO2 ≥ 45 mmHg in ambulatory setting"
replace tcco2_amb_hypercap =  1 if !missing(tcco2_amb) & tcco2_amb >= 45
replace tcco2_amb_hypercap =  0 if !missing(tcco2_amb) & tcco2_amb < 45
label define tcco2_amb_hypercap_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_amb_hypercap tcco2_amb_hypercap_lab2

*TcCO2 for Amb from Brian's uncertainty
gen tcco2_amb_hypercap_2 = .
label variable tcco2_amb_hypercap_2 "TcCO2 ≥ 45 mmHg in ambulatory setting"
replace tcco2_amb_hypercap_2 =  1 if !missing(tcco2_sim_amb) & tcco2_sim_amb >= 45
replace tcco2_amb_hypercap_2 =  0 if !missing(tcco2_sim_amb) & tcco2_sim_amb < 45
label define tcco2_amb_hypercap_2_lab2 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_amb_hypercap_2 tcco2_amb_hypercap_2_lab2

*======================================
*==5) Compare PaCO2 v TcCO2
*Build confusion matrices and compute sensitivity/specificity with diagt.
*======================================

*All
gen confusion_matrix_all = . //Given simulated readings, was an error made? 
label variable confusion_matrix_all "TcCO2 Confusion Matrix for all settings"
replace confusion_matrix_all = 1 if paco2_all_hypercap == 0  & tcco2_all_hypercap == 0 //TN
replace confusion_matrix_all = 2 if paco2_all_hypercap == 0 & tcco2_all_hypercap == 1 //FP
replace confusion_matrix_all = 3 if paco2_all_hypercap == 1 & tcco2_all_hypercap == 1 //TP
replace confusion_matrix_all = 4 if paco2_all_hypercap == 1  & tcco2_all_hypercap == 0 //FN
label define confusion_matrix_all_lab 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_all confusion_matrix_all_lab
diagt paco2_all_hypercap tcco2_all_hypercap //calculate operating characteristics
tab confusion_matrix_all

*All v2
drop confusion_matrix_all_2
label drop confusion_matrix_all_lab_2
gen confusion_matrix_all_2 = . //Given simulated readings, was an error made? 
label variable confusion_matrix_all_2 "TcCO2 Confusion Matrix for all settings"
replace confusion_matrix_all_2 = 1 if paco2_all_hypercap == 0  & tcco2_all_hypercap_2 == 0 //TN
replace confusion_matrix_all_2 = 2 if paco2_all_hypercap == 1 & tcco2_all_hypercap_2 == 1 //TP
replace confusion_matrix_all_2 = 3 if paco2_all_hypercap == 1  & tcco2_all_hypercap_2 == 0 //FN
replace confusion_matrix_all_2 = 4 if paco2_all_hypercap == 0 & tcco2_all_hypercap_2 == 1 //FP
label define confusion_matrix_all_lab_2 1 "True Negative" 2 "True Positive" 3 "False Negative" 4 "False Positive"
label values confusion_matrix_all_2 confusion_matrix_all_lab_2
diagt paco2_all_hypercap tcco2_all_hypercap_2 //calculate operating characteristics
tab confusion_matrix_all_2

*Ambulatory
gen confusion_matrix_amb = . //Given simulated readings, was an error made? 
label variable confusion_matrix_amb "TcCO2 Confusion Matrix for ambulatory setting"
replace confusion_matrix_amb = 1 if paco2_amb_hypercap == 0  & tcco2_amb_hypercap == 0 //TN
replace confusion_matrix_amb = 2 if paco2_amb_hypercap == 0 & tcco2_amb_hypercap == 1 //FP
replace confusion_matrix_amb = 3 if paco2_amb_hypercap == 1 & tcco2_amb_hypercap == 1 //TP
replace confusion_matrix_amb = 4 if paco2_amb_hypercap == 1  & tcco2_amb_hypercap == 0 //FN
label define confusion_matrix_amb_lab 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_amb confusion_matrix_amb_lab
diagt paco2_amb_hypercap tcco2_amb_hypercap //calculate operating characteristics
tab confusion_matrix_amb

*Ambulatory V2
gen confusion_matrix_amb_2 = . //Given simulated readings, was an error made? 
label variable confusion_matrix_amb "TcCO2 Confusion Matrix for ambulatory setting"
replace confusion_matrix_amb_2 = 1 if paco2_amb_hypercap == 0  & tcco2_amb_hypercap_2 == 0 //TN
replace confusion_matrix_amb_2 = 2 if paco2_amb_hypercap == 1 & tcco2_amb_hypercap_2 == 1 //TP
replace confusion_matrix_amb_2 = 3 if paco2_amb_hypercap == 1  & tcco2_amb_hypercap_2 == 0 //FN
replace confusion_matrix_amb_2 = 4 if paco2_amb_hypercap == 0 & tcco2_amb_hypercap_2 == 1 //FP
label define confusion_matrix_amb_lab_2 1 "True Negative" 2 "True Positive" 3 "False Negative" 4 "False Positive"
label values confusion_matrix_amb_2 confusion_matrix_amb_lab_2
diagt paco2_amb_hypercap tcco2_amb_hypercap_2 //calculate operating characteristics
tab confusion_matrix_amb_2

*Inpatient and Emergency
gen confusion_matrix_inp_emerg = . //Given simulated readings, was an error made? 
label variable confusion_matrix_inp_emerg "TcCO2 Confusion Matrix for Inpatient/Emergency Setting"
replace confusion_matrix_inp_emerg = 1 if paco2_inp_emerg_hypercap == 0  & tcco2_inp_emerg_hypercap == 0 //TN
replace confusion_matrix_inp_emerg = 2 if paco2_inp_emerg_hypercap == 0 & tcco2_inp_emerg_hypercap == 1 //FP
replace confusion_matrix_inp_emerg = 3 if paco2_inp_emerg_hypercap == 1 & tcco2_inp_emerg_hypercap == 1 //TP
replace confusion_matrix_inp_emerg = 4 if paco2_inp_emerg_hypercap == 1  & tcco2_inp_emerg_hypercap == 0 //FN
label define confusion_matrix_inp_emerg_lab 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_inp_emerg confusion_matrix_inp_emerg_lab
diagt paco2_inp_emerg_hypercap tcco2_inp_emerg_hypercap //calculate operating characteristics
tab confusion_matrix_inp_emerg

*Inpatient and Emergency V2
gen confusion_matrix_inp_emerg_2 = . //Given simulated readings, was an error made? 
label variable confusion_matrix_inp_emerg_2 "TcCO2 Confusion Matrix for Inpatient/Emergency Setting"
replace confusion_matrix_inp_emerg_2 = 1 if paco2_inp_emerg_hypercap == 0  & tcco2_inp_emerg_hypercap_2 == 0 //TN
replace confusion_matrix_inp_emerg_2 = 2 if paco2_inp_emerg_hypercap == 0 & tcco2_inp_emerg_hypercap_2 == 1 //FP
replace confusion_matrix_inp_emerg_2 = 3 if paco2_inp_emerg_hypercap == 1 & tcco2_inp_emerg_hypercap_2 == 1 //TP
replace confusion_matrix_inp_emerg_2 = 4 if paco2_inp_emerg_hypercap == 1  & tcco2_inp_emerg_hypercap_2 == 0 //FN
label define confusion_matrix_inp_emerg_lab_2 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_inp_emerg_2 confusion_matrix_inp_emerg_lab_2
diagt paco2_inp_emerg_hypercap tcco2_inp_emerg_hypercap_2 //calculate operating characteristics
tab confusion_matrix_inp_emerg_2

*ICU
gen confusion_matrix_cc = . //Given simulated readings, was an error made? 
label variable confusion_matrix_cc "TcCO2 Confusion Matrix for ICU Setting"
replace confusion_matrix_cc = 1 if paco2_cc_hypercap == 0  & tcco2_cc_hypercap == 0 //TN
replace confusion_matrix_cc = 2 if paco2_cc_hypercap == 0 & tcco2_cc_hypercap == 1 //FP
replace confusion_matrix_cc = 3 if paco2_cc_hypercap == 1 & tcco2_cc_hypercap == 1 //TP
replace confusion_matrix_cc = 4 if paco2_cc_hypercap == 1  & tcco2_cc_hypercap == 0 //FN
label define confusion_matrix_cc_lab 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_cc confusion_matrix_cc_lab
diagt paco2_cc_hypercap tcco2_cc_hypercap //calculate operating characteristics
tab confusion_matrix_cc

*ICU V2
gen confusion_matrix_cc_2 = . //Given simulated readings, was an error made? 
label variable confusion_matrix_cc_2 "TcCO2 Confusion Matrix for ICU Setting"
replace confusion_matrix_cc_2 = 1 if paco2_cc_hypercap == 0  & tcco2_cc_hypercap_2 == 0 //TN
replace confusion_matrix_cc_2 = 2 if paco2_cc_hypercap == 0 & tcco2_cc_hypercap_2 == 1 //FP
replace confusion_matrix_cc_2 = 3 if paco2_cc_hypercap == 1 & tcco2_cc_hypercap_2 == 1 //TP
replace confusion_matrix_cc_2 = 4 if paco2_cc_hypercap == 1  & tcco2_cc_hypercap_2 == 0 //FN
label define confusion_matrix_cc_lab_2 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_cc_2 confusion_matrix_cc_lab_2
diagt paco2_cc_hypercap tcco2_cc_hypercap_2 //calculate operating characteristics
tab confusion_matrix_cc_2
 
 
*======================================
*==MC) Montecarlo simulation - All as example. Problems are 1) only getting summary SD for all repetitions. 2) Dataset would be too massive to run sims on all PaCO2s. This example is truncated to only 100 repetitions for first 20 PaCO2s
* some IPW sampling and selection-weight code (commented).
*======================================

program define simulate_tcco2, rclass
    version 16.0
    preserve
        // Restrict to the first 20 observations of the current dataset
        keep in 1/20
        // Generate meta-analysis parameters
        gen bias_mc      = - 0.0852
        gen sd_mc        = 3.51
        gen tau_mc       = 2.97
        gen LOA_upper_mc = 6.9
        gen LOA_lower_mc = -7.1
        // Generate simulated TcCO2 values
        gen tcco2_mc = paco2 - bias_mc + rnormal(0, sd_mc) //note, there was a sign error here I fixed.
        // Summarize simulated values and return mean and standard deviation
        summarize tcco2_mc, meanonly
        return scalar mean_tcco2_mc = r(mean)
        return scalar sd_tcco2_mc   = r(sd)
  
end
simulate mean_tcco2_mc = r(mean_tcco2_mc) sd_tcco2_mc = r(sd_tcco2_mc), reps(`sims') nodots: simulate_tcco2

restore

/*======================================
*==IPW) Propensity weighting - Not really working yet. Treating the binary 'outcome' as hypercap vs not then seeing distribution w/ holding constant bmi, ph, sedatives. Unclear if this is correct approach.
*======================================


/// Estimate propensity score
logit paco2_all_hypercap bmi sedatives phtn
predict ps, pr 
/// Generate inverse propensity weights
gen ipw = paco2_all_hypercap/ps + (1 - paco2_all_hypercap)/(1 - ps)

///Now generate a pseudopopulation of 1000 PaCO2 where these covariables are held constant
egen total_ipw = total(ipw)
gen norm_weight = ipw / total_ipw
// Sort by paco2 (or any order—you just need a consistent order)
sort paco2
// Compute the cumulative probability for each observation
gen cum_weight = sum(norm_weight)
* Create a dummy variable (same value for all observations)
gen dummy = 1
* Save the original data with cumulative weights and the dummy variable
tempfile orig
save `orig', replace
* Now, create your simulated draws dataset (Only did 10 bc 1000 crashed my computer!)
clear
set obs 10
gen u = runiform()
* Also create the dummy variable here:
gen dummy = 1
* Use joinby to combine each simulated draw with all original observations by the dummy variable:
joinby dummy using `orig'
* Now, keep only those rows where the simulated random number u is less than or equal to cum_weight
keep if u <= cum_weight
* For each simulated draw (identified by u), keep the first matching original observation
bysort u (cum_weight): keep if _n == 1
* At this point, you have 10 observations drawn from your original weighted distribution.
* The variable paco2 represents the drawn PaCO2 values.
list paco2 in 1/10

*-----
*IPW New attempt, modeling the probability of ABG having been sampled
*-----
gen abg_sampled = paco2 !=.
replace abg_s =1 if paco2 !=.
replace abg_s =0 if paco2 ==.

// 1) Model ABG sampling
logit abg_sam osa asthma copd chf oud sedatives phtn has_j9611 has_j9610 has_j961 has_j9612 
predict ps_abg, pr
// 2) Generate IPW
gen abg_ipw = 1/ps_abg if abg_sampled==1
// 3) Stabilized weights to reduce variance (include?)
sum abg_sampled
local p_abg = r(mean)   // overall probability of ABG sampling
gen abg_sw = .
replace abg_sw = (`p_abg'/ps_abg) if abg_sampled==1
// 4: Outcome model among sampled patients
logit paco2_all_hypercap tcco2_all_hypercap [pweight=abg_sw], vce(robust)
*/

*======================================
*==6) Graphs
*Stacked bar (catplot) of misclassification vs PaCO₂.
*======================================
preserve
recode paco2 (min/30=.) (60/max=.)
replace paco2 = round(paco2, 1)
catplot, over(confusion_matrix_all_2) percent over(paco2, gap(0.01)) percent stack ///
	recast(bar) /// 
	asyvars ///
	bar(1, fcolor(teal%25) lcolor(teal)) ///
	bar(2, fcolor(purple%25) lcolor(purple)) ///
	bar(3, fcolor(cranberry%25) lcolor(cranberry)) ///
	bar(4, fcolor(orange_red%25) lcolor(orange_red)) ///
	ylabel(, labsize(small)) ///
	legend(pos(6) rows(1) size(large)) ///
	b1title("Arterial Partial Pressure of Carbon Dioxide", size(huge)) ///
	ytitle("TcCO{sub:2} Categorization Proportion", size(large)) ///
	xsize(5) ysize(2.5)
restore
