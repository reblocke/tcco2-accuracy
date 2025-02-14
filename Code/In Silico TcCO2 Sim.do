* Hypercapnia TriNetX 
* In Silico Simulation of Se and Sp of TcCO2 
* Updated 2025-2-13 BWL

/* ---------

Databse Documentation / Pre-processing: 

This contains all encounters in 2022, with at one of the 76 healthcare organizations contributing to data, with at least 1 clinical feature that should
cause a clinician to consider if hypercapnia is present: predisposing condition 
(including obesity), blood gas obtained, diagnosis code for respiratory failure
or NIV or IMV procedure code. 


BWL - I have a data-base that includes more elements from the chart if we need them


The inverse propensity weight scores are generated with logistic regression 
and a bunch of predictors (intentionally overfit). May be useful as a supplementary
analysis)

Definitions of the icd-10-cm codes that count as each: 
Any prior: 
g47.3 osa
j45 asthma
j43* or j44* copd
i50* chf
i63* stroke
n18* ckd
m05*, m06*, m30*, m31*, m32*, m33*, m34*, m35*, m36* ctd
f01-f09* dementia
e08-e13* dm
i70* PVD
f11* opiate use disorder
f13* sedative use disorder
e84* cf 
i27* pulm htn
D75.1* polycythemia
g12*, g14*, g70*, g35*, g71*, g95*, g36*, 37* nmd
f17*, f12*, f18* nicotine dependence


Notes: 
- ASSUMES normality of errors - major limitation

The pooled standard deviation is the geometric mean of the within-study standard deviation and the variance in the bias between studies (tau) (essentially, two different components of a prediction error)

SD of agreement when restricting to studies in other contexts: 

Low risk studies only: bias: +0.17, sd = 2.67, tau2 = 3.82
Senntec only: bias +0.119, sd =2.68, tau2= 3.53
Adult ICU only ( would need to approx with crit care only = not perfect): bias -0.596, sd =2.42, tau2 = 1.89
Acute respiratory failure only: bias = +1.69, sd=2.69, tau =3.16
Chronic respiratory failure only: bias =-0.092, sd=2.36,tau=2.44

(these are all found in the code provided by the conway meta-analysis)

source of data: https://figshare.com/articles/dataset/Accuracy_of_TcCO2_monitoring_meta-analysis/6244058/2

OVERALL TODO LIST: 

Trouble shoot IPW score -> moves distribution up = low PaCO2 over sampled? 
[ ] perhaps redo with lightGBM or XGBoost ML.  

-----------*/


capture log close // close any existing log

* Data processing
clear

cd "/Users/blocke/Box Sync/Residency Personal Files/Scholarly Work/Locke Research Projects/tcco2-accuracy" //change to your folder


/* This block just creates output folders for figures and log files, and makes it 
so that the do file is copied to a log each time it is run */ 

capture mkdir "Results and Figures"
capture mkdir "Results and Figures/$S_DATE/" //make new folder for figure output if needed
capture mkdir "Results and Figures/$S_DATE/Logs/" //new folder for stata logs
local a1=substr(c(current_time),1,2)
local a2=substr(c(current_time),4,2)
local a3=substr(c(current_time),7,2)
local b = "In Silico TcCO2 Sim.do" // do file name
copy "Code/`b'" "Results and Figures/$S_DATE/Logs/(`a1'_`a2'_`a3')`b'"
log using temp.log, replace
set scheme cleanplots
graph set window fontface "Times New Roman" //match publication font

clear
cd "data"
use in_silico_tcco2_db
cd ".."


label variable ipw "Inverse Propensity Weight"
label variable approx_ipw_fweight "Inverse Propensity Weight converted to frequency-weight"

/* MAIN Simulation */ 

//Overall approach to this code: simulate a *single* hypothetical TcCO2 reading for each observation - are there better ways to do this? So many observations that sampling uncertainty is not a major driver of undercertainty... so p-values etc. not very helpful. 

//Overall TcCO2 agreement

local mean_bias = 0.0852 //mean bias from raw data
local tau = sqrt(8.85) //variance in the between-study estimates of mean bias
local corr_sd 3.51 // variance in the intra-study agreement between measurements
gen error = rnormal(0, sqrt(`corr_sd'^2 + `tau'^2)) - `mean_bias' 
label variable error "Simulated Error using super-population from which studies taken from"
summ error, detail //check

keep if first_encounter == 1
keep if !missing(paco2) // only patients who had an ABG
//keep if !missing(is_inp)
//keep if is_emer == 1
//keep if is_amb == 1

hist paco2 
summ paco2, detail //All ABG results in the cohort

//Simulated TcCO2 reading 
gen tcco2_reading = paco2 + error
label variable tcco2_reading "TcCO2 Reading"

gen paco2_int = floor(paco2) //for visualization
label variable paco2_int "PaCO2 (integer)" 

* Generate histogram to visualize the distributions
histogram error, normal title("Error Sampling Distribution")
histogram tcco2_reading, normal title("Distribution of TcCO2 readings")

gen paco2_flag = .
label variable paco2_flag "PaCO2 >= 45 mmHg 1st day of enounter?"
replace paco2_flag = 1 if !missing(paco2) & paco2 >= 45
replace paco2_flag = 0 if !missing(paco2) & paco2 < 45
label define paco2_flag_lab2 0 "PaCO2 < 45 mmHg" 1 "PaCO2 >= 45 mmHg" 
label values paco2_flag paco2_flag_lab2

gen tcco2_hypercap_flag = .
label variable tcco2_hypercap_flag "TcCO2 >= 45 mmHg 1st day of encounter?"
replace tcco2_hypercap_flag =  1 if !missing(tcco2_reading) & tcco2_reading >= 45
replace tcco2_hypercap_flag =  0 if !missing(tcco2_reading) & tcco2_reading < 45
label define tcco2_hypercap_flag_lab 0 "TcCO2 < 45 mmHg" 1 "TcCO2 >= 45 mmHg" 
label values tcco2_hypercap_flag tcco2_hypercap_flag_lab

gen confusion_matrix = . //Given simulated readings, was an error made? 
label variable confusion_matrix "TcCO2 Confusion Matrix"
replace confusion_matrix = 1 if paco2_flag == 0  & tcco2_hypercap_flag == 0 //TN
replace confusion_matrix = 2 if paco2_flag == 0 & tcco2_hypercap_flag == 1 //FP
replace confusion_matrix = 3 if paco2_flag == 1 & tcco2_hypercap_flag == 1 //TP
replace confusion_matrix = 4 if paco2_flag == 1  & tcco2_hypercap_flag == 0 //FN
label define confusion_matrix_lab 1 "True Negative" 2 "False Posistive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix confusion_matrix_lab

* Summarize the generated variables to check the results
summarize error paco2 tcco2_reading, detail

tab confusion_matrix
diagt paco2_flag tcco2_hypercap_flag //calculate operating characteristics
//Not sure that the error bars and confidence intervals represent anything real here?


preserve
recode paco2 (min/20=.) (70/max=.) //truncate the range 
twoway kdensity paco2, recast(area) fcolor(ebg%25) lcolor(navy) lpattern(solid) lwidth(*2) bwidth(2) || , ///
	legend(off) ///
	xline(45, lwidth(medthick) lpattern(longdash)) ///
	xline(37.3, lwidth(medium) lpattern(dash_dot)) ///
	xline(52.5, lwidth(medium) lpattern(dash_dot)) ///
	text(0.04 60 "Hypercapnia threshold and", size(medlarge) color(gs8)) ///
	text(0.035 60 "95% Agreement Range", size(medlarge) color(gs8)) ///
	xlabel(20(10)70, labsize(large)) ///
	ylabel(, labsize(large)) ///
	xtitle("Arterial Partial Pressure of Carbon Dioxide (mmHg)", size(large)) ///
	ytitle("Relative Frequency", size(large)) ///
	scheme(white_tableau) ///
	xsize(9) ysize(3.5)
graph save total_distributions.gph, replace
restore

preserve
recode paco2_int (min/20=.) (70/max=.)  //Truncate to area of interest

catplot confusion_matrix, ///
	over(paco2_int) ///
	recast(bar) /// 
	asyvars ///
	stack ///
	percent(paco2_int) ///
	bar(1, fcolor(teal%25) lcolor(teal)) ///
	bar(2, fcolor(purple%25) lcolor(purple)) ///
	bar(3, fcolor(cranberry%25) lcolor(cranberry)) ///
	bar(4, fcolor(orange_red%25) lcolor(orange_red)) ///
	ylabel(, labsize(medlarge)) ///
	legend(pos(6) rows(1) size(large)) ///
	b1title("Arterial Partial Pressure of Carbon Dioxide", size(large)) ///
	ytitle("TcCO{sub:2} Categorization Proportion", size(large)) ///
	xsize(9) ysize(2.5)
	graph save blowout_confusion.gph, replace
restore

graph combine total_distributions.gph blowout_confusion.gph, ///
	cols(1) /// 
	xsize(9) ysize(6)
graph export "Results and Figures/$S_DATE/Distributions and Confusion.svg", name("Graph") replace




/* 

OTHER ANALYSES NOT CURRENTLY USED

*/ 

//Overlaps & ROC CURVES
twoway (hist paco2 if tcco2_hypercap_flag == 0, frac fcolor(%25) lcolor(ebblue) width(1)) (hist paco2 if tcco2_hypercap_flag == 1, frac fcolor(%25) lcolor(cranberry) width(1)), xtitle("PaCO2") xlabel(5(5)150) legend(pos(2) ring(0) label(1 "TcCO2 < 45 mmHg") label(2 "TcCO2 >= 45 mmHg"))

roctab paco2_flag tcco2_reading, graph plotopts(lwidth(thick) ylabel(,labsize(4)) xlabel(,labsize(4)) xtitle(, size(5)) ytitle(, size(5) height(5)) title("ROC for Hypercapnia by TcCO2 reading", size(5)) ) scheme(white_w3d) // this takes a very long time
//graph export "Results and Figures/$S_DATE/AMB ROC HCO3 Labels.png", as(png) name("Graph") replace
