*---------------------------------------------------------------*
* This starts with the "Pruned" but otherwise raw Conway dataset ("Conway_Tcco2_pruned_dataset") for just our populations created in another do-file (loaded into 'default frame', runs the meta-analysis on it then saves the results as the "Conway_Tcco2_working_dataset". 
*Then these values are used to create simulated TcCO2 for each individual patient's PaCO2 in the 'TriNetX_Working_Dataset' file (created by cleaning the raw TriNetX dataset in another do-file) and saves it as the new "Revised Working dataset" file.
*After these simulated values are created, I then go on to analyze the agreement of TcCO2 & PaCO2 using the separate "simulation_analysis_do" do-file. 
*---------------------------------------------------------------*

clear all
use "Conway_Tcco2_pruned_dataset", clear
save "Conway_Tcco2_working_dataset", replace

/***************************************************************************
  1. Define your meta-analysis and LOA maker programs
***************************************************************************/


capture program drop meta
program define meta, rclass
    version 17
    syntax varlist(min=2 max=2)
    quietly {
        local Te : word 1 of `varlist'
        local V_T : word 2 of `varlist'
        // number of non-missing observations for both Te and V_T.
        count if !missing(`Te', `V_T')
        local m = r(N) // number of studies stored as a macro
        // fixed-effect weights
        tempvar wt_FE
        gen double `wt_FE' = 1/`V_T' //Creates a new variable of fixed-effect weights = 1 / variance. 
        tempvar Te_wt  
        gen double `Te_wt' = `Te' * `wt_FE' // new variable, Effect size Te * the fixed-effect weight wt_FE
        quietly summarize `Te_wt'
        local sum_Te_wt = r(sum) //summation of the weight products
        quietly summarize `wt_FE'
        local sum_wt_FE = r(sum) //summation of the weights
        local T_FE = `sum_Te_wt' / `sum_wt_FE' //The fixed-effect pooled estimate (mean)
        // Q-statistic
        tempvar diff_T diff_sqr // diff_T: The difference between each study's effect and the fixed-effect pooled estimate
								//diff_sqr: Weighted squared difference
        gen double `diff_T' = `Te' - `T_FE'
        gen double `diff_sqr' = `wt_FE'*(`diff_T')^2
        quietly summarize `diff_sqr'
        local Q = r(sum) //Summarize -> Q = r(sum): The sum of weighted squared differences (Cochran's Q in metas) 
        // S1, S2
        local S1 = `sum_wt_FE' // Sum of all weights (already found as sum_wt_FE).
        tempvar wt_FE_sq
        gen double `wt_FE_sq' = `wt_FE'^2 
        quietly summarize `wt_FE_sq' // Creates a variable for wt_FE^2, and then sums it for S2.
        local S2 = r(sum) 
        // tau^ 
        local o2 = (`Q' - (`m' - 1)) / (`S1' - `S2' / `S1') // DerSimonian and Laird
        // random-effects weights
        tempvar wt_RE
        gen double `wt_RE' = 1/(`V_T' + `o2') // Creates random-effects weights = 1 / (V_T + \tau^2).
        // T_RE
        tempvar Te_wt_RE
        gen double `Te_wt_RE' = `Te'*`wt_RE'
        quietly summarize `Te_wt_RE' 
        local sum_Te_wt_RE = r(sum) // Weighted sum of the effects using wt_RE => sum_Te_wt_RE
        quietly summarize `wt_RE'
        local sum_wt_RE = r(sum) // total random-effects weight
        local T_RE = `sum_Te_wt_RE' / `sum_wt_RE' // The random-effects pooled effect size.
        // V_T_RE_mod
        local V_T_RE_mod = 1 / `sum_wt_RE' // Classic formula for the variance of the random-effects mean under the
			   							   // model-based approach
        // V_T_RE_rve - Robust variance estimate (Another approach to get standard errors less sensitive to model 
		// assumptions)
        tempvar diff_TR wt_RE_sq
        gen double `diff_TR' = `Te' - `T_RE'
        gen double `wt_RE_sq' = (`wt_RE')^2 * (`diff_TR')^2
        quietly summarize `wt_RE_sq'
        local num = r(sum)
        local denom = (`sum_wt_RE')^2
        local V_T_RE_rve = (`m' / (`m' - 1)) * (`num' / `denom')
        // return
        return scalar m          = `m' // number of studies
        return scalar T_RE       = `T_RE' // random effects mean
        return scalar tau2       = `o2' // tau2
        return scalar V_T_RE_mod = `V_T_RE_mod' // variance estimate from the model
        return scalar V_T_RE_rve = `V_T_RE_rve' // robust variance estimate.
		return scalar Q   = `Q' // return cochrane statistic for DL delta method
		return scalar S1  = `S1' // return FE weight sum for the same
		return scalar S2  = `S2' // return FE weight squares for the same
    }
end

	// Create the LOA maker
capture program drop loa_maker
program define loa_maker, rclass
    version 17
    syntax varlist(min=4 max=4)
    // The user is expected to pass: bias V_bias logs2 V_logs2
    local bias    : word 1 of `varlist' // mean paco2-tcco2
    local V_bias  : word 2 of `varlist' // the variance in that bias
    local logs2   : word 3 of `varlist' // natural log of studies difference variance
    local V_logs2 : word 4 of `varlist' // variance in the nat log of study variance
    // 1. Run meta for bias
    meta `bias' `V_bias'
    local m_bias       = r(m) // number of studies
    local bias_mean    = r(T_RE) // random-effects pooled estimate of the bias
    local tau2_bias    = r(tau2) // the between-study variance in the bias (i.e., how much each study's mean difference deviates from the overall mean difference).
    local V_bias_mod   = r(V_T_RE_mod) // model variance estimate for the pooled bias
    local V_bias_rve   = r(V_T_RE_rve) // a robust variance estimate for that pooled bias.
    
    // 2. Run meta again, but for the log of the within-study variances, to get a pooled estimate of "log(sigma^2)."
    meta `logs2' `V_logs2'
    local m_logs2      = r(m) // number of studies
    local logs2_mean   = r(T_RE) // random-effects pooled log variance
    local tau2_logs2   = r(tau2) // between-study variance in log(sigma2) - ie. how much the study-specific within-study variance differs from the overall average variance
    local V_logs2_mod  = r(V_T_RE_mod) // a model-based variance estimate for that pooled log-variance.
    local V_logs2_rve  = r(V_T_RE_rve) // a robust variance estimate for that pooled log-variance.
    
    // we expect the number of studies to be the same, so let's call that m
    local m = `m_bias'
    
    // 3. sd2_est = exp(logs2_row[2]) i.e. exp(logs2_mean) == Convert logs2 -> Pooled Standard Deviation
    //    bias_mean is already in local bias_mean
    //    tau_est   is tau2_bias
    local sd2_est = exp(`logs2_mean')
    local tau_est = `tau2_bias'
    
    // 4. LOA_L, LOA_U - The Bland-Altman 95% LOA, using the pooled bias ± 2 times the square root of the total variability (\sigma^2 + \tau^2).
	// the exponentiated pooled mean of logs2 => an estimate of \sigma^2.
	// the \tau^2 estimate for the bias (between-study variability).
	//    LOA_L = bias_mean - 2*sqrt(sd2_est + tau_est)
    //    LOA_U = bias_mean + 2*sqrt(sd2_est + tau_est)
    local LOA_L = `bias_mean' - 2*sqrt(`sd2_est' + `tau_est') 
    local LOA_U = `bias_mean' + 2*sqrt(`sd2_est' + `tau_est') 
	
    
    // 5. tcrit = qt(1-0.05/2, m-1) in R
    // In Stata, we can get that via invttail(m-1, .025)
    local df = `m' - 1 // degrees of freedom
    local tcrit = invttail(`df', 0.025) // tcrit = invttail(df, 0.025): The 2-sided 95% t critical value with df degrees of freedom. (Equivalent to qt(1 - 0.05/2, m-1) in R.)
    
    // Variance Decompositions: B1, B2
	// Weights used when combining the variance contributions from \sigma^2 vs. \tau^2.
    // B1 = sd2_est^2 / (sd2_est + tau_est)
    // B2 = tau_est^2   / (sd2_est + tau_est)
    local B1 = (`sd2_est'^2) / (`sd2_est' + `tau_est')
    local B2 = (`tau_est'^2) / (`sd2_est' + `tau_est')
    
    // Recomputes the sum of weights (S1), sum of squared weights (S2), sum of cubed weights (S3). 
	// for certain variance formulas.
    // We need to parse the data again (like we did in the meta program). 
    // We'll do it here but are mindful it might be a large data set.
    
    // We'll create some temp variables
    tempvar w
    gen double `w' = 1/`V_bias' if !missing(`V_bias', `bias')
    quietly summarize `w'
    scalar S1 = r(sum)
    
    // sum of w^2
    tempvar w2
    gen double `w2' = `w'^2
    quietly summarize `w2'
    scalar S2 = r(sum)
    
    // sum of w^3
    tempvar w3
    gen double `w3' = `w'^3
    quietly summarize `w3'
    scalar S3 = r(sum)
    
    // A0, A1, A2 - these are parts of a (commented) derivation for variance of \log(\tau^2)
    scalar A0 = 2*(`m'-1)/(S1 - S2/S1)^2
    scalar A1 = 4/(S1 - S2/S1)
    scalar A2 = 2*(S2 - 2*S3/S1 + (S2^2)/(S1^2)) / (S1 - S2/S1)^2
    
    // V_logT2 = 2/sum((V_bias + tau_est)^(-2))   
    tempvar denomvar
    gen double `denomvar' = ( `V_bias' + `tau_est')^-2 if !missing(`V_bias')
    quietly summarize `denomvar'
    local sum_denom = r(sum)
    
    local V_logT2 = 2 / `sum_denom' // Another approach used in the code to approximate the variance of the log of \tau^2.
    //The final V_logT2 is used to combine uncertainty from \tau^2.
	
    // Now compute V_LOA_mod, V_LOA_rve - Model-based variance of the LOA. Summing the variance from the meta-analysis of bias, from logs2, and from \tau^2.
    // from R code:
    // V_LOA_mod = bias_row[4] + B1*logs2_row[4] + B2*V_logT2
    // bias_row[4] = V_bias_mod; logs2_row[4] = V_logs2_mod
    local V_LOA_mod = `V_bias_mod' + `B1'*`V_logs2_mod' + `B2'*`V_logT2'
	
	// The robust-variance version
    // V_LOA_rve = bias_row[5] + B1*logs2_row[5] + B2*V_logT2
    // bias_row[5] = V_bias_rve; logs2_row[5] = V_logs2_rve
    local V_LOA_rve = `V_bias_rve' + `B1'*`V_logs2_rve' + `B2'*`V_logT2'
	
    // 6. final CI bounds
    // CI_L_mod = LOA_L - tcrit*sqrt(V_LOA_mod)
    // CI_U_mod = LOA_U + tcrit*sqrt(V_LOA_mod)
	//95% confidence interval for the LOA under model-based variance.
    local CI_L_mod = `LOA_L' - `tcrit'*sqrt(`V_LOA_mod')
    local CI_U_mod = `LOA_U' + `tcrit'*sqrt(`V_LOA_mod')
    
	//95% CI for the LOA under robust variance
    local CI_L_rve = `LOA_L' - `tcrit'*sqrt(`V_LOA_rve')
    local CI_U_rve = `LOA_U' + `tcrit'*sqrt(`V_LOA_rve')
    
    // 7. Return these 10 values in r()
    return scalar studies = `m'
    return scalar bias    = `bias_mean'
    // The R code's third item was sqrt(sd2_est), so let's store that:
    return scalar sd      = sqrt(`sd2_est')
    return scalar tau2    = `tau_est'
    return scalar LOA_L   = `LOA_L'
    return scalar LOA_U   = `LOA_U'
    return scalar CI_Lm   = `CI_L_mod'
    return scalar CI_Um   = `CI_U_mod'
    return scalar CI_Lr   = `CI_L_rve'
    return scalar CI_Ur   = `CI_U_rve'
end

/***************************************************************************
  2. Apply the programs to the restricted conway populations 
***************************************************************************/

* File paths
local src_meta "Conway_Tcco2_working_dataset"
local src_work "TriNetX_Working_Dataset"

* Keep meta dataset in default frame (already loaded)
* Open the working dataset in a separate frame so we can write per-group simulated vars
capture frame drop work
frame create work
frame work: use "`src_work'", clear

* Set seed once (for reproducibility of draws)
set seed 12345

*-------------------- Prepare result holders on FULL meta data -----------------*
foreach group in icu_group pft_group ed_inp_group {
    foreach v in n bias sd tau2 loa_l loa_u loa_ci_l_m loa_ci_u_m loa_ci_l_rve loa_ci_u_rve {
        capture confirm variable `v'_`group'
        if _rc gen double `v'_`group' = .
    }
}

*==================== Main loop: per restricted dataset ========================*
foreach group in icu_group pft_group ed_inp_group {

    *---- (A) RESTRICTED RUN: keep only this group, run loa_maker, get stats ---*
    preserve
        keep if `group' == 1

        * Run your meta-based LOA builder on the restricted subset
        loa_maker bias v_bias logs2 v_logs2

        * Collect results (the ones you used as *_cc before)
        local n_res        = r(studies)
        local bias_res     = r(bias)
        local sd_res       = r(sd)
        local tau2_res     = r(tau2)
        local loa_l_res    = r(LOA_L)
        local loa_u_res    = r(LOA_U)
        local ci_l_m_res   = r(CI_Lm)
        local ci_u_m_res   = r(CI_Um)
        local ci_l_rve_res = r(CI_Lr)
        local ci_u_rve_res = r(CI_Ur)

        *--------------- STEP 3: Random draws (logs2, bias, tau^2) --------------*
        * pooled log(sigma^2) & its variance (robust)
        meta logs2 v_logs2
        local logs2_mean = r(T_RE)
        local var_logs2  = r(V_T_RE_rve)

        local logs2_draw  = rnormal(`logs2_mean', sqrt(`var_logs2'))
        local sigma2_draw = exp(`logs2_draw')

        * pooled bias and its variance (robust)
        meta bias v_bias
        local bias_est = r(T_RE)
        local var_bias = r(V_T_RE_rve)
        local bias_draw = rnormal(`bias_est', sqrt(`var_bias'))

        * DerSimonian-Laird delta-method for tau^2 draw (log-normal approximation)
        local Q      = r(Q)
        local S1     = r(S1)
        local S2     = r(S2)
        local m      = r(m)
        local C      = `S1' - (`S2'/`S1')
        local varQ   = 2*(`m' - 1)
        local tau2_est = r(tau2)
        local var_tau2 = (1/(`C'^2)) * `varQ'
        local se_tau2   = sqrt(`var_tau2')
        local logtau2_est = log(`tau2_est')
        local var_logtau2 = `var_tau2' / (`tau2_est'^2)
        local se_logtau2  = sqrt(`var_logtau2')
        local logtau2_draw = rnormal(`logtau2_est', `se_logtau2')
        local tau2_draw    = exp(`logtau2_draw')

        *--------------- STEP 4: Combine -> total SD for simulation --------------*
        local total_sd_draw = sqrt(`sigma2_draw' + `tau2_draw')

    restore

    *---- (B) WRITE BACK: put subgroup constants on full meta dataset -----------*
    replace n_`group'              = `n_res'        if `group'==1
    replace bias_`group'           = `bias_res'     if `group'==1
    replace sd_`group'             = `sd_res'       if `group'==1
    replace tau2_`group'           = `tau2_res'     if `group'==1
    replace loa_l_`group'          = `loa_l_res'    if `group'==1
    replace loa_u_`group'          = `loa_u_res'    if `group'==1
    replace loa_ci_l_m_`group'     = `ci_l_m_res'   if `group'==1
    replace loa_ci_u_m_`group'     = `ci_u_m_res'   if `group'==1
    replace loa_ci_l_rve_`group'   = `ci_l_rve_res' if `group'==1
    replace loa_ci_u_rve_`group'   = `ci_u_rve_res' if `group'==1

    *---- (C) SIMULATE on WORKING DATASET: only for rows in this group ----------
	frame change work
    * create columns if absent
    capture confirm variable difference_draw_`group'
    if _rc gen double difference_draw_`group' = .

    capture confirm variable tcco2_`group'_sim
    if _rc gen double tcco2_`group'_sim = .

    * update ONLY rows where this group's flag == 1
    replace difference_draw_`group' = rnormal(`bias_draw', `total_sd_draw') if `group'==1
    replace tcco2_`group'_sim       = paco2 - difference_draw_`group'     if `group'==1
	* Change back to default frame so that the next iteration of the loop can run on the other groups
	frame change default
}
*Label for clarity in the working conway dataset
label var bias_icu "Matches reported bias from Conway"
label var bias_pft "Matches reported bias from Conway"
label var bias_ed_inp "Matches reported bias from Conway"
frame change work

/***************************************************************************
  3. Generate a variable for hypercapnic v not for both PaCO2 and TcCO2 for each setting
* Create binary indicators for PaCO₂ ≥45 mmHg and TcCO₂ ≥45 mmHg (overall and by setting).
* Create binary indicators for more extreme TcCO2 cutoffs of ≤40 and ≥50mmHg
***************************************************************************/


*Binary variable for PaCO2=hypercapnic or not:
capture drop paco2_hypercap
gen paco2_hypercap = .
capture label variable paco2_hypercap "PaCO2 ≥ 45 mmHg within 24h of admission"
replace paco2_hypercap = 1 if !missing(paco2) & paco2 >= 45
replace paco2_hypercap = 0 if !missing(paco2) & paco2 < 45
capture label define paco2_hypercap_lab 0 "PaCO2 < 45 mmHg" 1 "PaCO2 ≥ 45 mmHg" 
label values paco2_hypercap paco2_hypercap_lab
*Binary variable for Simulated TcCO2=hypercapnic or not:
foreach group in pft_group ed_inp_group icu_group{
capture drop tcco2_`group'_hypercap
gen tcco2_`group'_hypercap = .
capture label variable tcco2_`group'_hypercap "TcCO2 ≥ 45 mmHg for `group'?"
replace tcco2_`group'_hypercap =  1 if !missing(tcco2_`group'_sim) & tcco2_`group'_sim >= 45
replace tcco2_`group'_hypercap =  0 if !missing(tcco2_`group'_sim) & tcco2_`group'_sim < 45
capture label define tcco2_`group'_hypercap_lab 0 "TcCO2 < 45 mmHg" 1 "TcCO2 ≥ 45 mmHg" 
label values tcco2_`group'_hypercap tcco2_`group'_hypercap_lab
}
foreach group in pft_group ed_inp_group icu_group{
gen confusion_matrix_`group' = . //Given simulated readings, was an error made? 
label variable confusion_matrix_`group' "TcCO2 Confusion Matrix for `group'"
replace confusion_matrix_`group' = 1 if paco2_hypercap == 0  & tcco2_`group'_hypercap == 0 //TN
replace confusion_matrix_`group' = 2 if paco2_hypercap == 0 & tcco2_`group'_hypercap == 1 //FP
replace confusion_matrix_`group' = 3 if paco2_hypercap == 1 & tcco2_`group'_hypercap == 1 //TP
replace confusion_matrix_`group' = 4 if paco2_hypercap == 1  & tcco2_`group'_hypercap == 0 //FN
label define matrix_`group'_lab 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values confusion_matrix_`group' matrix_`group'_lab
diagt paco2_hypercap tcco2_`group'_hypercap //calculate operating characteristics
tab confusion_matrix_`group'
}

*Binary variable for Simulated TcCO2=hypercapnic or not:, more extreme TcCO2 cutoffs
foreach group in pft_group ed_inp_group icu_group{
capture drop tcco2_`group'_extreme
gen tcco2_`group'_extreme = .
capture label variable tcco2_`group'_extreme "TcCO2 ≤40 or ≥ 50 mmHg for `group'?"
replace tcco2_`group'_extreme =  1 if !missing(tcco2_`group'_sim) & tcco2_`group'_sim >= 50
replace tcco2_`group'_extreme =  0 if !missing(tcco2_`group'_sim) & tcco2_`group'_sim < 40
capture label define tcco2_extreme_lab 0 "TcCO2 < 40 mmHg" 1 "TcCO2 ≥ 50 mmHg" 
label values tcco2_`group'_extreme tcco2_extreme_lab
}
foreach group in pft_group ed_inp_group icu_group{
gen matrix_`group'_extreme = . //Given simulated readings, was an error made? 
label variable matrix_`group'_extreme "TcCO2 Confusion Matrix for `group' at extreme cutoffs"
replace matrix_`group'_extreme = 1 if paco2_hypercap == 0  & tcco2_`group'_extreme == 0 //TN
replace matrix_`group'_extreme = 2 if paco2_hypercap == 0 & tcco2_`group'_extreme == 1 //FP
replace matrix_`group'_extreme = 3 if paco2_hypercap == 1 & tcco2_`group'_extreme == 1 //TP
replace matrix_`group'_extreme = 4 if paco2_hypercap == 1  & tcco2_`group'_extreme == 0 //FN
label define matrix_`group'_extreme_lab 1 "True Negative" 2 "False Positive" 3 "True Positive" 4 "False Negative"
label values matrix_`group'_extreme matrix_`group'_extreme_lab
diagt paco2_hypercap tcco2_`group'_extreme //calculate operating characteristics
tab matrix_`group'_extreme
}


*Save the default frame as the working conway dataset and the working frame as the DAB_working dataset
frame default: save "Conway_Tcco2_working_dataset", replace
cd .. 
frame work: save "Final TcCO2 Dataset", replace
