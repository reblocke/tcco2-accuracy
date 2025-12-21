clear all
use "data.dta", clear


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
        gen double `Te_wt' = `Te' * `wt_FE' // new variable that is the product of the effect size Te and the fixed-effect weight wt_FE

        quietly summarize `Te_wt'
        local sum_Te_wt = r(sum) //summation of the weight products

        quietly summarize `wt_FE'
        local sum_wt_FE = r(sum) //summation of the weights
 
        local T_FE = `sum_Te_wt' / `sum_wt_FE' //The fixed-effect pooled estimate (mean)

        // Q-statistic
        tempvar diff_T diff_sqr 
		//diff_T: The difference between each study's effect and the fixed-effect pooled estimate
		//diff_sqr: Weighted squared difference, i.e., \text{wt_FE} \times (Te - T_{FE})^2.
        gen double `diff_T' = `Te' - `T_FE'
        gen double `diff_sqr' = `wt_FE'*(`diff_T')^2
        quietly summarize `diff_sqr'
        local Q = r(sum) //Summarize -> Q = r(sum): The sum of weighted squared differences. This is the Cochran's Q statistic used in meta-analysis to assess heterogeneity.

        // S1, S2
        local S1 = `sum_wt_FE' // Sum of all weights (already found as sum_wt_FE).
        tempvar wt_FE_sq
        gen double `wt_FE_sq' = `wt_FE'^2 
        quietly summarize `wt_FE_sq' // Creates a variable for wt_FE^2, and then sums it for S2.
        local S2 = r(sum) 

        // tau^2
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
        local V_T_RE_mod = 1 / `sum_wt_RE' // Classic formula for the variance of the random-effects mean under the model-based approach: \frac{1}{\sum w_i^{RE}}.

        // V_T_RE_rve - Robust variance estimate
		// Another approach to get standard errors less sensitive to model assumptions.
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
    local tau2_logs2   = r(tau2) // between-study veriance in log(sigma2) - ie. how much the study-specific within-study variance differs from the overall average variance
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
  2. Run the LOA maker to get the meta-analysis results for bias, etc.
***************************************************************************/

loa_maker bias v_bias logs2 v_logs2

display "Number of studies: " r(studies) 
display "Pooled bias:       " r(bias) // random-effects pooled bias
display "Pooled SD:         " r(sd) // \sqrt{\sigma^2} from the logs2 meta-analysis
display "tau^2 estimate:    " r(tau2) // between-study variance of the bias
display "LOA (lower):       " r(LOA_L) // the 95% limits of agreement
display "LOA (upper):       " r(LOA_U)
display "LOA CI (model):    [" r(CI_Lm) ", " r(CI_Um) "]" // model-based 95% CI for the LOA
display "LOA CI (robust):   [" r(CI_Lr) ", " r(CI_Ur) "]" // robust-variance 95% CI for the LOA


****************************************************************************
*3. Get draws for logs2 (=> sigma^2), tau^2, and the bias
* There overall errors are decomposed into 3 components: within-study (sigma), between study(tau), and average bias
* Each of these are models, as well as uncertainty in the parameters (e.g. how precisely do we know each component?)
****************************************************************************

//First get the log-variances for the log2-draw 
meta logs2 v_logs2 //easiest just to re-calculate these at the top-level
//Note: we do NOT use the TAU from this at all, Between-study variance in the *log of the within-study variance* 
local logs2_mean   = r(T_RE) // logs2_mean: the random-effects estimate from "meta logs2 V_logs2" - pooled log(sigma2)
display "local logs2_mean = `logs2_mean'"
local var_logs2    = r(V_T_RE_rve) // robust variance
display "local var_logs2   = `var_logs2'"

// (A) Random draws for logs2 => sigma^2
local logs2_draw   = rnormal(`logs2_mean', sqrt(`var_logs2'))
local sigma2_draw  = exp(`logs2_draw')
// If logs2_draw is extremely large or small, sigma2_draw can become big or near 0
display "Drawn logs2 = `logs2_draw', so sigma^2 = `sigma2_draw'"

//Second, get the bias for the bias_draw
meta bias v_bias
local bias_est     = r(T_RE) // the random-effects pooled bias.
local var_bias     = r(V_T_RE_rve) // variance estimate for pooled bias

display "Bias est= `bias_est'"
display "var_bias= `var_bias'"
local bias_draw  = rnormal(`bias_est', sqrt(`var_bias'))
display "Drawn bias = `bias_draw'"

//Third, estimate Tau^2 to get the between-study variance
// (B) Use Delta-method of DerSimonian and Laird to approx tau^2
// retrieve Q, S1, S2
local Q  = r(Q)
local S1 = r(S1)
local S2 = r(S2)
local m  = r(m)

// define C = denom
local C = `S1' - (`S2' / `S1')

// approximate Var(Q) ~ 2(m-1)
local varQ = 2*(`m' - 1)
local tau2_est     = r(tau2) //how much each study's mean difference deviates from overall 
local var_tau2 = (1/(`C'^2)) * `varQ' //calc var in tau2

local se_tau2 = sqrt(`var_tau2')
local logtau2_est = log(`tau2_est')
local var_logtau2 = `var_tau2' / (`tau2_est'^2)
//ignoring the variance contribution from C, as well as correlation with the estimate of the overall mean, etc. But it is often seen in elementary expositions of the DL method.

local se_logtau2 = sqrt(`var_logtau2')
local logtau2_draw = rnormal(`logtau2_est', `se_logtau2')
local tau2_draw    = exp(`logtau2_draw')

display "Delta-method tau2_draw = `tau2_draw'"

/* limitations: 	
1.	This method still has assumptions. For instance, we approximate Q\sim\chi^2_{m-1} with variance 2(m-1). If the data are very heterogeneous or outliers exist, this can be inaccurate.
2.	C treated as fixed. Strictly, S_1 and S_2 are random variables (they depend on data), so a more rigorous approach would require partial derivatives wrt S_1 and S_2 plus their covariance with Q. Often references ignore that to keep it simple.
*/


****************************************************************************
* 4. Combine them => total SD, then simulate on the datasimulate
****************************************************************************


local total_sd_draw = sqrt(`sigma2_draw' + `tau2_draw')
display "Drawn total SD = `total_sd_draw'"


use "/Users/reblocke/Research/tcco2-accuracy/Data/In Silico TCCO2 Database.dta", clear //swap to yours

// For each row, generate difference from Normal(bias_draw, total_sd_draw)
set seed 12345
display "Drawn bias = `bias_draw'"
display "Drawn total SD = `total_sd_draw'"
gen double difference_draw = rnormal(`bias_draw', `total_sd_draw')
gen double tcco2_sim       = paco2 - difference_draw

summarize paco2 tcco2_sim difference_draw
//save "co2_sim_data.dta", replace


