clear
use "TriNetX_Pruned_Dataset", clear

*----
*Generate population groups to match the terminology from the Conway dataset
**Note that the ambulatory group (which I'm calling PFT group here because that's how 'ambulatory' was defined in the Conway analysis) and the inpatient/emergency (which I'm lumping together as the Conway metaanalysis just had non-ICU acute respiratory failure group) are explicitly defined in the TriNetX database whereas the ICU population I am defining as being inpatient (not ED, not ambulatory) with critical care time billed; probably imperfect 
*----
gen pft_group = is_amb ==1 
tab pft_g, mis // 10,456; no missing values
gen icu_group = is_inp ==1 & cc_time==1 & is_emer==0 & is_amb==0
tab icu_g, mis // 66,704; no missing values
gen ed_inp_group = (is_inp ==1 | is_emer==1) & cc_time==0 & is_emer==0
tab ed_inp_g, mis // 91,503; no missing values


/*-----
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
drop if paco2==.


save "TriNetX_Working_Dataset", replace 
