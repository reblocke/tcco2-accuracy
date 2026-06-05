
*---------------------------------------------------------------------*
******You should be able to use this do-file for the full analysis*****
*---------------------------------------------------------------------*

clear
local project_root "."
if "`1'" != "" local project_root "`1'"
cd "`project_root'"

				/*==================================*
				Cleaning of the data and creation of working dataset
				*==================================*/

*Step 1: Cleaning of Conway Dataset
*Uses "Original Conway Dataset_Full", Makes "Conway_Tcco2_pruned_dataset"
run 1_conway_tcco2_cleaning_do

*Step 2: Cleaning of TriNetX Dataset
*USes "TriNetX_Pruned_Dataset", Makes "TriNetX_Working_Dataset"
run 2_trinetx_cleaning_do

*Step 3: Run Meta-Analysis on "Conway_Tcco2_pruned_dataset" (Makes "Conway_Tcco2_working_Dataset"), create simulated TcCO2 values using these values + incorporated uncertainty on TriNetX PaCO2 values (Using "TriNetX_Working_Dataset"), Makes "Final TcCO2 Dataset" saved to the parent folder.
run 3_tcco2_uncertainty_and_simulation_do

display as txt "All steps completed successfully."
				
				/*==================================*
						Analysis of the Data
				*==================================*/

use "Final TcCO2 Dataset", clear

*======================================
*Population characteristics
*======================================

**Age, Race, Sex
summ age, det // Median 64 (51, 74)
tab race // NH White 64.5%, Black 14.3%, Hispanic 5.4%, Asian 1.9%
tab sex // Male 54.7%

** Mean PaCO2 / group:
summ paco2, det // Mean 42.5 ± 16.1 for all
foreach group in pft_group ed_inp_group icu_group{
display = "`group'"
summ paco2 if `group'==1, det
} // Mean Amb (50.0 ± 9.3), Mean ED/Inp (42.0 ± 14.0), Mean ICU (43.7 ± 20.1)

**Prevalence Hypercapnia / group
foreach group in pft_group ed_inp_group icu_group{
display = "`group'"
tab paco2_hypercap if `group'==1
} // 26.5% PFT, 29.3% ED/Inp, 31.7% ICU
 
**Prevalence of midrange PaCO2 for each group
count if tcco2_pft_group_extreme ==1 // 1,410
count if tcco2_pft_group_extreme ==0 // 4,910
count if tcco2_pft_group_extreme ==. & pft_group==1 // 4,136 in Midrange TcCO2 40-50
count if pft_group==1 & missing(tcco2_pft_group_sim)

summ paco2, det // Mean 42.5 ± 16.1 for all
foreach group in pft_group ed_inp_group icu_group{
display = "`group'"
summ paco2 if `group'==1, det
} 



*======================================
*Compare PaCO2 v TcCO2
*Display confusion matrices and test characteristics for each population
*======================================

*TcCO2 Cutoff 45
foreach group in pft_group ed_inp_group icu_group{
diagt paco2_hypercap tcco2_`group'_hypercap //calculate operating characteristics
tab confusion_matrix_`group'
}

*TcCO2 Cutoff 40 and 50
foreach group in pft_group ed_inp_group icu_group{
diagt paco2_hypercap tcco2_`group'_extreme //calculate operating characteristics
tab matrix_`group'_extreme
}

*======================================
*Graphs
*======================================

preserve 
keep if paco2 < 150
hist paco2, freq color(navy) ylabel(, format(%9.0g)) ///
    title("Distribution of PaCO2 across all settings")
restore

local groups pft_group ed_inp_group icu_group
local glist
foreach group of local groups {
    preserve
        recode paco2 (min/30=.) (60/max=.)
        replace paco2 = round(paco2, 1)
		local x_min = 30
        local x_max = 60
        local x_step = 5
        catplot, over(confusion_matrix_`group') percent over(paco2, gap(0.01)) percent stack ///
            recast(bar) asyvars ///
            bar(1, fcolor(teal%25) lcolor(teal)) ///
            bar(2, fcolor(purple%25) lcolor(purple)) ///
            bar(3, fcolor(cranberry%25) lcolor(cranberry)) ///
            bar(4, fcolor(orange_red%25) lcolor(orange_red)) ///
            ylabel(, labsize(small)) ///
            legend(pos(6) rows(1) size(large)) ///
            xsize(5) ysize(2.5) ///
			title("`group'", size(med)) ///
            name(g_`group', replace)   // <— keep this graph in memory with a unique name
    restore
    local glist `glist' g_`group'
}
graph combine `glist', ///
	row (3) ///
    ycommon xcommon ///
    imargin(zero) ///
    title("Agreement of PaCO₂ and TcCO₂ Across Clinical Settings", size(large)) ///
    b1title("Arterial Partial Pressure of Carbon Dioxide (PaCO₂)", size(large)) ///
    l1title("TcCO₂ Categorization Percentage", size(large))
	graph export "Agreement graph.pdf", as(pdf) replace
	
preserve
recode paco2 (min/20=.) (100/max=.) //truncate the range 
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
	title("Distribution of PaCO2", size(huge)) ///
	scheme(white_tableau) ///
	xsize(9) ysize(3.5)
graph save total_distributions.gph, replace
restore
