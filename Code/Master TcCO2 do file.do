clear
cd "/Users/DustinAnderson/Desktop/Medicine/Fellowship/Research/Hypercapnia/TcCO2/Working Folder/Individual Data and Do-Files"

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

use "Final TcCO2 Dataset"

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
