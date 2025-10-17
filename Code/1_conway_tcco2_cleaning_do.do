
/*****
Conway Cleaning
*****/

//Methods note: source of data: https://figshare.com/articles/dataset/Accuracy_of_TcCO2_monitoring_meta-analysis/6244058/2

use "Original Conway Dataset_Full"

*Drop unecessary variables/subgroups
drop technology location_of_sensor device_temp funding_equip picu neonates surgery paed volunteer olv sedat crf cpex_11

*Rename and apply labels for populations of interest
rename icu1 icu_group
label var icu_group "Adult ICU patients"
rename respiratory_lft pft_group
label var pft "PFT Clinic" 
rename ed_arf ed_inp_group
label var ed_inp "ED and Inpatients (ARF)"

save "Conway_Tcco2_pruned_dataset", replace
