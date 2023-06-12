# README.md

Description of content and file structure of 'dataframe.csv', 'infusiondata.csv', 'modelparams.csv'.

## dataframe.csv
The original file published together with the supplementary material of the Eleveld paper.

ID - patient ID \
STUDY - number of the study (1-30) \
TIME - time in minutes \
CP - plasma concentration [ug/ml] \
ISMEAS - 1 if it is a measurement, 0 otherwise \
AMT - the amount of drugs given at this time instance [mg] \
RATE - the drug rate at which this amount what delivered [mg/min] \
EVID - opposite of ISMEAS \
AGE - age [years] \
WGT - weight [kg] \
HGT - height [cm] \
M1F2 - gender, male 1, female 2\
A1V2 - arterial 1, venous 2\
NOADD1ADD2 - if opiods were added, not added 1, added 2 \
PMA - post mentstrual age [years] \
BMI - bmi [kg/m^2]

## eleveld_modelparams.csv
ID - patient ID (1-1033) \
StudyNbr - number of the study (1-30) \
V1, V2, V3 - compartment volumes [l] \
CL, Q2, Q3 - clearances [l/min]

## eleveld_infusiondata.csv
ID - patient ID (1-1033) \
TIME - time in seconds \
InfusionRate - infusionrate [ug/s] \
Bolus - bolus dose [ug] \
IsMeasurement - if this time instance is a measurement (relates to measurements in dataframe.csv, CP). No 0, Yes 1.

