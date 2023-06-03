# Python-Graphing-Script
A python script that allows the user to visualize many different types of data without changing any hard-coded data. It accomplishes this by saving each different plot's settings, "presets", to JSON files. The script allows for sub-figures and sub-plots additionally, each using a different JSON file to store its settings (making three files in total). Additionally, the code contains an interface to allow the user to create new or movify existing presets, so that no files ever need to be modified.

Data is read in from .tsv, .csv, or .dat files, in which each column is a different data point, i.e. an x- and y-coordinate, separated by spaces, tabs, or a comma.

Included in this GitHub repository are examples of my personal figure, subplot, and axis presets. I have included images of some of the plots that can be generated using this script, along with the presets used to generate them.

![RMSD_First_CA](https://github.com/Sean-S1225/Python-Graphing-Script/assets/66101203/4c74129b-54a7-409c-b4bf-21b8e971acd8)
Figure preset: One, Subplot preset: Two, Axis presets: RMSD, RMSD

![RMSF_CA](https://github.com/Sean-S1225/Python-Graphing-Script/assets/66101203/7803a104-59e9-49a0-9be2-e122277789ab)
Figure preset: One, Subplot preset: Two, Axis presets: RMSF, RMSF

![Res 376](https://github.com/Sean-S1225/Python-Graphing-Script/assets/66101203/25cd0b5d-1a0a-4f15-afa9-559f0b32d032)
Figure preset: Rama4,4,3 Subplot preset: 1x4, 1x4, Rama3x1, presets: R_W, R_Y, R_P, R_All, R_W, R_Y, R_P, R_All, G_W, G_Y, G_P

![RMSD_Abox_dt_Variable_AllAtom_First](https://github.com/Sean-S1225/Python-Graphing-Script/assets/66101203/24f410a4-d720-4455-b67e-451f984a8af3)
Figure preset: One-BigRMSD, Subplot preset: Seven, Axis Presets: RMSD, RMSD_10FrAvg, RMSD, RMSD_10FrAvg, RMSD_10FrAvg, RMSD_10FrAvg, RMSD_10FrAvg
