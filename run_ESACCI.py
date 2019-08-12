
import ESACCI as ec
import os
import glob

os.chdir("../ESACCI/daily_files/COMBINED")
years = ec.get_subfolders()
#del years[:3]
#del years[-3:]
print(years)

for year in years:
    os.chdir(year)
    files = ec.get_files('nc')
    ymat = ec.combine_files(files, 720, 1440)
    #ec.plot_scene(ymat[0,:,:])
    os.chdir("..")
    break
