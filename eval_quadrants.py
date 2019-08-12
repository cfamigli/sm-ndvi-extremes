import FVD
import GIMMS as gm
import GLDAS as gld
import numpy as np
from PIL import Image
import os

os.chdir("matfiles")
mask = FVD.loadmat('land_mask.mat') # mask for land surface: land=-9999,other=0

pct = 15
val = 0.1
nrows = 720
ncols = 1440

F_init = FVD.loadmat('sm_numerator_monthly_controlled_fixed.mat')
D_init = FVD.loadmat('sm_denominator_monthly_controlled_fixed.mat')

npx_low = FVD.loadmat('num_anoms_low.mat')
npx_high = FVD.loadmat('num_anoms_high.mat')
ndvi = FVD.loadmat('mean_ndvi.mat')

F = np.copy(F_init)
D = np.copy(D_init)

F[(abs(F_init)<val) & (abs(D_init)<val)] = float('nan')
D[(abs(F_init)<val) & (abs(D_init)<val)] = float('nan')

FVD.savemat(F, 'sm_numerator_monthly_controlled_fixed_masked.mat')
#F[npx_high<5] = float('nan')
#D[npx_low<5] = float('nan')
full = abs(F) / (abs(F) + abs(D))

F[F==0] = float('nan')
D[D==0] = float('nan')

#FVD.plot_fvd(F, pct, -10, 10, mask=mask, kind='full')
#FVD.plot_fvd(D, pct, -10, 10, mask=mask, kind='full')

quads = FVD.define_quadrants(F, D, mask)
FVD.savemat(quads, 'quads_fixed_F_v.1_e.25.mat')
FVD.plot_quads(quads, 0, 4, mask=mask)

# at places where productivity changes most during drought, how is productivity changing?
'''directional_dry = np.copy(D)
directional_dry[full>0.5] = float('nan')

# at places where productivity changes most during wet conditions, how is productivity changing?
directional_wet = np.copy(F)
directional_wet[full<0.5] = float('nan')

FVD.plot_fvd(directional_dry, pct, -10,10, mask=mask)
FVD.plot_fvd(directional_wet, pct, -10,10, mask=mask)
FVD.plot_general_dist(directional_dry)
FVD.plot_general_dist(directional_wet)'''

# import other data
'''land_cover_type = np.array(Image.open('land_cover.tif'))*1. # land cover type
data = FVD.match_lc(land_cover_type, F)
#FVD.plot_lc(data)

X = gld.prepare_mat_with_statistics(nrows, ncols)
Yd = directional_dry.reshape(-1,1)
#Yd[abs(Yd)<1] = float('nan')
Yw = directional_wet.reshape(-1,1)
#Yw[abs(Yw)<1] = float('nan')

print(X[:,1].reshape(-1,1).shape)
print(Yd.shape)
print(Yw.shape)

for col in range(X.shape[1]):
    FVD.plot_scatter_density(X[:,col].reshape(-1,1), Yd, str(col))
    FVD.plot_scatter_density(X[:,col].reshape(-1,1), Yw, str(col))'''
