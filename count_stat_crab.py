#!/uufs/astro.utah.edu/common/home/u1019304/VENV2.7.10/bin/python

# syntax: count_stat_new.py det file lowenergy highenergy sun/nosun/full

# Finds the statistics of an image for Mi,j = a*Ai,j + b1*a0 + b2*a1 + b3*a2 + b4*a3 + Cn*c0, where M is the model value, A is the physical location of the pixel, a is a variable to find, a#'s are the base values of the individual dets with b being the norm tied the detector of choice, and Cn in the straylight norm with c0 being the bin representation.

# Both the excl.reg and the straylight.reg files are assumed to be in a standard 4.0 file format with 3 lines dedicated to: format, global conditions, and coordinate system
# It is also assumed that the only coordinate system used will be PHYSICAL. This is justified since the fitting is done on DET1 coordinate system with NuSTAR. 
# Finally, each region within the file should be formated with parentheses and delimited with a comma; i.e, circle(100,100,20) 

# It does not matter if the region is declared as an excluded or included region (signified with a - before the shape for DS9), the program only looks for keywords


import os, sys, numpy as np
from astropy.io import fits
import scipy.optimize as opt
from scipy.optimize import curve_fit
import matplotlib
import copy
from lmfit import Parameters, minimize
from scipy.integrate import quad

matplotlib.use('Agg')

import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.stats import norm


# Syntax: This should be ran from 
#   count_stat_crab.py 
#   telescope[A/B] 
#   fits_file 
#   lower_energy_limit        
#   high_energy_limit         
#   observation_ID(7 digit) 
#   write_to_directory 
#   Straylight_region(physical values assumed to have ds9 format) 
#   reference_detector(0,1,2,3) 
#   exclusion_region(physical values assumed to have ds9 format) 


######### Verify the loaded files to ensure it's what is being used 
if len(sys.argv) > 10:
    print('Too many arguments have been passed. Recieved: {}'.format(len(sys.argv)))
    sys.exit()

detp = sys.argv[1]
data = sys.argv[2]
lowe = sys.argv[3]
highe = sys.argv[4]
obsid = sys.argv[5]
homedir = sys.argv[6]
straylight_region = sys.argv[7]
reference_detector = int(sys.argv[8])
#x_c = float(sys.argv[6])


offx, offy = 0,0

def checkfile(fi):
  if os.path.isfile(fi) == False:
    return "BAD"
  else:
    return "GOOD"


data = os.path.join(homedir,data)

if checkfile(data) == "BAD":
  print('Data file not found, please check file path')
  sys.exit()

# NEED: crab mask, excl mask, Data

#homedir = '/uufs/chpc.utah.edu/common/home/astro/wik/NuSTAR/CRAB/'+obsid+'/'

######### HERE are the stray light masks: This is what needs to be updated. ###########
# The crab_mask is the mask that isolates the crab stray light, while
# the trans_mask are the edges of that mask and other areas that are unwanted due
# to blurred edges or other factors that may bias the measurement. For Renee, this may
# be a good replacement for the excl.reg file that represents any sources 

# This is a masking program that creates an 2-D array that masks active pixels from regions
# the h and w are the height and width of the array itself (default to 360x360), and the
# center is passed as coordinate of the radial center in the same order as the reg file
# reports.


##### Shape definitions ###############
# Circular
def create_circular_mask(h, w, center, radius):

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


# Polygon
def create_poly_mask(h,w,indicies):
    coords = []
    for i in range(0,len(indicies),2):
        pair = (indicies[i],indicies[i+1])
        coords.append(pair)
    x,y = np.meshgrid(np.arange(h),np.arange(w))
    x,y = x.flatten(),y.flatten()
    points = np.vstack((x,y)).T
    pat = Path(coords)
    grid = pat.contains_points(points)
    grid = grid.reshape((h,w))
    return grid


def create_ellipse_mask(h,w,centx,centy,dx,dy,rot):
    Y,X = np.ogrid[:h,:w]
    xp = (X-centx)*np.cos(rot)+(Y-centy)*np.sin(rot)
    yp = -(X-centx)*np.sin(rot)+(Y-centy)*np.cos(rot)
    grid = (xp/dx)**2+(yp/dy)**2 <= 1
    return grid


def get_box_coords(centx,centy,width,height,rot):
    # NOTE: This can be simplified with a simple rotation matrix, this is written out for debug
    rtx = centx + ((width/2) * np.cos(rot)) - ((height/2) * np.sin(rot))
    rty = centy + ((width/2) * np.sin(rot)) + ((height/2) * np.cos(rot))

    ltx = centx - ((width/2) * np.cos(rot)) - ((height/2) * np.sin(rot))
    lty = centy - ((width/2) * np.sin(rot)) + ((height/2) * np.cos(rot))

    lbx = centx - ((width/2) * np.cos(rot)) + ((height/2) * np.sin(rot))
    lby = centy - ((width/2) * np.sin(rot)) - ((height/2) * np.cos(rot))

    rbx = centx + ((width/2) * np.cos(rot)) + ((height/2) * np.sin(rot))
    rby = centy + ((width/2) * np.sin(rot)) - ((height/2) * np.cos(rot))
    
    coords = [rtx,rty,ltx,lty,lbx,lby,rbx,rby]
    return coords


#########################################
# These are fitting parameters for the estimated det norm values established 
# Through a stacked fitting of ~ 18Ms per telescope of filtered Occulted data
# When NuSTAR was within Earth's shadow.

aparams = np.array([[3.0344445961174137e-06, 0.49095924685628406, 3.2966298448587816e-08], \
    [2.8062974888119767e-06, 0.5112983664220534, 3.106952625755419e-08], \
    [4.400430023485417e-06, 0.5662638314570315, 3.859663516748116e-08], \
    [4.0619837918260075e-06, 0.532851541398607, 4.4697953111220014e-08]])
bparams = np.array([[3.1972692528864764e-06, 0.4536872824606454, 3.709356590195694e-08], \
    [5.6464991980026e-06, 0.5882584891219518, 4.2290117205190606e-08], \
    [4.543239243810085e-06, 0.5256260449949212, 4.214115020806879e-08], \
    [5.652738399347895e-06, 0.5624942207341803, 4.407630693825892e-08]])








with open(os.path.join(homedir,straylight_region)) as f:
    lines = f.readlines()


lin = lines[3:]

crab_mask = np.zeros((360,360))

##
# Assumptions: The nature of the stray light is from the circular profile of the aperture stops,
# so only a circular mask is applied to the stray light region profile and there currently is
# no plan to expand this concept.

# For the trans mask, this is possible to have any geometric shape, The assumptions here is that 
# they will be submitted through a region file in the format of DS9 and reported in physical pixels
# for det1 coords.
##
# It should be noted that only circular masking is possible, other shapes will need to be accounted
# for if they are wanted (i.e., box, annulus, polygon, elipse (though this one may be difficult))

# STRAY LIGHT REGION 
for li in lin:
    temp_mask = np.zeros((360,360))
    shape,vals = li.split('(')
    #shape = vals[0]
    vals = vals.split(')')[0]
    if "circle" in shape:
        X,Y,rad = vals.split(',')
        mask = create_circular_mask(360,360,(float(X),float(Y)),float(rad))
        #crab_mask[mask] = 0
        temp_mask[mask] = 1
    if 'polygon' in shape:
        vals = vals.split(',')
        vals = [float(i) for i in vals]
        mask = create_poly_mask(360,360,vals)
       #crab_mask[~mask] = 0
        temp_mask[mask] = 1
    if 'box' in shape:
        # Box is given as X,Y, width, height, rotation
        # create_box_mask will take it in this order
        centx,centy, width, height, rot = vals.split(',')
        val_2 = get_box_coords(float(centx),float(centy),float(width),float(height),float(rot))
        mask = create_poly_mask(360,360,val_2)
        temp_mask[mask] = 1
    if 'ellipse' in shape:
        X,Y,radx, rady, rotation = vals.split(',')
        mask = create_ellipse_mask(360,360,float(X),float(Y),float(radx),float(rady),float(rotation))
       #crab_mask[~mask] = 0
        temp_mask[mask] = 1

    crab_mask += temp_mask

crab_mask[crab_mask > 0] = 1

#with fits.open(homedir+'event_cl/crab'+detp+'.fits') as hdul:
#  crab_mask = hdul[0].data

# check for trans_mask region submission
if len(sys.argv) > 9:
    excl = sys.argv[9]

    with open(os.path.join(homedir,excl)) as f:
        lines = f.readlines()

    lin = lines[3:]
    trans_mask = np.ones((360,360))

    for li in lin:
        temp_mask = np.ones((360,360))
        shape, vals = li.split('(')
        #shape = vals[0]
        vals = vals.split(')')[0]
        #X,Y,rad = vals.split(',')
        # Here would be the place to enter other shapes into the masking function
        # The shapes are circle, polygon, box, and ellipse
        if 'circle' in shape:
            X,Y,rad = vals.split(',')
            mask = create_circular_mask(360,360,(float(X),float(Y)),float(rad))
            temp_mask[mask] = 0
        if 'polygon' in shape:
            vals = vals.split(',')
            vals = [float(i) for i in vals]
            mask = create_poly_mask(360,360,vals)
            temp_mask[mask] = 0
        if 'box' in shape:
            centx, centy, width, height, rot = vals.split(',')
            val_2 = get_box_coords(float(centx), float(centy), float(width), float(height), float(rot))
            mask = create_poly_mask(360,360,val_2)
            temp_mask[mask] = 0
        if 'ellipse' in shape:
            X,Y,radx,rady,rotation = vals.split(',')
            mask = create_ellipse_mask(360,360,float(X),float(Y),float(radx),float(rady),float(rotation))
            temp_mask[mask] = 0
        trans_mask *= temp_mask


else: 
    trans_mask = np.ones((360,360))

#with fits.open(homedir+'event_cl/mask'+detp+'.fits') as hdul:
#  trans_mask = hdul[0].data


##################
# files to use for counts:  pixmapA/B in nuskybgd auxil
#                           edgemaskA/B in auxil
#                           

###################### local files ################################
nusky_dir = '/uufs/astro.utah.edu/common/home/u1019304/my_idl/nuskybgd/auxil/'
edge = '/uufs/astro.utah.edu/common/home/u1019304/temp/fullmask{}_final.fits'.format(detp)
#edge = '/uufs/astro.utah.edu/common/home/u1019304/NuSTAR/auxil/fullmask'+detp+'.fits'
#edgeB = '/uufs/astro.utah.edu/common/home/u1019304/NuSTAR/auxil/fullmaskB.fits'
#local_dir = '/uufs/astro.utah.edu/common/home/u1019304/NuSTAR/auxil/'
orig_data = os.path.join(homedir,'nu{}{}01_cl.evt'.format(obsid,detp))
# Load data
with fits.open(orig_data) as exp:
  Exposure = exp[0].header['EXPOSURE']

with fits.open('/uufs/astro.utah.edu/common/home/u1019304/temp/detmap{}.fits'.format(detp)) as hA:
  dA = hA[0].data

with fits.open(edge) as hA:
  edgeA = hA[0].data  # To make a variable mask, make this mask a fraction of 1, this counts for the variable exposure time in excl.reg obs.

with fits.open(data) as hA:
  datA = hA[0].data

with fits.open('{}det{}_det1.img'.format(nusky_dir,detp)) as gA:
  gradA = gA[0].data

with fits.open('/uufs/astro.utah.edu/common/home/u1019304/temp/pixmap{}.fits'.format(detp)) as pm:
  pixmap = pm[0].data  #check for new pix map image possibly make :)

#Load in the expmap. This should be the exp version of the data file.  Used on one line, ref: [jkl;] to find
#exp_map = data.replace('Data','Exp')
#with fits.open(exp_map) as hdul:
#  exp_mask = hdul[0].data

eA_M = np.ones((360,360))
#eA_M /= Exposure

eA = np.copy(edgeA)*trans_mask
eA_M = eA_M*eA

#x_c *= Exposure

################################

# fix the Aij, Bij arrays (note: both arrays still use variable Aij) 
# The central position of the gradient was fixed here by analyzing the "best fit"
# given by allowing this central postions to move. This position is degenerate 
# to the gradient of the straylight model but the given position returned the 
# the most stable position for a variety of tests.


if detp == 'A':
  Aij = gradA[(int(len(gradA)/2) - 180) - (5 + offy):(int(len(gradA)/2) + 180) - (5 + offy), (int(len(gradA)/2) - 180) + (8 + offx):(int(len(gradA)/2) + 180) + (8 + offx)]*eA
else:
  Aij = gradA[(int(len(gradA)/2) - 180) - (16 + offy):(int(len(gradA)/2) + 180) - (16 + offy), (int(len(gradA)/2) - 180) + (10 + offx):(int(len(gradA)/2) + 180) + (10 + offx)]*eA

####
# Old version:
#Aij = gradA[len(gradA)/2 - 180:len(gradA)/2 + 180, len(gradA)/2 - 180:len(gradA)/2 + 180]*eA
####


# The current value of Aij is not conducive to the fitting program so it needs to be decreased 

Aij *= 0.0001
# So, for each value where the mask is not 0, we will run the fuction and minimize to that value.

dA += 1

vA = dA*eA

##################
# Counts of Aij for A 

a0 = np.zeros((360,360))
a1 = np.zeros((360,360))
a2 = np.zeros((360,360))
a3 = np.zeros((360,360))
Cn = np.copy(crab_mask)*eA

#####
### Here for testing the new masking ####
#####
# The masks were returned within expectations and allow for fitting of the
# general shapes DS9 can create (circle, box, polygon, and ellipse). Any more
# complicated shape will not be able to be fit

#fits.writeto('Resultant_stray.fits',Cn)
#fits.writeto('Resultant_excl.fits',eA_M)
#test_result = Cn*eA_M
#fits.writeto('Resultant_result.fits',test_result)
#sys.exit()

### End Test ############################



a0[vA == 1]=1
a1[vA == 2]=1
a2[vA == 3]=1
a3[vA == 4]=1
# NEED TO GET ACTIVE PIXELS FOR CRAB ARRAY

##################

PA = np.copy(datA)*eA

#################
# This is where this script diverges from count_stat.py, we will bin in a 1-D array 
# First attempt: create a tuple of data values that I can append and if it meets the
# criteria, then it will be converted into a numpy array.

# Create a long 1-D array that will represent the gradiant bins
def get_bins():
    A_bins, count_bins = [], []
    count, Aij_count = 0, 0.
    kA = 0
    min_value = 1.0 # min value in each bin, if it's below this, it will combine bins
    max_bin_length = 100000 # maximum number of bins, if it's longer, we will up the min count value

    bin_map = np.zeros((360,360))  # This array will be used to create the image
    bin_idx = 0  # This value is used as a counter to map from the bins to the image

    det0_bins,det1_bins,det2_bins,det3_bins = [],[],[],[]
    d0count, d1count, d2count, d3count = 0,0,0,0
    crab_bins = []
    crab_count = 0

# Think about using ravel to flatten out the 2-d arrays.
    for i in range(360):
        for j in range(360):
            bin_map[j,i] = bin_idx
            count += PA[j,i]     
            Aij_count += Aij[j,i]*eA_M[j,i] 
            d0count += a0[j,i]*eA_M[j,i]
            d1count += a1[j,i]*eA_M[j,i]
            d2count += a2[j,i]*eA_M[j,i]
            d3count += a3[j,i]*eA_M[j,i]
            crab_count += Cn[j,i]*eA_M[j,i] # CRAB <-------------
            if count >= min_value:
                bin_idx += 1
                count_bins.append(count)
                A_bins.append(Aij_count)
                count,Aij_count = 0, 0.
                kA += 1
                det0_bins.append(d0count)
                det1_bins.append(d1count)
                det2_bins.append(d2count)
                det3_bins.append(d3count)
                crab_bins.append(crab_count) # CRAB <-------------
                d0count,d1count,d2count,d3count = 0,0,0,0
                crab_count = 0 # CRAB <----------------
            if (i == 359) and (j == 359) and (count < min_value):  # Fix aij issue, not a value to put into the value
      #Aij_count += A_bins[-1:]  # If the last value in the array below the min count, it will
                A_bins[-1:] += Aij_count   # be added to the last bin. This may want to be redone
                det3_bins[-1:] += d3count  # it is also a det3 bin at this point
                crab_bins[-1:] += crab_count  # CRAB <---------------
            if len(A_bins) > max_bin_length:
                i, j, count, Aij_count, d0count, d1count, d2count, d3count, crab_count = 0,0,0, 0., 0,0,0,0,0 # CRAB
                count_bins, A_bins, det0_bins, det1_bins, det2_bins, det3_bins, crab_bins = [],[],[],[],[],[],[] # CRAB
                min_value += 10      
                kA = 0
    return kA, det0_bins, det1_bins, det2_bins, det3_bins, A_bins, count_bins, min_value, bin_map, crab_bins
################
# Need to find a way to seperate the bins into det binned arrays
# Think I found a way, it looks at the masks for each det and if 
# it exists in thier array, the count goes into there.

kA, det0_bins, det1_bins, det2_bins, det3_bins, A_bins, count_bins, min_value, bin_map, Cn_bins = get_bins()


if len(A_bins) != kA:
  print('inconsistant bin length')

if len(A_bins) != len(count_bins):
  print('A and count bins are of unequal length')

# Check for zeros in the bins

if A_bins.count(0) > 0:
  print("There are zeros in the bins")
  print(A_bins.count(0))
  print(A_bins.count(2.0))
  sys.exit()

# make it into a numpy array

A_bins = np.asarray(A_bins)
count_bins = np.asarray(count_bins)
det0_bins = np.asarray(det0_bins)
det1_bins = np.asarray(det1_bins)
det2_bins = np.asarray(det2_bins)
det3_bins = np.asarray(det3_bins)
Cn_bins = np.asarray(Cn_bins)

######################
# Function to be passed to the optimization function
######################
def fn(params):
#  x,y,z,w,t = params
  x = params['aCXB']
  y = params['det0']
#  z = params['det1']
#  w = params['det2']
#  t = params['det3']
  u = params['crab']
#  global A_bins,det0_bins,det1_bins,det2_bins,det3_bins 
  mod = (x*A_bins) + (y*det0_rat*det0_bins) + (y*det1_rat*det1_bins) + (y*det2_rat*det2_bins) + (y*det3_rat*det3_bins) + (u*Cn_bins)
  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
  return (2 * np.sum(numerA))

# This is to find the error values around the points, this is done
# by holding the aCXB norm constant (x_cerror) while allowing the other
# norms to be passed through the minimization function
# #### moved below to the error finding area


####################################

def detfn(param, detbin):
  mod = param*detbin
  idx = np.where((count_bins*detbin) != 0)
  numer = mod[idx] - count_bins[idx] + count_bins[idx]*(np.log(count_bins[idx])-np.log(mod[idx]))
  return(2 * np.sum(numer))

def f(X,x,y,z,w,t):
  x0 = X[:,0]
  x1 = X[:,1]
  x2 = X[:,2]
  x3 = X[:,3]
  x4 = X[:,4] 
  return (x*x0) + (y*x1) + (z*x2) + (w*x3) + (t*x4)

def residual(p):
  return (2 * np.sum(p-count_bins+count_bins*(np.log(count_bins)-np.log(p))))

def lmmin(pars,data=None):
  mod = pars['aCXB'].value*det0_rat*A_bins + pars['det0'].value*det0_bins + pars['det0'].value*det1_rat*det1_bins + pars['det0'].value*det2_rat*det2_bins + pars['det0'].value*det3_rat*det3_bins + pars['crab'].value*Cn_bins
  if data is None:
    return mod
  return mod - data



def getDetInit(tele,det,lener,hener):
    # This needs to be updated to account for energy gaps that may not represent the file
    # for now this is a work around
    #if tele == 'A':
    #    return quad(model_func,lener,hener,args=(aparams[det][0],aparams[det][1],aparams[det][2]))[0]
    #if tele == 'B':
    #    return quad(model_func,lener,hener,args=(bparams[det][0],bparams[det][1],bparams[det][2]))[0]
# These values are the distilled values from the ~18Ms fit of occulted data

    if tele == 'A':
        if det == 0:
            return 4.688213725012213e-08
        if det == 1:
            return 4.271946429615839e-08
        if det == 2:
            return 4.584367159703845e-08
        if det == 3:
            return 5.5204683992921586e-08
    if tele == 'B':
        if det == 0:
            return 4.99080884080508e-08
        if det == 1:
            return 5.029251032923879e-08
        if det == 2:
            return 5.370525725840809e-08
        if det == 3:
            return 5.4013373238992535e-08


def model_func(t,A,K,C):
    return A * np.exp(-K*t)+C


#    values = np.genfromtxt('/uufs/astro.utah.edu/common/home/u1019304/detvals/{}det{}normvals.txt'.format(tele,detector))
#    idx_low = np.where(values[:,0]==float(lener))[0]
#    idx_high = np.where(values[:,1]==float(hener))[0]
#    if idx_low != idx_high:
#        closest_idx = (np.abs(values[:,0]-float(lener))).argmin()
#        idx = copy.copy(closest_idx)
#    else:
#        idx = copy.copy(idx_low[0])
#    return values[idx][2]



#############################################################   

with open('/uufs/astro.utah.edu/common/home/u1019304/temp/normmeanvals.txt', 'r+') as f:
  lines = f.readlines()

#aCXB_initval, det0_initval, det1_initval, det2_initval, det3_initval = lines[0].split()
aCXB_initval = lines[0].split()[0]
det0_initval = getDetInit(detp, 0, float(lowe), float(highe))
det1_initval = getDetInit(detp, 1, float(lowe), float(highe))
det2_initval = getDetInit(detp, 2, float(lowe), float(highe))
det3_initval = getDetInit(detp, 3, float(lowe), float(highe))
reference_val = getDetInit(detp, reference_detector, float(lowe), float(highe))

aCXB_initval = float(aCXB_initval)
det0_initval = float(det0_initval)
det1_initval = float(det1_initval)
det2_initval = float(det2_initval)
det3_initval = float(det3_initval)
reference_val = float(reference_val)

aCXB_initval *= Exposure; aCXB_initval *= (float(highe)-float(lowe))
det0_initval *= Exposure; det0_initval *= (float(highe)-float(lowe))
det1_initval *= Exposure; det1_initval *= (float(highe)-float(lowe))
det2_initval *= Exposure; det2_initval *= (float(highe)-float(lowe))
det3_initval *= Exposure; det3_initval *= (float(highe)-float(lowe))
reference_val *= Exposure; reference_val *= (float(highe)-float(lowe))

# Here needs to be added the code to identify which det will be used as reference.
# What I need to solve: set the 'det0' param to the reference value
#                       write out the norm values with the solved for values
#                       errors


#det1_initval /= det0_initval
#det2_initval /= det0_initval
#det3_initval /= det0_initval

det0_rat = det0_initval/reference_val
det1_rat = det1_initval/reference_val
det2_rat = det2_initval/reference_val
det3_rat = det3_initval/reference_val

fit_params = Parameters()
fit_params.add('aCXB', value=aCXB_initval, min=0.0)
fit_params.add('det0', value=reference_val, min=(reference_val*0.0001))#, max=(reference_val*1.15))
#fit_params.add('det1', value=det1_initval, min=(det1_initval*0.0001))
#fit_params.add('det2', value=det2_initval, min=(det2_initval*0.0001))
#fit_params.add('det3', value=det3_initval, min=(det3_initval*0.0001))
fit_params.add('crab', value=(det3_initval*20), min=0.0)


 #!!!!!!!!!!!!!!!!!#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
resA = minimize(fn, fit_params, method = 'nelder', tol=1e-15)
 #!!!!!!!!!!!!!!!!!################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#print(fit_params['det0'].value, fit_params['det1'].value, fit_params['det2'].value, fit_params['det3'].value, fit_params['crab'].value)

#model = x_c*Aij+fit_params['det0'].value*a0+fit_params['det1'].value*a1+fit_params['det2'].value*a2+fit_params['det3'].value*a3+fit_params['crab'].value*Cn  # THIS NEEDS TO BE MULTI BY THE EXP IMAGE

cstat = fn(fit_params)
#resid = model - PA

yaxis = np.arange(0,len(count_bins))
resid_x = lmmin(fit_params, count_bins)


#bin_file = homedir+'BINS'+detp+'_'+str(lowe)+'_'+str(highe)+'_01_crab.fits'
#fits.writeto(bin_file, bin_map)
#with fits.open(bin_file, mode = 'update') as hdul:
#  hdul[0].header['COMMENT'] = 'bins image for energy of '+str(lowe)+' to '+str(highe)
#  hdul[0].header['COMMENT'] = 'A total number of bins is: '+str(kA) 

resA = minimize(fn, fit_params, method = 'nelder', tol=1e-15)
#################################
### Make a fits file of the model#
#################################

aCX = fit_params['aCXB'].value
d0 = fit_params['det0'].value*det0_rat
d1 = fit_params['det0'].value*det1_rat
d2 = fit_params['det0'].value*det2_rat
d3 = fit_params['det0'].value*det3_rat
stray_light = fit_params['crab'].value

scale = 1.0/0.31865/0.012096/0.012096 # The second value is from 1e4*3.1865e-5 due to the scaling cxb gradient file (see line284)
aCX *= scale/Exposure # fit norm to units of cts/s/cm^2/deg^2 over the energy range given
                        # The det norm values are reported as is, these can be scaled to get more
                        # information, however, unless needed, these values are just there for
                        # guidence.

# This aCXB value can then be used to populate a pha file columns


with open(os.path.join(homedir,'{}_params.txt'.format(detp)), 'a+') as O:
    O.write('{} {} {} {} {} {}{}'.format(aCX,d0,d1,d2,d3,stray_light,'\n'))
    


# From here is the error program, to use it needs to be uncommented and fixed a bit.
# HOWEVER, given that this program will be used to only measure a single observation, it is NOT adivised that this error analysis be used. The log liklihood used to fit the data is based on the Cash statistic which is Poissonian by nature and only when the data points are >> then those found within a single observation does a 1-sigma error make sense. If errors are needed, a new routine should be used. If you do have a large collection of data, then this brute force method has shown to be robust and reliable, but slow. 



##def gaussian(x, a, b, c):
##    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))


############### From here, it is a C-stat fit program ########### should move to own program
# Here we are going to fix the aCXB value (well, really we should fix it from the beginning).
# This allows the dets and the crab value to be free
############ FUNCTIONS ##############################
##def parabola(x,a,b,c):
##       return a*x**2+b*x+c

##def solu(cs,a,b,cv):
##  c = cv-cs-1
##  pos = (-b+np.sqrt(b**2-4*a*c))/(2*a)
##  neg = (-b-np.sqrt(b**2-4*a*c))/(2*a)
##  return max(pos,neg)

##def sold(cs,a,b,cv):
##  c = cv-cs-1
##  pos = (-b+np.sqrt(b**2-4*a*c))/(2*a)
##  neg = (-b-np.sqrt(b**2-4*a*c))/(2*a)
##  return min(pos,neg)

##def fit_Cstat(x,y,z,w,t,u):
#  global A_bins,det0_bins,det1_bins,det2_bins,det3_bins 
##  mod = (x*A_bins) + (y*det0_bins) + (z*det1_bins) + (w*det2_bins) + (t*det3_bins) + (u*Cn_bins)
##  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
##  return (2 * np.sum(numerA))

##def fn_cerrorC(params):
##  y = params['det0']
##  z = params['det1']
##  w = params['det2']
##  t = params['det3']
##  mod = (x_c*A_bins) + (y*det0_bins) + (z*det1_bins) + (w*det2_bins) + (t*det3_bins) + (u_c*Cn_bins)
##  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
##  return (2 * np.sum(numerA))

##def fn_cerror0(params):
##  z = params['det1']
##  w = params['det2']
##  t = params['det3']
##  u = params['crab']
##  mod = (x_c*A_bins) + (y_c*det0_bins) + (z*det1_bins) + (w*det2_bins) + (t*det3_bins) + (u*Cn_bins)
##  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
##  return (2 * np.sum(numerA))

##def fn_cerror1(params):
##  y = params['det0']
##  w = params['det2']
##  t = params['det3']
##  u = params['crab']
##  mod = (x_c*A_bins) + (y*det0_bins) + (z_c*det1_bins) + (w*det2_bins) + (t*det3_bins) + (u*Cn_bins)
##  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
##  return (2 * np.sum(numerA))

##def fn_cerror2(params):
##  y = params['det0']
##  z = params['det1']
##  t = params['det3']
##  u = params['crab']
##  mod = (x_c*A_bins) + (y*det0_bins) + (z*det1_bins) + (w_c*det2_bins) + (t*det3_bins) + (u*Cn_bins)
##  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
##  return (2 * np.sum(numerA))

##def fn_cerror3(params):
##  y = params['det0']
##  z = params['det1']
##  w = params['det2']
##  u = params['crab']
##  mod = (x_c*A_bins) + (y*det0_bins) + (z*det1_bins) + (w*det2_bins) + (t_c*det3_bins) + (u*Cn_bins)
##  numerA = mod - count_bins + count_bins*(np.log(count_bins)-np.log(mod))
##  return (2 * np.sum(numerA))

# The idea is this: We have standard parameters that are kept as THE values (standard). IF a new 
# lower cstat value is found, we set the standard to those values, reset everything, and start over.
# I know this is also done by the fit_params, but the name lets me keep straight what values are the 
# global values of error and which are the ones being adjusted to find bounds.

##standards = Parameters() # these will be the final reported on values
##standards.add('det0', value=fit_params['det0'].value)
##standards.add('det1', value=fit_params['det1'].value)
##standards.add('det2', value=fit_params['det2'].value)
##standards.add('det3', value=fit_params['det3'].value)
##standards.add('crab', value=fit_params['crab'].value)

##fit_params_errors = Parameters() # these will used to find the bounds
##fit_params_errors.add('det0', value=fit_params['det0'].value)
##fit_params_errors.add('det1', value=fit_params['det1'].value)
##fit_params_errors.add('det2', value=fit_params['det2'].value)
##fit_params_errors.add('det3', value=fit_params['det3'].value)

##def resetParams(val1, val2, val3, val4, val5):
##  fit_params_errors['det0'].set(value = val1)
##  fit_params_errors['det1'].set(value = val2)
##  fit_params_errors['det2'].set(value = val3)
##  fit_params_errors['det3'].set(value = val4)
##  fit_params_errors['crab'].set(value = val5)

##def setParameters(name1, var1, name2, var2, name3, var3, name4, var4):
##  global fit_params
##  fit_params_errors = None
##  fit_params_errors = Parameters()
##  fit_params_errors.add(name1, value=var1)
##  fit_params_errors.add(name2, value=var2)
##  fit_params_errors.add(name3, value=var3)
##  fit_params_errors.add(name4, value=var4)

##def resetStandards(val1,val2,val3,val4,val5):
##  standards['det0'].set(value = val1)
##  standards['det1'].set(value = val2)
##  standards['det2'].set(value = val3)
##  standards['det3'].set(value = val4)
##  standards['crab'].set(value = val5)


##detarray = ['det0','det1','det2','det3','crab']
##y_c = standards['det0'].value; z_c = standards['det1'].value; w_c = standards['det2'].value; t_c = standards['det3'].value; u_c = standards['crab'].value
##newcstat = np.copy(cstat)
##m_cstat = np.copy(cstat)
##mult_val = 10.0

########## MAIN ERROR FUNCTION: ####################
##def doCstatRun():
##  global y_c,z_c,t_c,w_c,x_c,u_c,newcstat, m_cstat, newdetvals, fit_params_errors
##  xvals, x1, x2 = [],[],[]
##  yvals, y1, y2 = [],[],[]
##  newdetvals = []
  #for i in [0,1,2,3,4]:
##  for i in [4]:
##    if i == 0:
##      setParameters('det1', standards['det1'].value, 'det2', standards['det2'].value, 'det3', standards['det3'].value, 'crab', standards['crab'].value)
##    if i == 1:
##      setParameters('det0', standards['det0'].value, 'det2', standards['det2'].value, 'det3', standards['det3'].value, 'crab', standards['crab'].value)
##    if i == 2:
##      setParameters('det0', standards['det0'].value, 'det1', standards['det1'].value, 'det3', standards['det3'].value, 'crab', standards['crab'].value)
##    if i == 3:
##      setParameters('det0', standards['det0'].value, 'det1', standards['det1'].value, 'det2', standards['det2'].value, 'crab', standards['crab'].value)
##    if i == 4:
##      setParameters('det0', standards['det0'].value, 'det1', standards['det1'].value, 'det2', standards['det2'].value, 'det3', standards['det3'].value)
    
##    for direction in ['high','low']:
##      if i == 0:
##        xvals.append(float(y_c))
##        if direction == 'high': x1.append(float(y_c))
##        if direction == 'low': x2.append(float(y_c))
##        st = 10**(math.floor(math.log(y_c,10))-4)
##      if i == 1:
##        xvals.append(float(z_c))
##        if direction == 'high': x1.append(float(z_c))
##        if direction == 'low': x2.append(float(z_c))
##        st = 10**(math.floor(math.log(z_c,10))-4)
##      if i == 2:
##        xvals.append(float(w_c))
##        if direction == 'high': x1.append(float(w_c))
##        if direction == 'low': x2.append(float(w_c))
##        st = 10**(math.floor(math.log(w_c,10))-4)
##      if i == 3:
##        xvals.append(float(t_c))
##        if direction == 'high': x1.append(float(t_c))
##        if direction == 'low': x2.append(float(t_c))
##        st = 10**(math.floor(math.log(t_c,10))-4)
##      if i == 4:
##        xvals.append(float(u_c))
##        if direction == 'high': x1.append(float(u_c))
##        if direction == 'low': x2.append(float(u_c))
##        st = 10**(math.floor(math.log(u_c,10))-3)
##      yvals.append(m_cstat)
##      if direction == 'high': step = st; fi = obsid+'/high_param_errors.txt'; y1.append(m_cstat)
##      elif direction == 'low': step = -st; fi = obsid+'/low_param_errors.txt'; y2.append(m_cstat)
##      while newcstat < 10.0+m_cstat:
##        if i == 0: # index for which det we are in and find cstat values with those parameters
##          y_c += step
##          resA = minimize(fn_cerror0,fit_params_errors, method='nelder', tol=1e-15)
##          newcstat = fit_Cstat(x_c,y_c,fit_params_errors['det1'].value,fit_params_errors['det2'].value,fit_params_errors['det3'].value,fit_params_errors['crab'].value)
##        elif i == 1:
##          z_c += step
##          resA = minimize(fn_cerror1,fit_params_errors, method='nelder', tol=1e-15)
##          newcstat = fit_Cstat(x_c,fit_params_errors['det0'].value,z_c,fit_params_errors['det2'].value,fit_params_errors['det3'].value,fit_params_errors['crab'].value)
##        elif i == 2:
##          w_c += step
##          resA = minimize(fn_cerror2,fit_params_errors, method='nelder', tol=1e-15)
##          newcstat = fit_Cstat(x_c,fit_params_errors['det0'].value,fit_params_errors['det1'].value,w_c,fit_params_errors['det3'].value,fit_params_errors['crab'].value)
##        elif i == 3:
##          t_c += step
##          resA = minimize(fn_cerror3,fit_params_errors, method='nelder', tol=1e-15)
##          newcstat = fit_Cstat(x_c,fit_params_errors['det0'].value,fit_params_errors['det1'].value,fit_params_errors['det2'].value,t_c,fit_params_errors['crab'].value)
##        elif i == 4:
##          u_c += step
##          resA = minimize(fn_cerrorC,fit_params_errors, method='nelder', tol=1e-13)
##          newcstat = fit_Cstat(x_c,fit_params_errors['det0'].value,fit_params_errors['det1'].value,fit_params_errors['det2'].value,fit_params_errors['det3'].value,u_c)
##        if newcstat < m_cstat:
##          print('NEW cstat VAL')
##          m_cstat = None
##          m_cstat = np.copy(newcstat)
##          if len(newdetvals) > 0: newdetvals = []
          # run newacxb val
##          if i == 0:
##            newdet0 = np.copy(y_c)
##            newdetvals.append(newdet0); newdetvals.append(fit_params_errors['det1'].value); newdetvals.append(fit_params_errors['det2'].value); newdetvals.append(fit_params_errors['det3'].value); newdetvals.append(fit_params_errors['crab'].value)
##            xvals,yvals,x1,y1,x2,y2 =  [],[],[],[],[],[]
##          if i == 1:
##            newdet1 = np.copy(z_c)
##            newdetvals.append(fit_params_errors['det0'].value); newdetvals.append(newdet1); newdetvals.append(fit_params_errors['det2'].value); newdetvals.append(fit_params_errors['det3'].value); newdetvals.append(fit_params_errors['crab'].value)
##            xvals,yvals,x1,y1,x2,y2 =  [],[],[],[],[],[]
##          if i == 2:
##            newdet2 = np.copy(w_c)
##            newdetvals.append(fit_params_errors['det0'].value); newdetvals.append(fit_params_errors['det1'].value); newdetvals.append(newdet2); newdetvals.append(fit_params_errors['det3'].value); newdetvals.append(fit_params_errors['crab'].value)
##            xvals,yvals,x1,y1,x2,y2 =  [],[],[],[],[],[]
##          if i == 3:
##            newdet3 = np.copy(t_c)
##            newdetvals.append(fit_params_errors['det0'].value); newdetvals.append(fit_params_errors['det1'].value); newdetvals.append(fit_params_errors['det2'].value); newdetvals.append(newdet3); newdetvals.append(fit_params_errors['crab'].value)
##            xvals,yvals,x1,y1,x2,y2 =  [],[],[],[],[],[]
##          if i == 4:
##            newdet4 = np.copy(u_c)
##            newdetvals.append(fit_params_errors['det0'].value); newdetvals.append(fit_params_errors['det1'].value); newdetvals.append(fit_params_errors['det2'].value); newdetvals.append(fit_params_errors['det3'].value); newdetvals.append(newdet4)
##            xvals,yvals,x1,y1,x2,y2 =  [],[],[],[],[],[]
##          blah = open(obsid+'/high_param_errors.txt','w+')
##          blah.close()
##          blah = open(obsid+'/low_param_errors.txt','w+')
##          blah.close()
##          newcstat = np.copy(m_cstat)
##          resetStandards(newdetvals[0], newdetvals[1], newdetvals[2], newdetvals[3], newdetvals[4])
##          return False
        
##        if i == 0:
##          xvals.append(float(y_c))
##          if direction == 'high': x1.append(float(y_c))
##          if direction == 'low': x2.append(float(y_c))
##        if i == 1:
##          xvals.append(float(z_c))
##          if direction == 'high': x1.append(float(z_c))
##          if direction == 'low': x2.append(float(z_c))
##        if i == 2:
##          xvals.append(float(w_c))
##          if direction == 'high': x1.append(float(w_c))
##          if direction == 'low': x2.append(float(w_c))
##        if i == 3:
##          xvals.append(float(t_c))
##          if direction == 'high': x1.append(float(t_c))
##          if direction == 'low': x2.append(float(t_c))
##        if i == 4:
##          xvals.append(float(u_c))
##          if direction == 'high': x1.append(float(u_c))
##          if direction == 'low': x2.append(float(u_c))
##        yvals.append(float(newcstat))
##        if direction == 'high': y1.append(float(newcstat))
##        if direction == 'low': y2.append(float(newcstat))
        
##        with open(fi,'a+') as O:
##          if i == 0:
##            O.write(str(x_c)+' '+str(y_c)+' '+str(fit_params_errors['det1'].value)+' '+str(fit_params_errors['det2'].value)+' '+str(fit_params_errors['det3'].value)+' '+str(fit_params_errors['crab'].value)+'\n')
##          if i == 1:
##            O.write(str(x_c)+' '+str(fit_params_errors['det0'].value)+' '+str(z_c)+' '+str(fit_params_errors['det2'].value)+' '+str(fit_params_errors['det3'].value)+' '+str(fit_params_errors['crab'].value)+'\n')
##          if i == 2:
##            O.write(str(x_c)+' '+str(fit_params_errors['det0'].value)+' '+str(fit_params_errors['det1'].value)+' '+str(w_c)+' '+str(fit_params_errors['det3'].value)+' '+str(fit_params_errors['crab'].value)+'\n')
##          if i == 3:
##            O.write(str(x_c)+' '+str(fit_params_errors['det0'].value)+' '+str(fit_params_errors['det1'].value)+' '+str(fit_params_errors['det2'].value)+' '+str(t_c)+' '+str(fit_params_errors['crab'].value)+'\n')
##          if i == 4:
##            O.write(str(x_c)+' '+str(fit_params_errors['det0'].value)+' '+str(fit_params_errors['det1'].value)+' '+str(fit_params_errors['det2'].value)+' '+str(fit_params_errors['det3'].value)+' '+str(u_c)+'\n')


      #reset parameters for next run:
##      if i == 0:
##        setParameters('det1', standards['det1'].value, 'det2', standards['det2'].value, 'det3', standards['det3'].value, 'crab', standards['crab'].value)
##      if i == 1:
##        setParameters('det0', standards['det0'].value, 'det2', standards['det2'].value, 'det3', standards['det3'].value, 'crab', standards['crab'].value)
##      if i == 2:
##        setParameters('det0', standards['det0'].value, 'det1', standards['det1'].value, 'det3', standards['det3'].value, 'crab', standards['crab'].value)
##      if i == 3:
##        setParameters('det0', standards['det0'].value, 'det1', standards['det1'].value, 'det2', standards['det2'].value, 'crab', standards['crab'].value)
##      if i == 4:
##        setParameters('det0', standards['det0'].value, 'det1', standards['det1'].value, 'det2', standards['det2'].value, 'det3', standards['det3'].value)
##      newcstat = np.copy(m_cstat)

##    fi_high = obsid+'/high_param_errors.txt'
##    fi_low = obsid+'/low_param_errors.txt'
##    fwrite = obsid+'/'+detp+'_'+detarray[i]+'_'+str(lowe)+'_'+str(highe)+'_params_errors.txt'

##    with open(fi_high) as fh:
##        lines = fh.readlines()
##        with open(fwrite,'w') as fw:
##                fw.write(str(lines)+'\n')
##    with open(fi_low) as fl:
##        lines = fl.readlines()
##        with open(fwrite,'a') as fw:
##                fw.write(str(lines)+'\n')

##    os.system('rm '+fi_high+' '+fi_low)
#need to write out values to parameter file
    # find the one sig errors
##    p,c = curve_fit(parabola,xvals,yvals)
##    pars1, cov1 = opt.curve_fit(parabola,x1,y1,p)
##    pars2, cov2 = opt.curve_fit(parabola,x2,y2,p)

##    sigup = solu(m_cstat,*pars1)
##    sigdown = sold(m_cstat,*pars2)

##    with open(obsid+'/'+detp+'_'+detarray[i]+"_params.txt",'a+') as O:
##      if i == 0:
##        O.write(lowe+' '+highe+' '+str(standards['det0'].value)+' '+str(sigup)+' '+str(sigdown)+'\n')
##      if i == 1:
##        O.write(lowe+' '+highe+' '+str(standards['det1'].value)+' '+str(sigup)+' '+str(sigdown)+'\n')
##      if i == 2:
##        O.write(lowe+' '+highe+' '+str(standards['det2'].value)+' '+str(sigup)+' '+str(sigdown)+'\n')
##      if i == 3:
##        O.write(lowe+' '+highe+' '+str(standards['det3'].value)+' '+str(sigup)+' '+str(sigdown)+'\n')
##      if i == 4:
##        O.write(lowe+' '+highe+' '+str(standards['crab'].value)+' '+str(sigup)+' '+str(sigdown)+'\n')



  #reset variables:
##    xvals, x1, x2 = [],[],[]
##    yvals, y1, y2 = [],[],[]
##    newcstat = np.copy(m_cstat)
##    y_c = standards['det0'].value; z_c = standards['det1'].value; w_c = standards['det2'].value; t_c = standards['det3'].value; u_c = standards['crab'].value
##  return True

##def clearFile(fi):
##  with open(fi, "r") as f:
##    lines = f.readlines()
##  with open(fi, "w") as f:
##    for line in lines:
##      if lowe in line:
##        if highe in line:
##          continue
##      f.write(line)

##while True:
##  if doCstatRun():
##    break
  # if I get a return of False, I need to clear any existing files that may have been written to:
##  for de in ['crab']:
##    fi_one = obsid+'/'+detp+'_'+de+'_params.txt'
##    fi_two = obsid+'/'+detp+'_'+de+'_params_errors.txt'
##    if os.path.isfile(fi_one):
##      clearFile(fi_one)
##    if os.path.isfile(fi_two):
##      clearFile(fi_two)

##print('finished')

##model = x_c*Aij+standards['det0'].value*a0+standards['det1'].value*a1+standards['det2'].value*a2+standards['det3'].value*a3+standards['crab'].value*Cn

##fii = homedir+'Model'+detp+'_'+str(lowe)+'_'+str(highe)+'_01_crab.fits'
##fits.writeto(fii, model)
##with fits.open(fii, mode='update') as hdul:
##  hdul[0].header['ACXBNORM'] = str(x_c)
##  hdul[0].header['DET0NORM'] = str(standards['det0'].value)
##  hdul[0].header['DET1NORM'] = str(standards['det1'].value)
##  hdul[0].header['DET2NORM'] = str(standards['det2'].value)
##  hdul[0].header['DET3NORM'] = str(standards['det3'].value)
##  hdul[0].header['CRABNORM'] = str(standards['crab'].value)
##  hdul[0].header['COMMENT'] = 'Model for '+detp+' Crab '
##  hdul.flush()

##resid = model - PA

##idx_for_hist = np.where(eA > 0)

##n, bins, patches = plt.hist(np.ravel(np.around(resid[idx_for_hist],decimals=2)), bins='auto', normed=True, facecolor='green')
##plt.title("Binned distribution of residual counts in for Crab")
##plt.xlabel("Counts per bin")
##plt.ylabel("# of bins")
##plt.savefig(homedir+"/hist_"+detp+"_"+str(lowe)+"_"+str(highe)+"_crab.png")
##plt.close()

##plt.matshow(model)
##plt.title("Background model for crab")
##plt.savefig(homedir+"/model_image_"+detp+"_"+str(lowe)+"_"+str(highe)+"_crab.png")
##plt.close()

##plt.matshow(resid)
##plt.title("Residual map for Crab")
##plt.savefig(homedir+"/resid_image_"+detp+"_"+str(lowe)+"_"+str(highe)+"_crab.png")
##plt.close()

#################################





####################   OLD VERSION: ################################
#midval = fit_params['aCXB']
#xvals,x1,x2 = [],[],[]
#yvals,y1,y2 = [],[],[]
#x_c = fit_params['aCXB']
#y_c = fit_params['det0']
#z_c = fit_params['det1']
#w_c = fit_params['det2']
#t_c = fit_params['det3']
#u_c = fit_params['crab'] 
#xvals.append(x_c.value)
#yvals.append(cstat)
#newcstat = np.copy(cstat)
#m_cstat = np.copy(cstat)
#mult_val = 10.0 
#onesigup = 0
#onesigdown = 0
#newdetvals = []
########### While loop #################
#while newcstat < mult_val+m_cstat:
#  x,y,z,w,t = params
#  x_c += 0.0001
#  fit_params_errors = Parameters()
#  fit_params_errors.add('det0', value=fit_params['det0'].value, min=0.0)
#  fit_params_errors.add('det1', value=fit_params['det1'].value, min=0.0)
#  fit_params_errors.add('det2', value=fit_params['det2'].value, min=0.0)
#  fit_params_errors.add('det3', value=fit_params['det3'].value, min=0.0)
#  resA = minimize(fn_cerror, fit_params_errors, method = 'nelder', tol=1e-15)
#  newcstat = fitcstat(x_c,fit_params_errors['det0'].value,fit_params_errors['det1'].value,fit_params_errors['det2'].value,fit_params_errors['det3'].value) 
  # Need to build in an exception to catch lower cstat values if they exist -- reset cstat value to new min
  # Do i need to reset the parameter values before each run? ##############################
#  if newcstat < m_cstat: 
#	print('exception exisits '+data+' high')
#	m_cstat = None
#	m_cstat = newcstat
 #       if len(newdetvals) > 0: newdetvals = []
#	newdetvals.append(fit_params_errors['det0'].value)
#	newdetvals.append(fit_params_errors['det1'].value)
#	newdetvals.append(fit_params_errors['det2'].value)
#	newdetvals.append(fit_params_errors['det3'].value)
#	newaCXB = x_c
#	xvals,yvals,x1,y1 =  [],[],[],[]
#	xvals.append(x_c.value)
#	yvals.append(m_cstat)
#	blah = open(homedir+'/'+detp+"_high_params_errors.txt",'w+')
 #	blah.close()
#	continue
 # xvals.append(float(x_c))
 # yvals.append(float(newcstat))
#  x1.append(float(x_c))
#  y1.append(float(newcstat))
#  with open(homedir+'/'+detp+"_high_params_errors.txt",'a+') as O:
#    O.write(str(x_c)+' '+str(fit_params_errors['det0'].value)+' '+str(fit_params_errors['det1'].value)+' '+str(fit_params_errors['det2'].value)+' '+str(fit_params_errors['det3'].value)+'\n')
#  fit_params_errors=None

#if len(newdetvals) > 0:
#	resetParams(newdetvals[0],newdetvals[1],newdetvals[2],newdetvals[3])
#	x_c = None
#	x_c = newaCXB
#	newcstat = np.copy(m_cstat)
#else:
#	fit_params_errors = None
#	fit_params_errors = Parameters()
#	fit_params_errors.add('det0', value=fit_params['det0'].value, min=0.0)
#	fit_params_errors.add('det1', value=fit_params['det1'].value, min=0.0)
#	fit_params_errors.add('det2', value=fit_params['det2'].value, min=0.0)
#	fit_params_errors.add('det3', value=fit_params['det3'].value, min=0.0)
#	newcstat=np.copy(cstat)
#	x_c = fit_params['aCXB']
#xvals_2 = xvals[:]
#yvals_2 = yvals[:]

#while newcstat < mult_val+m_cstat:
#  x_c -= 0.0001
#  fit_params_errors = Parameters()
#  fit_params_errors.add('det0', value=fit_params['det0'].value, min=0.0)
#  fit_params_errors.add('det1', value=fit_params['det1'].value, min=0.0)
#  fit_params_errors.add('det2', value=fit_params['det2'].value, min=0.0)
#  fit_params_errors.add('det3', value=fit_params['det3'].value, min=0.0)
#  resA = minimize(fn_cerror, fit_params_errors, method = 'nelder', tol=1e-15)
#  newcstat = fitcstat(x_c,fit_params_errors['det0'].value,fit_params_errors['det1'].value,fit_params_errors['det2'].value,fit_params_errors['det3'].value)
  # Need to build in an exception to catch lower cstat values if they exist
#  if newcstat < m_cstat: 
#	print('exception exisits '+data+' low')
#	m_cstat = None
#        m_cstat = newcstat
#        if len(newdetvals) > 0: newdetvals = []
#        newdetvals.append(fit_params_errors['det0'].value)
#        newdetvals.append(fit_params_errors['det1'].value)
#        newdetvals.append(fit_params_errors['det2'].value)
#        newdetvals.append(fit_params_errors['det3'].value)
#        newaCXB = x_c
#        x2,y2 =  [],[]
#        xvals = xvals_2[:]
#        yvals = yvals_2[:]
#        blah = open(homedir+'/'+detp+"_low_params_errors.txt",'w+')
#        blah.close()
#	continue
#  xvals.append(float(x_c))
#  yvals.append(float(newcstat))
#  y2.append(float(newcstat))
#  x2.append(float(x_c))
#  with open(homedir+'/'+detp+"_low_params_errors.txt",'a+') as O:
#    O.write(str(x_c)+' '+str(fit_params_errors['det0'].value)+' '+str(fit_params_errors['det1'].value)+' '+str(fit_params_errors['det2'].value)+' '+str(fit_params_errors['det3'].value)+'\n')

#fi_high = homedir+'/'+detp+"_high_params_errors.txt"
#fi_low = homedir+'/'+detp+"_low_params_errors.txt"
#fwrite = homedir+'/'+detp+"_params_errors.txt"
#with open(fi_high) as fh:
#	lines = fh.readlines()
#	with open(fwrite,'w') as fw:
#		fw.write(str(lines))
#with open(fi_low) as fl:
#	lines = fl.readlines()
#	with open(fwrite,'a') as fw:
#		fw.write(str(lines))
#
#os.system('rm '+fi_high+' '+fi_low) 

# if there are new lower cstat values, I will probably have to redo some images and plots. I will have to redo some files also. so check.


#  fit_params_errors = None

################################################################

######## Functions for errors #######################################
#def parabola(x,a,b,c):
#	return a*x**2+b*x+c

#def solu(cs,a,b,cv):
#  c = cv-cs-1
#  pos = (-b+np.sqrt(b**2-4*a*c))/(2*a)
#  neg = (-b-np.sqrt(b**2-4*a*c))/(2*a)
#  return max(pos,neg)

#def sold(cs,a,b,cv):
#  c = cv-cs-1
#  pos = (-b+np.sqrt(b**2-4*a*c))/(2*a)
#  neg = (-b-np.sqrt(b**2-4*a*c))/(2*a)
#  return min(pos,neg)
#####################################################################

####### Curve fit #################################
#p,c = curve_fit(parabola,xvals,yvals)

#p = [0.005,.01,10000]
#pars1, cov1 = opt.curve_fit(parabola,x1,y1,p)
#pars2, cov2 = opt.curve_fit(parabola,x2,y2,p)

#yfit1,yfit2 = [],[]

#for i in range(len(x1)): 
#    fitvalu = parabola(x1[i],*pars1)
#    yfit1.append(fitvalu)

#for i in range(len(x2)): 
#    fitvalu = parabola(x2[i],*pars2)
#    yfit2.append(fitvalu)
#################################################

############ 1-sig error values ###############
#sigup = solu(m_cstat,*pars1)
#sigdown = sold(m_cstat,*pars2)    
###############################################

#plt.scatter(xvals,yvals,s=1)
#plt.plot(x1,yfit1,label = 'fit1',linewidth=2)
#plt.plot(x2,yfit2,label = 'fit2',linewidth=2)
#plt.axvline(x=fit_params['aCXB'])
#plt.title("Cstat error values for 10 sigma in "+lowe+" to "+highe+" keV")
#plt.xlabel("aCXB")
#plt.ylabel(r'$\Delta$'+"Cstat")
#plt.legend()
#plt.show()
#plt.savefig(homedir+"/Cstat_errors_"+detp+"_"+sep+"_"+lowe+"_"+highe+".png")
#plt.close()
# save the plots and fit a parabola to both sides of the graph to recall exact values. 

############ Save Values to file ###################

# The format of the file will be the aCXB and det values with notes on the gradiant values used
# next will be the parameters for the parabolic fit with the range of xvals

#with open(homedir+'/'+detp+"_params.txt",'a+') as O:
#  O.write(lowe+' '+highe+' '+str(fit_params['aCXB'].value)+' '+str(sigup)+' '+str(sigdown)+' '+str(Exposure)+'\n')

#with open(homedir+'/'+detp+'_'+lowe+'_'+highe+'_'+'keVparams_longformat.txt','a+') as O: 
#  O.write(" Gradiant file: "+nusky_dir+'det'+detp+'_det1.img divided by 0.0001'+'\n')
#  O.write("Bins: "+str(len(count_bins))+" Total count:"+str(np.sum(count_bins))+'\n')
#  O.write("aCXB norm value: "+str(x_c)+'\n')
#  O.write("det0 norm value: "+str(standards['det0'].value)+'\n')
#  O.write("det1 norm value: "+str(standards['det1'].value)+'\n')
#  O.write("det2 norm value: "+str(standards['det2'].value)+'\n')
#  O.write("det3 norm value: "+str(standards['det3'].value)+'\n')
#  O.write("crab norm value: "+str(standards['crab'].value+'\n')
#  O.write("#"*25+'\n')
#  O.write("The following fit parameters are for the error values based on a parabola ax^2+bx+c"+'\n')
#  O.write("X-range: "+str(min(xvals))+" - "+str(max(xvals))+'\n')
#  O.write("Min Cstat value: "+str(cstat)+'\n')
#  O.write("The full range: "+str(p)+'\n')
#  O.write("Params for <= aCXB value: "+str(pars1)+'\n')
#  O.write("params for > aCXB value: "+str(pars2)+'\n')
