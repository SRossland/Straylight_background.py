# Straylight_background.py
Measures the CXB norm value in a DET1 NuSTAR image

The purpose of this code is to measure the aCXB norm value in NuSTAR observations that have straylight due to bright off axis sources which create circular regions on the detector.

NOTE:
  This program was written organically and is currently lacking in any aesthetic.
  This code currently only reports the aCXB norm value in usable values, that being in cts/s/cm^2/det^2 over the energy range given.
  Other norm values for the detectors and the stray light are given, but they are just relative to the fitting and should only be used as a reference of the quality of the run and not scientific analysis until scaled.
  Currently no error values are given, this portion of the code is the most time consuming while also exhibiting the greatest stability so it was commented out until such time as deemed necessary. ALSO, it should only be done sparingly with the certainty that enough counts are available since it is based on the Cash statistics and the reliability of 1 sigma errors are sensitive to this count. see https://arxiv.org/pdf/0811.2796.pdf for more information on this.
  If an exclusion region is to be used to mask out such things as a source, one must ensure that the radius is large enough to account for NuSTAR's extended PSF wings. An example of such a region is included that corresponds to the DET1 image of 90401321002.
 
# Before this code can be run, certain absolute paths must be updated:

# line 1: (OPTIONAL) If you require the use of the shebang, this will have to changed to something appropriate to you, i.e., #!/usr/bin/env python3
# line 229: nusky_dir --> This path points to the auxil directory within nuskybgd
# line 230: edge --> This path points to the mask fits file that is given in this hub, fullmask[A/B]__final.fits
# line 239: detmap[A/B].fits. This file is given in this hub, just need to have the path point to where you store it
# line 251: pixmap[A/B].fits. This file is given in this hub, it is ALSO given in the nuskybgd auxil, however, this one is more up-to-date
# line 493: normmeanvals.txt. This file is given in this hub and contains an initial value for norm fitting. This could be changed in the future.
==============================================
# There is an extra file you may need if you do not have nuskbgd locally (which is what the nusky_dir path is there for). det[A/B]_det1.img
==============================================
# Syntax:
#   count_stat_crab.py telescope[A/B]
#   fits_file (Data fits file in DET1 coords.)
#   lower_energy_limit (Initially used for reference values, currently this is only needed for errors)
#   high_energy_limit (Same as above)
#   observation_ID(7 digit) (The NuSTAR observation identifier)
#   home_directory (Currently this is assumed to be the directory where your files are stored, and also serves as your write out directory)
#   Straylight_region(physical values assumed to have ds9 format)
#   reference_detector(0,1,2,3)
#   exclusion_region(physical values assumed to have ds9 format)

example:
 ** Please note, this is currently ran by ssh on a local computational server that does not recognize a shebang and this calls python3 not python2
 
python count_stat_crab.py A nu90401321002A01_cl_2to20keV_det1.fits 2 20 90401321002 /uufs/astro.utah.edu/common/home/u1019304/TEST_OBS_RENEE/90401321002/clean 90401321002A_StrayCatsI_444.reg 1 excl_A.reg

python count_stat_crab.py B nu90401321002B01_cl_2to20keV_det1.fits 2 20 90401321002 /uufs/astro.utah.edu/common/home/u1019304/TEST_OBS_RENEE/90401321002/clean 90401321002B_StrayCatsI_445.reg 1 excl_B.reg
