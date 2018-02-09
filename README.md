# IARPA-data

The whole pipeline is:
1. Calculate thresholded data Yt. You can directly load Yt or calculate from denoised data Yd.
2. Correlation analysis to find out superpixels.
3. Rank-1 svd to find out the temporal trace of superpixels.
4. SPA to find pure superpixel. This is implemented by a matlab function "FastSepNMF.m".  You need to run spa.m when doing this.
Now you can get the temporal traces for all the pure superpixels!

