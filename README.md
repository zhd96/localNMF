# Superpixelization demixing method

Reference: https://www.biorxiv.org/content/early/2018/06/03/334706.

The whole pipeline is as follows:
1. Threshold data to increase SNR.
2. Correlation analysis to find out superpixels.
3. Rank-1 svd to extract the temporal trace of superpixels.
4. Successive Projection Algorithm to find pure superpixel.
5. NMF with temporal trace and spatial support of pure superpixels as initialization to extract neurons and their activities.

Example Demo_superpixel_pipeline.ipynb. <br />
Dataset from https://github.com/simonsfoundation/CaImAn. <br />
Recommend running PMD first and using the default parameters in axon_pipeline function. Note that parameters in demo have been tuned to adapt to the example data.
