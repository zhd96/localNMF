# Axon pipeline:

The whole pipeline is:
1. Threshold data to increase SNR.
2. Correlation analysis to find out superpixels.
3. Rank-1 svd to find out the temporal trace of superpixels.
4. Successive Projection Algorithm to find pure superpixel.
5. NMF with temporal traces of pure superpixels as initialization to extract neurons and their activities.

