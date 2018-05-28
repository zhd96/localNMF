# Axon pipeline

google document link for pipeline description: 

https://docs.google.com/document/d/1JqlBa7MC1HkmQwjjfPoqvRaoGOrCHntj0WR4-HjLo8c/edit?ts=5a71e6c7

The whole pipeline is:
1. Threshold data to increase SNR.
2. Correlation analysis to find out superpixels.
3. Rank-1 svd to find out the temporal trace of superpixels.
4. Successive Projection Algorithm to find pure superpixel.
5. NMF with temporal traces of pure superpixels as initialization to extract neurons and their activities.

Example Demo_superpixel_pipeline.ipynb <br />
Dataset from https://github.com/simonsfoundation/CaImAn
