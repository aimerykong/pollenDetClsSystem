# Automated Recognition and Counting of Pollen Grain Species from Field Samples

Please visit also visit the [webpage page of Pollen Morphology-based Classification and Analysis](http://www.ics.uci.edu/~fowlkes/bioshape/index.html) for more relevant projects.


![alt text](http://www.ics.uci.edu/~skong2/img/demo_BIC_lowRes.png "visualization")


The system first detects and segments pollen grains from a pile of scans over a sample with various z-plane (focus), then finely segments the grains over cropped window and classify them. Essentially, it involves proposal detection, segmentation, and classification. For classification, one is able to adopt multi-instance mechanism on multiple crops at different slides at the same location, due to various focuses.


Run 'main003_saveData.m' to get the annotated images. The path to dataset needs to be changed accordingly.

Download models at [google drive](https://drive.google.com/open?id=0BxeylfSgpk1MREgycndzNmJLT00)

Run 'main100_demo.m' for the demo.

Shu Kong @ UCI
06/5/2017



