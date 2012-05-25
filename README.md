CUHOG
=====

A HOG implementation on CUDA, which is compatible with Pedro Felzenszwalb's 31-dim HOG seamlessly.

This implementation is based on fastHOG [1] and groundHOG [2]. Our real-time object detection system will be released in next version associated with the Coarse-to-fine [3].
An example with matlab interface will also give. Currently, for a 640x480 image takes only 50 ms in average and the real-time object detection can achieve 10 FPS in GTX560 and 2.6GHZ 4-core CPU.

[1] V A Prisacariu, I D Reid. fastHOG - a real-time GPU implementation of HOG. http://www.robots.ox.ac.uk/~lav/Papers/prisacariu_reid_tr2310_09/prisacariu_reid_tr2310_09.html, 2009.

[2] P. Sudowe. GroundHOG - GPU-based Object Detection with Geometric Constraints. http://www.mmp.rwth-aachen.de/projects/groundhog, 2011.

[3] M. Pedersoli, A. Vedaldi, J. Gonzàlez, " A Coarse-to-fine approach for fast deformable object detection ", in 24th IEEE Computer Vision and Pattern Recognition (CVPR2011), Colorado Springs, CO, June, 2011.


ABOUT THE CODE

This implementation was developed by Xu Hu and Marco Pedersoli. It is provided without any warranty, express or implied. Please use it if it can assist in your research; we request you to cite our paper:

@article{
    author = "M. Pedersoli and J. Gonzalez and X.Hu and X. Roca",
    title = "Towards a Real-Time Pedestrian Detection based only on Vision",
    journal = "JOURNAL OF INTELLIGENT TRANSPORTATION SYSTEM",
    year = "under review"
}
