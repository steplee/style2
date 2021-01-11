# Style Transfer System
Some implementations of CNN based single-image style transfer.

The best model was trained using ideas from these [two](https://arxiv.org/pdf/1703.06868.pdf) [papers](https://arxiv.org/pdf/1909.13690v2.pdf), as well as adding two GAN discriminators on top to improve visual quality.

The AdaIn paper + GAN was implemented in `style2/adain.py`.
It is actually amazing how well transferring just the first two moments is able to copy style, but
it definately does not get the job fully done.

The second paper, which I call AdaInRA (rigid alignment), prescribes an additional step to AdaIN's moment matching that
allows the model to retain global style artifacts. It (plus two more GAN discriminators) are implemented in `style2/adain2.py`
It requires doing an SVD with columns proportional to square image size, which unforunately is very slow.
It definately gives better results, but is not suitable for running on videos. I haven't found a way to
run the svd at a lower resolution then upscale. Perhaps you can train an MLP to approximate the SVD instead.

AdaInRA results below (rows are **content**, **style**, **transferred style-to-content**)
![](/doc/adaRA006950.jpg)
![](/doc/adaRA007350.jpg)
