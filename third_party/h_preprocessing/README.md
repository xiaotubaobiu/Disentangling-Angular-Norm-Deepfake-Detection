# Primary code and model provenance

Pipeline A uses [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
at commit `b984b4b775b2c4dced95c1eadd195a5c7d32a60b`, its MobileNet0.25 architecture,
and the pretrained `mobilenet0.25_Final.pth` linked by that repository's README.
This is the repository author's PyTorch implementation of RetinaFace; the
original RetinaFace paper implementation is MXNet. No SCRFD model is used.
The vendored source is unmodified and retains its MIT license.

Pipeline B uses [dlib 20.0.0](https://github.com/davisking/dlib/tree/v20.0), the
HOG/SVM frontal detector, and the official
[68-point predictor](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
The [landmark example](https://dlib.net/face_landmark_detection.py.html) documents
the predictor and its iBUG 300-W training data. That data excludes commercial
use; this journal experiment is academic research. The dlib code has the Boost
Software License, copied in DLIB_LICENSE.txt. The chip geometry follows dlib's
`get_face_chip_details` and `get_mapping_to_chip` in
[interpolation.h](https://github.com/davisking/dlib/blob/v20.0/dlib/image_transforms/interpolation.h).

The five-point target is the usual ArcFace 112-pixel landmark template scaled
to 336 pixels. Its coordinates are recorded directly in the frozen protocol;
the geometry is this experiment's declared simulation, without a claim that
it reproduces any dataset publisher's preprocessing.

`PROVENANCE.json` records source, downloaded model URLs, revision and SHA256
identities. Checkpoints live under the artifact root's weights/preprocessing.
The RetinaFace adapter verifies model and source bytes before inference.
