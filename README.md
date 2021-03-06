# Deep transcoding based SAR change detection (CD)

**SAR Feature Extractor:**  We release a part of the network trained by us (Section III.A of the paper and Table I). This network can be used as SAR feature extractor in different SAR classification task and also change detection tasks. <br/>
For classification, append fully connected (FC) layer(s) at the end of the provided network and fine-tune the model for your task. <br/>
For change detection, no such fine-tuning is needed. Use the provided network as deep feature extractor in Deep Change Vector Analysis (DCVA)  (https://github.com/sudipansaha/dcvaVHROptical). For more details, please read the Section III.B of this paper or the original DCVA paper. <br/>
SAR Feature extractors are provided in the "SARFeatureExtractor" directory. Please download the models as instructed in "trainedModels" subdirectory.
We have provided feature extractor corresponding to two different layers, 3rd convolutional layer (layer 3 in the Table I of paper) and 1st residual block (layer 4
in the Table I of the paper). Code to load those models are provided in the SARFeatureExtractor/loadFeatureExtractor.py<br/>

**Main CD part:** - The CD method described in the paper (Section III.B and part of III.C) is based on Deep Change Vector Analysis (DCVA)  (https://github.com/sudipansaha/dcvaVHROptical) <br/>

**Details related to CycleGAN training:** Description of the architecture of discriminator and generator has been provided in netDiscriminator.txt and netGenerator.txt. Furthermore, some related parameters are defined in someParameters.txt






### Citation
If you find this code useful, please consider citing:
```[bibtex]
@article{saha2020building,
  title={Building Change Detection in VHR SAR Images via Unsupervised Deep Transcoding},
  author={Saha, Sudipan and Bovolo, Francesca and Bruzzone, Lorenzo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2020},
  publisher={IEEE}
}
```
