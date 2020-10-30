# Deep transcoding based SAR change detection (CD)"

**Details related to CycleGAN training** Description of the architecture of discriminator and generator has been provided in netDiscriminator.txt and netGenerator.txt. Furthermore, some related parameters are defined in someParameters.txt

**Main CD part** - The CD method described in the paper (Section III.B and part of III.C) is based on Deep Change Vector Analysis (DCVA)  (https://github.com/sudipansaha/dcvaVHROptical) <br/> 

**SAR Feature Extractor**  We release a part of the network trained by us (Section III.A of the paper and Table I). This network can be used as SAR feature extractor in different SAR classification task and also change detection tasks. <br/>
For classification, append fully connected (FC) layer(s) at the end of the provided network and fine-tune the model for your task. <br/>
For change detection, no such fine-tuning is needed. Use the provided network as deep feature extractor in Deep Change Vector Analysis (DCVA)  (https://github.com/sudipansaha/dcvaVHROptical). For more details, please read the Section III.B of this paper or the original DCVA paper. <br/>





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
