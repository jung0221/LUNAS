
# LUNAS: Lung automatic seeding and segmentation

LUNg Automatic Seeding and Segmentation (LUNAS) is a automatic seed-based method segmentation for CT thorax images. 

LUNAS utilizes a unique approach combining automatic seed generation and segmentation via the ROIFT (Relaxed Oriented Image Foresting Transform) algorithm. It has demonstrated superior performance compared to state-of-the-art deep learning methods like U-Net and traditional methods.

## Installation

To use LUNAS, clone this repository and install the required dependencies.

```bash
git clone https://github.com/jung0221/LUNAS.git
cd LUNAS
pip install -r requirements.txt
```

Usage

To run the segmentation process, use the following command:

```python lunas_segmentation.py --input <path_to_ct_image> --output <path_to_output_image>
```

Parameters

- --input: Path to the input CT image file.

- --output: Path where the segmented output image will be saved.

Datasets

LUNAS has been evaluated on several datasets, including:

- LCTSC: Lung CT Segmentation Challenge dataset.

- LOLA11: LObe and Lung Analysis 2011 dataset.

- EXACT: Extraction of Airways from CT dataset.


**[1]** Sousa AM, Martins SB, Falc&atilde;o AX, Reis F, Bagatin E, Irion K,
Med Phys. 2019 Nov; 46(11):4970-4982; doi: [10.1002/mp.13773](https://doi.org/10.1002/mp.13773).

**[2]** Yang, J., Sharp, G., Veeraraghavan, H., Van Elmpt, W., Dekker, A., Lustberg, T., & Gooding, M. (2017). Data from Lung CT Segmentation Challenge (LCTSC) (Version 3) [Data set]. The Cancer Imaging Archive. doi: [10.7937/K9/TCIA.2017.3R3FVZ08](https://doi.org/10.7937/K9/TCIA.2017.3R3FVZ08).

**[3]** Caio L. Demario, Paulo A.V. Miranda, RELAXED ORIENTED IMAGE FORESTING TRANSFORM FOR SEEDED IMAGE SEGMENTATION, 26th IEEE International Conference on Image Processing (ICIP). Sep 2019; Taipei, Taiwan, pp. 1520-1524; doi: [10.1109/ICIP.2019.8803080](http://dx.doi.org/10.1109/ICIP.2019.8803080).


If you are working with this code to your project, in addition to referencing **[1]**, please also cite:

> Choi J., Condori M. A. T., Miranda P. A. V. and Tsuzuki M. S. G. Lung automatic seeding and segmentation: a robust method based on relaxed oriented image foresting transform.

## Authors

This particular code was implemented by:

- Jungeui Choi
- Marcos A. T. Condori
- Paulo A.V. Miranda
- Marcos S. G. Tsuzuki

## Source code

The source code was implemented in Python and C/C++ language, compiled with gcc 9.4.0, and tested on a Linux operating system (Ubuntu 20.04.5 LTS 64-bit), running on an Intel® Core™ i5-10210U CPU @ 1.60GHz × 8 machine. 
The code natively supports volumes in the NIfTI format.



To compile the program, enter the folder and type **"make"**.
If you get the error **"fatal error: zlib.h: No such file or directory"**, then you have to install the zlib package: zlib1g-dev.



As output, the program generates the label image of the resulting segmentation in file **"segm_altis.nii.gz"** in the **"out"** subfolder, when **output_type** is zero.

### Program execution examples of ALTIS:

#### To execute the ALTIS method:

The following command computes the segmentation by ALTIS for the volume in the hypothetical file **"example01.nii.gz"**.

```
./altis example01.nii.gz 0
```

The following command computes the segmentation by ALTIS for the volume in the hypothetical file **"example01.nii.gz"**, using a fixed threshold of 200 on the residual image, instead of the default threshold defined as a percentage above Otsu's threshold.

```
./altis example01.nii.gz 0 T=200
```


### usage of ALTIS + ROIFT:

```
usage:
altis_roift <volume> [T=value] [left=file] [right=file]
Optional parameters:
	T................... threshold integer value
	                     (if not specified Otsu is used).
	left................ ground truth for left lung.
	right............... ground truth for right lung.
```

As output, the program generates improved segmentations of the left and right lungs, respectively, in the files **"segm_left_lung.nii.gz"** and **"segm_right_lung.nii.gz"** in the **"out"** subfolder.

### Program execution examples of ALTIS + ROIFT:

#### To execute seed generation by ALTIS and delineation by ROIFT:


The following command computes the segmentation of the lungs with seeds by ALTIS and delineation by ROIFT for the volume in the hypothetical file **"example01.nii.gz"**.

```
./altis_roift example01.nii.gz
```


The following command computes the segmentation of the lungs with seeds by ALTIS and delineation by ROIFT for the volume in the hypothetical file **"example01.nii.gz"**, using a fixed threshold of 200 on the residual image, instead of the default threshold defined as a percentage above Otsu's threshold.

```
./altis_roift example01.nii.gz T=200
```

## Referencing and citing
If you are working with this code to your project, please refer to:

Choi J., Condori M. A. T., Miranda P. A. V. and Tsuzuki M. S. G. Lung automatic seeding and segmentation: a robust method based on relaxed oriented image foresting transform.

## Contact

If you have any doubts, questions or suggestions to improve this code, please contact me at:
**jungchoi@usp.br**

