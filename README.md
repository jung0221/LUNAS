
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

## Usage

The following command computes the segmentation by LUNAS for the volume in the hypothetical file **"example01.nii"**.

```
python lunas.py --patient "example01.nii" --mesh "1" --pol "0.1" --dilperc "90" --iters "10"
```

### Parameters

- --patient: Path to the Nifti file (required).
- --mesh: Path to the mesh file (required). This parameter specifies the output path for the generated mesh file.
- --pol: Pol value for the ROIFT method (optional, default: 0.1). This parameter controls the level of polynomial smoothing applied during the segmentation process.
- --dilperc: Percentage value for conditional dilation (optional, default: 90). This parameter is used to improve the accuracy of the ground-truth of the datasets by applying a dilation operation conditionally.
- --iters: Number of iterations for the ROIFT method (optional, default: 10). This parameter specifies how many iterations the ROIFT algorithm should perform.

### Output

The program generates several output files in the **"out"** subfolder:

- **Nifti files**: The segmented parts are saved as `.nii` files in the `/out/nifti` directory. These include:
  - `left_lung-(input_file_name).nii`
  - `right_lung-(input_file_name).nii`
  - `ribs-(input_file_name).nii`
  - `skin-(input_file_name).nii`
  - `airways-(input_file_name).nii`

- **Mesh files**: If the `--mesh` parameter is provided, the program will generate `.stl` files for the segmented parts in the `/out/stl` directory. These include:
  - `left_lung-(input_file_name).stl`
  - `right_lung-(input_file_name).stl`
  - `ribs-(input_file_name).stl`
  - `skin-(input_file_name).stl`
  - `airways-(input_file_name).stl`


## Method

### 1. Automatic Seed Generator
This step involves generating internal and external seeds for lung segmentation, targeting specific regions of interest (lungs, trachea, ribs). The process includes:

- Thresholding: Identifying potential regions by applying intensity thresholds specific to the lungs, ribs, and trachea.
- Noise Removal: Cleaning binary images to remove irrelevant components like hospital bed artifacts.
- 2D Sampling: Extracting slices along the transverse plane for each region, considering spatial relationships and anatomical features.
- Seed Extraction: Identifying connected components, calculating their centers, and expanding seeds for better coverage.
- Verification: Validating seeds based on their position relative to anatomical structures, ensuring accuracy.
- Side Classification: Categorizing lung seeds as left or right based on trachea position and axial slice analysis.

### 2. Relaxed Oriented Image Foresting Transform

The Oriented Image Foresting Transform (OIFT) combines concepts from Image Foresting Transform (IFT), General Fuzzy Connectedness (GFC), and Generalized Graph Cut (GGC). It inherits key properties from these frameworks, such as robustness to seed placement. OIFT minimizes a specific energy function to compute an optimal partition that separates object and background regions on a symmetric digraph. It incorporates boundary polarity by assigning orientation-sensitive weights to arcs, allowing segmentation to favor either dark objects in bright backgrounds or the opposite, depending on the parameter settings.

Relaxation Procedure
The Relaxed Oriented Image Foresting Transform (ROIFT) extends OIFT by introducing an iterative relaxation process. Starting from an initial segmentation produced by OIFT, a sequence of progressively refined fuzzy segmentations is computed. This iterative approach adjusts arc weights dynamically, reflecting directed behavior. The final segmentation is determined by converting the fuzzy segmentation into a binary result.

The iterative relaxation smooths irregularities in the segmentation contours, improving both accuracy and visual quality compared to the original OIFT and related methods. Additionally, the approach balances the strengths of OIFT and Random Walks (RW), offering hybrid results that better align with human perception.

### Datasets used for evaluation

LUNAS has been evaluated on several datasets, including:

- LCTSC (Lung CT Segmentation Challenge dataset): A comprehensive resource for evaluation, chosen for its complexity and clinical significance. It encompasses 60 images with a variety of benign and malignant pulmonary lesion patterns, covering a wide range of pathological conditions. This diversity closely mirrors the real-world scenarios encountered by healthcare providers, establishing it as a suitable benchmark to evaluate the effectiveness and reliability of segmentation algorithms in clinical applications (https://doi.org/10.7937/K9/TCIA.2017.3R3FVZ08).

- LOLA11 (LObe and Lung Analysis 2011): LOLA11 provides 55 images of chest CT scans with varying abnormalities for which reference standards of lung and lobe segmentation have been established. (https://lola11.grand-challenge.org/). 

- EXACT (Extraction of Airways from CT 2009): The goal of the EXACT study is to compare algorithms to extract the airway tree from chest CT scans using a common dataset with 40 images and a performance evaluation method (https://doi.org/10.1109/TMI.2012.2209674).

- VIA/I-ELCAP (Vision and Image Analysis Group/International Early Lung Cancer Action Program): 50 CT scans were obtained in a single breath hold with a slice thickness of 1.25 mm. The locations of the nodules detected by the radiologist are also provided in this dataset (http://www.via.cornell.edu/lungdb.html).


## Authors

This particular code was implemented by:

- Jungeui Choi
- Marcos A. T. Condori
- Paulo A.V. Miranda
- Marcos S. G. Tsuzuki

## Source code

The source code was implemented in Python and C/C++ language, compiled with gcc 9.4.0, and tested on a Linux operating system (Ubuntu 20.04.5 LTS 64-bit), running on an Intel® Core™ I7-12700 CPU @ 4.90GHz × 8 machine. 
The code natively supports volumes in the NIfTI format.

To compile the program, enter the folder and type **"make"**.
If you get the error **"fatal error: zlib.h: No such file or directory"**, then you have to install the zlib package:

```
sudo apt-get install libz-dev
```

## Referencing and citing
If you are working with this code to your project, please refer to:

> Choi J., Condori M. A. T., Miranda P. A. V. and Tsuzuki M. S. G. Lung automatic seeding and segmentation: a robust method based on relaxed oriented image foresting transform.

## Contact

If you have any doubts, questions or suggestions to improve this code, please contact me at:
**jungchoi@usp.br**

