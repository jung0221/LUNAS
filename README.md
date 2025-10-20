
# LUNAS: Lung automatic seeding and segmentation

LUNg Automatic Seeding and Segmentation (LUNAS) is an automatic seed-based method for segmentation of CT thorax images. 

LUNAS utilizes a unique approach combining automatic seed generation and segmentation via the ROIFT (Relaxed Oriented Image Foresting Transform) algorithm. It has demonstrated superior performance compared to state-of-the-art deep learning methods like U-Net and traditional methods.

## Installation

To use LUNAS, clone this repository and install the required dependencies.

```bash
git clone https://github.com/jung0221/LUNAS.git
cd LUNAS
pip install -r requirements.txt
```

### Building the ROIFT Executable

LUNAS requires the `oiftrelax` executable to perform segmentation. Follow the platform-specific instructions below:

#### Linux

To compile the program on Linux, enter the folder and type:

```bash
make
```

If you get the error **"fatal error: zlib.h: No such file or directory"**, install the zlib development package:

```bash
sudo apt-get install libz-dev
```

#### Windows

For detailed instructions on building ROIFT on Windows with CMake and MSVC, see the [Windows Build Guide](docs/build-windows.md).

## Usage

### Basic Example

The following command computes the segmentation by LUNAS for the volume in the file **"example01.nii"**:

```bash
python lunas.py --patient "example01.nii" --output "lunas_output" --store-seeds
```

### Processing Multiple Files

Process all NIfTI files in a folder:

```bash
python lunas.py --patient-list-path "/path/to/nifti/folder" --output "lunas_output"
```

Process files from a CSV metadata file:

```bash
python lunas.py --metadata-path "patients.csv" --output "lunas_output"
```

### Command-Line Parameters

#### Required (mutually exclusive)
- `--patient`: Path to a single NIfTI file (`.nii` or `.nii.gz`)
- `--patient-list-path`: Folder containing NIfTI files to process in batch
- `--metadata-path`: CSV file with a "Full Path" column containing paths to NIfTI files

#### Optional Parameters
- `--output`: Output directory for segmentation results (default: `lunas_output`)
- `--pol`: Polarity value for the ROIFT method (default: `0.1`). Controls the boundary polarity during segmentation. Positive values favor bright objects on dark backgrounds, negative values favor the opposite
- `--dilperc`: Percentile value for conditional dilation (default: `90`). Used to improve segmentation accuracy by applying conditional dilation to the seeds
- `--iters`: Number of iterations for the ROIFT relaxation process (default: `10`). Higher values produce smoother segmentations but increase computation time
- `--mesh`: Create mesh files in STL format. Use `1` to enable, `0` to disable (default: `0`)
- `--store-seeds`: Store seed files for debugging or manual inspection. If not specified, seed files are automatically deleted after segmentation

### Example Commands

**Single file with custom parameters:**
```bash
python lunas.py --patient "CT_scan.nii.gz" --output "results" --pol 0.15 --iters 15 --mesh 1
```

**Batch processing with seed storage:**
```bash
python lunas.py --patient-list-path "C:/data/ct_scans" --output "batch_results" --store-seeds
```

**Processing from metadata file:**
```bash
python lunas.py --metadata-path "dataset.csv" --output "dataset_results" --dilperc 85
```

### Output

The program generates several output files organized in subdirectories under the specified output folder:

#### Directory Structure
```
lunas_output/
├── patient_name/
│   ├── 1st_left_lung_patient_name.nii.gz
│   ├── 1st_right_lung_patient_name.nii.gz
│   ├── 1st_trachea_patient_name.nii.gz
│   ├── left_lung_patient_name.txt (if --store-seeds)
│   ├── right_lung_patient_name.txt (if --store-seeds)
│   └── trachea_patient_name.txt (if --store-seeds)
└── segmentation_progress.log
```

#### Output Files

- **Segmentation files** (`.nii.gz`): Binary masks for each segmented structure
  - Left lung
  - Right lung
  - Trachea/Airways

- **Seed files** (`.txt`, optional): Coordinate lists of internal and external seeds used for segmentation. Only generated when `--store-seeds` flag is used

- **Mesh files** (`.stl`, optional): 3D surface meshes for visualization. Only generated when `--mesh 1` is specified

- **Log file** (`segmentation_progress.log`): Progress tracking and error logging for batch processing

## Method

### 1. Automatic Seed Generator

This step involves generating internal and external seeds for lung segmentation, targeting specific regions of interest (lungs, trachea). The process includes:

- **Thresholding**: Identifying potential regions by applying intensity thresholds specific to the lungs and trachea
- **Noise Removal**: Cleaning binary images to remove irrelevant components like hospital bed artifacts
- **2D Sampling**: Extracting slices along the transverse plane for each region, considering spatial relationships and anatomical features
- **Seed Extraction**: Identifying connected components, calculating their centers, and expanding seeds for better coverage
- **Verification**: Validating seeds based on their position relative to anatomical structures, ensuring accuracy
- **Side Classification**: Categorizing lung seeds as left or right based on trachea position and axial slice analysis

### 2. Relaxed Oriented Image Foresting Transform (ROIFT)

The Oriented Image Foresting Transform (OIFT) combines concepts from Image Foresting Transform (IFT), General Fuzzy Connectedness (GFC), and Generalized Graph Cut (GGC). It inherits key properties from these frameworks, such as robustness to seed placement. OIFT minimizes a specific energy function to compute an optimal partition that separates object and background regions on a symmetric digraph. It incorporates boundary polarity by assigning orientation-sensitive weights to arcs, allowing segmentation to favor either dark objects in bright backgrounds or the opposite, depending on the parameter settings.

#### Relaxation Procedure

The Relaxed Oriented Image Foresting Transform (ROIFT) extends OIFT by introducing an iterative relaxation process. Starting from an initial segmentation produced by OIFT, a sequence of progressively refined fuzzy segmentations is computed. This iterative approach adjusts arc weights dynamically, reflecting directed behavior. The final segmentation is determined by converting the fuzzy segmentation into a binary result.

The iterative relaxation smooths irregularities in the segmentation contours, improving both accuracy and visual quality compared to the original OIFT and related methods. Additionally, the approach balances the strengths of OIFT and Random Walks (RW), offering hybrid results that better align with human perception.

## Evaluation Datasets

LUNAS has been evaluated on several public datasets, including:

- **LCTSC** (Lung CT Segmentation Challenge): 60 images with various benign and malignant pulmonary lesion patterns ([DOI](https://doi.org/10.7937/K9/TCIA.2017.3R3FVZ08))

- **LOLA11** (LObe and Lung Analysis 2011): 55 chest CT scans with varying abnormalities ([DOI](https://doi.org/10.5281/zenodo.4708800))

- **EXACT09** (Extraction of Airways from CT 2009): 40 images for airway tree extraction evaluation ([DOI](https://doi.org/10.1109/TMI.2012.2209674))

- **VIA/I-ELCAP**: 50 CT scans with 1.25mm slice thickness ([Website](http://www.via.cornell.edu/lungdb.html))

## Source code

The source code was implemented in Python and C/C++ language, compiled with gcc 9.4.0, and tested on a Linux operating system (Ubuntu 20.04.5 LTS 64-bit), running on an Intel® Core™ I7-12700 CPU @ 4.90GHz × 8 machine. 
The code natively supports volumes in the NIfTI format.

To compile the program, enter the folder and type **"make"**.
If you get the error **"fatal error: zlib.h: No such file or directory"**, then you have to install the zlib package:

```
sudo apt-get install libz-dev
```

## Authors

This code was implemented by:

- Jungeui Choi
- Marcos A. T. Condori
- Paulo A.V. Miranda
- Marcos S. G. Tsuzuki

## Citation

If you use this code in your research, please cite:

> Choi J., Condori M. A. T., Miranda P. A. V. and Tsuzuki M. S. G. "Lung automatic seeding and segmentation: a robust method based on relaxed oriented image foresting transform."


## Contact

If you have any doubts, questions or suggestions to improve this code, please contact:
**jungchoi@usp.br**

