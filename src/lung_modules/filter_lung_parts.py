import nibabel as nib
from dataclasses import dataclass
import os
import cv2
import numpy as np
import sys
sys.path.append(os.getcwd())
from lung_segmentation.modules.close_holes import CloseHoles

@dataclass
class SegregateLungParts:
    patient_path: str
    lung_path: str
    def load_image(self):
        self.patient_image = nib.load(self.patient_path).get_fdata()
        lung_nifti = nib.load(self.lung_path)
        self.lung_image = lung_nifti.get_fdata().astype(np.uint8)
        self.lung_affine = lung_nifti.affine
        self.patient_name = os.path.basename(self.lung_path)
        self.dir_name = os.path.dirname(self.lung_path)

    def extract_ext_seeds(self):
        self.lung_image[self.patient_image > -300] = 0
        # self.lung_image[self.patient_image < -800] = 0
        ext_coords = np.where((self.lung_image == 1) & (self.patient_image < -1000))
        return ext_coords

    def morph_objects(self):
        kernel = np.ones((5, 5), np.uint8)
        for i in range(self.lung_image.shape[2]):
            self.lung_image[:,:,i] = cv2.erode(self.lung_image[:,:,i], kernel)
            self.lung_image[:,:,i] = cv2.erode(self.lung_image[:,:,i], kernel)
            self.lung_image[:,:,i] = cv2.erode(self.lung_image[:,:,i], kernel)

    def save_image(self, image, file_dir):
        nifti = nib.Nifti1Image(image.astype(np.int16), self.lung_affine)
        nib.loadsave.save(nifti, file_dir)

    def sample_seeds(self, coords, percent):
        seeds = np.vstack((coords[0], coords[1], coords[2])).T

        sample_size = int(coords[0].shape[0] * percent)
        num_seeds = seeds.shape[0]
        if sample_size > 0:
            sampled_indices = np.random.choice(num_seeds, size=sample_size, replace=False)
            seeds = seeds[sampled_indices]

        return seeds


    def run(self, filter_tumor=False):
        self.load_image()
        if filter_tumor:
            ext_coords = self.extract_ext_seeds()
        else: 
            ext_coords = None
        # TODO: Airways segmentation
        # airways_arr = self.lung_image.copy()  
        # for i in range(self.lung_image.shape[2]):
        #     airways_arr[:,:,i] = CloseHoles(self.lung_image[:,:,i]).run(option=False)
        #     airways_arr[:,:,i] = airways_arr[:,:,i] - self.lung_image[:,:,i] 
        # self.save_image(airways_arr, os.path.join(self.dir_name, "airways-" + self.patient_name))
        
        self.morph_objects()
        lung_coords = np.where(self.lung_image == 1)
        lung_seeds = self.sample_seeds(coords=lung_coords, percent=0.01)

        if np.any(ext_coords):
            ext_seeds = self.sample_seeds(coords=ext_coords, percent=1.0)
        else: 
            ext_seeds = None
        return lung_seeds, ext_seeds

if __name__ == "__main__":
    lung_path = "/mnt/segmentation/CT_segmented/Cornell-W0001-1.2.826.0.1.3680043.2.656.1.136-S02A01/left_lung_Cornell-W0001-1.2.826.0.1.3680043.2.656.1.136-S02A01.nii"
    patient_path = "/mnt/segmentation/CT_images/Cornell-W0001-1.2.826.0.1.3680043.2.656.1.136-S02A01.nii"
    
    run_seg = SegregateLungParts(patient_path, lung_path)
    run_seg.run()
