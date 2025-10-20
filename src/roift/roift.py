import subprocess
import nibabel as nib
import os
from skimage import measure
import numpy as np
import argparse
import glob
import trimesh
import shutil
import stat

class RelaxedOIFT:
    def convert_to_stl(self, nifti_path):
        img = nib.load(nifti_path)
        img_array = img.get_fdata().copy()
        verts, faces, _, _ = measure.marching_cubes(img_array, method='lewiner')

        # Scale vertices according to voxel spacing
        header = img.header.get_zooms()
        verts_scaled = verts * np.array(header[:3])
        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=verts_scaled, faces=faces)
        
        centroid_original = mesh.centroid.copy()
        
        mesh = trimesh.smoothing.filter_humphrey(mesh)
        
        centroid_after_smooth = mesh.centroid
        translation = centroid_original - centroid_after_smooth
        mesh.vertices += translation
        
        # Export to STL
        if nifti_path.endswith('.nii.gz'):
            stl_output_path = nifti_path[:-7] + '.stl'
        elif nifti_path.endswith('.nii'):
            stl_output_path = nifti_path[:-4] + '.stl'
        else:
            stl_output_path = nifti_path + '.stl'
        mesh.export(stl_output_path)
    def run(self, 
            patient_path: str, 
            seed_path: str, 
            percentile: int, 
            nitter: int, 
            pol: float, 
            as_stl: int, 
            out_path: str):
        self.out_path = out_path
        # Locate the oiftrelax executable in common build locations or in PATH
        exe_name = "oiftrelax.exe" if os.name == "nt" else "oiftrelax"
        cwd = os.getcwd()
        candidates = [
            os.path.join(cwd, "build", "src", "Release", exe_name),
            os.path.join(cwd, "build", "src", exe_name),
            os.path.join(cwd, "build", exe_name),
            os.path.join(cwd, exe_name),
        ]
        exe_path = None
        for c in candidates:
            if os.path.exists(c):
                exe_path = c
                break
        if exe_path is None:
            # try to find in PATH
            exe_path = shutil.which(exe_name)

        if exe_path is None:
            raise FileNotFoundError(f"oiftrelax executable not found. Checked: {candidates} and PATH")

        # Ensure executable bit on POSIX
        if os.name != "nt":
            try:
                st = os.stat(exe_path)
                if not (st.st_mode & stat.S_IXUSR):
                    os.chmod(exe_path, st.st_mode | stat.S_IXUSR)
            except Exception:
                # non-fatal; continue and let subprocess surface errors
                pass

        # Build command as a list to avoid shell quoting issues
        cmd = [exe_path, patient_path, seed_path, str(pol), str(nitter), str(percentile), out_path]
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Attach stdout/stderr to the exception message for clearer logs
            msg = f"oiftrelax failed (rc={e.returncode}). stdout:\n{e.stdout}\n stderr:\n{e.stderr}"
            raise RuntimeError(msg) from e
        
        # The file will be saved in the parent folder as "label.nii.gz"
        nifti_file = nib.load(out_path)
        nifti_arr = nifti_file.get_fdata()
        # Save as .nii format
        nib.loadsave.save(nib.Nifti1Image(nifti_arr.astype(np.int16), nifti_file.affine), out_path)

        if as_stl == 1: 
            self.convert_to_stl(self.out_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter")
    group = parser.add_mutually_exclusive_group(required=True)   
    group.add_argument("--nifti", help="Path to NIfTI image")
    group.add_argument("--nifti-folder", help="Path to NIfTI image")
    
    args = parser.parse_args()
    roift = RelaxedOIFT()

    if args.nifti: 
        images = [args.nifti]
    else:
        images = glob.glob(os.path.join(args.nifti_folder, '*.nii.gz'))
    for image in images:
        if not os.path.exists(image.replace(".nii.gz", ".stl")):
            roift.convert_to_stl(image)