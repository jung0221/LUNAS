import os
import subprocess
import logging
import glob
from src.lung_modules.auto_seeds import AutoSeeds
from src.roift.roift import RelaxedOIFT
from src.lung_modules.filter_lung_parts import SegregateLungParts
import argparse
import traceback
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time


class Lunas:
    def __init__(
        self, inputfolder, outputfolder, log_file_path="segmentation_progress.log"
    ):
        self.inputfolder = Path(inputfolder)
        if isinstance(outputfolder, str):
            of = outputfolder.strip()
            of = of.strip("\"'")
            self.outputfolder = Path(os.path.normpath(of))
        else:
            self.outputfolder = Path(outputfolder)
        self.log_file_path = log_file_path
        self.logger = self.setup_logger(log_file_path)
        self.roift = RelaxedOIFT()
        self.left_lung_out_path = None
        self.right_lung_out_path = None

    # TODO: Create a file to log codes.
    @staticmethod
    def setup_logger(log_file_path):
        logger = logging.getLogger(os.path.basename(__file__))
        logger.setLevel(logging.INFO)

        # FileHandler for logging to a file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # StreamHandler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter("%(asctime)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def setup_patient(self):
        patient_basename = os.path.basename(self.patient)
        if patient_basename.endswith(".nii.gz"):
            self.patient_name = patient_basename[:-7]  # Remove '.nii.gz'
        elif patient_basename.endswith(".nii"):
            self.patient_name = patient_basename[:-4]  # Remove '.nii'
        else:
            # If no known extension, just use the basename
            self.patient_name = patient_basename
        print("[INFO] Pulling image: ", self.patient_name)

        self.patient_output_dir = Path(
            os.path.join(self.outputfolder, self.patient_name)
        )

    def generate_seeds(self):
        self.logger.info(f"Generating seeds for {self.patient_name}")
        print("[INFO] Generating Seeds")
        trachea_seeds = self.lunas_autoseeds.run_trachea()
        left_lung_seeds, right_lung_seeds = self.lunas_autoseeds.run_lungs(
            trachea_seeds.copy()
        )
        external_seeds = self.lunas_autoseeds.run_external()

        left_ext_lung_seeds = [external_seeds, trachea_seeds, right_lung_seeds]
        right_ext_lung_seeds = [external_seeds, trachea_seeds, left_lung_seeds]

        if not os.path.exists(self.patient_output_dir):
            os.makedirs(self.patient_output_dir)

        # Save seed files
        self.trachea_seed_path = self.lunas_autoseeds.save_seeds(
            [trachea_seeds],
            [external_seeds, left_lung_seeds, right_lung_seeds],
            os.path.join(self.patient_output_dir, f"trachea_{self.patient_name}.txt"),
            0,
        )

        self.left_lung_seed_path = self.lunas_autoseeds.save_seeds(
            [left_lung_seeds],
            left_ext_lung_seeds,
            os.path.join(self.patient_output_dir, f"left_lung_{self.patient_name}.txt"),
            1,
        )
        self.right_lung_seed_path = self.lunas_autoseeds.save_seeds(
            [right_lung_seeds],
            right_ext_lung_seeds,
            os.path.join(
                self.patient_output_dir, f"right_lung_{self.patient_name}.txt"
            ),
            2,
        )
        return (
            left_lung_seeds,
            right_lung_seeds,
            left_ext_lung_seeds,
            right_ext_lung_seeds,
        )

    def run_roift(self):

        # Trachea segmentation
        self.logger.info(f"Performing segmentation for {self.patient_name} - Trachea")
        self.trachea_out_path = os.path.join(
            self.outputfolder,
            self.patient_name,
            f"1st_trachea_{self.patient_name}.nii.gz",
        )
        if not os.path.exists(self.trachea_out_path):
            print("[INFO] Performing Segmentation - Trachea")
            self.roift.run(
                self.patient, self.trachea_seed_path, 0, 0, -1, 0, self.trachea_out_path
            )

        # Left lung segmentation
        self.logger.info(f"Performing segmentation for {self.patient_name} - Left Lung")
        self.left_lung_out_path = os.path.join(
            self.outputfolder,
            self.patient_name,
            f"1st_left_lung_{self.patient_name}.nii.gz",
        )
        if not os.path.exists(self.left_lung_out_path):
            print("[INFO] Performing Segmentation - Left Lung")
            self.roift.run(
                self.patient,
                self.left_lung_seed_path,
                0,
                10,
                -0.8,
                0,
                self.left_lung_out_path,
            )

        # Right lung segmentation
        self.logger.info(
            f"Performing segmentation for {self.patient_name} - Right Lung"
        )
        self.right_lung_out_path = os.path.join(
            self.outputfolder,
            self.patient_name,
            f"1st_right_lung_{self.patient_name}.nii.gz",
        )
        if not os.path.exists(self.right_lung_out_path):
            print("[INFO] Performing Segmentation - Right Lung")
            self.roift.run(
                self.patient,
                self.right_lung_seed_path,
                0,
                10,
                -0.8,
                0,
                self.right_lung_out_path,
            )

    def filter_lung(self, left_ext_lung_seeds, right_ext_lung_seeds):
        print("[INFO] Removing irregular parts")
        filter_left = SegregateLungParts(self.patient, self.left_lung_out_path)
        left_seeds, ext_left_seeds = filter_left.run()

        filter_right = SegregateLungParts(self.patient, self.right_lung_out_path)
        right_seeds, ext_right_seeds = filter_right.run()
        if np.any(ext_left_seeds) and np.any(ext_right_seeds):
            left_ext_lung_seeds.append(ext_left_seeds)
            right_ext_lung_seeds.append(ext_right_seeds)

        self.left_lung_seed_path = self.lunas_autoseeds.save_seeds(
            [left_seeds],
            left_ext_lung_seeds,
            os.path.join(self.patient_output_dir, f"left_lung_{self.patient_name}.txt"),
            1,
        )

        self.right_lung_seed_path = self.lunas_autoseeds.save_seeds(
            [right_seeds],
            right_ext_lung_seeds,
            os.path.join(
                self.patient_output_dir, f"right_lung_{self.patient_name}.txt"
            ),
            1,
        )

    def run(self, patient_path, store_seeds=False):
        self.patient = patient_path
        self.lunas_autoseeds = AutoSeeds(self.patient)

        try:
            self.setup_patient()
            (
                self.left_lung_seeds,
                self.right_lung_seeds,
                self.left_ext_lung_seeds,
                self.right_ext_lung_seeds,
            ) = self.generate_seeds()

            self.run_roift()
            print("[INFO] Removing temporary files")
            if not store_seeds:
                seed_files = glob.glob(os.path.join(self.patient_output_dir, "*.txt"))
                for file in seed_files:
                    os.remove(file)
                
                

            return True

        except subprocess.CalledProcessError as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            func = tb.name
            line = tb.lineno
            filename = tb.filename
            self.logger.error(
                f"Subprocess failed for {self.patient_name} in {func}() at {filename}:{line}: {e}"
            )
            self.logger.error(f"Subprocess stderr: {e.stderr}")
            print(
                f"[ERROR] Subprocess failed for {self.patient_name} in {func}() at {filename}:{line}: {e}"
            )
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            func = tb.name
            line = tb.lineno
            filename = tb.filename
            self.logger.error(
                f"Segmentation failed for {self.patient_name} in {func}() at {filename}:{line}: {str(e)}"
            )
            print(
                f"[ERROR] Segmentation failed for {self.patient_name} in {func}() at {filename}:{line}: {str(e)}"
            )

    def second_segment(self):
        try:
            self.filter_lung(self.left_ext_lung_seeds, self.right_ext_lung_seeds)
            print("[INFO] Performing Second Segmentation - Left Lung")
            self.left_lung_out_path = os.path.join(
                self.outputfolder,
                self.patient_name,
                f"2nd_left_lung_{self.patient_name}.nii.gz",
            )

            self.roift.run(
                os.path.join(self.patient),
                self.left_lung_seed_path,
                90,
                10,
                -0.8,
                0,
                self.left_lung_out_path,
            )

            print("[INFO] Performing Second Segmentation - Right Lung")
            self.right_lung_out_path = os.path.join(
                self.outputfolder,
                self.patient_name,
                f"2nd_right_lung_{self.patient_name}.nii.gz",
            )

            self.roift.run(
                os.path.join(self.patient),
                self.right_lung_seed_path,
                90,
                10,
                -0.8,
                0,
                self.right_lung_out_path,
            )

            self.logger.info(f"Segmentation completed for {self.patient_name}")

        except subprocess.CalledProcessError as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            func = tb.name
            line = tb.lineno
            filename = tb.filename
            self.logger.error(
                f"Subprocess failed for {self.patient_name} in {func}() at {filename}:{line}: {e}"
            )
            self.logger.error(f"Subprocess stderr: {e.stderr}")
            print(
                f"[ERROR] Subprocess failed for {self.patient_name} in {func}() at {filename}:{line}: {e}"
            )
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            func = tb.name
            line = tb.lineno
            filename = tb.filename
            self.logger.error(
                f"Segmentation failed for {self.patient_name} in {func}() at {filename}:{line}: {str(e)}"
            )
            print(
                f"[ERROR] Segmentation failed for {self.patient_name} in {func}() at {filename}:{line}: {str(e)}"
            )


def main():
    parser = argparse.ArgumentParser(description="Process Nifti file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Nifti file path")
    group.add_argument("--patient-list-path", help="Folder containing Nifti files")
    group.add_argument("--metadata-path", help="Segment using metadata file (csv)")
    parser.add_argument(
        "--mesh", required=False, default="0", help="Create mesh (1) or not (0)"
    )
    parser.add_argument(
        "--output",
        default="lunas_output",
        help="Nifti segmentation output path",
    )
    parser.add_argument(
        "--pol", type=float, default=0.1, help="Pol value (default: 0.1)"
    )
    parser.add_argument(
        "--dilperc", type=int, default=90, help="Percentage value (default: 90)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Number of iterations (default: 10)"
    )
    parser.add_argument("--store-seeds", action="store_true", help="Store seeds file")

    args = parser.parse_args()

    if args.patient:
        inputfolder = os.path.dirname(args.patient)
        lunas = Lunas(inputfolder=inputfolder, outputfolder=args.output)
        lunas.run(patient_path=args.patient, store_seeds=args.store_seeds)
    elif args.patient_list_path:
        lunas = Lunas(args.patient_list_path, args.output)
        lists = glob.glob(os.path.join(args.patient_list_path, "*.nii")) + glob.glob(
            os.path.join(args.patient_list_path, "*.nii.gz")
        )
    elif args.metadata_path:
        df = pd.read_csv(args.metadata_path)
        lists = df["Full Path"]
        lunas = Lunas(lists, args.output)
    if args.patient_list_path or args.metadata_path:
        for i, patient_path in enumerate(lists):
            if not os.path.exists(
                os.path.join(
                    args.output, os.path.basename(patient_path).replace(".nii", "")
                )
            ):
                lunas.run(patient_path, store_seeds=args.only_seeds)


if __name__ == "__main__":
    main()
