import numpy as np
import nibabel as nib
import os
import cv2
import statistics
import imutils
from PIL import Image
import copy

class AutoSeeds:
    def __init__(self, nifti_file):
        self.nifti = nib.load(nifti_file)
        self.affine = self.nifti.affine
        self.nifti_image = self.nifti.get_fdata().copy()
        self.vox_dim = self.affine[:3, :3].diagonal()

    def highlight(self, thresh: int):
        """
        Thresholds the image by organ
        lungs: -500
        ribs: 100
        trachea: -370
        skin: -200
        heart: to be defined
        """

        thresh_image = copy.deepcopy(self.nifti_image)
        thresh_image[thresh_image >= thresh] = np.inf
        thresh_image[thresh_image < thresh] = -np.inf
        thresh_image[thresh_image > 0] = 1
        thresh_image[thresh_image <= 0] = 0

        return thresh_image

    def remove_noises_slice(self, img):
        img = 1 - img
        components, matrix, stats, _ = cv2.connectedComponentsWithStats(
            img, connectivity=8
        )
        sizes = stats[1:, -1]
        components = components - 1
        min_size = int(img.shape[0] / 3.4)
        img2 = np.zeros((matrix.shape))
        for j in range(0, components):
            if sizes[j] >= min_size:
                img2[matrix == j + 1] = 255
        img2 = 1 - img2
        return img2

    def sampling(self, thresh_image, organ):
        """
        Get the thersholded samples to obtain seeds.
        (1) Lung
        (2) Ribs
        (3) Trachea
        (4) Skin
        (5) Heart
        """
        z_list = []
        slices = []
        filtered_slices = []
        if organ == 1 or organ == 4:
            for j in range(10):
                z_value = int((j + 10) * thresh_image.shape[2] / 22)
                z_list.append(z_value)
                thresh_image[:, :, z_value] = np.rot90(thresh_image[:, :, z_value])
                thresh_image[:, :, z_value] = np.flip(
                    thresh_image[:, :, z_value], axis=0
                )

                if self.affine[0][0] < 0:
                    thresh_image[:, :, z_value] = np.flip(
                        thresh_image[:, :, z_value], axis=1
                    )
                if self.affine[1][1] < 0:
                    thresh_image[:, :, z_value] = np.flip(
                        thresh_image[:, :, z_value], axis=0
                    )
                img = np.array(thresh_image[:, :, z_value], dtype=np.uint8)
                slices.append(Image.fromarray(img * 255).convert("L"))

                img2 = self.remove_noises_slice(img) * 255
                kernel = np.ones((5, 5), np.uint8)
                change_times = int(thresh_image.shape[0] / 170)
                for i in range(1, change_times):
                    img2 = cv2.erode(img2, kernel)
                for i in range(1, change_times):
                    img2 = cv2.dilate(img2, kernel)

                filtered_slices.append(Image.fromarray(img2).convert("L"))

        elif organ == 2:
            for j in range(10):
                z_value = int((j + 5) * thresh_image.shape[2] / 15)
                z_list.append(z_value)

                thresh_image[:, :, z_value] = np.rot90(thresh_image[:, :, z_value])
                thresh_image[:, :, z_value] = np.flip(
                    thresh_image[:, :, z_value], axis=0
                )

                if self.affine[0][0] < 0:
                    thresh_image[:, :, z_value] = np.flip(
                        thresh_image[:, :, z_value], axis=1
                    )
                if self.affine[1][1] < 0:
                    thresh_image[:, :, z_value] = np.flip(
                        thresh_image[:, :, z_value], axis=0
                    )

                img = np.array(thresh_image[:, :, z_value], dtype=np.uint8)
                img2 = self.remove_noises_slice(img) * 255
                slices.append(Image.fromarray(img2).convert("L"))

        elif organ == 3:
            for j in range(30):
                z_value = int(
                    thresh_image.shape[2] - (j + 1) * thresh_image.shape[2] / 65
                )
                z_list.append(z_value)
                thresh_image[:, :, z_value] = np.rot90(thresh_image[:, :, z_value])
                thresh_image[:, :, z_value] = np.flip(
                    thresh_image[:, :, z_value], axis=0
                )

                if self.affine[0][0] < 0:
                    thresh_image[:, :, z_value] = np.flip(
                        thresh_image[:, :, z_value], axis=1
                    )
                if self.affine[1][1] < 0:
                    thresh_image[:, :, z_value] = np.flip(
                        thresh_image[:, :, z_value], axis=0
                    )

                img = np.array(thresh_image[:, :, z_value], dtype=np.uint8)
                img2 = self.remove_noises_slice(img) * 255

                slices.append(Image.fromarray(img2).convert("L"))

        elif organ == 5:
            for j in range(3):
                z_value = int(thresh_image.shape[2] * (j + 4) / 8)
                self.z_list.append(z_value)
                thresh_image[:, :, z_value] = np.rot90(thresh_image[:, :, z_value])
                thresh_image[:, :, z_value] = np.flip(
                    thresh_image[:, :, z_value], axis=0
                )
                slices.append(thresh_image[:, :, z_value])

        return slices, filtered_slices, z_list

    def seeds_by_area(self, slices, area_factor: tuple, debug=False):
        """
        Get the seeds calculating the CoG of each component, by divided area:
        lungs: (2.62, 260)
        trachea: (250, 10486)
        ribs: (44, 1310)
        """
        seeds = []

        # prepare debug directory
        if debug:
            debug_dir = "auto_seeds_debug"
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception:
                pass

        for idx, slice in enumerate(slices):
            slice = np.array(slice)

            ksize = 3
            gX = cv2.convertScaleAbs(
                cv2.Sobel(slice, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
            )
            gY = cv2.convertScaleAbs(
                cv2.Sobel(slice, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
            )
            combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

            thresh = cv2.threshold(
                combined, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            figarea = abs(
                thresh.shape[0] * thresh.shape[1] / (self.vox_dim[0] * self.vox_dim[1])
            )  # in mm^2
            centros = []
            # iterate components and collect centroids that meet area criteria
            for i in range(0, numLabels):
                area = stats[i, cv2.CC_STAT_AREA]
                (cX, cY) = centroids[i]
                if area < figarea / area_factor[0] and area > figarea / area_factor[1]:
                    cX = int(cX)
                    cY = int(cY)
                    centros.append([cX, cY])

            # Debug visualization: draw components, bounding boxes and centroids
            if debug:
                try:
                    vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                except Exception:
                    vis = cv2.merge([thresh, thresh, thresh])
                # draw each component's bounding box and centroid
                for i in range(1, numLabels):
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    y = int(stats[i, cv2.CC_STAT_TOP])
                    w = int(stats[i, cv2.CC_STAT_WIDTH])
                    h = int(stats[i, cv2.CC_STAT_HEIGHT])
                    a = int(stats[i, cv2.CC_STAT_AREA])
                    # choose color based on whether this component passed the filter
                    passed = (a < figarea / area_factor[0]) and (a > figarea / area_factor[1])
                    color = (0, 255, 0) if passed else (0, 0, 255)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
                    # draw centroid if available
                    try:
                        cx, cy = int(centroids[i][0]), int(centroids[i][1])
                        cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)
                    except Exception:
                        pass
                    # label index and area
                    cv2.putText(vis, f"#{i}:{a}", (x, max(y - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                # save visualization image for this slice
                try:
                    out_path = os.path.join(debug_dir, f"slice_{idx:03d}.png")
                    cv2.imwrite(out_path, vis)
                except Exception:
                    pass

            seeds.append(centros)
        return seeds

    def expand_seeds(self, xy_seeds):
        x_size = self.nifti_image.shape[0]
        x_range = int(x_size / (100 * abs(self.vox_dim[0])))
        for i in range(len(xy_seeds)):
            for j in range(len(xy_seeds[i])):
                current_seed = xy_seeds[i][j]
                for k in range(1, x_range):
                    xy_seeds[i].append([current_seed[0], current_seed[1] + 2 * k])
                    xy_seeds[i].append([current_seed[0], current_seed[1] - 2 * k])
                    xy_seeds[i].append([current_seed[0] - 2 * k, current_seed[1]])
                    xy_seeds[i].append([current_seed[0] - 2 * k, current_seed[1]])
                    xy_seeds[i].append(
                        [current_seed[0] - 2 * k, current_seed[1] + 2 * k]
                    )
                    xy_seeds[i].append(
                        [current_seed[0] - 2 * k, current_seed[1] - 2 * k]
                    )
                    xy_seeds[i].append(
                        [current_seed[0] + 2 * k, current_seed[1] + 2 * k]
                    )
                    xy_seeds[i].append(
                        [current_seed[0] + 2 * k, current_seed[1] - 2 * k]
                    )
        return xy_seeds

    def check_seeds(self, seeds, slices, organ):
        # removes seeds that are a different color from the chosen organ
        xy_seeds = []
        borders = []

        for j, image in enumerate(slices):
            image = np.array(image)

            # Find the limits of the body in the image
            cnts = imutils.grab_contours(
                cv2.findContours(
                    image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            )
            c = max(cnts, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            borders.append([extLeft, extRight, extTop, extBot])
            b_interval = int(0.03 * image.shape[0] * abs(self.vox_dim[0]))  # mm^2

            # remove the inverse pixel
            points_to_remove = []
            for i in range(len(seeds[j])):
                seed = seeds[j]
                if organ == 3 and image[seed[i][1]][seed[i][0]] == 1:
                    points_to_remove.append(i)
                if organ != 3 and image[seed[i][1]][seed[i][0]] == 0:
                    points_to_remove.append(i)
            for i in range(len(points_to_remove)):
                seed.pop(points_to_remove[i])
                points_to_remove = [item - 1 for item in points_to_remove]

            if len(seeds[j]) == 0:
                xy_seeds.append([])
            else:
                xy_seeds.append(seed)

        for j in range(len(xy_seeds)):
            i = 0
            while i < len(xy_seeds[j]):
                if (
                    xy_seeds[j][i][0] < borders[j][0][0] + b_interval
                    or xy_seeds[j][i][0] > borders[j][1][0] - b_interval
                    or xy_seeds[j][i][1] < borders[j][2][1] + b_interval
                    or xy_seeds[j][i][1] > borders[j][3][1] - b_interval
                ):
                    xy_seeds[j].pop(i)
                    i -= 1
                i += 1

        return xy_seeds

    def filter_by_boundary(self, xy_seeds, z_seeds):
        image_shape = self.nifti_image.shape[0]
        xy_traq = []

        # Gets the nearest seed from image center
        target = int(image_shape / 2)
        for j in range(len(xy_seeds)):
            if len(xy_seeds[j]) > 1:
                closest_coord = xy_seeds[j][0][0]
                for i in range(len(xy_seeds[j])):
                    if abs(xy_seeds[j][i][0] - target) <= abs(closest_coord - target):
                        closest_coord = xy_seeds[j][i][0]
                        closest_seed = [[xy_seeds[j][i][0], xy_seeds[j][i][1]]]
                seeds_within_radius = closest_seed  # Start with the closest seed
                for i in range(len(xy_seeds[j])):
                    # Calculate distance from the current seed to the closest seed
                    distance = (
                        (xy_seeds[j][i][0] - closest_seed[0][0]) ** 2
                        + (xy_seeds[j][i][1] - closest_seed[0][1]) ** 2
                    ) ** 0.5

                    if (
                        distance <= 5
                        and [xy_seeds[j][i][0], xy_seeds[j][i][1]]
                        not in seeds_within_radius
                    ):

                        seeds_within_radius.append(
                            [xy_seeds[j][i][0], xy_seeds[j][i][1]]
                        )  # Add seed if within radius
                xy_traq.append(seeds_within_radius)
            elif len(xy_seeds[j]) == 1:
                xy_traq.append([xy_seeds[j][0]])
            else:
                xy_traq.append([])

        x_coords, y_coords = [], []
        for i in range(len(xy_traq)):
            if len(xy_traq[i]) > 0:
                for j in range(len(xy_traq[i])):
                    x_coords.append(xy_traq[i][j][0])
                    y_coords.append(xy_traq[i][j][1])
        if x_coords:
            x_median = statistics.median(x_coords)
        else:
            x_median = image_shape[0] / 2
        if y_coords:
            y_median = statistics.median(y_coords)
        else:
            y_median = image_shape[1] / 2

        x_interval = 0.03 * image_shape
        x_coords = [
            x
            for x in x_coords
            if (x > x_median - x_interval) and (x < x_median + x_interval)
        ]
        y_coords = [
            y
            for y in y_coords
            if (y > y_median - x_interval) and (y < y_median + x_interval)
        ]
        min_traq = np.array([min(x_coords), min(y_coords)]) * 0.9
        max_traq = np.array([max(x_coords), max(y_coords)]) * 1.1

        # remove seeds outside of the trachea
        i = 0
        while i < len(xy_traq):
            if len(xy_traq[i]) > 0:
                for j in range(len(xy_traq[i])):
                    if (
                        xy_traq[i][j][0] < min_traq[0]
                        or xy_traq[i][j][0] > max_traq[0]
                        or xy_traq[i][j][1] < min_traq[1]
                        or xy_traq[i][j][1] > max_traq[1]
                    ):
                        del xy_traq[i][j]
            if len(xy_traq[i]) == 0:
                del xy_traq[i]
                del z_seeds[i]
            else:
                i += 1
        return xy_traq, z_seeds

    def fix_orientation(self, seeds):
        if self.affine[0][0] < 0:
            for i in range(len(seeds)):
                for j in range(len(seeds[i])):
                    seeds[i][j][0] = self.nifti_image.shape[0] - seeds[i][j][0]

        if self.affine[1][1] < 0:
            for i in range(len(seeds)):
                for j in range(len(seeds[i])):
                    seeds[i][j][1] = self.nifti_image.shape[1] - seeds[i][j][1]

        return seeds

    def constrain_seeds_within_bounds(
        self, xy_seeds, z_seeds, trachea_seeds, image_slices, filtered_slices
    ):

        z_trachea = trachea_seeds[:, -1]
        xy_trachea = trachea_seeds[:, :-1]
        # Trachea limitations for each slices
        left_seeds = []
        right_seeds = []

        for i, (filtered_image, image) in enumerate(zip(filtered_slices, image_slices)):
            image = np.array(image)
            filtered_image = np.array(filtered_image)

            left_seeds.append([])
            right_seeds.append([])
            nearest_trachea_slice = min(
                range(len(z_trachea)),
                key=lambda j: abs(z_trachea[j] - z_seeds[i]),
            )
            nearest_trachea_seed = xy_trachea[nearest_trachea_slice]
            # Find the limits of the body in the image
            cnts = cv2.findContours(
                filtered_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = imutils.grab_contours(cnts)
            max_cnts = max(cnts, key=cv2.contourArea)
            ext_left = tuple(max_cnts[max_cnts[:, :, 0].argmin()][0])
            ext_right = tuple(max_cnts[max_cnts[:, :, 0].argmax()][0])

            left_range = 0.2 * abs(nearest_trachea_seed[0] - ext_left[0])
            right_range = 0.2 * abs(nearest_trachea_seed[0] - ext_right[0])
            for j in range(len(xy_seeds[i])):
                xy_seed = xy_seeds[i][j]
                ppt_result = cv2.pointPolygonTest(max_cnts, xy_seed, False)

                if ppt_result > 0:
                    if (
                        image[xy_seed[1]][xy_seed[0]] == 0
                        and filtered_image[xy_seed[1]][xy_seed[0]] == 0
                    ):
                        if xy_seed[0] < nearest_trachea_seed[0] - left_range:
                            right_seeds[i].append(xy_seed)
                        if xy_seed[0] > nearest_trachea_seed[0] + right_range:
                            left_seeds[i].append(xy_seed)

        return left_seeds, right_seeds

    def merge_xyz_seeds(self, xy, z):
        xyz_seeds = []
        for xy_seeds, z_s in zip(xy, z):
            if xy_seeds:
                for xy_s in xy_seeds:
                    if xy_s:
                        xyz_seeds.append([xy_s[0], xy_s[1], z_s])
        return np.array(xyz_seeds)

    def run_trachea(self):
        thresh = self.highlight(-370)
        axial_samples, _, z_samples = self.sampling(thresh, 3)
        xy_seeds_by_area = self.seeds_by_area(axial_samples, (250, 10486))
        xy_limited = self.check_seeds(xy_seeds_by_area, axial_samples, 3)
        xy_filtered, z_filtered = self.filter_by_boundary(xy_limited, z_samples)
        final_xy_seeds = self.fix_orientation(xy_filtered)
        xyz_seeds = self.merge_xyz_seeds(final_xy_seeds, z_filtered)
        return xyz_seeds

    def run_lungs(self, trachea_seeds):
        thresh = self.highlight(-500)
        axial_samples, filtered_ax_slices, z_samples = self.sampling(thresh, 1)
        xy_seeds = self.seeds_by_area(axial_samples, (2.62, 260))
        xy_expanded_seeds = self.expand_seeds(xy_seeds)
        xy_expanded_seeds = self.fix_orientation(xy_expanded_seeds)
        left_seeds, right_seeds = self.constrain_seeds_within_bounds(
            xy_expanded_seeds,
            z_samples,
            trachea_seeds,
            axial_samples,
            filtered_ax_slices,
        )
        left_xyz_seeds = self.merge_xyz_seeds(left_seeds, z_samples)
        right_xyz_seeds = self.merge_xyz_seeds(right_seeds, z_samples)

        return left_xyz_seeds, right_xyz_seeds

    def run_external(self):
        thresh = self.highlight(0)
        axial_samples, _, z_samples = self.sampling(thresh, 2)
        xy_seeds = self.seeds_by_area(axial_samples, (44, 1310))
        xy_ribs = self.check_seeds(xy_seeds, axial_samples, 2)
        xy_ribs = self.fix_orientation(xy_ribs)
        xyz_seeds = self.merge_xyz_seeds(xy_ribs, z_samples)
        return xyz_seeds

    def save_seeds(self, internals, externals, file_path, marker):
        seeds_to_include = [0]
        with open(file_path, "w") as seed_file:
            for internal in internals:
                for seed in internal:
                    seeds_to_include.append(
                        f"{seed[0]} {seed[1]} {seed[2]} {marker} 1\r"
                    )
            for external in externals:
                for seed in external:
                    seeds_to_include.append(
                        f"{seed[0]} {seed[1]} {seed[2]} {marker} 0\r"
                    )
            xy_range = np.arange(
                int(self.nifti_image.shape[0] * 0.1),
                int(self.nifti_image.shape[0] * 0.9),
                10,
            )
            z_range = np.arange(1, self.nifti_image.shape[2], 20)

            for z in z_range:
                for xy in xy_range:
                    seeds_to_include.append(f"5 {xy} {z} 8 0\r")
                    seeds_to_include.append(f"{xy} 5 {z} 8 0\r")
                    seeds_to_include.append(f"5 5 {z} 8 0\r")
            seeds_to_include = np.unique(seeds_to_include)

            seeds_to_include[0] = f"{len(seeds_to_include) - 1}\r"
            seed_file.writelines(seeds_to_include)
        seed_file.close()

        return file_path
