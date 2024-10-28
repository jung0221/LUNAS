import numpy as np
from stl import mesh
import os
import logging
import platform
import time
import cv2
import imutils
import nibabel as nib
from PIL import Image
import argparse
from skimage import measure
import subprocess
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
logging.getLogger("stl").setLevel(logging.ERROR)
import statistics
from scipy.ndimage import zoom
import glob
import matplotlib.pyplot as plt


def search_dir_dcm (path):
    '''Recebe arquivo de texto com os diretorios, inverte barras e retorna em forma de lista'''
    files = open(path, 'r')
    files = files.readlines()
    for i in files:
        i.replace(os.sep,"/")
    return files

def str_to_list(exc):
    lis = []
    for l in exc:
        if "-" in l:
            li = list(map(int, l.split("-")))
            for a in range(li[0],li[1]+1):
                lis.append(a)
        else:
            lis.append(int(l))

    return lis

def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def highlight(nifti_file, outputfolder, patient, marker, organ):
    # nifti_file = nib.load(image)
    afim = nifti_file.affine
    nifti_image = nifti_file.get_fdata().copy()
    if organ == 1:
        nifti_image[nifti_image >= -500] = 1
        nifti_image[nifti_image < -500] = 0
    elif organ == 2:
        nifti_image[nifti_image < 100] = 0
        nifti_image[nifti_image >= 100] = 1
    elif organ == 3:
        nifti_image[nifti_image >= -370] = 1
        nifti_image[nifti_image < -370] = 0
    elif organ == 4:
        nifti_image[nifti_image >= -200] = 1
        nifti_image[nifti_image < -200] = 0
    elif organ == 5:
        nifti_image[nifti_image >= 100] = 0
        nifti_image[nifti_image < 15] = 0
        nifti_image[nifti_image >= 15] = 1

    matriz = nifti_image.astype(np.int16)
    nifti = nib.Nifti1Image(matriz, afim)
    if organ == 1:
        nib.loadsave.save(nifti, outputfolder + "lungs-" + patient + ".nii")
    
    # Different thresholds in upper side of the dicom image
    if organ == 2 and marker == 1:
        # nifti_file = nib.load(image)
        afim = nifti_file.affine
        nifti_image = nifti_file.get_fdata()
        for i in range(int(nifti_image.shape[2]/5)):
            z_axis = nifti_image.shape[2]-i-1
            slice = nifti_image[:,:,z_axis]
            slice[slice >= 150] += 1000
            nifti_image[:,:,z_axis] = slice
        for i in range(int(nifti_image.shape[2]/4), nifti_image.shape[2]-int(nifti_image.shape[2]/5)):
            slice = nifti_image[:,:,i]
            slice[slice >= 80] += 1000
            nifti_image[:,:,i] = slice
        for i in range(0, int(nifti_image.shape[2]/4)):
            slice = nifti_image[:,:,i]
            slice[slice >= 120] += 1000
            nifti_image[:,:,i] = slice

        matriz = nifti_image.astype(np.int16)
        nifti_file = nib.Nifti1Image(matriz, afim)
        nib.loadsave.save(nifti_file, outputfolder + "ribs-" + patient + ".nii")

    return nifti

def remove_noises_slice(img):
    img = 1 - img
    components, matrix, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; components = components - 1
    min_size = int(img.shape[0]/3.4)
    img2 = np.zeros((matrix.shape))
    for j in range(0, components):
        if sizes[j] >= min_size:
            img2[matrix == j + 1] = 255
    img2 = 1 - img2
    return img2

def remove_noises_airways(image):
    image = 1 - image
    mesh_stl = np.zeros(image.shape)
    for i in range(image.shape[2]):
        img = np.array(image[:,:,i], dtype=np.uint8)
        components, matrix, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:, -1]; components = components - 1
        min_size = 500
        img2 = np.zeros((matrix.shape))
        for j in range(0, components):
            if sizes[j] <= min_size:
                img2[matrix == j + 1] = 255
        mesh_stl[:,:,i] = img2
    mesh_stl = mesh_stl.astype(np.int16)
    return mesh_stl

def slicing(modified_nifti, patient_name, outputfolder, organ):
    matriz = modified_nifti.get_fdata()    
    afim = modified_nifti.affine
    z_list = []
    if organ == 1 or organ == 4:
        for j in range(10):
            z_value = int((j+10)*matriz.shape[2]/22)
            z_list.append(z_value)
            matriz[:,:,z_value] = np.rot90(matriz[:,:,z_value])
            
            matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)

            if afim[0][0] < 0:
                matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=1)
            if afim[1][1] < 0:
                matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)
            image = Image.fromarray(matriz[:,:,z_value]*255)
            image = image.convert("L")
            image.save(outputfolder + patient_name + "_image-lungs_{}.png".format(z_value))

            img = np.array(matriz[:,:,z_value], dtype=np.uint8)
            img2 = remove_noises_slice(img)*255
            kernel = np.ones((5, 5), np.uint8)
            change_times = int(matriz.shape[0]/170)
            for i in range(1, change_times):
                img2 = cv2.erode(img2, kernel)
            for i in range(1, change_times):
                img2 = cv2.dilate(img2, kernel)
            image = Image.fromarray(img2)
            image = image.convert("L")
            image.save(outputfolder + patient_name + "_modified_image-lungs_{}.png".format(z_value))
    elif organ == 2:
        for j in range(10):
            z_value = int((j+5)*matriz.shape[2]/15)
            z_list.append(z_value)

            matriz[:,:,z_value] = np.rot90(matriz[:,:,z_value])
            matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)

            if afim[0][0] < 0:
                matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=1)
            if afim[1][1] < 0:
                matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)
            
            img = np.array(matriz[:,:,z_value], dtype=np.uint8)
            img2 = remove_noises_slice(img)*255

            image = Image.fromarray(img2)
            image = image.convert("L")
            image.save(outputfolder + patient_name + "_image-ribs_{}.png".format(z_value))
    elif organ == 3:
        for j in range(30):
            z_value = int(matriz.shape[2]-(j+1)*matriz.shape[2]/65)
            z_list.append(z_value)
            matriz[:,:,z_value] = np.rot90(matriz[:,:,z_value])
            matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)

            if afim[0][0] < 0:
                matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=1)
            if afim[1][1] < 0:
                matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)
            
            img = np.array(matriz[:,:,z_value], dtype=np.uint8)
            img2 = remove_noises_slice(img)*255

            image = Image.fromarray(img2*255)
            image = image.convert("L")
            image.save(outputfolder + patient_name + "_image-airways_{}.png".format(z_value))
            
    elif organ == 5:
        for j in range(3):
            z_value = int(matriz.shape[2]*(j+4)/8)
            z_list.append(z_value)
            matriz[:,:,z_value] = np.rot90(matriz[:,:,z_value])
            matriz[:,:,z_value] = np.flip(matriz[:,:,z_value], axis=0)
            image = Image.fromarray(matriz[:,:,z_value])
            image = image.convert("L")
            image.save(outputfolder + patient_name + "_image-heart_{}.png".format(z_value))

    return z_list

def seeds_by_area(z_list, voxel_dim, patient_name, outputfolder, organ):
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scharr", type=int, default=0, help="path to input image")
    ap.add_argument("-c", "--connectivity", type=int, default=4, help="connectivity for connected component analysus")
    args = vars(ap.parse_args())
    image_sobel = []
    for j in range(len(z_list)):
        if organ == 1 or organ == 4:
            image = cv2.imread(outputfolder + patient_name + "_image-lungs_{}.png".format(z_list[j]))
        elif organ == 2:
            image = cv2.imread(outputfolder + patient_name + "_image-ribs_{}.png".format(z_list[j]))
        elif organ == 3:
            image = cv2.imread(outputfolder + patient_name + "_image-airways_{}.png".format(z_list[j]))
        elif organ == 5:
            image = cv2.imread(outputfolder + patient_name + "_image-heart_{}.png".format(z_list[j]))
        elif organ == 6:
            image = cv2.imread(outputfolder + patient_name + "_image-liver.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = -1 if args["scharr"] > 0 else 3
        gX = cv2.convertScaleAbs(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize))
        gY = cv2.convertScaleAbs(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize))
        combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        image_sobel.append(combined)
    seeds_xy_organ = []
    for j in range(len(image_sobel)):
        thresh = cv2.threshold(image_sobel[j], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
        output = cv2.connectedComponentsWithStats(thresh, args["connectivity"], cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        figarea = abs(thresh.shape[0]*thresh.shape[1]/(voxel_dim[0]*voxel_dim[1])) #in mm^2
        centros = []
        for i in range(0, numLabels):

            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            if organ == 1:
                if (area > figarea/260 and area < figarea/2.62):
                    cX = int(cX)
                    cY = int(cY)
                    centros.append([cX, cY])
            if organ == 2:
                if (area > figarea/1310 and area < figarea/44):
                    cX = int(cX)
                    cY = int(cY)
                    centros.append([cX, cY])
            if organ == 3:
                if (area < figarea/250 and area > figarea/10486):
                    cX = int(cX)
                    cY = int(cY)
                    centros.append([cX, cY])
            # if organ == 4:
            #     if area > 100:
            #         if (area < 1000 or area > 20000):
            #             cX = int(cX)
            #             cY = int(cY)
            #             centros.append([cX,cY])
            # if organ == 5:
            #     if area > 2000 and area < 200000:
            #         cX = int(cX)
            #         cY = int(cY)
            #         centros.append([cX,cY])
            # if organ == 6:
            #     if area > 1000:
            #         cX = int(cX)
            #         cY = int(cY)
            #         centros.append([cX,cY])
        seeds_xy_organ.append(centros)
    return seeds_xy_organ

def expand_seeds(seeds, z_value, vox_dim, patient_name, outputfolder):
    fig = cv2.imread(outputfolder + patient_name + "_image-lungs_{}.png".format(z_value[0]))
    x_size = fig.shape[0]
    x_range = int(x_size/(100*abs(vox_dim[0])))
    for i in range(len(seeds)):
        for j in range(len(seeds[i])):
            selected_seed = seeds[i][j]
            for k in range(1, x_range):
                seeds[i].append([selected_seed[0],   selected_seed[1]+2*k])
                seeds[i].append([selected_seed[0],   selected_seed[1]-2*k])
                seeds[i].append([selected_seed[0]-2*k, selected_seed[1]  ])
                seeds[i].append([selected_seed[0]-2*k, selected_seed[1]  ])
                seeds[i].append([selected_seed[0]-2*k, selected_seed[1]+2*k])
                seeds[i].append([selected_seed[0]-2*k, selected_seed[1]-2*k])
                seeds[i].append([selected_seed[0]+2*k, selected_seed[1]+2*k])
                seeds[i].append([selected_seed[0]+2*k, selected_seed[1]-2*k])
    return seeds

def check_seeds(seeds_xy, z_values, voxel_dim, patient_name, outputfolder, organ):
    sum = 0
    # removes seeds that are a different color from the chosen organ
    xy_list = []
    borders = []
    
    if organ == 2:
        z_lungforribs = z_values[1] 
        z_values = z_values[0]
        
    for j in range(len(z_values)):
        if organ == 2:
            image = cv2.imread(outputfolder + patient_name + "_image-ribs_{}.png".format(z_values[j]))
            image_borders = cv2.imread(outputfolder + patient_name + "_modified_image-lungs_{}.png".format(z_lungforribs[j]))
        elif organ == 3:
            image = cv2.imread(outputfolder + patient_name + "_image-airways_{}.png".format(z_values[j]))
        elif organ == 4:
            image = cv2.imread(outputfolder + patient_name + "_image-lungs_{}.png".format(z_values[j]))
        if organ > 2:
            image_borders = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_borders = cv2.cvtColor(image_borders, cv2.COLOR_BGR2GRAY)
        # Find the limits of the body in the image
        cnts = imutils.grab_contours(cv2.findContours(gray_borders.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        borders.append([extLeft, extRight, extTop, extBot])
        b_interval = int(0.03*gray.shape[0]*abs(voxel_dim[0])) # mm^2

        # remove the inverse pixel
        points_to_remove = []
        for i in range(len(seeds_xy[j])):
            seed = seeds_xy[j]
            if organ == 3:
                if gray[seed[i][1]][seed[i][0]] == 1:
                    points_to_remove.append(i)
            else:
                if gray[seed[i][1]][seed[i][0]] == 0:
                    points_to_remove.append(i)
        for i in range(len(points_to_remove)):
            seed.pop(points_to_remove[i])
            points_to_remove = [item-1 for item in points_to_remove]

        height, width = gray.shape[:2]

        if len(seeds_xy[j]) == 0:
            xy_list.append([])
        else:
            xy_list.append(seed)

    for j in range(len(xy_list)):
        sum += len(xy_list[j])
        i = 0
        while i < len(xy_list[j]):
            if (xy_list[j][i][0] < borders[j][0][0]+b_interval or xy_list[j][i][0] > borders[j][1][0]-b_interval or xy_list[j][i][1] < borders[j][2][1]+b_interval or xy_list[j][i][1] > borders[j][3][1]-b_interval):
                xy_list[j].pop(i)
                i -= 1
                sum -= 1
            i += 1

    return xy_list, sum

def coord_check_airways(xy_list, nifti_file):
    image_shape = nifti_file.shape[0]
    xy_traq = []
    # pega a semente (x) mais próxima do centro da imagem
    target = int(image_shape/2)
    for j in range(len(xy_list)):
        if len(xy_list[j]) > 1:
            closest_coord = xy_list[j][0][0]
            for i in range(len(xy_list[j])):
                if abs(xy_list[j][i][0]-target) <= abs(closest_coord-target):
                    closest_coord = xy_list[j][i][0]
                    closest_seed = [[xy_list[j][i][0], xy_list[j][i][1]]]
            seeds_within_radius = closest_seed  # Start with the closest seed
            for i in range(len(xy_list[j])):
                # Calculate distance from the current seed to the closest seed
                distance = ((xy_list[j][i][0] - closest_seed[0][0]) ** 2 + (xy_list[j][i][1] - closest_seed[0][1]) ** 2) ** 0.5
                if distance <= 5 and [xy_list[j][i][0], xy_list[j][i][1]] not in seeds_within_radius:
                    seeds_within_radius.append([xy_list[j][i][0], xy_list[j][i][1]])  # Add seed if within radius
            xy_traq.append(seeds_within_radius)
        elif len(xy_list[j]) == 1:
            xy_traq.append([xy_list[j][0]])
        else: xy_traq.append([])

    x_coords, y_coords = [], []
    for i in range(len(xy_traq)):
        if len(xy_traq[i]) > 0:
            for j in range(len(xy_traq[i])):
                x_coords.append(xy_traq[i][j][0])
                y_coords.append(xy_traq[i][j][1])
    x_median = statistics.median(x_coords)
    y_median = statistics.median(y_coords)
    nmatrix = nifti_file.get_fdata()
    x_interval = 0.03*nmatrix.shape[0]
    x_coords = [x for x in x_coords if (x > x_median - x_interval) and (x < x_median + x_interval)]
    y_coords = [y for y in y_coords if (y > y_median - x_interval) and (y < y_median + x_interval)]
    min_traq = np.array([min(x_coords), min(y_coords)])*0.9
    max_traq = np.array([max(x_coords), max(y_coords)])*1.1
    # remove seeds outside of the trachea
    new_xy_traq = []
    for i in range(len(xy_traq)):
        new_xy_traq.append([])
        if len(xy_traq[i]) > 0:
            for j in range(len(xy_traq[i])):
                if xy_traq[i][j][0] > min_traq[0] and xy_traq[i][j][0] < max_traq[0] and xy_traq[i][j][1] > min_traq[1] and xy_traq[i][j][1] < max_traq[1]:
                    new_xy_traq[i].append(xy_traq[i][j])

    return new_xy_traq

def open_notepad(sum, outputfolder, patient_name, organ):
    if organ == 1.0:
        notepad = open(outputfolder + "lungs_R-{}.txt".format(patient_name), "w")
    elif organ == 1.5:
        notepad = open(outputfolder + "lungs_L-{}.txt".format(patient_name), "w")
    elif organ == 2:
        notepad = open(outputfolder + "ribs-{}.txt".format(patient_name), "w")
    elif organ == 3:
        notepad = open(outputfolder + "airways-{}.txt".format(patient_name), "w")
    elif organ == 4:
        notepad = open(outputfolder + "skin-{}.txt".format(patient_name), "w")
    elif organ == 5:
        notepad = open(outputfolder + "heart-{}.txt".format(patient_name), "w")
    notepad.write("{}\r".format(sum))
    return notepad

def write_internal_seeds(notepad, internal):
    for k in range(len(internal[0])):
        for j in range(len(internal[0][k])):
            notepad.write("{} {} {} 8 1\r".format(internal[0][k][j][0], internal[0][k][j][1], internal[1][k]))

def write_external_seeds(notepad, external, nifti_file):
    nmatrix = nifti_file.get_fdata()
    border_xy = int(nmatrix.shape[0]*0.9)
    border_z = int(nmatrix.shape[2])
    for i in range(1, border_z):
        notepad.write("{} {} {} 8 0\r".format(border_xy, border_xy, i))
        notepad.write("{} 5 {} 8 0\r".format(border_xy, border_xy, i))
        notepad.write("5 {} {} 8 0\r".format(border_xy, border_xy, i))
        notepad.write("5 5 {} 8 0\r".format(i))

    for k in range(len(external[0])):
        for j in range(len(external[0][k])):
            notepad.write("{} {} {} 8 0\r".format(external[0][k][j][0], external[0][k][j][1], external[1][k]))
    return

def stl_conversor(image):
    # Converte a matriz binária para um array STL

    verts, faces, _, _ = measure.marching_cubes(image, method='lorensen')
    mesh_stl = mesh.Mesh(np.zeros(len(verts[faces]), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_stl.vectors[i][j] = verts[f[j],:]
    return mesh_stl

def reshape(your_mesh, img):
    header = img.header.get_zooms()
    # Redimensiona o stl
    your_mesh.x*=header[0]
    your_mesh.y*=header[1]
    your_mesh.z*=header[2]
    return your_mesh

def read_label(patient_name, outputfolder):
    mkdir_p(outputfolder)
    img = nib.load("label.nii.gz")
    img_array = np.array(img.dataobj)
    mes = stl_conversor(img_array)
    organ = reshape(mes, img)
    organ.save(outputfolder + '/{}.stl'.format(patient_name))
    return

def fix_orientation(xy_seeds, afim, shape):
    if afim[0][0] < 0:
        for i in range(len(xy_seeds)):
            for j in range(len(xy_seeds[i])):
                xy_seeds[i][j][0] = shape[0] - xy_seeds[i][j][0]

    if afim[1][1] < 0:
        for i in range(len(xy_seeds)):
            for j in range(len(xy_seeds[i])):
                xy_seeds[i][j][1] = shape[1] - xy_seeds[i][j][1]

    return xy_seeds


def check_seeds_lung(xy_seeds, z_seeds, traq_seeds, outputfolder, patient_name):
    z_traq_seeds = traq_seeds[1]
    xy_traq_seeds = traq_seeds[0]
    remove_index = []
    nxy_traq_seeds = []
    nz_traq_seeds = []

    # Remove empty list of seeds in trachea
    for i in range(len(xy_traq_seeds)):
        if len(xy_traq_seeds[i]) >= 1:
            nxy_traq_seeds.append(xy_traq_seeds[i])
            nz_traq_seeds.append(z_traq_seeds[i])
        else:
            remove_index.append(i)

    # Trachea limitations for each slices 
    left_seeds = []
    right_seeds = []
    l_sum = 0
    r_sum = 0
    
    for i in range(len(z_seeds)):
        left_seeds.append([])
        right_seeds.append([])
        close_index = min(range(len(nz_traq_seeds)), key=lambda j: abs(nz_traq_seeds[j] - z_seeds[i]))
        
        traq_limit = nxy_traq_seeds[close_index][0]
        

        image = cv2.imread(outputfolder + patient_name + "_image-lungs_{}.png".format(z_seeds[i]))
        image_borders = cv2.imread(outputfolder + patient_name + "_modified_image-lungs_{}.png".format(z_seeds[i]))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_borders = cv2.cvtColor(image_borders, cv2.COLOR_BGR2GRAY)
        # Find the limits of the body in the image
        # Douglas - Peucker
        cnts = cv2.findContours(gray_borders.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])

        L_range = 0.2*abs(traq_limit[0]-extLeft[0])
        R_range = 0.2*abs(traq_limit[0]-extRight[0])
        for j in range(len(xy_seeds[i])):
            xy_seed = xy_seeds[i][j]

            result = cv2.pointPolygonTest(c, xy_seed, False)

            if result > 0:
                if gray[xy_seed[1]][xy_seed[0]] == 0 and gray_borders[xy_seed[1]][xy_seed[0]] == 0:
                    if xy_seed[0] < traq_limit[0]-L_range:
                        right_seeds[i].append(xy_seed)
                    if xy_seed[0] > traq_limit[0]+R_range:
                        left_seeds[i].append(xy_seed)
        l_sum += len(left_seeds[i])
        r_sum += len(right_seeds[i])
        
    return left_seeds, right_seeds, l_sum, r_sum

def airways_seeds(nifti_file, outputfolder, patient_name):
    organ = 3
    afim = nifti_file.affine
    voxel_dimensions = afim[:3, :3].diagonal() # mm^2
    modified_nifti = highlight(nifti_file, outputfolder, patient_name, 0, organ)
    z_traq_seeds = slicing(modified_nifti, patient_name, outputfolder, organ)
    xy_traq = seeds_by_area(z_traq_seeds, voxel_dimensions, patient_name, outputfolder, organ)
    xy_traq, sum = check_seeds(xy_traq, z_traq_seeds, voxel_dimensions, patient_name, outputfolder, organ)
    xy_traq = coord_check_airways(xy_traq, nifti_file)

    new_xy_traq = fix_orientation(xy_traq, afim, nifti_file.get_fdata().shape)
    return [new_xy_traq, z_traq_seeds, sum]


def lung_seeds(nifti_file, outputfolder, patient_name, traq_seeds):
    organ = 1
    afim = nifti_file.affine
    voxel_dimensions = afim[:3, :3].diagonal() # mm^2

    modified_nifti = highlight(nifti_file, outputfolder, patient_name, 0, organ)
    z_lung_seeds = slicing(modified_nifti, patient_name, outputfolder, organ)
    xy_list = seeds_by_area(z_lung_seeds, voxel_dimensions, patient_name, outputfolder, organ)
    xy_list = expand_seeds(xy_list, z_lung_seeds, voxel_dimensions, patient_name, outputfolder)
    traq_seeds[0] = fix_orientation(traq_seeds[0], afim, nifti_file.get_fdata().shape)
    left_seeds, right_seeds, l_sum, r_sum = check_seeds_lung(xy_list, z_lung_seeds, traq_seeds, outputfolder, patient_name)
    left_seeds = fix_orientation(left_seeds, afim, nifti_file.get_fdata().shape)
    right_seeds = fix_orientation(right_seeds, afim, nifti_file.get_fdata().shape)
    traq_seeds[0] = fix_orientation(traq_seeds[0], afim, nifti_file.get_fdata().shape)

    return [left_seeds, z_lung_seeds, l_sum], [right_seeds, z_lung_seeds, r_sum]

def ribs_seeds(nifti_file, outputfolder, patient_name, z_lung):
    organ = 2
    afim = nifti_file.affine
    voxel_dimensions = afim[:3, :3].diagonal() # mm^2
    modified_nifti = highlight(nifti_file, outputfolder, patient_name, 0, organ)
    highlight(nifti_file, outputfolder, patient_name, 1, organ)
    z_ribs_seeds = slicing(modified_nifti, patient_name, outputfolder, organ)
    xy_list = seeds_by_area(z_ribs_seeds, voxel_dimensions, patient_name, outputfolder, organ)
    z_ribs = [z_ribs_seeds, z_lung]
    xy_ribs, sum = check_seeds(xy_list, z_ribs, voxel_dimensions, patient_name, outputfolder, organ)
    xy_ribs = fix_orientation(xy_ribs, afim, nifti_file.get_fdata().shape)
    return [xy_ribs, z_ribs_seeds, sum]


def skin_seeds(nifti_file, outputfolder, patient_name, to_heart):
    organ = 4
    afim = nifti_file.affine
    voxel_dimensions = afim[:3, :3].diagonal() # mm^2
    modified_nifti = highlight(nifti_file, outputfolder, patient_name, 0, organ)
    # noise_removed = remove_noises(modified_nifti, patient_name, outputfolder, organ)
    z_skin_seeds = slicing(modified_nifti, patient_name, outputfolder, organ)
    xy_list = seeds_by_area(z_skin_seeds, voxel_dimensions, patient_name, outputfolder, organ)
    xy_skin, sum = check_seeds(xy_list, z_skin_seeds, voxel_dimensions, patient_name, outputfolder, organ)
    
    return [xy_skin, z_skin_seeds, sum]

def compress_binary_array(arr, new_shape):
    old_shape = np.array(arr.shape)
    compress_factor = new_shape / old_shape

    new_shape = zoom(arr, compress_factor, order = 0)

    return new_shape.astype(np.uint8)

def compare_images(original_matrix, original_affine, contender_lung, patient_name, segmentationfolder, organ):
    # original_matrix = original_lung.get_fdata()
    original_matrix = original_matrix.astype(np.int16)
    original_matrix[original_matrix >= 1] = 1
    contender_matrix = contender_lung.get_fdata()
    contender_matrix[contender_matrix >= 1] = 1
    if contender_matrix.shape != original_matrix.shape:
        contender_matrix = compress_binary_array(contender_matrix, (original_matrix.shape[0], original_matrix.shape[1], original_matrix.shape[2]))
        nib.loadsave.save(nib.Nifti1Image(contender_matrix.astype(np.int16), original_affine), segmentationfolder + organ + patient_name + "_compressed_.nii")

    original_matrix = original_matrix.flatten()
    contender_matrix = contender_matrix.flatten()
    
    f1 = f1_score(original_matrix, contender_matrix)
    return f1

def SEED_GEN(nifti_file, outputfolder, patient_name):
    nifti_file = nib.load(nifti_file)

    afim = nifti_file.affine
    
    # voxel_to_mm(nifti_file)
    # Lung
    print("Generating Seeds...")
    traq = airways_seeds(nifti_file, outputfolder, patient_name)
    l_lung, r_lung = lung_seeds(nifti_file, outputfolder, patient_name, traq.copy())
    ribs = ribs_seeds(nifti_file, outputfolder, patient_name, l_lung[1])
    skin = skin_seeds(nifti_file, outputfolder, patient_name, False)
    sum_skin = skin[2] + l_lung[2] + r_lung[2] + ribs[2] + 2
    sum_lung = l_lung[2] + r_lung[2] + traq[2] + ribs[2] + 2
    notepad_lung_left = open_notepad(sum_lung, outputfolder, patient_name, 1)
    notepad_lung_right = open_notepad(sum_lung, outputfolder, patient_name, 1.5)
    notepad_airways = open_notepad(sum_lung, outputfolder, patient_name, 3)
    notepad_ribs = open_notepad(sum_lung, outputfolder, patient_name, 2)
    notepad_skin = open_notepad(sum_skin, outputfolder, patient_name, 4)

    # Lungs
    write_internal_seeds(notepad_lung_left, l_lung)
    write_internal_seeds(notepad_lung_right, r_lung)
    write_external_seeds(notepad_lung_left, r_lung, nifti_file)
    write_external_seeds(notepad_lung_right, l_lung, nifti_file)
    write_external_seeds(notepad_lung_left, ribs, nifti_file)
    write_external_seeds(notepad_lung_right, ribs, nifti_file)
    write_external_seeds(notepad_lung_left, traq, nifti_file)
    write_external_seeds(notepad_lung_right, traq, nifti_file)

    # Traquea
    write_internal_seeds(notepad_airways, traq)
    write_external_seeds(notepad_airways, ribs, nifti_file)
    write_external_seeds(notepad_airways, l_lung, nifti_file)
    write_external_seeds(notepad_airways, r_lung, nifti_file)

    # Ribs
    write_internal_seeds(notepad_ribs, ribs)
    write_external_seeds(notepad_ribs, traq, nifti_file)
    write_external_seeds(notepad_ribs, l_lung, nifti_file)
    write_external_seeds(notepad_ribs, r_lung, nifti_file)

    # Skin
    write_external_seeds(notepad_skin, skin, nifti_file)
    write_internal_seeds(notepad_skin, l_lung)
    write_internal_seeds(notepad_skin, r_lung)
    write_internal_seeds(notepad_skin, ribs)
    
    notepad_lung_left.close()
    notepad_lung_right.close()
    notepad_airways.close()
    notepad_ribs.close()
    notepad_skin.close()


    # remove wrong and duplicates for lungs and trachea
    sides = ['lungs_R-', 'lungs_L-', 'airways-', 'ribs-', 'skin-']
    for side in sides:
        with open(outputfolder + side + patient_name + '.txt', 'r') as file:
            lines = file.readlines()
            seed_to_remove = []
            for i in range(len(lines)):
                if len(lines[i]) < 10:
                    seed_to_remove.append(i)
            for i in range(len(seed_to_remove)):
                lines.pop(seed_to_remove[i])
            # remove duplicates
            lines = np.unique(lines)
            seeds = len(set(lines)) - 1
            lines[0] = str(seeds) + '\n'
        with open(outputfolder + side + patient_name + '.txt', 'w') as output_file:
            output_file.writelines(lines)
        file.close()
        output_file.close()

def send_to_linux(nFilePath, patient_name, outputfolder, wslfolder, roiftfolder):
    anifti = nib.load(nFilePath)
    aaffine = anifti.affine
    amatrix = anifti.get_fdata()
    amatrix[amatrix > 500] = 500
    amatrix[amatrix < -3000] = -1024
    nib.loadsave.save(nib.Nifti1Image(amatrix.astype(np.int16), aaffine), outputfolder + patient_name + '.nii')
    # Transfer the .nii and .txt files into linux
    lista_items = os.listdir(outputfolder)
    for k in range(len(lista_items)):
        if (patient_name in lista_items[k] and '.txt' in lista_items[k]) or  (patient_name in lista_items[k] and '.nii' in lista_items[k]):

            seed = "wsl cp -ar " + wslfolder + lista_items[k] + " " + roiftfolder
            p = subprocess.run(seed, shell=True)
    return


def ROIFT(roiftfolder, mcubesfolder, segmentationfolder, patient_name, percentile, nitter, mode, pol):
    print("perc: ", percentile)

    # mkdir_p(segmentationfolder)
    # print("Trachea ROIFT...")
    # file = "wsl " + roiftfolder + 'oiftrelax' + " " + roiftfolder + patient_name + ".nii" + " " + roiftfolder + "airways-" + patient_name + '.txt ' + str(-1.0) + ' 0' + ' ' + str(percentile)
    # p = subprocess.run(file, shell=True)
    # trachea_nifti = nib.load("label.nii.gz")
    # afim = trachea_nifti.affine
    # trachea_array = np.array(trachea_nifti.dataobj)
    # nib.loadsave.save(nib.Nifti1Image(trachea_array.astype(np.int16), afim), segmentationfolder + "airways-" + patient_name + ".nii")
    # read_label("airways-" + patient_name, mcubesfolder + nitter + patient_name)

    # print("Left Lung ROIFT...")
    # file = "wsl " + roiftfolder + 'oiftrelax' + " " + roiftfolder + patient_name + ".nii" + " " + roiftfolder + "lungs_L-" + patient_name + '.txt ' + str(-pol) + ' ' + nitter + ' ' + str(percentile)
    # p = subprocess.run(file, shell=True)
    # right_nifti = nib.load("label.nii.gz")
    # afim = right_nifti.affine
    # right_array = np.array(right_nifti.dataobj)
    # # right_array -= trachea_array
    # right_array[right_array < 0] = 0
    # nib.loadsave.save(nib.Nifti1Image(right_array.astype(np.int16), afim), segmentationfolder + "lungs_left-" + patient_name + ".nii")
    # read_label("lungs_left-" + patient_name, mcubesfolder + nitter + patient_name)

    # print("Right Lung ROIFT...")
    # file = "wsl " + roiftfolder + 'oiftrelax' + " " + roiftfolder + patient_name + ".nii" + " " + roiftfolder + "lungs_R-" + patient_name + '.txt ' + str(-pol) + ' ' + nitter + ' ' + str(percentile)
    # p = subprocess.run(file, shell=True)
    # left_nifti = nib.load("label.nii.gz")
    # afim = left_nifti.affine
    # left_array = np.array(left_nifti.dataobj)
    # # left_array -= trachea_array
    # left_array[left_array < 0] = 0
    # nib.loadsave.save(nib.Nifti1Image(left_array.astype(np.int16), afim), segmentationfolder + "lungs_right-" + patient_name + ".nii")
    # read_label("lungs_right-" + patient_name, mcubesfolder + nitter + patient_name)



    print("Ribs ROIFT...")
    file = "wsl " + roiftfolder + 'oiftrelax_2gb_1dc' + " " + roiftfolder + "ribs-" + patient_name + ".nii" + " " + roiftfolder + "ribs-" + patient_name + '.txt' + ' 1.0 ' + nitter + ' ' + str(percentile)
    p = subprocess.run(file, shell=True)

    ribs_nifti = nib.load("label.nii.gz")
    ribs_array = np.array(ribs_nifti.dataobj)
    afim = ribs_nifti.affine
    nib.loadsave.save(nib.Nifti1Image(ribs_array.astype(np.int16), afim), segmentationfolder + "ribs-" + patient_name + ".nii")   
    read_label("ribs-" + patient_name, mcubesfolder + nitter + patient_name) 


def mean_accs(inputfolder, txt_name):
    with open(inputfolder + txt_name, 'r') as file:
        lines = file.readlines()
    dic_L = np.zeros(int(len(lines)/2))
    dic_R = np.zeros(int(len(lines)/2))

    for i in range(len(lines)):
        if i%2 == 1:
            dic_L[int(i/2)] = float(lines[i].split()[0].replace("DL", ""))
            dic_R[int(i/2)] = float(lines[i].split()[1].replace("DR", ""))
    print(dic_L.mean())
    print(dic_R.mean())

    return dic_L.mean(), dic_R.mean()

def VAL(answerfolder, segmentationfolder, notepad_dil, patient_name, dataset):
    print("Validation: ")
    if dataset != "LCTSC":
        # Gabarito
        ans = nib.load(answerfolder + patient_name + ".nii")
        ans_R = ans.get_fdata().copy()
        ans_R[ans_R != 1] = 0
        ans_L = ans.get_fdata().copy()
        ans_L[ans_L != 2] = 0
        ans_A = ans.get_fdata().copy()
        ans_A[ans_A != 3] = 0
    else: 
        ans = nib.load(answerfolder + "L_" + patient_name + ".nii.gz")
        ans_L = nib.load(answerfolder + "L_" + patient_name + ".nii.gz").get_fdata().copy()
        ans_R = nib.load(answerfolder + "R_" + patient_name + ".nii.gz").get_fdata().copy()
    
    contender_lung_L = nib.load(segmentationfolder + "lungs_left-" + dataset + "-" + patient_name + ".nii")
    contender_lung_R = nib.load(segmentationfolder + "lungs_right-" + dataset + "-" + patient_name + ".nii")
    if dataset != "LCTSC":
        contender_lung_A = nib.load(segmentationfolder + "airways-" + dataset + "-" + patient_name + ".nii")
        f1_A = compare_images(ans_A, ans.affine, contender_lung_A, patient_name, segmentationfolder, "airways-")
    
    f1_L = compare_images(ans_L, ans.affine, contender_lung_L, patient_name, segmentationfolder, "lungs_left-")
    f1_R = compare_images(ans_R, ans.affine, contender_lung_R, patient_name, segmentationfolder, "lungs_right-")

    print("Dice Left Lung Iteration: {}".format(f1_L))
    print("Dice Right Lung Iteration: {}".format(f1_R))
    if dataset != "LCTSC":
        print("Dice Airways Iteration: {}".format(f1_A))
        # Write all accuracies in two notepads (not dilated and dilated)
        notepad_dil.write("DL{} DR{} DA{}\r".format(f1_L, f1_R, f1_A))
    else:
        notepad_dil.write("DL{} DR{}\r".format(f1_L, f1_R))

    return

def remove_files(patient_name, roiftfolder, wslfolder, outputfolder, segmentationfolder):
    print("Deleting files on this computer...")
    file = "wsl ls " + roiftfolder
    # Remove files from ROIFT folder
    p = subprocess.check_output(file, shell=True).decode("utf-8")
    lista_items = p.replace("\n", " ").split()
    for k in range(len(lista_items)):
        # Remove seeds file (.txt)
        if patient_name in lista_items[k] and '.txt' in lista_items[k]:
            remove_command = "wsl rm " + roiftfolder + lista_items[k]
            p = subprocess.run(remove_command, shell=True)
        
        # Remove nifti files (.nii)
        if patient_name in lista_items[k] and '.nii' in lista_items[k]:
            remove_command = "wsl rm " + roiftfolder + lista_items[k]
            p = subprocess.run(remove_command, shell=True)

        # Remove png files (.png)
        if patient_name in lista_items[k] and '.png' in lista_items[k]:
            remove_command = "wsl rm " + roiftfolder + lista_items[k]
            p = subprocess.run(remove_command, shell=True)

        # if patient_name in lista_items[k] and '.txt' in lista_items[k]:
        #     p = subprocess.run("wsl mkdir -p " + wslfolder + "Debug_Analysis/" + patient_name, shell=True)
        #     seed = "wsl cp -ar " + wslfolder + lista_items[k] + " " + wslfolder.replace("Seeds", "Debug_Analysis") + patient_name + "/"
        #     p = subprocess.run(seed, shell=True)
            
    # Find all .png files in the specified folder
    lista_items = glob.glob(os.path.join(outputfolder, "*"))
    
    for k in range(len(lista_items)):
        # Remove png files (.png)
        if patient_name in lista_items[k] and '.txt' in lista_items[k]:
            os.remove(lista_items[k])

    for k in range(len(lista_items)):
        # Remove png files (.png)
        if patient_name in lista_items[k] and '.png' in lista_items[k]:
            os.remove(lista_items[k])

    for k in range(len(lista_items)):
        # Remove png files (.png)
        if patient_name in lista_items[k] and '.nii' in lista_items[k]:
            os.remove(lista_items[k])

    # try:
    #     os.remove('label.nii.gz')
    # except:
    #     pass

    return

def main():
    start = time.process_time()
    dataset = int(input("1 - LOLA, 2 - LCTSC, 3 - fator 4, 4 - VIA ELCAP, 5 = EXACT: "))
    # Pasta que contem pasta de oiftrelax, imagens originais e gabarito
    inputfolder = "C:/Users/" + os.path.join(os.getlogin()) + "/OneDrive/Documentos/Poli/IC/"
    # roift path
    linuxfolder = "~/Documents/IC/"
    roiftfolder = linuxfolder + "MSales21_3D/"

    datasets = ['LOLA', 'LCTSC', 'fator_4', 'VIAELCAP', 'EXACT']
    dataset = datasets[dataset-1]

    outputfolder = inputfolder + "Seeds/" + dataset + "/"
    wslfolder = "/mnt/c/Users/" + outputfolder.replace("C:/Users/", "")
    mkdir_p(outputfolder)
    mkdir_p(wslfolder)
    originalfolder = inputfolder + "Dataset/" + dataset + "/"
    mcubesfolder = inputfolder + "STL/" + dataset + "/"
    answerfolder = inputfolder + "Gabarito/" + dataset + "/"
    segmentationfolder = inputfolder + "Segmented/" + dataset + "/"
    if dataset == "LCTSC": arq = "lista" + dataset + ".txt" 
    else: arq = "lista" + dataset + ".txt"
    txt_names = ["results_" + dataset]
    mode = '2gb_1dc'

    arquive = search_dir_dcm(inputfolder + "DatasetLists/" + arq)

    # no intervalo, escolher entre 1 a 24
    a = int(input("1 - Specific patients, 2 - Interval: "))

    if a == 1:
        while True:
            ex = str(input("Just: "))
            exc = ex.split(",")
            exc = str_to_list(exc)
            print("Just these patients?")
            for o in exc:
                print(arquive[o-1].replace('\n','').replace(inputfolder.replace('/','\\'),'').replace('\\','-'))
            p = int(input("1 - Yes, 2 - No: "))
            if p == 1:
                n = exc[0]
                m = exc[-1]
                aux = list(range(n,m))
                exc = np.setdiff1d(aux,exc)
                n -=1
                break
        print()
    elif a == 2:
        n = int(input("Start: "))
        m = int(input("Stop: "))
        n -= 1
        while True:
            ex = str(input("Except: "))
            if ex == "":
                exc = []
                break
            exc = ex.split(",")
            exc = str_to_list(exc)
            print("Except these patients?")
            for o in exc:
                print(arquive[o-1].replace('\n','').replace(inputfolder.replace('/','\\'),'').replace('\\','-'))
            p = int(input("1 - Yes, 2 - No: "))
            if p == 1:
                break
        print()

    relaxs = [0, 10]
    # Relax parameter
    for j in relaxs:
        for txt_name in txt_names:
            notepad_dil = open(inputfolder + "Results/" + txt_name + "it{}.txt".format(j), "w")
            counter = 1
            for i in range(n,m):
                if (i+1) not in exc:
                    input_name = arquive[i].replace('\n','')
                    patient_name = input_name.replace('NIFTI_SEEDS\\','')
                    nifti_file = originalfolder + patient_name + ".nii"
                    patient_name = dataset + "-" + patient_name
                    print("Patient {}/{}: ".format(counter,m-n) + patient_name)

                    try:
                        SEED_GEN(nifti_file, outputfolder, patient_name)
                        send_to_linux(nifti_file, patient_name, outputfolder, wslfolder, roiftfolder)
                        pol = 0.1
                        percs = [90]
                        for perc in percs:
                            ROIFT(roiftfolder, mcubesfolder, segmentationfolder, patient_name, perc, str(j), mode, pol)
                            # if dataset == "LOLA" or dataset == "VIAELCAP" or dataset == "EXACT" or dataset == "LCTSC":
                            #     patient_name = patient_name.replace(dataset + "-", '')
                            #     VAL(answerfolder, segmentationfolder, notepad_dil, patient_name, dataset)
                        remove_files(patient_name, roiftfolder, wslfolder, outputfolder, segmentationfolder)

                    except ZeroDivisionError as e:
                        # Print the exception error
                        print("An exception occurred:", e)
                    counter += 1


        notepad_dil.close()

    # mean_accs(inputfolder + "Results/v3/", "results_EXACTit0.txt")
    # mean_accs(inputfolder + "Results/v3/", "results_viaelcapit0.txt")
    # mean_accs(inputfolder + "Results/v3/", "results_lolait0.txt")
    # mean_accs(inputfolder + "Results/v3/", "results_LCTSCit0.txt")

    print("Done.")
    print("Time: {}".format(time.process_time() - start))

main()
