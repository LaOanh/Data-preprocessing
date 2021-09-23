'''
created on 27 April 2021
author: Oanh Thi La
purposes: processing data for CNN model
1. convert Digital number to Top of Atmosphere reflectance
2. calculating Geometry angles (cosine of Relative azimuth angle, Sun zenith angle, and sensor zenith angle)
3. recaling (0.0001) for AOT (downloaded from Landsat 8 level 2 by USGS)
3. merging 8 TOA bands + 3 Geometry angles + 1 AOT into 1 TIF image
'''



import numpy as np
from osgeo import gdal
import rasterio as rio
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

Root = tkinter.Tk()  # Create a Tkinter.Tk() instance
Root.withdraw()  # Hide the Tkinter.Tk() instance
# read MTL file to get rescaling parameters and sun_elevation angle
metadata = askopenfilename(title=u'Open MTL file', filetypes=[("MTL", ".txt")])

# read 8 band (B1,....B7, B9) for convert to TOA reflectance
bandlist = filedialog.askopenfilenames(title='Choose band 1 to 7 and band 9 files', filetypes=[("TIF", ".tif")])

# Read sun azimuth angle (SAA) and sensor azimuth angle (VAA)
SAA_path = askopenfilename(title=u'Open SAA file', filetypes=[("TIF", ".tif")]) #choose your SAA angle
VAA_path = askopenfilename(title=u'Open VAA file', filetypes=[("TIF", ".tif")]) # choose your VAA angle

# Read Sun zenith angle (SZA) and Sensor zenith angle (VZA)
SZA_path = askopenfilename(title=u'Open SZA file', filetypes=[("TIF", ".tif")]) #choose your SZA angle
VZA_path = askopenfilename(title=u'Open VZA file', filetypes=[("TIF", ".tif")]) # choose your VZA angle

#Read AOT path
AOT_path = askopenfilename(title=u'Open sr_aerosol file', filetypes=[("TIF", ".tif")])

def get_MTLfile(metadata):
    fh = open(metadata)
    # Get rescaling parameters and sun_elevation angle
    mult_term = []
    add_term = []
    sun_elevation = float()
    for line in fh:
        # Read the file line-by-line looking for the reflectance transformation parameters
        if "REFLECTANCE_MULT_BAND_" in line:
            mult_term.append(float(line.split("=")[1].strip()))
        elif "REFLECTANCE_ADD_BAND_" in line:
            add_term.append(float(line.split("=")[1].strip()))
        elif "SUN_ELEVATION" in line:
            # We're also getting the sun elevation from the metadata. It has

            sun_elevation = float(line.split("=")[1].strip())
    fh.close()  # Be sure to close an open file

    return mult_term, add_term, sun_elevation

[mult_term, add_term, sun_elevation] = get_MTLfile(metadata)


def DN_toTOAreflectance(mult_term, add_term, sun_elevation, bandlist):

    with rio.open(bandlist[0]) as src1:
        image_band1 = src1.read(1)

    image_masked_band1 = np.ma.masked_array(image_band1, mask=(image_band1 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa1 = (mult_term[0] * image_masked_band1.astype(float) + add_term[0])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band1 = (toa1.astype(float) / solar_z)

    with rio.open(bandlist[1]) as src2:
        image_band2 = src2.read(1)
    image_masked_band2 = np.ma.masked_array(image_band2, mask=(image_band2 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa2 = (mult_term[1] * image_masked_band2.astype(float) + add_term[1])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band2 = (toa2.astype(float) / solar_z)

    with rio.open(bandlist[2]) as src3:
        image_band3 = src3.read(1)
    image_masked_band3 = np.ma.masked_array(image_band3, mask=(image_band3 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa3 = (mult_term[2] * image_masked_band3.astype(float) + add_term[2])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band3 = (toa3.astype(float) / solar_z)

    with rio.open(bandlist[3]) as src4:
        image_band4 = src4.read(1)
    image_masked_band4 = np.ma.masked_array(image_band4, mask=(image_band4 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa4 = (mult_term[3] * image_masked_band4.astype(float) + add_term[3])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band4 = (toa4.astype(float) / solar_z)

    with rio.open(bandlist[4]) as src5:
        image_band5 = src5.read(1)
    image_masked_band5 = np.ma.masked_array(image_band5, mask=(image_band5 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa5= (mult_term[4] * image_masked_band5.astype(float) + add_term[4])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band5 = (toa5.astype(float) / solar_z)

    with rio.open(bandlist[5]) as src6:
        image_band6 = src6.read(1)
    image_masked_band6 = np.ma.masked_array(image_band6, mask=(image_band6 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa6= (mult_term[5] * image_masked_band6.astype(float) + add_term[5])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band6 = (toa6.astype(float) / solar_z)

    with rio.open(bandlist[6]) as src7:
        image_band7 = src7.read(1)
    image_masked_band7 = np.ma.masked_array(image_band7, mask=(image_band7 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa7= (mult_term[6] * image_masked_band7.astype(float) + add_term[6])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band7 = (toa7.astype(float) / solar_z)

    with rio.open(bandlist[7]) as src8:
        image_band8 = src8.read(1)
    image_masked_band8 = np.ma.masked_array(image_band8, mask=(image_band8 == 0))  # exclude 0 value
    constant = 0.01745329251994444444444444444444  # Constant is calculated (3.14/180) which is converting the sun-angle to sun_radians which was suggested by WOlfgang
    toa8= (mult_term[7] * image_masked_band8.astype(float) + add_term[7])
    solar_z = np.cos((90 - float(sun_elevation)) * float(constant))
    toa_band8 = (toa8.astype(float) / solar_z)

    return src1, toa_band1, toa_band2, toa_band3, toa_band4, toa_band5, toa_band6, toa_band7, toa_band8

[src1, toa_band1, toa_band2, toa_band3, toa_band4, toa_band5, toa_band6, toa_band7, toa_band8] = \
    DN_toTOAreflectance(mult_term, add_term, sun_elevation, bandlist)


## CALCULATE RELATIVE AZIMUTH ANGLE FROM SUN AZIMUTH AND SENSOR AZIMUTH ANGLE
def relative_azimuth_angle(SAA_path, VAA_path):
    ## read raster image as array
    SAA = gdal.Open(SAA_path).ReadAsArray()
    VAA = gdal.Open(VAA_path).ReadAsArray()
    sun_azi_angle = SAA / 100
    sen_azi_angle = VAA / 100
    difference_value = abs(sun_azi_angle - sen_azi_angle) #abs(Sensor Azimuth - 180.0 - Solar azimuth)
    dif_row = difference_value.shape[0]
    dif_col = difference_value.shape[1]
    rel_azi_angle = np.zeros([dif_row, dif_col])
    for i in range(dif_row):
        for j in range(dif_col):
            if difference_value[i, j] > 180.0:
                rel_azi_angle[i, j] = 360.0 - difference_value[i, j]
            elif difference_value[i, j] == 0:
                rel_azi_angle[i, j] = 0.0
            else:
                rel_azi_angle[i, j] = 180.0 - difference_value[i, j]
    return rel_azi_angle, difference_value

[rel_azi_angle, difference_value] = relative_azimuth_angle(SAA_path, VAA_path)

#calculating cosine of RAA angle
rel_azi_angle[rel_azi_angle==0] = np.nan # convert 0 value to nan to avoid calculate cosine for pixel have 0 value
cos_RAA = np.cos(rel_azi_angle)
cos_RAA[np.isnan(cos_RAA)] = 0 # convert nan back to 0 value
cos_RAA = np.float32(cos_RAA) # convert float64 to float32


'''2. COSINE OF SZA and VZA'''
def cos_SZA_VZA(SZA_path, VZA_path):
    ## read raster image as array
    SZA = gdal.Open(SZA_path).ReadAsArray()
    VZA = gdal.Open(VZA_path).ReadAsArray()
    sun_ze_angle = SZA / 100
    sen_ze_angle = VZA / 100

    sun_ze_angle[sun_ze_angle == 0] = np.nan  # convert 0 value to nan to avoid calculate cosine for pixel have 0 value
    cos_SZA = np.cos(sun_ze_angle)
    cos_SZA[np.isnan(cos_SZA)] = 0  # convert nan back to 0 value
    cos_SZA_final = np.float32(cos_SZA)  # convert float64 to float32

    sen_ze_angle[sen_ze_angle == 0] = np.nan  # convert 0 value to nan to avoid calculate cosine for pixel have 0 value
    cos_VZA = np.cos(sen_ze_angle)
    cos_VZA[np.isnan(cos_VZA)] = 0  # convert nan back to 0 value
    cos_VZA_final = np.float32(cos_VZA)  # convert float64 to float32
    return cos_SZA_final, cos_VZA_final

[cos_SZA_final, cos_VZA_final] = cos_SZA_VZA(SZA_path, VZA_path)


## 3. Recaling AOT data for 10000
# read AOT file
AOT_array = gdal.Open(AOT_path).ReadAsArray()
AOT_array[AOT_array==1] = 0 # convert nan back to 0 value
AOT = AOT_array*0.0001 ## recaling for 10000
AOT = np.float32(AOT) # convert float64 to float32


'MERGING BANDS INTO 1 TIF FILE'

toab1 = np.array(toa_band1, dtype='float32')
toab1[toab1 == 2.e-05] = 0
toab2 = np.array(toa_band2, dtype='float32')
toab2[toab2 == 2.e-05] = 0
toab3 = np.array(toa_band3, dtype='float32')
toab3[toab3 == 2.e-05] = 0
toab4 = np.array(toa_band4, dtype='float32')
toab4[toab4 == 2.e-05] = 0
toab5 = np.array(toa_band5, dtype='float32')
toab5[toab5 == 2.e-05] = 0
toab6 = np.array(toa_band6, dtype='float32')
toab6[toab6 == 2.e-05] = 0
toab7 = np.array(toa_band7, dtype='float32')
toab7[toab7 == 2.e-05] = 0
toab8 = np.array(toa_band8, dtype='float32')
toab8[toab8 == 2.e-05] = 0
filelist = np.concatenate((toab1, toab2, toab3, toab4, toab5, toab6, toab7, toab8, cos_VZA_final, cos_SZA_final, cos_RAA, AOT), axis=0)
filelist_reshape = filelist.reshape(12, toab1.shape[0], toab1.shape[1]).astype('float32')

## SAVING to TIF image
with rio.open(SAA_path) as src: # choose one image
    ras_data = src.read()
    ras_meta = src.meta
ras_meta.update(count=len(filelist_reshape))
ras_meta['dtype'] = "float32"
ras_meta['No Data'] = 0.0

Fname1 = tkinter.filedialog.asksaveasfilename(title=u'Select output filename', filetypes=[("TIF", ".tif")])
with rio.open(Fname1, 'w', **ras_meta) as dst: # write image with the same shape with the selected image aboved
    dst.write(filelist_reshape)








# import matplotlib.pyplot as plt
# imgplot = plt.imshow(cos_VZA_final)
# plt.show()







