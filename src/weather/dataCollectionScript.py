'''
This file uses the code from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7 
for the below two functions:
    earth_radius()
    area_grid()
These functions aggregate the weather data over a specified bounding box.
'''


import subprocess
import sys
from collections import namedtuple
from calendar import monthrange
import os.path
import pandas as pd
import csv
import math
import numpy as np

# Regions named on the command line override this list. That is how one region
# gets processed on its own for a test, and how a full run gets sharded across
# several pods without editing code.
ISO_LIST = sys.argv[1:] or ["AECI", "AT", "AZPS", "BE", "BG", "BPAT", "CH", "CISO", "CZ", "DE",
            "DK", "DUK", "EE", "EPE", "ERCOT", "ES", "FI", "FPL", "FR", "GR",
            "HR", "HU", "IPCO", "ISNE", "IT", "LDWP", "LT", "LV", "MISO", "NEVP",
            "NL", "NWMT", "NYISO", "PACE", "PACW", "PJM", "PL", "PSCO", "PT", "RS",
            "SC", "SCEG", "SE", "SI", "SK", "SOCO", "SPA", "SRP", "SWPP", "TIDC",
            "TVA"]
# IF 2 CONSECUTIVE FILES ARE NOT PRESENT, MANUALLY ADD THEM

# Plain name, not a relative build path: the CarbonCast README installs the
# binary with `cp -rfv wgrib2/wgrib2 /usr/local/bin/wgrib2`, and the
# weather_download image puts it on PATH via conda-forge.
#
# -inv /dev/null discards wgrib2's inventory listing, which it otherwise writes
# to stdout on every single call. That is ~2 lines x 206,590 calls per region --
# enough to bury the "doesn't exist" warnings that matter and to blow through a
# Kubernetes node's log rotation. The flag must precede the filename, and the
# CSV it produces is byte-identical either way.
GRIB2_CMD = "wgrib2 -inv /dev/null"
# FILE_DIR = "gfs.0p25.2020010100-2021010100.f000.grib2/"

FILE_PREFIX = "gfs.0p25."
# Where <REGION>/<var>/*.grib2 live. The default is the weather-data PVC, which
# the weather_download repo's k8s/tar-job.yaml and k8s/debug-pod.yaml mount
# whole at /data; the download Job instead mounts just the retraining_archive
# subPath at /app/retraining_archive.
#
# Overridable because reading the GRIB2 files straight off CephFS is ruinous:
# each is only ~4.3 KB, every one is read exactly once so nothing caches, and
# the measured cost was 1.58 s per file against 5.3 ms on local disk -- 300x,
# and 206,590 files per region. weather_download's scripts/aggregate_weather.sh
# therefore unpacks the tars onto node-local scratch and points this at that
# instead, turning 206,590 small CephFS reads into 32 large sequential ones.
ARCHIVE_ROOT = os.environ.get("WEATHER_ARCHIVE_ROOT", "/data/retraining_archive")
UGRD_VGRD_SEPRATOR = {"CISO": 1886, "PJM": 2556, "SE": 2296, "DK-DK2": 221, 
                        "ERCO": 2116, "ISNE": 1056, "GB": 1978, "DE": 1254, 
                        "PL": 984, "AUS_NSW": 64, "AUS_QLD": 5695, "AUS_SA": 2809}
TMP_DPT_SEPARATOR = {"CISO": 1886, "PJM": 2556, "SE": 2296, "DK-DK2": 221, 
                        "ERCO": 2116, "ISNE": 1056, "GB": 1978, "DE": 1254, 
                        "PL": 984, "AUS_NSW": 64, "AUS_QLD": 5695, "AUS_SA": 2809}

HOUR = ["00"] ##, "06", "12", "18"]
MAX_FCST_HOUR = 168 # was 96
FCST = ["000"] + ["%03d" % h for h in range(3, MAX_FCST_HOUR+1, 3)]
FCST_AVG_ACC = ["%03d" % h for h in range(3, MAX_FCST_HOUR+1, 3)]
YEARS = [2022, 2023]
HEADER = ["startDate", "endDate", "param", "level", "longitude", "latitude", "value"]
CSV_FILE_FIELDS_FCST = ["datetime", "param", "level", "latitude", "longitude",
        "Analysis"] + ["%d hr fcst" % h for h in range(3, MAX_FCST_HOUR+1, 3)]
# ds084.1 returns averages/accumulations as a (3-hour, 6-hour) pair per 6-hour
# block -- 0-3, 0-6, 6-9, 6-12, ... -- and that is the order they are stored in.
CSV_FILE_FIELDS_AVG = ["datetime", "param", "level", "latitude", "longitude"] + [
        lbl for s in range(0, MAX_FCST_HOUR, 6)
        for lbl in ("%d-%d hr avg" % (s, s+3), "%d-%d hr avg" % (s, s+6))]
CSV_FILE_FIELDS_ACC = ["datetime", "param", "level", "latitude", "longitude"] + [
        lbl for s in range(0, MAX_FCST_HOUR, 6)
        for lbl in ("%d-%d hr acc" % (s, s+3), "%d-%d hr acc" % (s, s+6))]

def getFileList(yearList=[2020], fileDir = None, fcstCol=FCST):
    fileList = []
    prevFile = None
    for year in yearList:
        for month in range(1, 13): # Month is always 1..12
            for day in range(1, monthrange(year, month)[1] + 1):
                curDate = str(year)+f"{month:02d}"+f"{day:02d}"
                fileName = FILE_PREFIX + str(curDate)
                oldFileName = fileName
                for hr in HOUR:
                    for fcst in fcstCol:
                        fileName = oldFileName
                        fileName +=str(hr) + ".f"+str(fcst)+".grib2"
                        filePath = ""
                        filePath = fileDir + fileName
                        if (os.path.exists(filePath) == False):
                            print(filePath + " doesn't exist")
                            filePath = fileDir + str(prevFile)
                            if (os.path.exists(filePath) == True):
                                fileList.append(filePath)
                                print("Using previous forecast value with file: ", prevFile)
                            else:
                                print(filePath + " doesn't exist also")
                        else:
                            fileList.append(filePath)
                            prevFile = fileName # assuming the first file to be searched is always present
    return fileList

# A cycle may borrow at most this many forecast hours from earlier in the SAME
# cycle before it is dropped instead. Six steps is 18 hours of a 168-hour
# horizon; past that the row is mostly synthetic, and a visible gap beats a
# plausible fabrication.
MAX_SUBSTITUTED_PER_CYCLE = 6


def buildDateGroups(fileDir, fcstCol, yearList=YEARS):
    """Expected GRIB2 paths grouped by init cycle, in forecast-hour order.

    Returns [(YYYYMMDDHH, [path, ...]), ...]. A cycle missing a forecast hour
    reuses the nearest earlier hour FROM THE SAME CYCLE, so a gap costs a few
    hours of staleness rather than a missing day.

    Two things this does that getFileList() above does not.

    First, paths are built from fcstCol directly, so position i can only ever
    hold forecast hour fcstCol[i]. getFileList returns one flat list that the
    caller slices len(fcstCol) at a time, and when its fallback cannot find a
    substitute it appends nothing at all -- leaving the list short, so every
    later cycle silently pairs the wrong forecast hour with the wrong day for
    the rest of the run.

    Second, substitution never crosses a cycle boundary. getFileList carries
    its `prevFile` across days, so a missing f000 is filled with YESTERDAY's
    last forecast hour -- a day-old value in the analysis slot, which is the
    present-time field. Here a cycle with nothing earlier to borrow from is
    dropped instead, as is one needing more than MAX_SUBSTITUTED_PER_CYCLE
    substitutions.
    """
    groups = []
    filled = []
    dropped = []

    for year in yearList:
        for month in range(1, 13):
            for day in range(1, monthrange(year, month)[1] + 1):
                curDate = str(year) + f"{month:02d}" + f"{day:02d}"
                for hr in HOUR:
                    label = curDate + hr
                    paths, subs, lastGood, reason = [], [], None, None

                    for fcst in fcstCol:
                        path = os.path.join(fileDir, "%s%s%s.f%s.grib2"
                                            % (FILE_PREFIX, curDate, hr, fcst))
                        if os.path.exists(path):
                            lastGood = path
                            paths.append(path)
                        elif lastGood is None:
                            reason = ("f%s missing with nothing earlier in the "
                                      "cycle to substitute" % fcst)
                            break
                        else:
                            subs.append((fcst, os.path.basename(lastGood).split(".f")[1][:3]))
                            paths.append(lastGood)

                    if reason is None and len(subs) > MAX_SUBSTITUTED_PER_CYCLE:
                        reason = ("%d of %d files missing, over the substitution "
                                  "cap of %d" % (len(subs), len(fcstCol),
                                                 MAX_SUBSTITUTED_PER_CYCLE))
                    if reason is not None:
                        dropped.append((label, reason))
                        continue
                    if subs:
                        filled.append((label, subs))
                    groups.append((label, paths))

    if filled:
        print("    %d cycle(s) completed by substitution (%d file(s)):"
              % (len(filled), sum(len(sub) for _, sub in filled)))
        for label, sub in filled[:5]:
            print("      %s  %s" % (label, ", ".join("f%s<-f%s" % (a, b) for a, b in sub)))
        if len(filled) > 5:
            print("      ... and %d more" % (len(filled) - 5))
    if dropped:
        print("    %d cycle(s) DROPPED:" % len(dropped))
        for label, reason in dropped[:5]:
            print("      %s  %s" % (label, reason))
        if len(dropped) > 5:
            print("      ... and %d more" % (len(dropped) - 5))
    return groups


def _gridOf(dataset):
    """Unique latitudes and longitudes, and the cell-area weights over them."""
    lat = np.unique(dataset["latitude"].values)
    lon = np.unique(dataset["longitude"].values)
    cellArea = area_grid(lat, lon)
    return lat, lon, cellArea, np.sum(cellArea)


def reportMissingCycles(name, writtenLabels, yearList=YEARS):
    """Print how many init cycles made it into a CSV, and list every one that did not.

    A cycle dropped by buildDateGroups or skipped while reading is an ABSENT
    row, not a blank one, so anything reading the CSV must not assume one row
    per day. Consecutive missing cycles are printed as one range.
    """
    expected = [str(year) + f"{month:02d}" + f"{day:02d}" + hr
                for year in yearList
                for month in range(1, 13)
                for day in range(1, monthrange(year, month)[1] + 1)
                for hr in HOUR]

    runs, start, prev = [], None, None
    for label in expected:
        if label in writtenLabels:
            if start is not None:
                runs.append((start, prev))
                start = None
        else:
            start = start or label
            prev = label
    if start is not None:
        runs.append((start, prev))

    numMissing = len(expected) - len(writtenLabels)
    print("    %s: wrote %d of %d cycles, %d missing"
          % (name, len(writtenLabels), len(expected), numMissing))
    if runs:
        print("      missing: " + ", ".join(a if a == b else "%s-%s" % (a, b) for a, b in runs))


def getWeatherData(dateGroups, csvFields, outFileName, weatherVariable):
    with open(outFileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvFields)
    rows, written = [], set()
    pairedVar = weatherVariable in ("TEMP", "DPT")

    for label, dateFiles in dateGroups:
        row = None
        lat = lon = cellArea = totalArea = gridSize = None
        failed = None

        for i in range(len(dateFiles)):
            val = subprocess.call(GRIB2_CMD + " " + dateFiles[i] + " -csv tmp.csv", shell=True)
            if val != 0:
                failed = "wgrib2 exited %d on %s" % (val, dateFiles[i])
                break
            dataset = pd.read_csv("tmp.csv", names=HEADER)
            os.remove("tmp.csv")

            if i == 0:
                lat, lon, cellArea, totalArea = _gridOf(dataset)
                gridSize = len(lat) * len(lon)

            # TMP/DPT and U/V arrive as two params over one grid, so a paired
            # file holds exactly twice the grid. Anything else means the file
            # is truncated or was subset differently, and its cycle is dropped
            # rather than contributing a short row.
            expected = gridSize * 2 if pairedVar else gridSize
            if len(dataset) != expected:
                failed = "%s: expected %d rows, got %d" % (dateFiles[i], expected, len(dataset))
                break

            if weatherVariable == "TEMP":
                dataset = dataset[:gridSize]
            elif weatherVariable == "DPT":
                dataset = dataset[gridSize:2 * gridSize]

            if row is None:
                row = [dataset["startDate"].iloc[0], dataset["param"].iloc[0],
                       dataset["level"].iloc[0], dataset["latitude"].iloc[0],
                       dataset["longitude"].iloc[0]]

            value = np.reshape(dataset["value"].values, (len(lat), len(lon)))
            row.append(np.sum((value * cellArea) / totalArea))

        if failed is not None or row is None:
            print("    SKIP %s %s -- %s" % (weatherVariable, label, failed or "no data"))
        else:
            rows.append(row)
            written.add(label)

    with open(outFileName, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
    reportMissingCycles(weatherVariable, written)
    return


def getWindData(dateGroups, csvFields, outFileName):
    with open(outFileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvFields)
    rows, written = [], set()

    for label, dateFiles in dateGroups:
        row = None
        lat = lon = cellArea = totalArea = gridSize = None
        failed = None

        for i in range(len(dateFiles)):
            val = subprocess.call(GRIB2_CMD + " " + dateFiles[i] + " -csv tmp.csv", shell=True)
            if val != 0:
                failed = "wgrib2 exited %d on %s" % (val, dateFiles[i])
                break
            dataset = pd.read_csv("tmp.csv", names=HEADER)
            os.remove("tmp.csv")

            if i == 0:
                lat, lon, cellArea, totalArea = _gridOf(dataset)
                gridSize = len(lat) * len(lon)

            if len(dataset) != gridSize * 2:
                failed = "%s: expected %d rows, got %d" % (
                    dateFiles[i], gridSize * 2, len(dataset))
                break

            # U GRD first, V GRD second -- confirmed against the GRIB2 section 4
            # parameter numbers and against wgrib2 -csv output on the archive.
            udataset = dataset[:gridSize]
            vdataset = dataset[gridSize:2 * gridSize]

            if row is None:
                row = [dataset["startDate"].iloc[0], dataset["param"].iloc[0],
                       dataset["level"].iloc[0], dataset["latitude"].iloc[0],
                       dataset["longitude"].iloc[0]]

            windSpeed = (udataset["value"].values**2 + vdataset["value"].values**2)**(0.5)
            value = np.reshape(windSpeed, (len(lat), len(lon)))
            row.append(np.sum((value * cellArea) / totalArea))

        if failed is not None or row is None:
            print("    SKIP WIND %s -- %s" % (label, failed or "no data"))
        else:
            rows.append(row)
            written.add(label)

    with open(outFileName, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
    reportMissingCycles("WIND", written)
    return


def addColumnToCSVFile(fileName, newColName, newColValue):
    dataset = pd.read_csv(fileName, header=0)
    dataset[newColName] = newColValue
    dataset.to_csv(fileName)
    return

def getWindSpeedAcrossAllRegions(inFileName):
    maxWindSpeedList = []
    totalWindSpeedList = []
    avgWindSpeedList = []
    uWindFileList = None
    if (inFileName is None):
        uWindFileList = getFileList(2020, "SE/ugrd_vgrd/")
    else:
        uWindFileList = getFileList(2020, inFileName)
    # vWindFileList = getFileList(2020, "ds084_1/vGRD/")
    fileIdx = 0
    rows = []
    while fileIdx < len(uWindFileList):
        row = None
        # weatherValues = []
        for i in range(len(FCST)):
            uval = subprocess.call(GRIB2_CMD + " " + uWindFileList[fileIdx] + " -csv utmp.csv", shell=True)
            if(uval == 0):
                vval = 0 #subprocess.call(GRIB2_CMD + " " + vWindFileList[fileIdx] + " -csv vtmp.csv", shell=True)
                fileIdx+=1
                if (vval == 0):
                    udataset = pd.read_csv("utmp.csv", names=HEADER)
                    # vdataset = pd.read_csv("vtmp.csv", infer_datetime_format=True, 
                    #         names=HEADER)
                    vdataset = udataset[UGRD_VGRD_SEPRATOR[ISO]:]
                    udataset = udataset[:UGRD_VGRD_SEPRATOR[ISO]]
                    # if row is None:
                    #     row = [dataset["startDate"].iloc[cityIdx], dataset["param"].iloc[cityIdx], dataset["level"].iloc[cityIdx],
                    #                 dataset["latitude"].iloc[cityIdx], dataset["longitude"].iloc[cityIdx]]

                    maxWindSpeed = 0
                    totalWindSpeed = 0
                    avgWindSpeed = 0
                    for j in range(len(udataset)):
                        u = udataset["value"].iloc[j]
                        v = vdataset["value"].iloc[j]
                        windSpeed = math.sqrt(u*u + v*v)
                        totalWindSpeed +=windSpeed
                        if  windSpeed> maxWindSpeed:
                            maxWindSpeed = windSpeed
                    avgWindSpeed = totalWindSpeed/len(udataset)
                    maxWindSpeedList.append([udataset["startDate"].iloc[0], maxWindSpeed])
                    totalWindSpeedList.append([udataset["startDate"].iloc[0], totalWindSpeed])
                    avgWindSpeedList.append([udataset["startDate"].iloc[0], avgWindSpeed])
                    # delFile = subprocess.call("rm utmp.csv", shell=True)
                    # if(delFile != 0):
                    #     print("Error: Process call failed -- rm")
                else:
                    print("Error: Process call failed -- ", GRIB2_CMD)
            else:
                print("Error: Process call failed -- ", GRIB2_CMD)
        # writing to csv file
        # print(weatherValues)
        # row.extend(weatherValues) 
        # rows.append(row)
    return maxWindSpeedList, totalWindSpeedList, avgWindSpeedList

def getTotalDWSRFcrossAllRegions():
    totalWindSpeedList = []
    avgWindSpeedList = []
    dswrfFileList = getFileList(2020, "PJM/dswrf/")
    # vWindFileList = getFileList(2020, "ds084_1/vGRD/")
    fileIdx = 0
    rows = []
    while fileIdx < len(dswrfFileList):
        row = None
        # weatherValues = []
        for i in range(len(FCST)):
            uval = subprocess.call(GRIB2_CMD + " " + dswrfFileList[fileIdx] + " -csv utmp.csv", shell=True)
            if(uval == 0):
                fileIdx+=1
                udataset = pd.read_csv("utmp.csv", names=HEADER)
                totalWindSpeed = np.sum(udataset["value"].values)
                avgWindSpeed = np.average(udataset["value"].values)
                totalWindSpeedList.append([udataset["startDate"].iloc[0], totalWindSpeed])
                avgWindSpeedList.append([udataset["startDate"].iloc[0], avgWindSpeed])
            else:
                print("Error: Process call failed -- ", GRIB2_CMD)
        # writing to csv file
        # print(weatherValues)
        # row.extend(weatherValues) 
        # rows.append(row)
    return totalWindSpeedList, avgWindSpeedList

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    # print("Earth radius:", r ,len(r))
    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos

    xlon, ylat = meshgrid(lon, lat)
    # print(ylat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx
    # print("Area shape: ", area.shape, type(area))
    return area

# # maxWindSpeedList, totalWindSpeedList, avgWindSpeedList = getWindSpeedAcrossAllRegions()
# totalDwsrfList, avgDwsrfList = getTotalDWSRFcrossAllRegions()
# # pd.DataFrame(maxWindSpeedList).to_csv("SE_maxWS.csv")
# pd.DataFrame(totalDwsrfList).to_csv("PJM_totalDWSRF.csv")
# pd.DataFrame(avgDwsrfList).to_csv("PJM_avgDWSRF.csv")



for ISO in ISO_LIST:
    # FILE_DIR = ["../final_weather_data/"+ISO+"/ugrd_vgrd/", #/2019_weather_data
    #         "../final_weather_data/"+ISO+"/tmp_dpt/",
    #         "../final_weather_data/"+ISO+"/tmp_dpt/",
    #         "../final_weather_data/"+ISO+"/dswrf/",
    #         "../final_weather_data/"+ISO+"/apcp/"]

    # OUT_FILE_NAME_LIST = ["../final_weather_data/"+ISO+"/"+ISO+"_AVG_WIND_SPEED.csv", #/2019_weather_data
    #                     "../final_weather_data/"+ISO+"/"+ISO+"_AVG_TEMP.csv",
    #                     "../final_weather_data/"+ISO+"/"+ISO+"_AVG_DPT.csv",
    #                     "../final_weather_data/"+ISO+"/"+ISO+"_AVG_DSWRF.csv",
    #                     "../final_weather_data/"+ISO+"/"+ISO+"_AVG_PCP.csv"]

    FILE_DIR = [ARCHIVE_ROOT+"/"+ISO+"/wind/",
            ARCHIVE_ROOT+"/"+ISO+"/temp/",
            ARCHIVE_ROOT+"/"+ISO+"/temp/",
            ARCHIVE_ROOT+"/"+ISO+"/dswrf/",
            ARCHIVE_ROOT+"/"+ISO+"/rain/"]

    OUT_FILE_NAME_LIST = [ARCHIVE_ROOT+"/"+ISO+"/"+ISO+"_AVG_WIND_SPEED.csv",
                        ARCHIVE_ROOT+"/"+ISO+"/"+ISO+"_AVG_TEMP.csv",
                        ARCHIVE_ROOT+"/"+ISO+"/"+ISO+"_AVG_DPT.csv",
                        ARCHIVE_ROOT+"/"+ISO+"/"+ISO+"_AVG_DSWRF.csv",
                        ARCHIVE_ROOT+"/"+ISO+"/"+ISO+"_AVG_PCP.csv"]

    # FILE_DIR = ["../extn/"+ISO+"/weather_data/tmp_dpt/",
    #         "../extn/"+ISO+"/weather_data/tmp_dpt/"]
            
    # OUT_FILE_NAME_LIST = ["../extn/"+ISO+"/weather_data/"+ISO+"_AVG_TEMP.csv",
    #                     "../extn/"+ISO+"/weather_data/"+ISO+"_AVG_DPT.csv"]

    print("*******************", ISO, "*******************")
    for xx in range(len(OUT_FILE_NAME_LIST)):
        if ("WIND" in OUT_FILE_NAME_LIST[xx]):
            print("WIND: ", OUT_FILE_NAME_LIST[xx])
            groups = buildDateGroups(FILE_DIR[xx], FCST)
            getWindData(groups, CSV_FILE_FIELDS_FCST, OUT_FILE_NAME_LIST[xx])
        else:
            if ("TEMP" in OUT_FILE_NAME_LIST[xx]):
                print("TEMP: ", OUT_FILE_NAME_LIST[xx])
                groups = buildDateGroups(FILE_DIR[xx], FCST)
                getWeatherData(groups, CSV_FILE_FIELDS_FCST, OUT_FILE_NAME_LIST[xx], "TEMP")
            elif ("DPT" in OUT_FILE_NAME_LIST[xx]):
                print("DPT: ", OUT_FILE_NAME_LIST[xx])
                groups = buildDateGroups(FILE_DIR[xx], FCST)
                getWeatherData(groups, CSV_FILE_FIELDS_FCST, OUT_FILE_NAME_LIST[xx], "DPT")
            elif ("DSWRF" in OUT_FILE_NAME_LIST[xx]):
                print("DSWRF: ", OUT_FILE_NAME_LIST[xx])
                groups = buildDateGroups(FILE_DIR[xx], FCST_AVG_ACC)
                getWeatherData(groups, CSV_FILE_FIELDS_AVG, OUT_FILE_NAME_LIST[xx], "DSWRF")
            elif ("PCP" in OUT_FILE_NAME_LIST[xx]):
                print("PCP: ", OUT_FILE_NAME_LIST[xx])
                groups = buildDateGroups(FILE_DIR[xx], FCST_AVG_ACC)
                getWeatherData(groups, CSV_FILE_FIELDS_ACC, OUT_FILE_NAME_LIST[xx], "PCP")
    print("*******************", ISO, " done *******************")

