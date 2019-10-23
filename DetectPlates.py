# coding: utf-8

# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random
import sys
import os

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.7
PLATE_HEIGHT_PADDING_FACTOR = 1.5

MAX_OVERLAP_RATIO, MIN_OVERLAP_RATIO = 1, 0.8
MAX_RATIO, MIN_RATIO = 1.5, 0.5
MAX_ANGLE_DIFF, MIN_ANGLE_DIFF = 30, -30
SHAPE_OF_POSSIBLE_PLATE = (120, 96)

listOfPossiblePlates =[]

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.8
PLATE_HEIGHT_PADDING_FACTOR = 1.5

MAX_OVERLAP_RATIO, MIN_OVERLAP_RATIO = 1, 0.75
MAX_RATIO, MIN_RATIO = 1.3, 0.7
MAX_ANGLE_DIFF, MIN_ANGLE_DIFF = 20, -20

listOfPossiblePlates =[]

###################################################################################################
def detectPlatesInScene(imgOriginalScene, location):
    listOfRawPossiblePlates = []                   # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps == True: # show steps #######################################################
        cv2.imshow("0", imgOriginalScene)
    # end if # show steps #########################################################################

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)         # preprocess to get grayscale and threshold images

    if Main.showSteps == True: # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if # show steps #########################################################################

            # find all possible chars in the scene,
            # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)
    listOfPossibleCharsInScene.sort(key = lambda Char: Char.intCenterX)

    if Main.showSteps == True: # show steps #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))         # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        # picture 2a - list of all contours
        # list of possible chars
        cv2.imshow("2b", imgContours)
        cv2.waitKey(0)
    # end if # show steps #########################################################################

            # given a list of all possible chars, find groups of matching chars
            # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True: # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(len(listOfListsOfMatchingCharsInScene)))    # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)

    # end if # show steps #########################################################################

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each group of matching chars
        possiblePlate = extractPlate(listOfMatchingChars)         # attempt to extract plate
        listOfRawPossiblePlates.append(possiblePlate)                  # add to list of possible plates
        # end if
    # end for

    listOfPossiblePlates = groupPossiblePlates(imgOriginalScene, listOfRawPossiblePlates)
    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")          # 13 with MCLRNF1 image

    if Main.showSteps == True: # show steps #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a"+str(i), imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b"+str(i), listOfPossiblePlates[i].imgPlate)
        # end for

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # end if # show steps #########################################################################



    return listOfPossiblePlates
# end function


import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

def groupPossiblePlates(imgOrigion, listOfPossiblePlates):
    duplicateSet = set([])
    for i, value in enumerate(listOfPossiblePlates):
        for j in range(i):
            if j == i:
                continue
            else:
                (center1, (width1, height1), angle1) = listOfPossiblePlates[i].rrLocationOfPlateInScene
                (center2, (width2, height2), angle2) = listOfPossiblePlates[j].rrLocationOfPlateInScene
                relativeDistance = math.sqrt( (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2 ) / height1
                heightRatio = height2 / height1
                angleDifference = angle2 - angle1
                
                # check if 2 lists has the same direction
                if angleDifference < MAX_ANGLE_DIFF and angleDifference > MIN_ANGLE_DIFF and relativeDistance < MAX_RATIO \
                   and relativeDistance > MIN_RATIO and heightRatio < MAX_RATIO and heightRatio > MIN_RATIO:
                    r1 = RotatedRect(center1[0], center1[1], width1, height1, angle1)
                    r2 = RotatedRect(center2[0], center2[1], width2, height2, angle2)
                    
                    Area1 = r1.get_contour().area
                    Area2 = r2.get_contour().area
                    intersectArea = r1.intersection(r2).area
                    relativeArea1 = intersectArea / Area1
                    relativeArea2 = intersectArea / Area2
                    # check if 2 list is the same location (overlapped area)
                    if (relativeArea1 < MAX_OVERLAP_RATIO and relativeArea1 > MIN_OVERLAP_RATIO) and (relativeArea2 < MAX_RATIO and relativeArea2 > MIN_RATIO):
                        if Area1 < Area2:
                            duplicateSet.add(i)
                            break
                        else:
                            duplicateSet.add(j)
                            continue
                    
                    # merge 2 lists in the same plate
                    if intersectArea > 0:
                        groupCenter = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)
                        groupWidth = width1 if width1 > width2 else width2
                        groupHeight = height1 + height2
                        groupAngle = (angle1 + angle2)/2
                        
                        listOfPossiblePlates[i].isTwoRow = listOfPossiblePlates[j].isTwoRow = True
                        listOfPossiblePlates[i].rrLocationOfPlateInScene = (groupCenter, (groupWidth, groupHeight), groupAngle)
                        duplicateSet.add(j)
                        break
    
    finalList = []
    for index, plate in enumerate(listOfPossiblePlates):
        if index not in list(duplicateSet) and plate.isTwoRow:
            plateWithImage = appendImageOfList(imgOrigion, plate)
            print(plateWithImage.rrLocationOfPlateInScene)
            center, (width, height), _ = plateWithImage.rrLocationOfPlateInScene
            shapeRatio = width / height
            if plateWithImage.imgPlate is not None:       # if plate was found and shape is square
                finalList.append(plateWithImage)                  # add to list of possible plates
        
    return finalList

###################################################################################################
def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour

        if Main.showSteps == True: # show steps ###################################################
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
        # end if # show steps #####################################################################

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # increment count of possible chars
            listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars
        # end if
    # end for

    if Main.showSteps == True: # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))                       # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))       # 131 with MCLRNF1 image
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################

    return listOfPossibleChars
# end function

###################################################################################################
def extractPlate(listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # this will be the return value

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

            # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    return possiblePlate
# end function

def appendImageOfList(imgOriginal ,possiblePlate):
    ptPlateCenter, (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg = possiblePlate.rrLocationOfPlateInScene
            # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), ptPlateCenter)

    possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate
    
    return possiblePlate