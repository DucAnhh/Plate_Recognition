# DetectChars.py

import cv2
import numpy as np
import math
import random
from sklearn import preprocessing
import tensorflow as tf

import Main
import Preprocess
import PossibleChar


# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 30

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 3.2

MAX_CHANGE_IN_AREA = 0.4

MAX_CHANGE_IN_WIDTH = 0.6
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 15.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 2

RESIZED_CHAR = (28, 28)
RESIZED_CHAR_IMAGE_HEIGHT = 28

MIN_CONTOUR_AREA = 100

MARGIN = 5

# CNN_MODEL = "models/cnn.ckpt"


def showContours(possiblePlate, listOfPossibleCharsInPlate):
    height, width, numChannels = possiblePlate.imgPlate.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    contours = []
    
    for possibleChar in listOfPossibleCharsInPlate:
        contours.append(possibleChar.contour)
    # end for

    cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

    cv2.imshow("6.getContours", imgContours)
    
def showListOfLists(possiblePlate, listOfCharsInPlate):
    print("step 7 - listOfListsOfChras = " + str(len(listOfCharsInPlate)))    # 13 with MCLRNF1 image
    height, width, numChannels = possiblePlate.imgPlate.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    intRandomBlue = random.randint(0, 255)
    intRandomGreen = random.randint(0, 255)
    intRandomRed = random.randint(0, 255)

    contours = []

    for matchingChar in listOfCharsInPlate:
        contours.append(matchingChar.contour)
    # end for

    cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
    # end for

    cv2.imshow("7.combineListOfLists", imgContours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###################################################################################################
def detectCharsInPlates(imgOriginalScene, listOfPossiblePlates, filePath):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # if list of possible plates is empty
        return listOfPossiblePlates             # return
    # end if

            # at this point we can be sure the list of possible plates has at least one plate

    for index, possiblePlate in enumerate(listOfPossiblePlates):          # for each possible plate, this is a big for loop that takes up most of the function
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # preprocess to get grayscale and threshold images
        
        adaptivePlate = cv2.adaptiveThreshold(possiblePlate.imgGrayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        blurPlate = cv2.GaussianBlur(adaptivePlate, (5,5),0)
        ret, processedPlate = cv2.threshold(blurPlate, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if Main.showSteps == True: # show steps ###################################################
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c.adaptive", adaptivePlate)
            cv2.imshow("5d.blur", blurPlate)
            cv2.imshow("5e.otsu", processedPlate)
            
        # end if # show steps #####################################################################

                # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # find all possible chars in the plate,
                # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(adaptivePlate)
        listOfPossibleCharsInPlate.sort(key = lambda Char: Char.intCenterX)
        
        if Main.showSteps == True: # show steps ###################################################
            showContours(possiblePlate, listOfPossibleCharsInPlate)
            
        listOfListsOfChars= findListOfListsOfMatchingChars(listOfPossibleCharsInPlate, minChars=3, maxAngle=10)
        if len(listOfListsOfChars) == 0:
            continue
        # find chars that have same heights
        listOfListsOfChars1 = [getEqualHeightList(x) for x in listOfListsOfChars]
        listOfListsOfChars2 = getEqualHeightList(listOfListsOfChars1, mode=1)
        # remove Distance Char
        listOfListsOfChars3 = [removeDistanceChar(x) for x in listOfListsOfChars2]
        # flatten list
        listOfCharsInPlate = [char for listChars in listOfListsOfChars3 for char in listChars]
        # remove inner Chars
        listOfCharsInPlate = removeInnerChars(listOfCharsInPlate)
        
        # number of plate elements must be > 6
        if len(listOfCharsInPlate) >= 6:
            possiblePlate.isPlate = True
            if Main.showSteps == True: # show steps #######################################################
                showListOfLists(possiblePlate, listOfCharsInPlate)

            # end of big for loop that takes up most of the function
            possiblePlate.strChars = recognizeCharsInPlate(imgOriginalScene, possiblePlate.imgGrayscale, listOfCharsInPlate)
            print("predict: ", possiblePlate.strChars)
        else:
            continue
    
    listOfPlates = [plate for plate in listOfPossiblePlates if plate.isPlate]
    
    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPlates
# end function

###################################################################################################
def findPossibleCharsInPlate(imgThresh):
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)       # add to list of possible chars
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
            # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

### eliminate different height
def getBounding(data):
    m = np.median(data)
    std = np.std(data)
    ratio = std / m
    # if ratio is small, use large size from standard deviation
    n = [1, 3][ratio < 0.05]

    return m + n * std, m - n * std

# mode = 1 for list of List, mode = 0 for list
def getEqualHeightList(listChars, mode=0):
    if mode:
        listHeight = [x[0].intBoundingRectHeight for x in listChars]
    else:
        listHeight = [x.intBoundingRectHeight for x in listChars]
    
    upperBound, lowerBound = getBounding(listHeight)
    if mode:
        listChars = [x for x in listChars if x[0].intBoundingRectHeight <= upperBound and x[0].intBoundingRectHeight >= lowerBound]
    else:
        listChars = [x for x in listChars if x.intBoundingRectHeight <= upperBound and x.intBoundingRectHeight >= lowerBound]
    return listChars

### remove inner chars
def removeInnerChars(listOfCharsInPlate):
    for i, char1 in enumerate(listOfCharsInPlate):
        for j, char2 in enumerate(listOfCharsInPlate):
            charStart_1X, charStart_1Y = char1.intBoundingRectX, char1.intBoundingRectY
            charEnd_1X, charEnd_1Y = char1.intBoundingRectX + char1.intBoundingRectWidth,                                         char1.intBoundingRectY + char1.intBoundingRectHeight
                
            charStart_2X, charStart_2Y = char2.intBoundingRectX, char2.intBoundingRectY
            charEnd_2X, charEnd_2Y = char2.intBoundingRectX + char2.intBoundingRectWidth,                                         char2.intBoundingRectY + char2.intBoundingRectHeight
            if charStart_1X < charStart_2X and charStart_1Y < charStart_2Y and                charEnd_1X > charEnd_2X and charEnd_1Y > charEnd_2Y:
                char2.isChar = False
    
    listOfChars = [char for char in listOfCharsInPlate if char.isChar == True]
    return listOfChars

###################################################################################################
def removeDistanceChar(charsInPlate):
    for i, char1 in enumerate(charsInPlate):
        validChar = False
        
        for j, char2 in enumerate(charsInPlate):
            relativeDistance = math.sqrt( (char1.intCenterX - char2.intCenterX)**2 + (char1.intCenterY - char2.intCenterY)**2 ) / char1.fltDiagonalSize
            if relativeDistance > 0 and relativeDistance < 1:
                validChar = True
        
        if not validChar:
            char1.isChar = False
            
    listOfChars = [char for char in charsInPlate if char.isChar == True]
    return listOfChars

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars, minChars = MIN_NUMBER_OF_MATCHING_CHARS, maxAngle = MAX_ANGLE_BETWEEN_CHARS):
            # with this function, we start off with all the possible chars in one big list
            # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
            # note that chars that are not found to be in a group of matches do not need to be considered further
    listOfListsOfMatchingChars = []                  # this will be the return value

    for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars, maxAngle)        # find all chars in the big list that match the current char

        listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars

        if len(listOfMatchingChars) < minChars:     # if current possible list of matching chars is not long enough to constitute a possible plate
            continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary
                                                # to save the list in any way since it did not have enough chars to be a possible plate
        # end if

                                                # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved, minChars)      # recursive call

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# end function
###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars, maxAngle = MAX_ANGLE_BETWEEN_CHARS):
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []                # this will be the return value

    for possibleMatchingChar in listOfChars:                # for each char in big list
        if possibleMatchingChar == possibleChar:    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                                                    # then we should not include it in the list of matches b/c that would end up double including the current char
            continue                                # so do not add to list of matches and jump back to top of for loop
        # end if
                    # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < maxAngle and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # if the chars are a match, add the current char to list of matching chars
        # end if
    # end for

    return listOfMatchingChars                  # return result
# end function

###################################################################################################
# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

    return fltAngleInDeg
# end function

def charPlace(char):
    return char.intCenterX + 10 * char.intCenterY


## Recognize letter ##
num_classes = 36 # 10 digits + 26 characters
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
class CNN_Model(tfe.Network):
  def __init__(self):
    super(CNN_Model, self).__init__()
    self.conv1 = self.track_layer(tf.layers.Conv2D(32, 5, activation = tf.nn.relu, padding = "SAME"))
    self.pool1 = self.track_layer(tf.layers.MaxPooling2D(2, 2))
    self.conv2 = self.track_layer(tf.layers.Conv2D(64, 5, activation = tf.nn.relu, padding = "SAME"))
    self.pool2 = self.track_layer(tf.layers.MaxPooling2D(2, 2))
    self.flatten = self.track_layer(tf.layers.Flatten())
    self.fc1 = self.track_layer(tf.layers.Dense(256, activation = tf.nn.relu))
    self.dropout = self.track_layer(tf.layers.Dropout(0.75))
    self.fc2 = self.track_layer(tf.layers.Dense(num_classes, activation = None))
    
  def call(self, input):
    input = tf.reshape(input, [-1, 28, 28, 1])
    result = self.conv1(input)
    result = self.pool1(result)
    result = self.conv2(result)
    result = self.flatten(result)
    result = self.fc1(result)
    result = self.dropout(result)
    result = self.fc2(result)
    return result

def recognizeLetter(test_imgs):

    test_imgs = np.asarray(test_imgs, dtype=np.float32)

    # use Label Encoding to decode later
    le = preprocessing.LabelEncoder()
    string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labels = list(string)
    le.fit(labels)
    x = tf.random_normal((1, 784))
    model = CNN_Model()
    model(x) # intitiate model - model.variables
    tfe.restore_network_checkpoint(model,'./models/AZ_09.ckpt')
    
    logits = model(test_imgs)
    test_labels = tf.argmax(tf.nn.softmax(logits), axis = 1)
    test_labels = le.inverse_transform(test_labels)
    
    return test_labels

    
###################################################################################################
# this is where we apply the actual char recognition
def recognizeCharsInPlate(imgOriginalScene, imgThresh, listOfMatchingChars):
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key = charPlace)        # sort chars from left to right
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # make color version of threshold image so we can draw contours in color on it
    
    charImages = []
    for i, currentChar in enumerate(listOfMatchingChars):                                         # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # draw green box around the char

                # crop char out of threshold image
        imgROI = imgThresh[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        imgROIResized = cv2.resize(imgROI, RESIZED_CHAR)           # resize image, this is necessary for char recognition
        
        # retreive binary image from the char images
        adaptivePlate = cv2.adaptiveThreshold(imgROIResized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        blurPlate = cv2.GaussianBlur(adaptivePlate, (5,5),0)
        ret, im = cv2.threshold(blurPlate,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # cv2.destroyAllWindows()
        if Main.showSteps == True:
            cv2.imshow("resize_" + str(i), im)

            cv2.waitKey(0)
            # cv2.imshow("img_thresh", imgThresh)
        
        charImages.append(im)
        
    # end for

    if Main.showSteps == True: # show steps #######################################################
        cv2.imshow("8", imgThreshColor)
    # end if # show steps #########################################################################
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    chars = recognizeLetter(charImages)
    strChars = "".join(chars)
   
    return strChars
# end function
