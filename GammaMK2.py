# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:44:52 2020

@author: Hayes
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import UsefulFunctions as uFunc
import os as os
import copy as copy
from sympy import symbols, diff
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D

#this is to allow for flipping between the console and separate windows without having to reload
inConsole = True





class TestRun:
    #contains all of the data for a single test run: one gamma source through one blocker

    
    def __init__(this, name, location):
                
        #initalize dummy values, then fill each field via a methods
        this.blockerMaterial = ""
        this.arealDensity = ""
        this.time = 0 #TODO
        
        #energy and counts are linked together: each index of the two of them refer to the same measurement
        this.counts = []
        
        
        def determineBlockerMaterial(this):
        #this function uses the name of the test run (drawn from the text file) to determine
        #the material of the blocker, and the thickness of the blocker
            
        #
            if((name.lower()).endswith("al")):
                this.blockerMaterial = "Aluminum"
        
            elif( (name.lower()).endswith("pb") or (name.lower()).endswith("lead")):
                this.blockerMaterial = "Lead"
        
            else:
                print("file was mislabeled")
        
        def determineArealDensity(this):
            
            stringHolder = uFunc.getNums(name)
            densityHolder = []
            mappingHolder = []
            
            if this.blockerMaterial == "Aluminum":
                densityHolder = GammaSource.alDensities
                mappingHolder = GammaSource.alMapping
            
            elif this.blockerMaterial == "Lead":
                densityHolder = GammaSource.pbDensities
                mappingHolder = GammaSource.pbMapping
                
            else:
                print("test run contains non standard blockerMaterial: ", this.blockerMaterial)
            
            
            assignedFlag = False 
            #this is a boolean flag to make sure each test run is properly assigned
            
            for x in range(0, len(densityHolder)):
                #looks at each slot in mapping array
                for label in mappingHolder[x]:
                #looks at each possible mapping for a single density value
                
                    if stringHolder == label:
                        this.arealDensity = densityHolder[x]
                        assignedFlag = True
            
            if (not assignedFlag):
                print("testRun was not assigned areal density")
            
        #end determineBlockerThickness
        
        def loadData(this):
            #this function loads in the majority of the data: the counts and energy of each reading
            
            arraysHolder = uFunc.arrayFromFile(location, 2, 2)
            this.counts =  np.array(arraysHolder[1])
            this.time = int(uFunc.lineFromFile(location, 0))
         #end loadData   

        #actual initialization
        determineBlockerMaterial(this)
        determineArealDensity(this)
        loadData(this)

    #end __init__()
       
    def getAllCounts(this):
        #helper that retuns the sum of all the counts of a test run
        return sum(this.counts)
    
    
#end TestRun class
   
    
    
class GammaSource:
    #this class contains all of the data for a single source of gamma emmisions, it contains
    #arrays of test run objects, each of which contains the data for this source, for a particualr
    #test run
        
    #these arrays are the areal densities of each blocker source, ordered from least to greatest
    #pbDensities = [0.044, 0.088, 100, 0.132, 0.199, 0.243, ]
    #alDensities = [ 10, 0.187, 350, 0.75, 1.50, 3.0]


    alDensities = [10, 350, 1282.4460000000001, 5143.5, 10287.0, 20574.0] #in mg/cm^2
    
    pbDensities = [100, 1267.3584, 2534.7168, 3802.0752, 5731.9164, 6999.2748] #in mg/cm^2


    #due to poorly labeled files, these arrays are used to assign each
    #test run it's appropriate position, each sub array contains file labels that corespond
    #to the density in pb/alDensities arrays. 
    alMapping = [["10"], ["350"], ["0187", "0137"], ["075"], ["150", "15"], ["3", "300"]]
    pbMapping = [["100"], ["0044"], ["0088", "088"], ["0132"], ["0199", "099"], ["0243"]]
    #"099" for lead feels worrying
    
    #TODO:
    #aight, so abunch of the above values are actaully the thickness in inches,
    #meaning we need to convert a thickness in inches to an areal density
    #in mg/cm^2

    
    #the energy values are the same for all test runs, so we store them here statically.
    energyValues = uFunc.arrayFromFile("C:/Users/Hayes/Documents/MATLAB/GammaTextFiles/Cesium0044inPB.txt", 2, 2)
    energyValues = np.array(energyValues[0]) #nned to do this, because inital read returns energy values and counts
    #These initial energy values are in keV, not MeV
            
    energyValues = energyValues/(1000)

    
    
    def __init__(this, name):
        
        this.name = name #the name of the source
        this.lead = []  #an array of all the test runs through lead
        this.aluminum = [] #an array of all the test runs through aluminum
        this.peaks = [] #the energy of each peak we are looking for
        
        for x in range(0,6):
        #this is done to set the size of the lists, from this point on nothing should be 
        #appended to these lists, onlt values in slots should be edited
            this.lead.append(0)
            this.aluminum.append(0)
        
        
    #end __init__
    
    #TODO: figure out how to use static methods
    #@staticmethod
    def calculateEnergyValues(this, location):
    
        dummy = 0
        #the energy values are the same for all test runs, so we store them here statically.
        #this is inefficiant, instead should do this once durring 

        
    #end calculateEnergyValues
    
    
    class Peak:
        
        def __init__(this, left, right):
            
            #these are the approximate energy values for the boundaries of the peak
            
            #this.leftEnergy = left
            #this.rightEnergy = right
            #TODO: instead of having this, should we change left and right to be 
            #the actual values of the indices, me thinks yess
            
            
            #rather than the highest point, this is the center as baised on the bounds
            #this.center = ((right-left)/2)+left
            
            
            #these are the positions in the energy array that most closely corespond to
            #the values in leftEnergy and rightEnergy
            
            this.leftEnergyPosition = uFunc.closestIndex(GammaSource.energyValues, left)
            this.rightEnergyPosition = uFunc.closestIndex(GammaSource.energyValues, right)
            
            this.leftEnergy = GammaSource.energyValues[this.leftEnergyPosition]
            this.rightEnergy = GammaSource.energyValues[this.rightEnergyPosition]
            
            
            def calculateCenter(this):
            #TODO: consider instead of this, defining the center of the peak as the average
            #position of the highest point accross all test runs
            
                #helper method that calculates the center of the peak, then returns it
                centerHolder = ((this.rightEnergy - this.leftEnergy)/2) + this.leftEnergy
                return GammaSource.energyValues[uFunc.closestIndex(GammaSource.energyValues, centerHolder)]
            
            this.center = calculateCenter(this)
            
        #defines the area around a peak, must include enough points
        #around peak to fit for background
        
        def fitBackground(this, testRun):
            #this method should fit out the background between the left and right
            #sides of the peak. The book says i want to try multiple lefts and rights
            #per peak, so ill wanna throw that in later
        
            #what to return: this could return a function that is then used
            #to determine how much to suubtract from each index, or it could
            #return a list where each indice corresponds to the background at that point
            
            #lets do the latter
            
            #to establish a more legitmate fit , howabout we take several points on either
            #side of the bounds, and do an actual linear fit to figure out the background
            
            xValsToFit = []
            yValsToFit = []
            
            
            for x in range(0 , 10):
                #adds points to either side of the bounds to the list used for fitting
                xValsToFit.append(GammaSource.energyValues[this.leftEnergyPosition + x])
                #xValsToFit.append(GammaSource.energyValues[this.leftEnergyPosition - x])
                
                yValsToFit.append(testRun.counts[this.leftEnergyPosition + x])
                #yValsToFit.append(testRun.counts[this.leftEnergyPosition - x])
                
                #xValsToFit.append(GammaSource.energyValues[this.rightEnergyPosition + x])
                xValsToFit.append(GammaSource.energyValues[this.rightEnergyPosition - x])
                
                #yValsToFit.append(testRun.counts[this.rightEnergyPosition + x])
                yValsToFit.append(testRun.counts[this.rightEnergyPosition - x])
                #TODO: figure out; do i really want to fit using values outside of the peak?
                #maybe i only want to go right of the left bound, and vice verse
            
            
            uFunc.sortLists(xValsToFit, yValsToFit)
            
            
            #lets just use the same method for fitting, and write an equation for a striaght line
            
            def linearFitFunction(xVals, m, b):
                
                return m*xVals + b
            
            #now we perform a fit, then create the appropriate list of values to
            #subtract
            
            xValsToFit = np.array(xValsToFit)
            yValsToFit = np.array(yValsToFit)
    
            
            fitLine, backgroundSigma = optimize.curve_fit(linearFitFunction, xValsToFit, yValsToFit)
            
            #the fit values for m and b
            mFit = fitLine[0]
            bFit = fitLine[1]
            
            
            #now we use a function we wrote for this to find the uncertainty in each of these
            #values due to the uncertainty in the fit parameters
            sigma = np.sqrt(np.diag(backgroundSigma))
            
            #there is probably a more efficient way to do this, but I am more focued on
            #getting it working in the first place
            m,b = symbols('m b', real = True)
            
            
            #to handle this with my current code i will have to carry this out 
            #separatley for each value of x, but i think that's not cosher
            #according to what i remember when you add two uncertainties
            #like this the propagation is just the root mean square
            #so there shouldnt be anything wrong with this
            
            
            
            formatedData = []
            formatedData.append([])
            formatedData.append([])
        
            formatedData[0].append(mFit)
            formatedData[0].append(sigma[0])
            
            formatedData[1].append(bFit)
            formatedData[1].append(sigma[1])
            
            errorSummation = 0
            
            for x in xValsToFit:
                #at the moment this appears to 
                
                
                sigmaFunc = m*x + b
                errorSummation = errorSummation + (uFunc.propagateError(sigmaFunc, formatedData, uncertaintiesKnown = True))**2
            
            
            errorSummation = errorSummation**.5
            
            #to find the values of the background that we need to subtract out
            #we then input our fit values for m and b along with the xvalues that 
            #define the peak to find the fit value of the background at each point in the peak
            
            peakXVals = np.array(this.getPeakXVals())
            backgroundVals = linearFitFunction(peakXVals, mFit, bFit)
        
            
            
            #print("sum of backgroundVals: ", sum(backgroundVals))
            #print("error summation: ", errorSummation)
            
            #returns the background to be removed, along with the uncertainty due to 
            #this subtraction
            return (backgroundVals, errorSummation)
        
        def integratePeak(this, testRun, rate = True):
            #this will subtract out the background,then total all the counts in the peak, and
            #return this value, along with the associated uncertainty
            #the rate parameter determines if you return counts or rate, you will almost always
            #want the rate, not the counts
            
            backgroundData = this.fitBackground(testRun)
            
            background = backgroundData[0]
            backgroundUncertainty = backgroundData[1]
            
            #plt.yscale('log')
            #plt.plot(np.array(this.getPeakXVals()), background/testRun.time, 'bs')
            
            
            
            counts = this.filterData(testRun)
            
            countsHolder = sum(counts)
            #to account for the detectors uncertainty, we use poisson statistics

            countUncertainty = np.sqrt(countsHolder)
            
            #TODO: Figure out how to account for the background uncertainty
            backgroundHolder = sum(background)
            
            countsHolder = countsHolder - backgroundHolder
            
            if rate:
                countsHolder = countsHolder/testRun.time
                countUncertainty = countUncertainty/testRun.time

            return (countsHolder, countUncertainty)
            
        
        def filterData(this, testRun):
            #filters out count data for energy levels outside of the bounds defined by the peak
            #then returns this new data
            countsHolder = copy.deepcopy(testRun.counts) #this step may not be needed
            countsHolder = countsHolder[this.leftEnergyPosition : this.rightEnergyPosition]
            
            return countsHolder
            
        
        
        def getPeakXVals(this):
            #this method returns a list of all the energy values that the peak ranges
            return GammaSource.energyValues[this.leftEnergyPosition : this.rightEnergyPosition]
        
        
    #end peak class  
    
    def addPeak(this, left, right):
        this.peaks.append( this.Peak(left, right))
        #this.peaks.append(this.Peak(left+0.0001, right+0.0001))
        #this.peaks.append(this.Peak(left-0.0001, right-0.0001))        
    
    def plotPeak(this, blockerMaterial, peak = 0):
        #by default plots the first peak for the gamma source, howwever the peak parameter can change this
        #plots one peak, or all peaks of a blocker material for a gamma source
        

        holder = this.getPeakData(blockerMaterial, peak)
        xVals = holder[0]
        yVals = holder[1]
        yUncer = holder[2]
        
        
        titleString = this.name + " through " + blockerMaterial 
        #uFunc.newPlotConsole(yLog = True, title = titleString, yLabel = "rate", xLabel = "areal density of blocker ")
        uFunc.newPlotConsole( title = titleString, yLabel = "rate", xLabel = "areal density of blocker ")

        
        #plt.plot(xVals, yVals, 'bs')
        plt.errorbar(xVals, yVals, yerr = yUncer, fmt = 'none')
        plt.plot(xVals, yVals, 'r--')    
        
    def getPeakData(this, blockerMaterial, peak = 0):
        #this method returns three sublists , where the first contains
        #the thickness of each blocker, the second contains the integrated height of 
        #the peak for the corresponding blocker, and the third contains the uncertainty
        #for the height of each peak
        
        dataSet = []
        xVals = []
        yVals = []
        yUncer = []
        toPlot = this.peaks[peak]
        
        if(blockerMaterial.lower() == "aluminum"):
            dataSet = this.aluminum
        
        elif(blockerMaterial.lower() == "lead"):
            dataSet = this.lead
            
        else:
            print("improper blocker material, please enter the name of a blocker as a string, you entered: " , blockerMaterial)
       
        
        for testRun in dataSet:
            
            yData = toPlot.integratePeak(testRun)
            
            yVals.append(yData[0])
            yUncer.append(yData[1])
            xVals.append(testRun.arealDensity)
           
        bucket = []
        bucket.append(xVals)
        bucket.append(yVals)
        bucket.append(yUncer)
        return bucket    
    
#end GammaSource class
 


def loadData(location):
    #this is the highest level of abstraction function: it is called once to load all the data
    #stored in a directory into classes that this program can work with
    

    gammaSources = []
    #this list contains all of the GammaSource objects
    def createGammaSources():
        #this function is used primarily to organize things: it creates
        #each GammaSource object, and adds it to the proper array
        #also pulls the appropriate values for the energy levels of each reading
        #once 
        
        gammaSources.append(GammaSource("barium"))
        gammaSources.append(GammaSource("cesium"))
        gammaSources.append(GammaSource("cobalt57"))
        gammaSources.append(GammaSource("cobalt60"))
        gammaSources.append(GammaSource("manganese"))
        gammaSources.append(GammaSource("sodium"))
        
    #end createGammaSources
    

    def generateTestRuns(testRunName, testRunLocation):
        #takes in the name and location of each new testRun,
        #determines the associated gamma source, strips this from the name string,
        #then creates new test run object
    
        for source in gammaSources:
            
            if testRunName.lower().startswith(source.name):
                
                testRunName = testRunName[len(source.name):]
                #this should strip the first part of the file name
                testRunHolder = TestRun(testRunName, testRunLocation)
                

                
                if testRunHolder.blockerMaterial == "Aluminum":
                    source.aluminum[GammaSource.alDensities.index(testRunHolder.arealDensity)] = testRunHolder
                
                
                elif testRunHolder.blockerMaterial == "Lead":
                    
                    source.lead[GammaSource.pbDensities.index(testRunHolder.arealDensity)] = testRunHolder
                
                
                else :
                    print("error cascade, improperly labeled file was dropped to continue process, file was named: ", testRunLocation)
    
    def readDataFromText(directory):
        #reads data in from directory, performs percursory checks on file to make sure
        #its properly formatted, then calls new method to create and sort testRun objects
        
        for file in os.listdir(directory):
            #we are only interested in text files
            if file.endswith(".txt"):
            
                generateTestRuns(file[:-4], (location + "/" + file))
            
    #end readDataFromText
    
    #first we create the list of gamma sources
    createGammaSources()
    
    #next we read in the data from text files, creting objects, and placing them
    #where they need to go
    readDataFromText(location)
    
    return gammaSources
    




#end loadData


def makePeakBox(gammaSource, peak, blockerMaterial = "aluminum"):

    #returns a list that contains two 
   
    package = []
    
    yVals = []
    xVals = []
    
    
    #the first trial of the aluminum test runs should have the largest energy,
    #this then finds the largest energy within the reigion defined by the peak
    maxY = max( (gammaSource.aluminum[0].counts)[peak.leftEnergyPosition : peak.rightEnergyPosition])
    maxY = maxY/gammaSource.aluminum[0].time
    
    #maxY = uFunc.recursiveMax(testRunArrayHolder)
    #this wont work because i dont know which time to divide by
    
    #maxY = max(toPlot.counts/toPlot.time)
        
    xVals.append(peak.leftEnergy)        
    yVals.append(0)
    
    xVals.append(peak.leftEnergy)
    yVals.append(maxY)
    
    xVals.append(peak.rightEnergy)
    yVals.append(maxY)
        
    xVals.append(peak.rightEnergy)
    yVals.append(0)
        
    yVals.append(0)
    xVals.append(peak.leftEnergy)

    package.append(xVals)
    package.append(yVals)
    
    return package



def drawPeakBox(gammaSource, peak, blockerMaterial, drawCenter = True):
    
    peakBox = makePeakBox(gammaSource, peak, blockerMaterial)
    
    xVals = peakBox[0]
    yVals = peakBox[1]
    
    plt.plot(xVals, yVals, 'r--')
    
    if drawCenter:
        #this draws a blue square at the center of the peak
        plt.plot(peak.center, max(yVals) ,'bs')


def drawSpectra(gammaSource, blockerMaterial = -1 ,testRun = -1, oneFigure = True, peakBounds = True, log = True):
    #method used to draw the spectra of test runs, depending on paramaters
    #draws the spectra of one test run or all test runs of a particular gamma source
    #one figure determines if this all goes down on a single figure, or if each
    #spectra has its own figure

    #TODO:
    #detect if testRun is a float or an int, if its an int you do that index in the array,
    #if its a float try to do the test run where the areal density is equal to that float
    #dont know if this is needed anymore
    
    
    
    if blockerMaterial == -1:
        #this means we plot both spectra
        drawSpectra(gammaSource, "aluminum",testRun, oneFigure, peakBounds, log)
        drawSpectra(gammaSource, "lead", testRun, oneFigure, peakBounds, log)
        return
    
    #this is used to switch between aluminum and lead blocker arrays
    testRunArrayHolder = []
    materialString = ""
    
    if blockerMaterial.lower() == "aluminum":
        testRunArrayHolder = gammaSource.aluminum
        materialString = "aluminum"
    
    elif blockerMaterial.lower() == "lead":
        testRunArrayHolder = gammaSource.lead
        materialString = "lead"
        
    else:
        print("tried to make a spectra for a non-existant blocker")
        return
        
    
    #used for printing one or all of the spectra
    counter = 0
    endValue = len(GammaSource.pbDensities) - 1
    
    #if the user designates a specific testRun
    if not testRun == -1 :
        
        counter = testRun
        endValue = testRun+1
        
    
    axisSize = 32
    labelSize = 28
    tickSize = 22
    thickness = .5
    
    titleString = gammaSource.name + " in " + materialString
    
       
        
    
    for x in range(counter, endValue):
    
        if( (not oneFigure) or (x == counter)):
            #regardless of if theres one figure or multiple, if its the first itteration
            #we must generate a new figure
            plt.figure()
            
            if(log):
                plt.yscale('log')
                
            plt.title(titleString)
        #end if one Figure branch
        
        
        toPlot = testRunArrayHolder[x]
        
        plt.plot( GammaSource.energyValues, toPlot.counts/toPlot.time, linewidth = thickness)
        
        
        drawBackground = True
        if(drawBackground):
            #this draws a line that shows the fit for the background of each test run
            #backgroundVals = GammaSource.pea
            dummy = 0    #end if background branch
        
        
    if(peakBounds):
        #draws lines to show where the reigons of interest are
        for peak in gammaSource.peaks:
            drawPeakBox(gammaSource, peak, blockerMaterial)
        
            
        #for x in range(0, len(peaksToPlot)/2):
            #plt.plot(peaksToPlot[x*2], peaksToPlot[(x*2)+1], 'r--')
        

        
        
        
#end draw Spectra
    
def plotAllSpectra(sources, material = -1):
    
    #by default draws all spectra for all blockers, with lead and aluminum on separate graphs,
    #but if user chooses, can instead only do a specific blocker material

    if ((material == -1) or (material.lower() == "aluminum")):
        
        for gammaSource in sources:
            drawSpectra(gammaSource, "aluminum")
    
    if ((material == -1) or (material.lower() == "lead")):
        
        for gammaSource in sources:
            drawSpectra(gammaSource, "lead")

def plotAllPeaks(sources, material = -1):
    
    if ((material == -1) or (material.lower() == "aluminum")):
        
        for gammaSource in sources:
            counter = 0
            
            for peak in gammaSource.peaks:
                gammaSource.plotPeak("aluminum", counter)
                counter += 1
    
    if ((material == -1) or (material.lower() == "lead")):
        
        for gammaSource in sources:
            counter = 0
            
            for peak in gammaSource.peaks:
                gammaSource.plotPeak("lead", counter)
                counter += 1


def peakFitFunction(arealDensity, R0, attenCoeff, yOffset):
        #fit used to find the value of k' for each gamma source 
        
        #print("attenCoeff: ", attenCoeff)
        #print("attenCoeffType: ", type(attenCoeff))
        #print("R0: ", R0)
        #print("attenCoeff: ", attenCoeff)
        #print("yOffset: ", yOffset)
        
        yVals = (R0 * (np.exp(-attenCoeff * arealDensity)))+ yOffset 
        
        
        #plt.plot(arealDensity, yVals)
        
    
        return(yVals)
# =============================================================================
#     r, a, y = symbols('r a y')
#     
#     func = r*2.718**(-a * arealDensity) + y
#     
#     print("func type: ", type(func))
#     print("func: ", func)
#     
#     
#     #should write a general form of this in u func that makes use to the position of the params
#     #evaluated = func.subs([(r, R0), (a, attenCoeff), (y, yOffset)])
#     evaluated = func.subs([(r, R0), (y, yOffset)])
# 
#     
#     print("evaluated: ",evaluated)
# =============================================================================
    
    #print("R0: ", R0)
    #print("attenCoeff: ", attenCoeff)
    #print("arealDensity: ", arealDensity)
    #print(" ")
    #return (R0 * (np.exp(-attenCoeff * arealDensity)) + )
    #return evaluated


def fitPeak(gammaSource, blockerMaterial, peak = 0):
    #this will perform a nonlinear fit to find the value of k' for a particular peek    
    
    data = gammaSource.getPeakData(blockerMaterial, peak)
    xVals = data[0]
    yVals = data[1]
    yUncer = data[2]
    
    #added to try to deal with overflow values: change everything from float to decimal:
    
# =============================================================================
#     for i in range(0, len(xVals)):
#         xVals[i] = Decimal(xVals[i])
#         yVals[i] = Decimal(yVals[i])
#     
# =============================================================================
    
    
      
    xVals = np.array(xVals)
    yVals = np.array(yVals)
    
    initialGuess = []

    initialGuess.append(float(max(yVals))) #intialGuess for R0
    initialGuess.append(.01) #initialGuess for k'
    initialGuess.append(1) #initiaGuess for yOffset

    print("source: ", gammaSource.name)
    print("blocker: ", blockerMaterial)
    print("peak: ", peak)


    #return optimize.curve_fit(peakFitFunction, xVals, yVals, p0 = initialGuess, sigma = yUncer, absolute_sigma = True)
    #below is the currently working, except for a few peaks solution, im gonna try 
    #something i saw on stack overflow to make it more robust
    
    # arealDensity, R0, attenCoeff, yOffset
    
    paramBounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 50])

    return optimize.curve_fit(peakFitFunction, xVals, yVals, p0 = initialGuess, sigma = yUncer, bounds = paramBounds, absolute_sigma = True)
    #return optimize.curve_fit(peakFitFunction, xVals, yVals, p0 = initialGuess, sigma = yUncer)


    #return optimize.curve_fit(peakFitFunction, xVals, yVals, p0 = initialGuess)



    

def plotPeakFit(gammaSource, blockerMaterial, peak = 0):
    #plots the fit of a single peak across all of the gamma sources for a particular emmiter
    
    data = gammaSource.getPeakData(blockerMaterial, peak)
    xVals = data[0]
    yVals = data[1]
    yUncer = data[2]
    
    xVals = np.array(xVals)
    yVals = np.array(yVals)
    
    fit, sigma = fitPeak(gammaSource, blockerMaterial, peak)
    
    print("fit: ", fit)
    print("sigma: ", np.sqrt(np.diag(sigma)))
    
    uFunc.newPlotWindow(title = (gammaSource.name + " " + blockerMaterial + " peak at " +  str(gammaSource.peaks[peak].center) + "(MeV)"), yLabel = "integrated Rate", xLabel = "Areal density (mg cm^-2)", )
    #uFunc.newPlotConsole(yLabel = "integrated Rate", xLabel = "Areal density (mg/cm^-2)", )
    
    labelString = "R0: "+ str(fit[0]) + "\nk': " + str(fit[1]) +"\nY Offset: "+str(fit[2])
    
    plotSpace = uFunc.makePlotSpace(min(xVals), max(xVals), 300)
    #a plot space is just a more detailed version of the current plot, to
    #show of the real shape of the fit
    
    
    
    
    #plt.plot(xVals, yVals, 'bs')
    plt.errorbar(xVals, yVals, yerr = yUncer, fmt = 'none')
    plt.plot(plotSpace, peakFitFunction(plotSpace, fit[0], fit[1], fit[2]), 'r--', label = labelString)

    #goal posts:
    #these are points ive added based on the graphis from the lab manual to see how
    #far from my goals the first and last points are

    plt.legend(fontsize = 26)
    

def plotAllPeakFits(gammaSources, blockerMaterial = -1):
    
    
    if blockerMaterial == -1:
        plotAllPeakFits(gammaSources, "aluminum")
        plotAllPeakFits(gammaSources, "lead")
        return
    
    elif blockerMaterial == "aluminum":
        for gammaSource in gammaSources:  
            for peak in range(0, len(gammaSource.peaks)):
                
                plotPeakFit(gammaSource, "aluminum", peak)  

    elif blockerMaterial == "lead":
        for gammaSource in gammaSources:  
            for peak in range(0, len(gammaSource.peaks)):
                
                plotPeakFit(gammaSource, "lead", peak)  
    
    

def loadFinalGraphFromBook(blockerMaterial):
    #this method is called to load in the values required to plot the example graph
    #provided in the book

    bucket = []#this is a super structure to hold the data together
    
    currentFolder = "C:/Users/Hayes/Documents/MATLAB/GammaTextFiles/GraphData"

    energyVals = uFunc.arrayFromFile( (currentFolder + "/energyData.txt"), 1, 0)
    bucket.append(energyVals)
    
    
    if blockerMaterial.lower() == "aluminum":
        
        
        
        currentFolder = currentFolder + "/Aluminum/"
        
        totalVals = uFunc.arrayFromFile(currentFolder + "alTotal.txt", 1, 0)
        comptonVals = uFunc.arrayFromFile(currentFolder + "Compton13.txt", 1, 0)
        pairVals = uFunc.arrayFromFile(currentFolder + "pairAL.txt", 1, 0)
        PEVals = uFunc.arrayFromFile(currentFolder + "PE13.txt", 1, 0)
      
        bucket.append(totalVals)
        bucket.append(comptonVals)
        bucket.append(pairVals)
        bucket.append(PEVals)
        
        return bucket
        
        
    elif blockerMaterial.lower() == "lead":
        
        currentFolder = currentFolder + "/Lead/"
        
        totalVals = uFunc.arrayFromFile(currentFolder + "leadTotal.txt", 1, 0)
        comptonVals = uFunc.arrayFromFile(currentFolder + "Compton82.txt", 1, 0)
        pairVals = uFunc.arrayFromFile(currentFolder + "pairPB.txt", 1, 0)
        PEVals = uFunc.arrayFromFile(currentFolder + "PE82.txt", 1, 0)
        
        bucket.append(totalVals)
        bucket.append(comptonVals)
        bucket.append(pairVals)
        bucket.append(PEVals)
        
        return bucket


def plotKPrime(gammaSources, blockerMaterial, logBars = True, graphFromBook = True):
    #this creates the final graph by using the fit to find the values of k'
    #for each fit, then using the center of the peak object as its energy value
    

    
    xVals = []
    yVals = []
    yUncertainty = []
    names = []
    #used to annotate data points
    for gammaSource in gammaSources:
        #for each GammaSource
        
        #used for annotation
        currentName = gammaSource.name
        for peakIndex in range(0, len(gammaSource.peaks)):
            #for each peak
            xVals.append(gammaSource.peaks[peakIndex].center)
            data, uncertainty = fitPeak(gammaSource, blockerMaterial, peakIndex)
            
            names.append((currentName + "peak: " + str(peakIndex)))
            
            
            yVals.append(data[1]) #data[1] contains k'
            sigma = np.sqrt(np.diag(uncertainty))
            
                #TODO: deterimne if this propagates the error in our fit for K' through
                #to the values of data points: it does not, it appears as though it only
                #takes into acount the error in k', which may be legit, but i dont
                #think it is. I think we need to use the propagation of error formula
                #to take into acount the uncertainty in R0 and stuff. That should
                #probably be handled in the fit peak part, but i dont know if that 
                #would cause an error cascade. I dont think it would, because at worst
                #i'll be changing it from a list to a float, and also that float should
                #be more accurate
            
            yUncertainty.append(sigma[1])
    
    
    
    #this is just done so the connecting line between the points looks right,
    #it should not change the data in any way
    uFunc.sortLists(xVals, yVals, names ,yUncertainty)
    
    
    
    #TODO:
    #figure out if the units on this are right, cause at the moment i dont htink they are
    #all of the graphs used to find k are in mg/cm^2
    #however this final graph is in cm^2/g
    #meaning i need to multiply by 1000
    yVals = np.array(yVals)
    yVals = yVals * 1000
    yUncertainty = np.array(yUncertainty)
    yUncertainty = yUncertainty * 1000
    
    print("uncertainty: ", yUncertainty)
    
    uFunc.newPlotWindow(yLog = True, xLog = True, xLabel = "Energy (MeV)", yLabel = "Mass attenuation coefficient (cm^2/g)", title = ("Mass attenuation Coefficient of "+blockerMaterial))
    
    plt.plot(xVals, yVals, 'bs')
    plt.errorbar(xVals, yVals, yerr = yUncertainty, fmt = 'none')
    #plt.plot(xVals, yVals, 'r--')
    
    for name in range(0, len(names)):
        plt.annotate(names[name], (xVals[name], yVals[name]))
    
    if logBars:
        uFunc.plotLogLogBars(-2,2,-2,2)
        
        
    if graphFromBook:
        
        data = loadFinalGraphFromBook(blockerMaterial)
        
        energy = data[0]
        total = data[1]
        compton = data[2]
        pair = data[3]
        PE = data[4]
        
        #due to how this was read in, it must have been wrapped in an extra list layer
        #or some such expereince. In any case, this is a list of lists with only one 
        #entrey which is the list we want
        energy = energy[0]
        total = total[0]
        compton = compton[0]
        pair = pair[0]
        PE = PE[0]
        
        plt.plot(energy, total, 'm-', label = "total Attenuation")
        plt.plot(energy, compton, label = "Attenuation due to Compton Scattering")
        plt.plot(energy, pair, label = "Attenuation due to pair production")
        plt.plot(energy, PE, label = "Attenuation due to PE")
        
        
        plt.legend(fontsize = 20)
        

def plotCleanSpectra(gammaSource, testRun, yLog = True):
    #This does the same thing as plot spectra, only it removes indices where the y value
    #is equal to zero, in an attempt to make the graph more ledigible
    
    xCopy = copy.deepcopy(GammaSource.energyValues)
    yCopy = copy.deepcopy(testRun.counts)
    
    yCopy = np.array(yCopy)
    yCopy = yCopy/testRun.time
    
    xHolder = []
    yHolder = []
    
    for i in range(0, len(xCopy)):
        if not (yCopy[i] == 0):
            xHolder.append(xCopy[i])
            yHolder.append(yCopy[i])
     
    if yLog:
        plt.yscale('log')
    
    plt.plot(xHolder, yHolder)



def plotFullCleanSpectra(gammaSource, blockerMaterial = -1, peakBounds = True):
    
    if blockerMaterial == -1:
        plotFullCleanSpectra(gammaSource, "lead")
        plotFullCleanSpectra(gammaSource, "aluminum")
        return
    
    elif blockerMaterial == "aluminum":
        plt.figure()
        plt.title( (gammaSource.name + " through " + blockerMaterial ))
        for testRun in gammaSource.aluminum:
            plotCleanSpectra(gammaSource, testRun)
                
    elif blockerMaterial  == "lead":
        plt.figure()
        plt.title( (gammaSource.name + " through " + blockerMaterial ))
        for testRun in gammaSource.lead:
            plotCleanSpectra(gammaSource, testRun)
       
    else:   
        print("invalid blocker input")
    
    
    if(peakBounds):
        #draws lines to show where the reigons of interest are
        for peak in gammaSource.peaks:
            drawPeakBox(gammaSource, peak, blockerMaterial)
        

    
def plot3DSpectra(gammaSource, blockerMaterial = -1 ,testRun = -1, oneFigure = True, peakBounds = True, log = True):
    #draws the spectra for a gamma source, for all different blocker materials
    
    #TODO: Add the box around the peak, but 3D
    #TODO: Add the fit line for k' to this plot. THat would be real spicy
    
    
    
    if blockerMaterial == -1:
        #this means we plot both spectra
        plot3DSpectra(gammaSource, "aluminum",testRun, oneFigure, peakBounds, log)
        plot3DSpectra(gammaSource, "lead", testRun, oneFigure, peakBounds, log)
        return
    
    #this is used to switch between aluminum and lead blocker arrays
    testRunArrayHolder = []
    materialString = ""
    
    if blockerMaterial.lower() == "aluminum":
        testRunArrayHolder = gammaSource.aluminum
        materialString = "aluminum"
    
    elif blockerMaterial.lower() == "lead":
        testRunArrayHolder = gammaSource.lead
        materialString = "lead"
        
    else:
        print("tried to make a spectra for a non-existant blocker")
        return
        
    
    #used for printing one or all of the spectra
    counter = -1
    endValue = len(GammaSource.pbDensities)-1
    
    
    
    #if the user designates a specific testRun
    if not testRun == -1 :
        
        counter = testRun
        endValue = testRun+1
        
    
    axisSize = 32
    labelSize = 20
    tickSize = 22
    thickness = .5
    
    titleString = gammaSource.name + " through " + materialString
    
    arealDensities = []
    
    if(blockerMaterial == "lead"):
        arealDensities = GammaSource.pbDensities
        
    else:
       arealDensities = GammaSource.alDensities
        
        
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d', title = titleString)
         
    ax.set_xlabel('Energy of gamma ray (Mev)', fontsize = labelSize)
    ax.set_ylabel('Areal Density Of Blocker (cm^2/g)', fontsize = labelSize)
    ax.set_zlabel('Rate of detection (counts/second)', fontsize = labelSize)
    ax.tick_params(labelsize=15)
    
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    
            
    if(log):
        ax.set_zscale('log')
    
    #this loops backwards to deal with a weird graphical phenomonon that was making things
    #hard to read
    for x in range(endValue, counter, -1):

        toPlot = testRunArrayHolder[x]
        yPosition = np.ones(len(toPlot.counts))*arealDensities[x]
        #this is used to place each spectra on a seperate y level
        
        ax.plot(GammaSource.energyValues, yPosition ,toPlot.counts/toPlot.time)
        
    
    
    #here is a section that adds the fit for k prime to this plot, allowing us to 
    #view that information on the context of the peaks relations to each other
    
    for peak in range(0, len(gammaSource.peaks)):
    
        kprimeFit, kprimeError = fitPeak(gammaSource, blockerMaterial, peak)
        kPrimeXVals = np.ones(len(GammaSource.energyValues)) * gammaSource.peaks[peak].center
        
        if(blockerMaterial == "lead"):
            kPrimeYVals = uFunc.makePlotSpace(0, max(GammaSource.pbDensities), len(GammaSource.energyValues))
    
        else:
            kPrimeYVals = uFunc.makePlotSpace(0, max(GammaSource.alDensities), len(GammaSource.energyValues))
            
            
        
        #issue with this atm is the fact that it's the k prime of the integrated peak,
        #so for this to fit on the graph well we need to divide by the width of the peak
        #so that didnt work
        kPrimeZVals = peakFitFunction(kPrimeYVals, kprimeFit[0], kprimeFit[1], kprimeFit[2])
        #kPrimeZVals = kPrimeZVals/(gammaSource.peaks[peak].leftEnergyPosition - gammaSource.peaks[peak].rightEnergyPosition)
    
        #add peak bounds
    
        boxCoords = makePeakBox(gammaSource, gammaSource.peaks[peak], blockerMaterial)
        yCoords = [0,0,0,0,0]
        
        print(max(arealDensities))
        
        for x in range(0, len(boxCoords[0])):
            boxCoords[0].append(boxCoords[0][x])
            boxCoords[1].append(boxCoords[1][x])
            yCoords.append(max(arealDensities))
            
        
        ax.plot(boxCoords[0], yCoords, boxCoords[1], 'b--')
    
    
        #ax.plot(kPrimeXVals, kPrimeYVals, kPrimeZVals, 'bs',label = "k fit")

    ax.legend()
    
    





#this code was used to find the areal densities of each blocker
densityOfLead = 11.34 #g/cm^3
densityOfAl = 2.70 #g/cm^3

#convert from g to mg:
densityOfLead = densityOfLead * 1000
densityOfAl = densityOfAl * 1000

leadToConvert = [0.044, 0.088, 0.132, 0.199, 0.243]#all of these values are in inches
alToConvert = [0.187, 0.75, 1.5, 3]#all of these values are in inches


def inchToAreal(inchThicknesses, materialDensity):
    
    #convert from inch to CM, 
    #then multiply thickness * density
    
    for x in range(0, len(inchThicknesses)):
        inchThicknesses[x] = inchThicknesses[x]*2.54
        #convert to CM
        inchThicknesses[x] = inchThicknesses[x]*materialDensity
        #convert to areal density 
        
    return inchThicknesses


#script:
                    
           
allSources = loadData("C:/Users/Hayes/Documents/MATLAB/GammaTextFiles")         
     
barium = allSources[0]
cesium = allSources[1]
cobalt57 = allSources[2]
cobalt60 = allSources[3]
manganese = allSources[4]
sodium = allSources[5]

lead = "lead"
al = "aluminum"


#TODO: add more peaks that are small variations around these peaks to account for 
#uncertainty due to the arbitrary nature of the location of the peak



barium.addPeak(0.0665, 0.11101)
barium.addPeak(0.31559, 0.3696)

cesium.addPeak(0.575, 0.6889)

cobalt57.addPeak(.1064, .1616)

cobalt60.addPeak(1.075, 1.14)
cobalt60.addPeak(1.21, 1.28)

manganese.addPeak(750/1000, 830/1000)

sodium.addPeak(.464, .5239)
sodium.addPeak(1.175, 1.23)



#this is here so we know which data point coresponds to which gamma source in the
#k' plot
peakCenters = []
peakMaterials = []
for source in allSources:
    for peak in source.peaks:
        peakCenters.append(peak.center)
        peakMaterials.append(source.name)

uFunc.sortLists(peakCenters, peakMaterials)


toPrint = [peakMaterials, peakCenters]




#plots all the final graphs needed for the lab report
def plotFinalGraphs():
    
    
    
    plotKPrime(allSources, "lead")
    plotKPrime(allSources, "aluminum")
    


    for gammaSource in allSources:
        plot3DSpectra(gammaSource)



