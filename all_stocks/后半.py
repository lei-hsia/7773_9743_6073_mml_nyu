
from jpype import *
import random
import math

def teCal(sourceArray, destArray):
	# Create a TE calculator and run it:
	teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
	teCalc = teCalcClass()
	teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
	teCalc.initialise(1, 0.5) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
	teCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))
	# For copied source, should give something close to 1 bit:
	result = teCalc.computeAverageLocalOfObservations()
	print("TE result %.4f bits" % (result))


# Change location of jar to match yours:
jarLocation = "./infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


arr_list = []
for i in quotes:
	arr_list.append(i["Close"].tolist())

# get every pair from those stocks
teArray = list(itertools.combinations(arr_list, 2))

for i in teArray:
	teCal(i[0], i[1])  # 1st & 2nd from each pair