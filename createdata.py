from math import sqrt; from itertools import count, islice
import csv
from numpy import binary_repr

def isPrime(n):
    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))

def decimalToBinary(dec, numberOfBits):
    answer = []
    
    strBin = (binary_repr(dec, width = numberOfBits))
    for bits in list(strBin):
        answer.append(bits)
    zeros = []
    for i in range(0, numberOfBits-len(answer)):
        zeros.append(0)
    return zeros + answer

numRange = 50
listOfBins = []
start = 0

for start in range(numRange):
    tempBinNum = decimalToBinary(start, 32)
    binNumber = []
    for bit in tempBinNum:
        if(bit == '0'):
            binNumber.append(.001)
        else:
            binNumber.append(.999)
        
    if(isPrime(start)):
        binNumber.append(1)
    else:
        binNumber.append(0)
    listOfBins.append(binNumber)
    start = start + 1

with open('primeBinary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(listOfBins)
