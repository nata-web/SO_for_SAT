import os
import numpy as np

import SO_for_SAT
import makeplots

path = os.path.abspath(os.getcwd())
path = os.path.join(path,'output','output_python','SAT')
path = os.path.join(path,"2023-07_IEEE_figures")

import matplotlib.pyplot as plt
plt.style.use('ieee_conf.mplstyle') 

def main():
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

    PO = SO_for_SAT.plotOptions()
    PO.colWidth  = 3.487
    PO.pageWidth  = 7.140
    PO.factr = [1.618,1.5][0]
    PO.path = path
    PO.plot6 = True
    PO.saveFigures = False
    
    makeplots.twoProblems(PO) # Fig 1, Fig 2

    makeplots.oneBigPlot(PO) # Fig 3
        
    makeplots.extraPlot(PO)  # Fig 4 Extra for weighted by borders

if __name__ == '__main__':
  main()            
