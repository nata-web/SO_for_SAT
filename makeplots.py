import SO_for_SAT

import numpy as np
import os
import shapely, shapely.affinity
import borders

# to avoid output from plots (when using HPC), uncomment the next two lines
# import matplotlib
# matplotlib.use('Agg') # to make a plot without needing an X-server use the Agg backend; Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# For plotting polygons
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

import matplotlib.transforms as mtransforms # Func: ScaledTranslation fot automatic labeling

# For creating your own colormap
from matplotlib.colors import LinearSegmentedColormap

SVGcolors = ["#ddaa33","#66ccee", "#004488", "#bb5566"]
# Create the 'iridescent' scheme (https://personal.sron.nl/~pault/)
clrs = ['#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF',
        '#D0E7CA', '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1',
        '#8DCBE4', '#81C4E7', '#7BBCE7', '#7EB2E4', '#88A5DD',
        '#9398D2', '#9B8AC4', '#9D7DB2', '#9A709E', '#906388',
        '#805770', '#684957', '#46353A']

cmap = LinearSegmentedColormap.from_list("",clrs)
cmap.set_bad('#999999')

elim_options = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1], [5,5,1]]

# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def plot_poly_or_multi(ax,polyC,**kwArgs):
    if type(polyC) == shapely.geometry.multipolygon.MultiPolygon:
        for poly in polyC.geoms:
            plot_polygon(ax, poly, **kwArgs)
    else:
        plot_polygon(ax, polyC, **kwArgs)

def plot_countries(ax,Countries,countries_colors):
    for i,Country in enumerate(Countries):
        polyC = Countries[Country]
        cInd = countries_colors[i]
        plot_poly_or_multi(ax, polyC, facecolor = SVGcolors[cInd%len(SVGcolors)], edgecolor='black', linewidth=0.2)

def make_1by2_plot(CO, result, fig, axs, labels, trans, matPlots, label_size = 0.75):
    
    resets = result.energies.shape[0]//3
    
    if matPlots:
        # subplot(121)
        label = labels[0]
        label_bold = r"$\boldsymbol{" + label + "}$" 
        
        ax = axs[label]
        
        img = ax.matshow(result.wOrig,  interpolation="none", cmap=cmap);
        ax.text(label_size, 1.0, label_bold, transform=ax.transAxes + trans, color='white', fontweight='bold',
                fontsize='medium', verticalalignment='top', fontfamily='serif', 
                bbox = dict(facecolor='black', alpha=0.4))
        ax.xaxis.set_ticks_position("bottom")
        
        # subplot(122)
        label = labels[1]
        label_bold = r"$\bm{" + label + "}$" 
        
        ax = axs[label]
        
        img = ax.matshow(result.w, vmin=result.wOrig.min(), vmax=result.wOrig.max(), interpolation="none", cmap=cmap);
        
        ax.text(label_size, 1.0, label_bold, transform=ax.transAxes + trans, color='white', fontweight='bold',
                fontsize='medium', verticalalignment='top', fontfamily='serif', 
                bbox = dict(facecolor='black', alpha=0.4))
        ax.xaxis.set_ticks_position("bottom")
        ax.set_yticks([])
        cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
        fig.colorbar(img, cax = cax)
 
    else:
        # subplot(121)
        label = labels[0]
        label_bold = r"$\bm{" + label + "}$" 
        
        ax = axs[label]
        plotEnergies(CO,ax,result.energies)
        ax.text(0.8, 1.0, label_bold, transform=ax.transAxes + trans, color='black', fontweight='bold',
            fontsize='medium', verticalalignment='top', fontfamily='serif')
        
        ax.set_ylabel("Energy")
        ax.tick_params(axis='both')
        
        # subplot(122)
        label = labels[1]
        label_bold = r"$\boldsymbol{" + label + "}$" 
        
        ax = axs[label]
        plot_countries(ax,CO.Countries,result.countries_colors)
        
        ax.text(0.8, 1.0, label_bold, transform=ax.transAxes + trans, color='black', fontweight='bold',
                fontsize='medium', verticalalignment='top', fontfamily='serif')
        ax.set_xticks([])
        ax.set_yticks([])

def make_Es_plot(CO, result, fig, axs, label, trans, xlabel, vline):
    
    resets = result.energies.shape[0]//3
    
    label_bold = r"$\bm{" + label + "}$" 
    ax = axs[label]
    plotEnergies(CO,ax,result.energies,labels=['Before learning','Learning','After learning'])
    ax.text(0.01, 1.0, label_bold, transform=ax.transAxes + trans, color='black', fontweight='bold',
            fontsize='medium', verticalalignment='top', fontfamily='serif')
    ax.set_ylabel("Energy")
    if xlabel:
        ax.set_xlabel("Resets")
    ax.tick_params(axis='both')
    if vline:
        # multiple lines all full height
        ax.vlines(x=[resets,resets*2], ymin = energies.min(), ymax = energies[:resets,-1].max(), 
                  colors='grey', ls=':', lw=1)
        
        textstr = '\n'.join((r"$\mathbf{Self -}$",r"$\mathbf{Optimization}$"))
                
        ax.text(0.34, 1.0, textstr, transform=ax.transAxes + trans, color='black', 
                fontweight='bold', fontsize='medium', verticalalignment='top', 
                fontfamily='serif', multialignment='center')

def plotEnergies(CO,ax,energies,colors=['#004488','#BB5566','#66ccee'],labels=[None,None,None]):
  for which in range(3):
    ax.plot(np.arange(which*CO.resets,(which+1)*CO.resets), energies[which*CO.resets:(which+1)*CO.resets,-1], c=colors[which], ls='None', marker='o', markersize=1, label=labels[which])
    
def plot_6(CO, result, clauses, check_sat=False, PO=SO_for_SAT.plotOptions()):
    
    resets = result.energies.shape[0]//3
    steps = result.energies.shape[1]
    
    fig,axs = plt.subplots(2,3,figsize=[15,10])
    # subplot(231)
    img = axs[0,0].matshow(result.wOrig, aspect="auto", cmap=cmap);
    axs[0,0].set_title('Initial weights')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)     
    fig.colorbar(img, ax = axs[0,0], cax=cax)
    
    # subplot(232)
    img = axs[0,1].matshow(result.w, aspect="auto", cmap=cmap);
    axs[0,1].set_title('Final weights')
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)     
    fig.colorbar(img, ax = axs[0,1], cax=cax)     

    # subplot(233)
    E_NL = result.energies[resets-50:resets]
    
    axs[0,2].plot(np.arange(steps), E_NL.T)
    axs[0,2].set_title('Energies WoL')
    axs[0,2].set_xlabel("Timesteps")
    axs[0,2].set_ylabel("Energy")
    
    # subplot(236)
    E_L = result.energies[resets+resets-50:resets*2]
    
    axs[1,2].plot(np.arange(steps)+steps, E_L.T)
    axs[1,2].set_title('Energies WL')
    axs[1,2].set_xlabel("Timesteps")
    axs[1,2].set_ylabel("Energy")
    
    # subplot(234)
    plotEnergies(CO,axs[1,0],result.energies,['#DDAA33','#BB5566','#004488'])
    
    # if SATproblem == "Liars":
    #     if check_sat: # is it SAT?
    #         Es_NL = energies[:resets,-1]
    #         wrong_s_Es_NL = Es_NL[sat_states[:resets,-1] == 0]
    #         Es_L = energies[resets:resets*2,-1]
    #         wrong_s_Es_L = Es_L[sat_states[resets:resets*2,-1] == 0]
    #         Es_NL2 = energies[resets*2:resets*3,-1]
    #         wrong_s_Es_NL2 = Es_NL2[sat_states[resets*2:resets*3,-1] == 0]
    
    #         # wrong_s_Es_L = Es_L[sat_states == False]
    #         axs[1,0].plot(np.arange(0,resets)[sat_states[:resets,-1] == 0], wrong_s_Es_NL, c='black', ls='None', marker='x', markersize=2)
    #         axs[1,0].plot(np.arange(resets,resets*2)[sat_states[resets:resets*2,-1] == 0], wrong_s_Es_L, c='black', ls='None', marker='x', markersize=2)
    #         axs[1,0].plot(np.arange(resets*2,resets*3)[sat_states[resets*2:resets*3,-1] == 0], wrong_s_Es_NL2, c='black', ls='None', marker='x', markersize=2)
    
    axs[1,0].set_title('Attractor states visited')
    axs[1,0].set_xlabel("Resets")
    axs[1,0].set_ylabel("Energy")
    
    # subplot(235)
    if CO.SATproblem == "Liars":
        avg_fac = int(3*resets / 100)
        eMin,eMax=(result.energies[:,-1].min(),result.energies[:,-1].max())
        h = np.array([np.histogram(r,30,range=(eMin,eMax))[0] for r in result.energies[:,-1].reshape((avg_fac,100))])
    
        img = axs[1,1].matshow(h.T,origin="lower",extent=[0,avg_fac,eMin,eMax],aspect='auto') 
        axs[1,1].set_title('Histogram of attractor energies')
        divider = make_axes_locatable(axs[1,1])
        cax = divider.append_axes("right", size="5%", pad=0.05)     
        fig.colorbar(img, ax = axs[1,1], cax=cax)
        axs[1,1].set_xlabel("Time window")
        axs[1,1].set_ylabel("Energy")
        
    elif CO.SATproblem == "MapColoring":
        plot_countries(axs[1,1],CO.Countries,result.countries_colors)
    
    fig_name = CO.SATproblem
    # if SATproblem == "Liars":
    #     main_title = r"" + SATproblem + " , N=" + str(N) + " people," + \
    #         str(M) + " statements, steps=" + str(steps) + ", resets=" + \
    #         str(resets) + ", α = " + str(round(1/eta,11)) + ", seed$_{sat}$ = " + \
    #         str(seed_sat) + ", seed$_{sim}$ = " + str(seed_sim)
            
    # elif SATproblem == "MapColoring":
    #     fig_name += "_" + Map
    #     main_title = r"" + SATproblem + ", n=" + str(n) + " countries," + \
    #         str(M) + " colors," + str(N) + " nodes," + " steps=" + str(steps) + \
    #         ", resets=" + str(resets) + ", α=" + str(round(1/eta,11)) + \
    #         ", seed$_{sim}$=" + str(seed_sim) + ", Weighted? " + str(weigh_by_border)
            
    #     if startState is not None:
    #         main_title += ", startState" 
    
    plt.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # fig.suptitle(main_title, y=0.99)
    
    if PO.saveFigures:
        fig_name += "_N_" + str(CO.N) + "_eta_" + str(CO.eta) + "_resets_" + str(CO.resets) + "_steps_" + str(CO.steps) + "_ss_" + str(CO.seed_sim)
        if CO.SATproblem == "Liars": 
            fig_name = fig_name + "_ssat_" + str(CO.seed_sat)
        else:
            if CO.weigh_by_border:
                fig_name += "_weighted"
                
            if CO.startState is not None:
                fig_name += "_startState" 
        
        plt.savefig(os.path.join(PO.path, fig_name + '.png'))

def getOptions(SATproblem,Map,PO):
    CO = SO_for_SAT.calcOptions()
    
    # Chose problem
    CO.SATproblem = SATproblem
    
    # Chose k for kSAT
    CO.SATtype = [2,3][0] 
    
    if SATproblem == "Liars":
        CO.N = 50  # Number of people
        CO.M = 34  # Number of statements, M <= N
        
        # seed_sat = np.random.randint(90000)
        CO.seed_sat = 59194
        statements, clauses_dimacs, clauses = SO_for_SAT.generate_LiarsSAT_problem(CO)

        write_dimacs_file(PO,CO, clauses_dimacs)
        CO.resets = 10
        CO.alphas = [5e-7]
        allClauses = (clauses,)
        
    elif SATproblem == "MapColoring":
               
        # Chose map
        CO.Map = Map
        
        CO.Countries = getattr(borders, CO.Map)
        n = len(CO.Countries)
        CO.n = n
        
        CO.M = 2  # Number of colors
        CO.N = n*CO.M  # Number of actual nodes for the SO model
        
        CO.weigh_by_border = False
        if CO.Map == "Checker":
            CO.resets = 100
            CO.alphas = [9e-7]
        elif CO.Map == "Europe":
            CO.resets = 100
            CO.alphas = [9e-7]
        elif CO.Map == "SouthAmerica":
            if CO.weigh_by_border:
                CO.resets = 1000
                CO.alphas = [1.5e-5]
            else:
                CO.resets = 1
                CO.alphas = [3e-5]
                # resets 1000, alpha 3e-5: 8 border violations
                # resets 1000, alpha 5e-5: 8 border violations (different coloring)
        elif CO.Map == "Japan":
            if CO.weigh_by_border:
                CO.resets = 1000
                CO.alphas = [2e-5]
            else:
                CO.resets = 1000
                CO.alphas = [2e-6]

        CO.elim_arr = elim_options[5] # '4' for not eliminating anything
        allClauses = clausesForBorders(PO,CO)

    return CO, allClauses

def clausesForBorders(PO, CO):
    n=len(CO.Countries)
    CO.bordersMap = borders.calcBorders(CO.Countries)
    bmin = [[b[0],b[2],b[4]] for b in CO.bordersMap]
    adj_borderMat = np.zeros((n,n),dtype=int)
    adj_borderMat[tuple(np.array(bmin).T[:2].astype(int))] = 1
    borderL = np.array(bmin)
    borderL_mat = np.zeros((n,n),dtype=int)
    borderL_mat[tuple(np.array(bmin).T[:2].astype(int))] = borderL[:,2]
    maxL = borderL_mat.max()
    borderL_mat = borderL_mat/maxL

    clauses, all_colored_clauses, one_cpn_clauses, adj_clauses, clauses_dimacs = SO_for_SAT.generate_SAT_coloring(CO, adj_borderMat)
    SO_for_SAT.write_dimacs_file(PO,CO, clauses_dimacs)

    return clauses, all_colored_clauses, one_cpn_clauses, adj_clauses, borderL_mat

def oneBigPlot(PO):

    CO, allClauses = getOptions("MapColoring", "SouthAmerica", PO)
    CO.steps = CO.N*20
    CO.seed_sim = 60135
    CO.unWeightedEnergies = True

    PO.saveFigures = True
    label_size = 0.8

    clausesW_arr = [[1,1,1], [5,5,1], [1,1,1]]
    alphas_arr = [8e-7, 2.1e-5, 2.1e-5]
    resets_arr = [1000, 1000, 1000]
    weigh_by_border_arr = [False, False, True]

    labels_arr = labels_mos = [['a','b','c','d'],['e','f','g','h'],['i','j','k','l']]

    fig3, axs3 = plt.subplot_mosaic(labels_mos, figsize=[PO.pageWidth,PO.colWidth*1.5], layout='constrained')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig3.dpi_scale_trans)

    for CO.elim_arr,CO.alpha,CO.resets,CO.weigh_by_border,labels in zip(clausesW_arr, 
                                                                alphas_arr, 
                                                                resets_arr, weigh_by_border_arr, labels_arr):
        CO.eta = int(1/CO.alpha)
        print("""The {} problem:""".format(CO.SATproblem))
        if CO.SATproblem == "MapColoring":
            print("""The {} map""".format(CO.Map), ", Weighted? ", str(CO.weigh_by_border))

        result = SO_for_SAT.simulate(CO, allClauses, PO=PO)
        if PO.plot6:
          plot_6(CO,result,allClauses[0], PO=PO)
          SO_for_SAT.checkBorders(CO, allClauses, result.stateLearn, elim_options)

        make_1by2_plot(CO,result, fig3, axs3, labels[:2], trans,  True, label_size)
        make_1by2_plot(CO,result, fig3, axs3, labels[2:], trans, False, label_size)


    label = labels_arr[-1][-2]
    ax = axs3[label]
    ax.set_xlabel("Resets")

    label = labels_arr[-1][-1]
    ax = axs3[label]
    ins = ax.inset_axes([0.02,0.3,0.45,0.35])
    plot_countries(ins, CO.Countries, result.countries_colors)

    ins.set_xlim([-87, -75])
    ins.set_ylim([6, 12])
    ins.set_xticks([])
    ins.set_yticks([])

    fig3.set_size_inches(PO.pageWidth, PO.colWidth*1.5)
    fig3.savefig(os.path.join(PO.path,"3by4bigPlot.pdf"))

def extraPlot(PO):
    CO, allClauses = getOptions("MapColoring", "SouthAmerica", PO)
    CO.steps = CO.N*20
    CO.seed_sim = 60135
    CO.unWeightedEnergies = True

    clausesW_arr = [[1,1,1]]
    alphas_arr = [6e-6]
    resets_arr = [1000]
    weigh_by_border_arr = [True, True]

    labels_arr = labels_mos = [['a','b']]

    label_size=0.8

    fig4, axs4 = plt.subplot_mosaic(labels_mos, figsize=[PO.colWidth,PO.colWidth/2], layout='constrained')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig4.dpi_scale_trans)

    for CO.elim_arr,CO.alpha,CO.resets,CO.weigh_by_border,labels in zip(clausesW_arr, 
                                                                alphas_arr, 
                                                                resets_arr, weigh_by_border_arr, labels_arr):
        CO.eta = int(1/CO.alpha)
        print("""The {} problem:""".format(CO.SATproblem))
        if CO.SATproblem == "MapColoring":
            print("""The {} map""".format(CO.Map), ", Weighted? ", str(CO.weigh_by_border))

        result = SO_for_SAT.simulate(CO, allClauses, PO=PO)
        if PO.plot6:
          plot_6(CO, result,allClauses[0], PO=PO)
          SO_for_SAT.checkBorders(CO,allClauses, result.stateLearn, elim_options)

        make_1by2_plot(CO,result, fig4, axs4, labels, trans, False, label_size)


    label = labels_arr[0][0]
    ax = axs4[label]
    ax.set_xlabel("Resets")

    label = labels_arr[0][1]
    ax = axs4[label]
    ins = ax.inset_axes([0.02,0.3,0.45,0.35])
    plot_countries(ins, CO.Countries, result.countries_colors)

    ins.set_xlim([-87, -75])
    ins.set_ylim([6, 12])
    ins.set_xticks([])
    ins.set_yticks([])

    fig4.set_size_inches(PO.colWidth,PO.colWidth/2)
    fig4.savefig(os.path.join(PO.path,"1by2Extra.pdf"))

def twoProblems(PO):

    seedsSim = [71266] 

    for seedSim in seedsSim:
        labels_arr = labels_mos = [['a','b'],['c','c'],['d','e'],['f','f']]
        fig1, axs1 = plt.subplot_mosaic(labels_mos, figsize=[PO.colWidth,PO.colWidth*1.8], layout='constrained')
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig1.dpi_scale_trans)

       # Liars Problem
        CO1 = SO_for_SAT.calcOptions()

        CO1.seed_sim = seedSim 

        CO1.SATproblem = "Liars"
        CO1.SATtype = 2
        CO1.N = 50  # Number of people
        CO1.n = CO1.N
        CO1.M = 34  # Number of statements, M <= N
        CO1.seed_sat = 58345 
        CO1.resets = 1000
        CO1.alpha = 2.5e-7 
        CO1.eta = int(1/CO1.alpha)
        CO1.steps = CO1.N*20
        print("""The {} problem:""".format(CO1.SATproblem))
        statements, clauses_dimacs, clauses = SO_for_SAT.generate_LiarsSAT_problem(CO1)
        SO_for_SAT.write_dimacs_file(PO,CO1,clauses_dimacs)
        result = SO_for_SAT.simulate(CO1,(clauses,),PO=PO)

        with open(os.path.join(PO.path, 'Es_seeds_' + str(CO1.resets) + "_" + str(CO1.eta) + ".txt"), 'a') as f:
            f.write(str(SO_for_SAT.calcE(result.wOrig, result.I, result.c, -1*result.stateLearn)) + " " + str(CO1.seed_sim))
            f.write("\n")

        if PO.plot6:
          plot_6(CO1, result, clauses, PO=PO)

        make_1by2_plot(CO1, result, fig1, axs1, labels_arr[0], trans,  True)
        xlabel = True
        vline = False
        make_Es_plot(CO1, result, fig1, axs1, labels_arr[1][0], trans, xlabel, vline)
        ax = axs1[labels_arr[1][0]]
        ax.legend(loc='upper right')

        # Checkerboard Map Coloring Problem
        CO2 = SO_for_SAT.calcOptions()
        CO2.seed_sim = seedSim 
        CO2.SATproblem = "MapColoring"
        CO2.SATtype = 2
        CO2.Map = "Checker"
        CO2.weigh_by_border = False
        n_tiles = 8
        CO2.Countries = borders.makeCheckerboard(n_tiles)
        n = len(CO2.Countries)
        CO2.n = n
        CO2.M = 2  # Number of colors
        CO2.N = n*CO2.M  # Number of actual nodes for the SO model
        
        CO2.resets = 40 
        CO2.alpha = 8e-21 
        CO2.eta = int(1/CO2.alpha)
        CO2.steps = CO2.N*10 
        print("""The {} problem:""".format(CO2.SATproblem))
        if CO2.SATproblem == "MapColoring":
            print("""The {} map""".format(CO2.Map), ", Weighted? ", str(CO2.weigh_by_border))
        CO2.elim_arr = [1,1,1]

        allClauses = clausesForBorders(PO, CO2)
        result = SO_for_SAT.simulate(CO2, allClauses, PO=PO)

        if PO.plot6:
          plot_6(CO2, result, allClauses[0], PO=PO)

        make_1by2_plot(CO2, result, fig1, axs1, labels_arr[2], trans,  True)
        xlabel = True
        vline = False
        make_Es_plot(CO2, result, fig1, axs1, labels_arr[3][0], trans, xlabel, vline)
        ax = axs1[labels_arr[3][0]]
        ax.legend(loc='upper right')
        
        fig2, axs2 = plt.subplots(figsize=(PO.colWidth/2,PO.colWidth/2))
        ax = axs2
        plot_countries(ax,CO2.Countries,result.countries_colors)
        ax.set_xticks([])
        ax.set_yticks([])
        fig2.tight_layout()

        fig1.set_size_inches(PO.colWidth, PO.colWidth*1.8)
        fig1.savefig(os.path.join(PO.path, "Liars_Checker_all.pdf"))
        fig2.set_size_inches(PO.colWidth/2, PO.colWidth/2)
        fig2.savefig(os.path.join(PO.path,"Colored ckeckerboard.pdf"))
