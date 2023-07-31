# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:58:24 2023

@author: Tesh
"""

import os
import sys
import numpy as np
import sympy as sp
import pickle as pickle      # Func: dump, load

import random

F = 1     # 1 to load Fortran module, 'compile' to dynamically compile it,
          # 0 to run without Fortran
speed = 1 # speed up by not explicitly adding w=w+dw

isBipolar = 1
    
class container():
  def __init__(self,copyFrom=None):
    if copyFrom is not None:
      for option in dir(copyFrom):
        if option[:2]!='__':
          setattr(self,option,getattr(copyFrom,option))

class plotOptions(container):
  def __init__(self,copyFrom=None):
    self.dump=False
    self.saveFigures=False
    super().__init__(copyFrom)

class calcOptions(container):
  def __init__(self,copyFrom=None):
    self.unWeightedEnergies=False
    self.startState=None
    super().__init__(copyFrom)

##################### SAT-related functions #####################

def generate_LiarsSAT_problem(CO):
    statements = []
    
    people_i = list(range(1, CO.N + 1))
    random.seed(CO.seed_sat)
    
    # Generate random statements
    for ind in range(CO.M):
        if CO.SATtype == 3: # 3SAT
            i = random.choice(people_i) # The person who makes the statement
            j = random.randint(1, CO.N)    # First person about whom the statement is made
            while j == i:               # People can only make statements about other people
                j = random.randint(1, CO.N)
            k = random.randint(1, CO.N)
            while k == i or k == j:     # Second person about whom the statement is made
                k = random.randint(1, CO.N)

            people_i.remove(i) # Each person gets to make only one statement
            
            # Generate a mix of statements with 2 or 3 literals
            statementType = random.choice([2, 3])
            
            if statementType == 3: # Statement about two people
                liar = random.choice([True, False, "Mix"])
                if liar:
                    statements.append("Person {} says person {} and person {} are both liars".format(i, j, k))
                elif liar == False:
                    statements.append("Person {} says person {} and person {} are both truth-tellers".format(i, j, k))
                elif liar == "Mix":
                    liar2 = random.choice([True, False])
                    if liar2:
                        statements.append("Person {} says person {} is a liar and person {} is a truth-teller".format(i, j, k))
                    else:
                        statements.append("Person {} says person {} is a truth-teller and person {} is a liar".format(i, j, k))
                        
            elif statementType == 2: # Statement about one person
                liar = random.choice([True, False])
                if liar:
                    statements.append("Person {} says person {} is a liar".format(i, j))
                else:
                    statements.append("Person {} says person {} is a truth-teller".format(i, j))
                    
        elif CO.SATtype == 2: # 2SAT
            i = random.choice(people_i)
            
            j = random.randint(1, CO.N)
            while j == i:
                
                j = random.randint(1, CO.N)

            people_i.remove(i) 
            
            
            liar = random.choice([True, False])
            if liar:
                statements.append("Person {} says person {} is a liar".format(i, j))
            else:
                statements.append("Person {} says person {} is a truth-teller".format(i, j))
    
    # Convert statements into clauses
    clauses = []
    clauses_dimacs = []
    ind = 0
    for statement in statements:
        #print("ind: ", ind)
        parts = statement.split()
        
        if CO.SATtype == 3: # 3SAT
            if len(parts) > 8: # Statement about two people
                if len(parts) > 11: # Mixed statement
                    i = int(parts[1])
                    j = int(parts[4])
                    k = int(parts[10])
                    j_liar = parts[7] == "liar"
                    k_liar = parts[13] == "liar"
                    
                    if j_liar and not k_liar:
                        clauses_dimacs.append(f"-{i} -{j} 0")
                        clauses_dimacs.append(f"-{i} {k} 0")
                        clauses_dimacs.append(f"{i} {j} -{k} 0")
                        clauses.append([-i, -j])
                        clauses.append([-i, k])
                        clauses.append([i, j, -k])
                        

                    elif k_liar and not j_liar:
                        clauses_dimacs.append(f"-{i} {j} 0")
                        clauses_dimacs.append(f"-{i} -{k} 0")
                        clauses_dimacs.append(f"{i} -{j} {k} 0")
                        clauses.append([-i, j])
                        clauses.append([-i, -k])
                        clauses.append([i, -j, k])
                    ind += 3
                    
                else: # Both are either liars or truth-tellers statement
                    i = int(parts[1])
                    j = int(parts[4])
                    k = int(parts[7])
                    liars = parts[10] == "liars"

                    if liars: # Both are liars statement
                        clauses_dimacs.append(f"-{i} -{j} 0")
                        clauses_dimacs.append(f"-{i} -{k} 0")
                        clauses_dimacs.append(f"{i} {j} {k} 0")
                        clauses.append([-i, -j])
                        clauses.append([-i, -k])
                        clauses.append([i, j, k])
                    else: # Both are truth-tellers statement
                        clauses_dimacs.append(f"-{i} {j} 0")
                        clauses_dimacs.append(f"-{i} {k} 0")
                        clauses_dimacs.append(f"{i} -{j} -{k} 0")
                        clauses.append([-i, j])
                        clauses.append([-i, k])
                        clauses.append([i, -j, -k])
                    ind += 3
                
            else: # Statement about one person
                i = int(parts[1])
                j = int(parts[4])
                liar = parts[7] == "liar"

                if liar:
                    clauses_dimacs.append(f"-{i} -{j} 0")
                    clauses_dimacs.append(f"{j} {i} 0")
                    clauses.append([-i, -j])
                    clauses.append([j, i])
                else:
                    clauses_dimacs.append(f"-{i} {j} 0")
                    clauses_dimacs.append(f"{i} -{j} 0")
                    clauses.append([-i, j])
                    clauses.append([i, -j])
                ind += 2
            
        elif CO.SATtype == 2: # 2SAT
            i = int(parts[1])
            j = int(parts[4])
            liar = parts[7] == "liar"

            if liar:
                clauses_dimacs.append(f"-{i} -{j} 0")
                clauses_dimacs.append(f"{j} {i} 0")
                clauses.append([-i, -j])
                clauses.append([j, i])
            else:
                clauses_dimacs.append(f"-{i} {j} 0")
                clauses_dimacs.append(f"{i} -{j} 0")
                clauses.append([-i, j])
                clauses.append([i, -j])
            ind += 2
    
    return statements, clauses_dimacs, clauses

def generate_SAT_coloring(CO, adj_mat):
    "N - number of nodes"
    "M - number of colors"
    clauses = []
    clauses_dimacs = []
    all_colored_clauses = []
    one_cpn_clauses = []        # CPN = color per node
    adj_clauses = []
    
    # Generate variables for each node and color combination
    variables = [[j + 1 + i * CO.M for j in range(CO.M)] for i in range(CO.n)]

    # Each node has to be colored
    for i in range(CO.n):
        clause = [variables[i][j] for j in range(CO.M)]
        clauses.append(clause)
        all_colored_clauses.append(clause)
        clauses_dimacs.append(f"{clause[0]} {clause[1]} 0")
    
    # Each node can only have one color
    for i in range(CO.n):
        for j in range(CO.M):
            for k in range(j + 1, CO.M):
                clause = [-variables[i][j], -variables[i][k]]
                clauses.append(clause)
                one_cpn_clauses.append(clause)
                clauses_dimacs.append(f"{clause[0]} {clause[1]} 0")
    
    # Adjacent nodes should have a different color
    for i in range(CO.n):
        for j in range(i + 1, CO.n):
            if adj_mat[i][j] == 1:
                #print("Have borders: ", i, j)
                for k in range(CO.M):
                    for sign in [-1]:
                        clause = [sign*variables[i][k], sign*variables[j][k]]
                        clauses.append(clause)
                        adj_clauses.append(clause)
                        clauses_dimacs.append(f"{clause[0]} {clause[1]} 0")
                    
    
    return clauses, all_colored_clauses, one_cpn_clauses, adj_clauses, clauses_dimacs

def WIcDirect(CO,clauses_dnf, borderL_mat=None,all_colored_clauses=[],one_cpn_clauses=[],adj_clauses=[]):
  W = np.zeros((CO.N,CO.N),order='F')
  I = np.zeros(CO.N)
  c = 0
  if borderL_mat is not None:
    maxL = borderL_mat.max()
  for row in clauses_dnf:
      a, b = row
      indA = abs(a) - 1
      valA = np.sign(a)
      indB = abs(b) - 1
      valB = np.sign(b)

      factor = 0.25

      clause = [-1*a, -1*b]
      if clause in all_colored_clauses:
          factor *= CO.elim_arr[0]

      if clause in one_cpn_clauses:
          factor *= CO.elim_arr[1]

      if clause in adj_clauses:
          factor *= CO.elim_arr[2]
          if CO.weigh_by_border:
              #print("clause: ", clause)
              factor *= borderL_mat[indA//2, indB//2]/maxL

      W[indA,indB] -= valA*valB*factor
      W[indB,indA] -= valA*valB*factor
      I[indA] -= valA*factor
      I[indB] -= valB*factor
      c -= factor

  return W,I,c
 
def convert_2SAT_to_W(CO, allClauses):
    clauses = allClauses[0]
    # Compute the cost function to be minimized
    
    # Convert CNF to DNF
    clauses_dnf = -1 * np.array(clauses)
    
    if CO.SATproblem == "Liars":
        W,I,c = WIcDirect(CO,clauses_dnf)
        
    elif CO.SATproblem == "MapColoring":
        all_colored_clauses, one_cpn_clauses, adj_clauses, borderL_mat=allClauses[1:]
        W,I,c = WIcDirect(CO,clauses_dnf, borderL_mat,all_colored_clauses,one_cpn_clauses,adj_clauses)
    return W, I, c

def write_dimacs_file(PO, CO, clauses_dimacs):
    if CO.SATproblem == "Liars":
        fileName = f"{CO.SATtype}SAT_{CO.SATproblem}_{CO.n}N_{CO.M}M.cnf"
        header = "c Truth-teller and liar problem\n"
        header += "c N={} people, M={} statements, C={} clauses"
    elif CO.SATproblem == "MapColoring":
        fileName = f"{CO.SATtype}SAT_{CO.SATproblem}_{CO.Map}_{CO.n}N_{CO.M}M.cnf"
        header = "c Graph coloring problem applied to maps\n"
        header += "c N={} countries, M={} colors, C={} clauses"
    with open(os.path.join(PO.path,fileName), "w") as file:
        file.write(header.format(CO.n, CO.M, len(clauses_dimacs)))
        file.write("\nc\n")
        file.write("p cnf {} {}".format(CO.N, len(clauses_dimacs)))
        file.write("\n")
        for clause in clauses_dimacs:
            file.write(clause + "\n")

def check_state(state, clauses):
    solution = []
    for ind,num in enumerate(state):
        solution.append((ind+1)*np.sign(num))
    
    sat = np.zeros(len(clauses))
    
    for num in solution:
        #print("num: ", num)
        if np.all(sat): #(all values evaluate to True)
            #print('SAT!')
            break
        else:
            for ind,row in enumerate(clauses):
                #print("ind: ", ind)
                #print("row: ", row)
                if row[0] == num or row[1] == num:
                    sat[ind] = 1
            #print("sat: ", sat)
    
    if np.all(sat): 
        #print('solution satisfies all the clauses')
        return True
    else:
        #print('UNSAT')
        return False
    
############## Functions for Self-Optimization algorithm ##############
    
def Binary_update(state, w, I):
    idx = np.random.randint(len(state))
    oldState = state[idx] # save the value of s_i before updating it
    # print(np.dot(w[idx,:], state) + I[idx])
    if (np.dot(w[idx,:], state) + I[idx] >= 0):
      state[idx] = 1
    else:
      state[idx] = -isBipolar
      
    return idx, oldState
                                                                                             
def learn(eta, w, I, c, wOrig, IOrig, cOrig, steps, N, energies, startState=None, doLearn=False):
    """Run the dynamics with or without learning (Eq. (3) in Weber et al. 2022 IEEE SSCI, 1276–1282), 
    the "regular" way using Binary_update()"""
    
    if startState is None: 
        state = np.random.randint(0,2,N)    # Randomize initial discrete behaviours/states s_i={+-1}
        if isBipolar:
            state = state*2-1
    else:
        state = startState
    
    
    for step in range(steps):
        idx, oldState = Binary_update(state, w, I)

        if(doLearn):
            if(step == 0):
                # create the dw (weight matrix change) only once per reset
                dw = state[:,np.newaxis] * state[np.newaxis,:]
            else:
                # since only one discrete state changes, we need to update only one column and row
                if state[idx] > 0:
                  dw[idx,:] = state
                  dw[:,idx] = state
                else:
                  dw[idx,:] = -isBipolar*state
                  dw[:,idx] = -isBipolar*state
            w += dw/eta # this line is the main bottleneck, the primary memory bandwidth constrain
        
        # Track history (including steps)
        if step==0:
            """Eq. (2), ibid."""
            energies[step] = calcE(wOrig,IOrig,cOrig,state)
        else:
            """Compute the energy from energy change from previous update, Eq. (8), ibid."""
            
            energies[step] = energies[step-1] - (state[idx]-oldState) \
                * (np.dot(state,wOrig[:,idx]) - state[idx]*wOrig[idx,idx] + IOrig[idx])
            if(energies[step] > energies[step-1]):
                print(state[idx], oldState, step, energies[step-1], energies[step])
            if not isBipolar:
                # since state**2==1 always for bipolar, this won't be needed in that case
                energies[step] -= (state[idx]**2-oldState**2) * wOrig[idx,idx]*0.5
    return state

def updateW(eta, w, dw, state, idx, idx2t, t2idx, t2state, t):
    
    w[idx,:] += dw[idx,:] * (t-idx2t[idx])/eta
    for i in range(idx2t[idx]+1,t):
        new_dw = state[idx] * t2state[i]
        if new_dw != dw[idx,t2idx[i]]:
            w[idx,t2idx[i]] += (new_dw - dw[idx,t2idx[i]]) * (t-i) / eta
            dw[idx,t2idx[i]] = new_dw
        
def calcE(W, I, c, state):
    return -0.5 * np.dot(state, np.dot(W, state)) - np.dot(I,state) - c
    
def learnSpeed(eta, w, I, c, wOrig, IOrig, cOrig, steps, N, energies, startState=None, doLearn=False):
    """Run the dynamics with or without learning, with the 'On the fly' calculation of w  (Algorithm 2 in Weber et al. 2022 IEEE SSCI, 1276–1282)"""
    if startState is None: 
        state = np.random.randint(0,2,N)    # Randomize initial discrete behaviours/states s_i={+-1}
        if isBipolar:
            state=state*2-1
    else:
        state = startState
    
    idx2t = np.zeros(N, dtype=int)          # Map between idx and time it was last changed ('ones' in Julia)
    t2idx = np.zeros(steps, dtype=int)      # The idx for which at time t the state was changed ('ones' in Julia)
    t2state = np.zeros(steps, dtype=int)    # history of all states after they were changed
    dw = np.zeros((N,N), dtype=int)
    
    for t in range(steps):
        idx = np.random.randint(len(state))
        oldState = state[idx]
        if doLearn:
            updateW(eta, w, dw, state, idx, idx2t, t2idx, t2state, t)
            idx2t[idx] = t  # at what time idx changed
            t2idx[t] = idx  # what idx that was

        # if (np.dot(w[idx,:],state) + I[idx] >= -isBipolar):
        if ((np.dot(w[idx,:],state) + I[idx]) + 0.5 * (1-isBipolar)*w[idx,idx] >= 0):
            state[idx] = 1
        else:
            state[idx] = -isBipolar
            
        if doLearn:
            t2state[t] = state[idx]         # save the state that was changed at time t
            if t==0:
                dw = state[:,np.newaxis] * state[np.newaxis,:]
            else:
                if state[idx] >= 0:
                    dw[idx,:] = state
                else:
                    dw[idx,:] = -state
                    
        if t==0:
            """Eq. (2), ibid."""
            energies[t] = calcE(wOrig,IOrig,cOrig,state)
        else:
            """Eq. (8), ibid."""
            if state[idx]==oldState:
                energies[t] = energies[t-1]
            else:
                energies[t] = energies[t-1] - (state[idx]-oldState) \
                           * (np.dot(state,wOrig[:,idx]) - state[idx]*wOrig[idx,idx] + IOrig[idx])
                if not isBipolar:
                    # since state**2==1 always for bipolar, this won't be needed in that case
                    energies[t] -= (state[idx]**2-oldState**2) * wOrig[idx,idx]*0.5
                
    if doLearn:
        t = steps
        for idx in range(N):
            updateW(eta, w, dw, state, idx, idx2t, t2idx, t2state, t)
            
    return state

def runReg(CO, w, I, c, wOrig, IOrig, cOrig, energies, startState, doLearn):
  for i in range(CO.resets):
      if speed:
          state = learnSpeed(CO.eta, w, I, c, wOrig, IOrig, cOrig, CO.steps, CO.N, energies[i], startState, doLearn)
      else:
          state = learn(CO.eta, w, I, c, wOrig, IOrig, cOrig, CO.steps, CO.N, energies[i], startState, doLearn)
    
    # # for process progress "visualization" print each 100 steps
    #   try:
    #       if i%(resets//10)==0:
    #           print('\r', i, end = '')
    #   except ZeroDivisionError:
    #       pass
    #   print('')
  return state
  
def beginRun(CO,w, I, c, wOrig, IOrig, cOrig, energies, startState, doLearn):
    
    start = np.array(os.times())
    
    np.random.seed(CO.seed_sim)
    
    if F: # run Fortran routine
        if not isBipolar:
          raise ValueError('Fortran hebbF.runsimple only works for bipolar states')
        state = np.zeros(CO.N,dtype=np.int8,order='F')
        randoms = np.zeros((CO.N+CO.steps,CO.resets),dtype=int,order='F')
        # Prefill random values for Fortran learning in the same order as
        # python learn 
        for r in range(CO.resets):
          randoms[:CO.N,r]=2*np.random.randint(0,2,CO.N)-1
          randoms[CO.N:,r]=np.random.randint(0,CO.N,CO.steps)+1 #Fortran uses 1 base indexing

        hebbF.hebb.runsimple(w, I, c, wOrig, IOrig, cOrig, energies.T, doLearn, CO.alpha, state, randoms) # transpose of energies because Fortran array ordering is reversed

        
        
    else:
        state = runReg(CO, w, I, c, wOrig, IOrig, cOrig, energies, startState, doLearn)
        
    
    duration = np.array(os.times()) - start
    print("""Execution time for N={} (ts={}, resets={}) with L={}:\n""".format(CO.N, CO.steps, CO.resets, doLearn), 
          [round(d, 4) for d in duration])
    if doLearn:
        print("eta =", CO.eta, ", α = " + str(round(1/CO.eta,11)) + "\n")
    sys.stdout.flush()
    return state


def simulate(CO,clauses,startState=None,PO=plotOptions()):
    
    energies = np.zeros((3*CO.resets, CO.steps), dtype=np.float64)

    w,I,c = convert_2SAT_to_W(CO,clauses)
    wWeights = w.copy()

    if CO.unWeightedEnergies:
      tempO = calcOptions(CO)
      tempO.weigh_by_border = False
      tempO.elim_arr = [1,1,1]
      wOrig, IOrig, cOrig = convert_2SAT_to_W(tempO,clauses)
    else:
      wOrig = w.copy()
      IOrig = I.copy()
      cOrig = c
        
    state = beginRun(CO, w, I, c, wOrig, IOrig, cOrig, energies[:CO.resets], startState, False)
    print("isSAT?", check_state(state, clauses[0]), "\n")
    stateLearn = beginRun(CO, w, I, c, wOrig, IOrig, cOrig, energies[CO.resets:CO.resets*2], startState, True)
    print("isSAT?", check_state(stateLearn, clauses[0]), "\n")
    stateEnd = beginRun(CO, w, I, c, wOrig, IOrig, cOrig, energies[CO.resets*2:CO.resets*3], startState, False)
    print("isSAT?", check_state(stateEnd, clauses[0]), "\n")
    
    if PO.dump:
        with open(os.path.join(PO.path,'output_{}_{}_ss{}'.format(CO.N, CO.eta, CO.seed_sim)),'wb') as out:
          for thing in [energies, w, wOrig, I, c, state, stateLearn]:
            pickle.dump(thing, out)

    result = container()
    result.energies = energies
    result.w = w
    result.wOrig = wWeights
    result.I = I
    result.c = c
    result.state = state
    result.stateLearn = stateLearn
    if CO.SATproblem == "MapColoring":
      result.countries_colors = colorsFromState(stateLearn)
    return result
        
def load_data(CO):
    result=container()
    with open(os.path.join(path,'output_{}_{}_ss{}'.format(CO.N, CO.eta, CO.seed_sim)),'rb') as inData:
        result.energies = pickle.load(inData)
        result.w = pickle.load(inData)
        result.wOrig = pickle.load(inData)
        result.I = pickle.load(inData)
        result.c = pickle.load(inData)
        result.state = pickle.load(inData)
        result.stateLearn = pickle.load(inData)
        
    if CO.SATproblem == "MapColoring":
      result.countries_colors = colorsFromState(stateLearn)
    return result

def computeContributions(CO, clauses, state, elim_options):
    tempO = calcOptions(CO)
    for elim_arr in elim_options[:3]:
        tempO.elim_arr = elim_arr
        w,I,c = convert_2SAT_to_W(tempO,clauses)
        E = calcE(w, I, c, -1*state)
        print(elim_arr, E)

def colorsFromState(state):
  return np.dot(state.reshape([int(len(state)/2),2])+1,[2,1])//2

def checkBorders(CO, clauses, state, elim_options):
    countries_colors = colorsFromState(state)
    for b in CO.bordersMap:
      if (countries_colors[b[0]] == countries_colors[b[2]])\
        and (b[0]>b[2])\
        and (countries_colors[b[0]] in [1,2])\
        and (countries_colors[b[2]] in [1,2]):
        print(b, countries_colors[b[0]], countries_colors[b[2]]) 

    computeContributions(CO, clauses, state, elim_options)

    print(state.reshape([int(len(state)/2),2]).sum(1))
    return countries_colors

if F=='compile':
  import importlib,subprocess
  modFile = 'hebbclean.F90'
  stamp = int(os.path.getmtime(modFile))
  module = 'hebbF{}'.format(stamp)
  try:
    hebbF=importlib.import_module(module)
  except ModuleNotFoundError:
    print('Compiling ',modFile)
    try:
      res=subprocess.check_output('f2py --f90flags="-g -fdefault-integer-8 -O3" -m {} -c {}'.format(module,modFile),shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as ex:
      print(ex.output.decode('utf-8'))
      raise ValueError() from None
    hebbF=importlib.import_module(module)
elif F==1:
  try:
    import hebbF
  except ModuleNotFoundError:
    print('hebbF.so compiled Fortran module not found\nTry compiling it with:\nf2py3 --f90flags="-g -fdefault-integer-8 -O3" -m hebbF -c hebbclean.F90')
    raise
