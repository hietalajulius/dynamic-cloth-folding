import cma
import numpy as np
es = cma.CMAEvolutionStrategy(12 * [0], 0.5)

def funkk(solutions):
     return np.abs(solutions-np.ones(12)*5).sum()

while not es.stop():
     #print("goo")
     solutions = es.ask()
     es.tell(solutions, [funkk(x) for x in solutions])
     es.disp()
     #es.result_pretty()

es.result_pretty()
cma.plot()  # shortcut for es.logger.plot()