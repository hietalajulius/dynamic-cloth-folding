import numpy as np
import osc_binding


r = osc_binding.step_controller(np.ones(16), np.ones(16), np.ones(7), np.ones(7), np.ones(7), np.ones(49), np.ones(42), np.ones(7), np.ones(7), np.ones(3), np.ones(3), 10.0, 10.0, 10.0, 10.0, 10.0, 1)
print(r)