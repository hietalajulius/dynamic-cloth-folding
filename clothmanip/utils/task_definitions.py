

constraints = dict(
                   diagonal=[
                                   dict(origin="S8_8", target="S0_0", distance=0.03, noise_directions=(1, 1, 0)),
                                   dict(origin="S0_0", target="S0_0",
                                        distance=0.03),
                                   dict(origin="S4_4", target="S4_4", distance=0.03), ],


                   sideways=[
                                        dict(origin="S8_8", target="S8_0", distance=0.05, noise_directions=(0, 1, 0)),
                                        dict(origin="S0_8", target="S0_0", distance=0.05, noise_directions=(0, 1, 0)),
                                        dict(origin="S0_4", target="S0_4", distance=0.05),
                                        dict(origin="S8_4", target="S8_4", distance=0.05),
                                        dict(origin="S8_0", target="S8_0", distance=0.05),
                                        dict(origin="S0_0", target="S0_0", distance=0.05)]
                   )
