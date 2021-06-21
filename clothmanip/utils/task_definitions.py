

constraints = dict(
                   diagonal = lambda min, mid, max: [
                                   dict(origin=f"S{max}_{max}", target=f"S{min}_{min}", distance=0.03, noise_directions=(1, 1, 0)),
                                   dict(origin=f"S{min}_{min}", target=f"S{min}_{min}",distance=0.03),
                                   dict(origin=f"S{mid}_{mid}", target=f"S{mid}_{mid}", distance=0.03), ],


                   sideways_3cm = lambda min, mid, max: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=0.03, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=0.03, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=0.03),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=0.03),
                                        dict(origin=f"S{max}_{min}", target=f"S{max}_{min}", distance=0.03),
                                        dict(origin=f"S{min}_{min}", target=f"S{min}_{min}", distance=0.03)],

                   sideways_5cm = lambda min, mid, max: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=0.05, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=0.05, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=0.05),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=0.05),
                                        dict(origin=f"S{max}_{min}", target=f"S{max}_{min}", distance=0.05),
                                        dict(origin=f"S{min}_{min}", target=f"S{min}_{min}", distance=0.05)]
                   )
