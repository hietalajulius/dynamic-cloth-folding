

constraints = dict(
                    diagonal = lambda min, mid, max, dist: [
                                   dict(origin=f"S{max}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(1, 1, 0)),
                                   dict(origin=f"S{min}_{min}", target=f"S{min}_{min}",distance=dist),
                                   dict(origin=f"S{mid}_{mid}", target=f"S{mid}_{mid}", distance=dist), ],


                    sideways = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{min}", target=f"S{max}_{min}", distance=dist),
                                        dict(origin=f"S{min}_{min}", target=f"S{min}_{min}", distance=dist)],

                    sideways_two_corners = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0))],

                    sideways_one_corner = lambda min, mid, max, dist: [
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0))],

                    sideways_two_corners_mid = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=dist)],
                   )
