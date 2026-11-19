# custom_runs.py  --  the only file that starts simulations.
#
# How to use it
#   1. Leave exactly ONE block below without the leading "# " (the run you want), keep the others commented.
#   2. Anaconda Prompt, in this folder:      python custom_runs.py
#   3. Watch it from the monitor notebooks (they only read files, they never start a run):
#          monitor_V2.ipynb            main run            (drc_runs/)
#          monitor_geometry.ipynb      geometry run, H, clusters, video   (drc_geo_runs/, snapshots of drc_runs/)
#          monitor_distribution.ipynb  h against the GFF   (drc_testfun_runs/, drc_runs/)
#
# Every run is checkpointed.  Starting a block again continues its chains; a block that is already finished
# returns in a few seconds and changes nothing, so re-running an old block is harmless.
# "samples" is per chain: total = chains x samples.  A larger "samples" on the same chains extends them
# (no new burn-in).  The standard CONFIG of each run lives in its library (custom_packages/<library>.py).

from custom_packages import drc_sim, drc_geometry, test_functions

WORKERS = 6          # processes; 6 = all cores of the i5-8400, 5 keeps the computer responsive

if __name__ == "__main__":

    # ======================================================================================================
    # 1. MAIN RUN  (drc_sim  ->  drc_runs/)      variance of h at the centre, N_odd, largest cluster, snapshots
    # ======================================================================================================

    # ---- 1a. standard run: drc_sim.CONFIG                                                      [DONE]
    #      L = 32, 64, 128, 256: 24 x 42,000 = 1,008,000 | L = 512: 12 x 16,000 = 192,000 | L = 1024: 12 x 4,000 = 48,000
    # drc_sim.run_all(drc_sim.CONFIG, workers=WORKERS)
    # drc_sim.summarize()

    # ---- 1b. extension of L = 512 and L = 1024                                                 [DONE]
    #      L = 1024: 48,000 -> 250,008 (12 x 20,834) | L = 512: 192,000 -> 1,000,008 (12 x 83,334)
    # drc_sim.run_all({1024: dict(chains=12, samples=20834, burn_in=500),
    #                  512:  dict(chains=12, samples=83334, burn_in=500)}, workers=WORKERS)
    # drc_sim.summarize()

    # ======================================================================================================
    # 2. GEOMETRY RUN  (drc_geometry  ->  drc_geo_runs/)   box-counting dimensions, sine modes, block kurtosis
    # ======================================================================================================

    # ---- 2a. standard run: drc_geometry.CONFIG                                                 [DONE]
    #      L = 64, 128: 12 x 32,000 = 384,000 | L = 256: 12 x 16,000 = 192,000 | L = 512: 12 x 8,000 = 96,000
    #      L = 1024: 12 x 4,000 = 48,000      (geometry measured on every 10th sample)
    # drc_geometry.run_all(drc_geometry.CONFIG, workers=WORKERS)
    # drc_geometry.summarize()

    # ======================================================================================================
    # 3. TEST FUNCTIONS  (test_functions  ->  drc_testfun_runs/, run name "main")      h(f) against the GFF
    # ======================================================================================================

    # ---- 3a. standard run: test_functions.CONFIG                                               [DONE]
    #      L = 256, 512, 1024: 12 x 8,334 = 100,008 each          (took 3.50 h wall-clock)
    # test_functions.run_all(test_functions.CONFIG, workers=WORKERS)

    # ---- 3b. extension to 250,008 samples per L (12 x 20,834)                                  [DONE]
    #      about 5.25 h wall-clock more (L = 256: 12 min, L = 512: 54 min, L = 1024: 4.2 h)
    #test_functions.run_all({256:  dict(chains=12, samples=20834, burn_in=500),
    #                        512:  dict(chains=12, samples=20834, burn_in=500),
    #                        1024: dict(chains=12, samples=20834, burn_in=500)}, workers=WORKERS)

    # ======================================================================================================
    # 4. TIMING ON THIS MACHINE  (about a minute each, nothing is saved)
    # ======================================================================================================
    # drc_sim.benchmark()
    # drc_geometry.benchmark()
