"""
Libraries of the double random current project.  Nothing in this folder starts a simulation by itself.

    drc_sim              sampler, main run (variance of h, snapshots)            -> results in  drc_runs/
    drc_geometry         clusters, outer boundaries, box-counting dimensions      -> results in  drc_geo_runs/
    test_functions       h(f) for test functions against the GFF                  -> results in  drc_testfun_runs/
    test_function_plots  figure of the test functions
    gff_compare          GFF samples, Green function, comparison figures
    drc_height           height function H of Theorem 2.1 on faces and vertices
    drc_clusters         drawing of the clusters, frames and videos
    drc_sample           one sample of the double current (edges, face signs, h, H, integer values) and its pictures
    presentation_figures figures for the slides, drawn at slide size                 -> saved in    Graficos/

Runs are started from  custom_runs.py  (project folder); the three monitor notebooks only read results.
The result folders and Graficos/ live in the project folder (the parent of this folder).
"""
