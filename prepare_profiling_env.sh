## This is necessary if you want to get correct detailed timings.
## The PROFILE mode inserts an implicit synchronization after each CUDA kernel
## call. The timing code in the cudaHOG library relies on this!
##
##
## for CUDA Profiling just do
##
## $  source prepare_profiling_env.sh
##
## in your shell -- do not try to run this as a script
## the valid options in cuda_profile_config are the following, choose 4 only
#
# gld_incoherent – Anzahl der nicht zusammengefügten lesenden Zugriffe auf den globalen Speicher, funktioniert nicht auf GT200-basierten Karten.
# gst_incoherent – Anzahl der nicht zusammengefügten schreibenden Zugriffe auf den globalen Speicher, funktioniert nicht auf GT200-basierten Karten.
# gld_coherent – Anzahl der zusammengefügten lesenden Speicherzugriffe.
# gst_coherent – Anzahl der zusammengefügten schreibenden Speicherzugriffe.
# local_load – Anzahl der lesenden Zugriffe auf den lokalen Speicher.
# local_store – Anzahl der schreibenden Zugriffe auf den lokalen Speicher.
# branch – Gesamte Anzahl der Verzweigungen
# divergent_branch – Gesamte Anzahl der Verzweigungen welche zu serialisierter Ausführung führten
# instructions – Befehlszähler
# warp_serialize – Anzahl von Threads in Warps die aufgrund von Adresskonflikten beim Zugfriff auf den gemeinsamen oder den konstanten Speicher.
# cta_launched – Anzahl der ausgeführten Blöcke.
##

export CUDA_PROFILE=1
export CUDA_PROFILE_CONFIG=$HOME/.cuda_profile_config
export CUDA_PROFILE_CSV=1
export CUDA_PROFILE_LOG=cuda_profile.csv

