# # For toy
# python exp.py -nd 2 -nb 4 -tphi 6 -t 200 -tt 200 -b 0.05 -ng 1000 -ngt 1000 -niter 100 -m toy

# For synthetic
python exp.py -nd 8 -nb 4 -tphi 6 -t 1000 -tt 1000 -b 0.05 -ng 2000 -ngt 2000 -niter 200 -m synthetic

# # For real
# python exp.py -nd 25 -nb 1 -tphi 10 -t 300 -tt 300 -b 0.1 -ng 1000 -ngt 1000 -niter 100 -m real