# SNMHawkesBeta
The repository implements the sigmoid nonlinear multivariate Hawkes processes in the paper "Efficient Inference of Flexible Interaction in Spiking-neuron Networks". It includes both simulation and inference. 

We recommend to read the tutorial to get familiar with this module. It introduces the key points in the model and illustrates how to perfrom simulation and inference.

For any further details, please refer to the paper. 

# Instructions
Please run the command line below for detailed guide:
```
python exp.py --help 
```
## Examples:

### Example 1: Synthetic data
```
python exp.py -nd 8 -nb 4 -tphi 6 -t 1000 -tt 1000 -b 0.05 -ng 2000 -ngt 2000 -niter 200 -m synthetic
```
### Example 2: Real data
```
python exp.py -nd 25 -nb 1 -tphi 10 -t 300 -tt 300 -b 0.1 -ng 1000 -ngt 1000 -niter 100 -m real
```
