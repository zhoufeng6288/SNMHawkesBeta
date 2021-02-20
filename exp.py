import numpy as np
import pandas as pd 
from snm_hawkes_beta import SNMHawkesBeta
from scipy.stats import beta 
import argparse

def snm_hawkes_run(num_of_dims: int,
                   num_of_basis: int,
                   T_phi: int,
                   T: int,
                   T_test: int,
                   b: float,
                   num_gq: int,
                   num_gq_test: int,
                   num_iter: int,
                   run_mode: str):


    snm_model = SNMHawkesBeta(num_of_dims,num_of_basis)
    

    if run_mode == "toy":
        beta_ab=np.array([[50,50,-2],[50,50,-1],[50,50,0],[50,50,1]])
        lamda_ub=np.array([5,5])
        base_activation=np.array([0,0])
        weight=np.array([[[1,0,0,0],[0,-0.5,0,0]],[[0,0,0,-0.5],[0,0,1,0]]])

        snm_model.set_hawkes_parameters(lamda_ub,base_activation,weight)
        points_hawkes = snm_model.simulation(T)
        points_hawkes_test = snm_model.simulation(T)


        

    elif run_mode == "synthetic":
        df=pd.read_csv('./synthetic_data.csv',index_col=0)
        points_hawkes=[]
        for i in range(8):
            points_hawkes.append(list(df.iloc[i].values[~np.isnan(df.iloc[i].values)]))
        df=pd.read_csv('./synthetic_data_test.csv',index_col=0)
        points_hawkes_test=[]
        for i in range(8):
            points_hawkes_test.append(list(df.iloc[i].values[~np.isnan(df.iloc[i].values)]))
        beta_ab=np.array([[50,50,-2],[50,50,-1],[50,50,0],[50,50,1]])

    elif run_mode == 'real':
        beta_ab=np.array([[50,50,-5]])
        points_area17=[]
        # read data from spk file
        file=['t00','t01','t02','t03','t04','t05','t06','t08','t09','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t22','t23','t24','t25','t26','t27']
        for f in file:
            with open('./spike_data_area17/'+f+'.spk', mode='rb') as file: 
                points_area17.append(list(np.fromfile(file, dtype=np.int64)/1e5))  # /1e5 to convert to unit of 100ms

        points_hawkes=[]
        points_hawkes_test=[]
        for i in range(len(points_area17)):
            temp=np.array(points_area17[i])
            points_hawkes.append(list(temp[temp<T]))
            points_hawkes_test.append(list(temp[(temp>=T) & (temp<T+T_test)]-T))

    else: 
        print("Type not supported!")

    snm_model.set_hawkes_hyperparameters(beta_ab,T_phi)
    lamda_ub_estimated,W_estimated,logL,logL_test = snm_model.EM_inference(points_hawkes, points_hawkes_test,\
                                            T, T_test, b, num_gq, num_gq_test, num_iter)

    snm_model.set_hawkes_parameters_estimated(lamda_ub_estimated, W_estimated)

        
    
    print("Estimated lambda ub")
    print(snm_model.lamda_ub_estimated)
    print("="*56)
    print("Estimated base activation")
    print(snm_model.base_activation_estimated)
    print("="*56)
    print("Estimated weight")
    print(snm_model.weight_estimated)
                                                               



if __name__ == "__main__":

    # num_of_dims = 2
    # num_of_basis = 4
    # beta_ab=np.array([[50,50,-2],[50,50,-1],[50,50,0],[50,50,1]])
    # T_phi=6
    # lamda_ub=np.array([5,5])
    # base_activation=np.array([0,0])
    # weight=np.array([[[1,0,0,0],[0,-0.5,0,0]],[[0,0,0,-0.5],[0,0,1,0]]])
    # T = 200
    # T_test = 200
    # b = 0.05
    # num_gq = 1000
    # num_gq_test = 1000
    # num_iter = 100

    # snm_hawkes_run(num_of_dims,num_of_basis,beta_ab,T_phi,lamda_ub,base_activation,weight,T,T_test,b,num_gq,num_gq_test,num_iter)



    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--num_of_dims",
                        dest="num_of_dims",
                        type=int,
                        required=True,
                        help="Number of dimensions."
                             "For example 2")
    parser.add_argument("-nb", "--num_of_basis",
                        dest="num_of_basis",
                        type=int,
                        required=True,
                        help="Number of basis functions."
                             "4")
    parser.add_argument("-tphi", "--T_phi",
                        dest="T_phi",
                        type=int,
                        required=True,
                        help=" ")
    parser.add_argument("-t", "--T",
                        dest="T",
                        type=int,
                        required=True,
                        help=" ")
    parser.add_argument("-tt", "--T_test",
                        dest="T_test",
                        type=int,
                        required=True,
                        help=" ")
    parser.add_argument("-b", "--b",
                        dest="b",
                        required=True,
                        type=float,
                        help=" ")
    parser.add_argument("-ng", "--num_gq",
                        dest="num_gq",
                        required=True,
                        type=int,
                        help=" ")
    parser.add_argument("-ngt", "--num_gq_test",
                        dest="num_gq_test",
                        required=True,
                        type=int,
                        help=" ")
    parser.add_argument("-niter", "--num_iter",
                        dest="num_iter",
                        required=True,
                        type=int,
                        help=" ")
    parser.add_argument("-m", "--run_mode",
                        dest="run_mode",
                        required=True,
                        type=str,
                        help=" ")
    args = parser.parse_args()
 
    snm_hawkes_run(args.num_of_dims,args.num_of_basis,args.T_phi,args.T,args.T_test,args.b,args.num_gq,args.num_gq_test,args.num_iter,args.run_mode)
    print("Done!")



