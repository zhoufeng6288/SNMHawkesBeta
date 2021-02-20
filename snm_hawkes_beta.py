import numpy as np
import copy
from scipy.stats import beta
from scipy.special import expit
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import multinomial
from numpy.polynomial import legendre

class SNMHawkesBeta:
	"""
	This class implements sigmoid nonlinear multivariate Hawkes processes with Beta densities as basis functions.
	The main features it provides include simulation and statistical inference. 
	"""
	def __init__(self, number_of_dimensions, number_of_basis):
		"""
		Initialises an instance.

		:type number_of_dimensions: int
		:param number_of_dimensions: number of dimensions (neurons)
		:type number_of_basis: int
		:param number_of_basis: number of basis functions (beta densities)
		"""
		self.number_of_dimensions = number_of_dimensions
		self.number_of_basis = number_of_basis
		self.beta_ab = np.zeros((number_of_basis, 3))
		self.T_phi = 0

		self.lamda_ub = np.zeros(number_of_dimensions)
		self.lamda_ub_estimated = None
		self.base_activation = np.zeros(number_of_dimensions)
		self.base_activation_estimated = None
		self.weight = np.zeros((number_of_dimensions, number_of_dimensions, number_of_basis))
		self.weight_estimated = None

	def set_hawkes_hyperparameters(self, beta_ab, T_phi):
		r"""
		Fix the hyperparameters : parameters a, b, shift and scale for basis functions (Beta densities). 

		:type beta_ab: numpy array
		:param beta_ab: [[a,b,shift],[a,b,shift]...] for basis functions.
		:type T_phi: float
		:param T_phi: the support of influence functions (the scale of basis functions)
		"""
		# Raise ValueError if the given parameters do not have the right shape
		if np.shape(beta_ab) != (self.number_of_basis, 3):
			raise ValueError('given basis functions have incorrect shape')
		if np.shape(T_phi) != ():
			raise ValueError('given scale parameter has incorrect shape')
		self.beta_ab = copy.copy(beta_ab)
		self.T_phi = copy.copy(T_phi)

	def set_hawkes_parameters(self, lamda_ub, base_activation, weight):
		r"""
		Fix the parameters: intensity upperbound, base activation and influence weight. 
		They are used in the simulation.

		:type lamda_ub: 1D numpy array
		:param lamda_ub: :math:`\bar{\lambda}`.
		:type base_activation: 1D numpy array
		:param base_activation: :math:`\mu`.
		:type weight: number_of_dimensions*number_of_dimensions*number_of_basis numpy array
		:param weight: :math:`w_{ijb}`.
		"""
		# Raise ValueError if the given parameters do not have the right shape
		if np.shape(lamda_ub) != (self.number_of_dimensions,):
			raise ValueError('given intensity upperbounds have incorrect shape')
		if np.shape(base_activation) != (self.number_of_dimensions,):
			raise ValueError('given base activations have incorrect shape')
		if np.shape(weight) != (self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis):
			raise ValueError('given weight have incorrect shape')
		self.lamda_ub = copy.copy(lamda_ub)
		self.base_activation = copy.copy(base_activation)
		self.weight = copy.copy(weight)

	def set_hawkes_parameters_estimated(self, lamda_ub_estimated, W_estimated):
		r"""
		Set the estimated intensity upperbound, base activation and influence weight. 
		They are used in the visualization.  

		:type lamda_ub_estimated: 1D numpy array
		:param lamda_ub_estimated: :math:`\hat\bar{\lamda}`.
		:type W_estimated: number_of_dimensions * (number_of_dimensions * number_of_basis + 1) numpy array
		:param W_estimated: `W[:,0]` is the estimated base activation, `W[:,1:]` is the estimated influence weight
		"""
		# Raise ValueError if the given parameters do not have the right shape
		if np.shape(lamda_ub_estimated) != (self.number_of_dimensions,):
			raise ValueError('given estimated intensity upperbounds have incorrect shape')
		if np.shape(W_estimated) != (self.number_of_dimensions, self.number_of_dimensions * self.number_of_basis + 1):
			raise ValueError('given estimated W have incorrect shape')
		self.lamda_ub_estimated = copy.copy(lamda_ub_estimated)
		self.base_activation_estimated = copy.copy(W_estimated[:,0])
		self.weight_estimated = copy.copy(W_estimated[:,1:])

	def intensity(self, t, target_dimension, timestamps_history, estimation = False):
		"""
		Given the historical timestamps, evaluate the conditional intensity at t on the target dimension.
		It is used in the simulation and visualization. If `estimation` is False, the intensity function is using 
		the ground truth parameters; if `estimation` is True, the intensity function is using the estimated parameters. 

		:type t: float
		:param t: the target time
		:type target_dimension: int
		:param target_dimension: the target dimension
		:type timestamps_history: list
		:param timestamps_history: [[t_1,t_2,...,t_N_1],[t_1,t_2,...,t_N_2],...], the historical timestamps before t
		:type estimation: bool
		:param estimation: indicate to use whether the ground-truth or estimated parameters

		:rtype: float
		:return: the conditional intensity at t
		"""
		# Raise ValueError if the given historical timestamps do not have the right shape
		if len(timestamps_history) != self.number_of_dimensions:
			raise ValueError('given historical timestamps have incorrect shape')
		if estimation == False:
			lamda_ub_target_dimension = self.lamda_ub[target_dimension]
			base_activation_target_dimension = self.base_activation[target_dimension]
			weight_target_dimension = self.weight[target_dimension]
		else:
			lamda_ub_target_dimension = self.lamda_ub_estimated[target_dimension]
			base_activation_target_dimension = self.base_activation_estimated[target_dimension]
			weight_target_dimension = self.weight_estimated[target_dimension]
		intensity = 0
		for n in range(self.number_of_dimensions):
			for i in range(len(timestamps_history[n])):
				if timestamps_history[n][i] >= t:
					break
				elif t - timestamps_history[n][i] > self.T_phi: 
					continue
				for b in range(self.number_of_basis):
					intensity += weight_target_dimension[n][b] * beta.pdf(t - timestamps_history[n][i], a = self.beta_ab[b][0], b = self.beta_ab[b][1], loc = self.beta_ab[b][2], scale = self.T_phi)
		return lamda_ub_target_dimension * expit(base_activation_target_dimension + intensity)

	def simulation(self, T):
		r"""
		Simulate a sample path of the sigmoid nonlinear multivariate Hawkes processes with Beta densities as basis functions.

		:type T: float
		:param T: time at which the simulation ends.
		:rtype: list
		:return: the timestamps when events occur on each dimension.
		"""
		t = 0
		points_hawkes = []
		for i in range(self.number_of_dimensions):
			points_hawkes.append([])
		intensity_sup = sum(self.lamda_ub)
		while(t < T):
			r = expon.rvs(scale = 1 / intensity_sup)
			t += r
			sum_intensity = sum(self.intensity(t,m,points_hawkes) for m in range(self.number_of_dimensions))
			assert sum_intensity <= intensity_sup, "intensity exceeds the upper bound"
			D = uniform.rvs(loc = 0,scale = 1)
			if D * intensity_sup <= sum_intensity:
				k = list(multinomial.rvs(1,[self.intensity(t,m,points_hawkes) / sum_intensity for m in range(self.number_of_dimensions)])).index(1)
				points_hawkes[k].append(t)
		if points_hawkes[k][-1] > T:
			del points_hawkes[k][-1]
		return points_hawkes

	'Inference'
	@staticmethod
	def gq_points_weights(a,b,Q):
		r"""
		Generate the Gaussian quadrature nodes and weights for the integral :math:`\int_a^b f(t) dt`

		:type a: float
		:param a: the lower end of the integral
		:type b: float
		:param b: the upper end of the integral
		:type Q: int
		:param Q: the number of Gaussian quadrature nodes (weights)
		:rtype: 1D numpy array, 1D numpy array
		:return: Gaussian quadrature nodes and the corresponding weights
		"""
		p,w = legendre.leggauss(Q)
		c = np.array([0] * Q + [1])
		p_new = (a + b + (b - a) * p) / 2
		w_new = (b - a) / (legendre.legval(p, legendre.legder(c))**2*(1-p**2))
		return p_new,w_new

	def Phi_t(self, t, points_hawkes):
		r"""
		Evaluate \Phi(t)=[1,\Phi_{11}(t),...,\Phi_{MB}(t)] where \Phi_{jb}(t) is the cumulative influence on t
		of the j-th dimensional observation by the b-th basis function

		:type t: float
		:param t: the target time
		:type points_hawkes: list
		:param points_hawkes: the timestamps when events occur on each dimension
		:rtype: 1D numpy array
		:return: \Phi(t)=[1,\Phi_{11}(t),...,\Phi_{MB}(t)]
		"""
		# Raise ValueError if the given timestamps do not have the right shape
		if len(points_hawkes) != self.number_of_dimensions:
			raise ValueError('given timestamps have incorrect shape')
		Phi_t = [1]
		for i in range(self.number_of_dimensions):
			for j in range(self.number_of_basis):
				index = (np.array(points_hawkes[i]) < t) & ((t - np.array(points_hawkes[i])) <= self.T_phi)
				Phi_t.append(sum(beta.pdf(t - np.array(points_hawkes[i])[index], a=self.beta_ab[j][0], b=self.beta_ab[j][1], loc=self.beta_ab[j][2], scale=self.T_phi)))
		return np.array(Phi_t)

	def loglikelyhood(self, W, lamda, Phi_n, Phi_gq, points_hawkes, w_gq):
		r"""
		Evaluate the log-likelihood for the given timestamps
		
		:type W: numpy array
		:param W: the input weight which includes the base activation
		:type lamda: 1D numpy array
		:param lamda: the input intensity upperbound
		:type Phi_n: list of 1D numpy arrays
		:param Phi_n: the cumulative influence \Phi on each observed timestamp
		:type Phi_gq: numpy array (Q, number_of_dimensions * number_of_basis + 1)
		:param Phi_gq: the cumulative influence \Phi on each Gaussian quadrature node
		:type points_hawkes: list
		:param points_hawkes: the timestamps when events occur on each dimension
		:type w_gq: 1D numpy array 
		:param w_gq: Gaussian quadrature weights
		:rtype: float
		:return: the log-likelihood for the given timestamps
		"""
		# Raise ValueError if the given timestamps do not have the right shape
		if len(points_hawkes) != self.number_of_dimensions:
			raise ValueError('given timestamps have incorrect shape')
		if np.shape(Phi_gq) != (len(w_gq), self.number_of_dimensions * self.number_of_basis + 1):
			raise ValueError('the dimension of Phi_gq or w_gq is incorrect')
		for i in range(self.number_of_dimensions):
			if len(Phi_n[i]) != len(points_hawkes[i]):
				raise ValueError('the dimension of Phi_n is incorrect')
		logl = 0
		for i in range(self.number_of_dimensions):
			N_i = len(points_hawkes[i])
			logl += sum(np.log(expit(W[i].dot(Phi_n[i].T))))+np.log(lamda[i])*N_i-(expit(W[i].dot(Phi_gq.T))*lamda[i]).dot(w_gq)
		return logl

	def EM_inference(self, points_hawkes, points_hawkes_test, T, T_test, b, num_gq, num_gq_test, num_iter, initial_W = None): 
		r"""
		EM algorithm which is used to estimate the MAP estimation of parameters: 
		lamda_ub and weight (base_activation is included in the weight). 
		
		:type points_hawkes: list
		:param points_hawkes: the training timestamps
		:type points_hawkes_test: list
		:param points_hawkes_test: the test timestamps
		:type T: float
		:param T: time at which the training timestamps ends
		:type T_test: float
		:param T_test: time at which the test timestamps ends
		:type b: float
		:param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
		:type num_gq: int
		:param num_gq: the number of Gaussian quadrature nodes on [0,T]
		:type num_gq_test: int
		:param num_gq_test: the number of Gaussian quadrature nodes on [0,T_test]
		:type num_iter: int
		:param num_iter: the number of EM iterations
		:type initial_W: numpy array
		:param initial_W: the initial value for W in the EM iterations

		:rtype: numpy array
		:return: the MAP estimation of lamda_ub (lamda) and weight (W), the training (logl) and test log-likelihood (logl_test)
		along EM iterations. 
		"""
		# number of points on each dimension 
		N = np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) 
		N_test = np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)])
		#initial W and lamda
		if initial_W is None:
			W = np.random.uniform(-1,1,size=(self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
		else:
			W = copy.copy(initial_W)
		lamda = N / T
		logl = []
		logl_test = []
		E_beta = np.zeros((self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
		E_w_n = [np.zeros(N[d]) for d in range(self.number_of_dimensions)] 
		p_gq, w_gq = self.gq_points_weights(0,T,num_gq) 
		p_gq_test, w_gq_test = self.gq_points_weights(0,T_test,num_gq_test) 
		Phi_n = [np.zeros((N[d], self.number_of_dimensions*self.number_of_basis+1)) for d in range(self.number_of_dimensions)]
		Phi_n_test = [np.zeros((N_test[d], self.number_of_dimensions*self.number_of_basis+1)) for d in range(self.number_of_dimensions)] 
		Phi_gq = np.zeros((num_gq, self.number_of_dimensions*self.number_of_basis+1)) 
		Phi_gq_test = np.zeros((num_gq_test, self.number_of_dimensions*self.number_of_basis+1))
		for d in range(self.number_of_dimensions):
			for n in range(N[d]):
				Phi_n[d][n] = self.Phi_t(points_hawkes[d][n], points_hawkes)
			for n in range(N_test[d]):
				Phi_n_test[d][n] = self.Phi_t(points_hawkes_test[d][n],points_hawkes_test)
		for m in range(num_gq):
			Phi_gq[m] = self.Phi_t(p_gq[m], points_hawkes)
		for m in range(num_gq_test):
			Phi_gq_test[m] = self.Phi_t(p_gq_test[m], points_hawkes_test)
		H_n = [W[d].dot(Phi_n[d].T) for d in range(self.number_of_dimensions)]
		H_gq = W.dot(Phi_gq.T)
		int_intensity = np.zeros(self.number_of_dimensions)
		
		for ite in range(num_iter):
			for d in range(self.number_of_dimensions):
				# update H_n_d,E_w_n_d; H_gq_d,int_intensity_d; E_beta_d
				H_n[d] = W[d].dot(Phi_n[d].T)
				E_w_n[d] = 1/2/H_n[d]*np.tanh(H_n[d]/2)
				H_gq[d] = W[d].dot(Phi_gq.T)
				int_intensity[d] = lamda[d]*expit(-H_gq[d]).dot(w_gq)
				E_beta[d] = b / W[d]
				
				# update lamda_d
				lamda[d]=(int_intensity[d]+N[d])/T

				# update W_d
				int_A=np.zeros((self.number_of_dimensions*self.number_of_basis+1,self.number_of_dimensions*self.number_of_basis+1))
				for n in range(N[d]):
					int_A+=E_w_n[d][n]*np.outer(Phi_n[d][n],Phi_n[d][n])
				for m in range(num_gq):
					int_A+=w_gq[m]*(lamda[d]/2/H_gq[d][m]*np.tanh(H_gq[d][m]/2)*expit(-H_gq[d][m])*np.outer(Phi_gq[m],Phi_gq[m]))
				int_B=np.zeros(self.number_of_dimensions*self.number_of_basis+1)
				for n in range(N[d]):
					int_B+=0.5*Phi_n[d][n]
				for m in range(num_gq):
					int_B+=-w_gq[m]/2*(lamda[d]*expit(-H_gq[d][m])*Phi_gq[m])
				W[d]=np.linalg.inv(int_A+np.diag(E_beta[d]/b/b)).dot(int_B)
				# for numerical stability, we truncate W if it is too close to 0
				W[d][np.abs(W[d])<1e-200]=1e-200*np.sign(W[d][np.abs(W[d])<1e-200])
			# compute the loglikelihood
			logl.append(self.loglikelyhood(W,lamda,Phi_n,Phi_gq,points_hawkes,w_gq))
			logl_test.append(self.loglikelyhood(W,lamda,Phi_n_test,Phi_gq_test,points_hawkes_test,w_gq_test))
		return lamda, W, logl, logl_test

	'tool functions'
	def influence_function_estimated(self, i, j, t, gt = False): 
		r"""
		Evaluate the influence function based on the basis functions and the influence weight W.
		It is used to visualize the influence functions. If gt = False, it is using the estimated parameters;
		if gt = True, it is using the ground truth parameters. 
		
		:type i: int
		:param i: the target dimension. \phi_{ij}(t)
		:type j: int
		:param j: the source dimension. \phi_{ij}(t)
		:type t: float
		:param t: the target time. \phi_{ij}(t)
		:type gt: bool
		:param gt: indicate to use whether the ground-truth or estimated parameters

		:rtype: float
		:return: the influence function \phi_{ij}(t)
		"""
		if gt == False:
			W_phi = self.weight_estimated.reshape(self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis)
		else:
			W_phi = self.weight.reshape(self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis)
		phi_t = np.array([beta.pdf(t, a = self.beta_ab[i][0], b = self.beta_ab[i][1], loc = self.beta_ab[i][2], scale = self.T_phi) for i in range(self.number_of_basis)])
		return W_phi[i][j].dot(phi_t)

	def heat_map(self, gt = False):
		r"""
		Evaluate the heatmap value based on the weight of the instance. 
		(It is assumed that the integral of all basis functions is 1). If gt = False, it is using the estimated parameters;
		if gt = True, it is using the ground truth parameters. 

		:type gt: bool
		:param gt: indicate to use whether the ground-truth or estimated parameters

		:rtype: numpy array
		:return: the estimated heatmap value (self.number_of_dimensions * self.number_of_dimensions)
		"""
		phi_heat=np.zeros((self.number_of_dimensions,self.number_of_dimensions))
		if gt == False:
			for i in range(self.number_of_dimensions):
				phi_heat[:,i]=np.sum(np.abs(self.weight_estimated[:,self.number_of_basis*i:self.number_of_basis*(i+1)]),axis=1)
		else:
			for i in range(self.number_of_dimensions):
				phi_heat[:,i]=np.sum(np.abs(self.weight[:,self.number_of_basis*i:self.number_of_basis*(i+1)]),axis=1)
		return phi_heat

