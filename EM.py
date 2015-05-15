from __future__ import division

import pandas as pd 
import numpy as np
import math

############### functions for EM algorithm ##################

class EM():

	def __init__(self,feature_data,features,N,K):

		"""
		feature_data: contains observations indexed by a 'key' column, rest of columns are features
		features: the list of features one wants to use in estimation
		K: number of latent types to estimate
		"""

		self.N = N
		self.K = K
		self.features = [f for feature in features for f in feature]
		self.F = len(self.features)
		data_index = [i for j,f in enumerate(features) for i in [j]*len(f)]

		# counts stored as lists because different features have different number of categories
		
		self.feature_counts = []
		self.M_f = np.empty(self.F,dtype=np.int)
		self.observations = np.empty((self.N,self.F),dtype=np.int)

		for i,f in enumerate(self.features):

			self.feature_counts.append(pd.crosstab(feature_data[data_index[i]]['key'],feature_data[data_index[i]][f]).values)
			self.M_f[i] = self.feature_counts[i].shape[1]
			self.observations[:,i] = self.feature_counts[i].sum(axis=1)
			
		# seed parameters

		self.rho = np.full(self.K,1/self.K) # equal probability of all types

		self.mu = []
		for M_f in self.M_f:
			self.mu.append(np.random.dirichlet(M_f*[1],self.K)) # uniform probability


	def set_seed(self,rho_seed,mu_seed):

		"""
		set seeds manually (should add dimensionality check)
		"""

		self.rho = rho_seed
		self.mu = mu_seed


	def feature_count_prob(self,f,i,k):

		"""
		For feature f, compute the log probability that document i with type k generates the observed counts.
		Called during E step to compute posterior distribution over types.
		"""

		temp = np.zeros(self.M_f[f])
		for m in range(self.M_f[f]):
			temp[m] = self.mu[f][k,m] ** self.feature_counts[f][i,m]
		prod = np.prod(temp)

		return prod


	def E_step(self):

		"""
		compute type probabilities given current parameter estimates.
		"""

		# on the first iteration of estimation, attribute is set
		if not hasattr(self, 'type_prob'): 
			self.type_prob = np.empty((self.N,self.K))

		temp_probs = np.empty((self.N,self.K))

		for i in range(self.N):
			for k in range(self.K):

				temp = np.zeros(self.F)
				for f in range(self.F): 
					temp[f] = self.feature_count_prob(f,i,k)

				temp_probs[i,k] = self.rho[k] * np.prod(temp)

		all_probs = temp_probs.sum(axis=1)[:, np.newaxis]
		self.type_prob = temp_probs / all_probs

		return np.log(all_probs).sum()


	def E_step_robust(self):

		"""
		compute type probabilities given current parameter estimates using log-scale.
		"""

		# on the first iteration of estimation, attribute is set
		if not hasattr(self, 'type_prob'): 
			self.type_prob = np.empty((self.N,self.K))

		temp_probs = np.empty((self.N,self.K))

		for i in range(self.N):
			for k in range(self.K):

				temp = np.zeros(self.F)
				for f in range(self.F): 
					temp[f] = np.dot(self.feature_counts[f][i,:],np.log(self.mu[f][k,:]))

				temp_probs[i,k] = np.log(self.rho[k]) + np.sum(temp)

		temp_probsZ = temp_probs - np.max(temp_probs,axis=1)[:, np.newaxis]
		self.type_prob = np.exp(temp_probsZ) / np.exp(temp_probsZ).sum(axis=1)[:, np.newaxis]

		return np.log(np.exp(temp_probsZ).sum(axis=1)).sum() + np.max(temp_probs,axis=1).sum()


	def M_step(self):

		"""
		generate new parameter estimates given updated type distribution
		"""

		for k in range(self.K): self.rho[k] = self.type_prob[:,k].sum() / self.N

		for f in range(self.F):

			for k in range(self.K):
				for m in range(self.M_f[f]):
					temp_prob = np.dot(self.type_prob[:,k],self.feature_counts[f][:,m])
					if temp_prob < 1e-99: temp_prob = temp_prob + 1e-99
					self.mu[f][k,m] = temp_prob / np.dot(self.type_prob[:,k],self.observations[:,f])


	def estimate(self, maxiter = 250, convergence = 1e-7):

		"""
		run EM algorithm until convergence, or until maxiter reached
		"""

		self.loglik = np.zeros(maxiter)

		iter = 0

		while iter < maxiter:

			self.loglik[iter] = self.E_step_robust()
			if np.isnan(self.loglik[iter]): 
				print "undefined log-likelihood"
				break
			self.M_step()

			if self.loglik[iter] - self.loglik[iter - 1] < 0 and iter > 0: 
				print("log-likelihood decreased by %f at iteration %d" % (self.loglik[iter] - self.loglik[iter - 1],iter))
			elif self.loglik[iter] - self.loglik[iter - 1] < convergence and iter > 0: 
				print ("convergence at iteration %d, loglik = %f" % (iter,self.loglik[iter]))
				self.loglik = self.loglik[self.loglik < 0]
				break

			iter += 1


	def feature_rank(self,type1,type2):

		def hellinger(x,y):

			""" compute the hellinger distance between topic count arrays """

			cons = 1 / np.sqrt(2)
			diff = np.sqrt(x) - np.sqrt(y)

			return cons * np.sqrt(np.power(diff,2).sum())

		distance = np.array([hellinger(self.mu[f][type1,:],self.mu[f][type2,:]) for f in range(self.F)])
		return [self.features[f] for f in distance.argsort()[::-1]], np.sort(distance)[::-1]


	def feature_rank2(self,type1,type2):

		distance = np.empty(self.F)

		for f in range(self.F):

			diff = self.mu[f][type1,:] - self.mu[f][type2,:]
			diff = diff * np.arange(self.M_f[f])
			distance[f] = diff.sum()

		feature_rank = [self.features[f] for f in distance.argsort()[::-1]]

		return feature_rank, np.sort(distance)[::-1]
