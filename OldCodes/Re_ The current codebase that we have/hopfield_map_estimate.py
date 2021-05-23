import numpy as np
import random
import hdnet.hopfield as hdn
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
import scipy.io as spio
import math
import scipy.special as sps
import scipy.optimize as spo
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from collections import deque
from sklearn.metrics import confusion_matrix


# TODO make data_type auto select, so that a folder can have multiple data types
# TODO print units for StimTimes and Ts when loading data

class ModifiedHopfieldNet:
	"""
		Argumentss:
		N = number of nodes to build Hopfield network with
		in_directory = the relative path to a folder containing the raw .mat files
		out_directory = the relative path to a folder that you want to store the python data. If not used, defaults to in_directory
		splits: the number of times to split the data to train on each portion (slices sequentially
			--> default: 3 splits = beginning, middle, end thirds of experiment)
		train_percent = the percentage of each chunk of data (the number of chunks as defined by splits) that will be
			used to train a Hopfield network --> default: 0.66
		num_nets: The number of replications to use when training Hopfield networks --> default: 5
		exp_type = 'map' to conduct a MAP estimate analysis, J to analyze the changing connectivity matrices through time.
			--> default: 'J'
	"""

	def __init__(self, in_directory, out_directory=None, splits=3, train_percent=0.66, num_nets=50, exp_type='J',
	             data_type='sim', dt=8000, n_jobs=1, N=None, num_bs=1, threshold=250, stim_shift=0):
		self.in_directory = in_directory
		self.type = exp_type
		if out_directory is None:
			self.out_directory = self.in_directory
		else:
			self.out_directory = out_directory
		if N is None:
			self.N = self.get_N()
		else:
			self.N = N
		self.splits = splits
		self.experiments = []
		self.train_percent = train_percent
		self.num_nets = num_nets
		self.networks = []
		self.data_type = data_type
		self.dt = dt
		self.n_jobs = n_jobs
		self.num_bs = num_bs
		self.threshold = threshold
		self.bs_inds = []
		self.filenames = []
		self.bs_inds = []
		self.thetas = []
		self.Js = []
		self.stim_shift = stim_shift
		if (self.n_jobs == 1):
			self.load_and_save_data(dt=self.dt)
		if (self.n_jobs > 1) & (self.type == 'J'):
			files = []
			for file in os.listdir(self.in_directory):
				filename = file[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
				if filename not in os.listdir(self.out_directory):
					files.append(file)
			p = Pool(self.n_jobs)
			p.map(self.run_multiprocessing_J, files)
			p.close()
			p.join()

	def get_N(self):
		dats = self.get_dats()
		try:
			Cs = dats[0][0]['Cs']
		except:
			Cs = dats[0][0]['data']['Cs']
			Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
		N = np.max(Cs)
		if self.type == 'map' or self.type == 'MI':
			N += 1
		return N

	def load_and_save_data(self, **kwargs):
		input_files = np.asarray(os.listdir(self.in_directory))
		print("The following files are in your input directory: \n")
		for k, file in enumerate(input_files):
			print(f'File {k}: {file}')
		requested_files = input('Please enter the index of files your would like to analyze separated by a comma. \n '
		                        'For example, if you wanted the first and third file, type "0, 2" \n '
		                        'If you would like to use all listed files, just press Enter')
		if requested_files == '':
			requested_files = input_files
		else:
			requested_files = requested_files.split(',')
			for k, file in enumerate(requested_files):
				requested_files[k] = int(file)
			requested_files = input_files[requested_files]
		for file in requested_files:
			filename = file[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
			if filename not in os.listdir(self.out_directory):
				print(f'---------------- The file {filename} was not found in the out directory  -----------------')
				print(f'---------------------- Importing .mat file: {file} instead ----------------------')
				dat = spio.loadmat(os.path.join(self.in_directory, file))
				ys = self.binaryVecs(dat, **kwargs)
				self.experiments.append(ys)
				y_sparse = csr_matrix(ys, dtype='uint8')
				save_npz(os.path.join(self.out_directory, filename), y_sparse)
			else:
				print(f'------------------ Loading file: {filename} from out directory --------------------')
				ys = load_npz(os.path.join(self.out_directory, filename)).toarray()
				self.experiments.append(ys)

	def build_and_train_networks(self):
		all_accs = []
		for i, memories in enumerate(self.experiments):
			accs = []
			print(f'---------------------- Conducting experiment: {i} ----------------------')
			memories_chunked = self.chunked(memories, self.splits)
			experiment_nets = []
			for j, memory_chunk in enumerate(memories_chunked):
				avg_acc = []
				chunked_nets = []
				for _ in tqdm(range(self.num_nets)):
					hop = hdn.HopfieldNetMPF(N=self.N)
					rand_memories = np.array(
						[random.choice(memory_chunk) for _ in range(round(len(memory_chunk) * self.train_percent))])
					hop.store_patterns_using_mpf(rand_memories + 0.0)
					if self.type == 'map':
						avg_acc.append(self.get_accuracy(memory_chunk, hop, precision_recall=False))
					chunked_nets.append(hop)
				experiment_nets.append(chunked_nets)
				accs.append(avg_acc)
				if self.type == 'map':
					print(f'Experiment: {i} // Chunk: {j} // Avg Accuracy: {round(np.mean(avg_acc), 3)} +/- '
					      f'{round(np.std(avg_acc), 3)}')
				else:
					print(f'Experiment: {i} // Chunk: {j}')
			all_accs.append(accs)
			print(f'---------------------- Finished experiment: {i} ----------------------')
			self.networks.append(experiment_nets)
		return all_accs

	@staticmethod
	def explicitMLE(means, corrs):
		n = len(means)
		if n > 20:
			raise ValueError('Cannot peform fitting when N>20. Reduce N and try again!')

		#
		def logL(x):
			J = np.reshape(x[:n ** 2], [n, n])
			h = x[n ** 2:]
			# get first term
			Z = 0
			for i in range(np.power(2, n)):
				x = np.asarray(list([int(j) for j in np.binary_repr(i, width=n)]))
				E = -np.inner(h, x) + np.inner(x, np.dot(J, x))
				Z += np.exp(E)
			# combine into logL
			logL = -np.sum(means * h) + np.sum(corrs * J) - np.log(Z)
			return logL

		# 1. Do MLE fit
		# For now, assume contrastive divergence unnecessary
		# record log likelihood and actual J's and h's
		def jac_MLE(x):
			J = np.reshape(x[:n ** 2], [n, n])
			h = x[n ** 2:]
			#
			moments_model = np.zeros(n ** 2 + n)
			Z = 0
			for i in range(np.power(2, n)):
				x = np.asarray(list([int(j) for j in np.binary_repr(i, width=n)]))
				E = -np.inner(h, x) + np.inner(x, np.dot(J, x))
				Z += np.exp(E)
				moments_model[:n ** 2] += np.exp(E) * np.reshape(np.outer(x, x), [n ** 2])
				moments_model[n ** 2:] += -np.exp(E) * x
			moments_model /= Z
			moments_data = np.hstack([np.reshape(corrs, n ** 2), -means])
			return moments_data - moments_model

		foo_MLE = spo.minimize(lambda x: -logL(x), x0=np.random.uniform(size=n ** 2 + n), jac=lambda x: -jac_MLE(x))
		logL_MLE = -foo_MLE.fun;
		J_guess = foo_MLE.x[:n ** 2];
		h_guess = foo_MLE.x[n ** 2:]
		return h_guess, np.reshape(J_guess, [n, n])

	def contrastiveDivergence(self, means, corrs, alpha=0.1, thresh=0.05):
		n = len(means)
		# choose initial J and theta
		J_guess = np.random.uniform(size=[n, n])
		theta_guess = np.random.uniform(size=n)

		grad_norm = thresh * 2
		while grad_norm > thresh:
			# do gradient ascent
			# get better estimate of gradient
			mean_model = np.zeros(n)
			corr_model = np.zeros([n, n])
			for k in range(10):
				foo = self.monte_carlo(J_guess, theta_guess, n, k=0.1)
				mean_model += foo[0]
				corr_model += foo[1]
			mean_model /= 10
			corr_model /= 10
			# gradient ascent
			grad_norm = np.sum(np.abs(means - mean_model) / n) + np.sum(np.abs(corrs - corr_model) / n ** 2)
			theta_guess += alpha * (means - mean_model)
			J_guess += alpha * (corrs - corr_model)
		return theta_guess, J_guess

	@staticmethod
	def monte_carlo(J, theta, n, k=0.1):
		# implement MonteCarlo to evaluate gradient and do gradient ascent
		# do this how many times?

		xold = np.asarray([int(i) for i in np.binary_repr(np.random.randint(0, np.power(2, n)), n)])
		Eold = np.inner(theta, xold) + np.inner(xold, np.dot(J, xold))
		# collect new samples and calculate means and correlations during.
		burn_in = 5000
		tot_time = int(30000)
		pm = np.exp(-k * np.arange(1, n))
		pm /= np.sum(pm)
		mean = np.zeros(n)
		corr = np.zeros([n, n])
		for t in range(burn_in + tot_time):
			# generate candidate
			# we'll look at all configurations that flip one bit, could modify this.
			m = np.random.choice(n - 1, size=1, p=pm) + 1
			# m gives the number of bits we flip
			foo = np.random.choice(n, size=m, replace=False)
			xnew = np.zeros(n)
			for i in range(n):
				if i in foo:
					xnew[i] = 1 - xold[i]
				else:
					xnew[i] = xold[i]
			# calculate the acceptance ratio
			Enew = np.inner(theta, xnew) + np.inner(xnew, np.dot(J, xnew))
			dE = Enew - Eold
			acceptance_ratio = np.exp(-dE)
			# accept or reject
			u = np.random.uniform()
			if u < acceptance_ratio:
				Eold = Enew
				if t > burn_in - 1:
					mean += xnew
					corr += np.outer(xnew, xnew)
				xold = xnew
			else:
				if t > burn_in - 1:
					mean += xold
					corr += np.outer(xold, xold)
		mean /= tot_time
		corr /= tot_time
		return mean, corr

	def get_means_and_corrs(self, binaryVec):
		means = np.mean(binaryVec, axis=0)
		corrs = np.dot(binaryVec, binaryVec.T)
		return means, corrs

	def run_explicitMLE(self):
		for i, memories in enumerate(self.experiments):
			print(f'---------------------- Conducting experiment: {i} ----------------------')
			memories_chunked = self.chunked(memories, self.splits)
			for j, memory_chunk in enumerate(memories_chunked):
				for _ in tqdm(range(self.num_nets)):
					rand_memories = np.array(
						[random.choice(memory_chunk) for _ in range(round(len(memory_chunk) * self.train_percent))])
					means, corrs = self.get_means_and_corrs(rand_memories)
					theta, J = self.explicitMLE(means, corrs)
					self.thetas.append(theta)
					self.Js.append(J)
			print(f'---------------------- Finished experiment: {i} ----------------------')
		return self.thetas, self.Js

	def run_bootstrap_Js(self, bootstrap_sizes):
		for size in bootstrap_sizes:
			bs_nets = []
			for bootstrap_num in range(self.num_bs):
				inds = np.random.choice(range(self.N), size, replace=False)
				self.bs_inds.append(inds)
				for i, memories in enumerate(self.experiments):
					memories = memories[:, inds]
					print(f'---------------------- Conducting experiment: {i} ----------------------')
					memories_chunked = self.chunked(memories, self.splits)
					experiment_nets = []
					for j, memory_chunk in enumerate(memories_chunked):
						chunked_nets = []
						for _ in range(self.num_nets):
							hop = hdn.HopfieldNetMPF(N=size)
							rand_memories = np.array(
								[random.choice(memory_chunk) for _ in
								 range(round(len(memory_chunk) * self.train_percent))])
							hop.store_patterns_using_mpf(rand_memories + 0.0)
							chunked_nets.append(hop)
						experiment_nets.append(chunked_nets)
						print(f'Experiment: {i}/{len(self.experiments)} // Chunk: {j}/{self.splits} // Bs Size: {size}')
					print(f'---------------------- Finished experiment: {i} ----------------------')
					bs_nets.append(experiment_nets)
			self.networks.append(bs_nets)

	def chunked(self, iterable, n):
		chunksize = int(math.ceil(len(iterable) / n))
		return (iterable[i * chunksize:i * chunksize + chunksize] for i in range(n))

	def getL(self, x, h1, A):
		L = np.exp(np.dot(h1.T, x) + A)
		if L > 1:
			return 1
		else:
			return 0

	def get_preds(self, y, hop):
		J = hop.J
		h = -hop.theta
		A = J[-1, -1]
		B = h[-1]
		# J0 = J[:-1, :-1]
		j = J[-1, :-1]
		# jT = J[-1, :-1]
		# J = J0
		h1 = 2 * j
		# h0 = h
		A = A + B
		x = y[:-1]
		p = self.getL(x, h1, A)
		return p

	def get_accuracy(self, memories, hop, precision_recall=False):
		accuracy = 0
		y_preds = []
		y_true = []
		for k, i in enumerate(memories):
			y_preds.append(self.get_preds(i, hop))
			y_true.append(i[-1])
			if y_preds[k] == y_true[k]:
				accuracy += 1
		accuracy = accuracy / len(memories)
		if not precision_recall:
			return round(accuracy * 100, 3)
		else:
			tn, fp, fn, tp = confusion_matrix(y_true, y_preds)
			return tn, fp, fn, tp, accuracy

	def get_js(self, filename='Js_Joost.pkl'):
		Js = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for networks in memory_chunk_networks:
					if type(networks) != list:
						chunk_nets.append(networks._J)
					else:
						bs_nets = []
						for network in networks:
							bs_nets.append(network._J)
						chunk_nets.append(bs_nets)
				experiment_nets.append(chunk_nets)
			Js.append(experiment_nets)
		Js = np.array(Js).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(Js, file)
		return Js

	def get_thetas(self, filename='Thetas_Joost_old.pkl'):
		thetas = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for networks in memory_chunk_networks:
					if type(networks) != list:
						chunk_nets.append(networks._theta)
					else:
						bs_nets = []
						for network in networks:
							bs_nets.append(network._theta)
						chunk_nets.append(bs_nets)
				experiment_nets.append(chunk_nets)
			thetas.append(experiment_nets)
		thetas = np.array(thetas).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(thetas, file)
		return thetas

	def binaryVecs(self, dat, dt=None):
		if ((self.type == 'map') or (self.type == 'MI')) & (self.data_type == 'old'):
			print("Since you're using the old data type, the following adjustments will be made: \n")
			print("The StimTimes variable will be assumed to be in the sample number form \n"
			      "The Ts variable will also be assumed to be in the sample number form \n"
			      "No changes will be made to either of the raw variables")
			if dt is None:
				dt = 0.05
			StimTimes = dat['StimTimes']
			Cs = np.array(dat['Cs'], dtype='uint32')
			Ts = np.array(dat['Ts'], dtype='uint32')
			foo_s = np.asarray([int(i / dt) for i in StimTimes])
			Tmax = np.max([np.max(StimTimes), np.max(Ts)])  # using just Ts will be more accurate
		elif (self.type == 'J') & (self.data_type == 'old'):
			if dt is None:
				dt = 800
			Cs = np.array(dat['Cs'], dtype='uint32')
			Ts = np.array(dat['Ts'], dtype='uint32')
			# Cs, Ts = self.clean_Cs_and_Ts(Cs, Ts)
			Tmax = np.max(Ts)
		elif (self.type == 'J') & (self.data_type == 'sim'):
			ys = dat['SPK_Times'].T
			return ys
		elif ((self.type == 'map') or (self.type == 'MI')) & (self.data_type == 'new'):
			print("Since you're using the new data type, the following adjustments will be made: \n")
			print("The StimTimes variable will be assumed to be in seconds \n"
			      "To get the sample number, we multiply StimTimes by 16,000 Hz \n"
			      "The Ts variable will be assumed to already be in sample number form, so no changes will be made to it")
			if dt is None:
				dt = 800
			dat = dat['data']
			StimTimes = dat['StimTimes']
			Cs = dat['Cs']
			Ts = dat['Ts']
			Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
			Ts = np.array([a[0] for a in Ts.tolist()[0][0].tolist()], dtype='uint32')
			StimTimes = np.array([a[0]*16000 for a in StimTimes.tolist()[0][0].tolist()], dtype='uint32')
			foo_s = np.asarray([int(i / dt) for i in StimTimes])
			Tmax = np.max([np.max(StimTimes), np.max(Ts)])  # using just Ts will be more accurate
		else:
			if dt is None:
				dt = 800
			Cs = dat['data']['Cs']
			Ts = dat['data']['Ts']
			Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
			Ts = np.array([a[0] for a in Ts.tolist()[0][0].tolist()], dtype='uint32')
			CsTs = self.clean_Cs_and_Ts(Cs, Ts)
			Cs = CsTs[0]
			Ts = CsTs[1]
			Tmax = np.max(Ts)

		foo_x = np.asarray([int(i / dt) for i in Ts])

		ys = []
		if self.stim_shift < 0:
			stim_hist = deque(maxlen=np.abs(self.stim_shift))
			for _ in range(self.stim_shift):
				stim_hist.append(0)

		for i in range(int(Tmax / dt)):
			if i in foo_x:
				# which neurons are firing
				inds = (i * dt <= Ts) * (Ts < (i + 1) * dt)
				neurons = Cs[inds]
				# for neuron in neurons:
				# 	counter[neuron] += 1
				neurons = list(set(neurons))
				foo2 = np.zeros(self.N)
				foo2[neurons] = 1
			# foo2[foo2 == 0] = np.nan
			else:
				foo2 = np.zeros(self.N)
			# foo2[foo2 == 0] = np.nan
			# is the stimulus firing
			if (self.type == 'map') or (self.type == 'MI'):
				if self.stim_shift < 0:
					foo2[-1] = stim_hist[0]
				if i in foo_s:
					if self.stim_shift < 0:
						stim_hist.append(1)
					else:
						foo2[-1] = 1
				else:
					if self.stim_shift < 0:
						stim_hist.append(0)
					else:
						foo2[-1] = 0
			ys.append(foo2)

		ys = np.asarray(ys, dtype='uint8').squeeze()
		if self.stim_shift > 0:
			stims = ys[:, -1]
			stims = stims[self.stim_shift:]
			ys = ys[:-self.stim_shift, :]
			ys[:, -1] = stims

		counter = np.sum(ys, axis=0)
		for i in range(ys.shape[
			               1] - 1):  # I believe we will not want to remove the stimulus channel even if it less than threshold but if not remove -1
			if counter[i] < self.threshold:
				ys[:, i] = 0

		return ys

	def clean_Cs_and_Ts(self, Cs, Ts, threshold=80_000, last_index=0):
		if 60 not in list(Cs):
			return np.array(Cs), np.array(Ts)
		first_marker = 0
		counter = 0
		Cs_beginning = list(Cs[:last_index])
		Ts_beginning = list(Ts[:last_index])
		Cs = list(Cs[last_index:])
		Ts = list(Ts[last_index:])
		index1 = 0
		index2 = 0
		for k, neuron in enumerate(Cs):
			if (neuron == 60) & (first_marker == 0):
				index1 = k
				first_marker = 1
				continue
			elif neuron == 60:
				index2 = k
				counter = 0
			elif first_marker == 1:
				counter += 1
			if (counter > threshold) or ((k + 1) == len(Cs)):
				cutout = list(range(index1, index2 + 1))
				Cs = [b for a, b in enumerate(Cs) if a not in cutout]
				Cs = Cs_beginning + Cs
				Ts = [b for a, b in enumerate(Ts) if a not in cutout]
				Ts = Ts_beginning + Ts
				return self.clean_Cs_and_Ts(Cs, Ts, threshold, index1 + len(Cs_beginning) + 2 * threshold)

	def run_multiprocessing_J(self, filename):
		dat = spio.loadmat(os.path.join(self.in_directory, filename))
		ys = self.binaryVecs(dat, dt=self.dt)
		self.experiments.append(ys)
		self.filenames.append(filename)
		y_sparse = csr_matrix(ys, dtype='uint8')
		filename = filename[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
		save_npz(os.path.join(self.out_directory, filename), y_sparse)

	def mutInfo_NSB(self, xs, ys, Kx, Ky):
		# use NSB entropy estimator
		# first get nXY and nX and nY
		# could probably just use np.histogram
		nX = {}
		for x in xs:
			if str(x) in nX:
				nX[str(x)] += 1
			else:
				nX[str(x)] = 1

		nY = {}
		for y in ys:
			if str(y) in nY:
				nY[str(y)] += 1
			else:
				nY[str(y)] = 1

		nXY = {}
		for i in range(len(xs)):
			x = xs[i]
			y = ys[i]
			if str(x) + '+' + str(y) in nXY:
				nXY[str(x) + '+' + str(y)] += 1
			else:
				nXY[str(x) + '+' + str(y)] = 1

		nX = np.asarray([nx for nx in nX.values()])
		nY = np.asarray([ny for ny in nY.values()])
		nXY = np.asarray([nxy for nxy in nXY.values()])
		#
		Kxy = Kx * Ky

		#
		# now use the following defn
		def entropy_NSB(ns, K):
			ns = ns[ns > 0]
			N = np.sum(ns)

			def Lagrangian(beta):
				K0 = K - len(ns)
				L = -np.sum(sps.gammaln(beta + ns)) - K0 * sps.gammaln(beta) + K * sps.gammaln(beta) - sps.gammaln(
					K * beta) + sps.gammaln(K * beta + N)
				return L

			# Before: find the beta that minimizes L
			ans = spo.minimize_scalar(lambda x: Lagrangian(x), bounds=[(0, None)])
			b = ans.x
			# calculate average S
			foos = (ns + b) * (sps.psi(N + K * b + 1) - sps.psi(ns + b + 1)) / (N + K * b)
			K0 = K - len(ns)
			S = np.sum(foos) + (K0 * b * (sps.psi(N + K * b + 1) - sps.psi(b + 1)) / (N + K * b))

			def avgS2(ns, K, b):
				N = np.sum(ns)
				K0 = K - len(ns)
				# calculate T
				foo1 = (sps.psi(ns + b + 1) - sps.psi(N + K * b + 1)) ** 2 + sps.polygamma(1,
				                                                                           ns + b + 2) - sps.polygamma(
					1, N + K * b + 2)
				T = np.sum((ns + b) * (ns + b + 1) * foo1 / (N + K * b) / (N + K * b + 1))
				foo1 = (sps.psi(b + 1) - sps.psi(N + K * b + 1)) ** 2 + sps.polygamma(1, b + 2) - sps.polygamma(1,
				                                                                                                N + K * b + 2)
				T += K0 * b * (b + 1) * foo1 / (N + K * b) / (N + K * b + 1)

				# calculate R
				def r(ni, nj, N, K, b):
					alphai = ni + b
					alphaj = nj + b
					foo1 = (sps.psi(alphai) - sps.psi(N + K * b + 1)) * (
							sps.psi(alphaj) - sps.psi(N + K * b + 1)) - sps.polygamma(1, N + K * b + 2)
					foo1 *= alphaj * alphai / (N + K * b) / (N + K * b + 1)
					return foo1

				foo1 = (ns + b) * (sps.psi(ns + b) - sps.psi(N + K * b + 1))
				R = (np.sum(np.outer(foo1, foo1)) - np.sum(np.outer(ns + b, ns + b)) * sps.polygamma(1,
				                                                                                     N + K * b + 2)) / (
						    N + K * b) / (N + K * b + 1)
				R -= np.sum(r(ns, ns, N, K, b))
				R += K0 * np.sum(r(ns, 0, N, K, b) + r(0, ns, N, K, b))
				if K0 > 0:
					R += np.exp(np.log(K0) + np.log(K0 - 1) + np.log(r(0, 0, N, K, b)))
				return R + T

			S2 = avgS2(ns, K, b)
			return S, S2 - S ** 2

		#
		SXY, varSXY = entropy_NSB(nXY, Kxy)
		SX, varSX = entropy_NSB(nX, Kx)
		SY, varSY = entropy_NSB(nY, Ky)
		return SX + SY - SXY, np.sqrt(varSXY + varSX + varSY)

	# figure out which neuron to focus on
	def MI_subset(self, xs, ys, maxChange=0.01, maxNeurons=7):
		# get the best neuron first
		mis = []
		var_mis = []
		for n in range(self.N):
			foo_y = ys[:, n]
			foo_y = [[y] for y in foo_y]
			foo = self.mutInfo_NSB(xs, foo_y, 2, 2)
			mis.append(foo[0])
			var_mis.append(foo[1])
		MI = [np.max(mis)]
		var_MI = [var_mis[np.argmax(mis)]]
		best_neurons = [np.argmax(mis)]
		#
		deltaMI = np.inf
		k = 1
		# len(best_neurons)
		while (deltaMI > maxChange) & (len(best_neurons) < maxNeurons):
			# choose the next neuron to add
			mis = []
			var_mis = []
			for j in range(self.N):
				if j in best_neurons:
					mis.append(0)
					var_mis.append(0)
				else:
					inds = np.hstack([best_neurons, j])
					foo_y = ys[:, inds]
					foo = list(self.mutInfo_NSB(xs, foo_y, 2, np.power(2, k + 1)))
					mis.append(foo[0])
					var_mis.append(foo[1])
			MI.append(np.max(mis))
			deltaMI = (MI[-1] - MI[-2]) / MI[-2]
			var_MI.append(var_mis[np.argmax(mis)])
			best_neurons = np.hstack([best_neurons, np.argmax(mis)])
			k += 1
		return MI[-1], var_MI[-1], best_neurons

	def get_dats(self, dts=None):
		dats = []
		filenames = []
		for file in os.listdir(self.in_directory):
			dat = spio.loadmat(os.path.join(self.in_directory, file))
			dats.append(dat)
			filename = f'{file} + dts={dts}'
			filenames.append(filename)
		return dats, filenames

	def get_MI_estimates(self, dts=np.asarray([0.01, 0.03, 0.1, 0.3, 1])):
		dats, filenames = self.get_dats(dts)
		allMIs = []
		allstdMIs = []
		for k, dat in enumerate(dats):
			MIs = np.zeros([len(dts), self.splits])
			stdMIs = np.zeros([len(dts), self.splits])
			dat_ys = []
			for i in range(len(dts)):
				ys = self.binaryVecs(dat, dt=dts[i])
				dat_ys.append(ys)
				xs = ys[:, :self.N]
				ys = ys[:, -1]
				xs_chunked = self.chunked(xs, self.splits)
				ys_chunked = self.chunked(ys, self.splits)
				MI_chunks = []
				varMI_chunks = []
				for xs_chunk, ys_chunk in zip(xs_chunked, ys_chunked):
					MI, varMI, best_neurons = self.MI_subset(ys_chunk, xs_chunk, 0.05)
					MI_chunks.append(MI)
					varMI_chunks.append(varMI)
				MIs[i, :] = np.array(MI_chunks).squeeze()
				stdMIs[i, :] = np.sqrt(np.array(varMI_chunks).squeeze())
			self.experiments.append(dat_ys)
			np.savez(filenames[k], MIs=MIs, stdMIs=stdMIs, dts=dts)
			allMIs.append(MIs)
			allstdMIs.append(stdMIs)
		allMIs = np.array(allMIs)
		allstdMIs = np.array(allstdMIs)
		return allMIs, allstdMIs
