import os
import time
import sys
# import utils
import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from numpy import sqrt, zeros, abs, floor, log, log2, eye, exp
# from geometry_utils import ExpMap, VecTransport, radial_proj, orthogonalize, renormalize

class CholeskyCMAES:
    """ Note this is a variant of CMAES Cholesky suitable for high dimensional optimization"""
    def __init__(self, space_dimen, population_size=None, init_sigma=3.0, init_code=None, Aupdate_freq=10,
                 maximize=True, random_seed=None, optim_params={}):
        N = space_dimen
        self.space_dimen = space_dimen
        # Overall control parameter
        self.maximize = maximize  # if the program is to maximize or to minimize
        # Strategy parameter setting: Selection
        if population_size is None:
            self.lambda_ = int(4 + floor(3 * log2(N)))  # population size, offspring number
            # the relation between dimension and population size.
        else:
            self.lambda_ = population_size  # use custom specified population size
        mu = self.lambda_ / 2  # number of parents/points for recombination
        #  Select half the population size as parents
        weights = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))))  # muXone array for weighted recombination
        self.mu = int(floor(mu))
        self.weights = weights / sum(weights)  # normalize recombination weights array
        mueff = self.weights.sum() ** 2 / sum(self.weights ** 2)  # variance-effectiveness of sum w_i x_i
        self.weights.shape = (1, -1)  # Add the 1st dim 1 to the weights mat
        self.mueff = mueff  # add to class variable
        self.sigma = init_sigma  # Note by default, sigma is None here.
        print("Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\nInitial sigma: %.3f"
              % (self.space_dimen, self.lambda_, self.mu, self.sigma))
        # Strategy parameter settiself.weightsng: Adaptation
        self.cc = 4 / (N + 4)  # defaultly  0.0009756
        self.cs = sqrt(mueff) / (sqrt(mueff) + sqrt(N))  # 0.0499
        self.c1 = 2 / (N + sqrt(2)) ** 2  # 1.1912701410022985e-07
        if "cc" in optim_params.keys():  # if there is outside value for these parameter, overwrite them
            self.cc = optim_params["cc"]
        if "cs" in optim_params.keys():
            self.cs = optim_params["cs"]
        if "c1" in optim_params.keys():
            self.c1 = optim_params["c1"]
        self.damps = 1 + self.cs + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1)  # damping for sigma usually  close to 1

        print("cc=%.3f, cs=%.3f, c1=%.3f damps=%.3f" % (self.cc, self.cs, self.c1, self.damps))
        if init_code is not None:
            self.init_x = np.asarray(init_code)
            self.init_x.shape = (1, N)
        else:
            self.init_x = None  # FIXED Nov. 1st
        self.xmean = zeros((1, N))
        self.xold = zeros((1, N))
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = zeros((1, N))
        self.ps = zeros((1, N))  # evolution paths for C and sigma
        self.A = eye(N, N)  # covariant matrix is represent by the factors A * A '=C
        self.Ainv = eye(N, N)

        self.eigeneval = 0  # track update of B and D
        self.counteval = 0
        if Aupdate_freq is None:
            self.update_crit = self.lambda_ / self.c1 / N / 10
        else:
            self.update_crit = Aupdate_freq * self.lambda_
        self.chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        # expectation of ||N(0,I)|| == norm(randn(N,1)) in 1/N expansion formula
        self._istep = 0

    def step_simple(self, scores, codes):
        """ Taking scores and codes to return new codes, without generating images
        Used in cases when the images are better handled in outer objects like Experiment object
        """
        # Note it's important to decide which variable is to be saved in the `Optimizer` object
        # Note to confirm with other code, this part is transposed.
        # set short name for everything to simplify equations
        N = self.space_dimen
        lambda_, mu, mueff, chiN = self.lambda_, self.mu, self.mueff, self.chiN
        cc, cs, c1, damps = self.cc, self.cs, self.c1, self.damps
        sigma, A, Ainv, ps, pc, = self.sigma, self.A, self.Ainv, self.ps, self.pc,
        # Sort by fitness and compute weighted mean into xmean
        if self.maximize is False:
            code_sort_index = np.argsort( scores)  # add - operator it will do maximization.
        else:
            code_sort_index = np.argsort(-scores)
        # scores = scores[code_sort_index]  # Ascending order. minimization
        if self._istep == 0:
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            if self.init_x is None:
                select_n = len(code_sort_index[0:mu])
                temp_weight = self.weights[:, :select_n] / np.sum(self.weights[:, :select_n]) # in case the codes is not enough
                self.xmean = temp_weight @ codes[code_sort_index[0:mu], :]
            else:
                self.xmean = self.init_x
        else:
            self.xold = self.xmean
            self.xmean = self.weights @ codes[code_sort_index[0:mu], :]  # Weighted recombination, new mean value
            # Cumulation statistics through steps: Update evolution paths
            randzw = self.weights @ self.randz[code_sort_index[0:mu], :]
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * randzw
            pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * randzw @ A
            # Adapt step size sigma
            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1))
            # self.sigma = self.sigma * exp((self.cs / self.damps) * (norm(ps) / self.chiN - 1))
            print("sigma: %.2f" % sigma)
            # Update A and Ainv with search path
            if self.counteval - self.eigeneval > self.update_crit:  # to achieve O(N ^ 2) do decomposition less frequently
                self.eigeneval = self.counteval
                t1 = time.time()
                v = pc @ Ainv
                normv = v @ v.T
                # Directly update the A Ainv instead of C itself
                A = sqrt(1 - c1) * A + sqrt(1 - c1) / normv * (
                            sqrt(1 + normv * c1 / (1 - c1)) - 1) * v.T @ pc  # FIXME, dimension error, # FIXED aug.13th
                Ainv = 1 / sqrt(1 - c1) * Ainv - 1 / sqrt(1 - c1) / normv * (
                            1 - 1 / sqrt(1 + normv * c1 / (1 - c1))) * Ainv @ v.T @ v
                t2 = time.time()
                print("A, Ainv update! Time cost: %.2f s" % (t2 - t1))
        # Generate new sample by sampling from Gaussian distribution
        new_samples = zeros((self.lambda_, N))
        self.randz = randn(self.lambda_, N)  # save the random number for generating the code.
        for k in range(self.lambda_):
            new_samples[k:k + 1, :] = self.xmean + sigma * (self.randz[k, :] @ A)  # m + sig * Normal(0,C)
            # Clever way to generate multivariate gaussian!!
            # Stretch the guassian hyperspher with D and transform the
            # ellipsoid by B mat linear transform between coordinates
            self.counteval += 1
        self.sigma, self.A, self.Ainv, self.ps, self.pc = sigma, A, Ainv, ps, pc,
        self._istep += 1
        return new_samples



def mutate(population, genealogy, mutation_size, mutation_rate, random_generator):
    do_mutate = random_generator.random_sample(population.shape) < mutation_rate
    population_new = population.copy()
    population_new[do_mutate] += random_generator.normal(loc=0, scale=mutation_size, size=np.sum(do_mutate))
    genealogy_new = ['%s+mut' % gen for gen in genealogy]
    return population_new, genealogy_new


def mate(population, genealogy, fitness, new_size, random_generator, skew=0.5):
    """
    fitness > 0
    """
    # clean data
    assert len(population) == len(genealogy)
    assert len(population) == len(fitness)
    if np.max(fitness) == 0:
        fitness[np.argmax(fitness)] = 0.001
    if np.min(fitness) <= 0:
        fitness[fitness <= 0] = np.min(fitness[fitness > 0])

    fitness_bins = np.cumsum(fitness)
    fitness_bins /= fitness_bins[-1]
    parent1s = np.digitize(random_generator.random_sample(new_size), fitness_bins)
    parent2s = np.digitize(random_generator.random_sample(new_size), fitness_bins)
    new_samples = np.empty((new_size, population.shape[1]))
    new_genealogy = []
    for i in range(new_size):
        parentage = random_generator.random_sample(population.shape[1]) < skew
        new_samples[i, parentage] = population[parent1s[i]][parentage]
        new_samples[i, ~parentage] = population[parent2s[i]][~parentage]
        new_genealogy.append('%s+%s' % (genealogy[parent1s[i]], genealogy[parent2s[i]]))
    return new_samples, new_genealogy


class Genetic():
    """Need rewrite or debugging. """
    def __init__(self, population_size, mutation_rate, mutation_size, kT_multiplier, recorddir,
                 parental_skew=0.5, n_conserve=0, random_seed=None, thread=None):
        super(Genetic, self).__init__(recorddir, random_seed, thread)

        # various parameters
        self._popsize = int(population_size)
        self._mut_rate = float(mutation_rate)
        self._mut_size = float(mutation_size)
        self._kT_mul = float(kT_multiplier)
        self._kT = None  # deprecated; will be overwritten
        self._n_conserve = int(n_conserve)
        assert (self._n_conserve < self._popsize)
        self._parental_skew = float(parental_skew)

        # # initialize dynamic parameters & their types
        # self._dynparams['mutation_rate'] = \
        #     DynamicParameter('d', self._mut_rate, 'probability that each gene will mutate at each step')
        # self._dynparams['mutation_size'] = \
        #     DynamicParameter('d', self._mut_size, 'stdev of the stochastic size of mutation')
        # self._dynparams['kT_multiplier'] = \
        #     DynamicParameter('d', self._kT_mul, 'used to calculate kT; kT = kT_multiplier * stdev of scores')
        # self._dynparams['n_conserve'] = \
        #     DynamicParameter('i', self._n_conserve, 'number of best individuals kept unmutated in each step')
        # self._dynparams['parental_skew'] = \
        #     DynamicParameter('d', self._parental_skew, 'amount inherited from one parent; 1 means no recombination')
        # self._dynparams['population_size'] = \
        #     DynamicParameter('i', self._popsize, 'size of population')

        # initialize samples & indices
        self._init_population = self._random_generator.normal(loc=0, scale=1, size=(self._popsize, 4096))
        self._init_population_dir = None
        self._init_population_fns = None
        self._curr_samples = self._init_population.copy()  # curr_samples is current population of codes
        self._genealogy = ['standard_normal'] * self._popsize
        self._curr_sample_idc = range(self._popsize)
        self._next_sample_idx = self._popsize
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]

        # # reset random seed to ignore any calls during init
        # if random_seed is not None:
        #     random_generator.seed(random_seed)

    def load_init_population(self, initcodedir, size):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        # make sure size <= population size
        assert size <= self._popsize, 'size %d too big for population of size %d' % (size, self._popsize)
        # load codes
        init_population, genealogy = utils_old.load_codes2(initcodedir, size)
        # fill the rest of population if size==len(codes) < population size
        if len(init_population) < self._popsize:
            remainder_size = self._popsize - len(init_population)
            remainder_pop, remainder_genealogy = mate(
                init_population, genealogy,  # self._curr_sample_ids[:size],
                np.ones(len(init_population)), remainder_size,
                self._random_generator, self._parental_skew
            )
            remainder_pop, remainder_genealogy = mutate(
                remainder_pop, remainder_genealogy, self._mut_size, self._mut_rate, self._random_generator
            )
            init_population = np.concatenate((init_population, remainder_pop))
            genealogy = genealogy + remainder_genealogy
        # apply
        self._init_population = init_population
        self._init_population_dir = initcodedir
        self._init_population_fns = genealogy  # a list of '*.npy' file names
        self._curr_samples = self._init_population.copy()
        self._genealogy = ['[init]%s' % g for g in genealogy]
        # no update for idc, idx, ids because popsize unchanged
        try:
            self._prepare_images()
        except RuntimeError:  # this means generator not loaded; on load, images will be prepared
            pass

    def save_init_population(self):
        '''Record experimental parameter: initial population
        in the directory "[:recorddir]/init_population" '''
        assert (self._init_population_fns is not None) and (self._init_population_dir is not None), \
            'please load init population first by calling load_init_population();' + \
            'if init is not loaded from file, it can be found in experiment backup_images folder after experiment runs'
        recorddir = os.path.join(self._recorddir, 'init_population')
        try:
            os.mkdir(recorddir)
        except OSError as e:
            if e.errno == 17:
                # ADDED Sep.17, To let user delete the directory if existing during the system running.
                chs = input("Dir %s exist input y to delete the dir and write on it, n to exit" % recorddir)
                if chs is 'y':
                    print("Directory %s all removed." % recorddir)
                    rmtree(recorddir)
                    os.mkdir(recorddir)
                else:
                    raise OSError('trying to save init population but directory already exists: %s' % recorddir)
            else:
                raise
        for fn in self._init_population_fns:
            copyfile(os.path.join(self._init_population_dir, fn), os.path.join(recorddir, fn))

    def step_simple(self, scores, codes):
        """ Taking scores and codes from outside to return new codes,
        without generating images
        Used in cases when the images are better handled in outer objects like Experiment object

        Discard the nan handling part!
        Discard the genealogy recording part
        """
        assert len(scores) == len(codes), \
            'number of scores (%d) != population size (%d)' % (len(scores), len(codes))
        new_size = self._popsize  # this may != len(curr_samples) if it has been dynamically updated
        new_samples = np.empty((new_size, codes.shape[1]))
        # instead of chaining the genealogy, alias it at every step
        curr_genealogy = np.array(self._curr_sample_ids, dtype=str)
        new_genealogy = [''] * new_size  # np array not used because str len will be limited by len at init

        # deal with nan scores:
        nan_mask = np.isnan(scores)
        n_nans = int(np.sum(nan_mask))
        valid_mask = ~nan_mask
        n_valid = int(np.sum(valid_mask))
        assert n_nans == 0  # discard the part dealing with nans
        # if some images have scores
        valid_scores = scores[valid_mask]
        self._kT = max((np.std(valid_scores) * self._kT_mul, 1e-8))  # prevents underflow kT = 0
        print('kT: %f' % self._kT)
        sort_order = np.argsort(valid_scores)[::-1]  # sort from high to low
        valid_scores = valid_scores[sort_order]
        # Note: if new_size is smalled than n_valid, low ranking images will be lost
        thres_n_valid = min(n_valid, new_size)
        new_samples[:thres_n_valid] = codes[valid_mask][sort_order][:thres_n_valid]
        new_genealogy[:thres_n_valid] = curr_genealogy[valid_mask][sort_order][:thres_n_valid]

        fitness = np.exp((valid_scores - valid_scores[0]) / self._kT)
        # skips first n_conserve samples
        n_mate = new_size - self._n_conserve - n_nans
        new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
            mate(
                new_samples[:thres_n_valid], new_genealogy[:thres_n_valid],
                fitness, n_mate, self._random_generator, self._parental_skew
            )
        new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
            mutate(
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid],
                self._mut_size, self._mut_rate, self._random_generator
            )

        self._istep += 1
        self._genealogy = new_genealogy
        self._curr_samples = new_samples
        self._genealogy = new_genealogy
        self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + new_size)  # cumulative id .
        self._next_sample_idx += new_size
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]
        return new_samples


    def add_immigrants(self, codedir, size, ignore_conserve=False):
        if not ignore_conserve:
            assert size <= len(self._curr_samples) - self._n_conserve, \
                'size of immigrantion should be <= size of unconserved population because ignore_conserve is False'
        else:
            assert size < len(self._curr_samples), 'size of immigrantion should be < size of population'
            if size > len(self._curr_samples) - self._n_conserve:
                print('Warning: some conserved codes are being overwritten')

        immigrants, immigrant_codefns = utils_old.load_codes2(codedir, size)
        n_immi = len(immigrants)
        n_conserve = len(self._curr_samples) - n_immi
        self._curr_samples = np.concatenate((self._curr_samples[:n_conserve], immigrants))
        self._genealogy = self._genealogy[:n_conserve] + ['[immi]%s' % fn for fn in immigrant_codefns]
        next_sample_idx = self._curr_sample_idc[n_conserve] + n_immi
        self._curr_sample_idc = range(self._curr_sample_idc[0], next_sample_idx)
        self._next_sample_idx = next_sample_idx
        if self._thread is None:
            new_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc[n_conserve:]]
        else:
            new_sample_ids = ['thread%02d_gen%03d_%06d' %
                              (self._thread, self._istep, idx) for idx in self._curr_sample_idc[n_conserve:]]
        self._curr_sample_ids = self._curr_sample_ids[:n_conserve] + new_sample_ids
        self._prepare_images()

    def update_dynamic_parameters(self):
        if self._dynparams['mutation_rate'].value != self._mut_rate:
            self._mut_rate = self._dynparams['mutation_rate'].value
            print('updated mutation_rate to %f at step %d' % (self._mut_rate, self._istep))
        if self._dynparams['mutation_size'].value != self._mut_size:
            self._mut_size = self._dynparams['mutation_size'].value
            print('updated mutation_size to %f at step %d' % (self._mut_size, self._istep))
        if self._dynparams['kT_multiplier'].value != self._kT_mul:
            self._kT_mul = self._dynparams['kT_multiplier'].value
            print('updated kT_multiplier to %.2f at step %d' % (self._kT_mul, self._istep))
        if self._dynparams['parental_skew'].value != self._parental_skew:
            self._parental_skew = self._dynparams['parental_skew'].value
            print('updated parental_skew to %.2f at step %d' % (self._parental_skew, self._istep))
        if self._dynparams['population_size'].value != self._popsize or \
                self._dynparams['n_conserve'].value != self._n_conserve:
            n_conserve = self._dynparams['n_conserve'].value
            popsize = self._dynparams['population_size'].value
            if popsize < n_conserve:  # both newest
                if popsize == self._popsize:  # if popsize hasn't changed
                    self._dynparams['n_conserve'].set_value(self._n_conserve)
                    print('rejected n_conserve update: new n_conserve > old population_size')
                else:  # popsize has changed
                    self._dynparams['population_size'].set_value(self._popsize)
                    print('rejected population_size update: new population_size < new/old n_conserve')
                    if n_conserve <= self._popsize:
                        self._n_conserve = n_conserve
                        print('updated n_conserve to %d at step %d' % (self._n_conserve, self._istep))
                    else:
                        self._dynparams['n_conserve'].set_value(self._n_conserve)
                        print('rejected n_conserve update: new n_conserve > old population_size')
            else:
                if self._popsize != popsize:
                    self._popsize = popsize
                    print('updated population_size to %d at step %d' % (self._popsize, self._istep))
                if self._n_conserve != n_conserve:
                    self._n_conserve = n_conserve
                    print('updated n_conserve to %d at step %d' % (self._n_conserve, self._istep))

    def save_current_genealogy(self):
        savefpath = os.path.join(self._recorddir, 'genealogy_gen%03d.npz' % self._istep)
        save_kwargs = {'image_ids': np.array(self._curr_sample_ids, dtype=str),
                       'genealogy': np.array(self._genealogy, dtype=str)}
        utils_old.savez(savefpath, save_kwargs)

    @property
    def generation(self):
        '''Return current step number'''
        return self._istep

