# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/00_basics.ipynb (unless otherwise specified).

__all__ = ['GenVars', 'CommonCauses', 'Instruments', 'EffectModifiers', 'Treatments', 'initialize', 'generate',
           'get_obs', 'initialize', 'get_obs', 'generate', 'initialize', 'get_obs', 'generate',
           'stochastically_convert_to_binary', 'initialize', 'generate', 'get_obs', 'Outcomes', 'plot_target_vs_rest',
           'plot_var_hists', 'show_correlations', 'get_Xy', 'get_model_feel', 'get_feature_importance',
           'get_partial_dependencies', 'plot_partial_dependencies', 'GraphGenerator', 'get_only_Xi_to_Y',
           'get_Xi_to_Y_with_ccs_and_such', 'vis_g', 'get_gml']

# Cell
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
from sklearn import ensemble, metrics
import dowhy as dw
from typing import List, Union
import itertools

# Cell
class GenVars:
    def __init__(self, name:str, n_vars:int, **kwargs):
        self.name = name
        self.n_vars = n_vars
        for name, var in kwargs.items():
            setattr(self, name, var)
    @classmethod
    def get_obs(self): raise NotImplementedError
    def initialize(self): raise NotImplementedError
    def generate(self): raise NotImplementedError


class CommonCauses(GenVars):
    def __init__(self, name:str='W', n_common_causes:int=1, rv_mean:stats.rv_continuous=stats.norm,
                 rv_mean_kwargs:dict=None):
        super().__init__(name, n_common_causes)
        self.initialize(rv_mean, rv_mean_kwargs)


class Instruments(GenVars):
    def __init__(self, name:str='Z', n_instruments:int=1, rv_p:stats.rv_continuous=stats.uniform,
                 rv_p_kwargs:dict=None):
        super().__init__(name, n_instruments)
        self.initialize(rv_p, rv_p_kwargs)


class EffectModifiers(GenVars):
    def __init__(self, name:str='X', n_eff_mods:int=1, rv_mean:stats.rv_continuous=stats.uniform,
                 rv_mean_kwargs:dict=None):
        super().__init__(name, n_eff_mods)
        self.initialize(rv_mean, rv_mean_kwargs)


class Treatments(GenVars):
    def __init__(self, name:str='V', n_treatments:int=1, rv:stats.rv_continuous=stats.norm,
                 rv_kwargs:dict=None, cc:CommonCauses=None, ins:Instruments=None,
                 beta:Union[float, List[int], List[float], np.ndarray]=10):
        super().__init__(name, n_treatments)
        self.initialize(rv, rv_kwargs, cc=cc, ins=ins, beta=beta)

# Cell
def initialize(self, rv_mean:stats.rv_continuous, rv_mean_kwargs:dict):
        if not rv_mean_kwargs:
            rv_mean_kwargs = {'loc': 0, 'scale': .1}
        self.rv_mean = rv_mean(**rv_mean_kwargs)
        self.mean = self.rv_mean.rvs(size=self.n_vars)
        self.cov = np.eye(self.n_vars)
        self.rv = stats.multivariate_normal(mean=self.mean, cov=self.cov)

def generate(self, n:int): return self.rv.rvs(size=n)

@classmethod
def get_obs(self, n:int, n_common_causes:int):
    cc = self(n_common_causes=n_common_causes)
    vals = cc.generate(n)
    cc.obs = pd.DataFrame(data=vals, columns=[f'{cc.name}{i}' for i in range(n_common_causes)])
    return cc

CommonCauses.initialize = initialize
CommonCauses.generate = generate
CommonCauses.get_obs = get_obs

# Cell
def initialize(self, rv_p:stats.rv_continuous, rv_p_kwargs:dict):
    if not rv_p_kwargs:
        rv_p_kwargs = {'loc': 0, 'scale': 1}
    self.rv_p = rv_p(**rv_p_kwargs)
    self.p = rv_p.rvs(size=self.n_vars) # np.random.uniform(0, 1, num_instruments)

@classmethod
def get_obs(self, n:int, n_instruments:int):
    ins = self(n_instruments=n_instruments)
    vals = ins.generate(n)
    ins.obs = pd.DataFrame(data=vals, columns=[f'{ins.name}{i}' for i in range(n_instruments)])
    return ins

def generate(self, n:int):
    Z = np.zeros((n, self.n_vars))
    for i in range(self.n_vars):
        if (i % 2) == 0:
            Z[:, i] = np.random.binomial(n=1, p=self.p[i], size=n)  # ???
        else:
            Z[:, i] = np.random.uniform(0, 1, size=n)  # ???
    return Z

Instruments.initialize = initialize
Instruments.get_obs = get_obs
Instruments.generate = generate

# Cell
def initialize(self, rv_mean:stats.rv_continuous, rv_mean_kwargs:dict):
    if not rv_mean_kwargs:
        rv_mean_kwargs = {'loc':-1, 'scale':1}
    self.rv_mean = rv_mean(**rv_mean_kwargs)
    self.mean = self.rv_mean.rvs(size=self.n_vars)
    self.cov = np.eye(self.n_vars)
    self.rv_obs = stats.multivariate_normal(mean=self.mean, cov=self.cov)

@classmethod
def get_obs(self, n:int, n_eff_mods:int):

    em = self(n_eff_mods=n_eff_mods)
    vals = em.generate(n)
    em.obs = pd.DataFrame(data=vals, columns=[f'{em.name}{i}' for i in range(n_eff_mods)])
    return em

def generate(self, n:int): return self.rv_obs.rvs(size=n)

EffectModifiers.initialize = initialize
EffectModifiers.get_obs = get_obs
EffectModifiers.generate = generate

# Cell
def stochastically_convert_to_binary(x:float):
    p = 1/(1+np.exp(-x))  # sigmoid
    return np.random.choice([0, 1], size=1, p=[1-p, p])

# Cell
def initialize(self, rv:stats.rv_continuous, rv_kwargs:dict,
               cc:CommonCauses=None, ins:Instruments=None,
               beta:Union[float, List[int], List[float], np.ndarray]=10):
    if not rv_kwargs:
        rv_kwargs = {'loc': 0, 'scale': 1}
    self.rv = rv(**rv_kwargs)
    self.n_common_causes = cc.n_vars if cc is not None else 0
    self.W = cc.obs.values.copy()
    self.n_instruments = ins.n_vars if ins is not None else 0
    self.Z = ins.obs.values.copy()

    if not isinstance(beta, (list, np.ndarray)):
        self.beta = np.repeat(beta, self.n_vars)


def generate(self, n:int, treatment_is_binary:bool=False):

    t = self.rv.rvs(size=(n, self.n_vars))

    range_c1 = max(self.beta) * .5
    c1 = np.random.uniform(0, range_c1,
                           size=(self.n_common_causes,
                                 self.n_vars))

    range_cz = self.beta
    cz = np.random.uniform(low=range_cz - range_cz * .05,
                           high=range_cz + range_cz * .05,
                           size=(self.n_instruments, self.n_vars))
    if self.n_common_causes > 0:
        t += self.W @ c1

    if self.n_instruments > 0:
        t += self.Z @ cz

    # Converting treatment to binary if required
    if treatment_is_binary:
        t = np.vectorize(stochastically_convert_to_binary)(t)

    return t

@classmethod
def get_obs(self, n:int, n_treatments:int, cc:CommonCauses,
            ins:Instruments, beta:Union[float, List[int], List[float], np.ndarray],
            treatment_is_binary:bool=False):
    treat = self(n_treatments=n_treatments, cc=cc, ins=ins,
                 beta=beta)
    vals = treat.generate(n, treatment_is_binary=treatment_is_binary)
    treat.obs = pd.DataFrame(data=vals, columns=[f'{treat.name}{i}' for i in range(n_treatments)])
    return treat

Treatments.initialize = initialize
Treatments.generate = generate
Treatments.get_obs = get_obs

# Cell
class Outcomes:
    def __init__(self, name:str='Y'):
        self.name = name

    def generate(self, treat:Treatments, cc:CommonCauses,
                em:EffectModifiers, outcome_is_binary:bool=False):
        # TODO: there is a bug when treat.obs.shape[1] > 1 to lead y to become to long
        def _compute_y(t, W, X, beta, c2, ce):
            y =  t @ beta
            if cc.n_vars > 0:
                y += W @ c2
            if em.n_vars > 0:
                y += (X @ ce) * np.prod(t, axis=1)
            return y

        W = cc.obs.values
        X = em.obs.values
        t = treat.obs.values.astype(float)
        beta = treat.beta

        range_c2 = max(beta) * .5
        c2 = np.random.uniform(0, range_c2, size=cc.n_vars)

        if isinstance(beta, (list, np.ndarray)):
            range_ce = max(beta) * .5
        else:
            range_ce = beta * .5

        ce = np.random.uniform(0, range_ce, em.n_vars)

        y = _compute_y(t, W, X, beta, c2, ce)

        if outcome_is_binary:
            y = np.vectorize(stochastically_convert_to_binary)(t).ravel()

        return y

    @classmethod
    def get_obs(self, treat:Treatments, cc:CommonCauses,
                em:EffectModifiers, outcome_is_binary:bool=False):
        out = self()
        vals = out.generate(treat, cc, em, outcome_is_binary)
        out.obs = pd.DataFrame(data=vals, columns=[f'{out.name}'])
        return out

# Cell
def plot_target_vs_rest(obs:pd.DataFrame, target:str='Y'):
    in_cols = [col for col in obs.columns if target!=col]

    for in_col in in_cols:
        fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
        ax.scatter(obs[in_col], obs[target], alpha=.1, marker='o')
        ax.set(xlabel=f'"{in_col}"', ylabel=f'"{target}"',
               title=f'"{target}" vs "{in_col}"')
        plt.show()

# Cell
def plot_var_hists(obs:pd.DataFrame, bins:int=50):
    for col in obs.columns:
        fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
        ax.hist(col, data=obs, bins=bins, density=True)
        ax.set(title=f'"{col}" distribution', xlabel=f'"{col}"',
               ylabel=f'Frequency')
        plt.show()

# Cell
def show_correlations(obs:pd.DataFrame, method:str='spearman'):
    cols = obs.columns
    n = len(cols)
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(obs.corr(method=method))
    ax.set(xticks=range(n), xticklabels=cols,
           yticks=range(n), yticklabels=cols)
    fig.colorbar(im, ax=ax, shrink=.6, label=f'{method} correlation')
    plt.show()

# Cell
def get_Xy(obs:pd.DataFrame, target:str='Y'):
    not_target = [c for c in obs.columns if c!=target]
    X, y = obs.loc[:, not_target].values, \
           obs.loc[:, target].values.ravel()
    return X, y, not_target

# Cell
def get_model_feel(model, obs:pd.DataFrame, target:str='Y',
                   bins:int=50):
    X, y, not_target = get_Xy(obs, target=target)
    _y = model.predict(X)
    Δ = _y - y

    fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
    ax.hist(Δ, bins=bins, density=True)
    ax.set(xlabel='$\Delta = y_p - y_t$', ylabel='Frequency',
           title='Model prediction residuals')
    plt.show()

    for var in not_target:
        fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
        ax.scatter(obs[var], obs[target], alpha=.1, marker='o', label='truth')
        ax.scatter(obs[var], _y, alpha=.1, marker='x', label='prediction')
        ax.set(xlabel=f'"{var}"', ylabel=f'"{target}"',
               title=f'Model prediction vs truth: "{var}"')
        ax.legend(loc='best')
        plt.show()

# Cell
def get_feature_importance(m, obs:pd.DataFrame, target:str='Y',
                           metric:callable=metrics.mean_squared_error):
    X, y, not_target = get_Xy(obs, target=target)
    n_obs, n_row = X.shape
    scores = {}
    for i in range(n_row):
        _X = X.copy()
        np.random.shuffle(_X[:,i])
        _y = m.predict(_X)
        scores[i] = metric(y, _y)

    scores = pd.DataFrame([{'variable': not_target[i], 'feature_importance': scores[i]} for i in scores])
    scores.sort_values('feature_importance', ascending=False, inplace=True)
    return scores

# Cell
def get_partial_dependencies(model, obs:pd.DataFrame,
                             target:str='Y', max_num_obs:int=100,
                             max_num_ys:int=10):

    assert max_num_ys > 0
    X, y, not_target = get_Xy(obs, target=target)
    n_obs, n_row = X.shape
    part_deps = {}

    for ix, var in enumerate(not_target):

        ys = {}
        idp_vals = np.unique(X[:,ix])
        n_u = len(idp_vals)

        step = n_u//max_num_ys if max_num_ys >= n_u else 1
        if step == 0: step = 1
        idp_vals = idp_vals[::step]

        if max_num_obs:
            if max_num_obs >= n_obs:
                ixs = np.arange(n_obs)
            else:
                ixs = np.random.choice(np.arange(n_obs), size=max_num_obs,
                                       replace=False)
        for i, val in enumerate(idp_vals):
            _X = X[ixs,:].copy()
            _X[:,ix] = val
            _y = model.predict(_X)
            ys[i] = _y.copy()

        part_deps[var] = pd.DataFrame({val: ys[i] for i,val in enumerate(idp_vals)})
    return part_deps

# Cell
def plot_partial_dependencies(part_deps:pd.DataFrame,
                              target:str='Y'):

    for var in part_deps:
        fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
        x = part_deps[var].columns.values
        for i, row in part_deps[var].iterrows():
            obs_line, = ax.plot(x, row.values, alpha=.1, lw=1, color='black', label='single obs.')
        avg = part_deps[var].mean(axis=0)
        avg_line, = ax.plot(x, avg, lw=2, color='yellow', label='avg.')
        ax.set(xlabel=f'"{var}"', ylabel=f'"{target}"', title=f'Partial dependency plot "{target}" vs "{var}"')
        ax.legend(loc='best', handles=[obs_line, avg_line])
        plt.show()

# Cell
class GraphGenerator:
    'Generates some specific directed graphs for `obs`'
    def __init__(self, obs:pd.DataFrame, target:str='Y'):
        self.not_targets = [col for col in obs.columns if col != target]
        self.target = target

# Cell
def get_only_Xi_to_Y(self):
    g = nx.DiGraph()
    for var in self.not_targets:
        g.add_edge(var, self.target)
    return g

GraphGenerator.get_only_Xi_to_Y = get_only_Xi_to_Y

# Cell
def get_Xi_to_Y_with_ccs_and_such(self, common_cause='W', effect_modifier='X',
                                  treatment='V', instrument='Z'):
    '''
    common cause → treatment
    common cause → outcome/target
    treatment → outcome/target
    instrument → treatment
    effect modifier → outcome/target
    '''
    g = nx.DiGraph()
    ccs = [v for v in self.not_targets if v.startswith(common_cause)]
    ems = [v for v in self.not_targets if v.startswith(effect_modifier)]
    treats = [v for v in self.not_targets if v.startswith(treatment)]
    inss = [v for v in self.not_targets if v.startswith(instrument)]

    g.add_edges_from([(cc, treat) for cc, treat in itertools.product(ccs, treats)])
    g.add_edges_from([(cc, self.target) for cc in ccs])
    g.add_edges_from([(treat, self.target) for treat in treats])
    g.add_edges_from([(ins, treat) for ins, treat in itertools.product(inss, treats)])
    g.add_edges_from([(em, self.target) for em in ems])
    return g


GraphGenerator.get_Xi_to_Y_with_ccs_and_such = get_Xi_to_Y_with_ccs_and_such

# Cell
def vis_g(self, g:nx.DiGraph, kind:str='spectral'):
    try:
        layout = getattr(nx, f'{kind}_layout')(g)
    except AttributeError as ae:
        raise AttributeError(f'No nx.{kind}_layout found')
    nx.draw(g, layout=layout, with_labels=True)
GraphGenerator.vis_g = vis_g

# Cell
def get_gml(self, g:nx.DiGraph):
    return ''.join([v for v in nx.readwrite.gml.generate_gml(g)]); gml

GraphGenerator.get_gml = get_gml