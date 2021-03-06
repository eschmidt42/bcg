# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/02_causal_model.ipynb (unless otherwise specified).

__all__ = ['CausalGraph', 'show_graph', 'view_graph', 'get_ancestors', 'cut_edges', 'get_causes', 'get_instruments',
           'get_effect_modifiers', 'CausalModel', 'identify_effect', 'construct_backdoor',
           'construct_instrumental_variable', 'RegressionEstimator', 'get_Xy_with_products', 'estimate_effect']

# Cell
import dowhy as dw
from typing import List, Union
import networkx as nx
import itertools
import sympy as sp
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model, neighbors
import numpy as np
from scipy import stats
from .basics import CommonCauses, Instruments, EffectModifiers, Treatments, Outcomes, get_Xy
import pandas as pd

# Cell
class CausalGraph:
    def __init__(self, treatments:List[str], outcome:str='Y',
                 common_causes:List[str]=None, effect_modifiers:List[str]=None,
                 instruments:List[str]=None, observed_nodes:List[str]=None,
                 missing_nodes_as_confounders:bool=False,
                 add_unobserved_confounder:bool=True):
        if common_causes is None: common_causes = []
        if effect_modifiers is None: effect_modifiers = []
        if instruments is None: instruments = []
        if missing_nodes_as_confounders:
            all_passed_nodes = treatments + [outcome] + \
                common_causes + effect_modifiers + instruments
            missing_nodes = [node for node in all_passed_nodes if node not in observed_nodes]
            common_causes = list(common_causes) + missing_nodes

        self.g = self.create_nx_digraph(treatments, outcome,
                                        common_causes, instruments,
                                        effect_modifiers, add_unobserved_confounder)

    @staticmethod
    def create_nx_digraph(treatments:List[str], outcome:str, common_causes:List[str],
                          instruments:List[str], effect_modifiers:List[str],
                          add_unobserved_confounder:bool=False):
        g = nx.DiGraph()
        g.add_edges_from([(treatment, outcome) for treatment in treatments])
        g.add_edges_from([(common_cause, treatment)
                          for common_cause, treatment in itertools.product(common_causes, treatments)])
        g.add_edges_from([(common_cause, outcome)
                          for common_cause in common_causes])
        g.add_edges_from([(effect_modifier, outcome) for effect_modifier in effect_modifiers])
        g.add_edges_from([(instrument, treatment) for instrument, treatment in itertools.product(instruments, treatments)])
        nx.set_node_attributes(g, True, 'observed')
        if add_unobserved_confounder:
            g.add_node('U', observed=False)
            g.add_edges_from([('U', treatment) for treatment in treatments])
            g.add_edge('U', outcome)

        return g


# Cell
def show_graph(g:nx.Graph, kind:str='spectral'):
    try:
        layout = getattr(nx, f'{kind}_layout')(g)
    except AttributeError as ae:
        raise AttributeError(f'No nx.{kind}_layout found')
    nx.draw(g, layout=layout, with_labels=True)

def view_graph(self, kind:str='spectral'):
    show_graph(self.g, kind=kind)

CausalGraph.view_graph = view_graph

# Cell
def get_ancestors(self, node:str, g:nx.DiGraph=None,
                  parents_only:bool=False):
    if parents_only:
        f = self.g if g is None else g
        return f.predecessors(node)
    return nx.ancestors(self.g if g is None else g, node)

CausalGraph.get_ancestors = get_ancestors

# Cell
def cut_edges(self, edges_to_cut:List[tuple]=None):
    if edges_to_cut is None: return None
    g_cut = self.g.copy()
    g_cut.remove_edges_from(edges_to_cut)
    return g_cut

CausalGraph.cut_edges = cut_edges

# Cell
def get_causes(self, nodes:List[str],
               edges_to_cut:List[tuple]=None):
    g_cut = self.cut_edges(edges_to_cut)
    causes = set()
    for node in nodes:
        causes.update(self.get_ancestors(node, g_cut))
    return causes

CausalGraph.get_causes = get_causes

# Cell
def get_instruments(self, treatments:List[str], outcome:str):
    treatment_parents_edges = set()
    treatment_parents = set()
    for treatment in treatments:
        parents = self.get_ancestors(treatment, parents_only=True)
        treatment_parents.update(parents)
        treatment_parents_edges.update([(parent, treatment)
                               for parent in parents])

    g_cut = self.cut_edges(treatment_parents_edges)

    outcome_ancestors = self.get_ancestors(outcome, g_cut)

    instruments_candidates = treatment_parents.difference(outcome_ancestors)

    descendants = set()
    for parent in outcome_ancestors:
        descendants.update(nx.descendants(g_cut, parent))
    instruments = instruments_candidates.difference(descendants)
    return instruments

CausalGraph.get_instruments = get_instruments

# Cell
def get_effect_modifiers(self, treatments:List[str], outcomes:List[str]):
    modifiers = set()
    for outcome in outcomes:
        modifiers.update(self.get_ancestors(outcome))
    modifiers = modifiers.difference(treatments)
    for treatment in treatments:
        modifiers = modifiers.difference(self.get_ancestors(treatment))
    return list(modifiers)

CausalGraph.get_effect_modifiers = get_effect_modifiers

# Cell
class CausalModel:

    def __init__(self, treatments:List[str], outcome:str='Y',
                 common_causes:List[str]=None, effect_modifiers:List[str]=None,
                 instruments:List[str]=None, causal_graph_kwargs=None):
        if not causal_graph_kwargs: causal_graph_kwargs = dict()
        self.cg = CausalGraph(treatments, outcome,
                              common_causes=common_causes,
                              effect_modifiers=effect_modifiers,
                              instruments=instruments,
                              **causal_graph_kwargs)
        self.treatments = treatments
        self.outcome = outcome
        self.common_causes = common_causes
        self.effect_modifiers = effect_modifiers
        self.instruments = instruments

    def identify_effect(self):
        pass
    def estimate_effect(self):
        pass
    def refute_estimate(self):
        pass

# Cell
def identify_effect(self, estimand_type:str='nonparametric-ate'):
    causes = {
        'treatments': self.cg.get_causes(self.treatments),
        'effects': self.cg.get_causes([self.outcome],
                                      edges_to_cut=[(t, self.outcome) for t in self.treatments])
    }
    print(f'causes: {causes}')
    common_causes = causes['treatments'].intersection(causes['effects'])
    print(f'common causes: {common_causes}')

    instruments = self.cg.get_instruments(self.treatments, self.outcome)

    # constructing backdoor estimand
    backdoor = self.construct_backdoor(self.treatments, self.outcome,
                                       common_causes, estimand_type=estimand_type)
    print('Backdoor:', backdoor)

    # constructing instrumental variable estimand
    instrumental_variable = None
    if len(instruments) > 0:
        instrumental_variable = self.construct_instrumental_variable(treatments, outcome, instruments,
                                                                 estimand_type=estimand_type)

    print('Instrumental variable:', instrumental_variable)
    return {
        'observed_common_causes': common_causes,
        'backdoor': backdoor,
        'instrumental_variable': instrumental_variable
    }

def construct_backdoor(self, treatments:List[str], outcome:str,
                       common_causes:List[str],
                       estimand_type:str='nonparametric-ate'):

    if estimand_type != 'nonparametric-ate': raise NotImplementedError

    # treatment variables
    sym_treatments = sp.Array([sp.Symbol(treatment) for treatment in treatments])

    # outcome given common causes
    expr = f'{outcome} | {",".join(common_causes)}' \
        if len(common_causes) > 0 else outcome

    # assigning a normal distribution to the outcome given common causes
    sym_mu = sp.Symbol("mu")
    sym_sigma = sp.Symbol("sigma", positive=True)
    sym_outcome = sp.stats.Normal(expr, sym_mu, sym_sigma)
    # expected outcome given common causes
    sym_conditional_outcome = sp.stats.Expectation(sym_outcome)

    # effect of treatment on outcome given common causes
    sym_effect = sp.Derivative(sym_conditional_outcome, sym_treatments)
    return sym_effect

def construct_instrumental_variable(self, treatments:List[str], outcome:str,
                                    instruments:List[str], estimand_type:str='nonparametric-ate'):

    if estimand_type != 'nonparametric-ate': raise NotImplementedError

    sym_mu, sym_sigma = 0, 1
    sym_outcome = sp.stats.Normal(outcome, sym_mu, sym_sigma)
    sym_treatments = sp.Array([sp.Symbol(sp.stats.Normal(treatment, sym_mu, sym_sigma))
                              for treatment in treatments])
    sym_instruments = sp.Array([sp.Symbol(instrument)
                                for instrument in instruments])
    sym_effect = sp.stats.Expectation(
        sp.Derivative(sym_outcome, sym_instruments)
        /
        sp.Derivative(sym_treatments, sym_instruments)
    )

    return sym_effect

CausalModel.construct_backdoor = construct_backdoor
CausalModel.construct_instrumental_variable = construct_instrumental_variable
CausalModel.identify_effect = identify_effect

# Cell
class RegressionEstimator:

    def __init__(self, model:sklearn.base.RegressorMixin):
        assert isinstance(model, sklearn.base.RegressorMixin)
        self.m = model

    def fit(self, X:np.ndarray, y:np.ndarray, ix:int, ix_confounders:List[int], reset:bool=True):
        if not isinstance(ix_confounders, list):
            ix_confounders = list(ix_confounders)
        self.ix = ix
        self.ix_confounders = ix_confounders
        _ix = [ix] + ix_confounders
        self._ix = _ix
        if reset:
            self.m.fit(X[:,self._ix],y)

    def estimate_effect(self, X:np.ndarray, treatment:Union[int, float], control:Union[int, float],
                        y:np.ndarray=None):
        n, _ = X.shape
        _X = X.copy()
        _X[:, self.ix] = treatment

        treatment_outcomes = self.m.predict(_X[:, self._ix])
        _X[:, self.ix] = control
        control_outcomes = self.m.predict(_X[:, self._ix])

        treatment_mean = treatment_outcomes.mean()
        control_mean = control_outcomes.mean()
        ate = treatment_mean - control_mean

        return ate

# Cell
def get_Xy_with_products(obs:pd.DataFrame, target:str='Y', feature_product_groups:List[list]=None):
    'feaure_product_groups (e.g. [["V0", "V1", "W0"], ["X0", "X1"]]) to compute products between each var in the first and second list (not within each list)'
    not_target = [c for c in obs.columns if c != target and c not in feature_product_groups[1]]
#     out_cols = [col for col in obs.columns if col != target ]

    X, y = obs.loc[:, not_target].values, \
           obs.loc[:, target].values.ravel()
    if feature_product_groups:
        assert isinstance(feature_product_groups, list)
        assert len(feature_product_groups) == 2
        assert all([isinstance(f, list) for f in feature_product_groups])
        product_cols = [(t,e) for t,e in itertools.product(*feature_product_groups)]
        cols = list(obs.columns.values)
        for t, e in product_cols:
            ix_t = cols.index(t)
            ix_e = cols.index(e)
            x = (obs[t] * obs[e]).values
            X = np.concatenate((X, x[:,None]), axis=1)
            not_target.append(f'{t}_{e}')
    return X, y, not_target

# Cell
def estimate_effect(self, estimands:dict, control_value:float,
                    treatment_name:str, treatment_value:float,
                    obs:pd.DataFrame, outcome:str='Y',
                    causal_method:str='backdoor',
                    model:Union[sklearn.base.RegressorMixin,sklearn.base.ClassifierMixin]=None, target_unit:str='ate',
                    effect_modifiers:List[str]=None,
                    supervised_type_is_regression:bool=True):
    assert causal_method in {'backdoor', 'instrumental_variable'}
    assert target_unit == 'ate'
    print('model', model)
    if model is None:
        if supervised_type_is_regression:
            model = linear_model.LinearRegression()
        else:
            model = linear_model.LogisticRegression(solver='lbfgs')

    if effect_modifiers is None:
        effect_modifiers = self.effect_modifiers

    # decide on approach given causal_method and model_type

    # estimate the effect using the arrived on approach
    X, y, not_outcome = get_Xy_with_products(obs, target=outcome, feature_product_groups=[treatments, effect_modifiers])
    if supervised_type_is_regression:
        estimator = RegressionEstimator(model)
    else:
        estimator = PropensityScoreMatcher(model)

    ix = [v.lower() for v in not_outcome].index(treatment_name)
    confounders = self.treatments + list(estimands['observed_common_causes']) + effect_modifiers
    print('confounders', confounders)
    ix_confounders = [_i for _i,_v in enumerate(obs.columns.values) if _v in confounders]
    estimator.fit(X, y, ix, ix_confounders)
    effect = estimator.estimate_effect(X=X, treatment=treatment_value, control=control_value, y=y)
    return effect

CausalModel.estimate_effect = estimate_effect

# propensity_model = linear_model.LogisticRegression(solver='lbfgs')
# estimator = PropensityScoreMatcher(propensity_model)
# estimator.fit(X, y, ix=0, ix_confounders=[1, 2])
# ate = estimator.estimate_effect(X=X, y=y, treatment=True, control=False)
# print(f'ate = {ate:.3f}')