import LOTlib3
from LOTlib3.Grammar import Grammar
from LOTlib3.Eval import primitive
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from LOTlib3.Miscellaneous import q
from LOTlib3.DataAndObjects import FunctionData
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler
from LOTlib3.TopN import TopN

import numpy as np
from functools import cache
import builtins
from copy import copy
from itertools import product
from pprint import pprint

from copy import deepcopy

#### PROGRAM SEMANTICS

@primitive
def f_not(a):
    return not a

@primitive
def f_and(a,b):
    return (a and b)

#assuming that memory[agent] is a set
@primitive
def f_in(string, memory, agent):
    return (string in memory[int(agent[1:])])

@primitive
def f_if(cond, thunk1, thunk2):
    if cond:
        thunk1()
    else:
        thunk2()

@primitive
def f_equals(a,b):
    return (a == b)

@primitive
def forallagents(func, game):
    for agent in range(game.n_agents):
        func('a' + str(agent))

@primitive
def forallobjects(func, game):
    for obj in range(game.n_objects):
        func('o' + str(obj))

@primitive
def forallstrings(func, agent, mem):
    for string in mem[int(agent[1:])].copy():
        func(string)

@primitive
def f_predict(agent, obj, preds):
    preds[int(agent[1:])] = int(obj[1:])

@primitive
def f_write(agent,string, mem):
    # writes string to the memory set for agent
    mem[int(agent[1:])].add(string)

@primitive
def f_remove(agent, string, mem):
    # removes string from memory set for agent
    mem[int(agent[1:])].discard(string)

@primitive
def f_closerthan(ag1,ag2,obj,game):
    # returns true if ag1 is closer to obj than ag2 in game
    for ag in game.state[int(obj[1:])]:
        if ag == int(ag1[1:]):
            return True
        elif ag == int(ag2[1:]):
            return False

@primitive
def f_chose(ag,obj,acts):
    #acts should be None when called in the prediction function
    if acts is None:
        return False
    # returns true if ag went for obj in acts
    return (acts[int(ag[1:])] == int(obj[1:]))

@primitive
def f_self(own_index):
    return 'a' + str(own_index)


def define_grammar(n_objects, n_agents, expr_nest_prob):

    grammar = Grammar(
        # programs consist of a move prediction function and of a memory update function
        # the memory structure is a set of strings for each agent
        # the stored strings are built from the alphabet {z, names of agents, names of objects}
        # the prediction function calls "predict(agent,object)" and then any agents not predicted are given a random prediction
        # the update function calls "write(agent, string)" and "remove(agent, string)" to add/remove string from the set corresponding to agent
        start='INF'
    )

    grammar.add_rule(
        'INF',
        # game is the Game object being played; accessed through f_closerthan
        # own_index is the agent's own index in game; accessed through
        # mem is the agent's current memory state: a list of sets of strings, one set for each agent in game
        #   accessed through f_in, f_write, f_remove, forallstrings
        # preds is a list of predictions made for the agents of the game
        #   initially [default] * game.n_agents, updated by each call to f_predict
        # acts is a list of actions taken by each other agent in game; accessed through f_chose
        #   acts should be None when calling the prediction function; calls to f_chose will always be false
        '[(lambda game, own_index, mem, acts, preds: %s), (lambda game, own_index, mem, acts: %s)]',
        ['P_EXPR', 'M_EXPR'],
        1
    )

    #### define data strings
    grammar.add_rule('STR', '%s + %s', ['STR','STR'], 0.5)
    grammar.add_rule('STR','%s', ['OBJ'], 1.0)
    grammar.add_rule('STR','%s', ['AG'], 1.0)
    grammar.add_rule('STR',q('z'), None, 1.0)

    for i in range(n_objects):
        grammar.add_rule('OBJ', q('o' + str(i)), None, 3/n_objects)

    for i in range(n_agents):
        grammar.add_rule('AG', q('a' + str(i)), None, 3/n_agents)

    grammar.add_rule('AG','f_self',['SELF'], 1)

    #### define boolean stuff, top-level variables
    grammar.add_rule('BOOL', 'f_and', ['BOOL', 'BOOL'], 0.5)
    grammar.add_rule('BOOL', 'f_not', ['BOOL'], 0.5)
    grammar.add_rule('BOOL', 'f_equals', ['AG','AG'],0.5)
    grammar.add_rule('BOOL', 'f_equals', ['OBJ','OBJ'],0.5)
    grammar.add_rule('BOOL', 'f_equals', ['STR','STR'],0.5)

    grammar.add_rule('GAME', 'game', None, 1)
    grammar.add_rule('SELF', 'own_index', None, 1)
    grammar.add_rule('MEM', 'mem', None, 1)
    grammar.add_rule('ACTS','acts', None, 1)
    grammar.add_rule('PREDS', 'preds', None, 1)

    #### ways for program to look at game, memory, acts
    grammar.add_rule('BOOL', 'f_closerthan',['AG','AG','OBJ','GAME'], 1.0)
    grammar.add_rule('BOOL', 'f_chose', ['AG','OBJ','ACTS'], 1)
    grammar.add_rule('BOOL', 'f_in', ['STR', 'MEM', 'AG'], 1)

    #### define program logic for prediction program
    grammar.add_rule('P_EXPR', 'f_if',['BOOL','P_THUNK','P_THUNK'], expr_nest_prob)
    grammar.add_rule('P_THUNK','lambda',['P_EXPR'], 1)

    grammar.add_rule('P_EXPR', 'f_predict',['AG','OBJ', 'PREDS'], 1.0)

    ##  iterate over agents
    grammar.add_rule('P_EXPR', 'forallagents', ['PA_FUNC', 'GAME'], expr_nest_prob)
    grammar.add_rule('PA_FUNC', 'lambda', ['P_EXPR'],1.0, bv_type='AG',bv_prefix='var_a')

    ##  iterate over objects
    grammar.add_rule('P_EXPR', 'forallobjects', ['PO_FUNC','GAME'], expr_nest_prob)
    grammar.add_rule('PO_FUNC', 'lambda', ['P_EXPR'],1.0, bv_type='OBJ',bv_prefix='var_o')

    ## iterate over strings in memory[agent]
    grammar.add_rule('P_EXPR','forallstrings',['PS_FUNC','AG','MEM'], expr_nest_prob)
    grammar.add_rule('PS_FUNC', 'lambda', ['P_EXPR'], 1.0, bv_type='STR',bv_prefix='var_s')

    #### define program logic for memory program
    grammar.add_rule('M_EXPR', 'f_if',['BOOL', 'M_THUNK', 'M_THUNK'], expr_nest_prob)
    grammar.add_rule('M_THUNK','lambda',['M_EXPR'], 1)

    grammar.add_rule('M_EXPR', 'f_write',['AG','STR','MEM'],1.0)
    grammar.add_rule('M_EXPR', 'f_remove',['AG','STR','MEM'],1.0)

    ##  iterate over agents
    grammar.add_rule('M_EXPR', 'forallagents', ['MA_FUNC','GAME'], expr_nest_prob)
    grammar.add_rule('MA_FUNC', 'lambda', ['M_EXPR'],1.0, bv_type='AG',bv_prefix='var_a')

    ##  iterate over objects
    grammar.add_rule('M_EXPR', 'forallobjects', ['MO_FUNC','GAME'],expr_nest_prob)
    grammar.add_rule('MO_FUNC', 'lambda', ['M_EXPR'],1.0, bv_type='OBJ',bv_prefix='var_o')

    ## iterate over strings in memory[agent]
    grammar.add_rule('M_EXPR','forallstrings',['MS_FUNC','AG','MEM'], expr_nest_prob)
    grammar.add_rule('MS_FUNC', 'lambda', ['M_EXPR'], 1.0, bv_type='STR',bv_prefix='var_s')

    return grammar


class MyHypothesis(LOTHypothesis):

    def __init__(self, index=None, grammar=define_grammar(n_objects, n_agents, expr_nest_prob), **kwargs):
        """
        Parameters
        ----------
        index: int
            Index of the agent themselves
        """
        self.index = index
        LOTHypothesis.__init__(
            self, 
            grammar=grammar,
            display='lambda: %s',
            maxnodes=1000,
            **kwargs
        )

    def compute_single_likelihood(self, datum):
        
        history = datum.input
        logalpha = np.log(datum.alpha)
        logonemalpha = np.log(1-datum.alpha)
        predf, memf = self()
        
        mem = [set() for _ in range(history[0]['game'].n_agents)]
        loglik = 0
        
        # loop over games
        for i, x in enumerate(history):
            
            game, actions = x['game'], x['actions']
            
            # lambda game, own_index, mem, acts, preds
            predictions = [np.inf]*game.n_agents
            
            predf(game, self.index, mem, None, predictions)
            
            # get total log lik score
            loglik += sum(
                logalpha if pr == ac else logonemalpha
                for pr, ac 
                in zip(predictions, actions)ii
            )

            # lambda game, own_index, mem, acts
            # mem modified in-place!
            memf(game, self.index, mem, actions)
        
        return loglik


class Game:
    
    def __init__(self, state=None, n_agents=5, n_objects=4):
        self.n_agents, self.n_objects = n_agents, n_objects
        if state is None:
            # state: np.array w/ shape (object, agent rank)
            # Containing the distance of each agent from each object.
            self.state = np.array([
                np.random.choice(n_agents, n_agents, False)
                for _ in range(n_objects)
            ])
        else:
            self.state = state

    def play(self, actions):
        """
        NOTE: This function is used for hypothetical games,
        so do NOT modify object state!
        
        Parameters
        ----------
        actions: array[int]
            An action index for each agent,
            Shape (agent).
            The values are the indices of the
            objects that the agent is moving towards,
            or np.inf for staying put.
        """
        assert len(actions)==self.n_agents, "actions must be for all&only agents!"
        reach = np.zeros(self.n_agents)
        for i, obj in enumerate(self.state):
            trying_obj = np.argwhere(actions==i).flatten()
            if trying_obj.size == 1:
                reach[trying_obj[0]] = 1
            elif trying_obj.size > 1:
                distances_of_trying = np.argsort(obj)[trying_obj]
                closest_agent_idx = np.argmin(distances_of_trying)
                reach[trying_obj[closest_agent_idx]] = 1
                reach[np.delete(trying_obj, closest_agent_idx)] = -1
        return reach
        
    def __str__(self):
        return np.array2string(self.state)


class Agent:
    
    def __init__(self, index, max_objects, n_agents, grammar, alpha=0.9999, prefs=None):
        """
        Arguments
        ---------
        max_objects: int
            Maximum number of objects that might appear in a game
        prefs: array[int]
            The ascending ranks of the agent's preferences
            e.g., [2,1,3] means that the agent prefers obj 1 over 2 etc.
        """

        self.alpha = alpha
        self.grammar = grammar
        self.max_objects = max_objects
        self.n_agents = n_agents
        # OWN index
        self.index = index
        # prediction and memory update functions
        hyp = self.define_hypothesis()
        self.program = hyp()
        self.predf, self.memf = self.program
        # memory
        self.mem = self.initialize_mem()
        self.library = []

        self.topN = None

        # (used for library learning)
        self.predf_hist = []
        self.mem_hist = []

        self.set_prefs(prefs)

    def initialize_mem(self):
        return [set() for _ in range(self.n_agents)]

    def define_hypothesis(self):
        return MyHypothesis(self.index, self.grammar)

    def set_prefs(self, prefs):
        if prefs is None:
            prefs = np.random.choice(self.max_objects, self.max_objects, False)
        self.prefs = prefs

    def induce_program(self, history):
        """
        Induce a distribution over functions
        
        Parameters
        ----------
        history: list[LOTlib3.Datum]
        """

        data = [
            FunctionData(
                input=history, 
                output=None,
                # probability of agents performing intended action
                alpha=self.alpha
            )
        ]
        
        h0 = self.define_hypothesis()
        tn = TopN(N=10)
        for h in MetropolisHastingsSampler(h0, data, steps=1e4):
            tn.add(h)

        self.program = tn.best().value
        self.topN = tn
        
        # Once it picked a program, it sets 
        # its self.program attribute and also returns it
        self.predf, self.memf = eval(str(self.program))
        self.predf_hist.append(self.predf)
        self.mem_hist.append(self.memf)
        
        return self.program

    def printBestHyps(self, single=False, **kwargs):
        if self.topN is None:
            print('topN not yet defined!')
        else:
            if single:
                print(self.topN.best())
                return
            # Print the best found hypotheses
            for h in self.topN.get_all(sorted=True, **kwargs):
                print(h.posterior_score, h.likelihood, h.prior, h)

    def move(self, game):
        """
        make a move in the game 
        (based on current memory state)
        """
        
        # choose an action by reward maximization
        # - Keep other agents' actions as predicted, but vary your own action
        # - For each possible action you could take, run a hypothetical game and check your own reward
        # - Pick the action with the highest reward

        # first predict what each of the other agents will do
        predictions = [np.inf]*game.n_agents
        self.predf(game, self.index, self.mem, None, predictions)
        preds = np.array(predictions, dtype='float64')
        
        # Possible actions: n_objects + 1 (stay)
        rewards = np.full(game.n_objects+1, np.inf)
        possible_actions = list(range(game.n_objects)) + [np.inf]
        for i, action in enumerate(possible_actions):
            # set own action to action, play hypothetical game
            h_actions = preds.copy()
            h_actions[self.index] = action
            # only consider reward of own action
            outcome = game.play(h_actions)[self.index]
            if outcome == 1:
                # if the agent reached the object,
                # utility is proportional to its preference
                # for the object
                outcome *= np.argwhere(action==self.prefs)[0,0]
            rewards[i] = outcome
        # pick action that maximises utility
        action = possible_actions[np.argmax(rewards)]
        # change preds to decided action 
        preds[self.index] = action
        # return the chosen action
        return action

    def update_memory(self, game, actions):
        # update the agent's memory
        self.mem_hist.append(deepcopy(self.mem))
        # self.index is agent's own index so it can self-locate
        self.memf(game, self.index, self.mem, actions)

    def reset(self, prefs=None):
        self.mem = self.initialize_mem()
        self.set_prefs(prefs)

    def learn_library(self):
        
        # TODO: library learning from history of programs

        # TODO: update self.grammar by adding new concepts
        
        self.library = []
        return library


def run_simulation(n_agents, n_objects, n_batches, batches_size, history_size, grammar, log=False):
    """
    Parameters
    ----------
    n_batches: int
        How many batches of histories to play?
    batches_size: int
        How many histories to play before library learning?
    history_size: int
        How many games in a history
    """
    
    agents = [
        Agent(i, n_objects, n_agents, grammar)
        for i in range(n_agents)
    ]
    
    batches = []
    for batch_i in range(n_batches):
        
        batch = []
        for hist_i in range(batches_size):

            # TODO: Reset agents' preferences and memories 
            # between sets of games
            for agent in agents:
                agent.reset()
            
            history = []
            for game_i in range(history_size):
                
                game = Game()
                
                # get a decided action from each agent
                actions = np.array([
                    agent.move(game)
                    for agent in agents
                ])

                # play the game, get results
                utils = game.play(actions)

                # update the agents' memories
                for agent in agents:
                    agent.update_memory(
                        game,
                        actions
                    )

                if log:
                    print('game:\n', game)
                    print('prefs:\n', np.array([ag.prefs for ag in agents]))
                    print('actions: ', actions)
                    print(utils)
                    print()
                
                history.append({
                    'game': game,
                    'actions': actions,
                    'utils': utils
                })

            print(f"Done with {batch_i}:{hist_i}:{game_i}") 
            for i, agent in enumerate(agents):
                print(i)
                agent.printBestHyps(single=True)
                print()
            
            # run program induction on the whole history
            learned_programs = [
                agent.induce_program(history)
                for agent in agents
            ]
            
            batch.append({
                'history': history,
                'programs': learned_programs
            })
        
        # run library learning to get more abstract concepts
        # for the next batch
        libraries = [
            agent.learn_library()
            for agent in agents
        ]
        
        batches.append({
            'batch': batch,
            'libraries': libraries
        })
    
    return agents, batches