from typing import Any, Tuple
from searchlight.headers import InitialInferencer2, State, ForwardTransitor2
from lean_dojo import *
from utils.llms import *
import logging

class LeanDojoState(State):
    '''
    Wraps around the lean_dojo TacticsState class
    '''
    def __init__(self, tactic_state: TacticState, notes: dict | None = None):
        super().__init__(tactic_state, notes)
        self.tactic_state = tactic_state

    def get_string(self) -> str:
        """
        Return the string state from the state.
        """
        if isinstance(self.tactic_state, TacticState):
            ts = self.tactic_state.pp
        elif isinstance(self.tactic_state, ProofFinished):
            ts = ''
        else:
            ts = self.tactic_state.unsolved_tactic_state
        return ts
    
    def get_tactic_state(self) -> TacticState:
        return self.tactic_state
    
class LowLevelInferencer(InitialInferencer2):
    '''
    
    '''
    def __init__(self, dojo, gen_method, prompt_fn_low, model, 
                 tokenizer, temperatures, 
                 num_samples_low, stop, max_tokens,
                 formal_statement, informal_statement, plan_high):
        super().__init__()
        self.dojo = dojo
        self.gen_method = gen_method
        self.model = model
        self.tokenizer = tokenizer
        self.temperatures = temperatures
        self.num_samples_low = num_samples_low
        self.stop = stop
        self.max_tokens = max_tokens
        self.prompt_fn_low = prompt_fn_low
        self.formal_statement = formal_statement
        self.informal_statement = informal_statement
        self.plan_high = plan_high

        # create logger
        self.logger = logging.getLogger(self.__class__.__name__)
        

    def _predict(self, state: LeanDojoState) -> tuple[dict, dict, dict[tuple[tuple[Any, Any], ...], Any], dict[tuple[tuple[Any, Any], ...], Any], dict]:
        
        # log the state
        self.logger.info(f"Infering for state: {state.get_string()}")

        # if state.get_tactic_state() is ProofFinished, return empty dicts
        if isinstance(state.get_tactic_state(), ProofFinished):
            # log the state as proof finished
            self.logger.info(f"Proof Finished: {state.get_string()}")
            # print it out too
            # print(f"Proof Finished: {state.get_string()}")
            return dict(), dict(), dict(), dict(), dict()
        
        processed_state = state.get_string() #TODO

        # generate actions and scores using gen_method
        step_cands, step_scores = self.gen_method(
            self.prompt_fn_low(tactic_state=processed_state,
                          formal_statement=self.formal_statement,
                          informal_statement=self.informal_statement,
                          plan_high=self.plan_high),
            model=self.model,
            tokenizer=self.tokenizer,
            temperatures=self.temperatures,
            num_samples=self.num_samples_low,
            stop=self.stop,
            max_tokens=self.max_tokens
        )
        step_cands = [s.strip() for s in step_cands] 

        actions = step_cands

        # make sure actions are unique
        actions = list(set(actions))
        
        action_to_next_state = {action: LeanDojoState(self.dojo.run_tac(state.get_tactic_state(), action)) for action in actions}
        
        # step scores are log probabilities, which will be negative
        action_to_reward = {action: step_scores[i] for i, action in enumerate(actions)}
        # unless the action happens to complete the proof, in which case the reward is float('inf')
        for action in actions:
            if isinstance(action_to_next_state[action].get_tactic_state(), ProofFinished):
                action_to_reward[action] = float('inf')

            # filter out any actions that are not valid (i.e. their next state is not ProofFinished or TacticState)
            if not isinstance(action_to_next_state[action].get_tactic_state(), ProofFinished) and not isinstance(action_to_next_state[action].get_tactic_state(), TacticState):
                action_to_reward.pop(action)
                action_to_next_state.pop(action)
                # log the invalid action
                self.logger.info(f"Invalid Action: {action}")

        # if action to reward is empty, return empty dicts (no valid actions left)
        if not action_to_reward:
            return dict(), dict(), dict(), dict(), dict()

        # set next_state_values to 0.0 for now
        next_state_values = {next_state: 0.0 for next_state in action_to_next_state.values()}

        # NOTE: fully expanding this instead of early stopping while expanding should not matter much

        policy = {action: 1.0/len(actions) for action in action_to_next_state.keys()}
        # process through self.single_actor_convert
        policies, heuristic_values, intermediate_rewards, transitions  = self.single_actor_convert(policy=policy,
                                                                        intermediate_rewards=action_to_reward,
                                                                        transitions=action_to_next_state,
                                                                        next_state_values=next_state_values)

        return policies, heuristic_values, intermediate_rewards, transitions, dict()