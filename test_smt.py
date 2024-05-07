from lean_dojo import *

repo = LeanGitRepo('https://github.com/ZIYU-DEEP/miniF2F-lean4',
                   '70a2e77ea57ead6adfc41376711962d7b509e037')

theorem = Theorem(repo, 'MiniF2F/Valid.lean', 'amc12a_2015_p10')

with Dojo(theorem) as (dojo, init_state):
    print(init_state)
    result = dojo.run_tac(init_state, 'smt!')
    # assert isinstance(result, ProofFinished)
    print(result)
    breakpoint()
