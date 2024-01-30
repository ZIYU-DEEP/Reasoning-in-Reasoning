import re
import pexpect
import os
import json
from typing import Optional


class ProofSearch:
    """
    from pysagredo.
    """
    def __init__(self, path_to_repl):
        # debug
        # print(f"LOOKING FOR REPL IN {path_to_repl}")
        self.proc = pexpect.spawn(
            "lake env lean --run REPL/Main.lean", cwd=path_to_repl, encoding="utf-8"
        )
        self.proc.debug = True

    def run_code(self, code, env=None, verbose=False):
        if env:
            command = (
                '{ "cmd" : "' + repr(code)[1:-1] + f'", "env" : {env}' + " }"
            )  # [1:-1] removes single quotes
        else:
            command = (
                '{ "cmd" : "' + repr(code)[1:-1] + '" }'
            )  # [1:-1] removes single quotes

        if verbose:
            print(command)
        self.proc.sendline(command)
        self.proc.expect_exact(command + "\r\n")

        # debugging
        # print(self.proc.before)

        self.proc.sendline()
        self.proc.expect_exact("\r\n")
        try:
            index = self.proc.expect('env": \d+\}', timeout=20)
            output = self.proc.before + self.proc.match.group()
            if verbose:
                print(output)
            return json.loads(output)
        except pexpect.exceptions.TIMEOUT:
            print("FAILED DUE TO TIMEOUT")


def verifier_feedback(ok: str, not_ok: str) -> Optional[str]:
    msg = "Consider previous issue"
    if msg in ok:
        return None
    _, err = calculateScoreHelper(not_ok)
    if err:
        err = err.strip()
        hint = f"\n/- {msg}: {err} -/\n"
        text = ok + hint
        return text
    return None


def calculateScore(msg: str) -> Optional[float]:
    score, _ = calculateScoreHelper(msg)
    return score


def calculateScoreHelper(msg: str) -> (Optional[float], Optional[str]):
    v = filterLean(msg + "```").strip()
    if v == "":
        return None, None
    # hack around the tokenizer not tokenizing '\n\n' as one id
    # if v.endswith('\n') and not v.endswith('\n\n'):
    #     if msg.count('```') % 2 == 1:
    #         return None, None
    r = checkLean(v)
    print(r)  # # DEBUG
    if r["status"] == 0:
        return 1.0, None
    critical_error = -1.0, r["error"]
    tmp_error = None, None
    if "unknown constant" in r["error"]:
        return critical_error
    elif "tactic 'rewrite' failed" in r["error"]:
        return critical_error
    if filterLean(msg).strip() != v:
        num_first_line = r.get('num_line_first', r.get('num_first_line', -1))
        if num_first_line >= v.count('\n'):
        # if r["num_line_first"] >= v.count("\n"):
            return tmp_error
        if "missing cases" in r["error"]:
            return tmp_error
    return critical_error


def score_func(sentence: str) -> Optional[float]:
    print("TEXT")
    print(sentence)
    score = calculateScore(sentence)
    print("SCORE")
    print(score)
    return score


def filterLean(msg: str) -> str:
    m = re.findall("```([Ll]ean4?)?(.*?)```", msg, re.MULTILINE | re.DOTALL)
    r = "\n".join([x[1] for x in m])
    # r = r.replace('\n#eval', '\n--#eval') # skip evaluations
    return r


def getErrorMessage(out: str):
    if "messages" in out:
        for m in out["messages"]:
            if m["severity"] == "error":
                return m
    return None


def checkLean(lean_code_block: str) -> dict:
    path_to_repl = os.environ.get('PATH_TO_LEAN_REPL')
    proofsearch = ProofSearch(path_to_repl=path_to_repl)
    try:
        out = proofsearch.run_code(lean_code_block.strip(), verbose=True)
    except pexpect.exceptions.EOF:
        print('Error usually due to your REPL setting.')
        print('First, check if your configure the path correctly.')
        print('Second, check if you have the mathlib correctly.')
        return {"status": 1, "num_first_line": 0, "error": ""}
    if out:  # failed due to timeout
        error_message = getErrorMessage(out)
        if error_message:
            return {
                "status": 1,
                "num_line_first": error_message["pos"]["line"],
                "error": error_message["data"],
            }
        else:
            return {"status": 0}
    else:
        return {
            "status": 1,
            "num_line_first": 0,
            "error": "Failed due to timeout after 20 seconds",
        }


filter_code = filterLean
check_code = checkLean

if __name__ == "__main__":
#     lean = f"""```lean
# import Mathlib

# def factorial : Nat → Nat
# | 0 => 1
# | n+1 => (n+1) * factorial n

# theorem factorial_pos : ∀ n : Nat, 0 < factorial n
# | 0 => Nat.zero_lt_one
# | n+1 => Nat.mul_pos (Nat.succ_pos n) (factorial_pos n)

# #eval factorial 5
# ```
# """
    lean = f"""```lean
    import data.nat.basic

    def parseAddition (s : String) : Nat :=
    let sumWithCarry := s.foldl
        (fun (sumCarry : Nat × Nat) c =>
        if c = '+' then (sumCarry.1 + sumCarry.2, 0)
        else (sumCarry.1, sumCarry.2 * 10 + (c.toNat - '0'.toNat)))
        (0, 0)
    sumWithCarry.1 + sumWithCarry.2

    #eval parseAddition "1" 
    ```
    """
    print(calculateScoreHelper(lean))