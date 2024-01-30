from .lang_config import LANG

if LANG == "Dafny":
    from dafny import (
        score_func,
        verifier_feedback,
        short_verifier_feedback,
        filter_code,
        check_code,
    )
elif LANG == "Coq":
    from coq import (
        score_func,
        verifier_feedback,
        short_verifier_feedback,
        filter_code,
        check_code,
    )
elif LANG == "Lean4":
    from lean import (
        score_func,
        verifier_feedback,
        filter_code,
        check_code,
    )
elif LANG == "Rust":
    from rust import (
        score_func,
        verifier_feedback,
        filter_code,
        check_code,
    )
elif LANG == "Scala":
    from scala import (
        score_func,
        verifier_feedback,
        filter_code,
        check_code,
    )
else:
    assert False

def can_be_solution(msg: str, min_lines: int, check_func=None) -> bool:
    if not (msg.count("```") % 2 == 0):
        return False
    v = filter_code(msg)
    r = v.count("\n") >= min_lines
    if r and check_func:
        r = check_func(v)
    return r

def find_largest_new_block(old_text: str, text: str) -> str:
    return find_largest_new_block_code(
        filter_code(old_text + "```").strip(), filter_code(text + "```").strip()
    )


def find_largest_new_block_code(old_code: str, code: str) -> str:
    while len(old_code) < len(code):
        r = check_code(code)
        if r["status"] == 0:
            return code
        try:
            code = code[0 : code.rindex(stop_word)].strip()
        except ValueError:
            return None
    return None
