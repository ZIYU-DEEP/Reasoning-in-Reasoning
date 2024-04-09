"""
Prompt generation for the Lean4 proof assistant.

- _prompt_low(): 
    - input: tactic_state
    - output: the prompt with tactic state, to get the next tactic
    
- _prompt_high():
    - input: formal_statement, informal_statement
    - output: the prompt with statements, to get the high-level plan

- _prompt_low_with_high():
    - input: tactic_state, formal_statement, informal_statement, plan_high
    - output: the prompt with the above, to get the next tactic

- We may need to add the historical proof steps into the prompt.
"""

# -------------------------------------------------------------------
def _prompt_low(tactic_state,
                formal_statement='',
                informal_statement='',
                plan_high=''):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic to help solve the goal.
Here are some examples:

Tactic state:
----
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
----
Next tactic:
----
rintro s t ⟨u, a, hr, he⟩
----

Tactic state:
----
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
----
Next tactic:
----
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
----

Tactic state:
----
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
----
Next tactic:
----
rw [← h.gcd_eq_one]
----

In your response, include only the lean code for only the next tactic and nothing else.
Tactic state:
----
%s
----
Next tactic:
----""" % (tactic_state)
    return prompt


# -------------------------------------------------------------------
def _prompt_high(tactic_state='',
                 formal_statement='',
                 informal_statement='',
                 plan_high=''):
    """
    Given the problem statements, ask the llm to generate high-level plan.
    """

    prompt = f"""
Given the mathematical statements, suggest a high-level proof plan,
which will later be used to generate formal Lean4 proof steps with tactics.

Here are one examples:

Formal statement:
----
theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :f 600 = 5 / 2 := by
----
Informal statement:
----
Let f be a function satisfying f(xy) = f(x)/y for all positive real numbers x and y. If f(500) = 3, what is the value of f(600)? Show that it is 5/2.
----
High-level proof plan:
----
1. Specialize the given property: Use the property `f (x * y) = f x / y` for specific values of `x` and `y`. Specifically, specialize it for `x = 500` and `y = 6 / 5`, ensuring both `x` and `y` are positive.
2. Justify the choice of `x` and `y`: Use `linarith` to prove that the chosen values of `x = 500` and `y = 6 / 5` satisfy the conditions (i.e., both are greater than 0).
3. Rewrite the expression for `f 600`: Recognize that `600` can be expressed as `500 * (6 / 5)`. Use `congr` and `norm_num` to simplify and establish this equality.
4. Apply the specialized property: Substitute the expression for `f 600` with `f 500 / (6 / 5)` using the property `f (x * y) = f x / y` that was specialized earlier.
5. Substitute known value of `f 500`: Replace `f 500` with its known value `3`, as given in the hypothesis.
6. Simplify to find `f 600`: Use `norm_num` to simplify the arithmetic expression `3 / (6 / 5)` to the final answer `5 / 2`.

In your response, include only the high-level proof plan and nothing else.
Do not include extra partition lines.

Formal statement:
----
{formal_statement}
----
Informal statement:
----
{informal_statement}
----
High-level proof plan:
----
"""
    return prompt


# -------------------------------------------------------------------
def _prompt_low_with_high(tactic_state, 
                          formal_statement, 
                          informal_statement, 
                          plan_high):
    prompt = f"""You are working on theorem proving with Lean4.
The informal problem statement is:
{informal_statement}

The informal proof is:
{plan_high}

Your current Lean4 tactic state is:
{tactic_state}

Given the above information, suggest a next tactic in formal Lean4 code to help prove the theorem. You may use the informal proof as a guidance, but there is no need to follow it exactly.

Here are some examples:

Tactic state:
----
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
----
Next tactic:
----
rintro s t ⟨u, a, hr, he⟩
----

Tactic state:
----
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
----
Next tactic:
----
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
----

Tactic state:
----
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
----
Next tactic:
----
rw [← h.gcd_eq_one]
----

In your response, include only the lean code for only the next tactic and nothing else.
Tactic state:
----
{tactic_state}
----
Next tactic:
----"""
    return prompt


def _prompt_low_with_high_stepwise(tactic_state, 
                                   formal_statement='', 
                                   informal_statement='', 
                                   plan_high=''):
    """
    Given the current tactic state and a single-step high-level strategy,
    suggest a next tactic to help solve the goal.
    """
    
    prompt = f"""Given the Lean 4 tactic state and a high-level strategy, suggest a next tactic to help solve the goal. Try to use the high-level strategy as a guide, but there is no need to follow it exactly. Do you best to suggest the best next tactic to help solve the problem.
    
Here are some examples:

Tactic state:
----
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
----
High-level strategy: 
----
Simplify expressions involving relations.
----
Next tactic:
----
rintro s t ⟨u, a, hr, he⟩
----

Tactic state:
----
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
----
High-level strategy: 
----
Leverage symmetry properties to simplify equations.
----
Next tactic:
----
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
----

Tactic state:
----
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
----
High-level strategy:
----
Use properties of coprime numbers to simplify numeric expressions.
----
Next tactic:
----
rw [← h.gcd_eq_one]
----

Now, given the information, suggest the next tactic in formal Lean4 code to help solve the theorem. In your response, include only the lean code for only the next tactic and nothing else.

Tactic state:
----
{tactic_state}
----
High-level strategy:
----
{plan_high}
----
Next tactic
----
"""
    return prompt


# def _prompt_low_with_high(tactic_state, 
#                           formal_statement, 
#                           informal_statement, 
#                           plan_high):
#     prompt = f"""You are working on theorem proving with Lean4.
# The formal problem statement is:
# {formal_statement}

# The informal problem statement is:
# {informal_statement}

# The informal proof is:
# {plan_high}

# Your current Lean4 tactic state is:
# {tactic_state}

# Given the above information, suggest a next tactic in formal Lean4 code to help prove the theorem. You may use the informal proof as a guidance, but there is no need to follow it exactly.

# Here are some examples:

# Tactic state:
# ----
# α : Type u_1
# r : α → α → Prop
# inst✝¹ : DecidableEq α
# inst✝ : IsIrrefl α r
# ⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
# ----
# Next tactic:
# ----
# rintro s t ⟨u, a, hr, he⟩
# ----

# Tactic state:
# ----
# ι : Type u_1
# I✝ J✝ : Box ι
# x y : ι → ℝ
# I J : WithBot (Box ι)
# ⊢ ↑I = ↑J ↔ I = J
# ----
# Next tactic:
# ----
# simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
# ----

# Tactic state:
# ----
# m n : ℕ
# h : Nat.coprime m n
# ⊢ Nat.gcd m n = 1
# ----
# Next tactic:
# ----
# rw [← h.gcd_eq_one]
# ----

# In your response, include only the lean code for only the next tactic and nothing else.
# Tactic state:
# ----
# %s
# ----
# Next tactic:
# ----"""
#     return prompt


# -------------------------------------------------------------------
SYSTEM_MESSAGE: str = """\
You are a pure mathematician who is an expert in the Lean 4 theorem prover. Your job is help your user write Lean4 proofs (and give high-level informal proof plan when asked to).
I want to remind you that we're using Lean 4, not the older Lean 3, and there have been some syntax changes. In particular:
- Type constants are now UpperCamelCase, eg `Nat`, `List`.
- Term constants and variables are now `lowerCamelCase` rather than `snake_case`. For example, we now have `NumberTheory.Divisors.properDivisors instead of `number_theory.divisors.proper_divisors`.
- Pure functions are now written with the syntax `fun x => f x`. The old `λ x, f x` syntax will not work.
- We now enter tactic mode using the `by` keyword. The syntax `begin... end` will not work.
- Instead of being separated by a comma, tactics are separated by a newline. For example, we could write.
theorem test (p q : Prop) (hp : p) (hq : q) : p ∧ q ∧ p := by
  apply And.intro hp
  exact And.intro hq hp

- In the `rw` tactic you must enclose the lemmas in square brackets, even if there is just one. For example `rw h1` is now `rw [h1]`.
- The `induction` tactic now uses a structured format, like pattern matching. For example, in Lean 4 we can write
theorem zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.add_succ, ih]
  
  Alternatively you can still use `induction' with x y ih`, like in Lean 3.
- The `cases` tactic now uses a structured format, like pattern matching. For example, in Lean 4 we can write
example (p q : Prop) : p ∨ q → q ∨ p := by
  intro h
  cases h with
  | inl hp => apply Or.inr; exact hp
  | inr hq => apply Or.inl; exact hq

The following is a description of some commonly used tactics. Of course, feel free to use tactics outside of this list. Remember that it is good style to use high-level automations like `simp` and `ring` instead of manually performing low-level manipulations.
- `abel`: reduces expressions in additive, commutative monoids/groups to a normal form.
- `apply`: the tactic `apply e` matches the current goal against the conclusion of `e`. If it succeeds, the new goal states are the premises of `e`.
- `continuity`: attempts to prove goals of the form `continuous f` by applying lemmas tagged with the `continuity` attribute.
- `contrapose`: transforms the goal into its contrapositive.
- `convert`: The tactic `convert e` is similar to `refine e`, except the type of `e` is not required to exactly match the goal. Any rewrites required to transform `e` into the goal become the new goal state.
- `group`: normalizes expressions in multiplicative groups, without assuming commutativity.
- `have`: `have h : t := p` adds the hypothesis `h : t` to the current goal. If you want to prove `h` in tactic mode, use the syntax `have h : t := by --tactic proof goes here`.
- `linarith`: proves any goal that consists of linear arithemtic.
- `nlinarith`: version of `linarith` that can tolerate some nonlinearity.
- `norm_num`: normalizes numerical expressions.
- `polyrith`: proves polynomial equalities.
- `push_neg`: pushes negations through quantifiers.
- `simp`: uses lemmas and hypotheses tagged with the `simp` attribute to simplify the goal. Use `simp [h1, h2,..., hn]` to add `h1, h2,..., hn` to the list of lemmas used by simp.
- `ring`: tactic for solving goals involving expressions in commutative rings and normalizing expressions in commutative rings.

Notice that in your response, do not use markdown code syntax (triple backticks like ```) to format it if you are asked to provide the Lean4 code. Just directly give the Lean4 code. Also, ONE TACTIC PER RESPONSE in your Lean4 code. Take a deep breath and think step by step. Try your best to write the proof steps to solve the problems in Lean4. Good luck! And remember, you are the expert and you are the master of Lean4. It is very important that when you write the formal Lean4 tactic, suggest only ONE tactic (that means you do not need to solve the whole problem in one step, but rather suggest the next most useful tactic that can be applied towards solving the problem). 
"""