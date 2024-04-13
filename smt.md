### SMT currently supported cases

* Supported case can include the following factors.
    * **trival arithmetic operation** like `[+, -, *, /, %, ^]`, please note that exponent following `^` should be constant number currently, that is `^ x` is not supported yet.
    * **logical operation** like `[<, >, <=, >=, ≠, ∧, ∨, ¬]`
    * **quantifier** like `[∀, ∃]`
    * **Self-defined function** like `(f : ℝ → ℝ)`
    * **Datatype** should be within `[Nat, Real, Rat]`. Please note that `Rat` is not fully supported, `Rat` numbers are forcely converted into `Real` numbers at present, so some operations unique to `Rat` type may not be tranlated correctly, e.g. `xx.den`, `xx.num`.


Some examples that can be resolved by SMT solvers.
```lean
theorem mathd_algebra_37 (x y : ℝ) (h₀ : x + y = 7) (h₁ : 3 * x + y = 45) : x ^ 2 - y ^ 2 = 217 := by 
  smt!

theorem imo_1961_p1 (x y z a b : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x ≠ y) (h₂ : y ≠ z)
    (h₃ : z ≠ x) (h₄ : x + y + z = a) (h₅ : x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆ : x * y = z ^ 2) :
    0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by 
  smt!

theorem mathd_algebra_206 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 + a * x + b) (h₁ : 2 * a ≠ b)
    (h₂ : f (2 * a) = 0) (h₃ : f b = 0) : a + b = -1 := by
  smt!
```

### SMT currently unsupported cases
When including the following factors, SMT may not resolve the theorem automatically,
* **Unsupported datatype**, e.g. `Complex`
```
theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
    ((∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2) * (∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  sorry

theorem mathd_numbertheory_221 (S : Finset ℕ)
    (h₀ : ∀ x : ℕ, x ∈ S ↔ 0 < x ∧ x < 1000 ∧ x.divisors.card = 3) : S.card = 11 := by sorry

theorem numbertheory_xsqpysqintdenomeq (x y : ℚ) (h₀ : (x ^ 2 + y ^ 2).den = 1) : x.den = y.den :=
  by sorry
```
* **Unsupported arithmetic operation**, e.g. `Real.sqrt`, `Real.logb`, `Int.floor` `Nat.xxx` ...
```
theorem imo_2006_p6 (a b c : ℝ) :
    a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2) ≤
      9 * Real.sqrt 2 / 32 * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2 :=
  by sorry

theorem mathd_algebra_151 : Int.ceil (Real.sqrt 27) - Int.floor (Real.sqrt 26) = 1 := by sorry
```
* **Range-related** theorem
```
theorem induction_sum2kp1npqsqm1 (n : ℕ) :
    ∑ k in Finset.range n, (2 * k + 3) = (n + 1) ^ 2 - 1 := by
  sorry

theorem mathd_numbertheory_109 (v : ℕ → ℕ) (h₀ : ∀ n, v n = 2 * n - 1) :
    (∑ k in Finset.Icc 1 100, v k) % 7 = 4 := by
  sorry
```

### TODO list

1. Encode some simple but currently unsupported arithmetic operations,
   * Sqrt
   * Abs (done)
   * Int.floor, Int.ceil
   * Logb, exp(n) --- will try
   * etc
2. Explore range-related theorem
