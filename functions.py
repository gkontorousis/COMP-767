from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

#y=m*x+b
def artihmetic(m,b):
    i=0
    while True:
        yield m * i + b
        i+=1

#y=a*x+b
def geometric(a,b):
    i=1
    while True:
        yield a * i + b
        i*=a

# it is known
def fibonacci(a=0,b=1):
    while True:
        yield a
        a,b=b,a+b

# 2 4 7 8 10 13 14
def times2_add3_every3digits():
    i=1
    while True:
        if i%3==0:
            yield i*2+1
        else:
            yield i*2
        i+=1

# 1 4 9 16 25
def prime_numbers():
    primes = []
    num = 2
    while True:
        is_prime = True
        for prime in primes:
            if prime * prime > num:
                break
            if num % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
            yield num
        num += 1

# 0 2 6 12 20
def square_numbers_minus_i():
    i=1
    while True:
        yield i*(i-1)
        i+=1

# 1 3 7 15 31
def powers_of_2_minus_i():
    i=0
    while True:
        yield 2**i - i
        i+=1

# 1 22 333 4444 55555
def repeated_digits():
    i=1
    while True:
        yield int(str(i)*i)
        i+=1
    

# this ones kinda hard lol
# 11 21 1211 111221 312211
def look_and_say():
    current = "1"
    while True:
        yield current
        next_seq = ""
        count = 1
        for j in range(1, len(current)):
            if current[j] == current[j-1]:
                count += 1
            else:
                next_seq += str(count) + current[j-1]
                count = 1
        next_seq += str(count) + current[-1]
        current = next_seq

# plus 2, plus 3, plus 4, plus 3, plus 2
def plus_2_3_4_3_2():
    i=0
    increments = [2, 3, 4, 3, 2]
    while True:
        yield i
        i += increments[i % len(increments)]


# start at 11 increase by fibonacci numbers
def fibonacci_starting_at_11():
    fib = fibonacci()
    current = 11
    while True:
        yield current
        current += next(fib)


# sum of prime numers squared
# 2^2, 2^2*3^2, 2^2*3^2*5^2, 2^2*3^2*5^2*7^2
def sum_of_prime_squares():
    primes = prime_numbers()
    current_sum = 1
    while True:
        prime = next(primes)
        current_sum *= prime ** 2
        yield current_sum

# good one!
# 2, 2^2*3, 2^2*3^2*5, 2^2*3^2*5^2*7, 2^2*3^2*5^2*7^2*11
def sum_of_prime_squares_v2():
    primes = prime_numbers()
    current_product = 1
    while True:
        prime = next(primes)
        current_product *= prime 
        yield current_product
        current_product *= prime


# odd positoins: (2n)^2, even positions: 2n
def odd_even_squares():
    i = 1
    while True:
        if i % 2 == 1:
            yield (2 * i) ** 2
        else:
            yield 2 * i
        i += 1

# mod3==0:(2n-1)^2-1, mod3==1: 2n^2-1, mod3==2: (2n+1)^2-1
def hard_v1():
    i = 1
    while True:
        if i % 3 == 0:
            yield (2 * i - 1) ** 2 - 1
        elif i % 3 == 1:
            yield 2 * i ** 2 - 1 
        else:
            yield (2 * i + 1) ** 2 - 1
        i += 1

# 2n+look_and_say(n)
def very_hard_v1():
    look_and_say_gen = look_and_say()
    i = 1
    while True:
        yield 2 * i + int(next(look_and_say_gen))
        i += 1


# simple closed-form generators used by training.py
def square_numbers():
    i = 1
    while True:
        yield i * i
        i += 1


def powers_of_two():
    i = 1
    while True:
        yield 2 ** i
        i += 1


def triangular_numbers():
    i = 1
    while True:
        yield i * (i + 1) // 2
        i += 1


def one_plus_triangular_numbers():
    i = 1
    while True:
        yield 1 + i * (i + 1) // 2
        i += 1


@dataclass(frozen=True)
class RuleGeneratorSpec:
    key: str
    generator_factory: Callable[[], Iterator[int]]
    rule_text: str


DEFAULT_RULE_SPECS: tuple[RuleGeneratorSpec, ...] = (
    RuleGeneratorSpec("artihmetic", lambda: artihmetic(3, 1), "y = 3n + 1"),
    RuleGeneratorSpec("geometric", lambda: geometric(2, 0), "geometric progression with ratio 2"),
    RuleGeneratorSpec("powers_of_2_minus_i", powers_of_2_minus_i, "2^n - n"),
    RuleGeneratorSpec("square_numbers", square_numbers, "n^2"),
    RuleGeneratorSpec("powers_of_two", powers_of_two, "2^(n+1)"),
    RuleGeneratorSpec("triangular_numbers", triangular_numbers, "n(n+1)/2"),
    RuleGeneratorSpec("one_plus_triangular_numbers", one_plus_triangular_numbers, "1 + n(n+1)/2"),
    RuleGeneratorSpec("square_numbers_minus_i", square_numbers_minus_i, "n(n-1)"),
    RuleGeneratorSpec("fibonacci", fibonacci, "fibonacci numbers"),
    RuleGeneratorSpec("prime_numbers", prime_numbers, "prime numbers"),
    RuleGeneratorSpec("repeated_digits", repeated_digits, "repeat digit n exactly n times"),
    RuleGeneratorSpec("plus_2_3_4_3_2", plus_2_3_4_3_2, "add cycle [2, 3, 4, 3, 2]"),
    RuleGeneratorSpec("times2_add3_every3digits", times2_add3_every3digits, "2n, except every 3rd term uses 2n+1"),
    RuleGeneratorSpec("fibonacci_starting_at_11", fibonacci_starting_at_11, "start at 11, then add successive Fibonacci numbers"),
    RuleGeneratorSpec("sum_of_prime_squares", sum_of_prime_squares, "cumulative product of squared primes"),
    RuleGeneratorSpec("sum_of_prime_squares_v2", sum_of_prime_squares_v2, "multiply by each prime twice before moving to next prime"),
    RuleGeneratorSpec("odd_even_squares", odd_even_squares, "odd positions: (2n)^2, even positions: 2n"),
    RuleGeneratorSpec("hard_v1", hard_v1, "piecewise quadratic by index modulo 3"),
    RuleGeneratorSpec("look_and_say", look_and_say, "look-and-say sequence"),
    # comment this one out as it results in ungodly large sequences (in terms of tokens) and led to OOM errors even with 80GB VRAM
    # RuleGeneratorSpec("very_hard_v1", very_hard_v1, "2n + look-and-say(n)"),
)


def _take_window(generator_factory: Callable[[], Iterator[int]], start_index: int, length: int) -> list[int]:
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    gen = generator_factory()
    for _ in range(start_index):
        next(gen)
    return [next(gen) for _ in range(length)]


def build_train_data(
    samples_per_rule: int = 3,
    sequence_length: int = 10,
    min_start_index: int = 0,
    max_start_index: int = 40,
    seed: int | None = 42,
    specs: Iterable[RuleGeneratorSpec] | None = None,
) -> list[tuple[list[int], str]]:
    """
    Build `(observed_sequence, gold_rule)` pairs from generator specs.

    Each rule contributes `samples_per_rule` windows from different start offsets,
    so the same underlying rule appears as multiple sequences.
    """
    if samples_per_rule <= 0:
        raise ValueError(f"samples_per_rule must be > 0, got {samples_per_rule}")
    if max_start_index < min_start_index:
        raise ValueError(
            "max_start_index must be >= min_start_index, got "
            f"{max_start_index} < {min_start_index}"
        )

    chosen_specs = tuple(DEFAULT_RULE_SPECS if specs is None else specs)
    rng = random.Random(seed)
    start_candidates = list(range(min_start_index, max_start_index + 1))

    train_data: list[tuple[list[int], str]] = []
    for spec in chosen_specs:
        if samples_per_rule <= len(start_candidates):
            starts = rng.sample(start_candidates, k=samples_per_rule)
        else:
            starts = [rng.choice(start_candidates) for _ in range(samples_per_rule)]

        for start in starts:
            seq = _take_window(spec.generator_factory, start, sequence_length)
            train_data.append((seq, spec.rule_text))

    rng.shuffle(train_data)
    return train_data


if __name__ == "__main__":
    gen = hard_v1()
    for _ in range(10):
        print(next(gen))



