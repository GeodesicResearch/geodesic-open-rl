#!/usr/bin/env python3
"""Generate a trivially simple Python debug dataset for RL-Zero code training.

Creates ~256 problems that a base LLM (e.g. OLMo-3-7B) should be able to solve
without any fine-tuning. Pushes to HuggingFace as geodesic-research/debug-code-rlzero.

Schema matches Dolci-RLZero-Code exactly:
    id: str, prompt: str, solution: str, ground_truth: list[str]

Usage:
    uv run python scripts/create_debug_dataset.py
    uv run python scripts/create_debug_dataset.py --dry-run  # print without pushing
"""

import argparse
import json

from datasets import Dataset


def make_problems():
    """Generate all problems as a list of dicts."""
    problems = []
    pid = 0

    def add(name, prompt, solution, tests):
        nonlocal pid
        problems.append(
            {"id": f"debug_{pid:03d}", "prompt": f"user: {prompt}", "solution": solution, "ground_truth": tests}
        )
        pid += 1

    # ── Category 1: Arithmetic ──────────────────────────────────────────
    add(
        "add",
        "Write a Python function called `add` that takes two integers and returns their sum.",
        "def add(a, b):\n    return a + b",
        ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0", "assert add(100, 200) == 300"],
    )

    add(
        "subtract",
        "Write a Python function called `subtract` that takes two integers and returns the first minus the second.",
        "def subtract(a, b):\n    return a - b",
        [
            "assert subtract(5, 3) == 2",
            "assert subtract(0, 0) == 0",
            "assert subtract(-1, 1) == -2",
            "assert subtract(10, 20) == -10",
        ],
    )

    add(
        "multiply",
        "Write a Python function called `multiply` that takes two integers and returns their product.",
        "def multiply(a, b):\n    return a * b",
        [
            "assert multiply(3, 4) == 12",
            "assert multiply(0, 5) == 0",
            "assert multiply(-2, 3) == -6",
            "assert multiply(7, 7) == 49",
        ],
    )

    add(
        "integer_divide",
        "Write a Python function called `integer_divide` that takes two integers and returns the integer division (floor division) of the first by the second.",
        "def integer_divide(a, b):\n    return a // b",
        [
            "assert integer_divide(7, 2) == 3",
            "assert integer_divide(10, 5) == 2",
            "assert integer_divide(1, 3) == 0",
            "assert integer_divide(-7, 2) == -4",
        ],
    )

    add(
        "modulo",
        "Write a Python function called `modulo` that takes two integers and returns the remainder when the first is divided by the second.",
        "def modulo(a, b):\n    return a % b",
        [
            "assert modulo(7, 3) == 1",
            "assert modulo(10, 5) == 0",
            "assert modulo(13, 4) == 1",
            "assert modulo(100, 7) == 2",
        ],
    )

    add(
        "power",
        "Write a Python function called `power` that takes two integers `base` and `exp` and returns base raised to the power of exp.",
        "def power(base, exp):\n    return base ** exp",
        [
            "assert power(2, 3) == 8",
            "assert power(5, 0) == 1",
            "assert power(3, 2) == 9",
            "assert power(10, 4) == 10000",
        ],
    )

    add(
        "absolute_value",
        "Write a Python function called `absolute_value` that takes an integer and returns its absolute value.",
        "def absolute_value(n):\n    return abs(n)",
        [
            "assert absolute_value(-5) == 5",
            "assert absolute_value(5) == 5",
            "assert absolute_value(0) == 0",
            "assert absolute_value(-100) == 100",
        ],
    )

    add(
        "max_of_two",
        "Write a Python function called `max_of_two` that takes two integers and returns the larger one.",
        "def max_of_two(a, b):\n    return max(a, b)",
        [
            "assert max_of_two(3, 5) == 5",
            "assert max_of_two(5, 3) == 5",
            "assert max_of_two(4, 4) == 4",
            "assert max_of_two(-1, -2) == -1",
        ],
    )

    add(
        "min_of_two",
        "Write a Python function called `min_of_two` that takes two integers and returns the smaller one.",
        "def min_of_two(a, b):\n    return min(a, b)",
        [
            "assert min_of_two(3, 5) == 3",
            "assert min_of_two(5, 3) == 3",
            "assert min_of_two(4, 4) == 4",
            "assert min_of_two(-1, -2) == -2",
        ],
    )

    add(
        "negate",
        "Write a Python function called `negate` that takes an integer and returns its negation.",
        "def negate(n):\n    return -n",
        ["assert negate(5) == -5", "assert negate(-3) == 3", "assert negate(0) == 0"],
    )

    add(
        "double",
        "Write a Python function called `double` that takes an integer and returns twice its value.",
        "def double(n):\n    return n * 2",
        ["assert double(5) == 10", "assert double(0) == 0", "assert double(-3) == -6"],
    )

    add(
        "square",
        "Write a Python function called `square` that takes an integer and returns its square.",
        "def square(n):\n    return n * n",
        ["assert square(3) == 9", "assert square(0) == 0", "assert square(-4) == 16", "assert square(7) == 49"],
    )

    add(
        "cube",
        "Write a Python function called `cube` that takes an integer and returns its cube.",
        "def cube(n):\n    return n ** 3",
        ["assert cube(2) == 8", "assert cube(3) == 27", "assert cube(0) == 0", "assert cube(-2) == -8"],
    )

    add(
        "average_two",
        "Write a Python function called `average_two` that takes two numbers and returns their average as a float.",
        "def average_two(a, b):\n    return (a + b) / 2",
        ["assert average_two(2, 4) == 3.0", "assert average_two(1, 2) == 1.5", "assert average_two(0, 0) == 0.0"],
    )

    add(
        "is_divisible",
        "Write a Python function called `is_divisible` that takes two integers `a` and `b` and returns True if `a` is divisible by `b`.",
        "def is_divisible(a, b):\n    return a % b == 0",
        [
            "assert is_divisible(10, 5) == True",
            "assert is_divisible(10, 3) == False",
            "assert is_divisible(0, 1) == True",
            "assert is_divisible(15, 3) == True",
        ],
    )

    add(
        "sum_three",
        "Write a Python function called `sum_three` that takes three integers and returns their sum.",
        "def sum_three(a, b, c):\n    return a + b + c",
        ["assert sum_three(1, 2, 3) == 6", "assert sum_three(0, 0, 0) == 0", "assert sum_three(-1, 0, 1) == 0"],
    )

    add(
        "increment",
        "Write a Python function called `increment` that takes an integer and returns it plus one.",
        "def increment(n):\n    return n + 1",
        ["assert increment(0) == 1", "assert increment(-1) == 0", "assert increment(99) == 100"],
    )

    add(
        "decrement",
        "Write a Python function called `decrement` that takes an integer and returns it minus one.",
        "def decrement(n):\n    return n - 1",
        ["assert decrement(1) == 0", "assert decrement(0) == -1", "assert decrement(100) == 99"],
    )

    add(
        "difference",
        "Write a Python function called `difference` that takes two integers and returns the absolute difference between them.",
        "def difference(a, b):\n    return abs(a - b)",
        ["assert difference(5, 3) == 2", "assert difference(3, 5) == 2", "assert difference(7, 7) == 0"],
    )

    add(
        "clamp",
        "Write a Python function called `clamp` that takes three integers: `value`, `low`, and `high`, and returns `value` clamped to the range [low, high].",
        "def clamp(value, low, high):\n    return max(low, min(value, high))",
        [
            "assert clamp(5, 0, 10) == 5",
            "assert clamp(-5, 0, 10) == 0",
            "assert clamp(15, 0, 10) == 10",
            "assert clamp(0, 0, 10) == 0",
        ],
    )

    add(
        "sum_of_squares",
        "Write a Python function called `sum_of_squares` that takes two integers and returns the sum of their squares.",
        "def sum_of_squares(a, b):\n    return a**2 + b**2",
        ["assert sum_of_squares(3, 4) == 25", "assert sum_of_squares(0, 0) == 0", "assert sum_of_squares(1, 1) == 2"],
    )

    add(
        "hypotenuse_squared",
        "Write a Python function called `hypotenuse_squared` that takes two integers (legs of a right triangle) and returns the square of the hypotenuse.",
        "def hypotenuse_squared(a, b):\n    return a**2 + b**2",
        [
            "assert hypotenuse_squared(3, 4) == 25",
            "assert hypotenuse_squared(5, 12) == 169",
            "assert hypotenuse_squared(0, 0) == 0",
        ],
    )

    # ── Category 2: String basics ───────────────────────────────────────
    add(
        "reverse_string",
        "Write a Python function called `reverse_string` that takes a string and returns it reversed.",
        "def reverse_string(s):\n    return s[::-1]",
        [
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('') == ''",
            "assert reverse_string('a') == 'a'",
            "assert reverse_string('abcd') == 'dcba'",
        ],
    )

    add(
        "string_length",
        "Write a Python function called `string_length` that takes a string and returns its length.",
        "def string_length(s):\n    return len(s)",
        ["assert string_length('hello') == 5", "assert string_length('') == 0", "assert string_length('a') == 1"],
    )

    add(
        "to_upper",
        "Write a Python function called `to_upper` that takes a string and returns it in uppercase.",
        "def to_upper(s):\n    return s.upper()",
        [
            "assert to_upper('hello') == 'HELLO'",
            "assert to_upper('Hello World') == 'HELLO WORLD'",
            "assert to_upper('') == ''",
        ],
    )

    add(
        "to_lower",
        "Write a Python function called `to_lower` that takes a string and returns it in lowercase.",
        "def to_lower(s):\n    return s.lower()",
        [
            "assert to_lower('HELLO') == 'hello'",
            "assert to_lower('Hello World') == 'hello world'",
            "assert to_lower('') == ''",
        ],
    )

    add(
        "concatenate",
        "Write a Python function called `concatenate` that takes two strings and returns them concatenated.",
        "def concatenate(a, b):\n    return a + b",
        [
            "assert concatenate('hello', ' world') == 'hello world'",
            "assert concatenate('', 'test') == 'test'",
            "assert concatenate('a', 'b') == 'ab'",
        ],
    )

    add(
        "first_char",
        "Write a Python function called `first_char` that takes a non-empty string and returns its first character.",
        "def first_char(s):\n    return s[0]",
        ["assert first_char('hello') == 'h'", "assert first_char('a') == 'a'", "assert first_char('xyz') == 'x'"],
    )

    add(
        "last_char",
        "Write a Python function called `last_char` that takes a non-empty string and returns its last character.",
        "def last_char(s):\n    return s[-1]",
        ["assert last_char('hello') == 'o'", "assert last_char('a') == 'a'", "assert last_char('xyz') == 'z'"],
    )

    add(
        "repeat_string",
        "Write a Python function called `repeat_string` that takes a string and an integer n and returns the string repeated n times.",
        "def repeat_string(s, n):\n    return s * n",
        [
            "assert repeat_string('ab', 3) == 'ababab'",
            "assert repeat_string('x', 5) == 'xxxxx'",
            "assert repeat_string('hi', 0) == ''",
        ],
    )

    add(
        "starts_with",
        "Write a Python function called `starts_with` that takes two strings `s` and `prefix` and returns True if `s` starts with `prefix`.",
        "def starts_with(s, prefix):\n    return s.startswith(prefix)",
        [
            "assert starts_with('hello', 'he') == True",
            "assert starts_with('hello', 'lo') == False",
            "assert starts_with('', '') == True",
        ],
    )

    add(
        "ends_with",
        "Write a Python function called `ends_with` that takes two strings `s` and `suffix` and returns True if `s` ends with `suffix`.",
        "def ends_with(s, suffix):\n    return s.endswith(suffix)",
        [
            "assert ends_with('hello', 'lo') == True",
            "assert ends_with('hello', 'he') == False",
            "assert ends_with('', '') == True",
        ],
    )

    add(
        "contains_substring",
        "Write a Python function called `contains_substring` that takes two strings `s` and `sub` and returns True if `sub` is found in `s`.",
        "def contains_substring(s, sub):\n    return sub in s",
        [
            "assert contains_substring('hello world', 'world') == True",
            "assert contains_substring('hello', 'xyz') == False",
            "assert contains_substring('abc', '') == True",
        ],
    )

    add(
        "count_char",
        "Write a Python function called `count_char` that takes a string and a character and returns how many times the character appears in the string.",
        "def count_char(s, c):\n    return s.count(c)",
        [
            "assert count_char('hello', 'l') == 2",
            "assert count_char('hello', 'z') == 0",
            "assert count_char('aaa', 'a') == 3",
        ],
    )

    add(
        "replace_char",
        "Write a Python function called `replace_char` that takes a string, an old character, and a new character, and returns the string with all occurrences of the old character replaced by the new character.",
        "def replace_char(s, old, new):\n    return s.replace(old, new)",
        [
            "assert replace_char('hello', 'l', 'r') == 'herro'",
            "assert replace_char('aaa', 'a', 'b') == 'bbb'",
            "assert replace_char('xyz', 'a', 'b') == 'xyz'",
        ],
    )

    add(
        "strip_whitespace",
        "Write a Python function called `strip_whitespace` that takes a string and returns it with leading and trailing whitespace removed.",
        "def strip_whitespace(s):\n    return s.strip()",
        [
            "assert strip_whitespace('  hello  ') == 'hello'",
            "assert strip_whitespace('hello') == 'hello'",
            "assert strip_whitespace('  ') == ''",
        ],
    )

    add(
        "split_words",
        "Write a Python function called `split_words` that takes a string and returns a list of words split by spaces.",
        "def split_words(s):\n    return s.split()",
        [
            "assert split_words('hello world') == ['hello', 'world']",
            "assert split_words('a b c') == ['a', 'b', 'c']",
            "assert split_words('single') == ['single']",
        ],
    )

    add(
        "join_words",
        "Write a Python function called `join_words` that takes a list of strings and returns them joined by a space.",
        "def join_words(words):\n    return ' '.join(words)",
        [
            "assert join_words(['hello', 'world']) == 'hello world'",
            "assert join_words(['a']) == 'a'",
            "assert join_words([]) == ''",
        ],
    )

    add(
        "capitalize_first",
        "Write a Python function called `capitalize_first` that takes a string and returns it with the first character capitalized and the rest lowercase.",
        "def capitalize_first(s):\n    return s.capitalize()",
        [
            "assert capitalize_first('hello') == 'Hello'",
            "assert capitalize_first('HELLO') == 'Hello'",
            "assert capitalize_first('') == ''",
        ],
    )

    add(
        "is_alpha",
        "Write a Python function called `is_alpha` that takes a string and returns True if all characters are alphabetic and the string is non-empty.",
        "def is_alpha(s):\n    return s.isalpha()",
        ["assert is_alpha('hello') == True", "assert is_alpha('hello123') == False", "assert is_alpha('') == False"],
    )

    add(
        "is_digit_string",
        "Write a Python function called `is_digit_string` that takes a string and returns True if all characters are digits and the string is non-empty.",
        "def is_digit_string(s):\n    return s.isdigit()",
        [
            "assert is_digit_string('123') == True",
            "assert is_digit_string('12a') == False",
            "assert is_digit_string('') == False",
        ],
    )

    add(
        "char_at_index",
        "Write a Python function called `char_at_index` that takes a string and an integer index and returns the character at that index.",
        "def char_at_index(s, i):\n    return s[i]",
        [
            "assert char_at_index('hello', 0) == 'h'",
            "assert char_at_index('hello', 4) == 'o'",
            "assert char_at_index('abc', 1) == 'b'",
        ],
    )

    # ── Category 3: List basics ─────────────────────────────────────────
    add(
        "sum_list",
        "Write a Python function called `sum_list` that takes a list of integers and returns their sum.",
        "def sum_list(lst):\n    return sum(lst)",
        [
            "assert sum_list([1, 2, 3]) == 6",
            "assert sum_list([]) == 0",
            "assert sum_list([-1, 1]) == 0",
            "assert sum_list([10]) == 10",
        ],
    )

    add(
        "max_list",
        "Write a Python function called `max_list` that takes a non-empty list of integers and returns the maximum value.",
        "def max_list(lst):\n    return max(lst)",
        ["assert max_list([1, 2, 3]) == 3", "assert max_list([-1, -2, -3]) == -1", "assert max_list([5]) == 5"],
    )

    add(
        "min_list",
        "Write a Python function called `min_list` that takes a non-empty list of integers and returns the minimum value.",
        "def min_list(lst):\n    return min(lst)",
        ["assert min_list([1, 2, 3]) == 1", "assert min_list([-1, -2, -3]) == -3", "assert min_list([5]) == 5"],
    )

    add(
        "list_length",
        "Write a Python function called `list_length` that takes a list and returns its length.",
        "def list_length(lst):\n    return len(lst)",
        ["assert list_length([1, 2, 3]) == 3", "assert list_length([]) == 0", "assert list_length([1]) == 1"],
    )

    add(
        "count_occurrences",
        "Write a Python function called `count_occurrences` that takes a list and a value and returns how many times the value appears in the list.",
        "def count_occurrences(lst, val):\n    return lst.count(val)",
        [
            "assert count_occurrences([1, 2, 2, 3], 2) == 2",
            "assert count_occurrences([1, 2, 3], 4) == 0",
            "assert count_occurrences([], 1) == 0",
        ],
    )

    add(
        "contains_element",
        "Write a Python function called `contains_element` that takes a list and a value and returns True if the value is in the list.",
        "def contains_element(lst, val):\n    return val in lst",
        [
            "assert contains_element([1, 2, 3], 2) == True",
            "assert contains_element([1, 2, 3], 4) == False",
            "assert contains_element([], 1) == False",
        ],
    )

    add(
        "first_element",
        "Write a Python function called `first_element` that takes a non-empty list and returns its first element.",
        "def first_element(lst):\n    return lst[0]",
        [
            "assert first_element([1, 2, 3]) == 1",
            "assert first_element(['a', 'b']) == 'a'",
            "assert first_element([42]) == 42",
        ],
    )

    add(
        "last_element",
        "Write a Python function called `last_element` that takes a non-empty list and returns its last element.",
        "def last_element(lst):\n    return lst[-1]",
        [
            "assert last_element([1, 2, 3]) == 3",
            "assert last_element(['a', 'b']) == 'b'",
            "assert last_element([42]) == 42",
        ],
    )

    add(
        "reverse_list",
        "Write a Python function called `reverse_list` that takes a list and returns a new list with the elements in reverse order.",
        "def reverse_list(lst):\n    return lst[::-1]",
        [
            "assert reverse_list([1, 2, 3]) == [3, 2, 1]",
            "assert reverse_list([]) == []",
            "assert reverse_list([1]) == [1]",
        ],
    )

    add(
        "sort_list",
        "Write a Python function called `sort_list` that takes a list of integers and returns a new sorted list in ascending order.",
        "def sort_list(lst):\n    return sorted(lst)",
        [
            "assert sort_list([3, 1, 2]) == [1, 2, 3]",
            "assert sort_list([]) == []",
            "assert sort_list([1]) == [1]",
            "assert sort_list([5, 3, 8, 1]) == [1, 3, 5, 8]",
        ],
    )

    add(
        "append_element",
        "Write a Python function called `append_element` that takes a list and a value and returns a new list with the value appended.",
        "def append_element(lst, val):\n    return lst + [val]",
        ["assert append_element([1, 2], 3) == [1, 2, 3]", "assert append_element([], 1) == [1]"],
    )

    add(
        "flatten_list",
        "Write a Python function called `flatten_list` that takes a list of lists and returns a single flat list with all elements.",
        "def flatten_list(lst):\n    return [x for sub in lst for x in sub]",
        [
            "assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]",
            "assert flatten_list([[], [1]]) == [1]",
            "assert flatten_list([]) == []",
        ],
    )

    add(
        "remove_duplicates",
        "Write a Python function called `remove_duplicates` that takes a list and returns a new list with duplicates removed, preserving order.",
        "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        [
            "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]",
            "assert remove_duplicates([]) == []",
            "assert remove_duplicates([1, 1, 1]) == [1]",
        ],
    )

    add(
        "zip_lists",
        "Write a Python function called `zip_lists` that takes two lists of equal length and returns a list of tuples pairing elements from each list.",
        "def zip_lists(a, b):\n    return list(zip(a, b))",
        ["assert zip_lists([1, 2], ['a', 'b']) == [(1, 'a'), (2, 'b')]", "assert zip_lists([], []) == []"],
    )

    add(
        "list_product",
        "Write a Python function called `list_product` that takes a list of integers and returns their product. Return 1 for an empty list.",
        "def list_product(lst):\n    result = 1\n    for x in lst:\n        result *= x\n    return result",
        [
            "assert list_product([1, 2, 3]) == 6",
            "assert list_product([]) == 1",
            "assert list_product([5]) == 5",
            "assert list_product([2, 3, 4]) == 24",
        ],
    )

    add(
        "take_first_n",
        "Write a Python function called `take_first_n` that takes a list and an integer n and returns the first n elements.",
        "def take_first_n(lst, n):\n    return lst[:n]",
        [
            "assert take_first_n([1, 2, 3, 4], 2) == [1, 2]",
            "assert take_first_n([1, 2, 3], 5) == [1, 2, 3]",
            "assert take_first_n([], 3) == []",
        ],
    )

    add(
        "filter_positive",
        "Write a Python function called `filter_positive` that takes a list of integers and returns a new list containing only the positive numbers.",
        "def filter_positive(lst):\n    return [x for x in lst if x > 0]",
        [
            "assert filter_positive([1, -2, 3, -4, 5]) == [1, 3, 5]",
            "assert filter_positive([-1, -2]) == []",
            "assert filter_positive([]) == []",
        ],
    )

    add(
        "double_list",
        "Write a Python function called `double_list` that takes a list of integers and returns a new list where each element is doubled.",
        "def double_list(lst):\n    return [x * 2 for x in lst]",
        [
            "assert double_list([1, 2, 3]) == [2, 4, 6]",
            "assert double_list([]) == []",
            "assert double_list([0, -1]) == [0, -2]",
        ],
    )

    add(
        "index_of",
        "Write a Python function called `index_of` that takes a list and a value and returns the index of the first occurrence of the value, or -1 if not found.",
        "def index_of(lst, val):\n    try:\n        return lst.index(val)\n    except ValueError:\n        return -1",
        ["assert index_of([1, 2, 3], 2) == 1", "assert index_of([1, 2, 3], 4) == -1", "assert index_of([], 1) == -1"],
    )

    add(
        "average_list",
        "Write a Python function called `average_list` that takes a non-empty list of numbers and returns their average as a float.",
        "def average_list(lst):\n    return sum(lst) / len(lst)",
        [
            "assert average_list([1, 2, 3]) == 2.0",
            "assert average_list([4]) == 4.0",
            "assert average_list([1, 2]) == 1.5",
        ],
    )

    # ── Category 4: Boolean logic ───────────────────────────────────────
    add(
        "is_even",
        "Write a Python function called `is_even` that takes an integer and returns True if it is even.",
        "def is_even(n):\n    return n % 2 == 0",
        [
            "assert is_even(2) == True",
            "assert is_even(3) == False",
            "assert is_even(0) == True",
            "assert is_even(-4) == True",
        ],
    )

    add(
        "is_odd",
        "Write a Python function called `is_odd` that takes an integer and returns True if it is odd.",
        "def is_odd(n):\n    return n % 2 != 0",
        [
            "assert is_odd(3) == True",
            "assert is_odd(2) == False",
            "assert is_odd(0) == False",
            "assert is_odd(-3) == True",
        ],
    )

    add(
        "is_positive",
        "Write a Python function called `is_positive` that takes an integer and returns True if it is positive (greater than 0).",
        "def is_positive(n):\n    return n > 0",
        ["assert is_positive(1) == True", "assert is_positive(0) == False", "assert is_positive(-1) == False"],
    )

    add(
        "is_negative",
        "Write a Python function called `is_negative` that takes an integer and returns True if it is negative (less than 0).",
        "def is_negative(n):\n    return n < 0",
        ["assert is_negative(-1) == True", "assert is_negative(0) == False", "assert is_negative(1) == False"],
    )

    add(
        "is_zero",
        "Write a Python function called `is_zero` that takes an integer and returns True if it is zero.",
        "def is_zero(n):\n    return n == 0",
        ["assert is_zero(0) == True", "assert is_zero(1) == False", "assert is_zero(-1) == False"],
    )

    add(
        "is_palindrome_str",
        "Write a Python function called `is_palindrome` that takes a string and returns True if it reads the same forwards and backwards.",
        "def is_palindrome(s):\n    return s == s[::-1]",
        [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('') == True",
            "assert is_palindrome('a') == True",
        ],
    )

    add(
        "is_empty_list",
        "Write a Python function called `is_empty` that takes a list and returns True if it is empty.",
        "def is_empty(lst):\n    return len(lst) == 0",
        ["assert is_empty([]) == True", "assert is_empty([1]) == False", "assert is_empty([1, 2, 3]) == False"],
    )

    add(
        "is_empty_string",
        "Write a Python function called `is_empty_string` that takes a string and returns True if it is empty.",
        "def is_empty_string(s):\n    return len(s) == 0",
        [
            "assert is_empty_string('') == True",
            "assert is_empty_string('a') == False",
            "assert is_empty_string('hello') == False",
        ],
    )

    add(
        "logical_and",
        "Write a Python function called `logical_and` that takes two booleans and returns True if both are True.",
        "def logical_and(a, b):\n    return a and b",
        [
            "assert logical_and(True, True) == True",
            "assert logical_and(True, False) == False",
            "assert logical_and(False, False) == False",
        ],
    )

    add(
        "logical_or",
        "Write a Python function called `logical_or` that takes two booleans and returns True if at least one is True.",
        "def logical_or(a, b):\n    return a or b",
        [
            "assert logical_or(True, False) == True",
            "assert logical_or(False, False) == False",
            "assert logical_or(True, True) == True",
        ],
    )

    add(
        "logical_not",
        "Write a Python function called `logical_not` that takes a boolean and returns its negation.",
        "def logical_not(a):\n    return not a",
        ["assert logical_not(True) == False", "assert logical_not(False) == True"],
    )

    add(
        "both_positive",
        "Write a Python function called `both_positive` that takes two integers and returns True if both are positive.",
        "def both_positive(a, b):\n    return a > 0 and b > 0",
        [
            "assert both_positive(1, 2) == True",
            "assert both_positive(-1, 2) == False",
            "assert both_positive(0, 1) == False",
        ],
    )

    add(
        "all_even",
        "Write a Python function called `all_even` that takes a list of integers and returns True if all are even.",
        "def all_even(lst):\n    return all(x % 2 == 0 for x in lst)",
        ["assert all_even([2, 4, 6]) == True", "assert all_even([2, 3, 4]) == False", "assert all_even([]) == True"],
    )

    add(
        "any_negative",
        "Write a Python function called `any_negative` that takes a list of integers and returns True if any element is negative.",
        "def any_negative(lst):\n    return any(x < 0 for x in lst)",
        [
            "assert any_negative([1, -2, 3]) == True",
            "assert any_negative([1, 2, 3]) == False",
            "assert any_negative([]) == False",
        ],
    )

    add(
        "is_between",
        "Write a Python function called `is_between` that takes three integers: `value`, `low`, and `high`, and returns True if `low <= value <= high`.",
        "def is_between(value, low, high):\n    return low <= value <= high",
        [
            "assert is_between(5, 1, 10) == True",
            "assert is_between(0, 1, 10) == False",
            "assert is_between(10, 1, 10) == True",
        ],
    )

    add(
        "xor",
        "Write a Python function called `xor` that takes two booleans and returns True if exactly one of them is True.",
        "def xor(a, b):\n    return a != b",
        [
            "assert xor(True, False) == True",
            "assert xor(False, True) == True",
            "assert xor(True, True) == False",
            "assert xor(False, False) == False",
        ],
    )

    add(
        "is_sorted",
        "Write a Python function called `is_sorted` that takes a list of integers and returns True if it is sorted in ascending order.",
        "def is_sorted(lst):\n    return lst == sorted(lst)",
        [
            "assert is_sorted([1, 2, 3]) == True",
            "assert is_sorted([3, 1, 2]) == False",
            "assert is_sorted([]) == True",
            "assert is_sorted([1]) == True",
        ],
    )

    # ── Category 5: Simple conditionals ─────────────────────────────────
    add(
        "sign",
        "Write a Python function called `sign` that takes an integer and returns 1 if positive, -1 if negative, and 0 if zero.",
        "def sign(n):\n    if n > 0:\n        return 1\n    elif n < 0:\n        return -1\n    else:\n        return 0",
        ["assert sign(5) == 1", "assert sign(-3) == -1", "assert sign(0) == 0"],
    )

    add(
        "fizzbuzz",
        "Write a Python function called `fizzbuzz` that takes an integer. Return 'FizzBuzz' if divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, otherwise return the number as a string.",
        "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    elif n % 3 == 0:\n        return 'Fizz'\n    elif n % 5 == 0:\n        return 'Buzz'\n    else:\n        return str(n)",
        [
            "assert fizzbuzz(15) == 'FizzBuzz'",
            "assert fizzbuzz(3) == 'Fizz'",
            "assert fizzbuzz(5) == 'Buzz'",
            "assert fizzbuzz(7) == '7'",
        ],
    )

    add(
        "grade_letter",
        "Write a Python function called `grade_letter` that takes an integer score (0-100) and returns 'A' for 90+, 'B' for 80-89, 'C' for 70-79, 'D' for 60-69, and 'F' for below 60.",
        "def grade_letter(score):\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    elif score >= 70:\n        return 'C'\n    elif score >= 60:\n        return 'D'\n    else:\n        return 'F'",
        [
            "assert grade_letter(95) == 'A'",
            "assert grade_letter(85) == 'B'",
            "assert grade_letter(75) == 'C'",
            "assert grade_letter(65) == 'D'",
            "assert grade_letter(55) == 'F'",
        ],
    )

    add(
        "max_of_three",
        "Write a Python function called `max_of_three` that takes three integers and returns the largest.",
        "def max_of_three(a, b, c):\n    return max(a, b, c)",
        [
            "assert max_of_three(1, 2, 3) == 3",
            "assert max_of_three(3, 2, 1) == 3",
            "assert max_of_three(2, 2, 2) == 2",
        ],
    )

    add(
        "min_of_three",
        "Write a Python function called `min_of_three` that takes three integers and returns the smallest.",
        "def min_of_three(a, b, c):\n    return min(a, b, c)",
        [
            "assert min_of_three(1, 2, 3) == 1",
            "assert min_of_three(3, 2, 1) == 1",
            "assert min_of_three(2, 2, 2) == 2",
        ],
    )

    add(
        "safe_divide",
        "Write a Python function called `safe_divide` that takes two numbers and returns their division. If the second number is 0, return 0.",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
        ["assert safe_divide(10, 2) == 5.0", "assert safe_divide(10, 0) == 0", "assert safe_divide(7, 2) == 3.5"],
    )

    add(
        "even_or_odd",
        "Write a Python function called `even_or_odd` that takes an integer and returns the string 'even' if even, 'odd' if odd.",
        "def even_or_odd(n):\n    return 'even' if n % 2 == 0 else 'odd'",
        ["assert even_or_odd(2) == 'even'", "assert even_or_odd(3) == 'odd'", "assert even_or_odd(0) == 'even'"],
    )

    add(
        "absolute_max",
        "Write a Python function called `absolute_max` that takes two integers and returns the one with the larger absolute value. If equal, return the first.",
        "def absolute_max(a, b):\n    if abs(a) >= abs(b):\n        return a\n    return b",
        ["assert absolute_max(3, -5) == -5", "assert absolute_max(-3, 2) == -3", "assert absolute_max(4, 4) == 4"],
    )

    add(
        "triangle_type",
        "Write a Python function called `triangle_type` that takes three positive integer side lengths. Return 'equilateral' if all sides equal, 'isosceles' if exactly two sides equal, and 'scalene' otherwise.",
        "def triangle_type(a, b, c):\n    if a == b == c:\n        return 'equilateral'\n    elif a == b or b == c or a == c:\n        return 'isosceles'\n    else:\n        return 'scalene'",
        [
            "assert triangle_type(3, 3, 3) == 'equilateral'",
            "assert triangle_type(3, 3, 4) == 'isosceles'",
            "assert triangle_type(3, 4, 5) == 'scalene'",
        ],
    )

    add(
        "leap_year",
        "Write a Python function called `is_leap_year` that takes a year (integer) and returns True if it is a leap year. A year is a leap year if divisible by 4, except centuries must also be divisible by 400.",
        "def is_leap_year(year):\n    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)",
        [
            "assert is_leap_year(2000) == True",
            "assert is_leap_year(1900) == False",
            "assert is_leap_year(2024) == True",
            "assert is_leap_year(2023) == False",
        ],
    )

    add(
        "vowel_or_consonant",
        "Write a Python function called `vowel_or_consonant` that takes a single lowercase letter and returns 'vowel' if it is a vowel (a, e, i, o, u) and 'consonant' otherwise.",
        "def vowel_or_consonant(c):\n    return 'vowel' if c in 'aeiou' else 'consonant'",
        [
            "assert vowel_or_consonant('a') == 'vowel'",
            "assert vowel_or_consonant('b') == 'consonant'",
            "assert vowel_or_consonant('e') == 'vowel'",
            "assert vowel_or_consonant('z') == 'consonant'",
        ],
    )

    add(
        "number_to_day",
        "Write a Python function called `number_to_day` that takes an integer 1-7 and returns the day of the week (1='Monday', 7='Sunday'). Return 'Invalid' for other inputs.",
        "def number_to_day(n):\n    days = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}\n    return days.get(n, 'Invalid')",
        [
            "assert number_to_day(1) == 'Monday'",
            "assert number_to_day(7) == 'Sunday'",
            "assert number_to_day(3) == 'Wednesday'",
            "assert number_to_day(8) == 'Invalid'",
        ],
    )

    add(
        "classify_number",
        "Write a Python function called `classify_number` that takes an integer and returns 'positive' if > 0, 'negative' if < 0, and 'zero' if 0.",
        "def classify_number(n):\n    if n > 0:\n        return 'positive'\n    elif n < 0:\n        return 'negative'\n    else:\n        return 'zero'",
        [
            "assert classify_number(5) == 'positive'",
            "assert classify_number(-3) == 'negative'",
            "assert classify_number(0) == 'zero'",
        ],
    )

    # ── Category 6: Simple loops ────────────────────────────────────────
    add(
        "factorial",
        "Write a Python function called `factorial` that takes a non-negative integer and returns its factorial.",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
            "assert factorial(3) == 6",
        ],
    )

    add(
        "fibonacci",
        "Write a Python function called `fibonacci` that takes a non-negative integer n and returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...).",
        "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(2) == 1",
            "assert fibonacci(6) == 8",
            "assert fibonacci(10) == 55",
        ],
    )

    add(
        "count_vowels",
        "Write a Python function called `count_vowels` that takes a string and returns the number of vowels (a, e, i, o, u, case-insensitive).",
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('AEIOU') == 5",
            "assert count_vowels('xyz') == 0",
            "assert count_vowels('') == 0",
        ],
    )

    add(
        "sum_digits",
        "Write a Python function called `sum_digits` that takes a non-negative integer and returns the sum of its digits.",
        "def sum_digits(n):\n    return sum(int(d) for d in str(n))",
        [
            "assert sum_digits(123) == 6",
            "assert sum_digits(0) == 0",
            "assert sum_digits(999) == 27",
            "assert sum_digits(10) == 1",
        ],
    )

    add(
        "count_digits",
        "Write a Python function called `count_digits` that takes a non-negative integer and returns the number of digits.",
        "def count_digits(n):\n    return len(str(n))",
        ["assert count_digits(123) == 3", "assert count_digits(0) == 1", "assert count_digits(10000) == 5"],
    )

    add(
        "reverse_integer",
        "Write a Python function called `reverse_integer` that takes a non-negative integer and returns it with its digits reversed.",
        "def reverse_integer(n):\n    return int(str(n)[::-1])",
        [
            "assert reverse_integer(123) == 321",
            "assert reverse_integer(100) == 1",
            "assert reverse_integer(0) == 0",
            "assert reverse_integer(5) == 5",
        ],
    )

    add(
        "sum_range",
        "Write a Python function called `sum_range` that takes two integers `a` and `b` and returns the sum of all integers from `a` to `b` inclusive.",
        "def sum_range(a, b):\n    return sum(range(a, b + 1))",
        ["assert sum_range(1, 5) == 15", "assert sum_range(3, 3) == 3", "assert sum_range(1, 10) == 55"],
    )

    add(
        "power_iterative",
        "Write a Python function called `power_iterative` that takes a base and a non-negative exponent and returns base**exponent using a loop.",
        "def power_iterative(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result",
        [
            "assert power_iterative(2, 3) == 8",
            "assert power_iterative(5, 0) == 1",
            "assert power_iterative(3, 3) == 27",
        ],
    )

    add(
        "repeat_char",
        "Write a Python function called `repeat_char` that takes a character and an integer n and returns a string of that character repeated n times.",
        "def repeat_char(c, n):\n    return c * n",
        [
            "assert repeat_char('a', 3) == 'aaa'",
            "assert repeat_char('x', 1) == 'x'",
            "assert repeat_char('z', 0) == ''",
        ],
    )

    add(
        "collatz_steps",
        "Write a Python function called `collatz_steps` that takes a positive integer and returns the number of steps to reach 1 using the Collatz conjecture (if even, divide by 2; if odd, multiply by 3 and add 1).",
        "def collatz_steps(n):\n    steps = 0\n    while n != 1:\n        if n % 2 == 0:\n            n = n // 2\n        else:\n            n = 3 * n + 1\n        steps += 1\n    return steps",
        [
            "assert collatz_steps(1) == 0",
            "assert collatz_steps(2) == 1",
            "assert collatz_steps(3) == 7",
            "assert collatz_steps(6) == 8",
        ],
    )

    add(
        "is_prime",
        "Write a Python function called `is_prime` that takes an integer and returns True if it is a prime number.",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        [
            "assert is_prime(2) == True",
            "assert is_prime(3) == True",
            "assert is_prime(4) == False",
            "assert is_prime(17) == True",
            "assert is_prime(1) == False",
        ],
    )

    add(
        "gcd",
        "Write a Python function called `gcd` that takes two positive integers and returns their greatest common divisor.",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        ["assert gcd(12, 8) == 4", "assert gcd(7, 3) == 1", "assert gcd(100, 25) == 25", "assert gcd(5, 5) == 5"],
    )

    add(
        "lcm",
        "Write a Python function called `lcm` that takes two positive integers and returns their least common multiple.",
        "def lcm(a, b):\n    def gcd(x, y):\n        while y:\n            x, y = y, x % y\n        return x\n    return a * b // gcd(a, b)",
        ["assert lcm(4, 6) == 12", "assert lcm(3, 5) == 15", "assert lcm(7, 7) == 7"],
    )

    add(
        "count_even",
        "Write a Python function called `count_even` that takes a list of integers and returns how many are even.",
        "def count_even(lst):\n    return sum(1 for x in lst if x % 2 == 0)",
        [
            "assert count_even([1, 2, 3, 4, 5, 6]) == 3",
            "assert count_even([1, 3, 5]) == 0",
            "assert count_even([]) == 0",
        ],
    )

    add(
        "sum_even",
        "Write a Python function called `sum_even` that takes a list of integers and returns the sum of the even ones.",
        "def sum_even(lst):\n    return sum(x for x in lst if x % 2 == 0)",
        ["assert sum_even([1, 2, 3, 4]) == 6", "assert sum_even([1, 3, 5]) == 0", "assert sum_even([]) == 0"],
    )

    add(
        "running_sum",
        "Write a Python function called `running_sum` that takes a list of integers and returns a new list where each element is the cumulative sum up to that index.",
        "def running_sum(lst):\n    result = []\n    total = 0\n    for x in lst:\n        total += x\n        result.append(total)\n    return result",
        [
            "assert running_sum([1, 2, 3]) == [1, 3, 6]",
            "assert running_sum([]) == []",
            "assert running_sum([5]) == [5]",
        ],
    )

    # ── Category 7: Dict/set basics ─────────────────────────────────────
    add(
        "get_value",
        "Write a Python function called `get_value` that takes a dictionary and a key and returns the value for that key, or None if the key is not found.",
        "def get_value(d, key):\n    return d.get(key)",
        [
            "assert get_value({'a': 1, 'b': 2}, 'a') == 1",
            "assert get_value({'a': 1}, 'b') == None",
            "assert get_value({}, 'x') == None",
        ],
    )

    add(
        "has_key",
        "Write a Python function called `has_key` that takes a dictionary and a key and returns True if the key exists in the dictionary.",
        "def has_key(d, key):\n    return key in d",
        [
            "assert has_key({'a': 1, 'b': 2}, 'a') == True",
            "assert has_key({'a': 1}, 'b') == False",
            "assert has_key({}, 'x') == False",
        ],
    )

    add(
        "dict_keys",
        "Write a Python function called `dict_keys` that takes a dictionary and returns a sorted list of its keys.",
        "def dict_keys(d):\n    return sorted(d.keys())",
        ["assert dict_keys({'b': 2, 'a': 1}) == ['a', 'b']", "assert dict_keys({}) == []"],
    )

    add(
        "dict_values",
        "Write a Python function called `dict_values` that takes a dictionary and returns a sorted list of its values.",
        "def dict_values(d):\n    return sorted(d.values())",
        ["assert dict_values({'a': 3, 'b': 1, 'c': 2}) == [1, 2, 3]", "assert dict_values({}) == []"],
    )

    add(
        "merge_dicts",
        "Write a Python function called `merge_dicts` that takes two dictionaries and returns a new dictionary that is the merge of both. If a key exists in both, the value from the second dictionary should be used.",
        "def merge_dicts(a, b):\n    result = {**a, **b}\n    return result",
        [
            "assert merge_dicts({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}",
            "assert merge_dicts({'a': 1}, {'a': 2}) == {'a': 2}",
            "assert merge_dicts({}, {}) == {}",
        ],
    )

    add(
        "invert_dict",
        "Write a Python function called `invert_dict` that takes a dictionary and returns a new dictionary with keys and values swapped.",
        "def invert_dict(d):\n    return {v: k for k, v in d.items()}",
        ["assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}", "assert invert_dict({}) == {}"],
    )

    add(
        "unique_elements",
        "Write a Python function called `unique_elements` that takes a list and returns a sorted list of unique elements.",
        "def unique_elements(lst):\n    return sorted(set(lst))",
        [
            "assert unique_elements([3, 1, 2, 1, 3]) == [1, 2, 3]",
            "assert unique_elements([]) == []",
            "assert unique_elements([1]) == [1]",
        ],
    )

    add(
        "set_intersection",
        "Write a Python function called `set_intersection` that takes two lists and returns a sorted list of elements that appear in both.",
        "def set_intersection(a, b):\n    return sorted(set(a) & set(b))",
        [
            "assert set_intersection([1, 2, 3], [2, 3, 4]) == [2, 3]",
            "assert set_intersection([1, 2], [3, 4]) == []",
            "assert set_intersection([], [1]) == []",
        ],
    )

    add(
        "set_union",
        "Write a Python function called `set_union` that takes two lists and returns a sorted list of all unique elements from both.",
        "def set_union(a, b):\n    return sorted(set(a) | set(b))",
        [
            "assert set_union([1, 2], [2, 3]) == [1, 2, 3]",
            "assert set_union([], [1]) == [1]",
            "assert set_union([], []) == []",
        ],
    )

    add(
        "set_difference",
        "Write a Python function called `set_difference` that takes two lists and returns a sorted list of elements in the first but not the second.",
        "def set_difference(a, b):\n    return sorted(set(a) - set(b))",
        [
            "assert set_difference([1, 2, 3], [2, 3, 4]) == [1]",
            "assert set_difference([1, 2], [1, 2]) == []",
            "assert set_difference([1, 2, 3], []) == [1, 2, 3]",
        ],
    )

    add(
        "count_unique",
        "Write a Python function called `count_unique` that takes a list and returns the number of unique elements.",
        "def count_unique(lst):\n    return len(set(lst))",
        [
            "assert count_unique([1, 2, 2, 3]) == 3",
            "assert count_unique([]) == 0",
            "assert count_unique([1, 1, 1]) == 1",
        ],
    )

    add(
        "word_count",
        "Write a Python function called `word_count` that takes a string and returns a dictionary mapping each word to its count.",
        "def word_count(s):\n    counts = {}\n    for word in s.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
        [
            "assert word_count('a b a') == {'a': 2, 'b': 1}",
            "assert word_count('hello') == {'hello': 1}",
            "assert word_count('') == {}",
        ],
    )

    add(
        "dict_from_lists",
        "Write a Python function called `dict_from_lists` that takes two lists (keys and values) of equal length and returns a dictionary.",
        "def dict_from_lists(keys, values):\n    return dict(zip(keys, values))",
        ["assert dict_from_lists(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}", "assert dict_from_lists([], []) == {}"],
    )

    add(
        "filter_dict",
        "Write a Python function called `filter_dict` that takes a dictionary and a list of keys, and returns a new dictionary containing only those keys.",
        "def filter_dict(d, keys):\n    return {k: d[k] for k in keys if k in d}",
        [
            "assert filter_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}",
            "assert filter_dict({'a': 1}, ['b']) == {}",
        ],
    )

    add(
        "dict_size",
        "Write a Python function called `dict_size` that takes a dictionary and returns the number of key-value pairs.",
        "def dict_size(d):\n    return len(d)",
        ["assert dict_size({'a': 1, 'b': 2}) == 2", "assert dict_size({}) == 0"],
    )

    add(
        "is_subset",
        "Write a Python function called `is_subset` that takes two lists and returns True if all elements of the first are in the second.",
        "def is_subset(a, b):\n    return set(a).issubset(set(b))",
        [
            "assert is_subset([1, 2], [1, 2, 3]) == True",
            "assert is_subset([1, 4], [1, 2, 3]) == False",
            "assert is_subset([], [1, 2]) == True",
        ],
    )

    # ── Category 8: Type conversions ────────────────────────────────────
    add(
        "int_to_str",
        "Write a Python function called `int_to_str` that takes an integer and returns it as a string.",
        "def int_to_str(n):\n    return str(n)",
        ["assert int_to_str(42) == '42'", "assert int_to_str(0) == '0'", "assert int_to_str(-5) == '-5'"],
    )

    add(
        "str_to_int",
        "Write a Python function called `str_to_int` that takes a string representation of an integer and returns the integer.",
        "def str_to_int(s):\n    return int(s)",
        ["assert str_to_int('42') == 42", "assert str_to_int('0') == 0", "assert str_to_int('-5') == -5"],
    )

    add(
        "float_to_int",
        "Write a Python function called `float_to_int` that takes a float and returns it truncated to an integer.",
        "def float_to_int(f):\n    return int(f)",
        ["assert float_to_int(3.7) == 3", "assert float_to_int(3.0) == 3", "assert float_to_int(-2.5) == -2"],
    )

    add(
        "int_to_float",
        "Write a Python function called `int_to_float` that takes an integer and returns it as a float.",
        "def int_to_float(n):\n    return float(n)",
        ["assert int_to_float(42) == 42.0", "assert int_to_float(0) == 0.0", "assert int_to_float(-5) == -5.0"],
    )

    add(
        "list_to_set",
        "Write a Python function called `list_to_set` that takes a list and returns a set.",
        "def list_to_set(lst):\n    return set(lst)",
        ["assert list_to_set([1, 2, 2, 3]) == {1, 2, 3}", "assert list_to_set([]) == set()"],
    )

    add(
        "set_to_sorted_list",
        "Write a Python function called `set_to_sorted_list` that takes a set and returns a sorted list.",
        "def set_to_sorted_list(s):\n    return sorted(s)",
        ["assert set_to_sorted_list({3, 1, 2}) == [1, 2, 3]", "assert set_to_sorted_list(set()) == []"],
    )

    add(
        "celsius_to_fahrenheit",
        "Write a Python function called `celsius_to_fahrenheit` that takes a temperature in Celsius and returns it in Fahrenheit.",
        "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
        [
            "assert celsius_to_fahrenheit(0) == 32.0",
            "assert celsius_to_fahrenheit(100) == 212.0",
            "assert celsius_to_fahrenheit(-40) == -40.0",
        ],
    )

    add(
        "fahrenheit_to_celsius",
        "Write a Python function called `fahrenheit_to_celsius` that takes a temperature in Fahrenheit and returns it in Celsius.",
        "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9",
        [
            "assert fahrenheit_to_celsius(32) == 0.0",
            "assert fahrenheit_to_celsius(212) == 100.0",
            "assert fahrenheit_to_celsius(-40) == -40.0",
        ],
    )

    add(
        "bool_to_int",
        "Write a Python function called `bool_to_int` that takes a boolean and returns 1 for True and 0 for False.",
        "def bool_to_int(b):\n    return int(b)",
        ["assert bool_to_int(True) == 1", "assert bool_to_int(False) == 0"],
    )

    add(
        "int_to_bool",
        "Write a Python function called `int_to_bool` that takes an integer and returns False if 0, True otherwise.",
        "def int_to_bool(n):\n    return bool(n)",
        [
            "assert int_to_bool(0) == False",
            "assert int_to_bool(1) == True",
            "assert int_to_bool(-1) == True",
            "assert int_to_bool(42) == True",
        ],
    )

    add(
        "list_to_string",
        "Write a Python function called `list_to_string` that takes a list of strings and returns them joined by commas.",
        "def list_to_string(lst):\n    return ','.join(lst)",
        [
            "assert list_to_string(['a', 'b', 'c']) == 'a,b,c'",
            "assert list_to_string(['hello']) == 'hello'",
            "assert list_to_string([]) == ''",
        ],
    )

    add(
        "string_to_list",
        "Write a Python function called `string_to_list` that takes a comma-separated string and returns a list of the parts.",
        "def string_to_list(s):\n    if not s:\n        return []\n    return s.split(',')",
        [
            "assert string_to_list('a,b,c') == ['a', 'b', 'c']",
            "assert string_to_list('hello') == ['hello']",
            "assert string_to_list('') == []",
        ],
    )

    add(
        "char_to_ascii",
        "Write a Python function called `char_to_ascii` that takes a single character and returns its ASCII code.",
        "def char_to_ascii(c):\n    return ord(c)",
        ["assert char_to_ascii('A') == 65", "assert char_to_ascii('a') == 97", "assert char_to_ascii('0') == 48"],
    )

    add(
        "ascii_to_char",
        "Write a Python function called `ascii_to_char` that takes an ASCII code (integer) and returns the corresponding character.",
        "def ascii_to_char(n):\n    return chr(n)",
        ["assert ascii_to_char(65) == 'A'", "assert ascii_to_char(97) == 'a'", "assert ascii_to_char(48) == '0'"],
    )

    add(
        "int_to_binary",
        "Write a Python function called `int_to_binary` that takes a non-negative integer and returns its binary representation as a string (without '0b' prefix).",
        "def int_to_binary(n):\n    return bin(n)[2:]",
        [
            "assert int_to_binary(10) == '1010'",
            "assert int_to_binary(0) == '0'",
            "assert int_to_binary(255) == '11111111'",
        ],
    )

    add(
        "binary_to_int",
        "Write a Python function called `binary_to_int` that takes a binary string and returns the integer it represents.",
        "def binary_to_int(s):\n    return int(s, 2)",
        [
            "assert binary_to_int('1010') == 10",
            "assert binary_to_int('0') == 0",
            "assert binary_to_int('11111111') == 255",
        ],
    )

    add(
        "round_to_n",
        "Write a Python function called `round_to_n` that takes a float and an integer n and returns the float rounded to n decimal places.",
        "def round_to_n(x, n):\n    return round(x, n)",
        [
            "assert round_to_n(3.14159, 2) == 3.14",
            "assert round_to_n(2.5, 0) == 2.0",
            "assert round_to_n(1.005, 2) == 1.0",
        ],
    )

    add(
        "int_to_hex",
        "Write a Python function called `int_to_hex` that takes a non-negative integer and returns its hexadecimal representation as a lowercase string (without '0x' prefix).",
        "def int_to_hex(n):\n    return hex(n)[2:]",
        ["assert int_to_hex(255) == 'ff'", "assert int_to_hex(0) == '0'", "assert int_to_hex(16) == '10'"],
    )

    add(
        "tuple_to_list",
        "Write a Python function called `tuple_to_list` that takes a tuple and returns it as a list.",
        "def tuple_to_list(t):\n    return list(t)",
        ["assert tuple_to_list((1, 2, 3)) == [1, 2, 3]", "assert tuple_to_list(()) == []"],
    )

    add(
        "list_to_tuple",
        "Write a Python function called `list_to_tuple` that takes a list and returns it as a tuple.",
        "def list_to_tuple(lst):\n    return tuple(lst)",
        ["assert list_to_tuple([1, 2, 3]) == (1, 2, 3)", "assert list_to_tuple([]) == ()"],
    )

    add(
        "enumerate_list",
        "Write a Python function called `enumerate_list` that takes a list and returns a list of (index, value) tuples.",
        "def enumerate_list(lst):\n    return list(enumerate(lst))",
        [
            "assert enumerate_list(['a', 'b', 'c']) == [(0, 'a'), (1, 'b'), (2, 'c')]",
            "assert enumerate_list([]) == []",
        ],
    )

    return problems


def main():
    parser = argparse.ArgumentParser(description="Create debug code RL-Zero dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without pushing to HF")
    parser.add_argument("--repo", default="geodesic-research/debug-code-rlzero", help="HF repo name")
    args = parser.parse_args()

    problems = make_problems()
    print(f"Generated {len(problems)} problems")

    # Validate all problems
    for p in problems:
        assert isinstance(p["id"], str), f"id must be str: {p['id']}"
        assert isinstance(p["prompt"], str), f"prompt must be str: {p['id']}"
        assert isinstance(p["solution"], str), f"solution must be str: {p['id']}"
        assert isinstance(p["ground_truth"], list), f"ground_truth must be list: {p['id']}"
        assert all(isinstance(t, str) for t in p["ground_truth"]), f"ground_truth items must be str: {p['id']}"
        # Verify solution passes tests
        try:
            exec(p["solution"], {})
            ns = {}
            exec(p["solution"], ns)
            for test in p["ground_truth"]:
                exec(test, ns)
        except Exception as e:
            print(f"FAILED: {p['id']} — {e}")
            print(f"  Solution: {p['solution']}")
            print(f"  Test: {test}")
            raise

    # Print category breakdown
    print(f"\nAll {len(problems)} solutions verified against their test cases.")

    if args.dry_run:
        print("\n--- DRY RUN: first 3 problems ---")
        for p in problems[:3]:
            print(json.dumps(p, indent=2))
        return

    # Serialize ground_truth as JSON strings (matching Dolci format)
    for p in problems:
        p["ground_truth"] = json.dumps(p["ground_truth"])

    ds = Dataset.from_list(problems)
    print(f"\nDataset: {ds}")
    print(f"Pushing to {args.repo}...")
    ds.push_to_hub(args.repo, split="train")
    print("Done!")


if __name__ == "__main__":
    main()
