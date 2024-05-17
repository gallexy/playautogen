# filename: perfect_numbers.py

def proper_divisors(n):
    divisors = [1]
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.extend([i, n // i])
    return divisors

def is_perfect_number(n):
    return sum(proper_divisors(n)) == n

perfect_numbers = []
for i in range(1, 1000001):
    if is_perfect_number(i):
        perfect_numbers.append(i)

print(f"Perfect numbers within the range of 1 to 1000000: {perfect_numbers}")