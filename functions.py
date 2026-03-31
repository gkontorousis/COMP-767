
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





if __name__ == "__main__":
    # run times2_add3_every3digits generator
    gen = look_and_say()
    for _ in range(10):
        print(next(gen))



