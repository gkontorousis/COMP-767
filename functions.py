
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


