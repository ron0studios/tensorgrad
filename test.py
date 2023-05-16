#import myModule
#print(myModule.fib(40))

a = 1
b = 1
for i in range(41):
    c = a + b
    a = b 
    b = c
print(b)
