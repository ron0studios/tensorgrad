from math import exp
from random import uniform

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __str__(self, level=0):
        if level == 6: return ""
        out = ""
        out += "\t"*level+ "data:"+str(self.data)+" "+ "\n"
        out += "\t"*level+ "op:"+self._op + "\n"
        out += "\t"*level+ "grad:"+str(self.grad) + "\n"
        out += "\t"*level+ "id:" + str(id(self)) + "\n"
        out += "\t"*level+ "children:" + "\n"

        
        if not self._prev:
            out += "\t"*level+"NONE" + "\n"
        else:
            for i in self._prev:
                out += i.__str__(level+1) + "\n"
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            
        out._backward = _backward
        return out

    def __radd__(self,other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self,other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out

    def __rmul__(self,other): # other * self
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self,other):
        assert isinstance(other, (int,float)), "only supporting ints and floats powers" 
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad = other * (self.data**(other-1)) * out.grad
            pass

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (exp(2*x) - 1)/(exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1- t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(uniform(-1,1)) for _ in range(nin)]
        self.b = Value(uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    n = MLP(3, [10,10,1])

    xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ]
    ys = [1.0, -1.0, -1.0, 1.0]
    
    xs = [tuple(uniform(-10,10) for d in range(3))for i in range(20)]
    ys = [sum(xs[i]) for i in range(20)]

    for i in range(1000):
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        for p in n.parameters():
            p.grad = 0
        loss.backward()
        
        print(loss.data)

        for p in n.parameters():
            p.data += -0.1 * p.grad

    pass
