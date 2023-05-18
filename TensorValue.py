from math import exp
from random import uniform
import time # debugging


# represents any non-scalar values
class TensorValue:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.shape = self._gen_shape(data)
        self._prev = children
        self._op = op

        self.grad = None
        self._backward = lambda: None

    # figures out 'shape' of self.data
    def _gen_shape(self, data):
        if not isinstance(data, (list, tuple)):
            return tuple()
        return (len(data),) + self._gen_shape(data[0])

    # recursively run a function on N tensors of the same shape
    def _piecewise(self, *inps, f):
        assert all(len(i) == len(inps[0]) for i in inps), f"cannot piecewise tensors of lengths {list(map(len, *inps))}"

        if not isinstance(inps[0][0], (list,tuple)):
            return tuple(f(*x) for x in zip(*inps))
        
        return tuple(self._piecewise(*(x[i] for x in inps), f=f) for i in range(len(inps[0])))

    def _dotVectors(self, a, b):
        return sum(self._piecewise(a,b,f = lambda x,y: x*y))

    def T(self): # transpose
        return tuple(zip(*self.data))

    def __add__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        assert other.shape == self.shape, f"Failed to add tensors of shape {self.shape} and {other.shape}"
        out = TensorValue(self._piecewise(self.data, other.data, f = lambda x,y: x+y), (self, other), '+')
        
        def _backward():
            assert len(self.shape) == len(other.shape) == 2, "only supporting tensors of rank 2!"
            assert self.shape[1] == other.shape[1] == 1, "only supporting column vectors now!"

            self.grad = 1*out.grad
            other.grad = 1*out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        assert other.shape == self.shape, f"Failed to add tensors of shape {self.shape} and {other.shape}"
        out = TensorValue(self._piecewise(self.data, other.data, f = lambda x,y: x*y), (self, other), '*')


        def _backward():
            assert len(self.shape) == len(other.shape) == 2, "only supporting tensors of rank 2!"
            assert self.shape[1] == other.shape[1] == 1, "only supporting column vectors now!"

            self.grad = other.grad*out.grad
            other.grad = self.grad*out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        assert len(self.shape) < 3 and len(other.shape) < 3, f"Cannot dot 2 tensors of rank {len(self.shape)}, {len(other.shape)}"
        assert self.shape[-1] == other.shape[0], f"Cannot dot 2 tensors of shape {self.shape} and {other.shape}"

        out = TensorValue(tuple(tuple(self._dotVectors(self.data[d], other.T()[e]) 
                                    for e in range(other.shape[-1])) 
                                    for d in range(self.shape[0])),
                                    (self,other), '.')
        
        def _bac

        return out

    def dot(self,other):
        return self.__matmul__(other)

    def sigmoid(self):
        out = TensorValue(self._piecewise(self.data, f=lambda x: pow(1+ exp(-x),-1)), (self), 'sigmoid')
        return out


class Layer:
    def __init__(self, nin, nout):
        self.W = TensorValue(tuple(tuple(uniform(-1,1) for d in range(nin)) for e in range(nout)))
        self.b = TensorValue(tuple((uniform(-1,1),) for _ in range(nout)))

    def __call__(self, x):
        out = (self.W @ x + self.b).sigmoid()
        return out

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    nn = MLP(1,[100,1000,1000,1000,1000,100,1])
    time1 = time.time()

    # before: 1.2s
    for i in range(1):
        a = nn(((1,),))
    time2 = time.time()
    print(time2-time1)
    # a = TensorValue(((1,2),(3,4)))
    # b = TensorValue(((5,),(6,)))

    # x = TensorValue(((1,),(0.5,),(0.3,),(0.7,)))
    # W = TensorValue(((0,1,2,3), (0,1,2,3), (0,1,2,3), (0,1,2,3)))
    # b = TensorValue(((8,),(7,),(-4,),(-6,)))
    # k = W @ x + b
    # h = k.sigmoid()
    # print(a.shape, b.shape)

