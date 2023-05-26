from math import exp
from random import uniform
import time # debugging



# represents any non-scalar values
class TensorValue:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.shape = self._gen_shape(data)
        self._prev = set(children)
        self._op = op

        self.grad = None
        self._backward = lambda: None
        
    def __str__(self, level=0):
        if level == 6: return ""
        out = ""
        out += "\t"*level+ "data:"+str(self.data)+" "+ str(self.shape) + "\n"
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
            

    # figures out 'shape' of self.data
    def _gen_shape(self, data):
        if not isinstance(data, (list, tuple)):
            return tuple()
        return (len(data),) + self._gen_shape(data[0])
    
    def converge(self, amt=-0.1):
        self.data = self._piecewise(self.data, self.grad, f=lambda x,y: x + amt*y)

    # recursively run a function on N tensors of the same shape
    def _piecewise(self, *inps, f):
        assert all(tuple(len(i) == len(inps[0]) for i in inps)), f"cannot piecewise tensors of lengths {list(map(len, *inps))}"

        if not isinstance(inps[0][0], (list,tuple)):
            return tuple(f(*x) for x in zip(*inps))
        
        return tuple(self._piecewise(*(x[i] for x in inps), f=f) for i in range(len(inps[0])))

    def _dotVectors(self, a, b):
        return sum(self._piecewise(a,b,f = lambda x,y: x*y))

    def T(self): # transpose
        return tuple(zip(*self.data))

    def __add__(self, other):
        if other == 0: # for the sum() function, which has a 'start' param
            return self
            
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        assert other.shape == self.shape, f"failed to add tensors of shape {self.shape} and {other.shape}"
        out = TensorValue(self._piecewise(self.data, other.data, f = lambda x,y: x+y), (self, other), '+')
        
        def _backward():
            assert len(self.shape) == len(other.shape) == 2, "only supporting tensors of rank 2!"
            assert self.shape[1] == other.shape[1] == 1, "only supporting column vectors now!"

            self.grad = 1*out.grad
            other.grad = 1*out.grad

        out._backward = _backward

        return out

    def __radd__(self,other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        
        return self + (-other)

    def __neg__(self):
        return TensorValue(self._piecewise(self.data, f = lambda x: -x), (self,), "-")

    def __mul__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        assert other.shape == self.shape, f"Failed to add tensors of shape {self.shape} and {other.shape}"
        out = TensorValue(self._piecewise(self.data, other.data, f = lambda x,y: x*y), (self, other), '*')


        def _backward():
            assert len(self.shape) == len(other.shape) == 2, "only supporting tensors of rank 2!"
            assert self.shape[1] == other.shape[1] == 1, "only supporting column vectors now!"

            self.grad = (TensorValue(other.data)*TensorValue(out.grad)).data
            other.grad = (TensorValue(self.data)*TensorValue(out.grad)).data

        out._backward = _backward

        return out

    def __rmul__(self,other): # other * self
        return self * other

    def __truediv__(self, other):
        return self * other**-1
    
    # rescales a single number to an tensor of that number
    def scale(self, num, shape):
        assert len(shape)==2, "only supporting tensors of rank 2!"
        return tuple(tuple(num for _ in range(shape[1])) for d in range(shape[0]))
        

    def __pow__(self, other):
        assert isinstance(other, (int,float)), "only supporting integer or floats"

        out = TensorValue(self._piecewise(self.data, f = lambda x: x**other), (self, ), '**')

        def _backward():
            assert len(self.shape) == 2, "only supporting tensors of rank 2!"
            assert self.shape[1] == 1, "only supporting column vectors now!"

            self.grad = (self.scale(other, self.shape)
                        * (self**(other-1))
                        * TensorValue(out.grad)).data

        out._backward = _backward

        return out


    def __matmul__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other)
        assert len(self.shape) < 3 and len(other.shape) < 3, f"Cannot dot 2 tensors of rank {len(self.shape)}, {len(other.shape)}"
        assert self.shape[-1] == other.shape[0], f"Cannot dot 2 tensors of shape {self.shape} and {other.shape}"

        out = TensorValue(tuple(tuple(self._dotVectors(self.data[d], other.T()[e]) 
                                    for e in range(other.shape[-1])) 
                                    for d in range(self.shape[0])),
                                    (self,other), '@')
        
        def _backward():
            assert len(self.shape) == len(other.shape) == 2, "only supporting tensors of rank 2!"
            #assert self.shape[1] > 1 and self.shape[1] == 1, "only supporting matrix @ vector autograd!"
            
            self.grad = TensorValue(out.grad).dot(TensorValue(other.data).T()).data
            other.grad= TensorValue(self.T()).dot(TensorValue(out.grad)).data

        out._backward = _backward
        return out
    
    def __rmatmul__(self, other):
        return self @ other

    def dot(self,other):
        return self.__matmul__(other)

    def sigmoid(self):
        out = TensorValue(self._piecewise(self.data, f=lambda x: pow(1+ exp(-x),-1)), (self,), 'sigmoid')
        
        def _backward():
            self.grad = (TensorValue(out.grad) * out*TensorValue(self._piecewise(out.data, f=lambda x: 1-x))).data

        out._backward = _backward
        return out
    
    def tanh(self):
        out = TensorValue(self._piecewise(self.data, f=lambda x: (exp(2*x)-1)/(exp(2*x)+1)), (self,), 'tanh')
        
        def _backward():
            self.grad = (TensorValue(out.grad)*TensorValue(self._piecewise(out.data, f=lambda x: 1-pow(x,2)))).data

        out._backward = _backward
        return out
    
                          

    def backward(self):
        assert self.shape[1] == 1, "only supporting column vectors and scalars now!"

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = tuple((1.0,) for d in range(self.shape[0]))
        for node in reversed(topo):
            node._backward()


class Layer:
    def __init__(self, nin, nout, output=False):
        self.W = TensorValue(tuple(tuple(uniform(-1,1) for d in range(nin)) for e in range(nout)))
        self.b = TensorValue(tuple((uniform(-1,1),) for _ in range(nout)))
        self.output = output

    def __call__(self, x):
        out = (self.W @ x + self.b)
        if not self.output: out = out.tanh()
        return out
    
    def parameters(self):
        return [self.W, self.b]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.layers[-1].output = True

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        out = []
        for i in self.layers:
            out += i.parameters()
        return out


if __name__ == "__main__":
    n = MLP(2,[4,4,1])

    # before: 1.2s
    # time1 = time.time()
    # for i in range(1):
    #     a = nn(((1,),))
    # time2 = time.time()
    # print(time2-time1)


    xs = [((2.0,), (3.0,), (-1.0,)), ((3.0,), (-1.0,),(0.5,)), ((0.5,), (1.0,), (1.0,)), ((1.0,), (1.0,), (-1.0,))]
    ys = [((4.0,),), ((2.5,),), ((2.5,),), ((1.0,),)]
    
    xs = [((uniform(-10,10),), (uniform(-10,10),)) for d in range(100)]
    ys = [((sum(xs[d][0]),),) for d in range(100)]

    for i in range(100000):
        ypred = [n(x) for x in xs]
        #print(tuple(zip(ys,(x.data for x in ypred))))
        
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        
        #loss = TensorValue(((0,),))
        #for ygt, yout in zip(ys,ypred):
        #    loss += (yout - ygt)**2    
        #loss /= TensorValue(((4,),))
        
        
        if not i % 100:
            print(int(loss.data[0][0]))

        for p in n.parameters():
            p.grad = None
        loss.backward()

        for p in n.parameters():
            p.converge(-0.001)
            #p.data += -0.01 * p.grad

#     tx = ( ((1,),(2,),(3,),(4,)), ((5,),(6,),(7,),(8,)), ((1,),(1,),(1,),(1,)) )
#     ty = (((10,),),((26,),),((4,),))
#     
#     for j in range(100):
#         loss = TensorValue(((0,),))
#         params = []
#         for i in range(3):
#             x = TensorValue(tx[i])
#             W = TensorValue(((0,1,2,3), (0,1,2,3), (0,1,2,3), (0,1,2,3)))
#             V = TensorValue(((0,1,2,3),))
#             b = TensorValue(((8,),(7,),(-4,),(-6,)))
#             b2= TensorValue(((1,),))
#             k = W @ x + b
#             h = k.sigmoid()
#             y = V @ h + b2
#             
#             W.grad = None
#             V.grad = None
#             b.grad = None
#             b2.grad = None
#             
#             params.extend([W,V,b,b2])
#             
#             loss = loss + (y - ty[i])**2
#         
#         loss.backward()
#         print(loss.data)
#         for i in params:
#             i.converge(-0.01)
    
        
    
        
        
     

#     for i in range(1000):
# 
#         
#         
#         print(o.data)
#         
#         W.grad = None
#         V.grad = None
#         b.grad = None
#         b2.grad = None
#         o.backward()
#         
#         W.converge(-0.1)
#         V.converge(-0.1)
#         b.converge(-0.1)
#         b2.converge(-0.1)
        
        
    #print(a.shape, b.shape)
