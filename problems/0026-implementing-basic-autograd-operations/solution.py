class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)
        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
            
        out._backward = _backward

        return out

    def backward(self):
        visited = set()
        topo = []
        def build_topo_safe(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo_safe(child)
                topo.append(v)
                
        build_topo_safe(self)

        self.grad = 1.0

        for v in reversed(topo):
            v._backward()