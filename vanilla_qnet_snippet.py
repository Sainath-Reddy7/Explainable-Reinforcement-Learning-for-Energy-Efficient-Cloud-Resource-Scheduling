

class VanillaQNetwork:
    """Plain 3-layer MLP (ablation control for the Dueling architecture).
    Same public interface as QNetwork so agent / XAI / trust metrics run
    unmodified when the Dueling streams are disabled."""

    def __init__(self, in_dim, out_dim, hidden=(128, 64), lr=5e-4, dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        dims = [in_dim, h1, h2, out_dim]
        self.params = []
        for i in range(len(dims) - 1):
            setattr(self, f"W{i+1}", rng.normal(0, np.sqrt(2/dims[i]), size=(dims[i], dims[i+1])).astype(np.float32))
            setattr(self, f"b{i+1}", np.zeros(dims[i+1], dtype=np.float32))
            self.params += [f"W{i+1}", f"b{i+1}"]
        self.n_layers = len(dims) - 1
        self.lr = lr
        self.dropout = dropout
        self._adam_init()
        self.run_mean = np.zeros(in_dim, dtype=np.float32)
        self.run_var = np.ones(in_dim, dtype=np.float32)
        self.run_count = 1e-4

    def _adam_init(self):
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def set_lr(self, lr):
        self.lr = lr

    def _normalize(self, x, update_stats=False):
        if update_stats:
            n = x.shape[0]
            delta = x.mean(0) - self.run_mean
            tot = self.run_count + n
            self.run_mean += delta * (n / tot)
            self.run_var += (x.var(0) - self.run_var) * (n / tot)
            self.run_count = tot
        return (x - self.run_mean) / np.sqrt(self.run_var + 1e-8)

    def forward(self, x, cache=False, train=False):
        x = np.asarray(x, np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        xn = self._normalize(x, update_stats=train)
        acts, zs, a = [xn], [], xn
        for i in range(1, self.n_layers + 1):
            z = a @ getattr(self, f"W{i}") + getattr(self, f"b{i}")
            zs.append(z)
            a = relu(z) if i < self.n_layers else z
            if i < self.n_layers and train and self.dropout > 0:
                m = (np.random.default_rng(self.t or 0).random(a.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
                a = a * m
            acts.append(a)
        if cache:
            return acts[-1], {"acts": acts, "zs": zs}
        return acts[-1][0] if single else acts[-1]

    def predict(self, x):
        return self.forward(x)

    def _backward(self, states, actions, targets, ws, grad_clip=10.0):
        batch = states.shape[0]
        out, c = self.forward(states, cache=True, train=True)
        td = out[np.arange(batch), actions] - targets
        dout = np.zeros_like(out)
        dout[np.arange(batch), actions] = 2.0 * (ws * td) / batch
        grads, da = {}, dout
        for i in range(self.n_layers, 0, -1):
            W = getattr(self, f"W{i}")
            grads[f"W{i}"] = c["acts"][i-1].T @ da
            grads[f"b{i}"] = da.sum(0)
            if i > 1:
                da = (da @ W.T) * relu_grad(c["zs"][i-2])
        self._adam_update(grads, grad_clip)
        return float(np.mean(ws * td ** 2))

    def train_step(self, s, a, t, grad_clip=10.0):
        return self._backward(s, a, t, np.ones(len(a), np.float32), grad_clip)

    def weighted_train_step(self, s, a, t, ws, grad_clip=10.0):
        return self._backward(s, a, t, ws, grad_clip)

    def _adam_update(self, grads, grad_clip):
        self.t += 1
        for p in self.params:
            g = np.clip(grads[p], -grad_clip, grad_clip)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * g ** 2
            setattr(self, p, getattr(self, p) - self.lr * (self.m[p] / (1 - self.beta1 ** self.t)) / (np.sqrt(self.v[p] / (1 - self.beta2 ** self.t)) + self.eps))

    def get_weights(self):
        return {p: getattr(self, p).copy() for p in self.params}

    def set_weights(self, w):
        for p in self.params:
            setattr(self, p, w[p].copy())

    def soft_update(self, other, tau):
        for p in self.params:
            setattr(self, p, (tau * getattr(other, p) + (1 - tau) * getattr(self, p)).astype(np.float32))

    def input_gradient(self, instance, action_idx):
        x = np.asarray(instance, np.float32)[None, :]
        out, c = self.forward(x, cache=True, train=False)
        d = np.zeros_like(out)
        d[0, action_idx] = 1.0
        for i in range(self.n_layers, 0, -1):
            W = getattr(self, f"W{i}")
            if i > 1:
                d = (d @ W.T) * relu_grad(c["zs"][i-2])
        return (d @ self.W1.T / np.sqrt(self.run_var + 1e-8))[0]
