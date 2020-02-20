import numpy as np
import mnist

def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    print(f"   feed_forward:")
    print(f"      len(a): {len(a)}")
    print(f"      a[0].shape: {a[0].shape}")
    print(f"      a[1].shape: {a[1].shape}")
    print(f"      a[2].shape: {a[2].shape}")
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)
    print("grads call:")
    print(f"   X.shape: {X.shape}")
    print(f"   Y.shape: {Y.shape}")
    print(f"   len(a): {len(a)}")
    print(f"   a[-1].shape: {(a[-1]).shape}")
    print(f"   a[-2].shape: {(a[-2]).shape}")
    print(f"   a[0].shape: {(a[0]).shape}")
    print(f"   a[1].shape: {(a[1]).shape}")
    print(f"   a[2].shape: {(a[2]).shape}")
    print(f"   Y.shape: {Y.shape}") #(20,10)
    delta = a[-1] - Y
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a)-2, 0, -1):
        print(f"   layer i is: {i}")
        print("   (weights[i].T).shape: " + str((weights[i].T).shape)) 
        delta = (a[i] > 0) * delta.dot(weights[i].T)
        print(f"   delta.shape: {delta.shape}")
        grads[i-1] = a[i-1].T.dot(delta)
        print(f"   grads[0].shape: {grads[0].shape}")
        print(f"   grads[1].shape: {grads[1].shape}")
    print("")
    return grads / len(X)

trX, trY, teX, teY = mnist.load_data()
weights = [np.random.randn(*w) * 0.1 for w in [(784, 100), (100, 10)]]
#uncomment following 2 lines to use saved weights
#np.save("weights.npy", weights)
#weights = np.load("weights.npy", allow_pickle=True)
num_epochs, batch_size, learn_rate = 30, 20, 0.1

for i in range(num_epochs):
    for j in range(0, len(trX), batch_size):
        # X is input, Y is expected
        X, Y = trX[j:j+batch_size], trY[j:j+batch_size]
        scaled_gs = learn_rate * grads(X, Y, weights)
        weights -= scaled_gs
        print(f"   scaled_gs.shape: {scaled_gs.shape}")
        print(f"   scaled_gs[0].shape: {scaled_gs[0].shape}")
        print(f"   scaled_gs[1].shape: {scaled_gs[1].shape}")
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print(f"i={i}, accuracy: {np.mean(prediction == np.argmax(teY, axis=1))}")
