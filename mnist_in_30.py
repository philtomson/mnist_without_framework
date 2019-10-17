import numpy as np
import mnist

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])
    
def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights):
    print("grads() called")
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)
    #a[0].shape = (20,784)
    #a[1].shape = (20,100)
    #a[2].shape = (20,10)
    delta = a[-1] - Y #delta.shape: (20,10), Y.shape:(20,10)
    print(f"   a[-1].shape: {(a[-1]).shape}")
    print(f"   a[-2].shape: {(a[-2]).shape}")
    #(a[-1]).shape:   (20,100)
    #(a[-2].T).shape: (100,20)
    grads[-1] = a[-2].T.dot(delta) #grads[-1].shape: (100,10)
    print(f"len(a) is: {len(a)}")
    i = 1
    #for i in range(len(a)-2, 0, -1):
    print(f'  in loop i: {i}')
    #print("   delta.shape: "+str(delta.shape))
    print("   (weights[i].T).shape: " + str((weights[i].T).shape))
    delta = (a[i] > 0) * delta.dot(weights[i].T)
    print("   delta.shape: "+str(delta.shape)) #(20,100)
    grads[i-1] = a[i-1].T.dot(delta)
    print(f"   grads.shape: {grads.shape}")
    return grads / len(X)

trX, trY, teX, teY = mnist.load_data()
weights = [np.random.randn(*w) * 0.1 for w in [(784, 100), (100, 10)]]
num_epochs, batch_size, learn_rate = 30, 20, 0.1

for i in range(num_epochs):
    print(f'i is: {i}')
    for j in range(0, len(trX), batch_size):
        print(f'j is {j}')
        X, Y = trX[j:j+batch_size], trY[j:j+batch_size]
        print(f'X.shape: {X.shape}')
        print(f'Y.shape: {Y.shape}')
        gs  = grads(X, Y, weights)
        print(f'gs[0].shape is: {(gs[0]).shape}')
        print(f'gs[1].shape is: {(gs[1]).shape}')
        print(f'weights[0].shape is: {(weights[0]).shape}')
        print(f'weights[1].shape is: {(weights[1]).shape}')
        weights -= learn_rate * gs
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print(f'Epoch: {i}, {np.mean(prediction == np.argmax(teY, axis=1))}')
