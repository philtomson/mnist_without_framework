using MLDatasets
using LinearAlgebra
using Statistics
using PyCall
np = pyimport("numpy")
using Debugger
using ImageShow
       
function feed_forward(X, weights::Array{Array{Float32,2},1})
    a = [float(X)]
    for w in weights
        p = max.(w'*a[end],0)
        push!(a,p)
    end
    return a
end

#Calculate Gradients
function grads(X, Y::Array{Float32,2}, weights::Array{Array{Float32,2},1})::Array{Array{Float32,2},1}
    grads = similar(weights)
    a = feed_forward(X, weights)
    delta = a[end]' - Y # difference between expected (Y) and final layer output (a[end])
    grads[end] = a[end-1]*delta 
    i = 2
        dw = (delta * weights[i]')
        relu = (a[i] .> 0)
        delta = dw .* relu'
        grads[i-1] = a[i-1] * delta   #* delta
    return grads ./ size(X)[2]
end

#breakpoint(grads)

function to_one_hot(v,classes)
   out_mat = zeros(size(v)[1],classes)
   for i in 1:length(v)
     out_mat[i,v[i]+1] = 1.0
   end
   return out_mat
end

trX, trY = MNIST.traindata()
#TODO: need to one-hot-encode the trY   
trY = to_one_hot(trY,10)

teX, teY = MNIST.testdata()
t = size(trX)
trX = reshape(trX, (t[1]*t[2]), t[3])
t = size(teX)
teX = reshape(teX, (t[1]*t[2]), t[3])
 #teY = reshape(teY, (t[1]*t[2]), t[3])

trainset = (trX,trY)
testset  = (teX,teY)
trY = convert(Array{Float32,2}, trY)
teY = convert(Array{Float32,1}, teY)

## Helper funcs  for debugging ###
#
function predict(n, mx=teX, my=teY)
    ff = feed_forward(mx, weights)
    prediction = argmax(ff[end][:,n]) - 1
    return prediction
end

function comp_tr_img(n)
    imshow(Gray.(reshape(trX[:,n], (28,28))))
    p = predict(n,trX, trY)
    expected = argmax(trY[n, :])-1
    println("expected: $expected")
    return expected
end

#show an image of the expected output
function compare_img(n,mx,my)
    imshow(Gray.(reshape(mx[:,n], (28,28))))
    p = predict(n, mx, my)
    expected = my[n,:][1]
    println("expected: $expected")
    println("predicted: $p")
    if( p != expected )
        println("MISMATCH")
    else
        println("Correct!")
    end
    return expected
end

#all weights are 0.5:
#weights = [fill(0.5, w)  for w in [(784, 100), (100, 10)]]

#random weights:
weights = [randn(Float32, w) * 0.1 for w in [(784, 100), (100, 10)]]
weights = convert(Array{Array{Float32,2},1}, weights)
@show typeof(weights)

##load Python weights instead:
#weights = np.load("weights.npy", allow_pickle=true)
#weights = convert(Array{Array{Float32,2},1}, weights)
num_epochs, batch_size, learn_rate = 3, 20, Float32(0.1)


function run(num_epochs, trX, trY, teX, teY, batch_size, w::Array{Array{Float32,2},1}, learn_rate::Float32 )
    weights::Array{Array{Float32,2},1} = w
    @show typeof(weights)
    for i in 1:num_epochs
        print("epoch: $i")
        for j in 1:batch_size:size(trX)[2]
            @inbounds X, Y = trX[:,j:j+batch_size-1], trY[j:j+batch_size-1,:]
            gs::Array{Array{Float32,2},1} = grads(X, Y, weights)
            weights -= learn_rate .* gs
        end
        ff = feed_forward(teX, weights)
        prediction = [x[1] for x in (argmax(ff[end], dims=1))[1,:]]
        prediction = [i[1]-1 for i in prediction]
        equiv = (prediction .== teY) #this was possibly the problem
        println(" accuracy%: $(mean(equiv))")
    end
end

#@enter run(num_epochs, trX, trY, teX, teY, batch_size, weights, learn_rate )
@time run(num_epochs, trX, trY, teX, teY, batch_size, weights, learn_rate )