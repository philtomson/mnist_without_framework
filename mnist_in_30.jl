using MLDatasets
using LinearAlgebra
using Statistics
#using Infiltrator
using PyCall
np = pyimport("numpy")
using Debugger
using ImageShow


       
function feed_forward(X, weights)
    a = [float(X)]
#    @show a
#    @show size(a)
#    @show typeof(a)
    #convert(Array{Array{Float64,2},1},a)
    for w in weights
        #w = convert(Array{Float32,2},w)
#        @show typeof(w)
#        println("size(a[end]):",size(a[end])) #784x20
#        println("size(w):", size(w)) #784x100
 #(20x784)*(784x100)
#        println(size(a[1,1]))
        #append!(a, maximum(dot(a[end],w),0)) #the relu
        p = max.(w'*a[end],0)
        #convert(Array{Float32,2}, p)
        push!(a,p)
    end
    return a
end

#Calculate Gradients
function grads(X, Y, weights)
    grads = similar(weights)
    a = feed_forward(X, weights)
    #Python:
    #a[0].shape = (20,784)
    #a[1].shape = (20,100)
    #a[2].shape = (20,10)
    #Julia:
    #size(aa[1]) (784, 20)
    #size(aa[2]) (100, 20)
    #size(aa[3]) (10, 20)

    #@show size(a)    # 3
    #@show size(a[1]) # (784, 20)
    #@show size(a[2]) # (100, 20)
    #@show size(a[3]) # (10,  20)
    #@show size(a[end]) #(10,20) Python: (20,10)
    #@show size(Y) #(20,10) same
    delta = a[end]' - Y # difference between expected (Y) and final layer output (a[end])
    #@show size(delta) # (20, 10)
    #size(delta) : (20,10)
    #@show size(a[end-1]') #(20,100)
    #was: grads[end] = dot(a[end-1],delta)
    grads[end] = a[end-1]*delta 
    #@show size(grads[end]) # (100, 10)
    #for i in range(len(a)-2, 0, -1)
    #@show length(a)
    #for i in Iterators.reverse(1:length(a)-1)
    i = 2
    #    println(" i = 2 now")
        #@show i
        #delta = (a[i] .> 0) .* dot(delta, transpose(weights[i]))
        #@show size(delta)
    #    @show size(weights[i])
        dw = (delta * weights[i]')
    #    @show size(dw) # (20, 100)
        relu = (a[i] .> 0)
    #    @show size(relu)[1] # (100)
    #    @show size(relu)    # (100,20)
        #size(relu): Julia: (100,20), Python: (20,784) **DISCREPANCY**
        #delta = dw * relu #relu * dw
        delta = dw .* relu'
    #    @show size(a[i-1]) #(784, 20)
    #    @show size(delta) #(20,20) : python (20,100) *** DISCREPANCY ***
        # was: grads[i-1] = dot(a[i-1]',delta)
        # was: grads[i-1] = delta * a[i-1]  #* delta
        
        grads[i-1] = a[i-1] * delta   #* delta
    #end
    #@show size(grads[1]) #(20, 784)   Python: (784,100) DISCRPEANCY
    #@show size(grads[2]) #(100, 10)
    #@show size(X)
    tmp_g = grads ./ size(X)[2]
    #@show size(tmp_g[1])
    #@show size(tmp_g[2])
    #@show grads
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
@show typeof(teX)
@show size(teX)
@show typeof(teY)
@show size(teY)
t = size(trX)
trX = reshape(trX, (t[1]*t[2]), t[3])
t = size(trX)
t = size(teX)
teX = reshape(teX, (t[1]*t[2]), t[3])
 #teY = reshape(teY, (t[1]*t[2]), t[3])

trainset = (trX,trY)
testset  = (teX,teY)
## Helper funcs ###
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
weights = [randn(w) * 0.1 for w in [(784, 100), (100, 10)]]
#weights = convert(Array{Array{Float32,2},1}, weights)

##load Python weights instead:
#weights = np.load("weights.npy", allow_pickle=true)
#weights = convert(Array{Array{Float64,2},1}, weights)
num_epochs, batch_size, learn_rate = 30, 20, 0.1


function run(num_epochs, trX, trY, teX, teY, batch_size, weights, learn_rate )
    weights_prev = copy(weights)
    for i in 1:num_epochs
        print("epoch: $i")
        for j in 1:batch_size:size(trX)[2]
            #global weights
            X, Y = trX[:,j:j+batch_size-1], trY[j:j+batch_size-1,:]
            gs = grads(X, Y, weights)
            #size(weights[1]) is: (784,100)
            #size(gs[1])      is: (20,784) : in python: (784,100) *** DISCREPENCY ***
            #size(gs[2])      is: (100,10) : in python: (100,10)
            scaled_gs = learn_rate .* gs
            #@show size(scaled_gs)
            #@show size(scaled_gs[1])
            #@show size(scaled_gs[2])
            #@show gs
            weights -= learn_rate .* gs #where the error crops up
        end
        #TODO: argmax here not working like np.argmax!!!!
        ff = feed_forward(teX, weights)
        #@show size(ff)#(3,)
        #@show size(ff[1])#(784, 10000)
        #@show size(ff[2])#(100, 10000)
        #@show size(ff[3])#(10,  10000)
        #@show size(teX) #(784, 10000)
        #was: prediction = argmax(ff[end], dims=1) #trouble here
        prediction = [ x[1] for x in (argmax(ff[end], dims=1))[1,:]]
        prediction = [i[1]-1 for i in prediction]
        equiv = (prediction .== teY) #this was possibly the problem
        println(" accuracy%: $(mean(equiv))")
        println()
    end
    weights_diff = weights - weights_prev
    return weights, weights_diff
end
#@show teY

#@enter run(num_epochs, trX, trY, teX, teY, batch_size, weights, learn_rate )
run(num_epochs, trX, trY, teX, teY, batch_size, weights, learn_rate )
#run(3, trX, trY, teX, teY, batch_size, weights, learn_rate )