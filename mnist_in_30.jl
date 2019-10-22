using MLDatasets
using LinearAlgebra
using Debugger
break_on(:error)


       
function feed_forward(X, weights)
    a = [float(X)]
#    @show a
#    @show size(a)
#    @show typeof(a)
    #convert(Array{Array{Float64,2},1},a)
    for w in weights
        w = convert(Array{Float32,2},w)
#        @show typeof(w)
#        println("size(a[end]):",size(a[end])) #784x20
#        println("size(w):", size(w)) #784x100
 #(20x784)*(784x100)
#        println(size(a[1,1]))
        #append!(a, maximum(dot(a[end],w),0)) #the relu
        p = max.(w'*a[end],0)
        #convert(Array{Float32,2}, p)
#        @show p
        @show typeof(p)
        @show size(p)
        #append!(a, p)
        push!(a,p)
    end
    return a
end

#Calculate Gradients
function grads(X, Y, weights)
    println()
    println(">>> grads()")
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

    @show size(a[end])
    @show size(Y)
    delta = a[end]' - Y
    @show size(delta)
    #size(delta) : (20,10)
    @show size(a[end-1]')
    #was: grads[end] = dot(a[end-1],delta)
    grads[end] = a[end-1]*delta
    #for i in range(len(a)-2, 0, -1)
    @show length(a)
    #for i in Iterators.reverse(1:length(a)-1)
    i = 2
        #@show i
        #delta = (a[i] .> 0) .* dot(delta, transpose(weights[i]))
        #@show size(delta)
        @show size(weights[i])
        dw = (delta * weights[i]')
        @show size(dw)
        @show size(a[i] .> 0) #this is the relu
        relu = (a[i] .> 0)
        @show size(relu)[1]
        @show size(relu)
        #size(relu): Julia: (100,20), Python: (20,784) **DISCREPANCY**
        delta = dw * relu #relu * dw
        @show size(a[i])
        @show size(delta) #(20,20) : python (20,100) *** DISCREPANCY ***
        # was: grads[i-1] = dot(a[i-1]',delta)
        grads[i-1] = delta * a[i-1]'  #* delta
    #end
    @show size(grads[1])
    @show size(grads[2])
    @show size(X)
    tmp_g = grads ./ length(X)
    @show size(tmp_g[1])
    @show size(tmp_g[2])
    println("<<< grads")
    return grads ./ length(X)
end

breakpoint(grads)

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

weights = [randn(w) * 0.1 for w in [(784, 100), (100, 10)]]
weights = convert(Array{Array{Float32,2},1}, weights)
num_epochs, batch_size, learn_rate = 30, 20, 0.1

for i in 1:num_epochs
    @show i
    for j in 1:length(trX):batch_size
        @show j
        global weights
        X, Y = trX[:,j:j+batch_size-1], trY[j:j+batch_size-1,:]
        gs = grads(X, Y, weights)
        @show size(gs[1])
        @show size(gs[2])
        #size(weights[1]) is: (784,100)
        #size(gs[1])      is: (20,784) : in python: (784,100) *** DISCREPENCY ***
        #size(gs[2])      is: (100,10) : in python: (100,10)
        weights -= learn_rate .* gs
    end
    prediction = argmax(feed_forward(teX, weights)[-1], 2)
    @show prediction
    println(i, mean(prediction == argmax(teY, 2)))
    println()
end
