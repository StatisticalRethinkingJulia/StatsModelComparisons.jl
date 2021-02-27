using Random

# CVIT - Create itr and itst indeces for k-fold-cv
#
#    Description
#     [ITR,ITST]=CVITR(N,K) returns 1xK cell arrays ITR and ITST holding 
#      cross-validation indeces for train and test sets respectively. 
#      K-fold division is balanced with all sets having floor(N/K) or 
#      ceil(N/K) elements.
#
#     [ITR,ITST]=CVITR(N,K,RS) with integer RS=true also makes random 
#      permutation, using substream RS. This way different permutations 
#      can be produced with different RS values, but same permutation is 
#      obtained when called again with same RS. Function restores the 
#      previous random stream before exiting.
#


# Copyright (c) 2010 Aki Vehtari

# This software is distributed under the GNU General Public
# License (version 2 or later); please refer to the file
# License.txt, included with the software, for details.
    
function cvit(n, k=10, rsubstream=false)

    a = k-rem(n,k)
    b = floor(Int, n/k);

    itst = Any[]
    itr = Any[]

    for cvi in 1:a
        push!(itst, collect(1:b) .+ (cvi-1) * b)
        push!(itr, setdiff(1:n,itst[cvi])) 
    end
    for cvi in (a+1):k
        push!(itst, (a * b) + collect(1:(b + 1)) + (cvi - a - 1) * (b + 1)) 
        push!(itr, setdiff(1:n,itst[cvi])) 
    end  

    if rsubstream
        rng = MersenneTwister()
        rii = randperm(rng, n)
        for cvi in 1:k
            itst[cvi] = rii[itst[cvi]]
            itr[cvi] = rii[itr[cvi]]
        end
    end
    itr, itst
end

