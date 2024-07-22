using LinearAlgebra
using Distributions

# Include the dictionary files
include("dictionaries/common_probs.jl")
include("dictionaries/common_values.jl")
include("dictionaries/unique_probs.jl")
include("dictionaries/unique_values.jl")
include("dictionaries/shared_probs.jl")
include("dictionaries/shared_values.jl")

# Access the dictionaries through the module names
common_prob_dict = common_probs.common_prob_dict
common_value_dict = common_values.common_value_dict
unique_prob_dict = unique_probs.unique_prob_dict
unique_value_dict = unique_values.unique_value_dict
shared_prob_dict = shared_probs.shared_prob_dict
shared_value_dict = shared_values.shared_value_dict

# Calculate expected values for each table
function calculate_expected_value(probs, values)
    return sum(probs[k] * values[k] for k in keys(probs))
end

EV_common = calculate_expected_value(common_prob_dict, common_value_dict)
EV_unique = calculate_expected_value(unique_prob_dict, unique_value_dict)
EV_shared = calculate_expected_value(shared_prob_dict, shared_value_dict)

# Probability of hitting the unique table
prUnique = 1/16

# Probability of getting a reroll token (alpha)
alpha = (1/16) * (15/16)^3  # Probability of hitting shared table on last roll and getting a token

# Distribution of gold value from a single casket opening (2-4 items)
function calculate_DistY(num_rolls_min, num_rolls_max)
    DistY = zeros(num_rolls_max - num_rolls_min + 1)
    for num_rolls in num_rolls_min:num_rolls_max
        roll_values = []
        for i in 1:num_rolls
            # Probability of hitting unique table
            p_unique = 1/16
            
            # Calculate expected value for this roll
            ev_unique = sum(unique_prob_dict[k] * unique_value_dict[k] for k in keys(unique_prob_dict))
            ev_common = sum(common_prob_dict[k] * common_value_dict[k] for k in keys(common_prob_dict))
            
            # For the last roll, we might hit the shared table
            if i == num_rolls
                ev_shared = sum(shared_prob_dict[k] * shared_value_dict[k] for k in keys(shared_prob_dict))
                ev_roll = p_unique * ev_unique + (1 - p_unique) * ((15/16) * ev_common + (1/16) * ev_shared)
            else
                ev_roll = p_unique * ev_unique + (1 - p_unique) * ev_common
            end
            
            push!(roll_values, ev_roll)
        end
        DistY[num_rolls - num_rolls_min + 1] = sum(roll_values)
    end
    raw_ev = mean(DistY)
    return DistY / sum(DistY), raw_ev  # Return both normalized distribution and raw expected value
end

# Calculate DistY and raw expected value
DistY, raw_ev = calculate_DistY(2, 4)
println("Raw expected value per casket: ", raw_ev)

yDistInd = 1:length(DistY)

# Define other necessary variables
GlobeTrotter = 2
maxI = 11 + 3 * GlobeTrotter
maxX = 10000  # Number of caskets
maxJ = 5  # Maximum number of reroll tokens

# Calculate alpha (probability of getting a reroll token)
alpha = (1/16) * (15/16)^3  # Probability of hitting shared table on last roll and getting a token


for y in 0:4
    V[1, 1:3, y+1, :, :] .= y
    sRR[1, 1:3, y+1, :, :] .= 0
    sRT[1, 1:3, y+1, :, :] .= 0
end
## When I am on my last casket, but have rerolls available.
for i in 3:maxI
    for y in 0:4
        ## This is the option value of the reroll
        ## Can't depend on reroll tokens since we are on the last casket.
        opt2 = sum(DistY .* (V[1, i-2, yDistInd, 1, 1]))
        if y > opt2
            V[1, i+1, y+1, :, :] .= y
            sRR[1, i+1, y+1, :, :] .= 0
            sRT[1, i+1, y+1, :, :] .= 0
        else
            V[1, i+1, y+1, :, :] .= opt2
            sRR[1, i+1, y+1, :, :] .= 1
            sRT[1, i+1, y+1, :, :] .= 0
        end
    end
end

## Main loop, order on j is important, as we use the cap to create
## a boundary that makes this able to be solved with simple
## backwards induction.

## There are a shit ton of states now:

## I have x caskets remaining,
## I have i progress on rerolls
## I have j reroll tokens in my bank

## Casket is then opened:
## There are y fortunate components in the casket,
## If k = 2, then there is a reroll token in the casket.
for x in 2:maxX
    for i in 0:2
        for j in 4:-1:0
            for y in 0:4
                for k in 1:2
                    ## Case: No reroll available, no reroll tokens left in bank/casket
                    if j == 0 && k == 1
                        ## I gain y value, and my states moves as follows:
                        ## x: -1, i: + 1, j: + (k-1), 
                        V[x, i+1, y+1, j+1, k] = y +
                            sum(DistY .* (alphaC * V[x-1, i+2, yDistInd, j+k, 1] +
                            alpha * V[x-1, i+2, yDistInd, j+k, 2]))
                        sRR[x, i+1, y+1, j+1, k] = 0
                        sRT[x, i+1, y+1, j+1, k] = 0
                    else
                        ## Reroll not available, but there are
                        ## tokens in the bank/casket. If I use the
                        ## token, my states move by i+3, but j-1
                        ## for future caskets.
                        useTokenVal = y +
                            sum(DistY .* (alphaC *
                            V[x-1, i+5, yDistInd, min(j + k - 1, maxJ), 1] +
                            alpha * V[x-1, i+5, yDistInd, min(j + k - 1, maxJ), 2]))

                        nUseTVal = y +
                            sum(DistY .* (alphaC *
                            V[x-1, i+2, yDistInd, min(j + k, maxJ), 1] +
                            alpha * V[x-1, i+2, yDistInd, min(j + k, maxJ), 2]))

                        if useTokenVal > nUseTVal
                            V[x, i+1, y+1, j+1, k] = useTokenVal
                            sRR[x,i+1,y+1,j+1,k] = 0
                            sRT[x,i+1,y+1,j+1,k] = 1
                        else
                            V[x, i+1, y+1, j+1, k] = nUseTVal
                            sRR[x,i+1,y+1,j+1,k] = 0
                            sRT[x,i+1,y+1,j+1,k] = 0
                        end
                    end
                end
            end
        end
    end
    for i in 3:maxI
        for j in 4:-1:0
            for y in 0:4
                for k in 1:2
                    ## Rerolls are available, but no tokens available.
                    if j == 0 && k == 1
                        opt1NT = y +
                            sum(DistY .* (alphaC *
                            V[x-1, min(i + 2, maxI + 1), yDistInd, min(j + k, maxJ), 1] +
                            alpha * V[x-1, min(i + 2, maxI + 1), yDistInd, min(j + k, maxJ), 2]))
                        ## If I choose to reroll, I gain the average value of the state:
                        ## (x,i-2,j)
                        opt2NT = sum(DistY .* (alphaC *
                            V[x, i-1, yDistInd, min(j + 1, maxJ), 1] +
                            alpha * V[x, i-1, yDistInd, min(j + 1, maxJ), 2]))
                        if opt1NT > opt2NT
                            V[x, i+1, y+1, j+1, k] = opt1NT
                            sRR[x, i+1, y+1, j+1, k] = 0
                            sRT[x, i+1, y+1, j+1, k] = 0
                        else
                            V[x, i+1, y+1, j+1, k] = opt2NT
                            sRR[x, i+1, y+1, j+1, k] = 1
                            sRT[x, i+1, y+1, j+1, k] = 0
                        end
                    else
                        ## Keep casket, don't use token.
                        opt1NT = y + sum(DistY .* (alphaC *
                            V[x-1, min(i + 2, maxI + 1), yDistInd, min(j + k, maxJ), 1] +
                            alpha * V[x-1, min(i + 2, maxI + 1), yDistInd, min(j + k, maxJ), 2]))
                        ## Reroll casket, don't use token.
                        ## Note that when I reroll a casket, I
                        ## don't gain the reroll token that was
                        ## contained in it
                        opt2NT = sum(DistY .* (alphaC *
                            V[x, i-1, yDistInd, min(j + 1, maxJ), 1] +
                            alpha * V[x, i-1, yDistInd, min(j + 1, maxJ), 2]))

                        ## Keep casket, use reroll token
                        opt1T = y + sum(DistY .* (alphaC *
                            V[x-1, min(i + 5, maxI + 1), yDistInd, min(j + k-1, maxJ), 1] +
                            alpha * V[x-1, min(i + 5, maxI + 1), yDistInd, min(j+k-1, maxJ), 2]))

                        ## Note that if you reroll, you cannot use a
                        ## token before observing the reroll
                        ## values. So opt2T does not exist.

                        if opt1NT > opt2NT && opt1NT > opt1T
                            V[x, i+1, y+1,j+1,k] = opt1NT
                            sRR[x, i+1, y+1,j+1,k] = 0
                            sRT[x, i+1, y+1,j+1,k] = 0
                        elseif opt2NT > opt1NT && opt2NT > opt1T
                            V[x, i+1, y+1,j+1,k] = opt2NT
                            sRR[x, i+1, y+1,j+1,k] = 1
                            sRT[x, i+1, y+1,j+1,k] = 0
                        else
                            V[x, i+1, y+1,j+1,k] = opt1T
                            sRR[x, i+1, y+1,j+1,k] = 0
                            sRT[x, i+1, y+1,j+1,k] = 1
                        end
                    end
                end
            end
        end
    end
end
return V,sRR,sRT
end



V,sRR,sRT = FillV( maxX, DistY, alphaC, alpha, yDistInd, maxJ, maxI )

mean( sRR[1,:,:,i,1] for i in 1:5)
mean( sRR[2,:,:,i,1] for i in 1:5)
mean( sRR[maxX-1,:,:,i,1] for i in 1:5)
mean( sRR[maxX,:,:,i,1] for i in 1:5)

mean( sRR[1,:,:,i,2] for i in 1:5)
mean( sRR[2,:,:,i,2] for i in 1:5)
mean( sRR[maxX-1,:,:,i,2] for i in 1:5)
mean( sRR[maxX,:,:,i,2] for i in 1:5)


mean( sRT[1,:,:,i,j] for i in 1:5, j in 1:2)
mean( sRT[2,:,:,i,j] for i in 1:5, j in 1:2)
mean( sRT[maxX,:,:,i,j] for i in 1:5, j in 1:2)


alpha*sum(DistY .* V[maxX,1,yDistInd,1,2]) + alphaC*sum(DistY .* V[maxX,1,yDistInd,1,1])
