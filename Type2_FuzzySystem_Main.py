import Type2_FuzzySystem_Functions as it2fuzz
import numpy as np

'''
input format:
x -- the starting value of the range.
y -- the upper bound of the range
z -- the incrementation value (the difference two values of the range)
'''
x, y, z = [float(x) for x in input("Enter the range of weight:").split(' ')]
x_weight = np.arange(x, y, z)

x, y, z = [float(x) for x in input("Enter the range of height:").split(' ')]
x_height = np.arange(x, y, z)

x, y, z = [float(x) for x in input("Enter the range of fitnes level:").split(' ')]
x_fitnessLevel = np.arange(x, y, z)


'''
w -- 2D tuple containing seperate UMF and LMF of weight(here).
w_types -- list containg the linguistic terms for weight(here).
'''
w, w_types = it2fuzz.fuzz_IT2_Inputs(x_weight, 'weight')


h, h_types = it2fuzz.fuzz_IT2_Inputs(x_height, 'height')


f, f_types = it2fuzz.fuzz_IT2_Inputs(x_fitnessLevel, 'fitness level')



'''
Ploting the membership values for each of antecedent and consequent
'''
it2fuzz.fuzz_IT2_plot_mf(x_weight, w, w_types, 'Weight')
it2fuzz.fuzz_IT2_plot_mf(x_height, h, h_types, 'Height')
it2fuzz.fuzz_IT2_plot_mf(x_fitnessLevel, f, f_types, 'Fitness Level')



'''
Displays the list of linguistic terms of the consequent, corresponding to a value.
'''
for i in range(len(f_types)):
    print(f'{i+1}) {f_types[i]}')



'''
rule_lst -- list of rules decided
'''
rule_lst = it2fuzz.fuzz_make_rules(w_types, h_types)


weight = int(input('Enter Value for weight:'))
height = float(input('Enter value for height:'))


'''
x_memvalue -- membership value at a particular single value for the antecedent x(weight and height here).
'''
w_memvalue = it2fuzz.fuzz_IT2_Interplot_mem(x_weight, w, weight)
h_memvalue = it2fuzz.fuzz_IT2_Interplot_mem(x_height, h, height)


'''
rule -- 2D tuple of maped rule for upper and lower membership values.
fitness_used -- list of fitness values decided based the rule_lst(to be used for ploting)
'''
rule, fitness_used = it2fuzz.fuzz_mapRule(w_memvalue, h_memvalue, f, rule_lst)

it2fuzz.fuzz_plot_outputMf(x_fitnessLevel, rule, fitness_used)


'''
R_combined -- 2D tuple containing aggregated rule for upper and lower membership values.
'''
R_combined = it2fuzz.fuzz_IT2_aggregation(rule)



'''
fitnessLevel -- output value(centroid value)
fitness_activation -- corresponding membership value of output
'''
fitnessLevel, fitness_activation = it2fuzz.fuzz_IT2_defuzz(x_fitnessLevel, R_combined)


it2fuzz.fuzz_IT2_output(x_fitnessLevel, f, fitnessLevel, fitness_activation, R_combined)


