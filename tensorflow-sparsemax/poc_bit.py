import math
def bitonic_par(list_input):
  n = len(list_input)
  k = 2
  m = 2 ** int(math.floor(math.log(n, 2)))
  if m != n:
    m = int(2 ** (int(math.floor(math.log(n, 2))) + 1))
  print(m)
  while k <= m:
    j = k / 2
    #print("j", j)
    while j > 0:
      i = 0
      print("k(bit position),j(bit spacing): ", k, j)
      while i < n:
        print(j)
        ixj = (i^j) 
        
        #print(i,j, ixj)

        #print(i, ixj)
        if ixj > i and ixj < n:
            print("i(pos), ixj(partner):", i, ixj)
            print("comparing position {0} and {1} (asc: {2})".format(i, ixj, i&k==0))
            if (i&k == 0) and (list_input[i] > list_input[ixj]):
              temp = list_input[i]
              list_input[i] = list_input[ixj]
              list_input[ixj] = temp

            if (i&k != 0) and (list_input[i] < list_input[ixj]):
              temp = list_input[i]
              list_input[i] = list_input[ixj]
              list_input[ixj] = temp
        i += 1
      
      """
      i = m
      while i < n :
        i_prime = i
        ixj = i_prime^(j)
        print('boom', i_prime, ixj)
        print("comparing position {0} and {1} (asc: {2})".format(i, ixj, i&k!=0))
        if (i&k != 0) and (list_input[i] > list_input[ixj]):
          temp = list_input[i]
          list_input[i] = list_input[ixj]
          list_input[ixj] = temp

        if (i&k == 0) and (list_input[i] < list_input[ixj]):
          temp = list_input[i]
          list_input[i] = list_input[ixj]
          list_input[ixj] = temp
        i += 1
      """
      j = j / 2
    k = k * 2 

"""def bitonic_par_2(list_input):
  print("boom")
  n = len(list_input)
  number_of_rounds = math.ceil(math.log(n, 2)) +1
  print("n rounds: ", number_of_rounds)
  for direction_bit in range(number_of_rounds):
    print("", )
    for bit_comp_distance in range(direction_bit+1)[::-1]:
      print("direction, distance", direction_bit, bit_comp_distance)
      for i in range(1, n + 1):
        partner_i = i^bit_comp_distance
        print("pair", i, partner_i)
"""
def bitonic_par_3(list_input):
  N = len(list_input)
  print("3rd time lucky")
  m = 2 ** int(math.ceil(math.log(N, 2)))
  print(m)
  k = 2
  shift_allign = 0
  while k <= m:
    j = int(k / 2)

    shift_allign = shift_allign^(N % 2)
    while j > 0:
      i = 0
      while i < N:
        ixj = i^(j)  # xor to find partner
        temp_i = i + shift_allign#(int(math.ceil(k / 2)) - j) # k % 2
        temp_ixj = ixj + shift_allign #- (int(math.ceil(k / 2)) - j) # k % 2
        print(i, ixj)
        print("temp", temp_i, temp_ixj)
        #if temp_ixj > temp_i and temp_i >=0 and temp_ixj >= 0 and temp_i < N and temp_ixj < N:    
        if ixj > i and ixj < N:
            if temp_ixj == N:
              temp_ixj -=1
              temp_i -=1
            if (temp_i&k == 0) and (list_input[temp_i] > list_input[temp_ixj]):
              temp = list_input[temp_i]
              list_input[temp_i] = list_input[temp_ixj]
              list_input[temp_ixj] = temp

            if (temp_i&k != 0) and (list_input[temp_i] < list_input[temp_ixj]):
              temp = list_input[temp_i]
              list_input[temp_i] = list_input[temp_ixj]
              list_input[temp_ixj] = temp
            print(list_input)

        i += 1
      j = int(j / 2)
    k = k * 2

def compare(input_list, i, j, direction):
  print("comparing {0}, {1}. (asc: {2})".format(i, j, direction))
  print("pre comp", input_list)
  if direction == (input_list[i] > input_list[j]):
    temp = input_list[i]
    input_list[i] = input_list[j]
    input_list[j] = temp
  print("post comp", input_list)

def bitonic_merge(input_list, low, n, direction):
  if n > 1:
    #print("merging from {0} plus {1}. (asc: {2})".format(low, n, direction))
    #print("pre", input_list)
    m = 2**int(math.floor(math.log(n-0.00001, 2)))
    for i in range(low, low + n - m):
      compare(input_list, i, i + m, direction)
    #print("comp", input_list) 
    bitonic_merge(input_list, low, m, direction)
    #print("post first merge ", input_list) 
    bitonic_merge(input_list, low + m, n - m, direction)
    #print("post second merge", input_list)   

def bitonic_sort(input_list, low, n, direction):
  #print(input_list)
  if n > 1:
    #print("sorting from {0} plus {1}. (asc: {2})".format(low, n, direction))
    m = int(math.floor(n / 2))
    bitonic_sort(input_list, low, m, not direction)
    bitonic_sort(input_list, low + m, n - m, direction)
    bitonic_merge(input_list, low, n, direction)
  else:
    pass
    #print("n=1. already sorted")

def bitonic_rec(input_list, direction=True):
  n = len(input_list)
  bitonic_sort(input_list, 0, n, direction)


test = [3, 2, 1]
print(test)
bitonic_rec(test, False)
print(test)

test_2 = [4,3,8, 2, 1, 5, 6, 7]
print("")
print( test_2)
#bitonic_par(test_2)
#bitonic_par_2(test_2)
bitonic_par_3(test_2)
print(test_2)

