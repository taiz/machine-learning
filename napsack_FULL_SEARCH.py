# coding:utf-8

import time

#
# 問題設定
#
#CANDIES = [(2,2), (1,2), (3,6), (2,1), (1,3), (5,85)] # (weight,value)
CANDIES = [(2,2), (1,2), (3,6), (2,1), (1,3), (5,85), (1,1), (1,1), (1,1), (1,1), (1,1)] # (weight,value)
W = 8

start = time.time()

def search(candies, total_weight):
    if len(candies) == 0:
        return 0
    candy = candies[0]
    weight, value  = candy
    if total_weight + weight > W:
        return 0
    return max(value + search(candies[1:], weight + total_weight), search(candies[1:], total_weight))

print('answer:', search(CANDIES, 0))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
