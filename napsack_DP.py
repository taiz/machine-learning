# coding:utf-8

import time

#
# 問題設定
#
#CANDIES = [(2,2), (1,2), (3,6), (2,1), (1,3), (5,85)] # (weight,value)
CANDIES = [(2,2), (1,2), (3,6), (2,1), (1,3), (5,85), (1,0), (1,0), (1,0), (1,0), (1,0)] # (weight,value)

W = 8
N = len(CANDIES)

start = time.time()

dp = [[0] * (W+1) for i in range(N+1)]

for i in range(N):
    weight, value = CANDIES[i]
    for j in range(W+1):
        if j >= weight:
            dp[i+1][j] = max([ dp[i][j-weight] + value, dp[i][j] ])
        else:
            dp[i+1][j] = dp[i][j]

print('answer:', dp[N][W])

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
