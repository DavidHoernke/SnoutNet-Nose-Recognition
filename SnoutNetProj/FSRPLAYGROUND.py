a = [1,1,0,1,1,0]
a5 = 0
a4 = 1
a3= 2
a2 = 3
a1 = 4
a0 = 5


count =100
stg = ""
for i in range(0,100):
    # temp = (a[a3]*a[a1]) % 2
    # temp = (temp + a[a5]) % 2
    temp = (a[a3] + a[a0])%2

    stg = stg + str(a[a0])
    a.insert(0,temp)
    a.pop()

print(stg)