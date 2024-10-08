import os

node_num = 50
for i in range(10):
    print(i)
    command = f"python NSGAII.py data/{node_num}/{i}_0.txt data/{node_num}/{i}_1.txt result/{node_num}/{i}.txt"
    os.system(command)