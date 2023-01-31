# SLP-ML-assignment-
import matplotlib.pyplot as plt
import random
import time
import numpy as np
def dotProduct (p1,p2):
  return np.dot(p1,p2)



def sign(y):
  if y>0:
    return "positive"
  else:
    return "negative"


p0 = [1, 0, 0]
y0 = 1

p1 = [1, 0, 1]
y1 = 1

p2 = [1, 1, 0]
y2 = 1

p3 = [1, 1, 1]
y3 = -1

w = [0, 0,1]
learning_rate=0.2
it=0

def act_func(y):
  if y >=0:
    o = 1
  else:
    o = -1
  return o

print("dotProduct(w,p0) = ", dotProduct(w,p0))
print("dotProduct(w,p1) = ", dotProduct(w,p1))
print("dotProduct(w,p2) = ", dotProduct(w,p2))
print("dotProduct(w,p3) = ", dotProduct(w,p3))
while not(dotProduct(w,p0)>=0 and dotProduct(w,p1)>=0 and dotProduct(w,p2)>=0 and dotProduct(w,p3) <0): 
  # it +=1
  (strp,p,y) =random.choice([("p0",p0,y0),("p1",p1,y1),("p2",p2,y2),("p3",p3,y3)])  # Dataset [sample, label]  # strp: tag
  # p: sample
  # y: label
  print("strp:", strp)
  print("y = ", y)
  # print("sample = ", p)
  # continue
  print("Dot product = ", dotProduct(w, p))
  out = dotProduct(w, p) # y^

  # Err = act_func(out) - y  # 1 - 1 = 0; 1 - (-1) = 2; -1 -(-1) = 0
  if ( dotProduct(w, p) > 0 and y > 0):
    continue
  it +=1
  print("We have chosen", (strp,y), "because this point is misclassified")
  print("dotProduct of", strp, "and w is", dotProduct(p,w), "while y of", strp, "is", sign(y))
  w = [w[i] + learning_rate * (y - out) * p[i] for i in range(3)]
  print("new w=", w)
  
if (w[2]!=0):
    plt.plot([0,1],[-w[0]/w[2],(-w[0]-w[1])/w[2]],'k')
else:
  if(w[1]!=0):
    plt.plot([-w[0]/w[1],-w[0]/w[1]],[0,1],'k')

  # plt.plot([0,0],[0,1],'k')
plt.annotate('p0',(0,0))
plt.annotate('p1',(0,1))
plt.annotate('p2',(1,0))
plt.annotate('p3',(1,1))
plt.scatter(0,0,marker='o',color='b')
plt.scatter(0,1,marker='o',color='b')
plt.scatter(1,0,marker='s',color='b')
plt.scatter(1,1,marker='s',color='r')
plt.show()
# print("Now", strp, "is well classified")
# print("dotProduct of p and w =", dotProduct(p,w), "while y is",sign(y))
print('Done after {}'.format(it))
print("Done! :)")
  
