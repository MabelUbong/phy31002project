import torch


# Tensors are basically numpy arrays and can have functions performed on them

t = torch.tensor([[1,2,3,4],
                  [5,6,7,8]], dtype=torch.float32)

#print('t size', t.size())

t1 = torch.tensor([[0,1],
                   [2,3],
                   [4,5],
                   [6,7]], dtype=torch.float32)
#print('t1 size', t1.size())

#Tensors can be re-shaped

#print('t shape', t.shape)
#print('t1 shape', t1.shape)

t1 = t1.reshape([2,4])
#print('New t1 shape after re-shaping', t1.shape)

#We can also flatten a Tensor

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

t3 = torch.ones(4, 3)

#print(t3)
#Now we use the flatten function

#print(flatten(t3))
#This takes the elements from the next row and adds them to the end of the previous row

#We can combine Tensors using the concatenate function

t4 = torch.tensor([
    [1,2],
    [3,4]
], dtype=torch.float32)
t5 = torch.tensor([
    [5,6],
    [7,8]
], dtype=torch.float32)
#print(torch.cat((t4,t5), dim=0))
#The 'dim=0' is so the cat func knows to do it row-wise.
#print(torch.cat((t4,t5), dim=1))

#Now for an image example, the first image is identified by 0's, 2nd by 1's and 3rd by 2's

T1 = torch.zeros(4,4, dtype=torch.float32)
T2 = torch.ones(4,4, dtype=torch.float32)
T3 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
], dtype= torch.float32)

T = torch.stack((T1,T2,T3))
#print(T.shape)
#print(T)
#Now we want this of the for [Batch, Colour, Height, Width]

T = T.reshape(3,1,4,4)
#print(T)
#Thie has added an extra axes

#We can index this  combined image to locate specific elements of each image

#print('First colour of first image', T[0][0])
#print('First row of pixels in the first colour of the first image', T[0][0][0])

#For convolutional neural networks we will need flat tensors

#print(T.flatten())

#We can choose which axes we flatten too
#print('We have flattened from the colour axes onwards so the images are still separated in batches', T.flatten(start_dim=1))

#Tensors can be computed on 'element-wise'

t6 = torch.ones(2,2)
t7 = torch.tensor([
    [3,4],
    [5,6]
], dtype=torch.float32)
t8=t6+t7
t9=t6-t7
t10=t6*t7
t11=t6/t7
#print('Add', t8)
#print('Subtract', t9)
#print('Multiply', t10)
#print('Divide', t11)

#Broadcasting

#When computing two tensors they need to be the same size and when they arent the smaller one is broadcast onto the larger one

t12 = torch.tensor([
    [1,2],
    [7,8]
], dtype=torch.float32)

t13 = torch.tensor([4,5], dtype=torch.float32)

t14 = t12+t13
#print(t14)
#Here the bottom row elements are just a mirror of the top row elements to add to the other tensor

#Can also compute comparisons

#print(T.eq(0)) #Equal to 0
#print(T.le(3)) #Less than or equal to 3

#Each element is compared to the number in the function

