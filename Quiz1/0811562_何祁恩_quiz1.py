import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

#Problem 1:
def Q1():
  a = list(range(1000))
  print(a)
  print(sum(a))

#Problem 2:
def Q2():
  a = np.random.randint(1,10, size=(100,))
  f = np.zeros((9,), dtype = int)
  print(a)
  for x in a:
    f[x-1] = f[x-1] + 1
  
  for i in range(len(f)):
    print('%2d elements for label %2d' %(f[i], i+1))

#Problem 3:
def Q3():
  list1=[]
  list2=[]
  for i in range(50, 100):
    list1.append(i)

  for i in range(1, 6):
    for j in range(10):
      list2.append(i)

  #shuffle data
  data_temp=np.c_[list1,list2]
  np.random.shuffle(data_temp)

  list1=data_temp[:,0]
  list2=data_temp[:,1]

  print(list1)
  print(list2)

  #train test spilt
  data80=np.array(list1[:math.floor(len(list1)*(0.8))])
  data20=np.array(list1[math.floor(len(list1)*(0.8)):])
  print(data80)
  print(data20)

  target80=np.array(list2[:math.floor(len(list2)*(0.8))])
  target20=np.array(list2[math.floor(len(list2)*(0.8)):])
  print(target80)
  print(target20)

def Q4():
  text = "He lifted the bottle to his lips and took a sip of the drink. He had tasted this before, but he couldn't quite remember the time and place it had happened. He desperately searched his mind trying to locate and remember where he had tasted this when the bicycle ran over his foot. There were two things that were important to Tracey. The first was her dog. Anyone that had ever met Tracey knew how much she loved her dog. Most would say that she treated it as her child. The dog went everywhere with her and it had been her best friend for the past five years. The second thing that was important to Tracey, however, would be a lot more surprising to most people. “Ingredients for life,” said the backside of the truck. They mean food, but really food is only 1 ingredient of life. Life has so many more ingredients such as pain, happiness, laughter, joy, tears, and smiles. Life also has hard work, easy play, sleepless nights, and sunbathing by the ocean. Love, hatred, envy, self-assurance, and fear could be just down aisle 3 ready to be bought when needed. How I wish I could pull ingredients like these off shelves in a store."
  text = text.lower()

  f_dict={}
  for letter in text:
    if letter.isalpha():
      if letter not in f_dict:
        f_dict[letter] = 1
      else:
        f_dict[letter] = f_dict[letter] + 1
  print(f_dict)

def Q5():
  text = "He lifted the bottle to his lips and took a sip of the drink. He had tasted this before, but he couldn't quite remember the time and place it had happened. He desperately searched his mind trying to locate and remember where he had tasted this when the bicycle ran over his foot. There were two things that were important to Tracey. The first was her dog. Anyone that had ever met Tracey knew how much she loved her dog. Most would say that she treated it as her child. The dog went everywhere with her and it had been her best friend for the past five years. The second thing that was important to Tracey, however, would be a lot more surprising to most people. “Ingredients for life,” said the backside of the truck. They mean food, but really food is only 1 ingredient of life. Life has so many more ingredients such as pain, happiness, laughter, joy, tears, and smiles. Life also has hard work, easy play, sleepless nights, and sunbathing by the ocean. Love, hatred, envy, self-assurance, and fear could be just down aisle 3 ready to be bought when needed. How I wish I could pull ingredients like these off shelves in a store."
  text = text.lower()
  f_dict={}
  for letter in text:
    if letter.isalpha():
      if letter not in f_dict:
        f_dict[letter] = 1
      else:
        f_dict[letter] = f_dict[letter] + 1

  #sort dictionary by value in decreasing order
  f_dict = sorted(f_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=1)
  print(f_dict)
  top_ten_f_dict = dict(f_dict[0:10])
  plt.bar(list(top_ten_f_dict.keys()), list(top_ten_f_dict.values()), width=0.3, color='r')
  plt.show()

def word_count(wf_dict, word):
  #preprocessing the word for '.' ',' '"'
  if word[-1] == '.' or word[-1] == ',' or word[-1] == '!' or word[-1] == '?':
    word = word[0:len(word)-1]
  if word[0] == "\"":
    word = word[1:len(word)]

  if word not in wf_dict:
      wf_dict[word] = 1
  else:
      wf_dict[word] = wf_dict[word] + 1

def Q6():
  text1 = "The bush began to shake. Brad couldn't see what was causing it to shake, but he didn't care. he had a pretty good idea about what was going on and what was happening. He was so confident that he approached the bush carefree and with a smile on his face. That all changed the instant he realized what was actually behind the bush."
  text2 = "Devon couldn't figure out the color of her eyes. He initially would have guessed that they were green, but the more he looked at them he almost wanted to say they were a golden yellow. Then there were the flashes of red and orange that seemed to be streaked throughout them. It was almost as if her eyes were made of opal with the sun constantly glinting off of them and bringing out more color. They were definitely the most unusual pair of eyes he'd ever seen."
  text3 = "Spending time at national parks can be an exciting adventure, but this wasn't the type of excitement she was hoping to experience. As she contemplated the situation she found herself in, she knew she'd gotten herself in a little more than she bargained for. It wasn't often that she found herself in a tree staring down at a pack of wolves that were looking to make her their next meal."
  text4 = "There weren't supposed to be dragons flying in the sky. First and foremost, dragons didn't exist. They were mythical creatures from fantasy books like unicorns. This was something that Pete knew in his heart to be true so he was having a difficult time acknowledging that there were actually fire-breathing dragons flying in the sky above him."
  text5 = "It really shouldn't have mattered to Betty. That's what she kept trying to convince herself even if she knew it mattered to Betty more than practically anything else. Why was she trying to convince herself otherwise? As she stepped forward to knock on Betty's door, she still didn't have a convincing answer to this question that she'd been asking herself for more than two years now."
  text6 = "Sometimes it's just better not to be seen. That's how Harry had always lived his life. He prided himself as being the fly on the wall and the fae that blended into the crowd. That's why he was so shocked that she noticed him."
  text7 = "She looked at her little girl who was about to become a teen. She tried to think back to when the girl had been younger but failed to pinpoint the exact moment when she had become a little too big to pick up and carry. It hit her all at once. She was no longer a little girl and she stood there speechless with fear, sadness, and pride all running through her at the same time."
  text8 = "Barbara had been waiting at the table for twenty minutes. it had been twenty long and excruciating minutes. David had promised that he would be on time today. He never was, but he had promised this one time. She had made him repeat the promise multiple times over the last week until she'd believed his promise. Now she was paying the price."
  text9 = "The trail to the left had a \"Danger! Do Not Pass\" sign telling people to take the trail to the right. This wasn't the way Zeke approached his hiking. Rather than a warning, Zeke read the sign as an invitation to explore an area that would be adventurous and exciting. As the others in the group all shited to the right, Zeke slipped past the danger sign to begin an adventure he would later regret."
  text10 = "Brock would have never dared to do it on his own he thought to himself. That is why Kenneth and he had become such good friends. Kenneth forced Brock out of his comfort zone and made him try new things he'd never imagine doing otherwise. Up to this point, this had been a good thing. It had expanded Brock's experiences and given him a new appreciation for life. Now that both of them were in the back of a police car, all Brock could think was that he would have never dared do it except for the influence of Kenneth."
  text = text1 + text2 + text3 + text4 + text5 + text6 + text7 + text8 + text9 + text10
  text = text.lower()
  word_list = text.split(' ')
  
  wf_dict={}
  for word in word_list:
    word_count(wf_dict, word)
  
  wf_dict = sorted(wf_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=1)
  top_ten_wf_dict = dict(wf_dict[0:10])
  plt.bar(list(top_ten_wf_dict.keys()), list(top_ten_wf_dict.values()), width=0.3, color='r')
  plt.show()

def Q7():
  x=np.array([0,1,2,3,4,5,6,7,8,9])
  y=x.reshape((5,2))
  z=np.hstack([y,y])

  #swap column index 1 and 2
  z_index1 = np.copy(z[:,1])
  z_index2 = np.copy(z[:,2])
  z[:,2]=z_index1
  z[:,1]=z_index2

  first=z[0,:]
  last=z[-1,:]

  print(first)
  print(last)

  result=first + last
  print(result)

def Q8():
  t = np.arange(0, 10, 0.1)
  x = np.cos(t)
  y = np.sin(t)
  plt.plot(t, x)
  plt.plot(t, y)
  plt.xlabel('Time')
  plt.ylabel('Amplitude')
  plt.show()

def euclidean_distance(x_train, x):
  distance = []
  for i in range(np.shape(x_train)[0]):
    dis = 0
    for j in range(np.shape(x_train)[1]):
      dis +=(x_train[i, j] -x[j])**2
    distance.append(np.sqrt(dis))
  return distance

def vote(distance, k, y_train):
  ind_sort = np.argsort(distance)
  ind=[]
  for i in range(k):
    ind.append(ind_sort[i])
  
  #flag to count
  flag = [0, 0, 0] #label: 0 1 2

  for i in ind:
    flag[int(y_train[i])]+=1

  max_vote=np.max(flag)
  result=[]
  if flag[0]==max_vote:
    result = 0
  elif flag[1]==max_vote:
    result = 1
  else:
    result = 2
  return result

def accuracy(result, y_valid):
  correct_count = 0
  for i in range(len(result)):
    if result[i] == y_valid[i]:
      correct_count+=1
  return correct_count / len(result)

#using validation data to determine the best k
def KNN_score(x_train, x_valid, y_train, y_valid):
  accuracy_store=[] #store the accuracy of different k value

  #change value of k from 1 to 10
  for k in range(1, 11):
    result=[]
    for i in range(np.shape(x_valid)[0]):
      distance = euclidean_distance(x_train, x_valid[i,:])
      result.append(vote(distance, k, y_train))
    accuracy_store.append(accuracy(result, y_valid))
  
  k=np.argsort(accuracy_store)[-1]
  return accuracy_store, k

def predect(x_train, x_test, y_train, k):
  result = []
  for i in range(np.shape(x_test)[0]):
    distance = euclidean_distance(x_train, x_test[i,:])
    result.append(vote(distance, k, y_train))
  return result


def KNN_Classifier():
  #load data
  dataset = load_iris()
  x = dataset.data
  y = dataset.target

  #shuffle data
  data_temp=np.c_[x,y]
  np.random.shuffle(data_temp)
  x=data_temp[:,0:-1]
  y=data_temp[:,-1]

  #train test spilt / using validation data to find the best k
  x_train=np.array(x[0:90,:])
  x_valid=np.array(x[90:120,:])
  x_test=np.array(x[120:,:])

  y_train=np.array(y[0:90])
  y_valid=np.array(y[90:120])
  y_test=np.array(y[120:])

  #normalization
  mean=[]
  std=[]
  for i in range(x_train.shape[1]):
    mean.append(np.mean(x_train[:,i]))
    std.append(np.std(x_train[:,i]))
  #normalize training/validation data
  x_tran_norm=np.zeros(np.shape(x_train))
  for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
      x_tran_norm[i,j]=(x_train[i,j]-mean[j])/std[j]

  x_valid_norm=np.zeros(np.shape(x_valid))
  for i in range(x_valid.shape[0]):
    for j in range(x_valid.shape[1]):
      x_valid_norm[i,j]=(x_valid[i,j]-mean[j])/std[j]

  x_test_norm=np.zeros(np.shape(x_test))
  for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
      x_test_norm[i,j]=(x_test[i,j]-mean[j])/std[j]

  #find out what k is best in this case
  accuracy_store, k = KNN_score(x_tran_norm, x_valid_norm, y_train, y_valid)


  #using test data to evaluate our model
  result = predect(x_tran_norm, x_test_norm, y_train, k)


  #compare with sklearn package
  knn=KNeighborsClassifier(k)
  knn.fit(x_tran_norm, y_train)
  y_pred_sklearn= knn.predict(x_test_norm)


  #plot the K value effect accuracy
  x=np.linspace(1, 10, 10)
  plt.plot(x, accuracy_store)
  plt.xlabel('K value')
  plt.ylabel('accuracy')
  plt.title('accuracy versus different value of K')
  plt.grid(True)
  plt.show()

  #plot the predict result
  X=np.linspace(1, len(result), len(result))
  plt.plot(X, result, "ro", label='My Predict Result')
  plt.plot(X, y_test, "gx", label='Actual Result')
  plt.plot(X, y_pred_sklearn, "b*", label='Sklearn Result')
  plt.legend()
  plt.yticks([0, 1, 2])
  plt.xlabel('xtest')
  plt.ylabel('prediction result')
  plt.title('Final result')
  plt.show()

  #print out the result
  print('my prediction on x_test:', result)
  print('actual solution on y_test:', y_test)
  print('predict result by sklearn', y_pred_sklearn)
  print('predict accuracy of my KNN model on testing dataset:', accuracy(result, y_test))

  
if __name__ == "__main__":
  Q1()
  Q2()
  Q3()
  Q4()
  Q5()
  Q6()
  Q7()
  Q8()
  KNN_Classifier()