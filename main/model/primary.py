import pandas as pd
import numpy as np 
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from .spotify_api import Spotify
from datetime import datetime
import csv
import json
import uuid

# GLOBAL VARIABLES (TODO: REMOVE!)
df = None
train_data_np = None
music = None
num_attributes = None

#-------------------- MODEL ARCHITECTURE ----------------------
class MusicNet(nn.Module):
  def __init__(self, num_attributes):
    super(MusicNet, self).__init__()
    self.name = "MusicNet"
    self.layer1 = nn.Linear(num_attributes - 1, 6)
    self.layer2 = nn.Linear(6, 3)
    self.layer3 = nn.Linear(3, 1)
    self.num_attributes = num_attributes
    
  def forward(self, img):
    flattened = img.view(-1, self.num_attributes - 1)
    activation1 = self.layer1(flattened)
    activation1 = F.relu(activation1)
    activation2 = self.layer2(activation1)
    activation3 = self.layer3(activation2)
    return activation3

#------------------------- MODEL HELPER FUNCTIONS ---------------------------------#

# Load input tracks data
def data_loader(dataset, batch_size):
  # Getting a list of indices
  total_data = len(dataset)
  indices = []
  for i in range(total_data):
    indices.append(i)

  print("Length of dataset: ", len(dataset))

  np.random.seed(1) # Fixed numpy random seed for reproducible shuffling
  np.random.shuffle(indices)
  train_split = int(len(indices) * 0.6)
  val_split = train_split + int(len(indices) * 0.2)
  test_split = total_data - val_split

  print("Number of training songs = ", str(train_split))
  print("Number of validation songs = ", str(val_split - train_split))
  print("Number of testing songs = ", str(test_split))

  # Finding the relevant indices for the different sets
  relevant_train_indices, relevant_val_indices, relevant_test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

  train_sampler = SubsetRandomSampler(relevant_train_indices)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=train_sampler)

  val_sampler = SubsetRandomSampler(relevant_val_indices)
  val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=val_sampler)

  test_sampler = SubsetRandomSampler(relevant_test_indices)
  test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=test_sampler)

  return train_loader, val_loader, test_loader

# Evaluate model performance using validation/test set
def evaluate(net, loader, criterion, num_attributes):
  total_loss = 0.0
  total_err = 0.0
  total_epoch = 0
  total_predicted = 0
  val_total = 0
  for i, data in enumerate(loader, 0):
    np_data = data.numpy()[:,:]
    inputs = torch.from_numpy(np_data[:,:num_attributes-1]).float()
    labels = torch.from_numpy(np_data[:,num_attributes-1:]).float()
    labels_1d = torch.from_numpy(np_data[:,num_attributes-1]).float()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    pred = (outputs > 0.0).squeeze().long()
    corr = pred != labels
    max, max_indices = torch.max(outputs.data, 1)
    val_total += labels.size(0)
    total_predicted += (pred == labels_1d).sum().item()
    total_err += (pred != labels_1d).sum().item()
    total_loss += loss.item()
    total_epoch += len(labels)
  acc = float(total_predicted) / val_total
  err = float(total_err) / total_epoch
  loss = float(total_loss) / (i + 1)
  return err, loss, acc

# Returns path to model
def get_model_name(name, batch_size, learning_rate, epoch):
  path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                  batch_size,
                                                  learning_rate,
                                                  epoch)
  return path

# Training Curve
def plot_training_curve(path):
  """ Plots the training curve for a model run, given the csv files
  containing the train/validation error/loss.

  Args:
      path: The base path of the csv files produced during training
  """
  import matplotlib.pyplot as plt
  train_acc = np.loadtxt("{}_train_acc.csv".format(path))
  val_acc = np.loadtxt("{}_val_acc.csv".format(path))
  train_err = np.loadtxt("{}_train_err.csv".format(path))
  val_err = np.loadtxt("{}_val_err.csv".format(path))
  train_loss = np.loadtxt("{}_train_loss.csv".format(path))
  val_loss = np.loadtxt("{}_val_loss.csv".format(path))
  n = len(train_err) # number of epochs
  plt.title("Train vs Validation Accuracy")
  plt.plot(range(1,n+1), train_acc, label="Train")
  plt.plot(range(1,n+1), val_acc, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend(loc='best')
  plt.show()
  plt.title("Train vs Validation Error")
  plt.plot(range(1,n+1), train_err, label="Train")
  plt.plot(range(1,n+1), val_err, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.legend(loc='best')
  plt.show()
  plt.title("Train vs Validation Loss")
  plt.plot(range(1,n+1), train_loss, label="Train")
  plt.plot(range(1,n+1), val_loss, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

# Make song recommendations after model training
def recommend_songs(sp, net, np_database, py_database, num_attributes, distance_array, number_of_songs):  
  np_database = np.insert(np_database, num_attributes, -1, axis=1)
  liked, disliked = 0, 0
  results = []

  for i in range (len(np_database)):
    output = net(py_database[i].float())
    pred = (output > 0.0).squeeze().long()
    if (pred.item() == 0):
      disliked += 1
    elif (pred.item() == 1):
      liked += 1
    np_database[i][num_attributes] = pred.item()
  print(f"Liked: {liked} and Disliked: {disliked}\n")
  
  closest_tracks = shortest_distances(distance_array, 1000)
  
  count = 0
  for i in range(0, 100):
    if np_database[closest_tracks[i]][num_attributes] == 1:
      additional_data = sp.get_track_info(['preview_url', 'album.images'], [df.iloc[closest_tracks[i]]['id']]).iloc[0].to_dict()
      results.append({**df.iloc[closest_tracks[i]].to_dict(), **additional_data})
      count += 1
    else:
      print("SKIP")
    if count == number_of_songs:
      break

  return results

#----------------------------- HELPER FUNCTIONS ---------------------------------#

# Calculates and returns the attributes of the "perfect" track from the user's 
# list of liked tracks.
# np_track_array is an array that contains only the required attributes of the 
# tracks.
def calculate_perfect_track(np_track_array):
  perfect_track = np.mean(np_track_array, axis=0) # Calculates mean of column attributes to output perfect track's attributes
  return perfect_track

# Calculates and returns the euclidean distance between the "perfect" track 
# attributes and each of the tracks' attributes in the modified spotify database.
# The modifed spotify database currently contains only the required attributes
# of all tracks in the database, and uses the indexes to locate the tracks in
# the database.
def euclidean_distance(modified_spotify_database, perfect_track):
  distance_array = np.empty(shape=len(modified_spotify_database))
  for i, track in enumerate(modified_spotify_database):
    distance = np.linalg.norm(perfect_track - track) # Calculates euclidean distance between perfect track and current track
    distance_array[i] = distance
  return distance_array

# Find list of disliked tracks using largest euclidean distances
def longest_distances(distance_array, number_of_tracks):
  idx = np.argsort(distance_array)[-number_of_tracks:]
  return idx[::-1]

# Calculates and returns the indexes of the shortest distances from the
# distance_array calculated in the euclidean_distance function. The
# number_of_distances can change depending on how many tracks we want to output.
# As mentioned above, we are using indexes to locate the tracks in the database.
def shortest_distances(distance_array, number_of_distances):
  idx = np.argpartition(distance_array, number_of_distances)
  return idx[:number_of_distances]

# Calculate input tracks
def find_input_tracks(sp, spotify_playlist_url):
  # Find liked tracks
  liked_tracks = (sp.fetch_spotify_attributes_from_ids(sp.get_playlist_track_ids(Spotify.get_playlist_id_from_url(str(spotify_playlist_url))))).to_numpy()

  perfect_track = calculate_perfect_track(liked_tracks)
  distance_array = euclidean_distance(train_data_np, perfect_track)

  # Number of disliked tracks to take as input
  res_num = len(liked_tracks)

  # Find the index array of the largest euclidean distances
  index_array = longest_distances(distance_array, res_num)

  # Number of attributes in each track
  num_attributes = np.shape(liked_tracks)[1]
  

  # Adding all disliked tracks from entire dataset using above index array
  disliked_tracks = np.empty(shape=(res_num, num_attributes))
  
  for i in range(len(index_array)):
    disliked_tracks[i] = train_data_np[index_array[i]]
  

  # Adding labels to tracks
  liked_tracks = np.insert(liked_tracks, num_attributes, 1, axis=1) # label liked tracks with 1
  disliked_tracks = np.insert(disliked_tracks, num_attributes, 0, axis=1) # label disliked tracks with 0

  # Combining liked and disliked tracks to create dataset used for training and validation
  input_tracks = torch.from_numpy(np.concatenate((liked_tracks, disliked_tracks), axis=0)) # combine the liked and disliked tracks to create input tracks (train dataset)
  num_attributes += 1

  return input_tracks, num_attributes, perfect_track, distance_array

#------------------------ MAIN MODEL TRAINING FUNCTION ------------------------------------#

def train_net(net, user, train_loader, val_loader, num_attributes, batch_size=8, learning_rate=0.01, num_epochs=10):
  # Fixed PyTorch random seed for reproducible result
  torch.manual_seed(1) # set the random seed

  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

  # Set up some numpy arrays to store the training/test loss/erruracy
  train_acc = np.zeros(num_epochs)
  train_err = np.zeros(num_epochs)
  train_loss = np.zeros(num_epochs)
  val_acc = np.zeros(num_epochs)
  val_err = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)

  start_time = time.time()
  for epoch in range(num_epochs):  # loop over the dataset multiple times
    total_train_loss = 0.0
    total_train_err = 0.0
    total_epoch = 0
    train_total = 0
    train_predicted = 0
    for i, data in enumerate(train_loader, 0):
      # Get the inputs
      #print(data)
      #print(data.size())
      np_data = data.numpy()[:,:]
      inputs = torch.from_numpy(np_data[:,:num_attributes-1]).float()
      labels = torch.from_numpy(np_data[:,num_attributes-1:]).float()
      labels_1d = torch.from_numpy(np_data[:,num_attributes-1]).float()
      # Zero the parameter gradients
      optimizer.zero_grad()
      #print("inputs: ", inputs)
      #print("labels: ", labels)
      # Forward pass, backward pass, and optimize
      outputs = net(inputs)
      #print(outputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      # Calculate the statistics
      pred = (outputs > 0.0).squeeze().long()
      corr = pred != labels
      max, _ = torch.max(outputs.data, 1)
      #print(outputs.data)
      #print(max)
      #print(pred)
      #print("predictions: ", pred)
      #print("labels: ",labels_1d)
      train_total += labels.size(0)
      train_predicted += (pred == labels_1d).sum().item()
      total_train_err += (pred != labels_1d).sum().item()
      total_train_loss += loss.item()
      total_epoch += len(labels)
    #print(train_predicted)
    #print(train_total)
    train_acc[epoch] = float(train_predicted) / train_total
    train_err[epoch] = float(total_train_err) / total_epoch
    train_loss[epoch] = float(total_train_loss) / (i+1)
    val_err[epoch], val_loss[epoch], val_acc[epoch] = evaluate(net, val_loader, criterion, num_attributes)
    print(("Epoch {}: Train acc: {}, Train err: {}, Train loss: {} | "+
            "Validation acc: {}, Validation err: {}, Validation loss: {}").format(
                epoch + 1,
                train_acc[epoch],
                train_err[epoch],
                train_loss[epoch],
                val_acc[epoch],
                val_err[epoch],
                val_loss[epoch]))
    # Save the current model (checkpoint) to a file
    model_path = f'user_data/{user}/' + get_model_name(net.name, batch_size, learning_rate, epoch)
    print(model_path)
    torch.save(net.state_dict(), model_path)
  print('Finished Training')
  end_time = time.time()
  elapsed_time = end_time - start_time
  print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
  # Write the train/test loss/err into CSV file for plotting later
  epochs = np.arange(1, num_epochs + 1)
  np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
  np.savetxt("{}_train_err.csv".format(model_path), train_err)
  np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
  np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
  np.savetxt("{}_val_err.csv".format(model_path), val_err)
  np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

def main(user, url = '', num_songs = 10):
  # Setup globals (REMOVE!)
  global df, train_data_np, num_attributes
  # Setup Spotify
  sp = Spotify()

  #--------------------------- DATA PROCESSING ------------------------------#

  # Load data
  df = pd.read_csv("./tracks.csv")

  # Columns to train by
  train_data = df.drop(['id', 'name', 'duration_ms', 'explicit', 'artists', 'id_artists', 'release_date', 'time_signature'], axis=1)
  train_data_np = train_data.to_numpy()
  train_data_py = torch.from_numpy(train_data_np)

  # Get input playlist
  input_tracks, num_attributes, perfect_track, distance_array = find_input_tracks(sp, url)

  #----------------------------- TRAIN MODEL -----------------------------------#

  music = MusicNet(num_attributes)
  train_loader, val_loader, test_loader = data_loader(input_tracks, batch_size=8)
  train_net(music, user, train_loader, val_loader, num_attributes, batch_size=8, learning_rate=0.01, num_epochs=5)

  #------------------- EVALUATE MODEL ON TEST SET -----------------------------#

  test_error, test_loss, test_accuracy = evaluate(music, test_loader, nn.BCEWithLogitsLoss(), num_attributes)
  print("Test Accuracy: ", test_accuracy, "\nTest Error: ", test_error, "\nTest Loss: ", test_loss)

  #------------------------------- RECOMMEND SONGS ------------------------------#

  results = recommend_songs(sp, music, train_data_np, train_data_py, num_attributes-1, distance_array, num_songs)
  
  # Save history entry
  fields = ['id', 'track', 'artists', 'images', 'preview'] 
  with open(f'user_data/{user}/history/{uuid.uuid4().hex}_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv', 'w', encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(fields)

      for res in results:
        writer.writerow([res['id'], res['name'], res['artists'], json.dumps(res['album.images']), res['preview_url']])

  return results

if __name__ == "__main__":
    main()