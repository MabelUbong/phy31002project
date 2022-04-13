# import standard PyTorch modules
import torch
import torch.nn as nn
# import torchvision module to handle image manipulation
import torchvision
# import modules to build RunManager helper classes
from collections import OrderedDict
# importing other important modules
import time
from torch.utils.tensorboard import SummaryWriter

class RunManager():

    def __init__(self):
        # tracking every epoch count, loss, accuracy and time
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_accuracy = 0
        self.epoch_start_time = None
        # tracking every run count, run data, hyper-params  and time used
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        # record model,device, loader and TensorBoard 
        self.network = None
        self.device = None
        self.loader = None
        self.tb = None

    def get_accuracy(self, preds, labels):
        #  calculates the sum of RMS of each pixel
        """
        Uses an MSE loss class provided by pytorch to calculate a pixel by
        pixel loss between the output tensor and training image data. Then
        to obtain an 'accuracy' value the Root Mean Squared differnce is
        calculated.

        Args:
        --------
          - preds = The tensor predicted by the network.
          - labels = The training image tensor, ISM.
        
        Returns:
        --------
          - RMSE = The accuracy value for the network run.
        """
        m = nn.MSELoss()
        MSE = m(preds, labels)
        RMSE = torch.sqrt(torch.tensor([MSE], dtype=torch.float32).to(0)).item()
        return RMSE

        # record the count,time,  hyper-param, model, loader of each run
        # record sample images and network graph to TensorBoard

    def begin_run(self, run, network, loader):
        """
        Initiates all of the measured values that are used for data analysis
        such as run count and adds the images to a tensorboard session.

        Args:
        --------
          -run = A set of hyperparameters.
          -network = The Convolutional Neural Network.
          -loader = The Confocal and ISM training images.
        """
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(
            self.network
            , images.to(getattr(run, 'device', 'cpu'))
        )

    def end_run(self):
      """
      Closes the tensorboard session and resets the epoch count.
      Args:
        --------
          - self 
      """
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
      """
      Starts the epoch specific variable; epoch count, epoch loss and
      epoch accuracy.
      Args:
        --------
          - self 
      """
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_accuracy = 0

    def end_epoch(self):
        """
        Ends the epoch specific variable; epoch count, epoch loss and
        epoch accuracy.
        
        Args:
        --------
          - self 
        """
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_accuracy

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

    def track_loss(self, loss, batch):
      """
      Tracks loss by accumlating the loss of batch into entire epoch loss.
      Args:
        --------
          - self
      """"
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_accuracy(self, preds, labels):
      """
      Tracks accuracy by utilising the 'get_accuracy' class method.
      Args:
        --------
          - self
      """"
        self.epoch_accuracy += self.get_accuracy(preds, labels)

    def save(self, model, path):
      """
      Saves the end results of all runs into csv, json for further analysis
      """"
        torch.save(model.state_dict(), path)
        
    # save end results of all runs into csv
    def csv(self, fileName):
      """
    Saves a csv file.
    Args:
        fileName (str): Full or relative path to file.
    """
          
      pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')
