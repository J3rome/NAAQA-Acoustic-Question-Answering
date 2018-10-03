import os
import json


def save_training_stats(stats_output_file, epoch_nb, train_accuracy, train_loss, val_accuracy, val_loss):
  """
  Will read the stats file from disk and append new epoch stats (Will create the file if not present)
  """
  if os.path.isfile(stats_output_file):
    with open(stats_output_file, 'r') as f:
      stats = json.load(f)
  else:
    stats = {}

  stats["epoch_%d" % (epoch_nb + 1)] = {
    'train_acc': train_accuracy,
    'train_loss': train_loss,
    'val_accuracy': val_accuracy,
    'val_loss': val_loss
  }

  with open(stats_output_file, 'w') as f:
    json.dump(stats, f, indent=2, sort_keys=True)
