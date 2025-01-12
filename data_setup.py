import os 
from  torchvision import datasets,transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir,test_dir,transform,batch_size,num_workers):

    train_data = datasets.ImageFolder(train_dir,transform = transform)
    test_data = datasets.ImageFolder(test_dir,transform = transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size,
        True, # shuffle
        num_workers,
        True, # pin memory
    ) 
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
  