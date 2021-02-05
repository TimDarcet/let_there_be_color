from model import LTBC
from data import places365DataModule


##Reload a checkpoint if needed
from_checkpoint = True
checkpoint = '../weights.ckpt'

if from_checkpoint:
    ltbc = LTBC.load_from_checkpoint(checkpoint)

## Select new images
data_folder = '../places365_standard/'
dm = places365DataModule(data_folder, batch_size=1)
dm.setup(stage='test')

## Retrieve a visualizable RGB Image using a LAB image (from the dataset or network output)
def convert_back_to_rgb(L_image,ab_image):
  L = np.transpose(100.0* L_image.detach().numpy(),(1,2,0))
  ab = np.transpose(254.0*ab_image.detach().numpy() - 127.0,(1,2,0))

  Lab = np.dstack((L,ab)).astype(np.float64)
  img = color.lab2rgb(Lab)

  return img

## Prediction test
test_dataloader = dm.test_dataloader()
batch = next(iter(test_dataloader))

images, labels = batch

L_image = images[:, :1, :, :]
ab_image = images[:, 1:, :, :]
pred_ab, pred_label = ltbc(L_image)


## Comparison (Ground truth vs. Prediction vs. Grayscale Image)

for img_idx in range(images.shape[0]):

  #Compare colored images

  img = images[img_idx].detach()
  img = convert_back_to_rgb(img[:1,:,:],img[1:,:,:])

  pred_Lab_image = torch.cat([L_image,pred_ab], dim=1)
  pred_rgb_image = convert_back_to_rgb(pred_Lab_image.detach()[img_idx,:1,:,:],pred_Lab_image.detach()[img_idx,1:,:,:])

  plt.figure(figsize=(15,10))
  plt.subplot(1,3,1)
  plt.title("Ground truth")
  plt.imshow(img)
  plt.subplot(1,3,2)
  plt.imshow(pred_rgb_image)
  plt.title("Prediction")
  plt.subplot(1,3,3)
  plt.imshow(color.rgb2gray(pred_rgb_image),cmap='gray')
  plt.title("Grayscale")


  #Compare label prediction
  #print(labels[img_idx])
  #print(np.sort(pred_label[img_idx].detach().numpy())[-10:-1])
  #print(np.argsort(pred_label[img_idx].detach().numpy())[-10:-1])
