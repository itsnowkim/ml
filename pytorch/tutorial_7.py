import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # 기본 가중치를 불러오지 않으므로 pretrained=True를 지정하지 않습니다.
model.load_state_dict(torch.load('model_weights.pth'))
# be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. 
# Failing to do this will yield inconsistent inference results.
model.eval()

# save with model
torch.save(model, 'model.pth')
model = torch.load('model.pth')