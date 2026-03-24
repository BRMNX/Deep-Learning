from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

class simpleDETR(nn.Module):
    def __init__(self, num_classes, num_encoder_layers=6, num_decoder_layers=6, hidden_dim=256, nheads=8):
        super().__init__()
        # Backbone: ResNet50 (N, 3, h, w) -> (N, 2048, H = h/32, W = w/32)
        self.backbone = resnet50()
        # Delete the fully-connected layers in ResNet50
        del self.backbone.fc

        # Conversion Layer 
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # Transformer 
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # Prediction heads(linear layers) 
        # Note that linear_bbox in baseline DETR is a 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes+1) # One extra class for predicting non-empty slots(background)
        self.linear_bbox = nn.Linear(hidden_dim, 4) # A bbox has shape (cx,cy,w,h)
        # Object queries
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # Spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim//2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim//2))

    def forward(self, x):
        # Propagate inputs through ResNet50 up to avg_pool layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # Use 1x1 convolutional kernel to convert 2048 to 256 features
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1),
            self.row_embed[:H].unsqueeze(1).repeat(1,W,1)],
            dim = -1
        ).flatten(0,1).unsqueeze(1) # (H*W, 1, 256)
        # Transformer encoder input size = (seq_len, batch_size, features)
        # h.size = (batch_size, 256, H, W)
        # -> .flatten(2) = (batch_size, 256, H*W)
        # -> .permute(2,0,1) = (H*W, batch_size, 256)
        h = self.transformer(pos + 0.1*h.flatten(2).permute(2,0,1),
                             self.query_pos.unsqueeze(1)
                             ).transpose(0,1) # -> Output size = (batch_size, 100, 256)
        return {'pred_logits': self.linear_class(h), 'pred_boxes': self.linear_bbox(h).sigmoid()}
        # pred_logits size = (batch_size, num_queries = 100, num_classes+1)
        # pred_boxes size = (batch_size, num_queries = 100, 4)
    

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x): # x = [cx, cy, w, h] ranging [0,1]
    x_c, y_c, w, h = x.unbind(1) # Like *x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1) # return b = [[min_x,min_y,max_x,max_y]] ranging [0,1]

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model):
    transform = T.Compose([T.Resize(800),T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Normalize the input image and se batch_size = 1
    img = transform(im).unsqueeze(0)
    # Check whether the img has an appropriate shape
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    outputs = model(img)
    # keep only predictions with 0.7+ confidence
    # Note that outputs['pred_logits'] size = (batch_size=1, num_queries=100, num_classes+1=91)
    # -> .softmax(-1) generates a probability distribution among 91 classes.
    # -> [0] Take the first sample in a batch -> shape:(100,91)
    # -> [:] Save all 100 queries(100 bboxes)
    # -> [:-1] Delete the "background" class, we don't want <0.7 and background.
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def plot_results(pil_img, prob, boxes):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示原图
    axes[0].imshow(pil_img)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    
    # 显示带有边界框的预测图
    axes[1].imshow(pil_img)
    axes[1].axis('off')
    axes[1].set_title('Detection Results')

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        axes[1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        axes[1].text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor='orange', alpha=0.5))
    plt.tight_layout()
    plt.show()

detr = simpleDETR(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# im = Image.open(requests.get(url, stream=True).raw)
im = Image.open(r"C:\Users\27093\Desktop\Deep Learning\else\desk_and_chair.jpg")
scores, boxes = detect(im, detr)
plot_results(im, scores, boxes)