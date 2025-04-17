# demo.py
import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as T
from model import SimpleMedNet
import time

# 加载模型
model = SimpleMedNet(num_classes=2)
model.load_state_dict(torch.load('med_model.pth', map_location='cpu'))
model.eval()

classes = ['Normal', 'Abnormal']

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

def predict(image: Image.Image):
    start = time.time()
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    end = time.time()
    return f"{classes[pred_class]} ({confidence*100:.1f}%)\n耗时: {end - start:.2f} 秒"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传医学图像"),
    outputs=gr.Textbox(label="模型判断结果"),
    title="医学图像分类 Demo",
    description="上传一张医学图像（CT、X-ray、MRI），模型将判断为 Normal 或 Abnormal。"
)

if __name__ == '__main__':
    demo.launch(share=True)
