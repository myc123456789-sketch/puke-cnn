import cv2
import onnxruntime as ort
from torchvision import transforms

transformdata = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ]
)


def imgread(img_path):
    imgdata = cv2.imread(img_path)
    # 转换成RGB
    imgdata = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    imgdata = transformdata(imgdata)
    # tensor ---> CHW  --->NCHW
    imgdata = imgdata.unsqueeze(0).numpy()
    return imgdata


def inference():
    # 加载onnx模型
    model = ort.InferenceSession(
        "runs/weights/resnet18.onnx", providers=["CPUExecutionProvider"]
    )
    imgdata = imgread("./images/test/ace of clubs/1.jpg")
    out = model.run(None, {"input": imgdata})
    classlabels =  ['ace of clubs', 'ace of diamonds','ace of hearts', 'ace of spades', 'eight of clubs',
                    'eight of diamonds','eight of hearts','eight of spades','five of clubs','five of diamonds',
                    'five of hearts','five of spades','four of clubs',
                    'four of diamonds',
                    'four of hearts',
                    'four of spades',
                    'jack of clubs',
                    'jack of diamonds',
                    'jack of hearts',
                    'jack of spades',
                    'joker',
                    'king of clubs',
                    'king of diamonds',
                    'king of hearts',
                    'king of spades',
                    'nine of clubs', 
                    'nine of diamonds',
                    'nine of hearts',
                    'nine of spades',
                    'queen of clubs',
                    'queen of diamonds',
                    'queen of hearts',
                    'queen of spades',
                    'seven of clubs',
                    'seven of diamonds',
                    'seven of hearts',
                    'seven of spades',
                    'six of clubs',
                    'six of diamonds',
                    'six of hearts',
                    'six of spades',
                    'ten of clubs',
                    'ten of diamonds',
                    'ten of hearts',
                    'ten of spades',
                    'three of clubs',
                    'three of diamonds',
                    'three of hearts',
                    'three of spades',
                    'two of clubs',
                    'two of diamonds',
                    'two of hearts',
                    'two of spades']
    print(classlabels[list(out[0][0]).index(max(out[0][0]))])


if __name__ == "__main__":
    inference()
