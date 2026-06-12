##Requirements & Dependencies
# # Core Machine Learning & Math Engine
# torch==2.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# numpy>=1.26.0

# # Native Audio Pipeline Dependencies
# scipy>=1.12.0
# soundfile>=0.12.1
# audioread>=3.1.0
# lazy_loader>=0.3
# pooch>=1.8.0
# msgpack>=1.0.7

# # Native Vision Pipeline Dependencies
# pillow>=10.2.0

# # Optional Math Helpers (Installed during setup)
# scikit-learn>=1.4.0
# joblib>=1.3.2
# decorator>=5.1.1



import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw

# Set random seeds for consistent results
torch.manual_seed(42)
random.seed(42)
np.seed = 42

print("--- Step 1: Generating Synthetic Chart Images ---")
num_samples_per_class = 40
X_train_list = []
Y_train_list = []

for _ in range(num_samples_per_class):
    # 1. Generate a Synthetic Bar Chart (Class 0)
    img_bar = Image.new('L', (64, 64), color=255)
    draw_bar = ImageDraw.Draw(img_bar)
    for i in range(3):
        x1 = random.randint(4 + (i * 18), 10 + (i * 18))
        y1 = random.randint(10, 45)
        x2 = x1 + random.randint(6, 10)
        y2 = 58
        draw_bar.rectangle([x1, y1, x2, y2], fill=random.randint(50, 180))
    
    X_train_list.append(np.array(img_bar, dtype=np.float32) / 255.0)
    Y_train_list.append(0)

    # 2. Generate a Synthetic Pie Chart (Class 1)
    img_pie = Image.new('L', (64, 64), color=255)
    draw_pie = ImageDraw.Draw(img_pie)
    cx, cy, r = 32, 32, random.randint(16, 24)
    draw_pie.ellipse([cx-r, cy-r, cx+r, cy+r], outline=100, width=2)
    for _ in range(3):
        angle = random.uniform(0, 2 * np.pi)
        x_end = cx + r * np.cos(angle)
        y_end = cy + r * np.sin(angle)
        draw_pie.line([cx, cy, x_end, y_end], fill=100, width=2)
        
    X_train_list.append(np.array(img_pie, dtype=np.float32) / 255.0)
    Y_train_list.append(1)

X_train = torch.tensor(np.array(X_train_list)).unsqueeze(1) 
Y_train = torch.tensor(np.array(Y_train_list), dtype=torch.long)

print(f"Generated {len(X_train)} synthetic images. Training matrix shape: {X_train.shape}")

# --- Step 2: Defining the Convolutional Neural Network (CNN) ---
print("\n--- Step 2: Defining the Convolutional Neural Network (CNN) ---")
class CNNChartClassifier(nn.Module):
    def __init__(self):
        super(CNNChartClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         
            nn.Conv2d(8, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x

model = CNNChartClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# --- Step 3: Training the AI Model ---
print("\n--- Step 3: Training the AI Model ---")
epochs = 20
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == Y_train).sum().item()
    accuracy = (correct / Y_train.size(0)) * 100
    
    if (epoch + 1) % 4 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {loss.item():.4f} | Accuracy: {accuracy:.1f}%")

# --- Step 4: Loading and Evaluating Your Local Image ---
print("\n--- Step 4: Loading and Evaluating Your Local Image ---")
target_image_path = r"C:\Users\v-nadusumill\OneDrive - Microsoft\Desktop\temp2-delete\visual\chart.jpeg"

if os.path.exists(target_image_path):
    print(f"Found user image at: {target_image_path}")
    
    user_img = Image.open(target_image_path).convert('L').resize((64, 64))
    user_arr = np.array(user_img, dtype=np.float32) / 255.0
    user_tensor = torch.tensor(user_arr).unsqueeze(0).unsqueeze(0) 
    
    model.eval()
    with torch.no_grad():
        prediction_output = model(user_tensor)
        _, final_choice = torch.max(prediction_output, 1)
    
    label_map = {0: "Bar Chart", 1: "Pie Chart"}
    print(f"\n🎯 Prediction Result: The AI classified your chart as a [{label_map[final_choice.item()]}]!")
else:
    print(f"❌ Could not find file at: {target_image_path}")
    print("Please double check that 'chart.jpeg' is sitting perfectly inside your '\\visual' folder.")