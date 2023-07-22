import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from model.model import VIT
import torch
import torchvision

name = []


def select_image():
    # Open the file dialog to select an image
    image_path = filedialog.askopenfilename(initialdir='/', title='Select an Image',
                                            filetypes=(('Image Files', '*.png *.jpg *.jpeg'), ('All Files', '*.*')))
    name.append(image_path)
    if image_path:
        # Display the selected image
        image = Image.open(image_path)
        image.thumbnail((300, 300))  # Resize image to fit the GUI
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image


Actions = {1: "applauding",
           2: "blowing bubbles",
           3: "brushing teeth",
           4: "cleaning the floor",
           5: "climbing",
           6: "cooking",
           7: "cutting trees",
           8: "cutting vegetables",
           9: "drinking",
           10: "feeding a horse",
           11: "fishing",
           12: "fixing a bike",
           13: "fixing a car",
           14: "gardening",
           15: "holding an umbrella",
           16: "jumping",
           17: "looking through a microscope",
           18: "looking through a telescope",
           19: "playing guitar",
           20: "playing violin",
           21: "pouring liquid",
           22: "pushing a_cart",
           23: "reading",
           24: "phoning",
           25: "riding a bike",
           26: "riding a horse",
           27: "rowing a boat",
           28: "running",
           29: "shooting an arrow",
           30: "smoking",
           31: "taking photos",
           32: "texting message",
           33: "throwing frisby",
           34: "using a computer",
           35: "walking the dog",
           36: "washing dishes",
           37: "watching TV",
           38: "waving hands",
           39: "writing on a board",
           40: "writing on a book"}


def analyze_image():
    img_path = name[len(name) - 1]
    img = Image.open(img_path)
    model = VIT("ViT-B_16", 224).to('cpu')
    state_dict = torch.load('model_best.pth', map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    img = torchvision.transforms.functional.resize(img, (224, 224))
    tensor = torchvision.transforms.functional.to_tensor(img)
    tensor = tensor.unsqueeze(dim=0)
    out, _ = model(tensor)
    _, predicted_label = torch.max(out, 1)
    action = Actions[predicted_label.item() + 1]
    label_gender.config(text=f'Action: {action}')


# Create the GUI
root = tk.Tk()
root.title('Image Action Recognition')

# Configure the window size and background color
root.geometry('600x600')
root.configure(bg='#F4F4F4')

# Add text label above the Select Image button
label_select = tk.Label(root, text='Select an Image:', font=('Arial', 14, 'bold'), bg='#F4F4F4')
label_select.pack(pady=10)

# Create a frame for the image display
frame_image = tk.Frame(root, bg='#F4F4F4')
frame_image.pack(pady=20)

# Add label to display the selected image
image_label = tk.Label(frame_image, bg='white')
image_label.pack()

# Create a frame for the buttons
frame_buttons = tk.Frame(root, bg='#F4F4F4')
frame_buttons.pack(pady=10)

# Add button to select image
button_select = tk.Button(frame_buttons, text='Select Image', command=select_image, width=20)
button_select.grid(row=0, column=0, padx=10)

# Add button to analyze image
button_analyze = tk.Button(frame_buttons, text='Analyze Image', command=analyze_image, width=20)
button_analyze.grid(row=0, column=1, padx=10)

# Create a frame for the labels
frame_labels = tk.Frame(root, bg='#F4F4F4')
frame_labels.pack(pady=10)

# Add labels for gender, and race
label_gender = tk.Label(frame_labels, text='Action: ', font=('Arial', 14), bg='#F4F4F4')
label_gender.grid(row=0, column=0, padx=5)

root.mainloop()
