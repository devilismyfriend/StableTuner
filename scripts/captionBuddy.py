import tkinter as tk
from tkinter import ttk, Menu
import os
import subprocess
from PIL import Image, ImageTk
import tkinter.filedialog as fd
import json
import sys
import os
import sys
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import subprocess
import numpy as np
import requests
import random
#main class

class TitleBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.title_bar = tk.Frame(self.parent, bg="#1e2124", relief="raised", bd=0)
        #width and height of the title bar
        self.title_bar.config(width=1000, height=30)
        self.title_bar.pack(expand=False, fill="x")
        self.close_button = tk.Button(self.title_bar, text="X", command=self.parent.destroy, bg="#1e2124", fg="#7289da", bd=0, activebackground="#1e2124", activeforeground="#7289da", highlightthickness=0)
        self.close_button.pack(side="right")
        self.minimize_button = tk.Button(self.title_bar, text="-", command=self.minimize_window, bg="#1e2124", fg="#7289da", bd=0, activebackground="#1e2124", activeforeground="#7289da", highlightthickness=0)
        self.minimize_button.pack(side="right")
        #add icon to the title bar
        #self.icon = tk.PhotoImage(self.title_bar,file="resources/stableTuner_icon.png")
        #self.icon = self.icon.subsample(6, 6)
        #self.icon_label = tk.Label(self.title_bar, image=self.icon, bg="#1e2124")
        #self.icon_label.pack(side="left")
        self.title_label = tk.Label(self.title_bar, text="StableTuner", bg="#1e2124", fg="#7289da", font=("Arial", 10, "bold"))
        self.title_label.pack(side="left")
        self.title_bar.bind("<B1-Motion>", self.move_window)
        self.title_bar.bind("<Button-1>", self.click_window)
        self.title_bar.bind("<Double-Button-1>", self.double_click)
        self.parent.bind("<Map>", self.on_resume)
        self.x = self.y = 0
        self.log_size = 0
        self.log_pos = 0
        
    def minimize_window(self):
        self.parent.overrideredirect(False)
        #self.parent.state("iconic")
        self.parent.iconify()
    def move_window(self, event):
        #get the current x and y coordinates of the mouse in relation to screen coordinates
        x = self.parent.winfo_pointerx() - self.x
        y = self.parent.winfo_pointery() - self.y
        #move the window to the new coordinates
        self.parent.geometry("+{}+{}".format(x, y))
    def double_click(self, event):
        if self.parent.state() == "normal":
            #self.parent.overrideredirect(False)
            #resize the window to the screen size
            self.log_size = self.parent.winfo_width(), self.parent.winfo_height()
            self.log_pos = self.parent.winfo_x(), self.parent.winfo_y()
            self.parent.geometry("{0}x{1}+0+0".format(self.parent.winfo_screenwidth(), self.parent.winfo_screenheight()))
            self.parent.state("zoomed")
        else:
            #self.parent.overrideredirect(True)
            #resize the window to the original size
            self.parent.geometry("{0}x{1}+0+0".format(self.log_size[0], self.log_size[1]))\
            #move the window to the original position
            self.parent.geometry("+{0}+{1}".format(self.log_pos[0], self.log_pos[1]))
            self.parent.state("normal")
    def click_window(self, event):
        self.x = event.x
        self.y = event.y
    def on_resume(self, event):
        if self.parent.state() == "normal":
            self.parent.overrideredirect(True)
            self.parent.state("normal")
            self.parent.deiconify()
class ImageBrowser(tk.Frame):
    def __init__(self, master=None,mainProcess=None):
        super().__init__(master)
        if not os.path.exists("scripts/BLIP"):
            print("Getting BLIP from GitHub.")
            subprocess.run(["git", "clone", "https://github.com/salesforce/BLIP", "scripts/BLIP"])
        #if not os.path.exists("scripts/CLIP"):
        #    print("Getting CLIP from GitHub.")
        #    subprocess.run(["git", "clone", "https://github.com/pharmapsychotic/clip-interrogator.git', 'scripts/CLIP"])
        blip_path = "scripts/BLIP"
        sys.path.append(blip_path)
        #clip_path = "scripts/CLIP"
        #sys.path.append(clip_path)
        self.mainProcess = mainProcess
        self.captioner_folder = os.path.dirname(os.path.realpath(__file__))
        self.master = master
        #self.master.overrideredirect(True)
        #self.title_bar = TitleBar(self.master)
        #self.title_bar.pack(side="top", fill="x")
        #make not user resizable
        self.master.title("Caption Buddy")
        #self.master.resizable(False, False)
        self.master.geometry("820x820")
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(side="top")
        self.tip_frame = tk.Frame(self.master)
        self.tip_frame.pack(side="top")
        self.dark_mode_var = "#202020"
        #self.dark_purple_mode_var = "#1B0F1B"
        self.dark_mode_title_var = "#286aff"
        self.dark_mode_button_pressed_var = "#BB91B6"
        self.dark_mode_button_var = "#8ea0e1"
        self.dark_mode_text_var = "#c6c7c8"
        self.master.configure(bg=self.dark_mode_var)
        self.canvas = tk.Canvas(self.master, width=750, height=750, bg=self.dark_mode_var, highlightthickness=0, relief='flat', borderwidth=0)
        self.canvas.configure(bg=self.dark_mode_var)
        #create temporary image for canvas
        self.canvas.pack()
        self.cur_img_index = 0
        self.image_count = 0
        #make a frame with a grid under the canvas
        self.frame = tk.Frame(self.master)
        self.frame.configure(bg=self.dark_mode_var)
        self.top_frame.configure(bg=self.dark_mode_var)
        self.tip_frame.configure(bg=self.dark_mode_var)
        #grid
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=100)
        self.frame.grid_columnconfigure(2, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)
        
        #show the frame
        self.frame.pack(side="bottom", fill="x")
        #bottom frame
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.configure(bg=self.dark_mode_var)
        #make grid
        self.bottom_frame.grid_columnconfigure(0, weight=0)
        self.bottom_frame.grid_columnconfigure(1, weight=2)
        self.bottom_frame.grid_columnconfigure(2, weight=0)
        self.bottom_frame.grid_columnconfigure(3, weight=2)
        self.bottom_frame.grid_columnconfigure(4, weight=0)
        self.bottom_frame.grid_columnconfigure(5, weight=2)
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        #show the frame
        self.bottom_frame.pack(side="bottom", fill="x")

        self.image_index = 0
        self.image_list = []
        self.caption = ''
        self.caption_file = ''
        self.caption_file_path = ''
        self.caption_file_name = ''
        self.caption_file_ext = ''
        self.caption_file_name_no_ext = ''
        self.output_format='text'

        self.use_blip = True
        self.debug = False
        self.create_widgets()
        self.load_blip_model()
        self.load_options()
        self.open_folder()
        
        self.canvas.focus_force()
        self.canvas.bind("<Right>", self.next_image)
        self.canvas.bind("<Left>", self.prev_image)
        
    def create_widgets(self):
        #add a checkbox to toggle auto generate caption
        self.auto_generate_caption = tk.BooleanVar(self.top_frame)
        self.auto_generate_caption.set(True)
        self.auto_generate_caption_checkbox = tk.Checkbutton(self.top_frame, text="Auto Generate Caption", variable=self.auto_generate_caption,fg=self.dark_mode_title_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.auto_generate_caption_checkbox.pack(side="right")
        #add a batch folder button
        self.batch_folder_button = tk.Button(self.top_frame,activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.batch_folder_button["text"] = "Batch Folder"
        self.batch_folder_button["command"] = self.batch_folder
        self.batch_folder_button.pack(side="right")
        self.open_button = tk.Button(self.top_frame,activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.open_button["text"] = "Open Folder"
        self.open_button["command"] = self.open_folder
        #self.open_button.grid(row=0, column=1)
        self.open_button.pack(side="right")
        #add an options button to the same row as the open button
        self.options_button = tk.Button(self.top_frame,activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.options_button["text"] = "Options"
        self.options_button["command"] = self.open_options
        self.options_button.pack(side="right")
        #add generate caption button
        self.generate_caption_button = tk.Button(self.top_frame, text="Generate Caption", command=self.generate_caption,activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.generate_caption_button.pack(side="right")
        
        #add a label for tips under the buttons
        self.tips_label = tk.Label(self.tip_frame, text="Use the left and right arrow keys to navigate images, enter to save the caption.", fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.tips_label.pack(side="top")
        #add image count label
        self.image_count_label = tk.Label(self.tip_frame, text=f"Image {self.cur_img_index} of {self.image_count}", fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.image_count_label.pack(side="top")

        self.image_label = tk.Label(self.canvas, bg=self.dark_mode_var)
        self.image_label.pack(side="top")
        #previous button
        self.prev_button = tk.Button(self.frame,command= lambda event=None: self.prev_image(event),activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        #grid
        self.prev_button.grid(row=1, column=0, sticky="w")

        self.prev_button["text"] = "Previous"
        #self.prev_button["command"] = self.prev_image
        #self.prev_button.pack(side="left")
        #self.prev_button.bind("<Left>", self.prev_image)
        self.caption_entry = tk.Entry(self.frame,fg=self.dark_mode_text_var, bg=self.dark_mode_var, relief='flat', highlightthickness=2, highlightbackground=self.dark_mode_button_var,insertbackground=self.dark_mode_text_var)
        #grid
        self.caption_entry.grid(row=1, column=1, rowspan=3, sticky="nsew")
        #bind to enter key
        self.caption_entry.bind("<Return>", self.save_caption)
        self.canvas.bind("<Return>", self.save_caption)
        #next button

        self.next_button = tk.Button(self.frame, command= lambda event=None: self.next_image(event),activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.next_button["text"] = "Next"
        #grid
        self.next_button.grid(row=1, column=2, sticky="e")
        #add two entry boxes and labels in the style of :replace _ with _
        #create replace string variable
        self.replace_label = tk.Label(self.bottom_frame, text="Replace:", fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.replace_label.grid(row=0, column=0, sticky="w")
        self.replace_entry = tk.Entry(self.bottom_frame, fg=self.dark_mode_text_var, bg=self.dark_mode_var, relief='flat', highlightthickness=2, highlightbackground=self.dark_mode_button_var,insertbackground=self.dark_mode_text_var)
        self.replace_entry.grid(row=0, column=1, sticky="nsew")
        self.replace_entry.bind("<Return>", self.save_caption)
        #self.replace_entry.bind("<Tab>", self.replace)
        #with label
        #create with string variable
        self.with_label = tk.Label(self.bottom_frame, text="With:", fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.with_label.grid(row=0, column=2, sticky="w")
        self.with_entry = tk.Entry(self.bottom_frame, fg=self.dark_mode_text_var, bg=self.dark_mode_var, relief='flat', highlightthickness=2, highlightbackground=self.dark_mode_button_var,insertbackground=self.dark_mode_text_var)
        self.with_entry.grid(row=0, column=3,  sticky="nswe")
        self.with_entry.bind("<Return>", self.save_caption)
        #add another entry with label, add suffix
        
        #create prefix string var
        self.prefix_label = tk.Label(self.bottom_frame, text="Add to start:", fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.prefix_label.grid(row=0, column=4, sticky="w")
        self.prefix_entry = tk.Entry(self.bottom_frame, fg=self.dark_mode_text_var, bg=self.dark_mode_var, relief='flat', highlightthickness=2, highlightbackground=self.dark_mode_button_var,insertbackground=self.dark_mode_text_var)
        self.prefix_entry.grid(row=0, column=5, sticky="nsew")
        self.prefix_entry.bind("<Return>", self.save_caption)

        #create suffix string var
        self.suffix_label = tk.Label(self.bottom_frame, text="Add to end:", fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.suffix_label.grid(row=0, column=6, sticky="w")
        self.suffix_entry = tk.Entry(self.bottom_frame, fg=self.dark_mode_text_var, bg=self.dark_mode_var, relief='flat', highlightthickness=2, highlightbackground=self.dark_mode_button_var,insertbackground=self.dark_mode_text_var)
        self.suffix_entry.grid(row=0, column=7, sticky="nsew")
        self.suffix_entry.bind("<Return>", self.save_caption)
        self.all_entries = [self.replace_entry, self.with_entry, self.suffix_entry, self.caption_entry, self.prefix_entry]
        #bind right click menu to all entries
        for entry in self.all_entries:
            entry.bind("<Button-3>", self.create_right_click_menu)
    def batch_folder(self):
        #show imgs in folder askdirectory
        #ask user if to batch current folder or select folder
        ask = tk.messagebox.askquestion("Batch Folder", "Batch current folder?")
        if ask == 'yes':
            batch_input_dir = self.folder
        else:
            batch_input_dir = fd.askdirectory(title="Select Folder to Batch Process", initialdir=os.getcwd())
        ask2 = tk.messagebox.askquestion("Batch Folder", "Save output to same directory?")
        if ask2 == 'yes':
            batch_output_dir = batch_input_dir
        else:
            batch_output_dir = fd.askdirectory(title="Select Folder to Save Batch Processed Images", initialdir=os.getcwd())
        if batch_input_dir == '':
            return
        if batch_output_dir == '':
            batch_output_dir = batch_input_dir

        self.caption_file_name = os.path.basename(batch_input_dir)
        self.image_list = []
        for file in os.listdir(batch_input_dir):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                self.image_list.append(os.path.join(batch_input_dir, file))
        self.image_index = 0
        #use progress bar class
        pba = tk.Tk()
        pba.title("Batch Processing")
        #remove icon
        pba.wm_attributes('-toolwindow','True')
        pb = ProgressbarWithCancel(pba)
        pb.set_max(len(self.image_list))
        pb.set_progress(0)
        
        #if batch_output_dir doesn't exist, create it
        if not os.path.exists(batch_output_dir):
            os.makedirs(batch_output_dir)
        for i in range(len(self.image_list)):
            radnom_chance = random.randint(0,25)
            if radnom_chance == 0:
                pb.set_random_label()
            if pb.is_cancelled():
                pba.destroy()
                return
            self.image_index = i
            pb.set_progress(i)
            self.master.update()
            img = Image.open(self.image_list[i]).convert("RGB")
            tensor = transforms.Compose([
                        transforms.Resize((self.blipSize, self.blipSize), interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            torch_image = tensor(img).unsqueeze(0).to(torch.device("cuda"))
            if self.nucleus_sampling:
                captions = self.blip_decoder.generate(torch_image, sample=True, top_p=self.q_factor)
            else:
                captions = self.blip_decoder.generate(torch_image, sample=False, num_beams=16, min_length=self.min_length, \
                            max_length=48, repetition_penalty=self.q_factor)
            caption = captions[0]
            self.replace = self.replace_entry.get()
            self.replace_with = self.with_entry.get()
            self.suffix_var = self.suffix_entry.get()
            #prepare the caption
            caption = caption.replace(self.replace, self.replace_with)
            if caption.endswith(',') or caption.endswith('.'):
                caption = caption[:-1]
                caption = caption +', ' + self.suffix_var
            else:
                caption = caption + self.suffix_var
            #saving the captioned image
            if self.output_format == 'text':
                #text file with same name as image
                imgName = os.path.basename(self.image_list[self.image_index])
                imgName = imgName[:imgName.rfind('.')]
                caption_file = os.path.join(batch_output_dir, imgName + '.txt')
                with open(caption_file, 'w') as f:
                    f.write(caption)
            elif self.output_format == 'filename':
                #duplicate image with caption as file name
                img.save(os.path.join(batch_output_dir, caption+'.png'))
        #show message box when done
        pba.destroy()
        donemsg = tk.messagebox.showinfo("Batch Folder", "Batching complete!",parent=self.master)
        #ask user if we should load the batch output folder
        ask3 = tk.messagebox.askquestion("Batch Folder", "Load batch output folder?")
        if ask3 == 'yes':
            self.image_index = 0
            self.open_folder(folder=batch_output_dir)

        #focus on donemsg
        #donemsg.focus_force()
    def generate_caption(self):
        #get the image
        tensor = transforms.Compose([
                        #transforms.CenterCrop(SIZE),
                        transforms.Resize((self.blipSize, self.blipSize), interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        torch_image = tensor(self.PILimage).unsqueeze(0).to(torch.device("cuda"))
        if self.nucleus_sampling:
            captions = self.blip_decoder.generate(torch_image, sample=True, top_p=self.q_factor)
        else:
            captions = self.blip_decoder.generate(torch_image, sample=False, num_beams=16, min_length=self.min_length, \
                        max_length=48, repetition_penalty=self.q_factor)
        self.caption = captions[0]
        self.caption_entry.delete(0, tk.END)
        self.caption_entry.insert(0, self.caption)
        #change the caption entry color to red
        self.caption_entry.configure(fg='red')
    def load_blip_model(self):
        self.blipSize = 384
        blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
        #check if options file exists
        if os.path.exists(os.path.join(self.captioner_folder, 'options.json')):
            with open(os.path.join(self.captioner_folder, 'options.json'), 'r') as f:
                self.nucleus_sampling = json.load(f)['nucleus_sampling']
                self.q_factor = json.load(f)['q_factor']
                self.min_length = json.load(f)['min_length']
        else:
            self.nucleus_sampling = False
            self.q_factor = 1.0
            self.min_length = 22
        config_path = os.path.join(self.captioner_folder, "BLIP/configs/med_config.json")
        cache_folder = os.path.join(self.captioner_folder, "BLIP/cache")
        model_path = os.path.join(self.captioner_folder, "BLIP/models/model_base_caption_capfilt_large.pth")
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
    
        if not os.path.exists(model_path):
            print(f"Downloading BLIP to {cache_folder}")
            with requests.get(blip_model_url, stream=True) as session:
                session.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in session.iter_content(chunk_size=1024): 
                        f.write(chunk)
            print('Download complete')
        else:
            print(f"Found BLIP model")
        import models.blip
        blip_decoder = models.blip.blip_decoder(pretrained=model_path, image_size=self.blipSize, vit='base', med_config=config_path)
        blip_decoder.eval()
        self.blip_decoder = blip_decoder.to(torch.device("cuda"))
        
    def open_folder(self,folder=None):
        if folder is None:
            self.folder = fd.askdirectory()
        else:
            self.folder = folder
        if self.folder == '':
            return
        self.output_folder = self.folder
        self.image_list = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        self.image_list.sort()
        self.image_count = len(self.image_list)
        if self.image_count == 0:
            tk.messagebox.showinfo("No Images", "No images found in the selected folder")
            return
        #update the image count label
        
        self.image_index = 0
        self.image_count_label.configure(text=f'Image {self.image_index+1} of {self.image_count}')
        self.output_folder = self.folder
        self.load_image()
    def load_image(self):
        self.PILimage = Image.open(self.image_list[self.image_index]).convert('RGB')
        #print(self.image_list[self.image_index])
        #self.image = self.image.resize((600, 600), Image.ANTIALIAS)
        #resize to fit 600x600 while maintaining aspect ratio
        width, height = self.PILimage.size
        if width > height:
            new_width = 700
            new_height = int(700 * height / width)
        else:
            new_height = 700
            new_width = int(700 * width / height)
        self.image = self.PILimage.resize((new_width, new_height), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.image, master=self.canvas)
        #print(self.image)
        self.image_label.configure(image=self.image)
        self.caption_file_path = self.image_list[self.image_index]
        self.caption_file_name = os.path.basename(self.caption_file_path)
        self.caption_file_ext = os.path.splitext(self.caption_file_name)[1]
        self.caption_file_name_no_ext = os.path.splitext(self.caption_file_name)[0]
        self.caption_file = os.path.join(self.folder, self.caption_file_name_no_ext + '.txt')
        if os.path.isfile(self.caption_file) and self.auto_generate_caption.get() == False:
            with open(self.caption_file, 'r') as f:
                self.caption = f.read()
                self.caption_entry.delete(0, tk.END)
                self.caption_entry.insert(0, self.caption)
                self.caption_entry.configure(fg=self.dark_mode_text_var)
                self.use_blip = False
        elif os.path.isfile(self.caption_file) and self.auto_generate_caption.get() == True:
            self.use_blip = True
            self.caption_entry.delete(0, tk.END)
        elif self.auto_generate_caption.get() == False:
            self.caption_entry.delete(0, tk.END)
            return
        if self.use_blip and self.debug==False:
            tensor = transforms.Compose([
                        transforms.Resize((self.blipSize, self.blipSize), interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            torch_image = tensor(self.PILimage).unsqueeze(0).to(torch.device("cuda"))
            if self.nucleus_sampling:
                captions = self.blip_decoder.generate(torch_image, sample=True, top_p=self.q_factor)
            else:
                captions = self.blip_decoder.generate(torch_image, sample=False, num_beams=16, min_length=self.min_length, \
                            max_length=48, repetition_penalty=self.q_factor)
            self.caption = captions[0]
            self.caption_entry.delete(0, tk.END)
            self.caption_entry.insert(0, self.caption)
            #change the caption entry color to red
            self.caption_entry.configure(fg='red')

            
    def save_caption(self, event):
        
        

        self.caption = self.caption_entry.get()
        self.replace = self.replace_entry.get()
        self.replace_with = self.with_entry.get()
        self.suffix_var = self.suffix_entry.get()
        self.prefix = self.prefix_entry.get()
        #prepare the caption
        self.caption = self.caption.replace(self.replace, self.replace_with)
        if self.prefix != '':
            if self.prefix.endswith(' '):
                self.prefix = self.prefix[:-1]
            if not self.prefix.endswith(','):
                self.prefix = self.prefix+','
            self.caption = self.prefix + ' ' + self.caption
        if self.caption.endswith(',') or self.caption.endswith('.'):
            self.caption = self.caption[:-1]
            self.caption = self.caption +', ' + self.suffix_var
        else:
            self.caption = self.caption + self.suffix_var
        if self.output_folder != self.folder:
            outputFolder = self.output_folder
        else:
            outputFolder = self.folder
        if self.output_format == 'text':
            #text file with same name as image
            #image name
            #print('test')
            imgName = os.path.basename(self.image_list[self.image_index])
            imgName = imgName[:imgName.rfind('.')]
            self.caption_file = os.path.join(outputFolder, imgName + '.txt')
            with open(self.caption_file, 'w') as f:
                f.write(self.caption)
        elif self.output_format == 'filename':
            #duplicate image with caption as file name
            self.PILimage.save(os.path.join(outputFolder, self.caption+'.png'))
        self.caption_entry.delete(0, tk.END)
        self.caption_entry.insert(0, self.caption)
        self.caption_entry.configure(fg='green')

        self.canvas.focus_force()
    def prev_image(self, event):
        if self.image_index > 0:
            self.image_index -= 1
            self.image_count_label.configure(text=f'Image {self.image_index+1} of {self.image_count}')
            self.load_image()
            self.canvas.focus_force()
    def next_image(self, event):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.image_count_label.configure(text=f'Image {self.image_index+1} of {self.image_count}')
            self.load_image()
            self.canvas.focus_force()
    def open_options(self):
        self.options_window = tk.Toplevel(self.master)
        self.options_window.configure(bg=self.dark_mode_var)
        self.options_window.title("Options")
        self.options_window.geometry("320x320")
        self.options_window.focus_force()
        self.options_window.grab_set()
        self.options_window.transient(self.master)
        self.options_window.protocol("WM_DELETE_WINDOW", self.close_options)
        #add title label
        self.options_title_label = tk.Label(self.options_window, text="Options",font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.options_title_label.pack(side="top")
        #add an entry with a button to select a folder as output folder
        self.output_folder_label = tk.Label(self.options_window, text="Output Folder", bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.output_folder_label.pack(side="top")
        self.output_folder_entry = tk.Entry(self.options_window, bg=self.dark_mode_var, fg=self.dark_mode_text_var, width=40)
        self.output_folder_entry.pack(side="top")
        self.output_folder_entry.insert(0, self.output_folder)
        self.output_folder_button = tk.Button(self.options_window, text="Select Folder", command=self.select_output_folder,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var,relief="flat")
        self.output_folder_button.pack(side="top")
        #add radio buttons to select the output format between text and filename
        self.output_format_label = tk.Label(self.options_window, text="Output Format", bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.output_format_label.pack(side="top")
        self.output_format_var = tk.StringVar(self.options_window)
        self.output_format_var.set(self.output_format)
        self.output_format_text = tk.Radiobutton(self.options_window, text="Text File", variable=self.output_format_var, value="text", bg=self.dark_mode_var, fg=self.dark_mode_text_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.output_format_text.pack(side="top")
        self.output_format_filename = tk.Radiobutton(self.options_window, text="File name", variable=self.output_format_var, value="filename",bg=self.dark_mode_var, fg=self.dark_mode_text_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.output_format_filename.pack(side="top")
        #add BLIP settings section
        self.blip_settings_label = tk.Label(self.options_window, text="BLIP Settings",font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.blip_settings_label.pack(side="top")
        #add a checkbox to use nucleas sampling or not
        self.nucleus_sampling_var = tk.IntVar(self.options_window)
        self.nucleus_sampling_checkbox = tk.Checkbutton(self.options_window, text="Use nucleus sampling", variable=self.nucleus_sampling_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.nucleus_sampling_checkbox.pack(side="top")
        self.nucleus_sampling_var.set(self.nucleus_sampling)
        #add a float entry to set the q factor
        self.q_factor_label = tk.Label(self.options_window, text="Q Factor", bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.q_factor_label.pack(side="top")
        self.q_factor_entry = tk.Entry(self.options_window, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.q_factor_entry.insert(0, self.q_factor)
        self.q_factor_entry.pack(side="top")
        #add a int entry to set the number minimum length
        self.min_length_label = tk.Label(self.options_window, text="Minimum Length", bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.min_length_label.pack(side="top")
        self.min_length_entry = tk.Entry(self.options_window, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.min_length_entry.insert(0, self.min_length)
        self.min_length_entry.pack(side="top")
        #add a horozontal radio button to select between None, ViT-L-14/openai, ViT-H-14/laion2b_s32b_b79k
        #self.model_label = tk.Label(self.options_window, text="CLIP Interrogation", bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        #self.model_label.pack(side="top")
        #self.model_var = tk.StringVar(self.options_window)
        #self.model_var.set(self.model)
        #self.model_none = tk.Radiobutton(self.options_window, text="None", variable=self.model_var, value="None", bg=self.dark_mode_var, fg=self.dark_mode_text_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        #self.model_none.pack(side="top")
        #self.model_vit_l_14 = tk.Radiobutton(self.options_window, text="ViT-L-14/openai", variable=self.model_var, value="ViT-L-14/openai", bg=self.dark_mode_var, fg=self.dark_mode_text_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        #self.model_vit_l_14.pack(side="top")
        #self.model_vit_h_14 = tk.Radiobutton(self.options_window, text="ViT-H-14/laion2b_s32b_b79k", variable=self.model_var, value="ViT-H-14/laion2b_s32b_b79k", bg=self.dark_mode_var, fg=self.dark_mode_text_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        #self.model_vit_h_14.pack(side="top")

        #add a save button
        self.save_button = tk.Button(self.options_window, text="Save", command=self.save_options,fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white", relief="flat")
        self.save_button.pack(side="top")
        #all entries list
        entries = [self.output_folder_entry, self.q_factor_entry, self.min_length_entry]
        #bind the right click to all entries
        for entry in entries:
            entry.bind("<Button-3>", self.create_right_click_menu)
        self.options_file = os.path.join(self.captioner_folder, 'captioner_options.json')
        if os.path.isfile(self.options_file):
            with open(self.options_file, 'r') as f:
                self.options = json.load(f)
                self.output_folder_entry.delete(0, tk.END)
                self.output_folder_entry.insert(0, self.output_folder)
                self.output_format_var.set(self.options['output_format'])
                self.nucleus_sampling_var.set(self.options['nucleus_sampling'])
                self.q_factor_entry.delete(0, tk.END)
                self.q_factor_entry.insert(0, self.options['q_factor'])
                self.min_length_entry.delete(0, tk.END)
                self.min_length_entry.insert(0, self.options['min_length'])
    def load_options(self):
        self.options_file = os.path.join(self.captioner_folder, 'captioner_options.json')
        if os.path.isfile(self.options_file):
            with open(self.options_file, 'r') as f:
                self.options = json.load(f)
                #self.output_folder = self.folder
                #self.output_folder = self.options['output_folder']
                self.output_format = self.options['output_format']
                self.nucleus_sampling = self.options['nucleus_sampling']
                self.q_factor = self.options['q_factor']
                self.min_length = self.options['min_length']
        else:
            self.output_folder = self.folder
            self.output_format = "text"
            self.nucleus_sampling = False
            self.q_factor = 0.9
            self.min_length =22
    def save_options(self):
        self.output_folder = self.output_folder_entry.get()
        self.output_format = self.output_format_var.get()
        self.nucleus_sampling = self.nucleus_sampling_var.get()
        self.q_factor = float(self.q_factor_entry.get())
        self.min_length = int(self.min_length_entry.get())
        #save options to a file
        self.options_file = os.path.join(self.captioner_folder, 'captioner_options.json')
        with open(self.options_file, 'w') as f:
            json.dump({'output_folder': self.output_folder, 'output_format': self.output_format, 'nucleus_sampling': self.nucleus_sampling, 'q_factor': self.q_factor, 'min_length': self.min_length}, f)
        self.close_options()

    def select_output_folder(self):
        self.output_folder = fd.askdirectory()
        self.output_folder_entry.delete(0, tk.END)
        self.output_folder_entry.insert(0, self.output_folder)
    def close_options(self):
        self.options_window.destroy()
        self.canvas.focus_force()
    def create_right_click_menu(self, event):
        #create a menu
        self.menu = Menu(self.master, tearoff=0)
        #add commands to the menu
        self.menu.add_command(label="Cut", command=lambda: self.master.focus_get().event_generate("<<Cut>>"))
        self.menu.add_command(label="Copy", command=lambda: self.master.focus_get().event_generate("<<Copy>>"))
        self.menu.add_command(label="Paste", command=lambda: self.master.focus_get().event_generate("<<Paste>>"))
        self.menu.add_command(label="Select All", command=lambda: self.master.focus_get().event_generate("<<SelectAll>>"))
        #display the menu
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            #make sure to release the grab (Tk 8.0a1 only)
            self.menu.grab_release()


#progress bar class with cancel button
class ProgressbarWithCancel(ttk.Frame):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        
        self.possibleLabels = ['Searching for answers...',"I'm working, I promise.",'ARE THOSE TENTACLES?!','Weird data man...','Another one bites the dust' ,"I think it's a cat?" ,'Looking for the meaning of life', 'Dreaming of captions']
        
        self.label = ttk.Label(self.master, text="Searching for answers...")
        self.label.pack(side="top", fill="x", expand=True)
        self.progress = ttk.Progressbar(self.master, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(self.master, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="right")
        self.cancelled = False
        self.count_label = ttk.Label(self.master, text="0/{0}".format(self.get_max()))
        self.count_label.pack(side="right")
    def set_random_label(self):
        import random
        self.label["text"] = random.choice(self.possibleLabels)
        #pop from list
        #self.possibleLabels.remove(self.label["text"])
    def cancel(self):
        self.cancelled = True
    def set_progress(self, value):
        self.progress["value"] = value
        self.count_label["text"] = "{0}/{1}".format(value, self.get_max())
    def get_progress(self):
        return self.progress["value"]
    def set_max(self, value):
        self.progress["maximum"] = value
    def get_max(self):
        return self.progress["maximum"]
    def is_cancelled(self):
        return self.cancelled
    #quit the progress bar window
        
    
#run when imported as a module
if __name__ == "__main__":

    root = tk.Tk()
    app = ImageBrowser(master=root)
    app.mainloop()