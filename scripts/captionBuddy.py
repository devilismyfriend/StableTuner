import tkinter as tk
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

class ImageBrowser(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        if not os.path.exists("scripts/BLIP"):
            print("Getting BLIP from GitHub.")
            subprocess.run(["git", "clone", "https://github.com/salesforce/BLIP", "scripts/BLIP"])
        blip_path = "scripts/BLIP"
        sys.path.append(blip_path)
        
        self.captioner_folder = os.path.dirname(os.path.realpath(__file__))
        self.master = master
        #make not user resizable
        self.master.title("Caption Buddy")
        self.master.resizable(False, False)
        self.master.geometry("800x800")
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(side="top")
        self.tip_frame = tk.Frame(self.master)
        self.tip_frame.pack(side="top")
        self.dark_mode_var = "#1e2124"
        self.dark_purple_mode_var = "#1B0F1B"
        self.dark_mode_title_var = "#7289da"
        self.dark_mode_button_pressed_var = "#BB91B6"
        self.dark_mode_button_var = "#8ea0e1"
        self.dark_mode_text_var = "#c6c7c8"
        self.master.configure(bg=self.dark_mode_var)
        self.canvas = tk.Canvas(self.master, width=700, height=700, bg=self.dark_mode_var, highlightthickness=0, relief='flat', borderwidth=0)
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
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_rowconfigure(2, weight=1)
        
        #show the frame
        self.frame.pack(side="bottom", fill="x")
        self.image_index = 0
        self.image_list = []
        self.caption = ''
        self.caption_file = ''
        self.caption_file_path = ''
        self.caption_file_name = ''
        self.caption_file_ext = ''
        self.caption_file_name_no_ext = ''
        self.use_blip = True
        self.create_widgets()
        self.load_blip_model()
        self.open_folder()
        self.load_options()
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
        self.prev_button = tk.Button(self.frame,activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        #grid
        self.prev_button.grid(row=1, column=0, sticky="w")

        self.prev_button["text"] = "Previous"
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
        #if batch_output_dir doesn't exist, create it
        if not os.path.exists(batch_output_dir):
            os.makedirs(batch_output_dir)
        for i in range(len(self.image_list)):
            self.image_index = i
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
        tk.messagebox.showinfo("Batch Folder", "Batching complete!")
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
        
    def open_folder(self):
        self.folder = fd.askdirectory()
        self.image_list = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        self.image_list.sort()
        self.image_count = len(self.image_list)
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
        if self.use_blip:
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
        #add a save button
        self.save_button = tk.Button(self.options_window, text="Save", command=self.save_options,fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white", relief="flat")
        self.save_button.pack(side="top")
        
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
                self.output_folder = self.folder
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

#run when imported as a module
if __name__ == "__main__":

    root = tk.Tk()
    app = ImageBrowser(master=root)
    app.mainloop()