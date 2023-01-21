import tkinter as tk
from tkinter import ttk, Menu
import os
import subprocess
from PIL import Image, ImageTk, ImageDraw
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
import customtkinter as ctk
from customtkinter import ThemeManager

from clip_segmentation import ClipSeg

#main class
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class BatchMaskWindow(ctk.CTkToplevel):
    def __init__(self, parent, path, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("Batch process masks")
        self.geometry("320x310")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.mode_var = tk.StringVar(self, "Create if absent")
        self.modes = ["Replace all masks", "Create if absent", "Add to existing", "Subtract from existing"]

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        self.path_label.grid(row=0, column=0, sticky="w",padx=5, pady=5)
        self.path_entry = ctk.CTkEntry(self.frame, width=150)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.path_button = ctk.CTkButton(self.frame, width=30, text="...", command=lambda: self.browse_for_path(self.path_entry))
        self.path_button.grid(row=0, column=1, sticky="e", padx=5, pady=5)

        self.prompt_label = ctk.CTkLabel(self.frame, text="Prompt", width=100)
        self.prompt_label.grid(row=1, column=0, sticky="w",padx=5, pady=5)
        self.prompt_entry = ctk.CTkEntry(self.frame, width=200)
        self.prompt_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mode_var, values=self.modes, dynamic_resizing=False, width=200)
        self.mode_dropdown.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.threshold_label = ctk.CTkLabel(self.frame, text="Threshold", width=100)
        self.threshold_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.threshold_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0.0 - 1.0")
        self.threshold_entry.insert(0, "0.3")
        self.threshold_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        self.smooth_label = ctk.CTkLabel(self.frame, text="Smooth", width=100)
        self.smooth_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="5")
        self.smooth_entry.insert(0, 5)
        self.smooth_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        self.expand_label = ctk.CTkLabel(self.frame, text="Expand", width=100)
        self.expand_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.expand_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="10")
        self.expand_entry.insert(0, 10)
        self.expand_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.create_masks_button = ctk.CTkButton(self.frame, text="Create Masks", width=310, command=self.create_masks)
        self.create_masks_button.grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.frame.pack(fill="both", expand=True)

    def browse_for_path(self, entry_box):
        # get the path from the user
        path = fd.askdirectory()
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, tk.END)
        entry_box.insert(0, path)
        self.focus_set()

    def set_progress(self, value, max_value):
        progress = value / max_value
        self.progress.set(progress)
        self.progress_label.configure(text="{0}/{1}".format(value, max_value))
        self.progress.update()

    def create_masks(self):
        self.parent.load_clip_seg_model()

        mode = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract"
        }[self.mode_var.get()]

        self.parent.clip_seg.mask_folder(
            sample_dir=self.path_entry.get(),
            prompts=[self.prompt_entry.get()],
            mode=mode,
            threshold=float(self.threshold_entry.get()),
            smooth_pixels=int(self.smooth_entry.get()),
            expand_pixels=int(self.expand_entry.get()),
            progress_callback=self.set_progress,
        )
        self.parent.load_image()

class ImageBrowser(ctk.CTkToplevel):
    def __init__(self,mainProcess=None):
        super().__init__()
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
        self.clip_seg = None
        self.PILimage = None
        self.PILmask = None
        self.mask_draw_x = 0
        self.mask_draw_y = 0
        self.mask_draw_radius = 20
        #self = master
        #self.overrideredirect(True)
        #self.title_bar = TitleBar(self)
        #self.title_bar.pack(side="top", fill="x")
        #make not user resizable
        self.title("Caption Buddy")
        #self.resizable(False, False)
        self.geometry("720x820")
        self.top_frame = ctk.CTkFrame(self,fg_color='transparent')
        self.top_frame.pack(side="top", fill="x",expand=False)
        self.top_subframe = ctk.CTkFrame(self.top_frame,fg_color='transparent')
        self.top_subframe.pack(side="bottom", fill="x",pady=10)
        self.top_subframe.grid_columnconfigure(0, weight=1)
        self.top_subframe.grid_columnconfigure(1, weight=1)
        self.tip_frame = ctk.CTkFrame(self,fg_color='transparent')
        self.tip_frame.pack(side="top")
        self.dark_mode_var = "#202020"
        #self.dark_purple_mode_var = "#1B0F1B"
        self.dark_mode_title_var = "#286aff"
        self.dark_mode_button_pressed_var = "#BB91B6"
        self.dark_mode_button_var = "#8ea0e1"
        self.dark_mode_text_var = "#c6c7c8"
        #self.configure(bg_color=self.dark_mode_var)
        self.canvas = ctk.CTkLabel(self,text='', width=600, height=600)
        #self.canvas.configure(bg_color=self.dark_mode_var)
        #create temporary image for canvas
        self.canvas.pack()
        self.cur_img_index = 0
        self.image_count = 0
        #make a frame with a grid under the canvas
        self.frame = ctk.CTkFrame(self)
        #grid
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=100)
        self.frame.grid_columnconfigure(2, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)
        
        #show the frame
        self.frame.pack(side="bottom", fill="x")
        #bottom frame
        self.bottom_frame = ctk.CTkFrame(self)
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
        #check if bad_files.txt exists
        if os.path.exists("bad_files.txt"):
            #delete it
            os.remove("bad_files.txt")
        self.use_blip = True
        self.debug = False
        self.create_widgets()
        self.load_blip_model()
        self.load_options()
        #self.open_folder()
        
        self.canvas.focus_force()
        self.canvas.bind("<Alt-Right>", self.next_image)
        self.canvas.bind("<Alt-Left>", self.prev_image)
        #on close window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    def on_closing(self):
        #self.save_options()
        self.mainProcess.deiconify()
        self.destroy()
    def create_widgets(self):
        self.output_folder = ''

        # add a checkbox to toggle auto generate caption
        self.auto_generate_caption = tk.BooleanVar(self.top_subframe)
        self.auto_generate_caption.set(True)
        self.auto_generate_caption_checkbox = ctk.CTkCheckBox(self.top_subframe, text="Auto Generate Caption", variable=self.auto_generate_caption,width=50)
        self.auto_generate_caption_checkbox.pack(side="left", fill="x", expand=True, padx=10)

        # add a checkbox to skip auto generating captions if they already exist
        self.auto_generate_caption_text_override = tk.BooleanVar(self.top_subframe)
        self.auto_generate_caption_text_override.set(False)
        self.auto_generate_caption_checkbox_text_override = ctk.CTkCheckBox(self.top_subframe, text="Skip Auto Generate If Text Caption Exists", variable=self.auto_generate_caption_text_override,width=50)
        self.auto_generate_caption_checkbox_text_override.pack(side="left", fill="x", expand=True, padx=10)

        # add a checkbox to enable mask editing
        self.enable_mask_editing = tk.BooleanVar(self.top_subframe)
        self.enable_mask_editing.set(False)
        self.enable_mask_editing_checkbox = ctk.CTkCheckBox(self.top_subframe, text="Enable Mask Editing", variable=self.enable_mask_editing, width=50)
        self.enable_mask_editing_checkbox.pack(side="left", fill="x", expand=True, padx=10)

        self.open_button = ctk.CTkButton(self.top_frame,text="Load Folder",fg_color=("gray75", "gray25"), command=self.open_folder,width=50)
        #self.open_button.grid(row=0, column=1)
        self.open_button.pack(side="left", fill="x",expand=True,padx=10)
        #add a batch folder button
        self.batch_folder_caption_button = ctk.CTkButton(self.top_frame, text="Batch Folder Caption", fg_color=("gray75", "gray25"), command=self.batch_folder_caption, width=50)
        self.batch_folder_caption_button.pack(side="left", fill="x", expand=True, padx=10)
        self.batch_folder_mask_button = ctk.CTkButton(self.top_frame, text="Batch Folder Mask", fg_color=("gray75", "gray25"), command=self.batch_folder_mask, width=50)
        self.batch_folder_mask_button.pack(side="left", fill="x", expand=True, padx=10)

        #add an options button to the same row as the open button
        self.options_button = ctk.CTkButton(self.top_frame, text="Options",fg_color=("gray75", "gray25"), command=self.open_options,width=50)
        self.options_button.pack(side="left", fill="x",expand=True,padx=10)
        #add generate caption button
        self.generate_caption_button = ctk.CTkButton(self.top_frame, text="Generate Caption",fg_color=("gray75", "gray25"), command=self.generate_caption,width=50)
        self.generate_caption_button.pack(side="left", fill="x",expand=True,padx=10)
        
        #add a label for tips under the buttons
        self.tips_label = ctk.CTkLabel(self.tip_frame, text="Use Alt with left and right arrow keys to navigate images, enter to save the caption.")
        self.tips_label.pack(side="top")
        #add image count label
        self.image_count_label = ctk.CTkLabel(self.tip_frame, text=f"Image {self.cur_img_index} of {self.image_count}")
        self.image_count_label.pack(side="top")

        self.image_label = ctk.CTkLabel(self.canvas,text='',width=100,height=100)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        #self.image_label.bind("<Button-3>", self.click_canvas)
        self.image_label.bind("<Motion>", self.draw_mask)
        self.image_label.bind("<Button-1>", self.draw_mask)
        self.image_label.bind("<Button-3>", self.draw_mask)
        self.image_label.bind("<MouseWheel>", self.draw_mask_radius)
        #self.image_label.pack(side="top")
        #previous button
        self.prev_button = ctk.CTkButton(self.frame,text="Previous", command= lambda event=None: self.prev_image(event),width=50)
        #grid
        self.prev_button.grid(row=1, column=0, sticky="w",padx=5,pady=10)
        #self.prev_button.pack(side="left")
        #self.prev_button.bind("<Left>", self.prev_image)
        self.caption_entry = ctk.CTkEntry(self.frame)
        #grid
        self.caption_entry.grid(row=1, column=1, rowspan=3, sticky="nsew",pady=10)
        #bind to enter key
        self.caption_entry.bind("<Return>", self.save)
        self.canvas.bind("<Return>", self.save)
        self.caption_entry.bind("<Alt-Right>", self.next_image)
        self.caption_entry.bind("<Alt-Left>", self.prev_image)
        self.caption_entry.bind("<Control-BackSpace>", self.delete_word)
        #next button

        self.next_button = ctk.CTkButton(self.frame,text='Next', command= lambda event=None: self.next_image(event),width=50)
        #self.next_button["text"] = "Next"
        #grid
        self.next_button.grid(row=1, column=2, sticky="e",padx=5,pady=10)
        #add two entry boxes and labels in the style of :replace _ with _
        #create replace string variable
        self.replace_label = ctk.CTkLabel(self.bottom_frame, text="Replace:")
        self.replace_label.grid(row=0, column=0, sticky="w",padx=5)
        self.replace_entry = ctk.CTkEntry(self.bottom_frame,   )
        self.replace_entry.grid(row=0, column=1, sticky="nsew",padx=5)
        self.replace_entry.bind("<Return>", self.save)
        #self.replace_entry.bind("<Tab>", self.replace)
        #with label
        #create with string variable
        self.with_label = ctk.CTkLabel(self.bottom_frame, text="With:")
        self.with_label.grid(row=0, column=2, sticky="w",padx=5)
        self.with_entry = ctk.CTkEntry(self.bottom_frame,   )
        self.with_entry.grid(row=0, column=3,  sticky="nswe",padx=5)
        self.with_entry.bind("<Return>", self.save)
        #add another entry with label, add suffix
        
        #create prefix string var
        self.prefix_label = ctk.CTkLabel(self.bottom_frame, text="Add to start:")
        self.prefix_label.grid(row=0, column=4, sticky="w",padx=5)
        self.prefix_entry = ctk.CTkEntry(self.bottom_frame,   )
        self.prefix_entry.grid(row=0, column=5, sticky="nsew",padx=5)
        self.prefix_entry.bind("<Return>", self.save)

        #create suffix string var
        self.suffix_label = ctk.CTkLabel(self.bottom_frame, text="Add to end:")
        self.suffix_label.grid(row=0, column=6, sticky="w",padx=5)
        self.suffix_entry = ctk.CTkEntry(self.bottom_frame,   )
        self.suffix_entry.grid(row=0, column=7, sticky="nsew",padx=5)
        self.suffix_entry.bind("<Return>", self.save)
        self.all_entries = [self.replace_entry, self.with_entry, self.suffix_entry, self.caption_entry, self.prefix_entry]
        #bind right click menu to all entries
        for entry in self.all_entries:
            entry.bind("<Button-3>", self.create_right_click_menu)
    def batch_folder_caption(self):
        #show imgs in folder askdirectory
        #ask user if to batch current folder or select folder
        #if bad_files.txt exists, delete it
        self.bad_files = []
        if os.path.exists('bad_files.txt'):
            os.remove('bad_files.txt')
        try:
            #check if self.folder is set
            self.folder
        except AttributeError:
            self.folder = ''
        if self.folder == '':
            self.folder = fd.askdirectory(title="Select Folder to Batch Process", initialdir=os.getcwd())
            batch_input_dir = self.folder
        else:
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
            if (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")) and not file.endswith('-masklabel.png'):
                self.image_list.append(os.path.join(batch_input_dir, file))
        self.image_index = 0
        #use progress bar class
        #pba = tk.Tk()
        #pba.title("Batch Processing")
        #remove icon
        #pba.wm_attributes('-toolwindow','True')
        pb = ProgressbarWithCancel(max=len(self.image_list))
        #pb.set_max(len(self.image_list))
        pb.set_progress(0)
        
        #if batch_output_dir doesn't exist, create it
        if not os.path.exists(batch_output_dir):
            os.makedirs(batch_output_dir)
        for i in range(len(self.image_list)):
            radnom_chance = random.randint(0,25)
            if radnom_chance == 0:
                pb.set_random_label()
            if pb.is_cancelled():
                pb.destroy()
                return
            self.image_index = i
            #get float value of progress between 0 and 1 according to the image index and the total number of images
            progress = i / len(self.image_list)
            pb.set_progress(progress)
            self.update()
            try:
                img = Image.open(self.image_list[i]).convert("RGB")
            except:
                self.bad_files.append(self.image_list[i])
                #skip file
                continue
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
            self.prefix = self.prefix_entry.get()
            #prepare the caption
            if self.suffix_var.startswith(',') or self.suffix_var.startswith(' '):
                self.suffix_var = self.suffix_var
            else:
                self.suffix_var = ' ' + self.suffix_var
            caption = caption.replace(self.replace, self.replace_with)
            if self.prefix != '':
                if self.prefix.endswith(' '):
                    self.prefix = self.prefix[:-1]
                if not self.prefix.endswith(','):
                    self.prefix = self.prefix+','
                caption = self.prefix + ' ' + caption
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
            progress = i + 1 / len(self.image_list)
            pb.set_progress(progress)
        #show message box when done
        pb.destroy()
        donemsg = tk.messagebox.showinfo("Batch Folder", "Batching complete!",parent=self.master)
        if len(self.bad_files) > 0:
            bad_files_msg = tk.messagebox.showinfo("Bad Files", "Couldn't process " + str(len(self.bad_files)) + "files,\nFor a list of problematic files see bad_files.txt",parent=self.master)
            with open('bad_files.txt', 'w') as f:
                for item in self.bad_files:
                    f.write(item + '\n')

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
        self.caption_entry.configure(fg_color='red')
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

    def batch_folder_mask(self):
        folder = ''
        try:
            # check if self.folder is set
            folder = self.folder
        except:
            pass

        dialog = BatchMaskWindow(self, folder)
        dialog.mainloop()

    def load_clip_seg_model(self):
        if self.clip_seg is None:
            self.clip_seg = ClipSeg()

    def open_folder(self,folder=None):
        if folder is None:
            self.folder = fd.askdirectory()
        else:
            self.folder = folder
        if self.folder == '':
            return
        self.output_folder = self.folder
        self.image_list = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')) and not f.endswith('-masklabel.png') and not f.endswith('-depth.png')]
        #self.image_list.sort()
        #sort the image list alphabetically so that the images are in the same order every time
        self.image_list.sort(key=lambda x: x.lower())

        self.image_count = len(self.image_list)
        if self.image_count == 0:
            tk.messagebox.showinfo("No Images", "No images found in the selected folder")
            return
        #update the image count label
        
        self.image_index = 0
        self.image_count_label.configure(text=f'Image {self.image_index+1} of {self.image_count}')
        self.output_folder = self.folder
        self.load_image()
        self.caption_entry.focus_set()

    def draw_mask(self, event):
        if not self.enable_mask_editing.get():
            return

        if event.widget != self.image_label.children["!label"]:
            return

        start_x = int(event.x / self.image_size[0] * self.PILimage.width)
        start_y = int(event.y / self.image_size[1] * self.PILimage.height)
        end_x = int(self.mask_draw_x / self.image_size[0] * self.PILimage.width)
        end_y = int(self.mask_draw_y / self.image_size[1] * self.PILimage.height)

        self.mask_draw_x = event.x
        self.mask_draw_y = event.y

        color = None

        if event.state & 0x0100 or event.num == 1:  # left mouse button
            color = (255, 255, 255)
        elif event.state & 0x0400 or event.num == 3:  # right mouse button
            color = (0, 0, 0)

        if color is not None:
            if self.PILmask is None:
                self.PILmask = Image.new('RGB', size=self.PILimage.size, color=(0, 0, 0))

            draw = ImageDraw.Draw(self.PILmask)
            draw.line((start_x, start_y, end_x, end_y), fill=color, width=self.mask_draw_radius + self.mask_draw_radius + 1)
            draw.ellipse((start_x - self.mask_draw_radius, start_y - self.mask_draw_radius, start_x + self.mask_draw_radius, start_y + self.mask_draw_radius), fill=color, outline=None)
            draw.ellipse((end_x - self.mask_draw_radius, end_y - self.mask_draw_radius, end_x + self.mask_draw_radius, end_y + self.mask_draw_radius), fill=color, outline=None)

            self.compose_masked_image()
            self.display_image()

    def draw_mask_radius(self, event):
        if event.widget != self.image_label.children["!label"]:
            return

        delta = -np.sign(event.delta) * 5
        self.mask_draw_radius += delta

    def compose_masked_image(self):
        np_image = np.array(self.PILimage).astype(np.float32) / 255.0
        np_mask = np.array(self.PILmask).astype(np.float32) / 255.0
        np_mask = np.clip(np_mask, 0.4, 1.0)
        np_masked_image = (np_image * np_mask * 255.0).astype(np.uint8)
        self.image = Image.fromarray(np_masked_image, mode='RGB')

    def display_image(self):
        #resize to fit 600x600 while maintaining aspect ratio
        width, height = self.image.size
        if width > height:
            new_width = 600
            new_height = int(600 * height / width)
        else:
            new_height = 600
            new_width = int(600 * width / height)
        self.image_size = (new_width, new_height)
        self.image = self.image.resize(self.image_size, Image.Resampling.LANCZOS)
        self.image = ctk.CTkImage(self.image, size=self.image_size)
        self.image_label.configure(image=self.image)

    def load_image(self):
        try:
            self.PILimage = Image.open(self.image_list[self.image_index]).convert('RGB')
        except:
            print(f'Error opening image {self.image_list[self.image_index]}')
            print('Logged path to bad_files.txt')
            #if bad_files.txt doesn't exist, create it
            if not os.path.exists('bad_files.txt'):
                with open('bad_files.txt', 'w') as f:
                    f.write(self.image_list[self.image_index]+'\n')
            else:
                with open('bad_files.txt', 'a') as f:
                    f.write(self.image_list[self.image_index]+'\n')
            return

        self.image = self.PILimage.copy()

        try:
            self.PILmask = None
            mask_filename = os.path.splitext(self.image_list[self.image_index])[0] + '-masklabel.png'
            if os.path.exists(mask_filename):
                self.PILmask = Image.open(mask_filename).convert('RGB')
                self.compose_masked_image()
        except Exception as e:
            print(f'Error opening mask for {self.image_list[self.image_index]}')
            print('Logged path to bad_files.txt')
            #if bad_files.txt doesn't exist, create it
            if not os.path.exists('bad_files.txt'):
                with open('bad_files.txt', 'w') as f:
                    f.write(self.image_list[self.image_index]+'\n')
            else:
                with open('bad_files.txt', 'a') as f:
                    f.write(self.image_list[self.image_index]+'\n')
            return

        self.display_image()

        self.caption_file_path = self.image_list[self.image_index]
        self.caption_file_name = os.path.basename(self.caption_file_path)
        self.caption_file_ext = os.path.splitext(self.caption_file_name)[1]
        self.caption_file_name_no_ext = os.path.splitext(self.caption_file_name)[0]
        self.caption_file = os.path.join(self.folder, self.caption_file_name_no_ext + '.txt')
        if os.path.isfile(self.caption_file) and self.auto_generate_caption.get() == False or os.path.isfile(self.caption_file) and self.auto_generate_caption.get() == True and self.auto_generate_caption_text_override.get() == True:
            with open(self.caption_file, 'r') as f:
                self.caption = f.read()
                self.caption_entry.delete(0, tk.END)
                self.caption_entry.insert(0, self.caption)
                self.caption_entry.configure(fg_color=ThemeManager.theme["CTkEntry"]["fg_color"])
                self.use_blip = False
        elif os.path.isfile(self.caption_file) and self.auto_generate_caption.get() == True and self.auto_generate_caption_text_override.get() == False or os.path.isfile(self.caption_file)==False and self.auto_generate_caption.get() == True and self.auto_generate_caption_text_override.get() == True:
            self.use_blip = True
            self.caption_entry.delete(0, tk.END)
        elif os.path.isfile(self.caption_file) == False and self.auto_generate_caption.get() == False:
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
            self.caption_entry.configure(fg_color='red')

    def save(self, event):
        self.save_caption()

        if self.enable_mask_editing.get():
            self.save_mask()

    def save_mask(self):
        mask_filename = os.path.splitext(self.image_list[self.image_index])[0] + '-masklabel.png'
        if self.PILmask is not None:
            self.PILmask.save(mask_filename)

    def save_caption(self):
        self.caption = self.caption_entry.get()
        self.replace = self.replace_entry.get()
        self.replace_with = self.with_entry.get()
        self.suffix_var = self.suffix_entry.get()
        self.prefix = self.prefix_entry.get()
        #prepare the caption
        self.caption = self.caption.replace(self.replace, self.replace_with)
        if self.suffix_var.startswith(',') or self.suffix_var.startswith(' '):
            self.suffix_var = self.suffix_var
        else:
            self.suffix_var = ' ' + self.suffix_var
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
        self.caption = self.caption.strip()
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
            #make sure self.caption doesn't contain any illegal characters
            illegal_chars = ['/', '\\', ':', '*', '?', '"', "'",'<', '>', '|', '.']
            for char in illegal_chars:
                self.caption = self.caption.replace(char, '')
            self.PILimage.save(os.path.join(outputFolder, self.caption+'.png'))
        self.caption_entry.delete(0, tk.END)
        self.caption_entry.insert(0, self.caption)
        self.caption_entry.configure(fg_color='green')

        self.caption_entry.focus_force()
    def delete_word(self,event):
        ent = event.widget
        end_idx = ent.index(tk.INSERT)
        start_idx = ent.get().rfind(" ", None, end_idx)
        ent.selection_range(start_idx, end_idx)
    def prev_image(self, event):
        if self.image_index > 0:
            self.image_index -= 1
            self.image_count_label.configure(text=f'Image {self.image_index+1} of {self.image_count}')
            self.load_image()
            self.caption_entry.focus_set()
            self.caption_entry.focus_force()
    def next_image(self, event):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.image_count_label.configure(text=f'Image {self.image_index+1} of {self.image_count}')
            self.load_image()
            self.caption_entry.focus_set()
            self.caption_entry.focus_force()
    def open_options(self):
        self.options_window = ctk.CTkToplevel(self)
        self.options_window.title("Options")
        self.options_window.geometry("320x550")
        #disable reszie
        self.options_window.resizable(False, False)
        self.options_window.focus_force()
        self.options_window.grab_set()
        self.options_window.transient(self)
        self.options_window.protocol("WM_DELETE_WINDOW", self.close_options)
        #add title label
        self.options_title_label = ctk.CTkLabel(self.options_window, text="Options",font=ctk.CTkFont(size=20, weight="bold"))
        self.options_title_label.pack(side="top", pady=5)
        #add an entry with a button to select a folder as output folder
        self.output_folder_label = ctk.CTkLabel(self.options_window, text="Output Folder")
        self.output_folder_label.pack(side="top", pady=5)
        self.output_folder_entry = ctk.CTkEntry(self.options_window)
        self.output_folder_entry.pack(side="top", fill="x", expand=False,padx=15, pady=5)
        self.output_folder_entry.insert(0, self.output_folder)
        self.output_folder_button = ctk.CTkButton(self.options_window, text="Select Folder", command=self.select_output_folder,fg_color=("gray75", "gray25"))
        self.output_folder_button.pack(side="top", pady=5)
        #add radio buttons to select the output format between text and filename
        self.output_format_label = ctk.CTkLabel(self.options_window, text="Output Format")
        self.output_format_label.pack(side="top", pady=5)
        self.output_format_var = tk.StringVar(self.options_window)
        self.output_format_var.set(self.output_format)
        self.output_format_text = ctk.CTkRadioButton(self.options_window, text="Text File", variable=self.output_format_var, value="text")
        self.output_format_text.pack(side="top", pady=5)
        self.output_format_filename = ctk.CTkRadioButton(self.options_window, text="File name", variable=self.output_format_var, value="filename")
        self.output_format_filename.pack(side="top", pady=5)
        #add BLIP settings section
        self.blip_settings_label = ctk.CTkLabel(self.options_window, text="BLIP Settings",font=ctk.CTkFont(size=20, weight="bold"))
        self.blip_settings_label.pack(side="top", pady=10)
        #add a checkbox to use nucleas sampling or not
        self.nucleus_sampling_var = tk.IntVar(self.options_window)
        self.nucleus_sampling_checkbox = ctk.CTkCheckBox(self.options_window, text="Use nucleus sampling", variable=self.nucleus_sampling_var)
        self.nucleus_sampling_checkbox.pack(side="top", pady=5)
        if self.debug:
            self.nucleus_sampling = 0
            self.q_factor = 0.5
            self.min_length = 10
        self.nucleus_sampling_var.set(self.nucleus_sampling)
        #add a float entry to set the q factor
        self.q_factor_label = ctk.CTkLabel(self.options_window, text="Q Factor")
        self.q_factor_label.pack(side="top", pady=5)
        self.q_factor_entry = ctk.CTkEntry(self.options_window)
        self.q_factor_entry.insert(0, self.q_factor)
        self.q_factor_entry.pack(side="top", pady=5)
        #add a int entry to set the number minimum length
        self.min_length_label = ctk.CTkLabel(self.options_window, text="Minimum Length")
        self.min_length_label.pack(side="top", pady=5)
        self.min_length_entry = ctk.CTkEntry(self.options_window)
        self.min_length_entry.insert(0, self.min_length)
        self.min_length_entry.pack(side="top", pady=5)
        #add a horozontal radio button to select between None, ViT-L-14/openai, ViT-H-14/laion2b_s32b_b79k
        #self.model_label = ctk.CTkLabel(self.options_window, text="CLIP Interrogation")
        #self.model_label.pack(side="top")
        #self.model_var = tk.StringVar(self.options_window)
        #self.model_var.set(self.model)
        #self.model_none = tk.Radiobutton(self.options_window, text="None", variable=self.model_var, value="None")
        #self.model_none.pack(side="top")
        #self.model_vit_l_14 = tk.Radiobutton(self.options_window, text="ViT-L-14/openai", variable=self.model_var, value="ViT-L-14/openai")
        #self.model_vit_l_14.pack(side="top")
        #self.model_vit_h_14 = tk.Radiobutton(self.options_window, text="ViT-H-14/laion2b_s32b_b79k", variable=self.model_var, value="ViT-H-14/laion2b_s32b_b79k")
        #self.model_vit_h_14.pack(side="top")

        #add a save button
        self.save_button = ctk.CTkButton(self.options_window, text="Save", command=self.save_options, fg_color=("gray75", "gray25"))
        self.save_button.pack(side="top",fill='x',pady=10,padx=10)
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
                if 'folder' in self.__dict__:
                    self.output_folder = self.folder
                else:
                    self.output_folder = ''
                self.output_format = self.options['output_format']
                self.nucleus_sampling = self.options['nucleus_sampling']
                self.q_factor = self.options['q_factor']
                self.min_length = self.options['min_length']
        else:
            #if self has folder, use it, otherwise use the current folder
            if 'folder' in self.__dict__ :
                self.output_folder = self.folder
            else:
                self.output_folder = ''
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
        self.caption_entry.focus_force()
    def create_right_click_menu(self, event):
        #create a menu
        self.menu = Menu(self, tearoff=0)
        #add commands to the menu
        self.menu.add_command(label="Cut", command=lambda: self.focus_get().event_generate("<<Cut>>"))
        self.menu.add_command(label="Copy", command=lambda: self.focus_get().event_generate("<<Copy>>"))
        self.menu.add_command(label="Paste", command=lambda: self.focus_get().event_generate("<<Paste>>"))
        self.menu.add_command(label="Select All", command=lambda: self.focus_get().event_generate("<<SelectAll>>"))
        #display the menu
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            #make sure to release the grab (Tk 8.0a1 only)
            self.menu.grab_release()


#progress bar class with cancel button
class ProgressbarWithCancel(ctk.CTkToplevel):
    def __init__(self,max=None, **kw):
        super().__init__(**kw)
        self.title("Batching...")
        self.max = max
        self.possibleLabels = ['Searching for answers...',"I'm working, I promise.",'ARE THOSE TENTACLES?!','Weird data man...','Another one bites the dust' ,"I think it's a cat?" ,'Looking for the meaning of life', 'Dreaming of captions']
        
        self.label = ctk.CTkLabel(self, text="Searching for answers...")
        self.label.pack(side="top", fill="x", expand=True,padx=10,pady=10)
        self.progress = ctk.CTkProgressBar(self, orientation="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True,padx=10,pady=10)
        self.cancel_button = ctk.CTkButton(self, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="right",padx=10,pady=10)
        self.cancelled = False
        self.count_label = ctk.CTkLabel(self, text="0/{0}".format(self.max))
        self.count_label.pack(side="right",padx=10,pady=10)
    def set_random_label(self):
        import random
        self.label["text"] = random.choice(self.possibleLabels)
        #pop from list
        #self.possibleLabels.remove(self.label["text"])
    def cancel(self):
        self.cancelled = True
    def set_progress(self, value):
        self.progress.set(value)
        self.count_label.configure(text="{0}/{1}".format(int(value * self.max), self.max))
    def get_progress(self):
        return self.progress.get
    def set_max(self, value):
        return value
    def get_max(self):
        return self.progress["maximum"]
    def is_cancelled(self):
        return self.cancelled
    #quit the progress bar window
        
    
#run when imported as a module
if __name__ == "__main__":

    #root = tk.Tk()
    app = ImageBrowser()
    app.mainloop()