import tkinter as tk
import os
import sys
import sysconfig
import subprocess
from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import json
from tkinter import messagebox
from PIL import Image, ImageTk
import glob
#from scripts import converters
#work in progress code, not finished, credits will be added at a later date.

#class to make popup right click menu with select all, copy, paste, cut, and delete when right clicked on an entry box


class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

class App(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        #define some colors
        self.stableTune_icon =PhotoImage(file = "resources/stableTuner_icon.png")
        self.master.iconphoto(False, self.stableTune_icon)
        self.dark_mode_var = "#1e2124"
        self.dark_purple_mode_var = "#1B0F1B"
        self.dark_mode_title_var = "#7289da"
        self.dark_mode_button_pressed_var = "#BB91B6"
        self.dark_mode_button_var = "#8ea0e1"
        self.dark_mode_text_var = "#c6c7c8"
        self.master.title("StableTune")
        self.master.configure(cursor="left_ptr")
        #resizable window
        self.master.resizable(True, True)
        #master canvas
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar = tk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.frame = tk.Frame(self.canvas)
        self.frame.pack(side="left", fill="both", expand=True)
        
        self.canvas.create_window((0,0), window=self.frame, anchor="nw")

        #create tabs
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        #create tabs
        self.general_tab = tk.Frame(self.notebook)
        self.training_tab = tk.Frame(self.notebook)
        self.dataset_tab = tk.Frame(self.notebook)
        self.sample_tab = tk.Frame(self.notebook)
        self.concepts_tab = tk.Frame(self.notebook)
        self.play_tab = tk.Frame(self.notebook)
        self.tools_tab = tk.Frame(self.notebook)
        self.notebook.add(self.general_tab, text="General Settings",sticky="n")
        self.notebook.add(self.training_tab, text="Training Settings",sticky="n")
        self.notebook.add(self.dataset_tab, text="Dataset Settings",sticky="n")
        self.notebook.add(self.sample_tab, text="Sample Settings",sticky="n")
        self.notebook.add(self.concepts_tab, text="Concept Settings",sticky="n")
        self.notebook.add(self.play_tab, text="Model Playground",sticky="n")
        self.notebook.add(self.tools_tab, text="Toolbox",sticky="n")
        self.general_tab.configure(bg=self.dark_mode_var)
        self.training_tab.configure(bg=self.dark_mode_var)
        self.dataset_tab.configure(bg=self.dark_mode_var)
        self.sample_tab.configure(bg=self.dark_mode_var)
        self.concepts_tab.configure(bg=self.dark_mode_var)
        self.play_tab.configure(bg=self.dark_mode_var)
        self.tools_tab.configure(bg=self.dark_mode_var)
        
        #notebook dark mode style
        self.notebook_style = ttk.Style()
        self.notebook_style.theme_use("clam")
        #dark mode
        self.notebook_style.configure("TNotebook", background=self.dark_mode_var, borderwidth=0, highlightthickness=0, lightcolor=self.dark_mode_var, darkcolor=self.dark_mode_var, bordercolor=self.dark_mode_var, tabmargins=[0,0,0,0], padding=[0,0,0,0], relief="flat")
        self.notebook_style.configure("TNotebook.Tab", background=self.dark_mode_var, borderwidth=0, highlightthickness=0, lightcolor=self.dark_mode_var, foreground=self.dark_mode_text_var, bordercolor=self.dark_mode_var,highlightcolor=self.dark_mode_var)
        self.notebook_style.map("TNotebook.Tab", background=[("selected", self.dark_mode_var)], foreground=[("selected", self.dark_mode_title_var), ("active", self.dark_mode_title_var)])
        
        #on tab change resize window
        self.notebook.bind("<<NotebookTabChanged>>", self.resize_window)

        #variables
        
        self.sample_prompts = []
        self.number_of_sample_prompts = len(self.sample_prompts)
        self.sample_prompt_labels = []
        self.diffusers_model_path = "stabilityai/stable-diffusion-2"
        self.vae_model_path = "stabilityai/sd-vae-ft-mse"
        self.output_path = "models/model_name"
        self.send_telegram_updates = False
        self.telegram_token = "TOKEN"
        self.telegram_chat_id = "ID"
        self.seed_number = 3434554
        self.resolution = 512
        self.batch_size = 1
        self.num_train_epochs = 5
        self.accumulation_steps = 1
        self.mixed_precision = "fp16"
        self.learning_rate = "5e-6"
        self.learning_rate_schedule = "constant"
        self.learning_rate_warmup_steps = 0
        self.concept_list_json_path = "concept_list.json"
        self.save_and_sample_every_x_epochs = 5
        self.train_text_encoder = True
        self.use_8bit_adam = True
        self.use_gradient_checkpointing = True
        self.num_class_images = 200
        self.add_class_images_to_training = False
        self.sample_batch_size = 1
        self.save_sample_controlled_seed = []
        self.delete_checkpoints_when_full_drive = True
        self.use_image_names_as_captions = True
        self.num_samples_to_generate = 1
        self.auto_balance_concept_datasets = True
        self.sample_width = 512
        self.sample_height = 512
        self.save_latents_cache = True
        self.regenerate_latents_cache = False
        self.use_aspect_ratio_bucketing = True
        self.do_not_use_latents_cache = True
        self.with_prior_reservation = False
        self.prior_loss_weight = 1.0
        self.sample_random_aspect_ratio = False
        self.add_controlled_seed_to_sample = []
        self.save_on_training_start = False
        self.concept_template = {'instance_prompt': 'subject', 'class_prompt': 'a photo of class', 'instance_data_dir':'./data/subject','class_data_dir':'./data/subject_class'}
        self.concepts = []
        self.play_diffusers_model_path = ""
        self.play_postive_prompt = ""
        self.play_negative_prompt = ""
        self.play_seed = -1
        self.play_num_samples = 1
        self.play_sample_width = 512
        self.play_sample_height = 512
        self.play_cfg = 7.5
        self.play_steps = 25
        self.schedulers = ["DPMSolverMultistepScheduler", "PNDMScheduler", 'DDIMScheduler','EulerAncestralDiscreteScheduler','EulerDiscreteScheduler']
        self.quick_select_models = ["Stable Diffusion 1.4", "Stable Diffusion 1.5", "Stable Diffusion 2 Base (512)", "Stable Diffusion 2 (768)", 'Stable Diffusion 2.1 Base (512)', "Stable Diffusion 2.1 (768)"]
        self.play_scheduler = 'DPMSolverMultistepScheduler'
        self.pipe = None
        self.current_model = None
        self.play_save_image_button = None
        self.dataset_repeats = 1
        self.limit_text_encoder = 0
        self.use_text_files_as_captions = False
        self.create_widgets()
 
        width = self.notebook.winfo_reqwidth()
        height = self.notebook.winfo_reqheight()
        self.master.geometry(f"{width}x{height}")
        self.master.update()
        #check if there is a stabletune_last_run.json file
        #if there is, load the settings from it
        if os.path.exists("stabletune_last_run.json"):
            try:
                self.load_config(file_name="stabletune_last_run.json")
                #try loading the latest generated model to playground entry
                self.playground_find_latest_generated_model()
            except:
                print("Error loading config file")
                #system sep
            #self.play_model_entry.insert(0, self.output_path_entry.get()+os.sep+self.train_epochs_entry.get())
        else:
            #self.load_config()
            pass
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    #create a right click menu for entry widgets
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
    def resize_window(self, event):
        #get the current selected notebook tab id
        tab_id = self.notebook.select()
        #get the tab index
        tab_index = self.notebook.index(tab_id)
        tabsSizes = {0 : [715,280], 1 : [715,490], 2 : [715,230],3 : [715,400],4 : [715,400],5 : [715,360],6 : [715,490]}
        #get the tab size
        tab_size = tabsSizes[tab_index]
        #resize the window to fit the widgets
        self.master.geometry(f"{tab_size[0]}x{tab_size[1]}")
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.master.update()
        
    def open_file(self):
        print("open file")
    def create_menu(self):
        #deprecated
        #menu frame dark mode
        self.menubar = tk.Menu(self.master,tearoff=0)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.load_config)
        self.filemenu.add_command(label="Save", command=self.save_config)
        #add clear settings
        #self.filemenu.add_command(label="Clear Settings", command=self.clear_settings)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.master.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)


        self.master.config(menu=self.menubar)
    def quick_select_model(self,*args):
        val = self.quick_select_var.get()
        if val != "Click to select model":
            #clear diffusers_model_path_entry
            self.diffusers_model_path_entry.delete(0, tk.END)
            if val == 'Stable Diffusion 1.4':
                self.diffusers_model_path_entry.insert(0,"CompVis/stable-diffusion-v1-4")
            elif val == 'Stable Diffusion 1.5':
                self.diffusers_model_path_entry.insert(0,"runwayml/stable-diffusion-v1-5")
            elif val == 'Stable Diffusion 2 Base (512)':
                self.diffusers_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-base")
            elif val == 'Stable Diffusion 2 (768)':
                self.diffusers_model_path_entry.insert(0,"stabilityai/stable-diffusion-2")
                self.resolution_var.set("768")
            elif val == 'Stable Diffusion 2.1 Base (512)':
                self.diffusers_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-1-base")
            elif val == 'Stable Diffusion 2.1 (768)':
                self.diffusers_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-1")
                self.resolution_var.set("768")
            self.master.update()
    def create_widgets(self):
        #create grid one side for labels the other for inputs
        #make the second column size 2x the first
        #create a model settings label in bold
        #add button to load config
        #add button to save config
        
        self.load_config_button = tk.Button(self.general_tab, text="Load Config", command=self.load_config,fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.load_config_button.configure(border=4, relief='flat')
        self.load_config_button.grid(row=0, column=1, sticky="nw")
        self.save_config_button = tk.Button(self.general_tab, text="Save Config", command=self.save_config,fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.save_config_button.configure(border=4, relief='flat')
        self.save_config_button.grid(row=0, column=1, sticky="ne")
        self.model_settings_label = tk.Label(self.general_tab, text="StableTune Settings",  font=("Arial", 12, "bold"), fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.model_settings_label.grid(row=0, column=0, sticky="nsew")

        self.quick_select_label = tk.Label(self.general_tab, text="Quick Select Model",  font=("Arial", 10, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        quick_select_label_ttp = CreateToolTip(self.quick_select_label, "Quick select another model to use.")
        self.quick_select_label.grid(row=1, column=0, sticky="nsew")
        self.quick_select_var = tk.StringVar()
        self.quick_select_var.set('Click to select')
        self.quick_select_dropdown = tk.OptionMenu(self.general_tab, self.quick_select_var, *self.quick_select_models, command=self.quick_select_model)
        self.quick_select_dropdown.config( activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.quick_select_dropdown["menu"].config( activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.quick_select_dropdown.grid(row=1, column=1, sticky="nsew")
        
        self.diffusers_model_path_label = tk.Label(self.general_tab, text="Diffusers model path / HuggingFace Repo",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        diffusers_model_path_label_ttp = CreateToolTip(self.diffusers_model_path_label, "The path to the diffusers model to use. Can be a local path or a HuggingFace repo path.")
        self.diffusers_model_path_label.grid(row=2, column=0, sticky="nsew")
        self.diffusers_model_path_entry = tk.Entry(self.general_tab,width=30,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        
        self.diffusers_model_path_entry.grid(row=2, column=1, sticky="nsew")
        self.diffusers_model_path_entry.insert(0, self.diffusers_model_path)
        #make a button to open a file dialog
        self.diffusers_model_path_button = tk.Button(self.general_tab, text="...", command=lambda: self.open_file_dialog(self.diffusers_model_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.diffusers_model_path_button.configure(border=4, relief='flat')
        self.diffusers_model_path_button.grid(row=2, column=2, sticky="nwse")
        #create vae model path dark mode
        self.vae_model_path_label = tk.Label(self.general_tab, text="VAE model path / HuggingFace Repo",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        vae_model_path_label_ttp = CreateToolTip(self.vae_model_path_label, "OPTINAL The path to the VAE model to use. Can be a local path or a HuggingFace repo path.")
        self.vae_model_path_label.grid(row=3, column=0, sticky="nsew")
        self.vae_model_path_entry = tk.Entry(self.general_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.vae_model_path_entry.grid(row=3, column=1, sticky="nsew")
        self.vae_model_path_entry.insert(0, self.vae_model_path)
        #make a button to open a file dialog
        self.vae_model_path_button = tk.Button(self.general_tab, text="...", command=lambda: self.open_file_dialog(self.vae_model_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.vae_model_path_button.configure(border=4, relief='flat')
        self.vae_model_path_button.grid(row=3, column=2, sticky="nsew")
        #create output path dark mode
        self.output_path_label = tk.Label(self.general_tab, text="Output Path",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        output_path_label_ttp = CreateToolTip(self.output_path_label, "The path to the output directory. If it doesn't exist, it will be created.")
        self.output_path_label.grid(row=4, column=0, sticky="nsew")
        self.output_path_entry = tk.Entry(self.general_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.output_path_entry.grid(row=4, column=1, sticky="nsew")
        self.output_path_entry.insert(0, self.output_path)
        #make a button to open a file dialog
        self.output_path_button = tk.Button(self.general_tab, text="...", command=lambda: self.open_file_dialog(self.output_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.output_path_button.configure(border=4, relief='flat')
        self.output_path_button.grid(row=4, column=2, sticky="nsew")
        #use telegram updates dark mode
        self.send_telegram_updates_label = tk.Label(self.general_tab, text="Send Telegram Updates",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        send_telegram_updates_label_ttp = CreateToolTip(self.send_telegram_updates_label, "Use Telegram updates to monitor training progress, must have a Telegram bot set up.")
        self.send_telegram_updates_label.grid(row=5, column=0, sticky="nsew")
        #create checkbox to toggle telegram updates and show telegram token and chat id
        self.send_telegram_updates_var = tk.IntVar()
        self.send_telegram_updates_checkbox = tk.Checkbutton(self.general_tab,variable=self.send_telegram_updates_var, command=self.toggle_telegram_settings,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.send_telegram_updates_checkbox.grid(row=5, column=1, sticky="nsew")
        #create telegram token dark mode
        self.telegram_token_label = tk.Label(self.general_tab, text="Telegram Token",  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        telegram_token_label_ttp = CreateToolTip(self.telegram_token_label, "The Telegram token for your bot.")
        self.telegram_token_label.grid(row=6, column=0, sticky="nsew")
        self.telegram_token_entry = tk.Entry(self.general_tab,  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var, disabledbackground=self.dark_mode_var,insertbackground="white")
        self.telegram_token_entry.grid(row=6, column=1, sticky="nsew")
        self.telegram_token_entry.insert(0, self.telegram_token)
        #create telegram chat id dark mode
        self.telegram_chat_id_label = tk.Label(self.general_tab, text="Telegram Chat ID",  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        telegram_chat_id_label_ttp = CreateToolTip(self.telegram_chat_id_label, "The Telegram chat ID to send updates to.")
        self.telegram_chat_id_label.grid(row=7, column=0, sticky="nsew")
        self.telegram_chat_id_entry = tk.Entry(self.general_tab,  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var, disabledbackground=self.dark_mode_var,insertbackground="white")
        self.telegram_chat_id_entry.grid(row=7, column=1, sticky="nsew")
        self.telegram_chat_id_entry.insert(0, self.telegram_chat_id)

        #Training settings label in bold
        self.training_settings_label = tk.Label(self.training_tab, text="Training Settings",  font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.training_settings_label.grid(row=0, column=0, sticky="nsew")
        #add a seed entry
        self.seed_label = tk.Label(self.training_tab, text="Seed",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        seed_label_ttp = CreateToolTip(self.seed_label, "The seed to use for training.")
        self.seed_label.grid(row=1, column=0, sticky="nsew")
        self.seed_entry = tk.Entry(self.training_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.seed_entry.grid(row=1, column=1, sticky="nsew")
        self.seed_entry.insert(0, self.seed_number)

        #create resolution dark mode dropdown
        self.resolution_label = tk.Label(self.training_tab, text="Resolution",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        resolution_label_ttp = CreateToolTip(self.resolution_label, "The resolution of the images to train on.")
        self.resolution_label.grid(row=2, column=0, sticky="nsew")
        self.resolution_var = tk.StringVar()
        self.resolution_var.set(self.resolution)
        self.resolution_dropdown = tk.OptionMenu(self.training_tab, self.resolution_var, "256", "320", "384", "448","512", "576", "640", "704", "768", "832", "896", "960", "1024")
        self.resolution_dropdown.config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_text_var, bg=self.dark_mode_var, highlightthickness=1, highlightbackground="black", highlightcolor="yellow")
        self.resolution_dropdown["menu"].config( activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.resolution_dropdown.grid(row=2, column=1, sticky="nsew")

        #create train batch size dark mode dropdown with values from 1 to 60
        self.train_batch_size_label = tk.Label(self.training_tab, text="Train Batch Size",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        train_batch_size_label_ttp = CreateToolTip(self.train_batch_size_label, "The batch size to use for training.")
        self.train_batch_size_label.grid(row=3, column=0, sticky="nsew")
        self.train_batch_size_var = tk.StringVar()
        self.train_batch_size_var.set(self.batch_size)
        self.train_batch_size_dropdown = tk.OptionMenu(self.training_tab, self.train_batch_size_var, *range(1, 61))
        self.train_batch_size_dropdown.config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_text_var, bg=self.dark_mode_var, highlightthickness=1, highlightbackground="black", highlightcolor="yellow")
        self.train_batch_size_dropdown["menu"].config( activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.train_batch_size_dropdown.grid(row=3, column=1, sticky="nsew")

        #create train epochs dark mode 
        self.train_epochs_label = tk.Label(self.training_tab, text="Train Epochs",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        train_epochs_label_ttp = CreateToolTip(self.train_epochs_label, "The number of epochs to train for. An epoch is one pass through the entire dataset.")
        self.train_epochs_label.grid(row=4, column=0, sticky="nsew")
        self.train_epochs_entry = tk.Entry(self.training_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.train_epochs_entry.grid(row=4, column=1, sticky="nsew")
        self.train_epochs_entry.insert(0, self.num_train_epochs)
        
        #create mixed precision dark mode dropdown
        self.mixed_precision_label = tk.Label(self.training_tab, text="Mixed Precision",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        mixed_precision_label_ttp = CreateToolTip(self.mixed_precision_label, "Use mixed precision training to speed up training, FP16 is recommended but requires a GPU with Tensor Cores.")
        self.mixed_precision_label.grid(row=5, column=0, sticky="nsew")
        self.mixed_precision_var = tk.StringVar()
        self.mixed_precision_var.set(self.mixed_precision)
        self.mixed_precision_dropdown = tk.OptionMenu(self.training_tab, self.mixed_precision_var, "fp16", "fp32")
        self.mixed_precision_dropdown.config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_text_var, bg=self.dark_mode_var, highlightthickness=1, highlightbackground="black", highlightcolor="yellow")
        self.mixed_precision_dropdown["menu"].config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.mixed_precision_dropdown.grid(row=5, column=1, sticky="nsew")

        #create use 8bit adam checkbox
        self.use_8bit_adam_var = tk.IntVar()
        self.use_8bit_adam_var.set(self.use_8bit_adam)
        #create label
        self.use_8bit_adam_label = tk.Label(self.training_tab, text="Use 8bit Adam",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        use_8bit_adam_label_ttp = CreateToolTip(self.use_8bit_adam_label, "Use 8bit Adam to speed up training, requires bytsandbytes.")
        self.use_8bit_adam_label.grid(row=6, column=0, sticky="nsew")
        #create checkbox
        self.use_8bit_adam_checkbox = tk.Checkbutton(self.training_tab, variable=self.use_8bit_adam_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.use_8bit_adam_checkbox.grid(row=6, column=1, sticky="nsew")
        #create use gradient checkpointing checkbox
        self.use_gradient_checkpointing_var = tk.IntVar()
        self.use_gradient_checkpointing_var.set(self.use_gradient_checkpointing)
        #create label
        self.use_gradient_checkpointing_label = tk.Label(self.training_tab, text="Use Gradient Checkpointing",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        use_gradient_checkpointing_label_ttp = CreateToolTip(self.use_gradient_checkpointing_label, "Use gradient checkpointing to reduce RAM usage.")
        self.use_gradient_checkpointing_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox
        self.use_gradient_checkpointing_checkbox = tk.Checkbutton(self.training_tab, variable=self.use_gradient_checkpointing_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.use_gradient_checkpointing_checkbox.grid(row=7, column=1, sticky="nsew")
        #create gradient accumulation steps dark mode dropdown with values from 1 to 60
        self.gradient_accumulation_steps_label = tk.Label(self.training_tab, text="Gradient Accumulation Steps",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        gradient_accumulation_steps_label_ttp = CreateToolTip(self.gradient_accumulation_steps_label, "The number of gradient accumulation steps to use, this is useful for training with limited GPU memory.")
        self.gradient_accumulation_steps_label.grid(row=8, column=0, sticky="nsew")
        self.gradient_accumulation_steps_var = tk.StringVar()
        self.gradient_accumulation_steps_var.set(self.accumulation_steps)
        self.gradient_accumulation_steps_dropdown = tk.OptionMenu(self.training_tab, self.gradient_accumulation_steps_var, *range(1, 61))
        self.gradient_accumulation_steps_dropdown.config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_text_var, bg=self.dark_mode_var, highlightthickness=1, highlightbackground="black", highlightcolor="yellow")
        self.gradient_accumulation_steps_dropdown["menu"].config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.gradient_accumulation_steps_dropdown.grid(row=8, column=1, sticky="nsew")
        #create learning rate dark mode entry
        self.learning_rate_label = tk.Label(self.training_tab, text="Learning Rate",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        learning_rate_label_ttp = CreateToolTip(self.learning_rate_label, "The learning rate to use for training.")
        self.learning_rate_label.grid(row=9, column=0, sticky="nsew")
        self.learning_rate_entry = tk.Entry(self.training_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.learning_rate_entry.grid(row=9, column=1, sticky="nsew")
        self.learning_rate_entry.insert(0, self.learning_rate)
        #create learning rate scheduler dropdown
        self.learning_rate_scheduler_label = tk.Label(self.training_tab, text="Learning Rate Scheduler",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        learning_rate_scheduler_label_ttp = CreateToolTip(self.learning_rate_scheduler_label, "The learning rate scheduler to use for training.")
        self.learning_rate_scheduler_label.grid(row=10, column=0, sticky="nsew")
        self.learning_rate_scheduler_var = tk.StringVar()
        self.learning_rate_scheduler_var.set(self.learning_rate_schedule)
        self.learning_rate_scheduler_dropdown = tk.OptionMenu(self.training_tab, self.learning_rate_scheduler_var, "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup")
        self.learning_rate_scheduler_dropdown.config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_text_var, bg=self.dark_mode_var, highlightthickness=1, highlightbackground="black", highlightcolor="yellow")
        self.learning_rate_scheduler_dropdown["menu"].config(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.learning_rate_scheduler_dropdown.grid(row=10, column=1, sticky="nsew")
        #create num warmup steps dark mode entry
        self.num_warmup_steps_label = tk.Label(self.training_tab, text="LR Warmup Steps",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        num_warmup_steps_label_ttp = CreateToolTip(self.num_warmup_steps_label, "The number of warmup steps to use for the learning rate scheduler.")
        self.num_warmup_steps_label.grid(row=11, column=0, sticky="nsew")
        self.num_warmup_steps_entry = tk.Entry(self.training_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.num_warmup_steps_entry.grid(row=11, column=1, sticky="nsew")
        self.num_warmup_steps_entry.insert(0, self.learning_rate_warmup_steps)
        #create use latent cache checkbox
        self.use_latent_cache_var = tk.IntVar()
        self.use_latent_cache_var.set(self.do_not_use_latents_cache)
        #create label
        self.use_latent_cache_label = tk.Label(self.training_tab, text="Use Latent Cache",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        use_latent_cache_label_ttp = CreateToolTip(self.use_latent_cache_label, "Cache the latents to speed up training.")
        self.use_latent_cache_label.grid(row=12, column=0, sticky="nsew")
        #create checkbox
        self.use_latent_cache_checkbox = tk.Checkbutton(self.training_tab, variable=self.use_latent_cache_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.use_latent_cache_checkbox.grid(row=12, column=1, sticky="nsew")
        #create save latent cache checkbox
        self.save_latent_cache_var = tk.IntVar()
        self.save_latent_cache_var.set(self.save_latents_cache)
        #create label
        self.save_latent_cache_label = tk.Label(self.training_tab, text="Save Latent Cache",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        save_latent_cache_label_ttp = CreateToolTip(self.save_latent_cache_label, "Save the latents cache to disk after generation, will be remade if batch size changes.")
        self.save_latent_cache_label.grid(row=13, column=0, sticky="nsew")
        #create checkbox
        self.save_latent_cache_checkbox = tk.Checkbutton(self.training_tab, variable=self.save_latent_cache_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.save_latent_cache_checkbox.grid(row=13, column=1, sticky="nsew")
        #create regnerate latent cache checkbox
        self.regenerate_latent_cache_var = tk.IntVar()
        self.regenerate_latent_cache_var.set(self.regenerate_latents_cache)
        #create label
        self.regenerate_latent_cache_label = tk.Label(self.training_tab, text="Regenerate Latent Cache",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        regenerate_latent_cache_label_ttp = CreateToolTip(self.regenerate_latent_cache_label, "Force the latents cache to be regenerated.")
        self.regenerate_latent_cache_label.grid(row=14, column=0, sticky="nsew")
        #create checkbox
        self.regenerate_latent_cache_checkbox = tk.Checkbutton(self.training_tab, variable=self.regenerate_latent_cache_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.regenerate_latent_cache_checkbox.grid(row=14, column=1, sticky="nsew")
        #create train text encoder checkbox
        self.train_text_encoder_var = tk.IntVar()
        self.train_text_encoder_var.set(self.train_text_encoder)
        #create label
        self.train_text_encoder_label = tk.Label(self.training_tab, text="Train Text Encoder",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        train_text_encoder_label_ttp = CreateToolTip(self.train_text_encoder_label, "Train the text encoder along with the UNET.")
        self.train_text_encoder_label.grid(row=15, column=0, sticky="nsew")
        #create checkbox
        self.train_text_encoder_checkbox = tk.Checkbutton(self.training_tab, variable=self.train_text_encoder_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.train_text_encoder_checkbox.grid(row=15, column=1, sticky="nsew")
        #create limit text encoder encoder entry
        self.limit_text_encoder_var = tk.StringVar()
        self.limit_text_encoder_var.set(self.limit_text_encoder)
        #create label
        self.limit_text_encoder_label = tk.Label(self.training_tab, text="Limit Text Encoder",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        limit_text_encoder_label_ttp = CreateToolTip(self.limit_text_encoder_label, "Stop training the text encoder after this many epochs, use % to train for a percentage of the total epochs.")
        self.limit_text_encoder_label.grid(row=16, column=0, sticky="nsew")
        #create entry
        self.limit_text_encoder_entry = tk.Entry(self.training_tab, textvariable=self.limit_text_encoder_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.limit_text_encoder_entry.grid(row=16, column=1, sticky="nsew")
        #create with prior loss preservation checkbox
        self.with_prior_loss_preservation_var = tk.IntVar()
        self.with_prior_loss_preservation_var.set(self.with_prior_reservation)
        #create label
        self.with_prior_loss_preservation_label = tk.Label(self.training_tab, text="With Prior Loss Preservation",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        with_prior_loss_preservation_label_ttp = CreateToolTip(self.with_prior_loss_preservation_label, "Use the prior loss preservation method. part of Dreambooth.")
        self.with_prior_loss_preservation_label.grid(row=17, column=0, sticky="nsew")
        #create checkbox
        self.with_prior_loss_preservation_checkbox = tk.Checkbutton(self.training_tab, variable=self.with_prior_loss_preservation_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.with_prior_loss_preservation_checkbox.grid(row=17, column=1, sticky="nsew")
        #create prior loss preservation weight entry
        self.prior_loss_preservation_weight_label = tk.Label(self.training_tab, text="Weight",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        prior_loss_preservation_weight_label_ttp = CreateToolTip(self.prior_loss_preservation_weight_label, "The weight of the prior loss preservation loss.")
        self.prior_loss_preservation_weight_label.grid(row=17, column=1, sticky="e")
        self.prior_loss_preservation_weight_entry = tk.Entry(self.training_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.prior_loss_preservation_weight_entry.grid(row=17, column=3, sticky="w")
        self.prior_loss_preservation_weight_entry.insert(0, self.prior_loss_weight)
        #create Dataset Settings label like the model settings label
        self.dataset_settings_label = tk.Label(self.dataset_tab, text="Dataset Settings", font=("Arial", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.dataset_settings_label.grid(row=0, column=0, sticky="nsew")
        
        #create use text files as captions checkbox
        self.use_text_files_as_captions_var = tk.IntVar()
        self.use_text_files_as_captions_var.set(self.use_text_files_as_captions)
        #create label
        self.use_text_files_as_captions_label = tk.Label(self.dataset_tab, text="Use Text Files as Captions",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        use_text_files_as_captions_label_ttp = CreateToolTip(self.use_text_files_as_captions_label, "Use the text files as captions for training, text files must have same name as image, instance prompt/token will be ignored.")
        self.use_text_files_as_captions_label.grid(row=1, column=0, sticky="nsew")
        #create checkbox
        self.use_text_files_as_captions_checkbox = tk.Checkbutton(self.dataset_tab, variable=self.use_text_files_as_captions_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.use_text_files_as_captions_checkbox.grid(row=1, column=1, sticky="nsew")
        # create use image names as captions checkbox
        self.use_image_names_as_captions_var = tk.IntVar()
        self.use_image_names_as_captions_var.set(self.use_image_names_as_captions)
        # create label
        self.use_image_names_as_captions_label = tk.Label(self.dataset_tab, text="Use Image Names as Captions",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        use_image_names_as_captions_label_ttp = CreateToolTip(self.use_image_names_as_captions_label, "Use the image names as captions for training, instance prompt/token will be ignored.")
        self.use_image_names_as_captions_label.grid(row=2, column=0, sticky="nsew")
        # create checkbox
        self.use_image_names_as_captions_checkbox = tk.Checkbutton(self.dataset_tab, variable=self.use_image_names_as_captions_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.use_image_names_as_captions_checkbox.grid(row=2, column=1, sticky="nsew")
        # create auto balance dataset checkbox
        self.auto_balance_dataset_var = tk.IntVar()
        self.auto_balance_dataset_var.set(self.auto_balance_concept_datasets)
        # create label
        self.auto_balance_dataset_label = tk.Label(self.dataset_tab, text="Auto Balance Dataset",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        auto_balance_dataset_label_ttp = CreateToolTip(self.auto_balance_dataset_label, "Will use the concept with the least amount of images to balance the dataset by removing images from the other concepts.")
        self.auto_balance_dataset_label.grid(row=3, column=0, sticky="nsew")
        # create checkbox
        self.auto_balance_dataset_checkbox = tk.Checkbutton(self.dataset_tab, variable=self.auto_balance_dataset_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.auto_balance_dataset_checkbox.grid(row=3, column=1, sticky="nsew")
        #create add class images to dataset checkbox
        self.add_class_images_to_dataset_var = tk.IntVar()
        self.add_class_images_to_dataset_var.set(self.add_class_images_to_training)
        #create label
        self.add_class_images_to_dataset_label = tk.Label(self.dataset_tab, text="Add Class Images to Dataset",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        add_class_images_to_dataset_label_ttp = CreateToolTip(self.add_class_images_to_dataset_label, "Will add class images without prior preservation to the dataset.")
        self.add_class_images_to_dataset_label.grid(row=4, column=0, sticky="nsew")
        #create checkbox
        self.add_class_images_to_dataset_checkbox = tk.Checkbutton(self.dataset_tab, variable=self.add_class_images_to_dataset_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.add_class_images_to_dataset_checkbox.grid(row=4, column=1, sticky="nsew")
        #create number of class images entry
        self.number_of_class_images_label = tk.Label(self.dataset_tab, text="Number of Class Images",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        number_of_class_images_label_ttp = CreateToolTip(self.number_of_class_images_label, "The number of class images to add to the dataset, if they don't exist in the class directory they will be generated.")
        self.number_of_class_images_label.grid(row=5, column=0, sticky="nsew")
        self.number_of_class_images_entry = tk.Entry(self.dataset_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.number_of_class_images_entry.grid(row=5, column=1, sticky="nsew")
        self.number_of_class_images_entry.insert(0, self.num_class_images)
        #create dataset repeat entry
        self.dataset_repeats_label = tk.Label(self.dataset_tab, text="Dataset Repeats",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        dataset_repeat_label_ttp = CreateToolTip(self.dataset_repeats_label, "The number of times to repeat the dataset, this will increase the number of images in the dataset.")
        self.dataset_repeats_label.grid(row=6, column=0, sticky="nsew")
        self.dataset_repeats_entry = tk.Entry(self.dataset_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.dataset_repeats_entry.grid(row=6, column=1, sticky="nsew")
        self.dataset_repeats_entry.insert(0, self.dataset_repeats)

        #add use_aspect_ratio_bucketing checkbox
        self.use_aspect_ratio_bucketing_var = tk.IntVar()
        self.use_aspect_ratio_bucketing_var.set(self.use_aspect_ratio_bucketing)
        #create label
        self.use_aspect_ratio_bucketing_label = tk.Label(self.dataset_tab, text="Use Aspect Ratio Bucketing",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        use_aspect_ratio_bucketing_label_ttp = CreateToolTip(self.use_aspect_ratio_bucketing_label, "Will use aspect ratio bucketing, may improve aspect ratio generations.")
        self.use_aspect_ratio_bucketing_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox
        self.use_aspect_ratio_bucketing_checkbox = tk.Checkbutton(self.dataset_tab, variable=self.use_aspect_ratio_bucketing_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.use_aspect_ratio_bucketing_checkbox.grid(row=7, column=1, sticky="nsew")
        #do something on checkbox click
        self.use_aspect_ratio_bucketing_checkbox.bind("<Button-1>", self.disable_with_prior_loss)
        #add download dataset entry
        
        #add download dataset entry
        #create Sampling Settings label like the model settings label
        self.sampling_settings_label = tk.Label(self.sample_tab, text="Sampling Settings", font=("Arial", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.sampling_settings_label.grid(row=0, column=0, sticky="nsew")
        #create saver every n epochs entry
        self.save_every_n_epochs_label = tk.Label(self.sample_tab, text="Save Every N Epochs",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        save_every_n_epochs_label_ttp = CreateToolTip(self.save_every_n_epochs_label, "Will save and sample the model every N epochs.")
        self.save_every_n_epochs_label.grid(row=1, column=0, sticky="nsew")
        self.save_every_n_epochs_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.save_every_n_epochs_entry.grid(row=1, column=1, sticky="nsew")
        self.save_every_n_epochs_entry.insert(0, self.save_and_sample_every_x_epochs)
        #create number of samples to generate entry
        self.number_of_samples_to_generate_label = tk.Label(self.sample_tab, text="Number of Samples to Generate",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        number_of_samples_to_generate_label_ttp = CreateToolTip(self.number_of_samples_to_generate_label, "The number of samples to generate per prompt.")
        self.number_of_samples_to_generate_label.grid(row=2, column=0, sticky="nsew")
        self.number_of_samples_to_generate_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.number_of_samples_to_generate_entry.grid(row=2, column=1, sticky="nsew")
        self.number_of_samples_to_generate_entry.insert(0, self.num_samples_to_generate)
        #create sample width entry
        self.sample_width_label = tk.Label(self.sample_tab, text="Sample Width",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_width_label_ttp = CreateToolTip(self.sample_width_label, "The width of the generated samples.")
        self.sample_width_label.grid(row=3, column=0, sticky="nsew")
        self.sample_width_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.sample_width_entry.grid(row=3, column=1, sticky="nsew")
        self.sample_width_entry.insert(0, self.sample_width)
        #create sample height entry
        self.sample_height_label = tk.Label(self.sample_tab, text="Sample Height",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_height_label_ttp = CreateToolTip(self.sample_height_label, "The height of the generated samples.")
        self.sample_height_label.grid(row=4, column=0, sticky="nsew")
        self.sample_height_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.sample_height_entry.grid(row=4, column=1, sticky="nsew")
        self.sample_height_entry.insert(0, self.sample_height)
        
        #create a checkbox to save_on_training_start
        self.save_on_training_start_var = tk.IntVar()
        self.save_on_training_start_var.set(self.save_on_training_start)
        #create label
        self.save_on_training_start_label = tk.Label(self.sample_tab, text="Save On Training Start",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        save_on_training_start_label_ttp = CreateToolTip(self.save_on_training_start_label, "Will save and sample the model on training start, useful for debugging and comparison.")
        self.save_on_training_start_label.grid(row=5, column=0, sticky="nsew")
        #create checkbox
        self.save_on_training_start_checkbox = tk.Checkbutton(self.sample_tab, variable=self.save_on_training_start_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.save_on_training_start_checkbox.grid(row=5, column=1, sticky="nsew")
        #create sample random aspect ratio checkbox
        self.sample_random_aspect_ratio_var = tk.IntVar()
        self.sample_random_aspect_ratio_var.set(self.sample_random_aspect_ratio)
        #create label
        self.sample_random_aspect_ratio_label = tk.Label(self.sample_tab, text="Sample Random Aspect Ratio",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_random_aspect_ratio_label_ttp = CreateToolTip(self.sample_random_aspect_ratio_label, "Will generate samples with random aspect ratios, useful to check aspect ratio bucketing.")
        self.sample_random_aspect_ratio_label.grid(row=6, column=0, sticky="nsew")
        #create checkbox
        self.sample_random_aspect_ratio_checkbox = tk.Checkbutton(self.sample_tab, variable=self.sample_random_aspect_ratio_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.sample_random_aspect_ratio_checkbox.grid(row=6, column=1, sticky="nsew")
        #create add sample prompt button
        self.add_sample_prompt_button = tk.Button(self.sample_tab, text="Add Sample Prompt",  command=self.add_sample_prompt,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.add_sample_prompt_button.configure(border=4, relief='flat')
        add_sample_prompt_button_ttp = CreateToolTip(self.add_sample_prompt_button, "Add a sample prompt to the list.")
        self.add_sample_prompt_button.grid(row=7, column=0, sticky="nsew")
        #create remove sample prompt button
        self.remove_sample_prompt_button = tk.Button(self.sample_tab, text="Remove Sample Prompt",  command=self.remove_sample_prompt,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.remove_sample_prompt_button.configure(border=4, relief='flat')
        remove_sample_prompt_button_ttp = CreateToolTip(self.remove_sample_prompt_button, "Remove a sample prompt from the list.")
        self.remove_sample_prompt_button.grid(row=7, column=1, sticky="nsew")

        #for every prompt in self.sample_prompts, create a label and entry
        self.sample_prompt_labels = []
        self.sample_prompt_entries = []
        self.sample_prompt_row = 8
        for i in range(len(self.sample_prompts)):
            #create label
            self.sample_prompt_labels.append(tk.Label(self.sample_tab, text="Sample Prompt " + str(i),fg=self.dark_mode_text_var, bg=self.dark_mode_var))
            self.sample_prompt_labels[i].grid(row=self.sample_prompt_row + i, column=0, sticky="nsew")
            #create entry
            self.sample_prompt_entries.append(tk.Entry(self.sample_tab, width=70,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white"))
            self.sample_prompt_entries[i].grid(row=self.sample_prompt_row + i, column=1, sticky="nsew")
            self.sample_prompt_entries[i].insert(0, self.sample_prompts[i])
        for i in self.sample_prompt_entries:
            i.bind("<Button-3>", self.create_right_click_menu)
        self.controlled_sample_row = 30 + len(self.sample_prompts)
        #create a button to add controlled seed sample
        self.add_controlled_seed_sample_button = tk.Button(self.sample_tab, text="Add Controlled Seed Sample",  command=self.add_controlled_seed_sample,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.add_controlled_seed_sample_button.configure(border=4, relief='flat')
        add_controlled_seed_sample_button_ttp = CreateToolTip(self.add_controlled_seed_sample_button, "Will generate a sample using the seed at every save interval.")
        self.add_controlled_seed_sample_button.grid(row=self.controlled_sample_row + len(self.sample_prompts), column=0, sticky="nsew")
        #create a button to remove controlled seed sample
        self.remove_controlled_seed_sample_button = tk.Button(self.sample_tab, text="Remove Controlled Seed Sample",  command=self.remove_controlled_seed_sample,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.remove_controlled_seed_sample_button.configure(border=4, relief='flat')
        remove_controlled_seed_sample_button_ttp = CreateToolTip(self.remove_controlled_seed_sample_button, "Will remove the last controlled seed sample.")
        self.remove_controlled_seed_sample_button.grid(row=self.controlled_sample_row + len(self.sample_prompts), column=1, sticky="nsew")
        #for every controlled seed sample in self.controlled_seed_samples, create a label and entry
        self.controlled_seed_sample_labels = []
        self.controlled_seed_sample_entries = []
        for i in range(len(self.add_controlled_seed_to_sample)):
            #create label
            self.controlled_seed_sample_labels.append(tk.Label(self.sample_tab, text="Controlled Seed Sample " + str(i),fg=self.dark_mode_text_var, bg=self.dark_mode_var))
            self.controlled_seed_sample_labels[i].grid(row=self.controlled_sample_row + len(self.sample_prompts) + i, column=0, sticky="nsew")
            #create entry
            self.controlled_seed_sample_entries.append(tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white"))
            self.controlled_seed_sample_entries[i].grid(row=self.controlled_sample_row + len(self.sample_prompts) + i, column=1, sticky="nsew")
            self.controlled_seed_sample_entries[i].insert(0, self.add_controlled_seed_to_sample[i])
        for i in self.controlled_seed_sample_entries:
            i.bind("<Button-3>", self.create_right_click_menu)
        
        #add concept settings label
        self.concept_settings_label = tk.Label(self.concepts_tab, text="Concept Settings",  font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.concept_settings_label.grid(row=0, column=0, sticky="nsew")

        #add load concept from json button
        self.load_concept_from_json_button = tk.Button(self.concepts_tab, text="Load Concepts From JSON",  command=self.load_concept_from_json,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.load_concept_from_json_button.configure(border=4, relief='flat')
        load_concept_from_json_button_ttp = CreateToolTip(self.load_concept_from_json_button, "Load concepts from a JSON file, compatible with Shivam's concept list.")
        self.load_concept_from_json_button.grid(row=1, column=0, sticky="nsew")
        #add save concept to json button
        self.save_concept_to_json_button = tk.Button(self.concepts_tab, text="Save Concepts To JSON",  command=self.save_concept_to_json,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.save_concept_to_json_button.configure(border=4, relief='flat')
        save_concept_to_json_button_ttp = CreateToolTip(self.save_concept_to_json_button, "Save concepts to a JSON file, compatible with Shivam's concept list.")
        self.save_concept_to_json_button.grid(row=1, column=1, sticky="nsew")
        #create a button to add concept
        self.add_concept_button = tk.Button(self.concepts_tab, text="Add Concept",  command=self.add_concept,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.add_concept_button.configure(border=4, relief='flat')
        self.add_concept_button.grid(row=2, column=0, sticky="nsew")
        #create a button to remove concept
        self.remove_concept_button = tk.Button(self.concepts_tab, text="Remove Concept",  command=self.remove_concept,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.remove_concept_button.configure(border=4, relief='flat')
        self.remove_concept_button.grid(row=2, column=1, sticky="nsew")
        self.concept_entries = []
        self.concept_labels = []
        self.concept_file_dialog_buttons = []
        
        #add play model entry with button to open file dialog
        self.play_model_label = tk.Label(self.play_tab, text="Diffusers Model Directory",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_model_label.grid(row=0, column=0, sticky="nsew")
        self.play_model_entry = tk.Entry(self.play_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.play_model_entry.grid(row=0, column=1, sticky="nsew")
        self.play_model_entry.insert(0, self.play_diffusers_model_path)
        self.play_model_file_dialog_button = tk.Button(self.play_tab, text="...",width=5, command=lambda: self.open_file_dialog(self.play_model_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.play_model_file_dialog_button.configure(border=4, relief='flat')
        self.play_model_file_dialog_button.grid(row=0, column=2, sticky="w")
        #add a prompt entry to play tab
        self.play_prompt_label = tk.Label(self.play_tab, text="Prompt",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_prompt_label.grid(row=1, column=0, sticky="nsew")
        self.play_prompt_entry = tk.Entry(self.play_tab, width=40,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.play_prompt_entry.grid(row=1, column=1, sticky="nsew")
        self.play_prompt_entry.insert(0, self.play_postive_prompt)
        #add a negative prompt entry to play tab
        self.play_negative_prompt_label = tk.Label(self.play_tab, text="Negative Prompt",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_negative_prompt_label.grid(row=2, column=0, sticky="nsew")
        self.play_negative_prompt_entry = tk.Entry(self.play_tab, width=40,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.play_negative_prompt_entry.grid(row=2, column=1, sticky="nsew")
        self.play_negative_prompt_entry.insert(0, self.play_negative_prompt)
        #add a seed entry to play tab
        self.play_seed_label = tk.Label(self.play_tab, text="Seed",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_seed_label.grid(row=3, column=0, sticky="nsew")
        self.play_seed_entry = tk.Entry(self.play_tab, width=10,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.play_seed_entry.grid(row=3, column=1, sticky="w")
        self.play_seed_entry.insert(0, self.play_seed)
        #create a steps slider from 1 to 100
        self.play_steps_label = tk.Label(self.play_tab, text="Steps",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_steps_label.grid(row=4, column=0, sticky="nsew")
        self.play_steps_slider = tk.Scale(self.play_tab, from_=1, to=100, orient=tk.HORIZONTAL,fg=self.dark_mode_text_var, bg=self.dark_mode_var,border=0, highlightthickness=0, troughcolor=self.dark_mode_title_var, sliderrelief='flat', sliderlength=20, length=200,activebackground=self.dark_mode_var)
        self.play_steps_slider.grid(row=4, column=1, sticky="nsew")
        self.play_steps_slider.set(self.play_steps)
        #add a scheduler selection box

        
        self.play_scheduler_label = tk.Label(self.play_tab, text="Scheduler",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_scheduler_label.grid(row=5, column=0, sticky="nsew")
        self.play_scheduler_variable = tk.StringVar(self.play_tab)
        self.play_scheduler_variable.set(self.play_scheduler)
        self.play_scheduler_option_menu = tk.OptionMenu(self.play_tab, self.play_scheduler_variable, *self.schedulers,)
        self.play_scheduler_option_menu.configure(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_text_var, bg=self.dark_mode_var, highlightthickness=1, highlightbackground="black", highlightcolor="yellow")
        self.play_scheduler_option_menu["menu"].configure(activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.play_scheduler_option_menu.grid(row=5, column=1, sticky="nsew")
        
        #add resoltuion slider from 256 to 1024 in increments of 64 for width and height
        self.play_resolution_label = tk.Label(self.play_tab, text="Resolution",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_resolution_label.grid(row=6,rowspan=2, column=0, sticky="nsew")
        self.play_resolution_label_height = tk.Label(self.play_tab, text="Height",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_resolution_label_width = tk.Label(self.play_tab, text="Width",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_resolution_label_height.grid(row=6, column=1, sticky="e")
        self.play_resolution_label_width.grid(row=6, column=1, sticky="w")
        #add sliders for height and width
        self.play_resolution_slider_height = tk.Scale(self.play_tab, from_=256, to=2048, orient=tk.HORIZONTAL, resolution=64,fg=self.dark_mode_text_var, bg=self.dark_mode_var,border=0, highlightthickness=0, troughcolor=self.dark_mode_title_var, sliderrelief='flat', sliderlength=20,activebackground=self.dark_mode_var)
        self.play_resolution_slider_width = tk.Scale(self.play_tab, from_=256, to=2048, orient=tk.HORIZONTAL, resolution=64,fg=self.dark_mode_text_var, bg=self.dark_mode_var,border=0, highlightthickness=0, troughcolor=self.dark_mode_title_var, sliderrelief='flat', sliderlength=20,activebackground=self.dark_mode_var)
        self.play_resolution_slider_height.grid(row=7, column=1, sticky="e")
        self.play_resolution_slider_width.grid(row=7, column=1, sticky="w")
        self.play_resolution_slider_height.set(self.play_sample_height)
        self.play_resolution_slider_width.set(self.play_sample_width)
        #add a cfg slider 0.5 to 25 in increments of 0.5
        self.play_cfg_label = tk.Label(self.play_tab, text="CFG",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_cfg_label.grid(row=8, column=0, sticky="nsew")
        self.play_cfg_slider = tk.Scale(self.play_tab, from_=0.5, to=25, orient=tk.HORIZONTAL, resolution=0.5,fg=self.dark_mode_text_var, bg=self.dark_mode_var,border=0, highlightthickness=0, troughcolor=self.dark_mode_title_var, sliderrelief='flat', sliderlength=20, length=200,activebackground=self.dark_mode_var)
        self.play_cfg_slider.grid(row=8, column=1, sticky="nsew")
        self.play_cfg_slider.set(self.play_cfg)
        #add Toolbox label
        self.play_toolbox_label = tk.Label(self.play_tab, text="Toolbox",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.play_toolbox_label.grid(row=9, column=0, sticky="nsew")
        self.play_generate_image_button = tk.Button(self.play_tab, text="Generate Image", command=lambda: self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), self.play_resolution_slider_height.get(), self.play_resolution_slider_width.get(), self.play_cfg_slider.get(), self.play_steps_slider.get()),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.play_generate_image_button.configure(border=4, relief='flat')
        self.play_generate_image_button.grid(row=10, column=0, columnspan=2, sticky="nsew")
        #create a canvas to display the generated image
        self.play_image_canvas = tk.Canvas(self.play_tab, width=512, height=512, bg=self.dark_mode_var, highlightthickness=0)
        self.play_image_canvas.grid(row=11, column=0, columnspan=3, sticky="nsew")
        #create a button to generate image
        self.play_prompt_entry.bind("<Return>", lambda event: self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), self.play_resolution_slider_height.get(), self.play_resolution_slider_width.get(), self.play_cfg_slider.get(), self.play_steps_slider.get()))
        self.play_negative_prompt_entry.bind("<Return>", lambda event: self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), self.play_resolution_slider_height.get(), self.play_resolution_slider_width.get(), self.play_cfg_slider.get(), self.play_steps_slider.get()))
        
        #add convert to ckpt button
        self.play_convert_to_ckpt_button = tk.Button(self.play_tab, text="Convert To CKPT", command=lambda:self.convert_ckpt(model_path=self.play_model_entry.get()),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.play_convert_to_ckpt_button.configure(border=4, relief='flat')
        self.play_convert_to_ckpt_button.grid(row=9, column=1, columnspan=1, sticky="e")
        #add interative generation button to act as a toggle
        self.play_interactive_generation_button_bool = tk.BooleanVar()
        self.play_interactive_generation_button = tk.Button(self.play_tab, text="Interactive Generation", command=self.interactive_generation_button,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.play_interactive_generation_button.configure(border=4, relief='flat')
        self.play_interactive_generation_button.grid(row=9, column=1, columnspan=1, sticky="w")
        self.play_interactive_generation_button_bool.set(False)

        #add label to tools tab
        self.tools_label = tk.Label(self.tools_tab, text="Toolbox",  font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.tools_label.grid(row=0, column=0,columnspan=3, sticky="nsew")
        #empty row
        self.empty_row = tk.Label(self.tools_tab, text="",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.empty_row.grid(row=1, column=0, sticky="nsew")
        #add a label model tools title
        self.model_tools_label = tk.Label(self.tools_tab, text="Model Tools",  font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.model_tools_label.grid(row=2, column=0,columnspan=3, sticky="nsew")
        #add a button to convert to ckpt
        self.convert_to_ckpt_button = tk.Button(self.tools_tab, text="Convert Diffusers To CKPT", command=lambda:self.convert_ckpt(),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.convert_to_ckpt_button.configure(border=4, relief='flat')
        self.convert_to_ckpt_button.grid(row=3, column=0, columnspan=1, sticky="nsew")
        #add a button to convert ckpt to diffusers
        self.convert_ckpt_to_diffusers_button = tk.Button(self.tools_tab, text="Convert CKPT To Diffusers", command=lambda:self.convert_ckpt_to_diffusers(),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.convert_ckpt_to_diffusers_button.configure(border=4, relief='flat',state="disabled")
        self.convert_ckpt_to_diffusers_button.grid(row=3, column=1, columnspan=1, sticky="nsew")
        #empty row
        self.empty_row = tk.Label(self.tools_tab, text="",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.empty_row.grid(row=6, column=0, sticky="nsew")
        #add a label dataset tools title
        self.dataset_tools_label = tk.Label(self.tools_tab, text="Dataset Tools",  font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.dataset_tools_label.grid(row=7, column=0,columnspan=3, sticky="nsew")

        #add a button for Caption Buddy
        self.caption_buddy_button = tk.Button(self.tools_tab, text="Launch Caption Buddy",font=("Helvetica", 10, "bold"), command=lambda:self.caption_buddy(),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.caption_buddy_button.configure(border=4, relief='flat')
        self.caption_buddy_button.grid(row=8, column=0, columnspan=3, sticky="nsew")


        self.download_dataset_label = tk.Label(self.tools_tab, text="Download Dataset from HF",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        download_dataset_label_ttp = CreateToolTip(self.download_dataset_label, "Will git clone a HF dataset repo")
        self.download_dataset_label.grid(row=9, column=0, sticky="nsew")
        self.download_dataset_entry = tk.Entry(self.tools_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.download_dataset_entry.grid(row=9, column=1, sticky="nsew")
        #add download dataset button
        self.download_dataset_button = tk.Button(self.tools_tab, text="Download Dataset", command=self.download_dataset,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var)
        self.download_dataset_button.grid(row=9, column=2, sticky="nsew")
        
        self.all_entries_list = [self.diffusers_model_path_entry, self.seed_entry,self.play_seed_entry,self.play_model_entry,self.output_path_entry,self.play_prompt_entry,self.sample_width_entry,self.train_epochs_entry,self.learning_rate_entry,self.sample_height_entry,self.telegram_token_entry,self.vae_model_path_entry,self.dataset_repeats_entry,self.download_dataset_entry,self.num_warmup_steps_entry,self.download_dataset_entry,self.telegram_chat_id_entry,self.save_every_n_epochs_entry,self.play_negative_prompt_entry,self.number_of_class_images_entry,self.number_of_samples_to_generate_entry,self.prior_loss_preservation_weight_entry]
        for entry in self.all_entries_list:
            entry.bind("<Button-3>", self.create_right_click_menu)
        self.generate_btn = tk.Button(self.general_tab)
        
        self.generate_btn.configure(border=4, relief='flat',fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var, font=("Helvetica", 12, "bold"))
        self.generate_btn["text"] = "Start Training!"
        self.generate_btn["command"] = self.process_inputs
        self.generate_btn.grid(row=100, column=0,columnspan=2, sticky="nsew")
    def playground_find_latest_generated_model(self):
        last_output_path = self.output_path_entry.get()
        last_num_epochs = self.train_epochs_entry.get()
        last_model_path = last_output_path + os.sep + last_num_epochs
        #convert last_model_path seperators to the correct ones for the os
        last_model_path = last_model_path.replace("/", os.sep)
        last_model_path = last_model_path.replace("\\", os.sep)
        #check if the output path is valid
        if last_output_path != "":
            #check if the output path exists
            if os.path.exists(last_output_path):
                #check if the output path has a model in it
                if os.path.exists(last_model_path):
                    #check if the model is a ckpt
                    self.play_model_entry.delete(0, tk.END)
                    self.play_model_entry.insert(0, last_model_path)
                else:
                    #find the newest directory in the output path
                    
                    newest_dir = max(glob.iglob(last_output_path + os.sep + '*'), key=os.path.getctime)
                    #convert newest_dir seperators to the correct ones for the os
                    newest_dir = newest_dir.replace("/", os.sep)
                    newest_dir = newest_dir.replace("\\", os.sep)
                    last_model_path = newest_dir
                    self.play_model_entry.delete(0, tk.END)
                    self.play_model_entry.insert(0, last_model_path)
            else:
                return
        else:
            return
                    
    def caption_buddy(self):
        import scripts.captionBuddy
        cb_root = tk.Tk()
        cb_icon =PhotoImage(master=cb_root,file = "resources/stableTuner_icon.png")
        cb_root.iconphoto(False, cb_icon)
        app2 = scripts.captionBuddy.ImageBrowser(cb_root)
        cb_root.mainloop()
    def convert_ckpt_to_diffusers(self):
        #get the model path
        model_path = fd.askopenfilename(title="Select Model", filetypes=(("Model", "*.ckpt"), ("All Files", "*.*")))
        #get the output path
        output_path = fd.askdirectory()
        #run the command
        os.system("python3 /content/CLIP/convert_ckpt_to_diffusers.py --model_path " + model_path + " --output_path " + output_path)
    def disable_with_prior_loss(self, *args):
        if self.use_aspect_ratio_bucketing_var.get() == 1:
            self.with_prior_loss_preservation_var.set(0)
            self.with_prior_loss_preservation_checkbox.configure(state="disabled")

        else:
            self.with_prior_loss_preservation_checkbox.configure(state="normal")
    
    def download_dataset(self):
        #get the dataset name
        #import datasets
        from git import Repo
        folder = fd.askdirectory()
        dataset_name = self.download_dataset_entry.get()
        url = "https://huggingface.co/datasets/" + dataset_name if "/" not in dataset_name[0] else "/" + dataset_name
        Repo.clone_from(url, folder)
        
        #dataset = load_dataset(dataset_name)
        #for each item in the dataset save it to a file in a folder with the name of the dataset
        #create the folder
        #get user to pick a folder
        #git clone hugging face repo
                
        #using 
    def interactive_generation_button(self):
        #get state of button
        button_state = self.play_interactive_generation_button_bool.get()
        #flip the state of the button
        self.play_interactive_generation_button_bool.set(not button_state)
        #if the button is now true
        if self.play_interactive_generation_button_bool.get():
            #change the background of the button to green
            self.play_interactive_generation_button.configure(bg=self.dark_mode_title_var, fg="white")
        else:
            #change the background of the button to normal
            self.play_interactive_generation_button.configure(fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
    def play_save_image(self):


        file = fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")]) 
        #check if png in file name
        if ".png" not in file and file != "" and self.play_current_image:
            file = file + ".png"
        self.play_current_image.save(file)
    def play_generate_image(self, model, prompt, negative_prompt, seed, scheduler, sample_height, sample_width, cfg, steps):
        
        import diffusers
        import torch
        self.play_height = sample_height
        self.play_width = sample_width
        interactive = self.play_interactive_generation_button_bool.get()
        #update generate image button text
        if self.pipe is None or self.play_model_entry.get() != self.current_model:
            if self.pipe is not None:
                del self.pipe
                #clear torch cache
                torch.cuda.empty_cache()
            self.play_generate_image_button["text"] = "Loading Model, Please stand by..."
            self.play_generate_image_button.configure(fg="red")
            self.play_generate_image_button.update()
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(model,torch_dtype=torch.float16)
            self.pipe.to('cuda')
            self.current_model = model
        if scheduler == 'DPMSolverMultistepScheduler':
            scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == 'PNDMScheduler':
            scheduler = diffusers.PNDMScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == 'DDIMScheduler':
            scheduler = diffusers.DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == 'EulerDiscreteScheduler':
            scheduler = diffusers.EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = scheduler
        
        def displayInterImg(step: int, timestep: int, latents: torch.FloatTensor):
            #tensor to image
            img = self.pipe.decode_latents(latents)
            image = self.pipe.numpy_to_pil(img)[0]
            #convert to PIL image
            self.play_current_image = ImageTk.PhotoImage(image)
            if step == 0:
                self.play_image_canvas.configure(width=self.play_width, height=self.play_height)
                if self.play_width < self.master.winfo_width():
                    self.play_width = self.master.winfo_width()
                    self.master.geometry(f"{self.play_width}x{self.play_height+300}")
                self.play_image = self.play_image_canvas.create_image(0, 0, anchor="nw", image=self.play_current_image)
                self.play_image_canvas.update()
            #update image
            self.play_image_canvas.itemconfig(self.play_image, image=self.play_current_image)
            self.play_image_canvas.update()
        with torch.autocast("cuda"), torch.inference_mode():
            seed = int(seed)
            if seed == -1:
                #random seed
                import random
                seed = random.randint(0, 1000000)
            generator = torch.Generator("cuda").manual_seed(seed)
            self.play_generate_image_button["text"] = "Generating, Please stand by..."
            self.play_generate_image_button.configure(fg=self.dark_mode_title_var)
            self.play_generate_image_button.update()
            image = self.pipe(prompt=prompt,negative_prompt=negative_prompt,height=sample_height,width=sample_width, guidance_scale=cfg, num_inference_steps=steps,generator=generator,callback=displayInterImg if interactive else None,callback_steps=1).images[0]
            self.play_current_image = image
            #image is PIL image
            image = ImageTk.PhotoImage(image)
            self.play_image_canvas.configure(width=sample_width, height=sample_height)
            self.play_image_canvas.create_image(0, 0, anchor="nw", image=image)
            self.play_image_canvas.image = image
            #resize app to fit image, add current height to image height
            #if sample width is lower than current width, use current width
            if sample_width < self.master.winfo_width():
                sample_width = self.master.winfo_width()
            self.master.geometry(f"{sample_width}x{sample_height+330}")
            #refresh the window
            if self.play_save_image_button == None:
                self.play_save_image_button = tk.Button(self.play_tab, text="Save Image", command=self.play_save_image,fg=self.dark_mode_text_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
                self.play_save_image_button.configure(border=4, relief='flat')
                self.play_save_image_button.grid(row=10, column=2, columnspan=1, sticky="nsew")
            self.master.update()
            self.play_generate_image_button["text"] = "Generate Image"
            #normal text
            self.play_generate_image_button.configure(fg=self.dark_mode_text_var)
        
    def convert_ckpt(self,model_path=None, output_path=None):
        # Script for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
        # *Only* converts the UNet, VAE, and Text Encoder.
        # Does not convert optimizer state or any other thing.

        import argparse
        import os.path as osp

        import torch


        # =================#
        # UNet Conversion #
        # =================#

        unet_conversion_map = [
            # (stable-diffusion, HF Diffusers)
            ("time_embed.0.weight", "time_embedding.linear_1.weight"),
            ("time_embed.0.bias", "time_embedding.linear_1.bias"),
            ("time_embed.2.weight", "time_embedding.linear_2.weight"),
            ("time_embed.2.bias", "time_embedding.linear_2.bias"),
            ("input_blocks.0.0.weight", "conv_in.weight"),
            ("input_blocks.0.0.bias", "conv_in.bias"),
            ("out.0.weight", "conv_norm_out.weight"),
            ("out.0.bias", "conv_norm_out.bias"),
            ("out.2.weight", "conv_out.weight"),
            ("out.2.bias", "conv_out.bias"),
        ]

        unet_conversion_map_resnet = [
            # (stable-diffusion, HF Diffusers)
            ("in_layers.0", "norm1"),
            ("in_layers.2", "conv1"),
            ("out_layers.0", "norm2"),
            ("out_layers.3", "conv2"),
            ("emb_layers.1", "time_emb_proj"),
            ("skip_connection", "conv_shortcut"),
        ]

        unet_conversion_map_layer = []
        # hardcoded number of downblocks and resnets/attentions...
        # would need smarter logic for other networks.
        for i in range(4):
            # loop over downblocks/upblocks

            for j in range(2):
                # loop over resnets/attentions for downblocks
                hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
                sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
                unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

                if i < 3:
                    # no attention layers in down_blocks.3
                    hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                    sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                    unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

            for j in range(3):
                # loop over resnets/attentions for upblocks
                hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
                sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
                unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

                if i > 0:
                    # no attention layers in up_blocks.0
                    hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                    sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
                    unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

            if i < 3:
                # no downsample in down_blocks.3
                hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
                sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
                unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

                # no upsample in up_blocks.3
                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
                unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

        hf_mid_atn_prefix = "mid_block.attentions.0."
        sd_mid_atn_prefix = "middle_block.1."
        unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

        for j in range(2):
            hf_mid_res_prefix = f"mid_block.resnets.{j}."
            sd_mid_res_prefix = f"middle_block.{2*j}."
            unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


        def convert_unet_state_dict(unet_state_dict):
            # buyer beware: this is a *brittle* function,
            # and correct output requires that all of these pieces interact in
            # the exact order in which I have arranged them.
            mapping = {k: k for k in unet_state_dict.keys()}
            for sd_name, hf_name in unet_conversion_map:
                mapping[hf_name] = sd_name
            for k, v in mapping.items():
                if "resnets" in k:
                    for sd_part, hf_part in unet_conversion_map_resnet:
                        v = v.replace(hf_part, sd_part)
                    mapping[k] = v
            for k, v in mapping.items():
                for sd_part, hf_part in unet_conversion_map_layer:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
            new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
            return new_state_dict


        # ================#
        # VAE Conversion #
        # ================#

        vae_conversion_map = [
            # (stable-diffusion, HF Diffusers)
            ("nin_shortcut", "conv_shortcut"),
            ("norm_out", "conv_norm_out"),
            ("mid.attn_1.", "mid_block.attentions.0."),
        ]

        for i in range(4):
            # down_blocks have two resnets
            for j in range(2):
                hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
                sd_down_prefix = f"encoder.down.{i}.block.{j}."
                vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

            if i < 3:
                hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
                sd_downsample_prefix = f"down.{i}.downsample."
                vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                sd_upsample_prefix = f"up.{3-i}.upsample."
                vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

            # up_blocks have three resnets
            # also, up blocks in hf are numbered in reverse from sd
            for j in range(3):
                hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
                sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
                vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

        # this part accounts for mid blocks in both the encoder and the decoder
        for i in range(2):
            hf_mid_res_prefix = f"mid_block.resnets.{i}."
            sd_mid_res_prefix = f"mid.block_{i+1}."
            vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))


        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "query."),
            ("k.", "key."),
            ("v.", "value."),
            ("proj_out.", "proj_attn."),
        ]


        def reshape_weight_for_sd(w):
            # convert HF linear weights to SD conv2d weights
            return w.reshape(*w.shape, 1, 1)


        def convert_vae_state_dict(vae_state_dict):
            mapping = {k: k for k in vae_state_dict.keys()}
            for k, v in mapping.items():
                for sd_part, hf_part in vae_conversion_map:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
            for k, v in mapping.items():
                if "attentions" in k:
                    for sd_part, hf_part in vae_conversion_map_attn:
                        v = v.replace(hf_part, sd_part)
                    mapping[k] = v
            new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
            weights_to_convert = ["q", "k", "v", "proj_out"]
            for k, v in new_state_dict.items():
                for weight_name in weights_to_convert:
                    if f"mid.attn_1.{weight_name}.weight" in k:
                        print(f"Reshaping {k} for SD format")
                        new_state_dict[k] = reshape_weight_for_sd(v)
            return new_state_dict


        # =========================#
        # Text Encoder Conversion #
        # =========================#

        import re
        textenc_conversion_lst = [
            # (stable-diffusion, HF Diffusers)
            ('resblocks.','text_model.encoder.layers.'),
            ('ln_1','layer_norm1'),
            ('ln_2','layer_norm2'),
            ('.c_fc.','.fc1.'),
            ('.c_proj.','.fc2.'),
            ('.attn','.self_attn'),
            ('ln_final.','transformer.text_model.final_layer_norm.'),
            ('token_embedding.weight','transformer.text_model.embeddings.token_embedding.weight'),
            ('positional_embedding','transformer.text_model.embeddings.position_embedding.weight')
        ]
        protected = {re.escape(x[1]):x[0] for x in textenc_conversion_lst}
        textenc_pattern = re.compile("|".join(protected.keys()))

        # Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
        code2idx = {'q':0,'k':1,'v':2}

        def convert_text_enc_state_dict_v20(text_enc_dict:dict[str, torch.Tensor]):
            new_state_dict = {}
            capture_qkv_weight = {}
            capture_qkv_bias = {}
            for k,v in text_enc_dict.items():
                if k.endswith('.self_attn.q_proj.weight') or k.endswith('.self_attn.k_proj.weight') or k.endswith('.self_attn.v_proj.weight'):
                    k_pre = k[:-len('.q_proj.weight')]
                    k_code = k[-len('q_proj.weight')]
                    if k_pre not in capture_qkv_weight:
                        capture_qkv_weight[k_pre] = [None,None,None]
                    capture_qkv_weight[k_pre][code2idx[k_code]] = v
                    continue

                if k.endswith('.self_attn.q_proj.bias') or k.endswith('.self_attn.k_proj.bias') or k.endswith('.self_attn.v_proj.bias'):
                    k_pre = k[:-len('.q_proj.bias')]
                    k_code = k[-len('q_proj.bias')]
                    if k_pre not in capture_qkv_bias:
                        capture_qkv_bias[k_pre] = [None,None,None]
                    capture_qkv_bias[k_pre][code2idx[k_code]] = v
                    continue

                relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
                #        if relabelled_key != k:
                #            print(f"{k} -> {relabelled_key}")

                new_state_dict[relabelled_key] = v

            for k_pre,tensors in capture_qkv_weight.items():
                if None in tensors:
                    raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
                relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
                new_state_dict[relabelled_key+'.in_proj_weight'] = torch.cat(tensors)

            for k_pre,tensors in capture_qkv_bias.items():
                if None in tensors:
                    raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
                relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
                new_state_dict[relabelled_key+'.in_proj_bias'] = torch.cat(tensors)

            return new_state_dict


        def convert_text_enc_state_dict(text_enc_dict:dict[str, torch.Tensor]):
            return text_enc_dict
        
        #IS_V20_MODEL = True
        #model_path = self.diffusers_model_path_entry.get()
        if model_path is None:
            model_path = fd.askdirectory(initialdir=self.output_path_entry.get(), title="Select Diffusers Model Directory")
        #check if model path has vae,unet,text_encoder,tokenizer,scheduler and args.json and model_index.json
        if output_path is None:
            output_path = fd.asksaveasfilename(initialdir=os.getcwd(),title = "Save CKPT file",filetypes = (("ckpt files","*.ckpt"),("all files","*.*")))
        if not os.path.exists(model_path) and not os.path.exists(os.path.join(model_path,"vae")) and not os.path.exists(os.path.join(model_path,"unet")) and not os.path.exists(os.path.join(model_path,"text_encoder")) and not os.path.exists(os.path.join(model_path,"tokenizer")) and not os.path.exists(os.path.join(model_path,"scheduler")) and not os.path.exists(os.path.join(model_path,"args.json")) and not os.path.exists(os.path.join(model_path,"model_index.json")):
            messagebox.showerror("Error", "Couldn't find model in path")
            return
            #check if ckpt in output path
        if not output_path.endswith(".ckpt") and output_path != "":
            #add ckpt to output path
            output_path = output_path + ".ckpt"
        if not output_path or output_path == "":
            return
        assert model_path is not None, "Must provide a model path!"

        assert output_path is not None, "Must provide a checkpoint path!"
        #create a progress bar
        progress = 0
        #tk inter progress bar
        # load the model
        unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")
        vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.bin")
        text_enc_path = osp.join(model_path, "text_encoder", "pytorch_model.bin")

        # Convert the UNet model
        unet_state_dict = torch.load(unet_path, map_location="cpu")
        unet_state_dict = convert_unet_state_dict(unet_state_dict)
        unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

        # Convert the VAE model
        vae_state_dict = torch.load(vae_path, map_location="cpu")
        vae_state_dict = convert_vae_state_dict(vae_state_dict)
        vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

        # Convert the text encoder model
        text_enc_dict = torch.load(text_enc_path, map_location="cpu")

    # Easiest way to identify v2.0 model seems to be that the text encoder (OpenCLIP) is deeper
        is_v20_model = "text_model.encoder.layers.22.layer_norm2.bias" in text_enc_dict

        if is_v20_model:
            # Need to add the tag 'transformer' in advance so we can knock it out from the final layer-norm
            text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()} 
            text_enc_dict = convert_text_enc_state_dict_v20(text_enc_dict)
            text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
        else:
            text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
            text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}

        # Put together new checkpoint
        state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
        #if args.half:
        #    state_dict = {k: v.half() for k, v in state_dict.items()}
        state_dict = {"state_dict": state_dict}
        torch.save(state_dict, output_path)
        messagebox.showinfo("Conversion Complete", "Conversion Complete")


    def add_concept(self, inst_prompt_val=None, class_prompt_val=None, inst_data_path_val=None, class_data_path_val=None, do_not_balance_val=False):
        #create a title for the new concept
        concept_title = tk.Label(self.concepts_tab, text="Concept " + str(len(self.concept_labels)+1), font=("Helvetica", 10, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        concept_title.grid(row=3 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance prompt label
        ins_prompt_label = tk.Label(self.concepts_tab, text="Instance Prompt",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        ins_prompt_label_ttp = CreateToolTip(ins_prompt_label, "The token for the concept, will be ignored if use image names as captions is checked.")
        ins_prompt_label.grid(row=4 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance prompt entry
        ins_prompt_entry = tk.Entry(self.concepts_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        ins_prompt_entry.grid(row=4 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if inst_prompt_val != None:
            ins_prompt_entry.insert(0, inst_prompt_val)
        #create class prompt label
        class_prompt_label = tk.Label(self.concepts_tab, text="Class Prompt",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        class_prompt_label_ttp = CreateToolTip(class_prompt_label, "The prompt will be used to generate class images and train the class images if added to dataset")
        class_prompt_label.grid(row=5 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create class prompt entry
        class_prompt_entry = tk.Entry(self.concepts_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        class_prompt_entry.grid(row=5 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if class_prompt_val != None:
            class_prompt_entry.insert(0, class_prompt_val)
        #create instance data path label
        ins_data_path_label = tk.Label(self.concepts_tab, text="Instance Data Path",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        ins_data_path_label_ttp = CreateToolTip(ins_data_path_label, "The path to the folder containing the concept's images.")
        ins_data_path_label.grid(row=6 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance data path entry
        ins_data_path_entry = tk.Entry(self.concepts_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        ins_data_path_entry.grid(row=6 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if inst_data_path_val != None:
            ins_data_path_entry.insert(0, inst_data_path_val)
        #add a button to open a file dialog to select the instance data path
        ins_data_path_file_dialog_button = tk.Button(self.concepts_tab, text="...", command=lambda: self.open_file_dialog(ins_data_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        ins_data_path_file_dialog_button.configure(border=4, relief='flat')
        ins_data_path_file_dialog_button.grid(row=6 + (len(self.concept_labels)*6), column=2, sticky="nsew")
        #create class data path label
        class_data_path_label = tk.Label(self.concepts_tab, text="Class Data Path",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        class_data_path_label_ttp = CreateToolTip(class_data_path_label, "The path to the folder containing the concept's class images.")
        class_data_path_label.grid(row=7 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #add a button to open a file dialog to select the class data path
        class_data_path_file_dialog_button = tk.Button(self.concepts_tab, text="...", command=lambda: self.open_file_dialog(class_data_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        class_data_path_file_dialog_button.configure(border=4, relief='flat')
        class_data_path_file_dialog_button.grid(row=7 + (len(self.concept_labels)*6), column=2, sticky="nsew")
        #create class data path entry
        class_data_path_entry = tk.Entry(self.concepts_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        class_data_path_entry.grid(row=7 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if class_data_path_val != None:
            class_data_path_entry.insert(0, class_data_path_val)
        #add a checkbox to do not balance dataset
        do_not_balance_dataset_var = tk.IntVar()
        #label for checkbox
        do_not_balance_dataset_label = tk.Label(self.concepts_tab, text="Do not balance dataset",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        do_not_balance_dataset_label_ttp = CreateToolTip(do_not_balance_dataset_label, "If checked, the dataset will not be balanced. this settings overrides the global auto balance setting, if there's a concept you'd like to train without balance while the others will.")
        do_not_balance_dataset_label.grid(row=8 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        do_not_balance_dataset_checkbox = tk.Checkbutton(self.concepts_tab, variable=do_not_balance_dataset_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        do_not_balance_dataset_checkbox.grid(row=8 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        do_not_balance_dataset_var.set(0)
        if do_not_balance_val != False:
            do_not_balance_dataset_var.set(1)
        #combine all the entries into a list
        concept_entries = [ins_prompt_entry, class_prompt_entry, ins_data_path_entry, class_data_path_entry,do_not_balance_dataset_var,do_not_balance_dataset_checkbox]
        for i in concept_entries[:4]:
            i.bind("<Button-3>", self.create_right_click_menu)
        #add the list to the list of concept entries
        self.concept_entries.append(concept_entries)
        #add the title to the list of concept titles
        self.concept_labels.append([concept_title, ins_prompt_label, class_prompt_label, ins_data_path_label, class_data_path_label,do_not_balance_dataset_label])
        self.concepts.append({"instance_prompt": ins_prompt_entry, "class_prompt": class_prompt_entry, "instance_data_dir": ins_data_path_entry, "class_data_dir": class_data_path_entry,'do_not_balance': do_not_balance_dataset_var})
        self.concept_file_dialog_buttons.append([ins_data_path_file_dialog_button, class_data_path_file_dialog_button])
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def open_file_dialog(self, entry):
        """Opens a file dialog and sets the entry to the selected file."""
        file_path = fd.askdirectory()
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
    def save_concept_to_json(self,filename=None):
        #dialog box to select the file to save to
        if filename == None:
            file = fd.asksaveasfile(mode='w', defaultextension=".json", filetypes=[("JSON", "*.json")])
            #check if file has json extension
            if 'json' not in file.name:
                file.name = file.name + '.json'
        else:
            file = open(filename, 'w')
        if file != None:
            self.formatted_concepts = []
            concepts = self.concepts.copy()
            for i in range(len(concepts)):
                newDict = {}
                conceptDict = concepts[i]
                for key in conceptDict:
                    if isinstance(conceptDict[key], str) or isinstance(conceptDict[key], int) or isinstance(conceptDict[key], float) or isinstance(conceptDict[key], bool):
                        val = conceptDict[key]
                    else:
                        val = conceptDict[key].get()
                    newDict[key] = val
                self.formatted_concepts.append(newDict)
            concepts = None
            #write the json to the file
            #if the file is not none
            if file != None:
                #write the json to the file
                json.dump(self.formatted_concepts, file, indent=4)
                #close the file
                file.close()
    def load_concept_from_json(self):
        #
        #dialog
        concept_json = fd.askopenfilename(title = "Select file",filetypes = (("json files","*.json"),("all files","*.*")))
        for i in range(len(self.concept_entries)):
                self.remove_concept()
        self.concept_entries = []
        self.concept_labels = []
        self.concepts = []
        with open(concept_json, "r") as f:
            concept_json = json.load(f)
        for concept in concept_json:
            self.add_concept(inst_prompt_val=concept["instance_prompt"], class_prompt_val=concept["class_prompt"], inst_data_path_val=concept["instance_data_dir"], class_data_path_val=concept["class_data_dir"], do_not_balance_val=concept["do_not_balance"])
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.master.update()
        return concept_json
    def remove_concept(self):
        #remove the last concept
        if len(self.concept_labels) > 0:
            for entry in self.concept_entries[-1]:
                #if the entry is an intvar
                if isinstance(entry, tk.IntVar):
                    #delete the entry
                    del entry
                else:
                    entry.destroy()
            for label in self.concept_labels[-1]:
                label.destroy()
            for button in self.concept_file_dialog_buttons[-1]:
                button.destroy()
            self.concept_entries.pop()
            self.concept_labels.pop()
            self.concepts.pop()
            self.concept_file_dialog_buttons.pop()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def toggle_telegram_settings(self):
        #print(self.send_telegram_updates_var.get())
        if self.send_telegram_updates_var.get() == 1:
            self.telegram_token_label.config(state="normal")
            self.telegram_token_entry.config(state="normal")
            self.telegram_chat_id_label.config(state="normal")
            self.telegram_chat_id_entry.config(state="normal")
        else:
            self.telegram_token_label.config(state="disabled")
            self.telegram_token_entry.config(state="disabled")
            self.telegram_chat_id_label.config(state="disabled")
            self.telegram_chat_id_entry.config(state="disabled")
    def add_controlled_seed_sample(self,value=""):
        self.controlled_seed_sample_labels.append(tk.Label(self.sample_tab, text="Controlled Seed Sample " + str(len(self.controlled_seed_sample_labels)),fg=self.dark_mode_text_var, bg=self.dark_mode_var))
        self.controlled_seed_sample_labels[-1].grid(row=self.controlled_sample_row + len(self.sample_prompts) + len(self.controlled_seed_sample_labels), column=0, sticky="nsew")
        #create entry
        entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        entry.bind("<Button-3>",self.create_right_click_menu)
        self.controlled_seed_sample_entries.append(entry)
        self.controlled_seed_sample_entries[-1].grid(row=self.controlled_sample_row + len(self.sample_prompts) + len(self.controlled_seed_sample_entries), column=1, sticky="nsew")
        if value != "":
            self.controlled_seed_sample_entries[-1].insert(0, value)
        self.add_controlled_seed_to_sample.append(value)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def remove_controlled_seed_sample(self):
        #get the entry and label to remove
        if len(self.controlled_seed_sample_labels) > 0:
            self.controlled_seed_sample_labels[-1].destroy()
            self.controlled_seed_sample_labels.pop()
            self.controlled_seed_sample_entries[-1].destroy()
            self.controlled_seed_sample_entries.pop()
            self.add_controlled_seed_to_sample.pop()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
    def remove_sample_prompt(self):
        if len(self.sample_prompt_labels) > 0:
            #remove the last label and entry
            #get entry value
            self.sample_prompt_labels[-1].destroy()
            self.sample_prompt_entries[-1].destroy()
            #remove the last label and entry from the lists
            self.sample_prompt_labels.pop()
            self.sample_prompt_entries.pop()
            #remove the last value from the list
            self.sample_prompts.pop()
            #print(self.sample_prompts)
            #print(self.sample_prompt_entries)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def add_sample_prompt(self,value=""):
        #add a new label and entry
        self.sample_prompt_labels.append(tk.Label(self.sample_tab, text="Sample Prompt " + str(len(self.sample_prompt_labels)),fg=self.dark_mode_text_var, bg=self.dark_mode_var))
        self.sample_prompt_labels[-1].grid(row=self.sample_prompt_row + len(self.sample_prompt_labels) - 1, column=0, sticky="nsew")
        entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        entry.bind("<Button-3>", self.create_right_click_menu)
        self.sample_prompt_entries.append(entry)
        self.sample_prompt_entries[-1].grid(row=self.sample_prompt_row + len(self.sample_prompt_labels) - 1, column=1, sticky="nsew")
        
        if value != "":
            self.sample_prompt_entries[-1].insert(0, value)
        #update the sample prompts list
        self.sample_prompts.append(value)
        #print(self.sample_prompts)
        #print(self.sample_prompt_entries)
        #update canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        
    def update_sample_prompts(self):
        #update the sample prompts list
        self.sample_prompts = []
        for i in range(len(self.sample_prompt_entries)):
            self.sample_prompts.append(self.sample_prompt_entries[i].get())
    def update_controlled_seed_sample(self):
        #update the sample prompts list
        self.add_controlled_seed_to_sample = []
        for i in range(len(self.controlled_seed_sample_entries)):
            self.add_controlled_seed_to_sample.append(self.controlled_seed_sample_entries[i].get())
        
        self.master.update()
    def update_concepts(self):
        #update the concepts list
        #if the first index is a dict
        if isinstance(self.concepts, dict):
            return
        self.concepts = []
        for i in range(len(self.concept_entries)):
            self.concepts.append({"instance_prompt": self.concept_entries[i][0].get(), "class_prompt": self.concept_entries[i][1].get(), "instance_data_dir": self.concept_entries[i][2].get(), "class_data_dir": self.concept_entries[i][3].get(), "do_not_balance": self.concept_entries[i][4].get()})
    def save_config(self, config_file=None):
        #save the config file
        import json
        #create a dictionary of all the variables
        #ask the user for a file name
        if config_file == None:
            file_name = fd.asksaveasfilename(title = "Select file",filetypes = (("json files","*.json"),("all files","*.*")))
            #check if json in file name
            if ".json" not in file_name:
                file_name += ".json"
        else:
            file_name = config_file
        config = {}
        self.update_controlled_seed_sample()
        self.update_sample_prompts()
        self.update_concepts()
        config["concepts"] = self.concepts
        config["sample_prompts"] = self.sample_prompts
        config['add_controlled_seed_to_sample'] = self.add_controlled_seed_to_sample
        config["model_path"] = self.diffusers_model_path_entry.get()
        config["vae_path"] = self.vae_model_path_entry.get()
        config["output_path"] = self.output_path_entry.get()
        config["send_telegram_updates"] = self.send_telegram_updates_var.get()
        config["telegram_token"] = self.telegram_token_entry.get()
        config["telegram_chat_id"] = self.telegram_chat_id_entry.get()
        config["resolution"] = self.resolution_var.get()
        config["batch_size"] = self.train_batch_size_var.get()
        config["train_epocs"] = self.train_epochs_entry.get()
        config["mixed_precision"] = self.mixed_precision_var.get()
        config["use_8bit_adam"] = self.use_8bit_adam_var.get()
        config["use_gradient_checkpointing"] = self.use_gradient_checkpointing_var.get()
        config["accumulation_steps"] = self.gradient_accumulation_steps_var.get()
        config["learning_rate"] = self.learning_rate_entry.get()
        config["warmup_steps"] = self.num_warmup_steps_entry.get()
        config["learning_rate_scheduler"] = self.learning_rate_scheduler_var.get()
        config["use_latent_cache"] = self.use_latent_cache_var.get()
        config["save_latent_cache"] = self.save_latent_cache_var.get()
        config["regenerate_latent_cache"] = self.regenerate_latent_cache_var.get()
        config["train_text_encoder"] = self.train_text_encoder_var.get()
        config["with_prior_loss_preservation"] = self.with_prior_loss_preservation_var.get()
        config["prior_loss_preservation_weight"] = self.prior_loss_preservation_weight_entry.get()
        config["use_image_names_as_captions"] = self.use_image_names_as_captions_var.get()
        config["auto_balance_concept_datasets"] = self.auto_balance_dataset_var.get()
        config["add_class_images_to_dataset"] = self.add_class_images_to_dataset_var.get()
        config["number_of_class_images"] = self.number_of_class_images_entry.get()
        config["save_every_n_epochs"] = self.save_every_n_epochs_entry.get()
        config["number_of_samples_to_generate"] = self.number_of_samples_to_generate_entry.get()
        config["sample_height"] = self.sample_height_entry.get()
        config["sample_width"] = self.sample_width_entry.get()
        config["sample_random_aspect_ratio"] = self.sample_random_aspect_ratio_var.get()
        config['save_on_training_start'] = self.save_on_training_start_var.get()
        config['concepts'] = self.concepts
        config['aspect_ratio_bucketing'] = self.use_aspect_ratio_bucketing_var.get()
        config['seed'] = self.seed_entry.get()
        config['dataset_repeats'] = self.dataset_repeats_entry.get()
        config['limit_text_encoder_training'] = self.limit_text_encoder_entry.get()
        config['use_text_files_as_captions'] = self.use_text_files_as_captions_var.get()
        #save the config file
        #if the file exists, delete it
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, "w",encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    def load_config(self,file_name=None):
        #load the config file
        #ask the user for a file name
        if file_name == None:
            file_name = fd.askopenfilename(title = "Select file",filetypes = (("json files","*.json"),("all files","*.*")))
        if file_name == "":
            return
        #load the config file
        with open(file_name, "r",encoding='utf-8') as f:
            config = json.load(f)

        #load concepts
        try:
            for i in range(len(self.concept_entries)):
                self.remove_concept()
            self.concept_entries = []
            self.concept_labels = []
            self.concepts = []
            for i in range(len(config["concepts"])):
                self.add_concept(inst_prompt_val=config["concepts"][i]["instance_prompt"], class_prompt_val=config["concepts"][i]["class_prompt"], inst_data_path_val=config["concepts"][i]["instance_data_dir"], class_data_path_val=config["concepts"][i]["class_data_dir"],do_not_balance_val=config["concepts"][i]["do_not_balance"])
        except:
            pass
        
        #destroy all the current labels and entries
        for i in range(len(self.sample_prompt_labels)):
            self.sample_prompt_labels[i].destroy()
            self.sample_prompt_entries[i].destroy()

        for i in range(len(self.controlled_seed_sample_labels)):
            self.controlled_seed_sample_labels[i].destroy()
            self.controlled_seed_sample_entries[i].destroy()
        self.sample_prompt_labels = []
        self.sample_prompt_entries = []
        self.controlled_seed_sample_labels = []
        self.controlled_seed_sample_entries = []
        #set the variables
        for i in range(len(config["sample_prompts"])):
            self.add_sample_prompt(value=config["sample_prompts"][i])
        for i in range(len(config['add_controlled_seed_to_sample'])):
            self.add_controlled_seed_sample(value=config['add_controlled_seed_to_sample'][i])
            
        self.diffusers_model_path_entry.delete(0, tk.END)
        self.diffusers_model_path_entry.insert(0, config["model_path"])
        self.vae_model_path_entry.delete(0, tk.END)
        self.vae_model_path_entry.insert(0, config["vae_path"])
        self.output_path_entry.delete(0, tk.END)
        self.output_path_entry.insert(0, config["output_path"])
        self.send_telegram_updates_var.set(config["send_telegram_updates"])
        if config["send_telegram_updates"]:
            self.telegram_token_entry.config(state='normal')
            self.telegram_chat_id_entry.config(state='normal')
            self.telegram_token_label.config(state='normal')
            self.telegram_chat_id_label.config(state='normal')
        self.telegram_token_entry.delete(0, tk.END)
        self.telegram_token_entry.insert(0, config["telegram_token"])
        self.telegram_chat_id_entry.delete(0, tk.END)
        self.telegram_chat_id_entry.insert(0, config["telegram_chat_id"])
        self.resolution_var.set(config["resolution"])
        self.train_batch_size_var.set(config["batch_size"])
        self.train_epochs_entry.delete(0, tk.END)
        self.train_epochs_entry.insert(0, config["train_epocs"])
        self.mixed_precision_var.set(config["mixed_precision"])
        self.use_8bit_adam_var.set(config["use_8bit_adam"])
        self.use_gradient_checkpointing_var.set(config["use_gradient_checkpointing"])
        self.gradient_accumulation_steps_var.set(config["accumulation_steps"])
        self.learning_rate_entry.delete(0, tk.END)
        self.learning_rate_entry.insert(0, config["learning_rate"])
        self.num_warmup_steps_entry.delete(0, tk.END)
        self.num_warmup_steps_entry.insert(0, config["warmup_steps"])
        self.learning_rate_scheduler_var.set(config["learning_rate_scheduler"])
        self.use_latent_cache_var.set(config["use_latent_cache"])
        self.save_latent_cache_var.set(config["save_latent_cache"])
        self.regenerate_latent_cache_var.set(config["regenerate_latent_cache"])
        self.train_text_encoder_var.set(config["train_text_encoder"])
        self.with_prior_loss_preservation_var.set(config["with_prior_loss_preservation"])
        self.prior_loss_preservation_weight_entry.delete(0, tk.END)
        self.prior_loss_preservation_weight_entry.insert(0, config["prior_loss_preservation_weight"])
        self.use_image_names_as_captions_var.set(config["use_image_names_as_captions"])
        self.auto_balance_dataset_var.set(config["auto_balance_concept_datasets"])
        self.add_class_images_to_dataset_var.set(config["add_class_images_to_dataset"])
        self.number_of_class_images_entry.delete(0, tk.END)
        self.number_of_class_images_entry.insert(0, config["number_of_class_images"])
        self.save_every_n_epochs_entry.delete(0, tk.END)
        self.save_every_n_epochs_entry.insert(0, config["save_every_n_epochs"])
        self.number_of_samples_to_generate_entry.delete(0, tk.END)
        self.number_of_samples_to_generate_entry.insert(0, config["number_of_samples_to_generate"])
        self.sample_height_entry.delete(0, tk.END)
        self.sample_height_entry.insert(0, config["sample_height"])
        self.sample_width_entry.delete(0, tk.END)
        self.sample_width_entry.insert(0, config["sample_width"])
        self.sample_random_aspect_ratio_var.set(config["sample_random_aspect_ratio"])
        self.save_on_training_start_var.set(config["save_on_training_start"])
        #self.concept_config_path_entry.delete(0, tk.END)
        #self.concept_config_path_entry.insert(0,config["concept_config_path"])
        self.use_aspect_ratio_bucketing_var.set(config["aspect_ratio_bucketing"])
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, config["seed"])
        self.dataset_repeats_entry.delete(0, tk.END)
        self.dataset_repeats_entry.insert(0, config["dataset_repeats"])
        self.limit_text_encoder_entry.delete(0, tk.END)
        self.limit_text_encoder_entry.insert(0, config["limit_text_encoder_training"])
        self.use_text_files_as_captions_var.set(config["use_text_files_as_captions"])
        #self.update_controlled_seed_sample()
        #self.update_sample_prompts()
        self.master.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def process_inputs(self):
        #collect and process all the inputs
        self.update_controlled_seed_sample()
        self.update_sample_prompts()
        
        self.save_concept_to_json(filename='stabletune_concept_list.json')
        self.update_concepts()
        for i in range(len(self.sample_prompts)):
            self.sample_prompts.append(self.sample_prompts[i])
        for i in range(len(self.add_controlled_seed_to_sample)):
            self.add_controlled_seed_to_sample.append(self.add_controlled_seed_to_sample[i])
        self.model_path = self.diffusers_model_path_entry.get()
        self.vae_path = self.vae_model_path_entry.get()
        self.output_path = self.output_path_entry.get()
        self.send_telegram_updates = self.send_telegram_updates_var.get()
        self.telegram_token = self.telegram_token_entry.get()
        self.telegram_chat_id = self.telegram_chat_id_entry.get()
        self.resolution = self.resolution_var.get()
        self.batch_size = self.train_batch_size_var.get()
        self.train_epocs = self.train_epochs_entry.get()
        self.mixed_precision = self.mixed_precision_var.get()
        self.use_8bit_adam = self.use_8bit_adam_var.get()
        self.use_gradient_checkpointing = self.use_gradient_checkpointing_var.get()
        self.accumulation_steps = self.gradient_accumulation_steps_var.get()
        self.learning_rate = self.learning_rate_entry.get()
        self.warmup_steps = self.num_warmup_steps_entry.get()
        self.learning_rate_scheduler = self.learning_rate_scheduler_var.get()
        self.use_latent_cache = self.use_latent_cache_var.get()
        self.save_latent_cache = self.save_latent_cache_var.get()
        self.regenerate_latent_cache = self.regenerate_latent_cache_var.get()
        self.train_text_encoder = self.train_text_encoder_var.get()
        self.with_prior_loss_preservation = self.with_prior_loss_preservation_var.get()
        self.prior_loss_preservation_weight = self.prior_loss_preservation_weight_entry.get()
        self.use_image_names_as_captions = self.use_image_names_as_captions_var.get()
        self.auto_balance_concept_datasets = self.auto_balance_dataset_var.get()
        self.add_class_images_to_dataset = self.add_class_images_to_dataset_var.get()
        self.number_of_class_images = self.number_of_class_images_entry.get()
        self.save_every_n_epochs = self.save_every_n_epochs_entry.get()
        self.number_of_samples_to_generate = self.number_of_samples_to_generate_entry.get()
        self.sample_height = self.sample_height_entry.get()
        self.sample_width = self.sample_width_entry.get()
        self.sample_random_aspect_ratio = self.sample_random_aspect_ratio_var.get()
        self.save_on_training_start = self.save_on_training_start_var.get()
        self.concept_list_json_path = 'stabletune_concept_list.json'
        self.use_aspect_ratio_bucketing = self.use_aspect_ratio_bucketing_var.get()
        self.seed_number = self.seed_entry.get()
        self.dataset_repeats = self.dataset_repeats_entry.get()
        self.limit_text_encoder = self.limit_text_encoder_entry.get()
        self.use_text_files_as_captions = self.use_text_files_as_captions_var.get()
        #open stabletune_concept_list.json
        if os.path.exists('stabletune_last_run.json'):
            with open('stabletune_last_run.json') as f:
                self.last_run = json.load(f)
            if self.regenerate_latent_cache == False:
                if self.last_run["concepts"] == self.concepts:
                    #check if resolution is the same
                    try:
                        #try because I keep adding stuff to the json file and it may error out for peeps
                        if self.last_run["resolution"] != self.resolution or self.use_text_files_as_captions != self.last_run['use_text_files_as_captions'] or self.last_run['dataset_repeats'] != self.dataset_repeats or self.last_run["batch_size"] != self.batch_size or self.last_run["train_text_encoder"] != self.train_text_encoder or self.last_run["use_image_names_as_captions"] != self.use_image_names_as_captions or self.last_run["auto_balance_concept_datasets"] != self.auto_balance_concept_datasets or self.last_run["add_class_images_to_dataset"] != self.add_class_images_to_dataset or self.last_run["number_of_class_images"] != self.number_of_class_images or self.last_run["aspect_ratio_bucketing"] != self.use_aspect_ratio_bucketing:
                            self.regenerate_latent_cache = True
                            #show message
                            
                            messagebox.showinfo("StableTune", "Configuration changed, regenerating latent cache")
                    except:
                        print("Error trying to see if regenerating latent cache is needed, this means it probably needs to be regenerated and ST was updated recently.")
                        pass
                

        #create a bat file to run the training
        if self.mixed_precision == 'fp16':
            batBase = 'accelerate "launch" "--mixed_precision=fp16" "trainer.py"'
        else:
            batBase = 'accelerate "launch" "trainer.py"'
        
        if self.use_text_files_as_captions == True:
            batBase += ' "--use_text_files_as_captions" '

        if '%' in self.limit_text_encoder or self.limit_text_encoder != '0' and len(self.limit_text_encoder) > 0:
            #calculate the epoch number from the percentage and set the limit_text_encoder to the epoch number
            self.limit_text_encoder = int(self.limit_text_encoder.replace('%','')) * int(self.train_epocs) / 100
            #round the number to the nearest whole number
            self.limit_text_encoder = round(self.limit_text_encoder)
            batBase += f' "--stop_text_encoder_training={self.limit_text_encoder}" '
        batBase += f' "--pretrained_model_name_or_path={self.model_path}" '
        batBase += f' "--pretrained_vae_name_or_path={self.vae_path}" '
        batBase += f' "--output_dir={self.output_path}" '
        batBase += f' "--seed={self.seed_number}" '
        batBase += f' "--resolution={self.resolution}" '
        batBase += f' "--train_batch_size={self.batch_size}" '
        batBase += f' "--num_train_epochs={self.train_epocs}" '
        batBase += f' "--mixed_precision={self.mixed_precision}" '
        if self.use_aspect_ratio_bucketing:
            batBase += f' "--use_bucketing" '
        if self.use_8bit_adam == True:
            batBase += f' "--use_8bit_adam" '
        if self.use_gradient_checkpointing == True:
            batBase += f' "--gradient_checkpointing" '
        batBase += f' "--gradient_accumulation_steps={self.accumulation_steps}" '
        batBase += f' "--learning_rate={self.learning_rate}" '
        batBase += f' "--lr_warmup_steps={self.warmup_steps}" '
        batBase += f' "--lr_scheduler={self.learning_rate_scheduler}" '
        if self.use_latent_cache == False:
            batBase += f' "--not_cache_latents" '
        if self.save_latent_cache == True:
            batBase += f' "--save_latents_cache" '
        if self.regenerate_latent_cache == True:
            batBase += f' "--regenerate_latent_cache" '
        if self.train_text_encoder == True:
            batBase += f' "--train_text_encoder" '
        if self.with_prior_loss_preservation == True and self.use_aspect_ratio_bucketing == False:
            batBase += f' "--with_prior_preservation" '
            batBase += f' "--prior_loss_weight={self.prior_loss_preservation_weight}" '
        elif self.with_prior_loss_preservation == True and self.use_aspect_ratio_bucketing == True:
            print('loss preservation isnt supported with aspect ratio bucketing yet, sorry!')
        if self.use_image_names_as_captions == True:
            batBase += f' "--use_image_names_as_captions" '
        if self.auto_balance_concept_datasets == True:
            batBase += f' "--auto_balance_concept_datasets" '
        if self.add_class_images_to_dataset == True and self.with_prior_loss_preservation == False:
            batBase += f' "--add_class_images_to_dataset" '
        batBase += f' "--concepts_list={self.concept_list_json_path}" '
        batBase += f' "--num_class_images={self.number_of_class_images}" '
        batBase += f' "--save_every_n_epoch={self.save_every_n_epochs}" '
        batBase += f' "--n_save_sample={self.number_of_samples_to_generate}" '
        batBase += f' "--sample_height={self.sample_height}" '
        batBase += f' "--sample_width={self.sample_width}" '
        batBase += f' "--dataset_repeats={self.dataset_repeats}" '
        if self.sample_random_aspect_ratio == True:
            batBase += f' "--sample_aspect_ratios" '
        if self.send_telegram_updates == True:
            batBase += f' "--send_telegram_updates" '
            batBase += f' "--telegram_token={self.telegram_token}" '
            batBase += f' "--telegram_chat_id={self.telegram_chat_id}" '
        #remove duplicates from self.sample_prompts
        
        self.sample_prompts = list(dict.fromkeys(self.sample_prompts))
        #remove duplicates from self.add_controlled_seed_to_sample
        self.add_controlled_seed_to_sample = list(dict.fromkeys(self.add_controlled_seed_to_sample))
        for i in range(len(self.sample_prompts)):
            batBase += f' "--add_sample_prompt={self.sample_prompts[i]}" '
        for i in range(len(self.add_controlled_seed_to_sample)):
            batBase += f' "--save_sample_controlled_seed={self.add_controlled_seed_to_sample[i]}" '
        if self.save_on_training_start == True:
            batBase += f' "--save_on_training_start" '
        #save config
        self.save_config('stabletune_last_run.json')
        
        #save the bat file
        with open("train.bat", "w", encoding="utf-8") as f:
            f.write(batBase)
        #close the window
        self.destroy()
        self.master.destroy()
        #run the bat file
        self.master.quit()
        train = os.system("train.bat")
        #if exit code is 0, then the training was successful
        if train == 0:
            root = tk.Tk()
            app = App(master=root)
            #self.play_model_entry.insert(0, self.output_path_entry.get()+os.sep+self.train_epochs_entry.get())
            #switch to the play tab
            #self.notebook.select(5)
            #self.master.update()
            app.mainloop()

            
        else:
            os.system("pause")
        #restart the app
        


        
root = tk.Tk()
app = App(master=root)
app.mainloop()
