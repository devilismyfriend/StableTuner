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
import converters
import shutil
import ctypes as ct
#from scripts import converters
#work in progress code, not finished, credits will be added at a later date.

#class to make popup right click menu with select all, copy, paste, cut, and delete when right clicked on an entry box

#class to make a title bar for the window instead of the default one with the minimize, maximize, and close buttons

def dark_title_bar(window):
    """
    MORE INFO:
    https://docs.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
    """
    window.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    set_window_attribute = ct.windll.dwmapi.DwmSetWindowAttribute
    get_parent = ct.windll.user32.GetParent
    hwnd = get_parent(window.winfo_id())
    rendering_policy = DWMWA_USE_IMMERSIVE_DARK_MODE
    value = 2
    value = ct.c_int(value)
    set_window_attribute(hwnd, rendering_policy, ct.byref(value), ct.sizeof(value))
class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        #parent of the widget
        #hack to get the master of the app
        
        self.parent = widget.winfo_toplevel()
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
        self.tw.wm_attributes("-topmost", 1)
        self.parent.wm_attributes("-topmost", 0)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        #top most 
        
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
        #deiconify event
        #self.master.bind("<Map>", self.on_resume)
        #remove the default title bar
        self.master.overrideredirect(False)
        #force keep window on top
        self.master.wm_attributes("-topmost", 1)
        #create gui at center of screen
        self.master.geometry("1000x600+{}+{}".format(int(self.master.winfo_screenwidth()/2-1000/2), int(self.master.winfo_screenheight()/2-600/2)))
        #create a title bar
        #self.master.configure(bg="#1e2124")
        #define some colors
        self.stableTune_icon =PhotoImage(file = "resources/stableTuner_icon.png")
        self.master.iconphoto(False, self.stableTune_icon)
        self.dark_mode_var = "#202020"
        #self.dark_purple_mode_var = "#1B0F1B"
        self.dark_mode_title_var = "#286aff"
        self.dark_mode_button_pressed_var = "#BB91B6"
        self.dark_mode_button_var = "#8ea0e1"
        self.dark_mode_text_var = "#c6c7c8"
        self.master.title("StableTune")
        self.master.configure(cursor="left_ptr")
        #resizable window
        self.master.resizable(True, True)
        #master canvas
        self.canvas = tk.Canvas(self.master)
        self.canvas.configure(highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        #create a dark mode style for the scrollbar
        style = ttk.Style()
        style.theme_use('clam')
        #create new style
        style.element_create('Vertical.Scrollbar.trough', "from", 'default')
        style.configure("Vertical.TScrollbar", gripcount=0, background=self.dark_mode_title_var, darkcolor="#202020", lightcolor="#202020", troughcolor="#202020", bordercolor="#202020", arrowcolor=self.dark_mode_title_var)
        self.scrollbar = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview,style="Vertical.TScrollbar")
        #create dark mdoe style for vertical scrollbar
        self.scrollbar.pack(side="right", fill="y")
        #bind mousewheel to scroll
        self.canvas.bind_all("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.frame = tk.Frame(self.canvas)
        self.frame.configure(highlightthickness=0)
        self.frame.pack(side="top", fill="both", expand=True)
        self.configure(bg=self.dark_mode_var)
        
        self.canvas.create_window((0,0), window=self.frame, anchor="nw")
        self.canvas.configure(bg=self.dark_mode_var)
        #create tabs
        self.tabsSizes = {0 : [695,400], 1 : [695,560], 2 : [695,300],3 : [695,440],4 : [695,500],5 : [695,400],6 : [695,490]}
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
        self.notebook.add(self.concepts_tab, text="Training Data",sticky="nw")
        self.notebook.add(self.play_tab, text="Model Playground",sticky="n")
        self.notebook.add(self.tools_tab, text="Toolbox",sticky="n")
        #pad the frames to make them look better
        self.general_tab.configure(padx=10, pady=10)
        self.training_tab.configure(padx=10, pady=10)
        self.dataset_tab.configure(padx=10, pady=10)
        self.sample_tab.configure(padx=10, pady=10)
        self.concepts_tab.configure(padx=10, pady=10)
        self.play_tab.configure(padx=10, pady=10)
        self.tools_tab.configure(padx=10, pady=10)

        self.general_tab.configure(bg=self.dark_mode_var)
        self.training_tab.configure(bg=self.dark_mode_var)
        self.dataset_tab.configure(bg=self.dark_mode_var)
        self.sample_tab.configure(bg=self.dark_mode_var)
        self.concepts_tab.configure(bg=self.dark_mode_var)
        self.play_tab.configure(bg=self.dark_mode_var)
        self.tools_tab.configure(bg=self.dark_mode_var)
        #make a bottom frame
        self.bottom_frame = tk.Frame(self.canvas, bg=self.dark_mode_var)
        self.bottom_frame.pack(side="bottom", fill="x", expand=False)
        #configure grid
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        self.bottom_frame.columnconfigure(2, weight=1)
        #rowconfigure
        self.bottom_frame.rowconfigure(0, weight=1)
        #notebook dark mode style
        self.notebook_style = ttk.Style()
        self.notebook_style.theme_use("clam")
        #dark mode
        self.notebook_style.configure("TNotebook", background=self.dark_mode_var, borderwidth=0, highlightthickness=0, lightcolor=self.dark_mode_var, darkcolor=self.dark_mode_var, bordercolor=self.dark_mode_var, tabmargins=[0,0,0,0], padding=[0,0,0,0], relief="flat")
        self.notebook_style.configure("TNotebook.Tab", background=self.dark_mode_var, borderwidth=0, highlightthickness=0, lightcolor=self.dark_mode_var, foreground=self.dark_mode_text_var, bordercolor=self.dark_mode_var,highlightcolor=self.dark_mode_var, relief="flat")
        self.notebook_style.map("TNotebook.Tab", background=[("selected", self.dark_mode_var)], foreground=[("selected", self.dark_mode_title_var), ("active", self.dark_mode_title_var)])
        
        #on tab change resize window
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        #variables
        
        self.sample_prompts = []
        self.number_of_sample_prompts = len(self.sample_prompts)
        self.sample_prompt_labels = []
        self.input_model_path = ""
        self.vae_model_path = ""
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
        self.sample_on_training_start = False
        self.concept_template = {'instance_prompt': 'subject', 'class_prompt': 'a photo of class', 'instance_data_dir':'./data/subject','class_data_dir':'./data/subject_class'}
        self.concepts = []
        self.play_input_model_path = ""
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
        self.ckpt_sd_version = None
        self.convert_to_ckpt_after_training = False
        self.execute_post_conversion = False
        self.preview_images = []
        self.disable_cudnn_benchmark = True
        self.sample_step_interval = 500
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
                #convert to ckpt if option is wanted
                if self.execute_post_conversion == True:
                    #construct unique name
                    epoch = self.play_model_entry.get().split(os.sep)[-1]
                    name_of_model = self.play_model_entry.get().split(os.sep)[-2]
                    res = self.resolution_var.get()
                    #time and date
                    from datetime import datetime
                    #format time and date to %month%day%hour%minute
                    now = datetime.now()
                    dt_string = now.strftime("%m-%d-%H-%M")
                    #construct name
                    name = name_of_model+'_'+res+"_e"+epoch+"_"+dt_string
                    #print(self.play_model_entry.get())
                    self.convert_to_ckpt(model_path=self.play_model_entry.get(), output_path=self.output_path_entry.get(),name=name)
                    #open stabletune_last_run.json and change convert_to_ckpt_after_training to False
                    with open("stabletune_last_run.json", "r") as f:
                        data = json.load(f)
                    data["execute_post_conversion"] = False
                    with open("stabletune_last_run.json", "w") as f:
                        json.dump(data, f, indent=4)
            except Exception as e:
                print(e)
                pass
            #self.play_model_entry.insert(0, self.output_path_entry.get()+os.sep+self.train_epochs_entry.get())
        else:
            #self.load_config()
            pass
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def on_resume(self, event):
        #if state is deiconified, then window is restored
        
        self.master.update()
        self.master.overrideredirect(True)
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
    def on_tab_changed(self, event):
        #get the current selected notebook tab id
        tab_id = self.notebook.select()
        #get the tab object
        tab = self.notebook.nametowidget(tab_id)
        #get the tab index
        tab_index = self.notebook.index(tab_id)
        
        #get the tab size
        tab_size = self.tabsSizes[tab_index]
        #resize the window to fit the widgets
        self.master.geometry(f"{tab_size[0]}x{tab_size[1]}")
        #hide self.start_training_btn if we are on the playground or tools tab
        if tab_index == 5 or tab_index == 6:
            self.start_training_btn.grid_remove()
        else:
            self.start_training_btn.grid()
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
            #clear input_model_path_entry
            self.input_model_path_entry.delete(0, tk.END)
            if val == 'Stable Diffusion 1.4':
                self.input_model_path_entry.insert(0,"CompVis/stable-diffusion-v1-4")
            elif val == 'Stable Diffusion 1.5':
                self.input_model_path_entry.insert(0,"runwayml/stable-diffusion-v1-5")
            elif val == 'Stable Diffusion 2 Base (512)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-base")
            elif val == 'Stable Diffusion 2 (768)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2")
                self.resolution_var.set("768")
            elif val == 'Stable Diffusion 2.1 Base (512)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-1-base")
            elif val == 'Stable Diffusion 2.1 (768)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-1")
                self.resolution_var.set("768")
            self.master.update()
    def create_widgets(self):
        #create grid one side for labels the other for inputs
        #make the second column size 2x the first
        #create a model settings label in bold
        #add button to load config
        #add button to save config
        self.model_settings_label = tk.Label(self.general_tab, text="StableTune Settings",  font=("Arial", 12, "bold"), fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.model_settings_label.grid(row=0, column=0, sticky="nsew")
        self.load_config_button = tk.Button(self.general_tab, text="Load Config", command=self.load_config,fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.load_config_button.configure(border=4, relief='flat')
        self.load_config_button.grid(row=0, column=1, sticky="nw")
        self.save_config_button = tk.Button(self.general_tab, text="Save Config", command=self.save_config,fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        self.save_config_button.configure(border=4, relief='flat')
        self.save_config_button.grid(row=0, column=1, sticky="ne")

        #add tip label
        self.tip_label = tk.Label(self.general_tab, text="Tip: Hover over settings for information ;)",  font=("Arial", 10), fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.tip_label.grid(row=1, column=0,columnspan=3, sticky="nsew",pady=(0,10))
        self.quick_select_label = tk.Label(self.general_tab, text="Quick Select Model",  font=("Arial", 10, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        quick_select_label_ttp = CreateToolTip(self.quick_select_label, "Quick select another model to use.")
        self.quick_select_label.grid(row=2, column=0, sticky="nsew")
        self.quick_select_var = tk.StringVar()
        self.quick_select_var.set('Click to select')
        self.quick_select_dropdown = tk.OptionMenu(self.general_tab, self.quick_select_var, *self.quick_select_models, command=self.quick_select_model)
        self.quick_select_dropdown.config( activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, border=0, relief='flat', fg=self.dark_mode_title_var, bg=self.dark_mode_var, highlightthickness=2, highlightbackground=self.dark_mode_button_var)
        self.quick_select_dropdown["menu"].config( activebackground=self.dark_mode_var, activeforeground=self.dark_mode_title_var, bg=self.dark_mode_var, fg=self.dark_mode_text_var)
        self.quick_select_dropdown.grid(row=2, column=1, sticky="nsew")
        
        self.input_model_path_label = tk.Label(self.general_tab, text="Input Model / HuggingFace Repo",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        input_model_path_label_ttp = CreateToolTip(self.input_model_path_label, "The path to the diffusers model to use. Can be a local path or a HuggingFace repo path.")
        self.input_model_path_label.grid(row=3, column=0, sticky="nsew")
        self.input_model_path_entry = tk.Entry(self.general_tab,width=30,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        
        self.input_model_path_entry.grid(row=3, column=1, sticky="nsew")
        self.input_model_path_entry.insert(0, self.input_model_path)
        #make a button to open a file dialog
        self.input_model_path_button = tk.Button(self.general_tab, text="...", command=self.choose_model,fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.input_model_path_button.configure(border=4, relief='flat')
        self.input_model_path_button.grid(row=3, column=2, sticky="nwse")
        #create vae model path dark mode
        self.vae_model_path_label = tk.Label(self.general_tab, text="VAE model path / HuggingFace Repo",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        vae_model_path_label_ttp = CreateToolTip(self.vae_model_path_label, "OPTINAL The path to the VAE model to use. Can be a local path or a HuggingFace repo path.")
        self.vae_model_path_label.grid(row=4, column=0, sticky="nsew")
        self.vae_model_path_entry = tk.Entry(self.general_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.vae_model_path_entry.grid(row=4, column=1, sticky="nsew")
        self.vae_model_path_entry.insert(0, self.vae_model_path)
        #make a button to open a file dialog
        self.vae_model_path_button = tk.Button(self.general_tab, text="...", command=lambda: self.open_file_dialog(self.vae_model_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.vae_model_path_button.configure(border=4, relief='flat')
        self.vae_model_path_button.grid(row=4, column=2, sticky="nsew")
        #create output path dark mode
        self.output_path_label = tk.Label(self.general_tab, text="Output Path",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        output_path_label_ttp = CreateToolTip(self.output_path_label, "The path to the output directory. If it doesn't exist, it will be created.")
        self.output_path_label.grid(row=5, column=0, sticky="nsew")
        self.output_path_entry = tk.Entry(self.general_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.output_path_entry.grid(row=5, column=1, sticky="nsew")
        self.output_path_entry.insert(0, self.output_path)
        #make a button to open a file dialog
        self.output_path_button = tk.Button(self.general_tab, text="...", command=lambda: self.open_file_dialog(self.output_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        self.output_path_button.configure(border=4, relief='flat')
        self.output_path_button.grid(row=5, column=2, sticky="nsew")
        #create a checkbox wether to convert to ckpt after training
        self.convert_to_ckpt_after_training_label = tk.Label(self.general_tab, text="Convert to CKPT after training?",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        convert_to_ckpt_label_ttp = CreateToolTip(self.convert_to_ckpt_after_training_label, "Convert the model to a tensorflow checkpoint after training.")
        self.convert_to_ckpt_after_training_label.grid(row=6, column=0, sticky="nsew")
        self.convert_to_ckpt_after_training_var = tk.IntVar()
        self.convert_to_ckpt_after_training_checkbox = tk.Checkbutton(self.general_tab,variable=self.convert_to_ckpt_after_training_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.convert_to_ckpt_after_training_checkbox.grid(row=6, column=1, sticky="nsew")
        #use telegram updates dark mode
        self.send_telegram_updates_label = tk.Label(self.general_tab, text="Send Telegram Updates",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        send_telegram_updates_label_ttp = CreateToolTip(self.send_telegram_updates_label, "Use Telegram updates to monitor training progress, must have a Telegram bot set up.")
        self.send_telegram_updates_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox to toggle telegram updates and show telegram token and chat id
        self.send_telegram_updates_var = tk.IntVar()
        self.send_telegram_updates_checkbox = tk.Checkbutton(self.general_tab,variable=self.send_telegram_updates_var, command=self.toggle_telegram_settings,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.send_telegram_updates_checkbox.grid(row=7, column=1, sticky="nsew")
        #create telegram token dark mode
        self.telegram_token_label = tk.Label(self.general_tab, text="Telegram Token",  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        telegram_token_label_ttp = CreateToolTip(self.telegram_token_label, "The Telegram token for your bot.")
        self.telegram_token_label.grid(row=8, column=0, sticky="nsew")
        self.telegram_token_entry = tk.Entry(self.general_tab,  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var, disabledbackground=self.dark_mode_var,insertbackground="white")
        self.telegram_token_entry.grid(row=8, column=1, sticky="nsew")
        self.telegram_token_entry.insert(0, self.telegram_token)
        #create telegram chat id dark mode
        self.telegram_chat_id_label = tk.Label(self.general_tab, text="Telegram Chat ID",  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        telegram_chat_id_label_ttp = CreateToolTip(self.telegram_chat_id_label, "The Telegram chat ID to send updates to.")
        self.telegram_chat_id_label.grid(row=9, column=0, sticky="nsew")
        self.telegram_chat_id_entry = tk.Entry(self.general_tab,  state="disabled",fg=self.dark_mode_text_var, bg=self.dark_mode_var, disabledbackground=self.dark_mode_var,insertbackground="white")
        self.telegram_chat_id_entry.grid(row=9, column=1, sticky="nsew")
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
        
        #create checkbox disable cudnn benchmark
        self.disable_cudnn_benchmark_var = tk.IntVar()
        self.disable_cudnn_benchmark_var.set(self.disable_cudnn_benchmark)
        #create label for checkbox
        self.disable_cudnn_benchmark_label = tk.Label(self.training_tab, text="EXPERIMENTAL: Disable cuDNN Benchmark",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        disable_cudnn_benchmark_label_ttp = CreateToolTip(self.disable_cudnn_benchmark_label, "Disable cuDNN benchmarking, may offer 2x performance on some systems and stop OOM errors.")
        self.disable_cudnn_benchmark_label.grid(row=17, column=0, sticky="nsew")
        #create checkbox
        self.disable_cudnn_benchmark_checkbox = tk.Checkbutton(self.training_tab, variable=self.disable_cudnn_benchmark_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.disable_cudnn_benchmark_checkbox.grid(row=17, column=1, sticky="nsew")

        #create label
        #create with prior loss preservation checkbox
        self.with_prior_loss_preservation_var = tk.IntVar()
        self.with_prior_loss_preservation_var.set(self.with_prior_reservation)
        #create label
        self.with_prior_loss_preservation_label = tk.Label(self.training_tab, text="With Prior Loss Preservation",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        with_prior_loss_preservation_label_ttp = CreateToolTip(self.with_prior_loss_preservation_label, "Use the prior loss preservation method. part of Dreambooth.")
        self.with_prior_loss_preservation_label.grid(row=18, column=0, sticky="nsew")
        #create checkbox
        self.with_prior_loss_preservation_checkbox = tk.Checkbutton(self.training_tab, variable=self.with_prior_loss_preservation_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.with_prior_loss_preservation_checkbox.grid(row=18, column=1, sticky="nsew")
        #create prior loss preservation weight entry
        self.prior_loss_preservation_weight_label = tk.Label(self.training_tab, text="Weight",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        prior_loss_preservation_weight_label_ttp = CreateToolTip(self.prior_loss_preservation_weight_label, "The weight of the prior loss preservation loss.")
        self.prior_loss_preservation_weight_label.grid(row=18, column=1, sticky="e")
        self.prior_loss_preservation_weight_entry = tk.Entry(self.training_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.prior_loss_preservation_weight_entry.grid(row=18, column=3, sticky="w")
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
        #create sample every n steps entry
        self.sample_step_interval_label = tk.Label(self.sample_tab, text="Sample Every N Steps",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_step_interval_label_ttp = CreateToolTip(self.sample_step_interval_label, "Will sample the model every N steps.")
        self.sample_step_interval_label.grid(row=1, column=0, sticky="nsew")
        self.sample_step_interval_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.sample_step_interval_entry.grid(row=1, column=1, sticky="nsew")
        self.sample_step_interval_entry.insert(0, self.sample_step_interval)
        #create saver every n epochs entry
        self.save_every_n_epochs_label = tk.Label(self.sample_tab, text="Save and sample Every N Epochs",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        save_every_n_epochs_label_ttp = CreateToolTip(self.save_every_n_epochs_label, "Will save and sample the model every N epochs.")
        self.save_every_n_epochs_label.grid(row=2, column=0, sticky="nsew")
        self.save_every_n_epochs_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.save_every_n_epochs_entry.grid(row=2, column=1, sticky="nsew")
        self.save_every_n_epochs_entry.insert(0, self.save_and_sample_every_x_epochs)
        #create number of samples to generate entry
        self.number_of_samples_to_generate_label = tk.Label(self.sample_tab, text="Number of Samples to Generate",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        number_of_samples_to_generate_label_ttp = CreateToolTip(self.number_of_samples_to_generate_label, "The number of samples to generate per prompt.")
        self.number_of_samples_to_generate_label.grid(row=3, column=0, sticky="nsew")
        self.number_of_samples_to_generate_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.number_of_samples_to_generate_entry.grid(row=3, column=1, sticky="nsew")
        self.number_of_samples_to_generate_entry.insert(0, self.num_samples_to_generate)
        #create sample width entry
        self.sample_width_label = tk.Label(self.sample_tab, text="Sample Width",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_width_label_ttp = CreateToolTip(self.sample_width_label, "The width of the generated samples.")
        self.sample_width_label.grid(row=4, column=0, sticky="nsew")
        self.sample_width_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.sample_width_entry.grid(row=4, column=1, sticky="nsew")
        self.sample_width_entry.insert(0, self.sample_width)
        #create sample height entry
        self.sample_height_label = tk.Label(self.sample_tab, text="Sample Height",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_height_label_ttp = CreateToolTip(self.sample_height_label, "The height of the generated samples.")
        self.sample_height_label.grid(row=5, column=0, sticky="nsew")
        self.sample_height_entry = tk.Entry(self.sample_tab,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        self.sample_height_entry.grid(row=5, column=1, sticky="nsew")
        self.sample_height_entry.insert(0, self.sample_height)
        
        #create a checkbox to sample_on_training_start
        self.sample_on_training_start_var = tk.IntVar()
        self.sample_on_training_start_var.set(self.sample_on_training_start)
        #create label
        self.sample_on_training_start_label = tk.Label(self.sample_tab, text="Sample On Training Start",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_on_training_start_label_ttp = CreateToolTip(self.sample_on_training_start_label, "Will save and sample the model on training start, useful for debugging and comparison.")
        self.sample_on_training_start_label.grid(row=6, column=0, sticky="nsew")
        #create checkbox
        self.sample_on_training_start_checkbox = tk.Checkbutton(self.sample_tab, variable=self.sample_on_training_start_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.sample_on_training_start_checkbox.grid(row=6, column=1, sticky="nsew")
        #create sample random aspect ratio checkbox
        self.sample_random_aspect_ratio_var = tk.IntVar()
        self.sample_random_aspect_ratio_var.set(self.sample_random_aspect_ratio)
        #create label
        self.sample_random_aspect_ratio_label = tk.Label(self.sample_tab, text="Sample Random Aspect Ratio",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        sample_random_aspect_ratio_label_ttp = CreateToolTip(self.sample_random_aspect_ratio_label, "Will generate samples with random aspect ratios, useful to check aspect ratio bucketing.")
        self.sample_random_aspect_ratio_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox
        self.sample_random_aspect_ratio_checkbox = tk.Checkbutton(self.sample_tab, variable=self.sample_random_aspect_ratio_var,fg=self.dark_mode_text_var, bg=self.dark_mode_var, activebackground=self.dark_mode_var, activeforeground=self.dark_mode_text_var, selectcolor=self.dark_mode_var)
        self.sample_random_aspect_ratio_checkbox.grid(row=7, column=1, sticky="nsew")
        #create add sample prompt button
        self.add_sample_prompt_button = tk.Button(self.sample_tab, text="Add Sample Prompt",  command=self.add_sample_prompt,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.add_sample_prompt_button.configure(border=4, relief='flat')
        add_sample_prompt_button_ttp = CreateToolTip(self.add_sample_prompt_button, "Add a sample prompt to the list.")
        self.add_sample_prompt_button.grid(row=8, column=0, sticky="nsew")
        #create remove sample prompt button
        self.remove_sample_prompt_button = tk.Button(self.sample_tab, text="Remove Sample Prompt",  command=self.remove_sample_prompt,fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.remove_sample_prompt_button.configure(border=4, relief='flat')
        remove_sample_prompt_button_ttp = CreateToolTip(self.remove_sample_prompt_button, "Remove a sample prompt from the list.")
        self.remove_sample_prompt_button.grid(row=8, column=1, sticky="nsew")

        #for every prompt in self.sample_prompts, create a label and entry
        self.sample_prompt_labels = []
        self.sample_prompt_entries = []
        self.sample_prompt_row = 9
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
        self.concept_settings_label = tk.Label(self.concepts_tab, text="Training Data",  font=("Helvetica", 12, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        self.concept_settings_label.ttp = CreateToolTip(self.concept_settings_label, "This is where you put your training data!")
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
        self.play_model_entry.insert(0, self.play_input_model_path)
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
        self.play_convert_to_ckpt_button = tk.Button(self.play_tab, text="Convert To CKPT", command=lambda:self.convert_to_ckpt(model_path=self.play_model_entry.get()),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
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
        self.convert_to_ckpt_button = tk.Button(self.tools_tab, text="Convert Diffusers To CKPT", command=lambda:self.convert_to_ckpt(),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.convert_to_ckpt_button.configure(border=4, relief='flat')
        self.convert_to_ckpt_button.grid(row=3, column=0, columnspan=1, sticky="nsew")
        #add a button to convert ckpt to diffusers
        self.convert_ckpt_to_diffusers_button = tk.Button(self.tools_tab, text="Convert CKPT To Diffusers", command=lambda:self.convert_ckpt_to_diffusers(),fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
        self.convert_ckpt_to_diffusers_button.configure(border=4, relief='flat')
        self.convert_ckpt_to_diffusers_button.grid(row=3, column=2, columnspan=1, sticky="nsew")
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
        
        self.all_entries_list = [self.input_model_path_entry, self.seed_entry,self.play_seed_entry,self.play_model_entry,self.output_path_entry,self.play_prompt_entry,self.sample_width_entry,self.train_epochs_entry,self.learning_rate_entry,self.sample_height_entry,self.telegram_token_entry,self.vae_model_path_entry,self.dataset_repeats_entry,self.download_dataset_entry,self.num_warmup_steps_entry,self.download_dataset_entry,self.telegram_chat_id_entry,self.save_every_n_epochs_entry,self.play_negative_prompt_entry,self.number_of_class_images_entry,self.number_of_samples_to_generate_entry,self.prior_loss_preservation_weight_entry]
        for entry in self.all_entries_list:
            entry.bind("<Button-3>", self.create_right_click_menu)
        self.start_training_btn = tk.Button(self.bottom_frame)
        #self.start_training_btn.pack(side="bottom", fill="both")
        self.start_training_btn.configure(border=4, relief='flat',fg=self.dark_mode_title_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var, font=("Helvetica", 12, "bold"))
        self.start_training_btn["text"] = "Start Training!"
        self.start_training_btn["command"] = self.process_inputs
        self.start_training_btn.grid(row=0, column=1,columnspan=1, sticky="nsew")
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
        import captionBuddy
        self.master.overrideredirect(False)
        self.master.iconify()
        cb_root = tk.Tk()
        cb_icon =PhotoImage(master=cb_root,file = "resources/stableTuner_icon.png")
        cb_root.iconphoto(False, cb_icon)
        app2 = captionBuddy.ImageBrowser(cb_root,self.master)

        app = cb_root.mainloop()
        #check if app2 is running
        
        
        self.master.overrideredirect(True)
        self.master.deiconify()
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
            self.master.geometry(f"{sample_width}x{sample_height+self.tabsSizes[5][1]}")
            #refresh the window
            if self.play_save_image_button == None:
                self.play_save_image_button = tk.Button(self.play_tab, text="Save Image", command=self.play_save_image,fg=self.dark_mode_text_var, bg=self.dark_mode_var,activebackground=self.dark_mode_title_var)
                self.play_save_image_button.configure(border=4, relief='flat')
                self.play_save_image_button.grid(row=10, column=2, columnspan=1, sticky="nsew")
            self.master.update()
            self.play_generate_image_button["text"] = "Generate Image"
            #normal text
            self.play_generate_image_button.configure(fg=self.dark_mode_text_var)
    def convert_ckpt_to_diffusers(self,ckpt_path=None, output_path=None):
        if ckpt_path is None:
            ckpt_path = fd.askopenfilename(initialdir=os.getcwd(),title = "Select CKPT file",filetypes = (("ckpt files","*.ckpt"),("all files","*.*")))
        if output_path is None:
            #file dialog to save diffusers model
            output_path = fd.askdirectory(initialdir=os.getcwd(), title="Select where to save Diffusers Model Directory")
        version, prediction = self.get_sd_version(ckpt_path)
        self.convert_model_dialog = tk.Toplevel(self)
        self.convert_model_dialog.title("Converting model")
        #label
        empty_label = tk.Label(self.convert_model_dialog, text="")
        empty_label.pack()
        label = tk.Label(self.convert_model_dialog, text="Converting CKPT to Diffusers. Please wait...")
        label.pack()
        self.convert_model_dialog.geometry("300x70")
        self.convert_model_dialog.resizable(False, False)
        self.convert_model_dialog.grab_set()
        self.convert_model_dialog.focus_set()
        self.master.update()
        convert = converters.Convert_SD_to_Diffusers(ckpt_path,output_path,prediction_type=prediction,version=version)
        self.convert_model_dialog.destroy()

    def convert_to_ckpt(self,model_path=None, output_path=None,name=None):
        if model_path is None:
            model_path = fd.askdirectory(initialdir=self.output_path_entry.get(), title="Select Diffusers Model Directory")
        #check if model path has vae,unet,text_encoder,tokenizer,scheduler and args.json and model_index.json
        if output_path is None:
            output_path = fd.asksaveasfilename(initialdir=os.getcwd(),title = "Save CKPT file",filetypes = (("ckpt files","*.ckpt"),("all files","*.*")))
        if not os.path.exists(model_path) and not os.path.exists(os.path.join(model_path,"vae")) and not os.path.exists(os.path.join(model_path,"unet")) and not os.path.exists(os.path.join(model_path,"text_encoder")) and not os.path.exists(os.path.join(model_path,"tokenizer")) and not os.path.exists(os.path.join(model_path,"scheduler")) and not os.path.exists(os.path.join(model_path,"args.json")) and not os.path.exists(os.path.join(model_path,"model_index.json")):
            messagebox.showerror("Error", "Couldn't find model structure in path")
            return
            #check if ckpt in output path
        if name != None:
            output_path = os.path.join(output_path,name+".ckpt")
        if not output_path.endswith(".ckpt") and output_path != "":
            #add ckpt to output path
            output_path = output_path + ".ckpt"
        if not output_path or output_path == "":
            return

        self.convert_model_dialog = tk.Toplevel(self)
        self.convert_model_dialog.title("Converting model")
        #label
        empty_label = tk.Label(self.convert_model_dialog, text="")
        empty_label.pack()
        label = tk.Label(self.convert_model_dialog, text="Converting Diffusers to CKPT. Please wait...")
        label.pack()
        self.convert_model_dialog.geometry("300x70")
        self.convert_model_dialog.resizable(False, False)
        self.convert_model_dialog.grab_set()
        self.convert_model_dialog.focus_set()
        self.master.update()
        converters.Convert_Diffusers_to_SD(model_path, output_path)
        self.convert_model_dialog.destroy()
        #messagebox.showinfo("Conversion Complete", "Conversion Complete")
    
    #function to act as a callback when the user adds a new concept data path to generate a new preview image
    def update_preview_image(self, event):
        #check if entry has changed
        indexOfEntry = 0
        for concept_entry in self.concept_entries:
            if event.widget in concept_entry:
                indexOfEntry = self.concept_entries.index(concept_entry)
                #stop the loop
                break
        #get the path from the entry
        path = event.widget.get()
        canvas = self.preview_images[indexOfEntry][0]
        image_container = self.preview_images[indexOfEntry][1]
        icon = 'resources/stableTuner_icon.png'
        #create a photoimage object of the image in the path
        icon = Image.open(icon)
        #resize the image
        image = icon.resize((150, 150), Image.Resampling.LANCZOS)
        if path != "":
            if os.path.exists(path):
                files = os.listdir(path)
                for i in range(4):
                    #get an image from the path
                    import random
                    
                    #filter files for images
                    files = [f for f in files if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
                    if len(files) != 0:
                        rand = random.choice(files)
                        image_path = os.path.join(path,rand)
                        #remove image_path from files
                        if len(files) > 4:
                            files.remove(rand)
                        #files.pop(image_path)
                        #open the image
                        #print(image_path)
                        image_to_add = Image.open(image_path)
                        #resize the image to 38x38
                        #resize to 150x150 closest to the original aspect ratio
                        image_to_add.thumbnail((150, 150), Image.Resampling.LANCZOS)
                        #decide where to put the image
                        if i == 0:
                            #top left
                            image.paste(image_to_add, (0, 0))
                        elif i == 1:
                            #top right
                            image.paste(image_to_add, (76, 0))
                        elif i == 2:
                            #bottom left
                            image.paste(image_to_add, (0, 76))
                        elif i == 3:
                            #bottom right
                            image.paste(image_to_add, (76, 76))
                    #convert the image to a photoimage
                    #image.show()
        newImage=ImageTk.PhotoImage(image)
        self.preview_images[indexOfEntry][2] = newImage
        canvas.itemconfig(image_container, image=newImage)
        

    def add_concept(self, inst_prompt_val=None, class_prompt_val=None, inst_data_path_val=None, class_data_path_val=None, do_not_balance_val=False):
        #create a title for the new concept
        concept_title = tk.Label(self.concepts_tab, text="Concept " + str(len(self.concept_labels)+1), font=("Helvetica", 10, "bold"),fg=self.dark_mode_title_var, bg=self.dark_mode_var)
        concept_title.grid(row=3 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance prompt label
        ins_prompt_label = tk.Label(self.concepts_tab, text="Token/Prompt",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
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
        class_prompt_entry = tk.Entry(self.concepts_tab,width=50,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        class_prompt_entry.grid(row=5 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if class_prompt_val != None:
            class_prompt_entry.insert(0, class_prompt_val)
        #create instance data path label
        ins_data_path_label = tk.Label(self.concepts_tab, text="Training Data Directory",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        ins_data_path_label_ttp = CreateToolTip(ins_data_path_label, "The path to the folder containing the concept's images.")
        ins_data_path_label.grid(row=6 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance data path entry
        ins_data_path_entry = tk.Entry(self.concepts_tab,width=50,fg=self.dark_mode_text_var, bg=self.dark_mode_var,insertbackground="white")
        ins_data_path_entry.bind("<FocusOut>", self.update_preview_image)
        #bind to insert
        ins_data_path_entry.grid(row=6 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if inst_data_path_val != None:
            #focus on the entry
            
            ins_data_path_entry.insert(0, inst_data_path_val)
            ins_data_path_entry.focus_set()
            #focus on main window
            self.master.focus_set()
        #add a button to open a file dialog to select the instance data path
        ins_data_path_file_dialog_button = tk.Button(self.concepts_tab, text="...", command=lambda: self.open_file_dialog(ins_data_path_entry),fg=self.dark_mode_text_var, bg=self.dark_mode_title_var, activebackground=self.dark_mode_button_var, activeforeground="white")
        ins_data_path_file_dialog_button.configure(border=4, relief='flat')
        ins_data_path_file_dialog_button.grid(row=6 + (len(self.concept_labels)*6), column=2, sticky="nsew")
        #create class data path label
        class_data_path_label = tk.Label(self.concepts_tab, text="Class Data Directory",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
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

        #create a preview of the images in the path on the right side of the concept
        #create a frame to hold the images
        #empty column to separate the images from the rest of the concept
        
        sep = tk.Label(self.concepts_tab,padx=3, text="",fg=self.dark_mode_text_var, bg=self.dark_mode_var).grid(row=4 + (len(self.concept_labels)*6), column=3, sticky="nsew")

        image_preview_frame = tk.Frame(self.concepts_tab, bg=self.dark_mode_var)
        image_preview_frame.grid(row=4 + (len(self.concept_labels)*6), column=4, rowspan=4, sticky="ne")
        #create a label for the images
        #image_preview_label = tk.Label(image_preview_frame, text="Image Preview",fg=self.dark_mode_text_var, bg=self.dark_mode_var)
        #image_preview_label.grid(row=0, column=0, sticky="nsew")
        #create a canvas to hold the images
        image_preview_canvas = tk.Canvas(image_preview_frame, bg=self.dark_mode_var)
        #flat border
        image_preview_canvas.configure(border=0, relief='flat', highlightthickness=0)
        #canvas size is 100x100
        image_preview_canvas.config(width=150, height=150)
        image_preview_canvas.grid(row=0, column=0, sticky="nsew")
        #debug test, image preview just white
        #if there's a path in the entry, show the images in the path
        #grab stableTuner_icon.png from the resources folder
        icon = 'resources/stableTuner_icon.png'
        #create a photoimage object of the image in the path
        icon = Image.open(icon)
        #resize the image
        image = icon.resize((150, 150), Image.Resampling.LANCZOS)
        image_preview = ImageTk.PhotoImage(image, master=image_preview_frame)
        if inst_data_path_val != None:
            if os.path.exists(inst_data_path_val):
                del image_preview
                #get 4 images from the path
                #create a host image 
                image = Image.new("RGB", (150, 150), "white")
                files = os.listdir(inst_data_path_val)
                if len(files) > 0:
                    for i in range(4):
                        #get an image from the path
                        import random
                        
                        #filter files for images
                        files = [f for f in files if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
                        rand = random.choice(files)
                        image_path = os.path.join(inst_data_path_val,rand)
                        #remove image_path from files
                        if len(files) > 4:
                            files.remove(rand)
                        #files.pop(image_path)
                        #open the image
                        #print(image_path)
                        image_to_add = Image.open(image_path)
                        #resize the image to 38x38
                        #resize to 150x150 closest to the original aspect ratio
                        image_to_add.thumbnail((150, 150), Image.Resampling.LANCZOS)
                        #decide where to put the image
                        if i == 0:
                            #top left
                            image.paste(image_to_add, (0, 0))
                        elif i == 1:
                            #top right
                            image.paste(image_to_add, (76, 0))
                        elif i == 2:
                            #bottom left
                            image.paste(image_to_add, (0, 76))
                        elif i == 3:
                            #bottom right
                            image.paste(image_to_add, (76, 76))
                    #convert the image to a photoimage
                    #image.show()
                    image_preview = ImageTk.PhotoImage(image, master=image_preview_frame)
                    #add the image to the canvas

        
        image_container = image_preview_canvas.create_image(0, 0, anchor="nw", image=image_preview)
        self.preview_images.append([image_preview_canvas,image_container,image_preview])
        image_preview_frame.update()
        if do_not_balance_val != False:
            do_not_balance_dataset_var.set(1)
        #combine all the entries into a list
        concept_entries = [ins_prompt_entry, class_prompt_entry, ins_data_path_entry, class_data_path_entry,do_not_balance_dataset_var,do_not_balance_dataset_checkbox]
        for i in concept_entries[:4]:
            i.bind("<Button-3>", self.create_right_click_menu)
        #add the list to the list of concept entries
        self.concept_entries.append(concept_entries)
        #add the title to the list of concept titles
        self.concept_labels.append([concept_title, ins_prompt_label, class_prompt_label, ins_data_path_label, class_data_path_label,do_not_balance_dataset_label,image_preview_frame])
        self.concepts.append({"instance_prompt": ins_prompt_entry, "class_prompt": class_prompt_entry, "instance_data_dir": ins_data_path_entry, "class_data_dir": class_data_path_entry,'do_not_balance': do_not_balance_dataset_var})
        self.concept_file_dialog_buttons.append([ins_data_path_file_dialog_button, class_data_path_file_dialog_button])
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def get_sd_version(self,file_path):
            import torch
            checkpoint = torch.load(file_path)
            answer = messagebox.askyesno("V-Model?", "Is this model using V-Parameterization? (based on SD2.x 768 model)")
            if answer == True:
                prediction = "vprediction"
            else:
                prediction = "epsilon"
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            checkpoint = checkpoint["state_dict"]
            if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
                version = "v2"
            else:
                version = "v1"
            del checkpoint
            return version, prediction
    def choose_model(self):
        """Opens a file dialog and to choose either a model or a model folder."""
        #open file dialog and show only ckpt and json files and folders
        file_path = fd.askopenfilename(filetypes=[("Model", "*.ckpt"), ("Model", "*.json"), ("Model", "*.safetensors")])
        #file_path = fd.askopenfilename() model_index.json
        if file_path == "":
            return
        #check if the file is a json file
        if file_path.endswith("model_index.json"):
            #check if the file is a model index file
            #check if folder has folders for: vae, unet, tokenizer, text_encoder
            model_dir = os.path.dirname(file_path)
            required_folders = ["vae", "unet", "tokenizer", "text_encoder"]
            for folder in required_folders:
                if not os.path.isdir(os.path.join(model_dir, folder)):
                    #show error message
                    messagebox.showerror("Error", "The selected model is missing the {} folder.".format(folder))
                    return
                file_path = model_dir
            #if the file is not a model index file
        if file_path.endswith(".ckpt"):
            sd_file = file_path
            version, prediction = self.get_sd_version(sd_file)
            #create a directory under the models folder with the name of the ckpt file
            model_name = os.path.basename(file_path).split(".")[0]
            #get the path of the script
            script_path = os.getcwd()
            #get the path of the models folder
            models_path = os.path.join(script_path, "models")
            #if no models_path exists, create it
            if not os.path.isdir(models_path):
                os.mkdir(models_path)
            #create the path of the new model folder
            model_path = os.path.join(models_path, model_name)
            #check if the model folder already exists
            if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "model_index.json")):
                file_path = model_path
            else:
                #create the model folder
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
                os.mkdir(model_path)
                #converter
                #show a dialog to inform the user that the model is being converted
                self.convert_model_dialog = tk.Toplevel(self)
                self.convert_model_dialog.title("Converting model")
                #label
                empty_label = tk.Label(self.convert_model_dialog, text="")
                empty_label.pack()
                label = tk.Label(self.convert_model_dialog, text="Converting CKPT to Diffusers. Please wait...")
                label.pack()
                self.convert_model_dialog.geometry("300x70")
                self.convert_model_dialog.resizable(False, False)
                self.convert_model_dialog.grab_set()
                self.convert_model_dialog.focus_set()
                self.master.update()
                convert = converters.Convert_SD_to_Diffusers(sd_file,model_path,prediction_type=prediction,version=version)
                self.convert_model_dialog.destroy()

                file_path = model_path
        if file_path.endswith(".safetensors"):
            #raise not implemented error
            raise NotImplementedError("The selected file is a safetensors file. This file type is not supported yet.")
            file_path = ''
        self.input_model_path_entry.delete(0, tk.END)
        self.input_model_path_entry.insert(0, file_path)
    
    def open_file_dialog(self, entry):
        """Opens a file dialog and sets the entry to the selected file."""
        indexOfEntry = None
        file_path = fd.askdirectory()
        #get the entry name
        
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
        #focus on the entry
        entry.focus_set()
        #unset the focus on the button
        self.master.focus_set()

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
            self.preview_images.pop()
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
        #print(self.concepts)
        config["sample_prompts"] = self.sample_prompts
        config['add_controlled_seed_to_sample'] = self.add_controlled_seed_to_sample
        config["model_path"] = self.input_model_path_entry.get()
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
        config['sample_on_training_start'] = self.sample_on_training_start_var.get()
        config['concepts'] = self.concepts
        config['aspect_ratio_bucketing'] = self.use_aspect_ratio_bucketing_var.get()
        config['seed'] = self.seed_entry.get()
        config['dataset_repeats'] = self.dataset_repeats_entry.get()
        config['limit_text_encoder_training'] = self.limit_text_encoder_entry.get()
        config['use_text_files_as_captions'] = self.use_text_files_as_captions_var.get()
        config['ckpt_version'] = self.ckpt_sd_version
        config['convert_to_ckpt_after_training'] = self.convert_to_ckpt_after_training_var.get()
        config['execute_post_conversion'] = self.convert_to_ckpt_after_training_var.get()
        config['disable_cudnn_benchmark'] = self.disable_cudnn_benchmark_var.get()
        config['sample_step_interval'] = self.sample_step_interval_entry.get()
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
        except Exception as e:
            print(e)
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
            
        self.input_model_path_entry.delete(0, tk.END)
        self.input_model_path_entry.insert(0, config["model_path"])
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
        self.sample_on_training_start_var.set(config["sample_on_training_start"])
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
        self.convert_to_ckpt_after_training_var.set(config["convert_to_ckpt_after_training"])
        if config["execute_post_conversion"]:
            self.execute_post_conversion = True
        self.disable_cudnn_benchmark_var.set(config["disable_cudnn_benchmark"])
        self.sample_step_interval_entry.delete(0, tk.END)
        self.sample_step_interval_entry.insert(0, config["sample_step_interval"])

            

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
        self.model_path = self.input_model_path_entry.get()
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
        self.sample_on_training_start = self.sample_on_training_start_var.get()
        self.concept_list_json_path = 'stabletune_concept_list.json'
        self.use_aspect_ratio_bucketing = self.use_aspect_ratio_bucketing_var.get()
        self.seed_number = self.seed_entry.get()
        self.dataset_repeats = self.dataset_repeats_entry.get()
        self.limit_text_encoder = self.limit_text_encoder_entry.get()
        self.use_text_files_as_captions = self.use_text_files_as_captions_var.get()
        self.convert_to_ckpt_after_training = self.convert_to_ckpt_after_training_var.get()
        self.disable_cudnn_benchmark = self.disable_cudnn_benchmark_var.get()
        self.sample_step_interval = self.sample_step_interval_entry.get()
        if int(self.train_epocs) == 0 or self.train_epocs == '':
            messagebox.showerror("Error", "Number of training epochs must be greater than 0")
            return
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
        if self.mixed_precision == 'fp16' or self.mixed_precision == 'bf16':
            batBase = f'accelerate "launch" "--mixed_precision={self.mixed_precision}" "scripts/trainer.py"'
        else:
            batBase = 'accelerate "launch" "--mixed_precision=no" "scripts/trainer.py"'
        if self.disable_cudnn_benchmark == True:
            batBase += ' "--disable_cudnn_benchmark" '
        if self.use_text_files_as_captions == True:
            batBase += ' "--use_text_files_as_captions" '
        if self.sample_step_interval != '0' or self.sample_step_interval != '' or self.sample_step_interval != ' ':
            batBase += f' "--sample_step_interval={self.sample_step_interval}" '
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
        if self.mixed_precision == 'fp16' or self.mixed_precision == 'bf16':
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
        if self.sample_on_training_start == True:
            batBase += f' "--sample_on_training_start" '
        #save config
        self.save_config('stabletune_last_run.json')
        
        #save the bat file
        with open("scripts/train.bat", "w", encoding="utf-8") as f:
            f.write(batBase)
        #close the window
        self.destroy()
        self.master.destroy()
        #run the bat file
        self.master.quit()
        train = os.system(r".\scripts\train.bat")
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
            #cancel conversion on restart
            with open('stabletune_last_run.json', 'r') as f:
                data = json.load(f)
            data['execute_post_conversion'] = False
            with open('stabletune_last_run.json', 'w') as f:
                json.dump(data, f)
            os.system("pause")
        #restart the app
        


        
root = tk.Tk()
app = App(master=root)
dark_title_bar(root)
app.mainloop()
