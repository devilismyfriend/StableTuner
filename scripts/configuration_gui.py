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
from PIL import Image, ImageTk,ImageOps,ImageDraw
import glob
import converters
import shutil
from datetime import datetime
import pyperclip
import random
import customtkinter as ctk
import random
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
#work in progress code, not finished, credits will be added at a later date.

#class to make a generated image preview for the playground window, should open a new window alongside the playground window
class GeneratedImagePreview(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        #title
        self.title("Viewfinder")
        self.parent = parent
        self.configure(bg_color="transparent")
        #frame
        self.frame = ctk.CTkFrame(self, bg_color="transparent")
        self.frame.pack(fill="both", expand=True)
        #add tip label
        self.tip_label = ctk.CTkLabel(self.frame,text='Press the right arrow or enter to generate a new image', bg_color="transparent")
        self.tip_label.pack(fill="both", expand=True)
        #image
        self.image_preview_label = ctk.CTkLabel(self.frame,text='', bg_color="transparent")
        self.image_preview_label.pack(fill="both", expand=True)
        # run on close
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        #bind next image to right arrow
        self.bind("<Right>", lambda event: self.next_image())
        #bind to enter to generate a new image
        self.bind("<Return>", lambda event: self.next_image())
    def next_image(self, event=None):
        self.parent.generate_next_image()
    def on_close(self):
        self.parent.generation_window = None
        self.destroy()
    def ingest_image(self, image):
        self.geometry(f"{image.width + 50}x{image.height + 50}")
        self.image_preview_label.configure(image=ctk.CTkImage(image,size=(image.width,image.height)))
        #resize window
#class to make a concept top level window
class ConceptWidget(ctk.CTkFrame):
    #a widget that holds a concept and opens a concept window when clicked
    def __init__(self, parent, concept=None,width=150,height=150, *args, **kwargs):
        ctk.CTkFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.concept = concept
        #if concept is none, make a new concept
        if self.concept == None:
            self.default_image_preview = Image.open("resources/stableTuner_logo.png").resize((150, 150), Image.Resampling.LANCZOS)
            #self.default_image_preview = ImageTk.PhotoImage(self.default_image_preview)
            self.concept_name = "New Concept"
            self.concept_data_path = ""
            self.concept_class_name = ""
            self.concept_class_path = ""
            self.flip_p = ''
            self.concept_do_not_balance = False
            self.process_sub_dirs = False
            self.image_preview = self.default_image_preview
            #create concept
            self.concept = Concept(self.concept_name, self.concept_data_path, self.concept_class_name, self.concept_class_path,self.flip_p, self.concept_do_not_balance,self.process_sub_dirs, self.image_preview, None)
        else:
            self.concept = concept
            self.concept.image_preview = self.make_image_preview()
        
        self.width = width
        self.height = height
        self.configure(fg_color='transparent',border_width=0)
        self.concept_frame = ctk.CTkFrame(self, width=400, height=300,fg_color='transparent',border_width=0)
        self.concept_frame.grid_columnconfigure(0, weight=1)
        self.concept_frame.grid_rowconfigure(0, weight=1)
        self.concept_frame.grid(row=0, column=0, sticky="nsew")
        #concept image
        #if self.concept.image_preview is type(str):
        #    self.concept.image_preview = Image.open(self.concept.image_preview)
        self.concept_image_label = ctk.CTkLabel(self.concept_frame,text='',width=width,height=height, image=ctk.CTkImage(self.concept.image_preview,size=(100,100)))
        
        self.concept_image_label.grid(row=0, column=0, sticky="nsew")
        #ctk button with name as text and image as preview
        self.concept_button = ctk.CTkLabel(self.concept_frame, text=self.concept.concept_name,bg_color='transparent', compound="top")
        self.concept_button.grid(row=1, column=0, sticky="nsew")
        #bind the button to open a concept window
        self.concept_button.bind("<Button-1>", lambda event: self.open_concept_window())
        self.concept_image_label.bind("<Button-1>", lambda event: self.open_concept_window())
    def resize_widget(self,width,height):
        self.image_preview = self.image_preview.configure(size=(width,height))
        self.concept_image_label.configure(width=width,height=height,image=self.image_preview)
    def make_image_preview(self):
        def add_corners(im, rad):
            circle = Image.new('L', (rad * 2, rad * 2), 0)
            draw = ImageDraw.Draw(circle)
            draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
            alpha = Image.new('L', im.size, "white")
            w, h = im.size
            alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
            alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
            alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
            alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
            im.putalpha(alpha)
            return im
        path = self.concept.concept_path
        icon = 'resources/stableTuner_icon.png'
        #create a photoimage object of the image in the path
        icon = Image.open(icon)
        #resize the image
        image = icon.resize((150, 150), Image.Resampling.LANCZOS)
        if path != "" and path != None:
            if os.path.exists(path):
                files = []
                #if there are sub directories
                if self.concept.process_sub_dirs:
                    #get a list of all sub directories
                    sub_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
                    #if there are sub directories
                    if len(sub_dirs) != 0:
                        #collect all images in sub directories
                        for sub_dir in sub_dirs:
                            #collect the full path of all files in the sub directory to files
                            files += [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)]
                #if there are no sub directories
                else:
                    files = [os.path.join(path, f) for f in os.listdir(path)]
                    #omit sub directories
                    files = [f for f in files if not os.path.isdir(f)]
                if len(files) != 0:
                    for i in range(4):
                        #get an image from the path
                        import random
                        
                        #filter files for images
                        files = [f for f in files if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
                        if len(files) != 0:
                            rand = random.choice(files)
                            image_path = rand
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
                    image = add_corners(image, 30)
                        #convert the image to a photoimage
                        #image.show()
        newImage=ctk.CTkImage(image,size=(100,100))
        #print(image)
        self.image_preview = image
        return image
    def open_concept_window(self, event=None):
        #open a concept window
        self.concept_window = ConceptWindow(parent=self.parent, conceptWidget=self, concept=self.concept)
        self.concept_window.mainloop()
    
    def update_button(self):
        #update the button with the new concept name
        self.concept_button.configure(text=self.concept.concept_name)
        #update the preview image
        self.concept_image_label.configure(image=ctk.CTkImage(self.concept.image_preview,size=(100,100)))
    
    

        
class ConceptWindow(ctk.CTkToplevel):
    #init function
    def __init__(self, parent,conceptWidget,concept,*args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        #set title
        self.title("Concept Editor")
        self.parent = parent
        self.conceptWidget = conceptWidget
        self.concept = concept
        self.geometry("576x297")
        self.resizable(False, False)
        #self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.default_image_preview = Image.open("resources/stableTuner_icon.png").resize((150, 150), Image.Resampling.LANCZOS)
        #self.default_image_preview = ImageTk.PhotoImage(self.default_image_preview)
        
        #make a frame for the concept window
        self.concept_frame = ctk.CTkFrame(self, width=600, height=300)
        self.concept_frame.grid(row=0, column=0, sticky="nsew",padx=10,pady=10)
        self.concept_frame_subframe=ctk.CTkFrame(self.concept_frame, width=600, height=300)
        #4 column grid
        #self.concept_frame.grid_columnconfigure(0, weight=1)
        #self.concept_frame.grid_columnconfigure(1, weight=5)
        #self.concept_frame.grid_columnconfigure(2, weight=1)
        #self.concept_frame.grid_columnconfigure(3, weight=3)
        #make a label for concept name
        self.concept_name_label = ctk.CTkLabel(self.concept_frame_subframe, text="Dataset Token/Name:")
        self.concept_name_label.grid(row=0, column=0, sticky="nsew",padx=5,pady=5)
        #make a entry box for concept name
        self.concept_name_entry = ctk.CTkEntry(self.concept_frame_subframe,width=200)
        #create right click menu
        self.concept_name_entry.bind("<Button-3>", self.create_right_click_menu)
        self.concept_name_entry.grid(row=0, column=1, sticky="e",padx=5,pady=5)
        self.concept_name_entry.insert(0, self.concept.concept_name)
        #make a label for concept path
        self.concept_path_label = ctk.CTkLabel(self.concept_frame_subframe, text="Data Path:")
        self.concept_path_label.grid(row=1, column=0, sticky="nsew",padx=5,pady=5)
        #make a entry box for concept path
        self.concept_path_entry = ctk.CTkEntry(self.concept_frame_subframe,width=200)
        #create right click menu
        self.concept_path_entry.bind("<Button-3>", self.create_right_click_menu)
        self.concept_path_entry.grid(row=1, column=1, sticky="e",padx=5,pady=5)
        #on focus out, update the preview image
        self.concept_path_entry.bind("<FocusOut>", lambda event: self.update_preview_image(self.concept_path_entry))
        
        self.concept_path_entry.insert(0, self.concept.concept_path)
        #make a button to browse for concept path
        self.concept_path_button = ctk.CTkButton(self.concept_frame_subframe,width=30, text="...", command=lambda: self.browse_for_path(self.concept_path_entry))
        self.concept_path_button.grid(row=1, column=2, sticky="w",padx=5,pady=5)
        #make a label for Class Name
        self.class_name_label = ctk.CTkLabel(self.concept_frame_subframe, text="Class Name:")
        self.class_name_label.grid(row=2, column=0, sticky="nsew",padx=5,pady=5)
        #make a entry box for Class Name
        self.class_name_entry = ctk.CTkEntry(self.concept_frame_subframe,width=200)
        #create right click menu
        self.class_name_entry.bind("<Button-3>", self.create_right_click_menu)
        self.class_name_entry.grid(row=2, column=1, sticky="e",padx=5,pady=5)
        self.class_name_entry.insert(0, self.concept.concept_class_name)
        #make a label for Class Path
        self.class_path_label = ctk.CTkLabel(self.concept_frame_subframe, text="Class Path:")
        self.class_path_label.grid(row=3, column=0, sticky="nsew",padx=5,pady=5)
        #make a entry box for Class Path
        self.class_path_entry = ctk.CTkEntry(self.concept_frame_subframe,width=200)
        #create right click menu
        self.class_path_entry.bind("<Button-3>", self.create_right_click_menu)
        self.class_path_entry.grid(row=3, column=1, sticky="e",padx=5,pady=5)
        self.class_path_entry.insert(0, self.concept.concept_class_path)
        #make a button to browse for Class Path
        self.class_path_button = ctk.CTkButton(self.concept_frame_subframe,width=30, text="...", command=lambda: self.browse_for_path(entry_box=self.class_path_entry))
        self.class_path_button.grid(row=3, column=2, sticky="w",padx=5,pady=5)
        #entry and label for flip probability
        self.flip_probability_label = ctk.CTkLabel(self.concept_frame_subframe, text="Flip Probability:")
        self.flip_probability_label.grid(row=4, column=0, sticky="nsew",padx=5,pady=5)
        self.flip_probability_entry = ctk.CTkEntry(self.concept_frame_subframe,width=200,placeholder_text="0.0 - 1.0")
        self.flip_probability_entry.grid(row=4, column=1, sticky="e",padx=5,pady=5)
        if self.concept.flip_p != '':
            self.flip_probability_entry.insert(0, self.concept.flip_p)
        #self.flip_probability_entry.bind("<button-3>", self.create_right_click_menu)
        
        #make a label for dataset balancingprocess_sub_dirs
        self.balance_dataset_label = ctk.CTkLabel(self.concept_frame_subframe, text="Don't Balance Dataset")
        self.balance_dataset_label.grid(row=5, column=0, sticky="nsew",padx=5,pady=5)
        #make a switch to enable or disable dataset balancing
        self.balance_dataset_switch = ctk.CTkSwitch(self.concept_frame_subframe, text="", variable=tk.BooleanVar())
        self.balance_dataset_switch.grid(row=5, column=1, sticky="e",padx=5,pady=5)
        if self.concept.concept_do_not_balance == True:
            self.balance_dataset_switch.toggle()

        self.process_sub_dirs = ctk.CTkLabel(self.concept_frame_subframe, text="Search Sub-Directories")
        self.process_sub_dirs.grid(row=6, column=0, sticky="nsew",padx=5,pady=5)
        #make a switch to enable or disable dataset balancing
        self.process_sub_dirs_switch = ctk.CTkSwitch(self.concept_frame_subframe, text="", variable=tk.BooleanVar())
        self.process_sub_dirs_switch.grid(row=6, column=1, sticky="e",padx=5,pady=5)
        if self.concept.process_sub_dirs == True:
            self.process_sub_dirs_switch.toggle()
        #self.balance_dataset_switch.set(self.concept.concept_do_not_balance)
        #add image preview 
        self.image_preview_label = ctk.CTkLabel(self.concept_frame_subframe,text='', width=150, height=150,image=ctk.CTkImage(self.default_image_preview,size=(150,150)))
        self.image_preview_label.grid(row=0, column=4,rowspan=5, sticky="nsew",padx=5,pady=5)
        if self.concept.image_preview != None or self.concept.image_preview != "":
            #print(self.concept.image_preview)
            self.update_preview_image(entry=None,path=None,pil_image=self.concept.image_preview)
        elif self.concept.concept_data_path != "":
            self.update_preview_image(entry=None,path=self.concept_data_path)
        #self.image_container = self.image_preview_label.create_image(0, 0, anchor="nw", image=test_image)

        #make a save button
        self.save_button = ctk.CTkButton(self.concept_frame_subframe, text="Save", command=self.save)
        self.save_button.grid(row=6, column=3,columnspan=3,rowspan=1, sticky="nsew",padx=10,pady=10)

        #make a delete button
        #self.delete_button = ctk.CTkButton(self.concept_frame_subframe, text="Delete", command=self.delete)
        #self.delete_button.grid(row=6, column=3,columnspan=2, sticky="nsew")
        self.concept_frame_subframe.pack(fill="both", expand=True)
        #placeholder hack focus in and out of the entry box flip probability
        
    def create_right_click_menu(self, event):
        #create a menu
        self.menu = Menu(self.master, tearoff=0)
        self.menu.config(font=("Segoe UI", 15))

        #set dark colors for the menu
        self.menu.configure(bg="#2d2d2d", fg="#ffffff", activebackground="#2d2d2d", activeforeground="#ffffff")
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
    def delete(self):
        del self.concept
        self.conceptWidget.destroy()
        del self.conceptWidget
        self.destroy()
    #function to update image preview on change
    def update_preview_image(self, entry=None, path=None, pil_image=None):

        def add_corners(im, rad):
            circle = Image.new('L', (rad * 2, rad * 2), 0)
            draw = ImageDraw.Draw(circle)
            draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
            alpha = Image.new('L', im.size, "white")
            w, h = im.size
            alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
            alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
            alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
            alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
            im.putalpha(alpha)
            return im
        #check if entry has changed
        if entry != None and path == None :
            #get the path from the entry
            path = entry.get()
            
        #get the path from the entry
        #path = event.widget.get()
        #canvas = self.canvas
        #image_container = self.image_container

        icon = 'resources/stableTuner_icon.png'
        #create a photoimage object of the image in the path
        icon = Image.open(icon)
        #resize the image
        image = icon.resize((150, 150), Image.Resampling.LANCZOS)
        if path != "" and path != None:
            if os.path.exists(path):
                files = []
                #if there are sub directories in the path
                if self.concept.process_sub_dirs or self.process_sub_dirs_switch.get() == 1:
                    #get a list of all sub directories
                    sub_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
                    #if there are sub directories
                    if len(sub_dirs) != 0:
                        #collect all images in sub directories
                        for sub_dir in sub_dirs:
                            #collect the full path of all files in the sub directory to files
                            files += [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)]
                #if there are no sub directories
                else:
                    files = [os.path.join(path, f) for f in os.listdir(path)]
                    #omit sub directories
                    files = [f for f in files if not os.path.isdir(f)]
                if len(files) != 0:
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
                        add_corners(image, 30)
                        #convert the image to a photoimage
                        #image.show()
        if pil_image != None:
            image = pil_image
        #if image is of type PIL.Image.
        
        newImage=ctk.CTkImage(image,size=(150,150))
        self.image_preview = image
        
        self.image_preview_label.configure(image=newImage)

    #function to browse for concept path
    def browse_for_path(self,entry_box):
        #get the path from the user
        path = fd.askdirectory()
        #set the path to the entry box
        #delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, tk.END)
        entry_box.insert(0, path)
        self.focus_set()
    #save the concept
    def save(self):
        #get the concept name
        concept_name = self.concept_name_entry.get()
        #get the concept path
        concept_path = self.concept_path_entry.get()
        #get the class name
        class_name = self.class_name_entry.get()
        #get the class path
        class_path = self.class_path_entry.get()
        #get the flip probability
        flip_p = self.flip_probability_entry.get()
        #get the dataset balancing
        balance_dataset = self.balance_dataset_switch.get()
        #create the concept
        process_sub_dirs = self.process_sub_dirs_switch.get()
        #image preview
        image_preview = self.image_preview
        #get the main window
        image_preview_label = self.image_preview_label
        #update the concept
        self.concept.update(concept_name, concept_path, class_name, class_path,flip_p,balance_dataset,process_sub_dirs,image_preview,image_preview_label)
        self.conceptWidget.update_button()
        #close the window
        self.destroy()

#class of the concept
class Concept:
    def __init__(self, concept_name, concept_path, class_name, class_path,flip_p, balance_dataset=None,process_sub_dirs=None,image_preview=None, image_container=None):
        if concept_name == None:
            concept_name = ""
        if concept_path == None:
            concept_path = ""
        if class_name == None:
            class_name = ""
        if class_path == None:
            class_path = ""
        if flip_p == None:
            flip_p = ""
        if balance_dataset == None:
            balance_dataset = False
        if process_sub_dirs == None:
            process_sub_dirs = False
        if image_preview == None:
            image_preview = ""
        if image_container == None:
            image_container = ""
        

        self.concept_name = concept_name
        self.concept_path = concept_path
        self.concept_class_name = class_name
        self.concept_class_path = class_path
        self.flip_p = flip_p
        self.concept_do_not_balance = balance_dataset
        self.image_preview = image_preview
        self.image_container = image_container
        self.process_sub_dirs = process_sub_dirs
    #update the concept
    def update(self, concept_name, concept_path, class_name, class_path,flip_p,balance_dataset,process_sub_dirs, image_preview, image_container):
        self.concept_name = concept_name
        self.concept_path = concept_path
        self.concept_class_name = class_name
        self.concept_class_path = class_path
        self.flip_p = flip_p
        self.image_preview = image_preview
        self.image_container = image_container
        self.concept_do_not_balance = balance_dataset
        self.image_preview = image_preview
        self.image_container = image_container
        self.process_sub_dirs = process_sub_dirs
    #get the cocept details
    def get_details(self):
        return self.concept_name, self.concept_path, self.concept_class_name, self.concept_class_path,self.flip_p, self.concept_do_not_balance,self.process_sub_dirs, self.image_preview, self.image_container
#class to make popup right click menu with select all, copy, paste, cut, and delete when right clicked on an entry box
class DynamicGrid(ctk.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkFrame.__init__(self, parent, *args, **kwargs)
        self.text = tk.Text(self, wrap="char", borderwidth=0, highlightthickness=0,
                            state="disabled")
        self.text.pack(fill="both", expand=True)
        self.boxes = []

    def add_box(self, color=None):
        #bg = color if color else random.choice(("red", "orange", "green", "blue", "violet"))
        box = ctk.CTkFrame(self.text,width=100, height=100)
        #add a ctkbutton to the frame
        #ctk.CTkButton(box,text="test",command=lambda:print("test")).pack()
        #add a ctklabel to the frame
        ctk.CTkLabel(box,text="test").pack()
        #add a ctkentry to the frame
        ctk.CTkEntry(box).pack()
        #add a ctkcombobox to the frame
        #add a button remove the frame
        ctk.CTkButton(box,text="remove",command=lambda:self.remove_box(box)).pack()
        self.boxes.append(box)
        self.text.configure(state="normal")
        self.text.window_create("end", window=box)
        self.text.configure(state="disabled")
    def remove_box(self,box):
        self.boxes.remove(box)
        box.destroy()
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for box in self.boxes:
            self.text.window_create("end", window=box)
        self.text.configure(state="disabled")
#class to make a title bar for the window instead of the default one with the minimize, maximize, and close buttons
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        #self.pack(fill="both", expand=True)
        self.grid(row=0,column=0,sticky="nsew")
        s = ttk.Style()
        s.configure('new.TFrame', background='#242424',borderwidth=0,highlightthickness=0)
        self.configure(style='new.TFrame')
        self.canvas = tk.Canvas(self,bg='#242424')
        self.canvas.config(bg="#333333",highlightthickness=0,borderwidth=0,highlightbackground="#333333")
        self.scrollbar = ctk.CTkScrollbar(
            self, orientation="vertical", command=self.canvas.yview,bg_color="#333333",
            width=10, corner_radius=10)
        #s = ttk.Style()
        #s.configure('new.TFrame', background='#242424',borderwidth=0,highlightthickness=0)
        self.scrollable_frame = ttk.Frame(self.canvas,style='new.TFrame')
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        #set background color of the scrollable frame
        #self.scrollable_frame.config(background="#333333")
        self.scrollable_frame.bind("<Configure>",
            lambda *args, **kwargs: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        #resize the scrollable frame to the size of the window capped at 1000x1000
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(width=min(750, e.width), height=min(750, e.height)))
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind("<Destroy>",
            lambda *args, **kwargs: self.unbind_all("<MouseWheel>"))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * round(event.delta / 120), "units")

    
    def update_scroll_region(self):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
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
        x += self.widget.winfo_rootx() + 50
        y += self.widget.winfo_rooty() + 50
        # creates a toplevel window
        self.tw = ctk.CTkToplevel(self.widget)
        #self.tw.wm_attributes("-topmost", 1)
        #self.parent.wm_attributes("-topmost", 0)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        #top most 
        
        label = ctk.CTkLabel(self.tw, text=self.text, justify='left',
                       wraplength = self.wraplength)
        label.pack(padx=10, pady=10 )

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

class App(ctk.CTk):    
    def __init__(self):
        super().__init__()
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.geometry(f"{1100}x{585}")
        self.stableTune_icon =PhotoImage(master=self,file = "resources/stableTuner_icon.png")
        self.iconphoto(False, self.stableTune_icon)
        self.dark_mode_var = "#1e2124"
        self.dark_purple_mode_var = "#1B0F1B"
        self.dark_mode_title_var = "#7289da"
        self.dark_mode_button_pressed_var = "#BB91B6"
        self.dark_mode_button_var = "#8ea0e1"
        self.dark_mode_text_var = "#c6c7c8"
        self.title("StableTuner")
        self.configure(cursor="left_ptr")
        #resizable window
        self.resizable(True, True)
        self.create_default_variables()
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=10, sticky="nsew")
        self.logo_img = ctk.CTkImage(Image.open("resources/stableTuner_logo.png").resize((300, 300), Image.Resampling.LANCZOS),size=(80,80))
        self.logo_img = ctk.CTkLabel(self.sidebar_frame, image=self.logo_img, text='', height=50,width=50, font=ctk.CTkFont(size=15, weight="bold"))
        self.logo_img.grid(row=0, column=0, padx=20, pady=20)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="StableTuner", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.place(x=90, y=105, anchor="n")
        self.empty_label = ctk.CTkLabel(self.sidebar_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.empty_label.grid(row=1, column=0, padx=0, pady=0)
        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame,text='General Settings',command=self.general_nav_button_event)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=5)
        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame,text='Trainer Settings',command=self.training_nav_button_event)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=5)
        self.sidebar_button_3 = ctk.CTkButton(self.sidebar_frame,text='Dataset Settings',command=self.dataset_nav_button_event)
        self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=5)
        self.sidebar_button_4 = ctk.CTkButton(self.sidebar_frame,text='Sampling Settings',command=self.sampling_nav_button_event)
        self.sidebar_button_4.grid(row=5, column=0, padx=20, pady=5)
        self.sidebar_button_5 = ctk.CTkButton(self.sidebar_frame,text='Data',command=self.data_nav_button_event)
        self.sidebar_button_5.grid(row=6, column=0, padx=20, pady=5)
        self.sidebar_button_6 = ctk.CTkButton(self.sidebar_frame,text='Model Playground',command=self.playground_nav_button_event)
        self.sidebar_button_6.grid(row=7, column=0, padx=20, pady=5)
        self.sidebar_button_7 = ctk.CTkButton(self.sidebar_frame,text='Toolbox',command=self.toolbox_nav_button_event)
        self.sidebar_button_7.grid(row=8, column=0, padx=20, pady=5)
        #empty label
        self.empty_label = ctk.CTkLabel(self.sidebar_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.empty_label.grid(row=9, column=0, padx=0, pady=0)
        #empty label
        self.empty_label = ctk.CTkLabel(self.sidebar_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.empty_label.grid(row=10, column=0, padx=0, pady=0)
        #empty label
        self.empty_label = ctk.CTkLabel(self.sidebar_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.empty_label.grid(row=11, column=0, padx=0, pady=0)
        self.sidebar_button_11 = ctk.CTkButton(self.sidebar_frame,text='Caption Buddy',command=self.caption_buddy)
        self.sidebar_button_11.grid(row=13, column=0, padx=20, pady=5)
        self.sidebar_button_12 = ctk.CTkButton(self.sidebar_frame,text='Start Training!', command=lambda : self.process_inputs(export=False))
        self.sidebar_button_12.bind("<Button-3>", self.create_right_click_menu_export)
        self.sidebar_button_12.grid(row=14, column=0, padx=20, pady=5)
        self.general_frame = ctk.CTkFrame(self, width=140, corner_radius=0,fg_color='transparent')
        self.general_frame.grid_columnconfigure(0, weight=5)
        self.general_frame.grid_columnconfigure(1, weight=10)
        self.general_frame_subframe = ctk.CTkFrame(self.general_frame,width=300, corner_radius=20)
        self.general_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=20)
        self.general_frame_subframe_side_guide = ctk.CTkFrame(self.general_frame,width=250, corner_radius=20)
        self.general_frame_subframe_side_guide.grid(row=2, column=1,sticky="nsew", padx=20, pady=20)
        self.create_general_settings_widgets()   
        self.apply_general_style_to_widgets(self.general_frame_subframe)
        self.override_general_style_widgets()
        self.training_frame = ctk.CTkFrame(self, width=400, corner_radius=0,fg_color='transparent')
        self.training_frame.grid_columnconfigure(0, weight=1)
        self.training_frame_subframe = ctk.CTkFrame(self.training_frame,width=400, corner_radius=20)
        self.training_frame_subframe.grid_columnconfigure(0, weight=1)
        self.training_frame_subframe.grid_columnconfigure(1, weight=1)
        self.training_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=20)
        self.create_trainer_settings_widgets()
        self.grid_train_settings()
        self.apply_general_style_to_widgets(self.training_frame_subframe)
        self.override_training_style_widgets()
        self.dataset_frame = ctk.CTkFrame(self, width=140, corner_radius=0,fg_color='transparent')
        self.dataset_frame.grid_columnconfigure(0, weight=1)        
        self.dataset_frame_subframe = ctk.CTkFrame(self.dataset_frame,width=400, corner_radius=20)
        self.dataset_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=20)
        self.create_dataset_settings_widgets()
        self.apply_general_style_to_widgets(self.dataset_frame_subframe)
        self.sampling_frame = ctk.CTkFrame(self, width=140, corner_radius=0,fg_color='transparent')
        self.sampling_frame.grid_columnconfigure(0, weight=1)
        self.sampling_frame_subframe = ctk.CTkFrame(self.sampling_frame,width=400, corner_radius=20)
        self.sampling_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=20)
        self.create_sampling_settings_widgets()
        self.apply_general_style_to_widgets(self.sampling_frame_subframe)
        self.data_frame = ctk.CTkFrame(self, width=140, corner_radius=0,fg_color='transparent')
        self.data_frame.grid_columnconfigure(0, weight=1)
        self.data_frame_subframe = ctk.CTkFrame(self.data_frame,width=400, corner_radius=20)
        self.data_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=5) 
        self.create_data_settings_widgets()
        self.apply_general_style_to_widgets(self.data_frame_subframe)
        self.data_frame_concepts_subframe = ctk.CTkFrame(self.data_frame,width=400, corner_radius=20)
        self.data_frame_concepts_subframe.grid(row=3, column=0,sticky="nsew", padx=20, pady=5)        
        self.playground_frame = ctk.CTkFrame(self, width=140, corner_radius=0,fg_color='transparent')
        self.playground_frame.grid_columnconfigure(0, weight=1)
        self.playground_frame_subframe = ctk.CTkFrame(self.playground_frame,width=400, corner_radius=20)
        self.playground_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=20)
        self.playground_frame_subframe.grid_columnconfigure(0, weight=1)
        self.playground_frame_subframe.grid_columnconfigure(1, weight=3)
        self.playground_frame_subframe.grid_columnconfigure(2, weight=1)
        self.create_plyaground_widgets()
        self.apply_general_style_to_widgets(self.playground_frame_subframe)
        self.override_playground_widgets_style()
        self.toolbox_frame = ctk.CTkFrame(self, width=140, corner_radius=0,fg_color='transparent')
        self.toolbox_frame.grid_columnconfigure(0, weight=1)
        self.toolbox_frame_subframe = ctk.CTkFrame(self.toolbox_frame,width=400, corner_radius=20)
        self.toolbox_frame_subframe.grid(row=2, column=0,sticky="nsew", padx=20, pady=20)
        self.create_toolbox_widgets()
        self.apply_general_style_to_widgets(self.toolbox_frame_subframe)

        

        self.select_frame_by_name('general') 
        self.update()
        
        if os.path.exists("stabletune_last_run.json"):
            try:
                self.load_config(file_name="stabletune_last_run.json")
                #try loading the latest generated model to playground entry
                self.find_latest_generated_model(self.play_model_entry)
                #convert to ckpt if option is wanted
                if self.execute_post_conversion == True:
                    #construct unique name
                    epoch = self.play_model_entry.get().split(os.sep)[-1]
                    name_of_model = self.play_model_entry.get().split(os.sep)[-2]
                    res = self.resolution_var.get()
                    #time and date
                    #format time and date to %month%day%hour%minute
                    now = datetime.now()
                    dt_string = now.strftime("%m-%d-%H-%M")
                    #construct name
                    name = name_of_model+'_'+res+"_e"+epoch+"_"+dt_string
                    #print(self.play_model_entry.get())
                    #if self.play_model_entry.get() is a directory and all required folders exist
                    if os.path.isdir(self.play_model_entry.get()) and all([os.path.exists(os.path.join(self.play_model_entry.get(), folder)) for folder in self.required_folders]):
                        #print("all folders exist")
                        self.convert_to_ckpt(model_path=self.play_model_entry.get(), output_path=self.output_path_entry.get(),name=name)

                    #self.convert_to_ckpt(model_path=self.play_model_entry.get(), output_path=self.output_path_entry.get(),name=name)
                    #open stabletune_last_run.json and change convert_to_ckpt_after_training to False
                    with open("stabletune_last_run.json", "r") as f:
                        data = json.load(f)
                    data["execute_post_conversion"] = False
                    with open("stabletune_last_run.json", "w") as f:
                        json.dump(data, f, indent=4)
            except Exception as e:
                print(e)
                pass
        else:
            pass

    def create_default_variables(self):
        self.model_variant = 'Regular'
        self.model_variants = ['Regular', 'Inpaint']
        self.required_folders = ["vae", "unet", "tokenizer", "text_encoder"]
        self.aspect_ratio_bucketing_mode = 'Dynamic Fill'
        self.dynamic_bucketing_mode = 'Duplicate'
        self.play_keep_seed = False
        self.use_ema = False
        self.clip_penultimate = False
        self.conditional_dropout = ''
        self.cloud_toggle = False
        self.generation_window = None
        self.concept_widgets = []
        self.sample_prompts = []
        self.number_of_sample_prompts = len(self.sample_prompts)
        self.sample_prompt_labels = []
        self.input_model_path = "stabilityai/stable-diffusion-2-1-base"
        self.vae_model_path = ""
        self.output_path = "models/new_model"
        self.send_telegram_updates = False
        self.telegram_token = "TOKEN"
        self.telegram_chat_id = "ID"
        self.seed_number = 3434554
        self.resolution = 512
        self.batch_size = 24
        self.num_train_epochs = 100
        self.accumulation_steps = 1
        self.mixed_precision = "fp16"
        self.learning_rate = "3e-6"
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
        self.auto_balance_concept_datasets = False
        self.sample_width = 512
        self.sample_height = 512
        #self.save_latents_cache = True
        self.regenerate_latents_cache = False
        self.use_aspect_ratio_bucketing = True
        self.do_not_use_latents_cache = True
        self.with_prior_reservation = False
        self.prior_loss_weight = 1.0
        self.sample_random_aspect_ratio = False
        self.add_controlled_seed_to_sample = []
        self.sample_on_training_start = True
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
        self.quick_select_models = ["Stable Diffusion 1.4", "Stable Diffusion 1.5", "Stable Diffusion 1.5 Inpaint", "Stable Diffusion 2 Base (512)", "Stable Diffusion 2 (768)", 'Stable Diffusion 2 Inpaint', 'Stable Diffusion 2.1 Base (512)', "Stable Diffusion 2.1 (768)"]
        self.play_scheduler = 'DPMSolverMultistepScheduler'
        self.pipe = None
        self.current_model = None
        self.play_save_image_button = None
        self.dataset_repeats = 1
        self.limit_text_encoder = 0
        self.use_text_files_as_captions = True
        self.ckpt_sd_version = None
        self.convert_to_ckpt_after_training = False
        self.execute_post_conversion = False
        self.preview_images = []
        self.disable_cudnn_benchmark = True
        self.sample_step_interval = 500
    def select_frame_by_name(self, name):
        # set button color for selected button
        self.sidebar_button_1.configure(fg_color=("gray75", "gray25") if name == "general" else "transparent")
        self.sidebar_button_2.configure(fg_color=("gray75", "gray25") if name == "training" else "transparent")
        self.sidebar_button_3.configure(fg_color=("gray75", "gray25") if name == "dataset" else "transparent")
        self.sidebar_button_4.configure(fg_color=("gray75", "gray25") if name == "sampling" else "transparent")
        self.sidebar_button_5.configure(fg_color=("gray75", "gray25") if name == "data" else "transparent")
        self.sidebar_button_6.configure(fg_color=("gray75", "gray25") if name == "playground" else "transparent")
        self.sidebar_button_7.configure(fg_color=("gray75", "gray25") if name == "toolbox" else "transparent")


        # show selected frame
        if name == "general":
            self.general_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.general_frame.grid_forget()
        if name == "training":
            self.training_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.training_frame.grid_forget()
        if name == "dataset":
            self.dataset_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.dataset_frame.grid_forget()
        if name == "sampling":
            self.sampling_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.sampling_frame.grid_forget()
        if name == "data":
            self.data_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.data_frame.grid_forget()
        if name == "playground":
            self.playground_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.playground_frame.grid_forget()
        if name == "toolbox":
            self.toolbox_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.toolbox_frame.grid_forget()

    def general_nav_button_event(self):
        self.select_frame_by_name("general")

    def training_nav_button_event(self):
        self.select_frame_by_name("training")

    def dataset_nav_button_event(self):
        self.select_frame_by_name("dataset")
    def sampling_nav_button_event(self):
        self.select_frame_by_name("sampling")
    def data_nav_button_event(self):
        self.select_frame_by_name("data")
    def playground_nav_button_event(self):
        self.select_frame_by_name("playground")
    def toolbox_nav_button_event(self):
        self.select_frame_by_name("toolbox")

    #create a right click menu for entry widgets
    def create_right_click_menu(self, event):
        #create a menu
        self.menu = Menu(self.master, tearoff=0)
        self.menu.config(font=("Segoe UI", 15))

        #set dark colors for the menu
        self.menu.configure(bg="#2d2d2d", fg="#ffffff", activebackground="#2d2d2d", activeforeground="#ffffff")
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
    def create_right_click_menu_export(self, event):
        #create a menu
        self.menu = Menu(self.master, tearoff=0)
        #set menu size and font size
        self.menu.config(font=("Segoe UI", 15))

        #set dark colors for the menu
        self.menu.configure(bg="#2d2d2d", fg="#ffffff", activebackground="#2d2d2d", activeforeground="#ffffff")
        #add commands to the menu
        self.menu.add_command(label="Export Trainer Command for Windows", command=lambda: self.process_inputs(export='Win'))
        self.menu.add_command(label="Copy Trainer Command for Linux", command=lambda: self.process_inputs(export='LinuxCMD'))
        #display the menu
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            #make sure to release the grab (Tk 8.0a1 only)
            self.menu.grab_release()
    def create_left_click_menu_config(self, event):
        #create a menu
        self.menu = Menu(self.master, tearoff=0)
        #set menu size and font size
        self.menu.config(font=("Segoe UI", 15))

        #set dark colors for the menu
        self.menu.configure(bg="#2d2d2d", fg="#ffffff", activebackground="#2d2d2d", activeforeground="#ffffff")
        #add commands to the menu
        self.menu.add_command(label="Load Config", command=self.load_config)
        self.menu.add_command(label="Save Config", command=self.save_config)
        #display the menu
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            #make sure to release the grab (Tk 8.0a1 only)
            self.menu.grab_release()
    def quick_select_model(self,*args):
        val = self.quick_select_var.get()
        if val != "Click to select model":
            #clear input_model_path_entry
            self.input_model_path_entry.delete(0, tk.END)
            if val == 'Stable Diffusion 1.4':
                self.input_model_path_entry.insert(0,"CompVis/stable-diffusion-v1-4")
                self.model_variant_var.set("Regular")
            elif val == 'Stable Diffusion 1.5':
                self.input_model_path_entry.insert(0,"runwayml/stable-diffusion-v1-5")
                self.model_variant_var.set("Regular")
            elif val == 'Stable Diffusion 1.5 Inpaint':
                self.input_model_path_entry.insert(0,"runwayml/stable-diffusion-inpainting")
                self.model_variant_var.set("Inpaint")
            elif val == 'Stable Diffusion 2 Base (512)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-base")
                self.model_variant_var.set("Regular")
            elif val == 'Stable Diffusion 2 (768)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2")
                self.resolution_var.set("768")
                self.sample_height_entry.delete(0, tk.END)
                self.sample_height_entry.insert(0,"768")
                self.sample_width_entry.delete(0, tk.END)
                self.sample_width_entry.insert(0,"768")
                self.model_variant_var.set("Regular")
            elif val == 'Stable Diffusion 2 Inpaint':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-inpainting")
                self.model_variant_var.set("Inpaint")
            elif val == 'Stable Diffusion 2.1 Base (512)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-1-base")
                self.model_variant_var.set("Regular")
            elif val == 'Stable Diffusion 2.1 (768)':
                self.input_model_path_entry.insert(0,"stabilityai/stable-diffusion-2-1")
                self.resolution_var.set("768")
                self.sample_height_entry.delete(0, tk.END)
                self.sample_height_entry.insert(0,"768")
                self.sample_width_entry.delete(0, tk.END)
                self.sample_width_entry.insert(0,"768")
                self.model_variant_var.set("Regular")
    def override_training_style_widgets(self):
        for i in self.training_frame_subframe.children.values():
            if 'ctkbutton' in str(i):
                i.grid(padx=5, pady=5,sticky="w")
            if 'ctkoptionmenu' in str(i):
                i.grid(padx=10, pady=5,sticky="w")
            if 'ctkentry' in str(i):
                i.configure(width=160)
                i.grid(padx=10, pady=5,sticky="w")
                i.bind("<Button-3>", self.create_right_click_menu)
            if 'ctkswitch' in str(i):
                i.configure(text='')
                i.grid(padx=10, pady=5,sticky="")
            if 'ctklabel' in str(i):
                i.grid(padx=10, pady=5,sticky="w")

    def override_playground_widgets_style(self):
        self.playground_title.grid(row=0, column=0, padx=20, pady=20)  
        self.play_model_label.grid(row=0, column=0, sticky="nsew")
        self.play_model_entry.grid(row=0, column=1, sticky="nsew")
        self.play_prompt_label.grid(row=1, column=0, sticky="nsew")
        self.play_prompt_entry.grid(row=1, column=1,columnspan=2, sticky="nsew")
        self.play_negative_prompt_label.grid(row=2, column=0, sticky="nsew")
        self.play_negative_prompt_entry.grid(row=2, column=1,columnspan=2, sticky="nsew")
        self.play_seed_label.grid(row=3, column=0, sticky="nsew")
        self.play_seed_entry.grid(row=3, column=1, sticky="w")
        self.play_keep_seed_checkbox.grid(row=3, column=1)
        self.play_steps_label.grid(row=4, column=0, sticky="nsew")
        self.play_steps_slider.grid(row=4, column=1, sticky="nsew")
        self.play_scheduler_label.grid(row=5, column=0, sticky="nsew")
        self.play_scheduler_option_menu.grid(row=5, column=1, sticky="nsew")
        self.play_resolution_label.grid(row=6,rowspan=2, column=0, sticky="nsew")
        self.play_resolution_label_height.grid(row=6, column=1, sticky="w")
        self.play_resolution_label_width.grid(row=6, column=1, sticky="e")
        self.play_resolution_slider_height.grid(row=7, column=1, sticky="w")
        self.play_resolution_slider_width.grid(row=7, column=1, sticky="e")
        self.play_resolution_slider_height.set(self.play_sample_height)
        self.play_cfg_label.grid(row=8, column=0, sticky="nsew")
        self.play_cfg_slider.grid(row=8, column=1, sticky="nsew")
        self.play_toolbox_label.grid(row=9, column=0, sticky="nsew")
        self.play_generate_image_button.grid(row=10, column=0, columnspan=2, sticky="nsew")
        self.play_convert_to_ckpt_button.grid(row=9, column=1, columnspan=1, sticky="w")
    def override_general_style_widgets(self):
        pass
    def apply_general_style_to_widgets(self,frame):
        for i in frame.children.values():
            if 'ctkbutton' in str(i):
                i.grid(padx=5, pady=10,sticky="w")
            if 'ctkoptionmenu' in str(i):
                i.grid(padx=10, pady=10,sticky="w")
            if 'ctkentry' in str(i):
                i.configure(width=160)
                i.grid(padx=10, pady=5,sticky="w")
                i.bind("<Button-3>", self.create_right_click_menu)
            if 'ctkswitch' in str(i):
                i.configure(text='')
                i.grid(padx=10, pady=10,sticky="")
            if 'ctklabel' in str(i):
                i.grid(padx=10,sticky="w")

    def grid_train_settings(self):
        #define grid row and column
        self.training_frame_subframe.grid_columnconfigure(0, weight=2)
        self.training_frame_subframe.grid_columnconfigure(1, weight=1)
        self.training_frame_subframe.grid_columnconfigure(2, weight=2)
        self.training_frame_subframe.grid_columnconfigure(3, weight=1)
        
        rows = 12
        columns = 4
        widgets = self.training_frame_subframe.children.values()
        #organize widgets in grid
        curRow = 0
        curColumn = 0
        #make widgets a list
        widgets = list(widgets)[1:]
        #find ctkcanvas in widgets and remove it
        for i in widgets:
            if 'ctkcanvas' in str(i):
                widgets.remove(i)
        #create pairs of widgets
        pairs = []
        for i in range(0,len(widgets),2):
            pairs.append([widgets[i],widgets[i+1]])
        for p in pairs:
            p[0].grid(row=curRow, column=curColumn, sticky="w",padx=1,pady=1)
            p[1].grid(row=curRow, column=curColumn+1, sticky="w",padx=1,pady=1)
            curRow += 1
            if curRow == rows:
                curRow = 0
                curColumn += 2
    
    def dreambooth_mode(self):
        try:
            if self.dreambooth_mode_selected:
                self.dreambooth_mode_selected.destroy()
        except:
            pass
        try:
            if self.fine_tune_mode_selected:
                self.fine_tune_mode_selected.destroy()
                #re-enable previous disabled widgets
                self.with_prior_loss_preservation_checkbox.configure(state='normal')
                self.with_prior_loss_preservation_label.configure(state='normal')
                self.prior_loss_preservation_weight_entry.configure(state='normal')
                self.prior_loss_preservation_weight_label.configure(state='normal')
                self.with_prior_loss_preservation_var.set(1)
        except:
            pass
        self.dreambooth_mode_selected = ctk.CTkLabel(self.general_frame_subframe_side_guide,fg_color='transparent', text="Dreambooth it is!\n I disabled irrelevant features for you.", font=ctk.CTkFont(size=14))
        self.dreambooth_mode_selected.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        self.use_text_files_as_captions_checkbox.configure(state='disabled')
        self.use_text_files_as_captions_label.configure(state='disabled')
        self.use_text_files_as_captions_var.set(0)
        #self.use_text_files_as_captions_checkbox.set(0)
        self.use_image_names_as_captions_label.configure(state='disabled')
        self.use_image_names_as_captions_checkbox.configure(state='disabled')
        self.use_image_names_as_captions_var.set(0)
        #self.use_image_names_as_captions_checkbox.set(0)
        self.add_class_images_to_dataset_checkbox.configure(state='disabled')
        self.add_class_images_to_dataset_label.configure(state='disabled')
        self.add_class_images_to_dataset_var.set(0)
        #self.add_class_images_to_dataset_checkbox.set(0)
        pass
    def fine_tune_mode(self):
        try:
            if self.dreambooth_mode_selected:
                self.dreambooth_mode_selected.destroy()
                #re-enable checkboxes
                self.use_text_files_as_captions_checkbox.configure(state='normal')
                self.use_text_files_as_captions_label.configure(state='normal')
                self.use_image_names_as_captions_label.configure(state='normal')
                self.use_image_names_as_captions_checkbox.configure(state='normal')
                self.add_class_images_to_dataset_checkbox.configure(state='normal')
                self.add_class_images_to_dataset_label.configure(state='normal')
                self.use_text_files_as_captions_var.set(1)
                self.use_image_names_as_captions_var.set(1)
                self.add_class_images_to_dataset_var.set(0)
        except:
            pass
        try:
            if self.fine_tune_mode_selected:
                self.fine_tune_mode_selected.destroy()
        except:
            pass
        self.fine_tune_mode_selected = ctk.CTkLabel(self.general_frame_subframe_side_guide,fg_color='transparent', text="Let's Fine-Tune!\n I disabled irrelevant features for you.", font=ctk.CTkFont(size=14))
        self.fine_tune_mode_selected.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        self.with_prior_loss_preservation_checkbox.configure(state='disabled')
        self.with_prior_loss_preservation_label.configure(state='disabled')
        #self.with_prior_loss_preservation_checkbox.set(0)
        self.prior_loss_preservation_weight_label.configure(state='disabled')
        self.prior_loss_preservation_weight_entry.configure(state='disabled')
        self.with_prior_loss_preservation_var.set(0)

        #self.prior_loss_preservation_weight_entry.set(1.0)
        pass
    def lora_mode(self):
        self.lora_mode_selected = ctk.CTkLabel(self.general_frame_subframe_side_guide,fg_color='transparent', text="Lora it is!\n I disabled irrelevant features for you.", font=ctk.CTkFont(size=14))
        self.lora_mode_selected.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        pass
    def create_general_settings_widgets(self):


        self.general_frame_title = ctk.CTkLabel(self.general_frame, text="General Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.general_frame_title.grid(row=0, column=0,columnspan=2, padx=20, pady=20)    
        #self.tip_label = ctk.CTkLabel(self.general_frame, text="Tip: Hover over settings for information",  font=ctk.CTkFont(size=14))
        #self.tip_label.grid(row=1, column=0, sticky="nsew")

        self.general_frame_sidebar_title = ctk.CTkLabel(self.general_frame_subframe_side_guide,fg_color='transparent', text="Welcome!", font=ctk.CTkFont(size=20, weight="bold"))
        #self.general_frame_sidebar_title.grid(row=0, column=0, sticky="nsew")
        self.general_frame_sidebar_title.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        #text
        self.general_frame_sidebar_text = ctk.CTkLabel(self.general_frame_subframe_side_guide,fg_color='transparent', text="Welcome To StableTuner\nHow do you want to train today?", font=ctk.CTkFont(size=14))
        self.general_frame_sidebar_text.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        #add dreambooth button
        self.dreambooth_button = ctk.CTkButton(self.general_frame_subframe_side_guide, text="Dreambooth", command=self.dreambooth_mode)
        self.dreambooth_button.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        #add fine-tune button
        self.fine_tune_button = ctk.CTkButton(self.general_frame_subframe_side_guide, text="Fine-Tune", command=self.fine_tune_mode)
        self.fine_tune_button.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        #add LORA button with disabled state
        self.lora_button = ctk.CTkButton(self.general_frame_subframe_side_guide, text="LORA", command=self.lora_mode, state="disabled")
        self.lora_button.pack(side="top", fill="x", expand=False, padx=10, pady=10)
        self.quick_select_var = tk.StringVar(self.master)
        self.quick_select_var.set('Quick Select Base Model')
        self.quick_select_dropdown = ctk.CTkOptionMenu(self.general_frame_subframe, variable=self.quick_select_var, values=self.quick_select_models, command=self.quick_select_model,dynamic_resizing=False, width=200)
        self.quick_select_dropdown.grid(row=0, column=0, sticky="nsew")
        self.load_config_button = ctk.CTkButton(self.general_frame_subframe, text="Load/Save Config")
        #bind the load config button to a function
        self.load_config_button.bind("<Button-1>", lambda event: self.create_left_click_menu_config(event))
        self.load_config_button.grid(row=0, column=1, sticky="nsew")
        #create another button to resume from latest checkpoint
        self.input_model_path_resume_button = ctk.CTkButton(self.general_frame_subframe, text="Resume From Last Session",width=50, command=lambda : self.find_latest_generated_model(self.input_model_path_entry))
        self.input_model_path_resume_button.grid(row=0, column=2, sticky="nsew")
        self.input_model_path_label = ctk.CTkLabel(self.general_frame_subframe, text="Input Model / HuggingFace Repo")
        input_model_path_label_ttp = CreateToolTip(self.input_model_path_label, "The path to the diffusers model to use. Can be a local path or a HuggingFace repo path.")
        self.input_model_path_label.grid(row=1, column=0, sticky="nsew")
        self.input_model_path_entry = ctk.CTkEntry(self.general_frame_subframe,width=30)
        
        self.input_model_path_entry.grid(row=1, column=1, sticky="nsew")
        self.input_model_path_entry.insert(0, self.input_model_path)
        #make a button to open a file dialog
        self.input_model_path_button = ctk.CTkButton(self.general_frame_subframe,width=30, text="...", command=self.choose_model)
        self.input_model_path_button.grid(row=1, column=2, sticky="w")
        
        self.vae_model_path_label = ctk.CTkLabel(self.general_frame_subframe, text="VAE model path / HuggingFace Repo")
        vae_model_path_label_ttp = CreateToolTip(self.vae_model_path_label, "OPTINAL The path to the VAE model to use. Can be a local path or a HuggingFace repo path.")
        self.vae_model_path_label.grid(row=2, column=0, sticky="nsew")
        self.vae_model_path_entry = ctk.CTkEntry(self.general_frame_subframe)
        self.vae_model_path_entry.grid(row=2, column=1, sticky="nsew")
        self.vae_model_path_entry.insert(0, self.vae_model_path)
        #make a button to open a file dialog
        self.vae_model_path_button = ctk.CTkButton(self.general_frame_subframe,width=30, text="...", command=lambda: self.open_file_dialog(self.vae_model_path_entry))
        self.vae_model_path_button.grid(row=2, column=2, sticky="w")

        self.output_path_label = ctk.CTkLabel(self.general_frame_subframe, text="Output Path")
        output_path_label_ttp = CreateToolTip(self.output_path_label, "The path to the output directory. If it doesn't exist, it will be created.")
        self.output_path_label.grid(row=3, column=0, sticky="nsew")
        self.output_path_entry = ctk.CTkEntry(self.general_frame_subframe)
        self.output_path_entry.grid(row=3, column=1, sticky="nsew")
        self.output_path_entry.insert(0, self.output_path)
        #make a button to open a file dialog
        self.output_path_button = ctk.CTkButton(self.general_frame_subframe,width=30, text="...", command=lambda: self.open_file_dialog(self.output_path_entry))
        self.output_path_button.grid(row=3, column=2, sticky="w")

        self.convert_to_ckpt_after_training_label = ctk.CTkLabel(self.general_frame_subframe, text="Convert to CKPT after training?")
        convert_to_ckpt_label_ttp = CreateToolTip(self.convert_to_ckpt_after_training_label, "Convert the model to a tensorflow checkpoint after training.")
        self.convert_to_ckpt_after_training_label.grid(row=4, column=0, sticky="nsew")
        self.convert_to_ckpt_after_training_var = tk.IntVar()
        self.convert_to_ckpt_after_training_checkbox = ctk.CTkSwitch(self.general_frame_subframe,text='',variable=self.convert_to_ckpt_after_training_var)
        self.convert_to_ckpt_after_training_checkbox.grid(row=4, column=1, sticky="nsew",padx=10)
        
        #use telegram updates dark mode
        self.send_telegram_updates_label = ctk.CTkLabel(self.general_frame_subframe, text="Send Telegram Updates")
        send_telegram_updates_label_ttp = CreateToolTip(self.send_telegram_updates_label, "Use Telegram updates to monitor training progress, must have a Telegram bot set up.")
        self.send_telegram_updates_label.grid(row=6, column=0, sticky="nsew")
        #create checkbox to toggle telegram updates and show telegram token and chat id
        self.send_telegram_updates_var = tk.IntVar()
        self.send_telegram_updates_checkbox = ctk.CTkSwitch(self.general_frame_subframe,variable=self.send_telegram_updates_var, command=self.toggle_telegram_settings)
        self.send_telegram_updates_checkbox.grid(row=6, column=1, sticky="nsew")
        #create telegram token dark mode
        self.telegram_token_label = ctk.CTkLabel(self.general_frame_subframe, text="Telegram Token",  state="disabled")
        telegram_token_label_ttp = CreateToolTip(self.telegram_token_label, "The Telegram token for your bot.")
        self.telegram_token_label.grid(row=7, column=0, sticky="nsew")
        self.telegram_token_entry = ctk.CTkEntry(self.general_frame_subframe,  state="disabled")
        self.telegram_token_entry.grid(row=7, column=1,columnspan=3, sticky="nsew")
        self.telegram_token_entry.insert(0, self.telegram_token)
        #create telegram chat id dark mode
        self.telegram_chat_id_label = ctk.CTkLabel(self.general_frame_subframe, text="Telegram Chat ID",  state="disabled")
        telegram_chat_id_label_ttp = CreateToolTip(self.telegram_chat_id_label, "The Telegram chat ID to send updates to.")
        self.telegram_chat_id_label.grid(row=8, column=0, sticky="nsew")
        self.telegram_chat_id_entry = ctk.CTkEntry(self.general_frame_subframe,  state="disabled")
        self.telegram_chat_id_entry.grid(row=8, column=1,columnspan=3, sticky="nsew")
        self.telegram_chat_id_entry.insert(0, self.telegram_chat_id)
        
        #add a switch to toggle runpod mode
        self.cloud_mode_label = ctk.CTkLabel(self.general_frame_subframe, text="Cloud Training Export")
        cloud_mode_label_ttp = CreateToolTip(self.cloud_mode_label, "Cloud mode will package up a quick trainer session for RunPod/Colab etc.")
        self.cloud_mode_label.grid(row=9, column=0, sticky="nsew")
        self.cloud_mode_var = tk.IntVar()
        self.cloud_mode_checkbox = ctk.CTkSwitch(self.general_frame_subframe,variable=self.cloud_mode_var, command=self.toggle_runpod_mode)
        self.cloud_mode_checkbox.grid(row=9, column=1, sticky="nsew")
    
    def toggle_runpod_mode(self):
        toggle = self.cloud_mode_var.get()
        #flip self.toggle
        if toggle == True:
            toggle = False
            self.sidebar_button_12.configure(text='Export for Cloud!')
        else:
            toggle = True
            self.sidebar_button_12.configure(text='Start Training!')
        
    
    def create_trainer_settings_widgets(self):
        self.training_frame_title = ctk.CTkLabel(self.training_frame, text="Training Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.training_frame_title.grid(row=0, column=0, padx=20, pady=20)   
        
        #add a model variant dropdown
        self.model_variant_label = ctk.CTkLabel(self.training_frame_subframe, text="Model Variant")
        model_variant_label_ttp = CreateToolTip(self.model_variant_label, "The model type you're training.")
        self.model_variant_label.grid(row=0, column=0, sticky="nsew")
        self.model_variant_var = tk.StringVar()
        self.model_variant_var.set(self.model_variant)
        self.model_variant_dropdown = ctk.CTkOptionMenu(self.training_frame_subframe, values=self.model_variants, variable=self.model_variant_var)
    
        
        #add a seed entry
        self.seed_label = ctk.CTkLabel(self.training_frame_subframe, text="Seed")
        seed_label_ttp = CreateToolTip(self.seed_label, "The seed to use for training.")
        #self.seed_label.grid(row=1, column=0, sticky="nsew")
        self.seed_entry = ctk.CTkEntry(self.training_frame_subframe)
        #self.seed_entry.grid(row=1, column=1, sticky="nsew")
        self.seed_entry.insert(0, self.seed_number)
        #create resolution dark mode dropdown
        self.resolution_label = ctk.CTkLabel(self.training_frame_subframe, text="Resolution")
        resolution_label_ttp = CreateToolTip(self.resolution_label, "The resolution of the images to train on.")
        #self.resolution_label.grid(row=2, column=0, sticky="nsew")
        self.resolution_var = tk.StringVar()
        self.resolution_var.set(self.resolution)
        self.resolution_dropdown = ctk.CTkOptionMenu(self.training_frame_subframe, variable=self.resolution_var, values=["256", "320", "384", "448","512", "576", "640", "704", "768", "832", "896", "960", "1024"])
        #self.resolution_dropdown.grid(row=2, column=1, sticky="nsew")
        
        #create train batch size dark mode dropdown with values from 1 to 60
        self.train_batch_size_label = ctk.CTkLabel(self.training_frame_subframe, text="Train Batch Size")
        train_batch_size_label_ttp = CreateToolTip(self.train_batch_size_label, "The batch size to use for training.")
        #self.train_batch_size_label.grid(row=3, column=0, sticky="nsew")
        self.train_batch_size_var = tk.StringVar()
        self.train_batch_size_var.set(self.batch_size)
        #make a list of values from 1 to 60 that are strings
        #train_batch_size_values = 
        self.train_batch_size_dropdown = ctk.CTkOptionMenu(self.training_frame_subframe, variable=self.train_batch_size_var, values=[str(i) for i in range(1,61)])
        #self.train_batch_size_dropdown.grid(row=3, column=1, sticky="nsew")

        #create train epochs dark mode 
        self.train_epochs_label = ctk.CTkLabel(self.training_frame_subframe, text="Train Epochs")
        train_epochs_label_ttp = CreateToolTip(self.train_epochs_label, "The number of epochs to train for. An epoch is one pass through the entire dataset.")
        #self.train_epochs_label.grid(row=4, column=0, sticky="nsew")
        self.train_epochs_entry = ctk.CTkEntry(self.training_frame_subframe)
        #self.train_epochs_entry.grid(row=4, column=1, sticky="nsew")
        self.train_epochs_entry.insert(0, self.num_train_epochs)
        
        #create mixed precision dark mode dropdown
        self.mixed_precision_label = ctk.CTkLabel(self.training_frame_subframe, text="Mixed Precision")
        mixed_precision_label_ttp = CreateToolTip(self.mixed_precision_label, "Use mixed precision training to speed up training, FP16 is recommended but requires a GPU with Tensor Cores.")
        #self.mixed_precision_label.grid(row=5, column=0, sticky="nsew")
        self.mixed_precision_var = tk.StringVar()
        self.mixed_precision_var.set(self.mixed_precision)
        self.mixed_precision_dropdown = ctk.CTkOptionMenu(self.training_frame_subframe, variable=self.mixed_precision_var,values=["fp16", "fp32"])
        #self.mixed_precision_dropdown.grid(row=5, column=1, sticky="nsew")

        #create use 8bit adam checkbox
        self.use_8bit_adam_var = tk.IntVar()
        self.use_8bit_adam_var.set(self.use_8bit_adam)
        #create label
        self.use_8bit_adam_label = ctk.CTkLabel(self.training_frame_subframe, text="Use 8bit Adam")
        use_8bit_adam_label_ttp = CreateToolTip(self.use_8bit_adam_label, "Use 8bit Adam to speed up training, requires bytsandbytes.")
        #self.use_8bit_adam_label.grid(row=6, column=0, sticky="nsew")
        #create checkbox
        self.use_8bit_adam_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.use_8bit_adam_var,text='')
        #self.use_8bit_adam_checkbox.grid(row=6, column=1, sticky="nsew")
        #create use gradient checkpointing checkbox
        self.use_gradient_checkpointing_var = tk.IntVar()
        self.use_gradient_checkpointing_var.set(self.use_gradient_checkpointing)
        #create label
        self.use_gradient_checkpointing_label = ctk.CTkLabel(self.training_frame_subframe, text="Use Gradient Checkpointing")
        use_gradient_checkpointing_label_ttp = CreateToolTip(self.use_gradient_checkpointing_label, "Use gradient checkpointing to reduce RAM usage.")
        #self.use_gradient_checkpointing_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox
        self.use_gradient_checkpointing_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.use_gradient_checkpointing_var)
        #self.use_gradient_checkpointing_checkbox.grid(row=7, column=1, sticky="nsew")
        #create gradient accumulation steps dark mode dropdown with values from 1 to 60
        self.gradient_accumulation_steps_label = ctk.CTkLabel(self.training_frame_subframe, text="Gradient Accumulation Steps")
        gradient_accumulation_steps_label_ttp = CreateToolTip(self.gradient_accumulation_steps_label, "The number of gradient accumulation steps to use, this is useful for training with limited GPU memory.")
        #self.gradient_accumulation_steps_label.grid(row=8, column=0, sticky="nsew")
        self.gradient_accumulation_steps_var = tk.StringVar()
        self.gradient_accumulation_steps_var.set(self.accumulation_steps)
        self.gradient_accumulation_steps_dropdown = ctk.CTkOptionMenu(self.training_frame_subframe, variable=self.gradient_accumulation_steps_var, values=['0','1','2','3','4','5','6','7','8','9','10'])
        #self.gradient_accumulation_steps_dropdown.grid(row=8, column=1, sticky="nsew")
        #create learning rate dark mode entry
        self.learning_rate_label = ctk.CTkLabel(self.training_frame_subframe, text="Learning Rate")
        learning_rate_label_ttp = CreateToolTip(self.learning_rate_label, "The learning rate to use for training.")
        #self.learning_rate_label.grid(row=9, column=0, sticky="nsew")
        self.learning_rate_entry = ctk.CTkEntry(self.training_frame_subframe)
        #self.learning_rate_entry.grid(row=9, column=1, sticky="nsew")
        self.learning_rate_entry.insert(0, self.learning_rate)
        #create learning rate scheduler dropdown
        self.learning_rate_scheduler_label = ctk.CTkLabel(self.training_frame_subframe, text="Learning Rate Scheduler")
        learning_rate_scheduler_label_ttp = CreateToolTip(self.learning_rate_scheduler_label, "The learning rate scheduler to use for training.")
        #self.learning_rate_scheduler_label.grid(row=10, column=0, sticky="nsew")
        self.learning_rate_scheduler_var = tk.StringVar()
        self.learning_rate_scheduler_var.set(self.learning_rate_schedule)
        self.learning_rate_scheduler_dropdown = ctk.CTkOptionMenu(self.training_frame_subframe, variable=self.learning_rate_scheduler_var, values=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
        #self.learning_rate_scheduler_dropdown.grid(row=10, column=1, sticky="nsew")
        #create num warmup steps dark mode entry
        self.num_warmup_steps_label = ctk.CTkLabel(self.training_frame_subframe, text="LR Warmup Steps")
        num_warmup_steps_label_ttp = CreateToolTip(self.num_warmup_steps_label, "The number of warmup steps to use for the learning rate scheduler.")
        #self.num_warmup_steps_label.grid(row=11, column=0, sticky="nsew")
        self.num_warmup_steps_entry = ctk.CTkEntry(self.training_frame_subframe)
        #self.num_warmup_steps_entry.grid(row=11, column=1, sticky="nsew")
        self.num_warmup_steps_entry.insert(0, self.learning_rate_warmup_steps)
        #create use latent cache checkbox
        #self.use_latent_cache_var = tk.IntVar()
        #self.use_latent_cache_var.set(self.do_not_use_latents_cache)
        #create label
        #self.use_latent_cache_label = ctk.CTkLabel(self.training_frame_subframe, text="Use Latent Cache")
        #use_latent_cache_label_ttp = CreateToolTip(self.use_latent_cache_label, "Cache the latents to speed up training.")
        #self.use_latent_cache_label.grid(row=12, column=0, sticky="nsew")
        #create checkbox
        #self.use_latent_cache_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.use_latent_cache_var)
        #self.use_latent_cache_checkbox.grid(row=12, column=1, sticky="nsew")
        #create save latent cache checkbox
        #self.save_latent_cache_var = tk.IntVar()
        #self.save_latent_cache_var.set(self.save_latents_cache)
        #create label
        #self.save_latent_cache_label = ctk.CTkLabel(self.training_frame_subframe, text="Save Latent Cache")
        #save_latent_cache_label_ttp = CreateToolTip(self.save_latent_cache_label, "Save the latents cache to disk after generation, will be remade if batch size changes.")
        #self.save_latent_cache_label.grid(row=13, column=0, sticky="nsew")
        #create checkbox
        #self.save_latent_cache_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.save_latent_cache_var)
        #self.save_latent_cache_checkbox.grid(row=13, column=1, sticky="nsew")
        #create regnerate latent cache checkbox
        self.regenerate_latent_cache_var = tk.IntVar()
        self.regenerate_latent_cache_var.set(self.regenerate_latents_cache)
        #create label
        self.regenerate_latent_cache_label = ctk.CTkLabel(self.training_frame_subframe, text="Regenerate Latent Cache")
        regenerate_latent_cache_label_ttp = CreateToolTip(self.regenerate_latent_cache_label, "Force the latents cache to be regenerated.")
        #self.regenerate_latent_cache_label.grid(row=14, column=0, sticky="nsew")
        #create checkbox
        self.regenerate_latent_cache_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.regenerate_latent_cache_var)
        #self.regenerate_latent_cache_checkbox.grid(row=14, column=1, sticky="nsew")
        #create train text encoder checkbox
        self.train_text_encoder_var = tk.IntVar()
        self.train_text_encoder_var.set(self.train_text_encoder)
        #create label
        self.train_text_encoder_label = ctk.CTkLabel(self.training_frame_subframe, text="Train Text Encoder")
        train_text_encoder_label_ttp = CreateToolTip(self.train_text_encoder_label, "Train the text encoder along with the UNET.")
        #self.train_text_encoder_label.grid(row=15, column=0, sticky="nsew")
        #create checkbox
        self.train_text_encoder_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.train_text_encoder_var)
        #self.train_text_encoder_checkbox.grid(row=15, column=1, sticky="nsew")
        #create limit text encoder encoder entry
        self.clip_penultimate_var = tk.IntVar()
        self.clip_penultimate_var.set(self.clip_penultimate)
        #create label
        self.clip_penultimate_label = ctk.CTkLabel(self.training_frame_subframe, text="Clip Penultimate")
        clip_penultimate_label_ttp = CreateToolTip(self.clip_penultimate_label, "Train using the Penultimate layer of the text encoder.")
        #create checkbox
        self.clip_penultimate_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.clip_penultimate_var)
        

        self.limit_text_encoder_var = tk.StringVar()
        self.limit_text_encoder_var.set(self.limit_text_encoder)
        #create label
        self.limit_text_encoder_label = ctk.CTkLabel(self.training_frame_subframe, text="Limit Text Encoder")
        limit_text_encoder_label_ttp = CreateToolTip(self.limit_text_encoder_label, "Stop training the text encoder after this many epochs, use % to train for a percentage of the total epochs.")
        #self.limit_text_encoder_label.grid(row=16, column=0, sticky="nsew")
        #create entry
        self.limit_text_encoder_entry = ctk.CTkEntry(self.training_frame_subframe, textvariable=self.limit_text_encoder_var)
        #self.limit_text_encoder_entry.grid(row=16, column=1, sticky="nsew")
        
        #create checkbox disable cudnn benchmark
        self.disable_cudnn_benchmark_var = tk.IntVar()
        self.disable_cudnn_benchmark_var.set(self.disable_cudnn_benchmark)
        #create label for checkbox
        self.disable_cudnn_benchmark_label = ctk.CTkLabel(self.training_frame_subframe, text="EXPERIMENTAL: Disable cuDNN Benchmark")
        disable_cudnn_benchmark_label_ttp = CreateToolTip(self.disable_cudnn_benchmark_label, "Disable cuDNN benchmarking, may offer 2x performance on some systems and stop OOM errors.")
        #self.disable_cudnn_benchmark_label.grid(row=17, column=0, sticky="nsew")
        #create checkbox
        self.disable_cudnn_benchmark_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.disable_cudnn_benchmark_var)
        #self.disable_cudnn_benchmark_checkbox.grid(row=17, column=1, sticky="nsew")
        #add conditional dropout entry
        self.conditional_dropout_label = ctk.CTkLabel(self.training_frame_subframe, text="Conditional Dropout")
        conditional_dropout_label_ttp = CreateToolTip(self.conditional_dropout_label, "Precentage of probability to drop out a caption token to train the model to be more robust to missing words.")
        self.conditional_dropout_entry = ctk.CTkEntry(self.training_frame_subframe)
        self.conditional_dropout_entry.insert(0, self.conditional_dropout)
        #create use EMA switch
        self.use_ema_var = tk.IntVar()
        self.use_ema_var.set(self.use_ema)
        #create label
        self.use_ema_label = ctk.CTkLabel(self.training_frame_subframe, text="Use EMA")
        use_ema_label_ttp = CreateToolTip(self.use_ema_label, "Use Exponential Moving Average to smooth the training paramaters. Will increase VRAM usage.")
        #self.use_ema_label.grid(row=18, column=0, sticky="nsew")
        #create checkbox
        self.use_ema_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.use_ema_var)

        #create with prior loss preservation checkbox
        self.with_prior_loss_preservation_var = tk.IntVar()
        self.with_prior_loss_preservation_var.set(self.with_prior_reservation)
        #create label
        self.with_prior_loss_preservation_label = ctk.CTkLabel(self.training_frame_subframe, text="With Prior Loss Preservation")
        with_prior_loss_preservation_label_ttp = CreateToolTip(self.with_prior_loss_preservation_label, "Use the prior loss preservation method. part of Dreambooth.")
        self.with_prior_loss_preservation_label.grid(row=19, column=0, sticky="nsew")
        #create checkbox
        self.with_prior_loss_preservation_checkbox = ctk.CTkSwitch(self.training_frame_subframe, variable=self.with_prior_loss_preservation_var)
        self.with_prior_loss_preservation_checkbox.grid(row=19, column=1, sticky="nsew")
        #create prior loss preservation weight entry
        self.prior_loss_preservation_weight_label = ctk.CTkLabel(self.training_frame_subframe, text="Weight")
        prior_loss_preservation_weight_label_ttp = CreateToolTip(self.prior_loss_preservation_weight_label, "The weight of the prior loss preservation loss.")
        self.prior_loss_preservation_weight_label.grid(row=19, column=1, sticky="e")
        self.prior_loss_preservation_weight_entry = ctk.CTkEntry(self.training_frame_subframe)
        self.prior_loss_preservation_weight_entry.grid(row=19, column=3, sticky="w")
        self.prior_loss_preservation_weight_entry.insert(0, self.prior_loss_weight)
        

    def create_dataset_settings_widgets(self):
        #self.dataset_settings_label = ctk.CTkLabel(self.dataset_tab, text="Dataset Settings", font=("Arial", 12, "bold"))
        #self.dataset_settings_label.grid(row=0, column=0, sticky="nsew")
        self.dataset_frame_title = ctk.CTkLabel(self.dataset_frame, text="Dataset Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.dataset_frame_title.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")  
        #create use text files as captions checkbox
        self.use_text_files_as_captions_var = tk.IntVar()
        self.use_text_files_as_captions_var.set(self.use_text_files_as_captions)
        #create label
        self.use_text_files_as_captions_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Use Text Files as Captions")
        use_text_files_as_captions_label_ttp = CreateToolTip(self.use_text_files_as_captions_label, "Use the text files as captions for training, text files must have same name as image, instance prompt/token will be ignored.")
        self.use_text_files_as_captions_label.grid(row=1, column=0, sticky="nsew")
        #create checkbox
        self.use_text_files_as_captions_checkbox = ctk.CTkSwitch(self.dataset_frame_subframe, variable=self.use_text_files_as_captions_var)
        self.use_text_files_as_captions_checkbox.grid(row=1, column=1, sticky="nsew")
        # create use image names as captions checkbox
        self.use_image_names_as_captions_var = tk.IntVar()
        self.use_image_names_as_captions_var.set(self.use_image_names_as_captions)
        # create label
        self.use_image_names_as_captions_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Use Image Names as Captions")
        use_image_names_as_captions_label_ttp = CreateToolTip(self.use_image_names_as_captions_label, "Use the image names as captions for training, instance prompt/token will be ignored.")
        self.use_image_names_as_captions_label.grid(row=2, column=0, sticky="nsew")
        # create checkbox
        self.use_image_names_as_captions_checkbox = ctk.CTkSwitch(self.dataset_frame_subframe, variable=self.use_image_names_as_captions_var)
        self.use_image_names_as_captions_checkbox.grid(row=2, column=1, sticky="nsew")
        # create auto balance dataset checkbox
        self.auto_balance_dataset_var = tk.IntVar()
        self.auto_balance_dataset_var.set(self.auto_balance_concept_datasets)
        # create label
        self.auto_balance_dataset_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Auto Balance Dataset")
        auto_balance_dataset_label_ttp = CreateToolTip(self.auto_balance_dataset_label, "Will use the concept with the least amount of images to balance the dataset by removing images from the other concepts.")
        self.auto_balance_dataset_label.grid(row=3, column=0, sticky="nsew")
        # create checkbox
        self.auto_balance_dataset_checkbox = ctk.CTkSwitch(self.dataset_frame_subframe, variable=self.auto_balance_dataset_var)
        self.auto_balance_dataset_checkbox.grid(row=3, column=1, sticky="nsew")
        #create add class images to dataset checkbox
        self.add_class_images_to_dataset_var = tk.IntVar()
        self.add_class_images_to_dataset_var.set(self.add_class_images_to_training)
        #create label
        self.add_class_images_to_dataset_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Add Class Images to Dataset")
        add_class_images_to_dataset_label_ttp = CreateToolTip(self.add_class_images_to_dataset_label, "Will add class images without prior preservation to the dataset.")
        self.add_class_images_to_dataset_label.grid(row=4, column=0, sticky="nsew")
        #create checkbox
        self.add_class_images_to_dataset_checkbox = ctk.CTkSwitch(self.dataset_frame_subframe, variable=self.add_class_images_to_dataset_var)
        self.add_class_images_to_dataset_checkbox.grid(row=4, column=1, sticky="nsew")
        #create number of class images entry
        self.number_of_class_images_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Number of Class Images")
        number_of_class_images_label_ttp = CreateToolTip(self.number_of_class_images_label, "The number of class images to add to the dataset, if they don't exist in the class directory they will be generated.")
        self.number_of_class_images_label.grid(row=5, column=0, sticky="nsew")
        self.number_of_class_images_entry = ctk.CTkEntry(self.dataset_frame_subframe)
        self.number_of_class_images_entry.grid(row=5, column=1, sticky="nsew")
        self.number_of_class_images_entry.insert(0, self.num_class_images)
        #create dataset repeat entry
        self.dataset_repeats_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Dataset Repeats")
        dataset_repeat_label_ttp = CreateToolTip(self.dataset_repeats_label, "The number of times to repeat the dataset, this will increase the number of images in the dataset.")
        self.dataset_repeats_label.grid(row=6, column=0, sticky="nsew")
        self.dataset_repeats_entry = ctk.CTkEntry(self.dataset_frame_subframe)
        self.dataset_repeats_entry.grid(row=6, column=1, sticky="nsew")
        self.dataset_repeats_entry.insert(0, self.dataset_repeats)

        #add use_aspect_ratio_bucketing checkbox
        self.use_aspect_ratio_bucketing_var = tk.IntVar()
        self.use_aspect_ratio_bucketing_var.set(self.use_aspect_ratio_bucketing)
        #create label
        self.use_aspect_ratio_bucketing_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Use Aspect Ratio Bucketing")
        use_aspect_ratio_bucketing_label_ttp = CreateToolTip(self.use_aspect_ratio_bucketing_label, "Will use aspect ratio bucketing, may improve aspect ratio generations.")
        self.use_aspect_ratio_bucketing_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox
        self.use_aspect_ratio_bucketing_checkbox = ctk.CTkSwitch(self.dataset_frame_subframe, variable=self.use_aspect_ratio_bucketing_var)
        self.use_aspect_ratio_bucketing_checkbox.grid(row=7, column=1, sticky="nsew")
        #do something on checkbox click
        self.use_aspect_ratio_bucketing_checkbox.bind("<Button-1>", self.aspect_ratio_mode_toggles)
        
        #option menu to select aspect ratio bucketing mode
        self.aspect_ratio_bucketing_mode_var = tk.StringVar()
        self.aspect_ratio_bucketing_mode_var.set(self.aspect_ratio_bucketing_mode)
        self.aspect_ratio_bucketing_mode_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Aspect Ratio Bucketing Mode")
        aspect_ratio_bucketing_mode_label_ttp = CreateToolTip(self.aspect_ratio_bucketing_mode_label, "Select what the Auto Bucketing will do in case the bucket doesn't match the batch size, dynamic will choose the least amount of adding/removing of images per bucket.")
        self.aspect_ratio_bucketing_mode_label.grid(row=8, column=0, sticky="nsew")
        self.aspect_ratio_bucketing_mode_option_menu = ctk.CTkOptionMenu(self.dataset_frame_subframe, variable=self.aspect_ratio_bucketing_mode_var, values=['Dynamic Fill', 'Drop Fill', 'Duplicate Fill'])
        self.aspect_ratio_bucketing_mode_option_menu.grid(row=8, column=1, sticky="nsew")
        #option menu to select dynamic bucketing mode (if enabled)
        self.dynamic_bucketing_mode_var = tk.StringVar()
        self.dynamic_bucketing_mode_var.set(self.dynamic_bucketing_mode)
        self.dynamic_bucketing_mode_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Dynamic Preference")
        dynamic_bucketing_mode_label_ttp = CreateToolTip(self.dynamic_bucketing_mode_label, "If you're using dynamic mode, choose what you prefer in the case that dropping and duplicating are the same amount of images.")
        self.dynamic_bucketing_mode_label.grid(row=9, column=0, sticky="nsew")
        self.dynamic_bucketing_mode_option_menu = ctk.CTkOptionMenu(self.dataset_frame_subframe, variable=self.dynamic_bucketing_mode_var, values=['Duplicate', 'Drop'])
        self.dynamic_bucketing_mode_option_menu.grid(row=9, column=1, sticky="nsew")
        #option menu to select dynamic bucketing mode (if enabled)

        

        #add download dataset entry
        #add a switch to duplicate fill bucket
        #self.duplicate_fill_buckets_var = tk.IntVar()
        #self.duplicate_fill_buckets_var.set(self.duplicate_fill_buckets)
        #create label
        #self.duplicate_fill_buckets_label = ctk.CTkLabel(self.dataset_frame_subframe, text="Force Fill Buckets with Duplicates")
        #duplicate_fill_buckets_label_ttp = CreateToolTip(self.duplicate_fill_buckets_label, "Will duplicate to fill buckets, enable this to avoid buckets dropping images.")
        #self.duplicate_fill_buckets_label.grid(row=8, column=0, sticky="nsew")
        #create checkbox
        #self.duplicate_fill_buckets_checkbox = ctk.CTkSwitch(self.dataset_frame_subframe, variable=self.duplicate_fill_buckets_var)
        #self.duplicate_fill_buckets_checkbox.grid(row=8, column=1, sticky="nsew")
        #self.use_aspect_ratio_bucketing_checkbox.bind("<Button-1>", self.duplicate_fill_buckets_label.configure(state="disabled"))
        #self.use_aspect_ratio_bucketing_checkbox.bind("<Button-1>", self.duplicate_fill_buckets_checkbox.configure(state="disabled"))
        
    def create_sampling_settings_widgets(self):
        self.sampling_title = ctk.CTkLabel(self.sampling_frame, text="Sampling Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.sampling_title.grid(row=0, column=0, padx=20, pady=20)  
        #create sample every n steps entry
        self.sample_step_interval_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Sample Every N Steps")
        sample_step_interval_label_ttp = CreateToolTip(self.sample_step_interval_label, "Will sample the model every N steps.")
        self.sample_step_interval_label.grid(row=1, column=0, sticky="nsew")
        self.sample_step_interval_entry = ctk.CTkEntry(self.sampling_frame_subframe)
        self.sample_step_interval_entry.grid(row=1, column=1, sticky="nsew")
        self.sample_step_interval_entry.insert(0, self.sample_step_interval)
        #create saver every n epochs entry
        self.save_every_n_epochs_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Save and sample Every N Epochs")
        save_every_n_epochs_label_ttp = CreateToolTip(self.save_every_n_epochs_label, "Will save and sample the model every N epochs.")
        self.save_every_n_epochs_label.grid(row=2, column=0, sticky="nsew")
        self.save_every_n_epochs_entry = ctk.CTkEntry(self.sampling_frame_subframe)
        self.save_every_n_epochs_entry.grid(row=2, column=1, sticky="nsew")
        self.save_every_n_epochs_entry.insert(0, self.save_and_sample_every_x_epochs)
        #create number of samples to generate entry
        self.number_of_samples_to_generate_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Number of Samples to Generate")
        number_of_samples_to_generate_label_ttp = CreateToolTip(self.number_of_samples_to_generate_label, "The number of samples to generate per prompt.")
        self.number_of_samples_to_generate_label.grid(row=3, column=0, sticky="nsew")
        self.number_of_samples_to_generate_entry = ctk.CTkEntry(self.sampling_frame_subframe)
        self.number_of_samples_to_generate_entry.grid(row=3, column=1, sticky="nsew")
        self.number_of_samples_to_generate_entry.insert(0, self.num_samples_to_generate)
        #create sample width entry
        self.sample_width_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Sample Width")
        sample_width_label_ttp = CreateToolTip(self.sample_width_label, "The width of the generated samples.")
        self.sample_width_label.grid(row=4, column=0, sticky="nsew")
        self.sample_width_entry = ctk.CTkEntry(self.sampling_frame_subframe)
        self.sample_width_entry.grid(row=4, column=1, sticky="nsew")
        self.sample_width_entry.insert(0, self.sample_width)
        #create sample height entry
        self.sample_height_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Sample Height")
        sample_height_label_ttp = CreateToolTip(self.sample_height_label, "The height of the generated samples.")
        self.sample_height_label.grid(row=5, column=0, sticky="nsew")
        self.sample_height_entry = ctk.CTkEntry(self.sampling_frame_subframe)
        self.sample_height_entry.grid(row=5, column=1, sticky="nsew")
        self.sample_height_entry.insert(0, self.sample_height)
        
        #create a checkbox to sample_on_training_start
        self.sample_on_training_start_var = tk.IntVar()
        self.sample_on_training_start_var.set(self.sample_on_training_start)
        #create label
        self.sample_on_training_start_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Sample On Training Start")
        sample_on_training_start_label_ttp = CreateToolTip(self.sample_on_training_start_label, "Will save and sample the model on training start, useful for debugging and comparison.")
        self.sample_on_training_start_label.grid(row=6, column=0, sticky="nsew")
        #create checkbox
        self.sample_on_training_start_checkbox = ctk.CTkSwitch(self.sampling_frame_subframe, variable=self.sample_on_training_start_var)
        self.sample_on_training_start_checkbox.grid(row=6, column=1, sticky="nsew")
        #create sample random aspect ratio checkbox
        self.sample_random_aspect_ratio_var = tk.IntVar()
        self.sample_random_aspect_ratio_var.set(self.sample_random_aspect_ratio)
        #create label
        self.sample_random_aspect_ratio_label = ctk.CTkLabel(self.sampling_frame_subframe, text="Sample Random Aspect Ratio")
        sample_random_aspect_ratio_label_ttp = CreateToolTip(self.sample_random_aspect_ratio_label, "Will generate samples with random aspect ratios, useful to check aspect ratio bucketing.")
        self.sample_random_aspect_ratio_label.grid(row=7, column=0, sticky="nsew")
        #create checkbox
        self.sample_random_aspect_ratio_checkbox = ctk.CTkSwitch(self.sampling_frame_subframe, variable=self.sample_random_aspect_ratio_var)
        self.sample_random_aspect_ratio_checkbox.grid(row=7, column=1, sticky="nsew")
        #create add sample prompt button
        self.add_sample_prompt_button = ctk.CTkButton(self.sampling_frame_subframe, text="Add Sample Prompt",  command=self.add_sample_prompt)
        add_sample_prompt_button_ttp = CreateToolTip(self.add_sample_prompt_button, "Add a sample prompt to the list.")
        self.add_sample_prompt_button.grid(row=1, column=2, sticky="nsew")
        #create remove sample prompt button
        self.remove_sample_prompt_button = ctk.CTkButton(self.sampling_frame_subframe, text="Remove Sample Prompt",  command=self.remove_sample_prompt)
        remove_sample_prompt_button_ttp = CreateToolTip(self.remove_sample_prompt_button, "Remove a sample prompt from the list.")
        self.remove_sample_prompt_button.grid(row=1, column=3, sticky="nsew")

        #for every prompt in self.sample_prompts, create a label and entry
        self.sample_prompt_labels = []
        self.sample_prompt_entries = []
        self.sample_prompt_row = 2
        for i in range(len(self.sample_prompts)):
            #create label
            self.sample_prompt_labels.append(ctk.CTkLabel(self.sampling_frame_subframe, text="Sample Prompt " + str(i)))
            self.sample_prompt_labels[i].grid(row=self.sample_prompt_row + i, column=2, sticky="nsew")
            #create entry
            self.sample_prompt_entries.append(ctk.CTkEntry(self.sampling_frame_subframe, width=70))
            self.sample_prompt_entries[i].grid(row=self.sample_prompt_row + i, column=3, sticky="nsew")
            self.sample_prompt_entries[i].insert(0, self.sample_prompts[i])
        for i in self.sample_prompt_entries:
            i.bind("<Button-3>", self.create_right_click_menu)
        self.controlled_sample_row = 2 + len(self.sample_prompts)
        #create a button to add controlled seed sample
        self.add_controlled_seed_sample_button = ctk.CTkButton(self.sampling_frame_subframe, text="Add Controlled Seed Sample",  command=self.add_controlled_seed_sample)
        add_controlled_seed_sample_button_ttp = CreateToolTip(self.add_controlled_seed_sample_button, "Will generate a sample using the seed at every save interval.")
        self.add_controlled_seed_sample_button.grid(row=self.controlled_sample_row + len(self.sample_prompts), column=2, sticky="nsew")
        #create a button to remove controlled seed sample
        self.remove_controlled_seed_sample_button = ctk.CTkButton(self.sampling_frame_subframe, text="Remove Controlled Seed Sample",  command=self.remove_controlled_seed_sample)
        remove_controlled_seed_sample_button_ttp = CreateToolTip(self.remove_controlled_seed_sample_button, "Will remove the last controlled seed sample.")
        self.remove_controlled_seed_sample_button.grid(row=self.controlled_sample_row + len(self.sample_prompts), column=3, sticky="nsew")
        #for every controlled seed sample in self.controlled_seed_samples, create a label and entry
        self.controlled_seed_sample_labels = []
        self.controlled_seed_sample_entries = []
        self.controlled_seed_buttons = [self.add_controlled_seed_sample_button, self.remove_controlled_seed_sample_button]
        
        for i in range(len(self.add_controlled_seed_to_sample)):
            #create label
            self.controlled_seed_sample_labels.append(ctk.CTkLabel(self.sampling_frame_subframe, text="Controlled Seed Sample " + str(i)))
            self.controlled_seed_sample_labels[i].grid(row=self.controlled_sample_row + len(self.sample_prompts) + i, column=2, sticky="nsew")
            #create entry
            self.controlled_seed_sample_entries.append(ctk.CTkEntry(self.sampling_frame_subframe))
            self.controlled_seed_sample_entries[i].grid(row=self.controlled_sample_row + len(self.sample_prompts) + i, column=3, sticky="nsew")
            self.controlled_seed_sample_entries[i].insert(0, self.add_controlled_seed_to_sample[i])
        for i in self.controlled_seed_sample_entries:
            i.bind("<Button-3>", self.create_right_click_menu)
    def create_data_settings_widgets(self):
        #add concept settings label
        self.data_frame_title = ctk.CTkLabel(self.data_frame, text='Data Settings', font=ctk.CTkFont(size=20, weight="bold"))
        self.data_frame_title.grid(row=0, column=0,columnspan=2, padx=20, pady=20)    
        #add load concept from json button
        #add empty label
        empty = ctk.CTkLabel(self.data_frame_subframe, text="",width=40)
        empty.grid(row=1, column=0, sticky="nsew")
        self.load_concept_from_json_button = ctk.CTkButton(self.data_frame_subframe, text="Load Concepts From JSON",  command=self.load_concept_from_json)
        self.load_concept_from_json_button.grid(row=1, column=1, sticky="e")
        load_concept_from_json_button_ttp = CreateToolTip(self.load_concept_from_json_button, "Load concepts from a JSON file, compatible with Shivam's concept list.")
        #self.load_concept_from_json_button.grid(row=1, column=0, sticky="nsew")
        #add save concept to json button
        self.save_concept_to_json_button = ctk.CTkButton(self.data_frame_subframe, text="Save Concepts To JSON",  command=self.save_concept_to_json)
        self.save_concept_to_json_button.grid(row=1, column=2, sticky="e")
        save_concept_to_json_button_ttp = CreateToolTip(self.save_concept_to_json_button, "Save concepts to a JSON file, compatible with Shivam's concept list.")
        #self.save_concept_to_json_button.grid(row=1, column=1, sticky="nsew")
        #create a button to add concept
        self.add_concept_button = ctk.CTkButton(self.data_frame_subframe, text="Add Concept",  command=self.add_new_concept,width=50)
        self.add_concept_button.grid(row=1, column=3, sticky="e")
        #self.add_concept_button.grid(row=2, column=0, sticky="nsew")
        #create a button to remove concept
        self.remove_concept_button = ctk.CTkButton(self.data_frame_subframe, text="Remove Concept",  command=self.remove_new_concept,width=50)
        self.remove_concept_button.grid(row=1, column=4, sticky="e")
        #self.remove_concept_button.grid(row=2, column=1, sticky="nsew")
        self.previous_page_button = ctk.CTkButton(self.data_frame_subframe, text="Previous Page",  command=self.next_concept_page,width=50, state="disabled")
        self.previous_page_button.grid(row=1, column=5, sticky="e")
        #self.remove_concept_button.grid(row=2, column=1, sticky="nsew")
        self.next_page_button = ctk.CTkButton(self.data_frame_subframe, text="Next Page",  command=self.next_concept_page,width=50, state="disabled")
        self.next_page_button.grid(row=1, column=6, sticky="e")
        #self.remove_concept_button.grid(row=2, column=1, sticky="nsew")
        #self.concept_entries = []
        #self.concept_labels = []
        #self.concept_file_dialog_buttons = []
    def next_concept_page(self):
        self.concept_page += 1
        self.update_concept_page()
    def create_plyaground_widgets(self):
        self.playground_title = ctk.CTkLabel(self.playground_frame, text="Model Playground", font=ctk.CTkFont(size=20, weight="bold"))
        #add play model entry with button to open file dialog
        self.play_model_label = ctk.CTkLabel(self.playground_frame_subframe, text="Diffusers Model Directory")
        self.play_model_entry = ctk.CTkEntry(self.playground_frame_subframe,placeholder_text="CTkEntry")
        self.play_model_entry.insert(0, self.play_input_model_path)
        self.play_model_file_dialog_button = ctk.CTkButton(self.playground_frame_subframe, text="...",width=5, command=lambda: self.open_file_dialog(self.play_model_entry))
        self.play_model_file_dialog_button.grid(row=0, column=2, sticky="w")
        #add a prompt entry to play tab
        self.play_prompt_label = ctk.CTkLabel(self.playground_frame_subframe, text="Prompt")
        self.play_prompt_entry = ctk.CTkEntry(self.playground_frame_subframe)
        self.play_prompt_entry.insert(0, self.play_postive_prompt)
        #add a negative prompt entry to play tab
        self.play_negative_prompt_label = ctk.CTkLabel(self.playground_frame_subframe, text="Negative Prompt")
        self.play_negative_prompt_entry = ctk.CTkEntry(self.playground_frame_subframe, width=40)
        self.play_negative_prompt_entry.insert(0, self.play_negative_prompt)
        #add a seed entry to play tab
        self.play_seed_label = ctk.CTkLabel(self.playground_frame_subframe, text="Seed")
        self.play_seed_entry = ctk.CTkEntry(self.playground_frame_subframe)
        self.play_seed_entry.insert(0, self.play_seed)
        #add a keep seed checkbox next to seed entry
        self.play_keep_seed_var = tk.IntVar()
        self.play_keep_seed_var.set(self.play_keep_seed)
        self.play_keep_seed_checkbox = ctk.CTkCheckBox(self.playground_frame_subframe, text="Keep Seed", variable=self.play_keep_seed_var)
        
        #add a temperature slider from 0.1 to 1.0
        
        #create a steps slider from 1 to 100
        self.play_steps_label = ctk.CTkLabel(self.playground_frame_subframe, text=f"Steps: {self.play_steps}")
        self.play_steps_slider = ctk.CTkSlider(self.playground_frame_subframe, from_=1, to=150, number_of_steps=149, command= lambda x: self.play_steps_label.configure(text="Steps: " + str(int(self.play_steps_slider.get()))))
        
        #on slider change update the value
        #self.play_steps_slider.bind("<Configure>", self.play_steps_label.configure(text="Steps: " + str(self.play_steps_slider.get())))
        self.play_steps_slider.set(self.play_steps)
        #add a scheduler selection box

        
        self.play_scheduler_label = ctk.CTkLabel(self.playground_frame_subframe, text="Scheduler")
        self.play_scheduler_variable = tk.StringVar(self.playground_frame_subframe)
        self.play_scheduler_variable.set(self.play_scheduler)
        self.play_scheduler_option_menu = ctk.CTkOptionMenu(self.playground_frame_subframe, variable=self.play_scheduler_variable, values=self.schedulers)
        
        #add resoltuion slider from 256 to 1024 in increments of 64 for width and height
        self.play_resolution_label = ctk.CTkLabel(self.playground_frame_subframe, text="Resolution")
        self.play_resolution_label_height = ctk.CTkLabel(self.playground_frame_subframe, text=f"Height: {self.play_sample_height}")
        self.play_resolution_label_width = ctk.CTkLabel(self.playground_frame_subframe, text=f"Width: {self.play_sample_width}")
        #add sliders for height and width
        #make a list of resolutions from 256 to 2048 in increments of 64
        #play_resolutions = []
        #for i in range(256,2049,64):
        #    play_resolutions.append(str(i))
        self.play_resolution_slider_height = ctk.CTkSlider(self.playground_frame_subframe,from_=128, to=2048, number_of_steps=30, command= lambda x: self.play_resolution_label_height.configure(text="Height: " + str(int(self.play_resolution_slider_height.get()))))
        self.play_resolution_slider_width = ctk.CTkSlider(self.playground_frame_subframe, from_=128, to=2048, number_of_steps=30, command= lambda x: self.play_resolution_label_width.configure(text="Width: " + str(int(self.play_resolution_slider_width.get()))))
        self.play_resolution_slider_width.set(self.play_sample_width)
        self.play_resolution_slider_height.set(self.play_sample_height)
        #add a cfg slider 0.5 to 25 in increments of 0.5
        self.play_cfg_label = ctk.CTkLabel(self.playground_frame_subframe, text=f"CFG: {self.play_cfg}")
        self.play_cfg_slider = ctk.CTkSlider(self.playground_frame_subframe, from_=0.5, to=25, number_of_steps=49, command= lambda x: self.play_cfg_label.configure(text="CFG: " + str(self.play_cfg_slider.get())))
        self.play_cfg_slider.set(self.play_cfg)
        #add Toolbox label
        self.play_toolbox_label = ctk.CTkLabel(self.playground_frame_subframe, text="Toolbox")
        self.play_generate_image_button = ctk.CTkButton(self.playground_frame_subframe, text="Generate Image", command=lambda: self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), int(self.play_resolution_slider_height.get()), int(self.play_resolution_slider_width.get()), self.play_cfg_slider.get(), self.play_steps_slider.get()))
        #create a canvas to display the generated image
        #self.play_image_canvas = tk.Canvas(self.playground_frame_subframe, width=512, height=512, highlightthickness=0)
        #self.play_image_canvas.grid(row=11, column=0, columnspan=3, sticky="nsew")
        #create a button to generate image
        self.play_prompt_entry.bind("<Return>", lambda event: self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), int(self.play_resolution_slider_height.get()), int(self.play_resolution_slider_width.get()), self.play_cfg_slider.get(), self.play_steps_slider.get()))
        self.play_negative_prompt_entry.bind("<Return>", lambda event: self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), int(self.play_resolution_slider_height.get()), int(self.play_resolution_slider_width.get()), self.play_cfg_slider.get(), self.play_steps_slider.get()))
        
        #add convert to ckpt button
        self.play_convert_to_ckpt_button = ctk.CTkButton(self.playground_frame_subframe, text="Convert To CKPT", command=lambda:self.convert_to_ckpt(model_path=self.play_model_entry.get()))
        #add interative generation button to act as a toggle
        #self.play_interactive_generation_button_bool = tk.BooleanVar()
        #self.play_interactive_generation_button = ctk.CTkButton(self.playground_frame_subframe, text="Interactive Generation", command=self.interactive_generation_button)
        #self.play_interactive_generation_button_bool.set(False)#add play model entry with button to open file dialog
    def create_toolbox_widgets(self):
        #add label to tools tab
        self.toolbox_title = ctk.CTkLabel(self.toolbox_frame, text="Toolbox", font=ctk.CTkFont(size=20, weight="bold"))
        self.toolbox_title.grid(row=0, column=0, padx=20, pady=20)  
        #empty row
        #self.empty_row = ctk.CTkLabel(self.toolbox_frame_subframe, text="")
        #self.empty_row.grid(row=1, column=0, sticky="nsew")
        #add a label model tools title
        self.model_tools_label = ctk.CTkLabel(self.toolbox_frame_subframe, text="Model Tools",  font=ctk.CTkFont(size=20, weight="bold"))
        self.model_tools_label.grid(row=2, column=0,columnspan=3, sticky="nsew",pady=10)
        #empty row
        #self.empty_row = ctk.CTkLabel(self.toolbox_frame_subframe, text="")
        #self.empty_row.grid(row=3, column=0, sticky="nsew")
        #add a button to convert to ckpt
        self.convert_to_ckpt_button = ctk.CTkButton(self.toolbox_frame_subframe, text="Convert Diffusers To CKPT", command=lambda:self.convert_to_ckpt())
        self.convert_to_ckpt_button.grid(row=4, column=0, columnspan=1, sticky="nsew")
        #add a button to convert ckpt to diffusers
        self.convert_ckpt_to_diffusers_button = ctk.CTkButton(self.toolbox_frame_subframe, text="Convert CKPT To Diffusers", command=lambda:self.convert_ckpt_to_diffusers())
        self.convert_ckpt_to_diffusers_button.grid(row=4, column=1, columnspan=1, sticky="nsew")
        #empty row
        self.empty_row = ctk.CTkLabel(self.toolbox_frame_subframe, text="")
        self.empty_row.grid(row=6, column=0, sticky="nsew")
        #add a label dataset tools title
        self.dataset_tools_label = ctk.CTkLabel(self.toolbox_frame_subframe, text="Dataset Tools",  font=ctk.CTkFont(size=20, weight="bold"))
        self.dataset_tools_label.grid(row=7, column=0,columnspan=3, sticky="nsew")

        #add a button for Caption Buddy
        #self.caption_buddy_button = ctk.CTkButton(self.toolbox_frame_subframe, text="Launch Caption Buddy",font=("Helvetica", 10, "bold"), command=lambda:self.caption_buddy())
        #self.caption_buddy_button.grid(row=8, column=0, columnspan=3, sticky="nsew")


        self.download_dataset_label = ctk.CTkLabel(self.toolbox_frame_subframe, text="Clone Dataset from HF")
        download_dataset_label_ttp = CreateToolTip(self.download_dataset_label, "Will git clone a HF dataset repo")
        self.download_dataset_label.grid(row=9, column=0, sticky="nsew")
        self.download_dataset_entry = ctk.CTkEntry(self.toolbox_frame_subframe)
        self.download_dataset_entry.grid(row=9, column=1, sticky="nsew")
        #add download dataset button
        self.download_dataset_button = ctk.CTkButton(self.toolbox_frame_subframe, text="Download Dataset", command=self.download_dataset)
        self.download_dataset_button.grid(row=9, column=2, sticky="nsew")
    def find_latest_generated_model(self,entry=None):
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
                    
                    if all(x in os.listdir(last_model_path) for x in self.required_folders):
                       # print(newest_dir)
                        last_model_path = last_model_path.replace("/", os.sep).replace("\\", os.sep)
                        if entry:
                            entry.delete(0, tk.END)
                            entry.insert(0, last_model_path)
                            return
                    else:
                        
                        newest_dirs = sorted(glob.iglob(last_output_path + os.sep + '*'), key=os.path.getctime, reverse=True)
                        #sort newest_dirs by date
                        for newest_dir in newest_dirs:
                            #check if the newest dir has all the required folders
                            if all(x in os.listdir(newest_dir) for x in self.required_folders):
                                last_model_path = newest_dir.replace("/", os.sep).replace("\\", os.sep)
                                if entry:
                                    entry.delete(0, tk.END)
                                    entry.insert(0, last_model_path)
                                    return
                else:
                        
                        newest_dirs = sorted(glob.iglob(last_output_path + os.sep + '*'), key=os.path.getctime, reverse=True)
                        #sort newest_dirs by date
                        for newest_dir in newest_dirs:
                            #check if the newest dir has all the required folders
                            if all(x in os.listdir(newest_dir) for x in self.required_folders):
                                last_model_path = newest_dir.replace("/", os.sep).replace("\\", os.sep)
                                if entry:
                                    entry.delete(0, tk.END)
                                    entry.insert(0, last_model_path)
                                    return
            else:
                return
        else:
            return

    def packageForCloud(self):
        #check if there's an export folder in the cwd and if not create one
        if not os.path.exists("exports"):
            os.mkdir("exports")
        exportDir = self.export_name
        if not os.path.exists("exports" + os.sep + exportDir):
            os.mkdir("exports" + os.sep + exportDir)
        else:
            #remove the old export folder
            shutil.rmtree("exports" + os.sep + exportDir)
            os.mkdir("exports" + os.sep + exportDir)
        self.full_export_path = "exports" + os.sep + exportDir
        os.mkdir(self.full_export_path + os.sep + 'output')
        os.mkdir(self.full_export_path + os.sep + 'datasets')

        #check if self.model_path is a directory
        if os.path.isdir(self.model_path):
            #get the directory name
            model_name = os.path.basename(self.model_path)
            #check if model_name can be an int
            try:
                model_name = int(model_name)
                #get the parent directory name
                model_name = os.path.basename(os.path.dirname(self.model_path))
            except:
                pass
            #create a folder in the export folder with the model name
            if not os.path.exists(self.full_export_path + os.sep + 'input_model'+ os.sep + model_name):
                os.mkdir(self.full_export_path + os.sep + 'input_model'+ os.sep + model_name)
            #copy the model to the export folder
            shutil.copytree(self.model_path, self.full_export_path + os.sep +'input_model'+ os.sep+ model_name + os.sep,dirs_exist_ok=True)
            self.model_path= 'input_model' + '/' + model_name
        if os.path.isdir(self.vae_path):
            #get the directory name
            vae_name = os.path.basename(self.vae_path)
            #create a folder in the export folder with the model name
            if not os.path.exists(self.full_export_path + os.sep + 'input_vae_model'+ os.sep + vae_name):
                os.mkdir(self.full_export_path + os.sep + 'input_vae_model'+ os.sep + vae_name)
            #copy the model to the export folder
            shutil.copytree(self.vae_path, self.full_export_path + os.sep +'input_vae_model'+ os.sep+ vae_name + os.sep + vae_name,dirs_exist_ok=True)
            self.vae_path= 'input_vae_model' + '/' + vae_name
        if self.output_path == '':
            self.output_path = 'output'
        else:
            #get the dirname
            output_name = os.path.basename(self.output_path)
            #create a folder in the export folder with the model name
            if not os.path.exists(self.full_export_path + os.sep + 'output'+ os.sep + output_name):
                os.mkdir(self.full_export_path + os.sep + 'output'+ os.sep + output_name)
            self.output_path = 'output' + '/' + output_name
        #loop through the concepts and add them to the export folder
        concept_counter = 0
        new_concepts = []
        for concept in self.concepts:
            concept_counter += 1
            concept_data_dir = os.path.basename(concept['instance_data_dir'])
            #concept is a dict
            #get the concept name
            concept_name = concept['instance_prompt']
            #if concept_name is ''
            if concept_name == '':
                concept_name = 'concept_' + str(concept_counter)
                
            #create a folder in the export/datasets folder with the concept name
            #if not os.path.exists(self.full_export_path + os.sep + 'datasets'+ os.sep + concept_name):
            #    os.mkdir(self.full_export_path + os.sep + 'datasets'+ os.sep + concept_name)
            #copy the concept to the export folder
            shutil.copytree(concept['instance_data_dir'], self.full_export_path + os.sep + 'datasets'+ os.sep + concept_data_dir ,dirs_exist_ok=True)
            concept_class_name = concept['class_prompt']
            if concept_class_name == '':
                #if class_data_dir is ''
                if concept['class_data_dir'] != '':
                    concept_class_name = 'class_' + str(concept_counter)
                    #create a folder in the export/datasets folder with the concept name
                    if not os.path.exists(self.full_export_path + os.sep + 'datasets'+ os.sep + concept_class_name):
                        os.mkdir(self.full_export_path + os.sep + 'datasets'+ os.sep + concept_class_name)
                    #copy the concept to the export folder
                    shutil.copytree(concept['class_data_dir'], self.full_export_path + os.sep + 'datasets'+ os.sep + concept_class_name+ os.sep,dirs_exist_ok=True)
            else:
                if concept['class_data_dir'] != '':
                    #create a folder in the export/datasets folder with the concept name
                    if not os.path.exists(self.full_export_path + os.sep + 'datasets'+ os.sep + concept_class_name):
                        os.mkdir(self.full_export_path + os.sep + 'datasets'+ os.sep + concept_class_name)
                    #copy the concept to the export folder
                    shutil.copytree(concept['class_data_dir'], self.full_export_path + os.sep + 'datasets'+ os.sep + concept_class_name+ os.sep,dirs_exist_ok=True)
            #create a new concept dict
            new_concept = {}
            new_concept['instance_prompt'] = concept_name
            new_concept['instance_data_dir'] = 'datasets' + '/' + concept_data_dir 
            new_concept['class_prompt'] = concept_class_name
            new_concept['class_data_dir'] = 'datasets' + '/' + concept_class_name if concept_class_name != '' else ''
            new_concept['do_not_balance'] = concept['do_not_balance']
            new_concept['use_sub_dirs'] = concept['use_sub_dirs']
            new_concepts.append(new_concept)
        #make scripts folder
        self.save_concept_to_json(filename=self.full_export_path + os.sep + 'stabletune_concept_list.json', preMadeConcepts=new_concepts)
        if not os.path.exists(self.full_export_path + os.sep + 'scripts'):
            os.mkdir(self.full_export_path + os.sep + 'scripts')
        #copy the scripts/trainer.py the scripts folder
        shutil.copy('scripts' + os.sep + 'trainer.py', self.full_export_path + os.sep + 'scripts' + os.sep + 'trainer.py')
        #copy trainer_utils.py to the scripts folder
        shutil.copy('scripts' + os.sep + 'trainer_util.py', self.full_export_path + os.sep + 'scripts' + os.sep + 'trainer_util.py')
        #copy converters.py to the scripts folder
        shutil.copy('scripts' + os.sep + 'converters.py', self.full_export_path + os.sep + 'scripts' + os.sep + 'converters.py')
        #copy model_util.py to the scripts folder
        shutil.copy('scripts' + os.sep + 'model_util.py', self.full_export_path + os.sep + 'scripts' + os.sep + 'model_util.py')
        
    def caption_buddy(self):
        import captionBuddy
        #self.master.overrideredirect(False)
        self.iconify()
        #cb_root = tk.Tk()
        cb_icon =PhotoImage(master=self,file = "resources/stableTuner_icon.png")
        #cb_root.iconphoto(False, cb_icon)
        app2 = captionBuddy.ImageBrowser(self)
        app2.iconphoto(False, cb_icon)
        #app = app2.mainloop()
        #check if app2 is running
        
        
        #self.master.overrideredirect(True)
        #self.master.deiconify()
    def aspect_ratio_mode_toggles(self, *args):
        if self.use_aspect_ratio_bucketing_var.get() == 1:
            self.with_prior_loss_preservation_var.set(0)
            self.with_prior_loss_preservation_checkbox.configure(state="disabled")
            self.aspect_ratio_bucketing_mode_label.configure(state="normal")
            self.aspect_ratio_bucketing_mode_option_menu.configure(state="normal")
            self.dynamic_bucketing_mode_label.configure(state="normal")
            self.dynamic_bucketing_mode_option_menu.configure(state="normal")
            

        else:
            self.with_prior_loss_preservation_checkbox.configure(state="normal")
            self.aspect_ratio_bucketing_mode_label.configure(state="disabled")
            self.aspect_ratio_bucketing_mode_option_menu.configure(state="disabled")
            self.dynamic_bucketing_mode_label.configure(state="disabled")
            self.dynamic_bucketing_mode_option_menu.configure(state="disabled")
            
    
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
            #self.play_interactive_generation_button.configure()
            pass
        else:
            #change the background of the button to normal
            pass
            #self.play_interactive_generation_button.configure(fg=self.dark_mode_title_var)
    def play_save_image(self):


        file = fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")]) 
        #check if png in file name
        if ".png" not in file and file != "" and self.play_current_image:
            file = file + ".png"
        self.play_current_image.save(file)
    def generate_next_image(self):
        self.play_generate_image(self.play_model_entry.get(), self.play_prompt_entry.get(), self.play_negative_prompt_entry.get(), self.play_seed_entry.get(), self.play_scheduler_variable.get(), int(self.play_resolution_slider_height.get()), int(self.play_resolution_slider_width.get()), self.play_cfg_slider.get(), self.play_steps_slider.get())
    def play_generate_image(self, model, prompt, negative_prompt, seed, scheduler, sample_height, sample_width, cfg, steps):
        
        import diffusers
        import torch
        from diffusers.utils.import_utils import is_xformers_available
        self.play_height = sample_height
        self.play_width = sample_width
        #interactive = self.play_interactive_generation_button_bool.get()
        #update generate image button text
        if self.pipe is None or self.play_model_entry.get() != self.current_model:
            if self.pipe is not None:
                del self.pipe
                #clear torch cache
                torch.cuda.empty_cache()
            self.play_generate_image_button["text"] = "Loading Model, Please stand by..."
            #self.play_generate_image_button.configure(fg="red")
            self.play_generate_image_button.update()
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(model,torch_dtype=torch.float16,safety_checker=None)
            self.pipe.to('cuda')
            self.current_model = model
            if scheduler == 'DPMSolverMultistepScheduler':
                scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            if scheduler == 'PNDMScheduler':
                scheduler = diffusers.PNDMScheduler.from_config(self.pipe.scheduler.config)
            if scheduler == 'DDIMScheduler':
                scheduler = diffusers.DDIMScheduler.from_config(self.pipe.scheduler.config)
            if scheduler == 'EulerAncestralDiscreteScheduler':
                scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            if scheduler == 'EulerDiscreteScheduler':
                scheduler = diffusers.EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler = scheduler
            if is_xformers_available():
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                    except Exception as e:
                        print(
                            "Could not enable memory efficient attention. Make sure xformers is installed"
                            f" correctly and a GPU is available: {e}"
                        )
        def displayInterImg(step: int, timestep: int, latents: torch.FloatTensor):
            #tensor to image
            img = self.pipe.decode_latents(latents)
            image = self.pipe.numpy_to_pil(img)[0]
            #convert to PIL image
            self.play_current_image = ctk.CTkImage(image)
            #if step == 0:
                #self.play_image_canvas.configure(width=self.play_width, height=self.play_height)
                #if self.play_width < self.master.winfo_width():
                    #self.play_width = self.master.winfo_width()
                    #self.master.geometry(f"{self.play_width}x{self.play_height+300}")
                #self.play_image = self.play_image_canvas.create_image(0, 0, anchor="nw", image=self.play_current_image)
                #self.play_image_canvas.update()
            #update image
            self.play_image_canvas.itemconfig(self.play_image, image=self.play_current_image)
            self.play_image_canvas.update()
        with torch.autocast("cuda"), torch.inference_mode():
            if seed == "" or seed == " ":
                seed = -1
            seed = int(seed)
            if seed == -1 or seed == 0 or self.play_keep_seed_var.get() == 0:
                #random seed
                seed = random.randint(0, 10000000)
                self.play_seed_entry.delete(0, "end")
                self.play_seed_entry.insert(0, seed)
            generator = torch.Generator("cuda").manual_seed(seed)
            #self.play_generate_image_button["text"] = "Generating, Please stand by..."
            #self.play_generate_image_button.configure(fg=self.dark_mode_title_var)
            #self.play_generate_image_button.update()
            image = self.pipe(prompt=prompt,negative_prompt=negative_prompt,height=int(sample_height),width=int(sample_width), guidance_scale=cfg, num_inference_steps=int(steps),generator=generator).images[0]
            self.play_current_image = image
            #image is PIL image
            if self.generation_window is None:
                self.generation_window = GeneratedImagePreview(self)
            self.generation_window.ingest_image(self.play_current_image)
            #focus
            self.generation_window.focus_set()
            
            #image = ctk.CTkImage(image)
            #self.play_image_canvas.configure(width=sample_width, height=sample_height)
            #self.play_image_canvas.create_image(0, 0, anchor="nw", image=image)
            #self.play_image_canvas.image = image
            #resize app to fit image, add current height to image height
            #if sample width is lower than current width, use current width
            #if sample_width < self.master.winfo_width():
            #    sample_width = self.master.winfo_width()
            #self.master.geometry(f"{sample_width}x{sample_height+self.tabsSizes[5][1]}")
            #refresh the window
            if self.play_save_image_button == None:
                self.play_save_image_button = ctk.CTkButton(self.playground_frame_subframe, text="Save Image", command=self.play_save_image)
                self.play_save_image_button.grid(row=10, column=2, columnspan=1, sticky="ew", padx=5, pady=5)
            #self.master.update()
            #self.play_generate_image_button["text"] = "Generate Image"
            #normal text
            #self.play_generate_image_button.configure(fg=self.dark_mode_text_var)
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
        empty_label = ctk.CTkLabel(self.convert_model_dialog, text="")
        empty_label.pack()
        label = ctk.CTkLabel(self.convert_model_dialog, text="Converting CKPT to Diffusers. Please wait...")
        label.pack()
        self.convert_model_dialog.geometry("300x70")
        self.convert_model_dialog.resizable(False, False)
        self.convert_model_dialog.grab_set()
        self.convert_model_dialog.focus_set()
        self.update()
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
        empty_label = ctk.CTkLabel(self.convert_model_dialog, text="")
        empty_label.pack()
        label = ctk.CTkLabel(self.convert_model_dialog, text="Converting Diffusers to CKPT. Please wait...")
        label.pack()
        self.convert_model_dialog.geometry("300x70")
        self.convert_model_dialog.resizable(False, False)
        self.convert_model_dialog.grab_set()
        self.convert_model_dialog.focus_set()
        self.update()
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
        newImage=ctk.CTkImage(image)
        self.preview_images[indexOfEntry][2] = newImage
        canvas.itemconfig(image_container, image=newImage)
    def remove_new_concept(self):
        #get the last concept widget
        if len(self.concept_widgets) > 0:
            concept_widget = self.concept_widgets[-1]
            #remove it from the list
            self.concept_widgets.remove(concept_widget)
            #destroy the widget
            concept_widget.destroy()
            #repack the widgets
            #self.repack_concepts()
    def add_new_concept(self,concept=None):
        #create a new concept
        #for concept in self.concept_widgets check if concept was deleted
        #if it was, remove it from the list
        row=0
        column=len(self.concept_widgets)
        
        if len(self.concept_widgets) > 6:
            row=1
            concept_widget = ConceptWidget(self.data_frame_concepts_subframe, concept,width=100,height=100)
            width=100
            height=100
            column=len(self.concept_widgets)-7
            if len(self.concept_widgets) > 13:
                row=2
                concept_widget = ConceptWidget(self.data_frame_concepts_subframe, concept,width=100,height=100)
                height=100
                width=100
                column=len(self.concept_widgets)-14
                if len(self.concept_widgets) > 20:
                    messagebox.showerror("Error", "You can only have 21 concepts")
                    return
        else:
            concept_widget = ConceptWidget(self.data_frame_concepts_subframe, concept,width=100,height=100)   
        #print(row)
        concept_widget.grid(row=row, column=column, sticky="e",padx=13, pady=10)
        self.concept_widgets.append(concept_widget)
        self.update()
        #print(len(self.concept_widgets))
        #if row == 2:
        #    for concept in self.concept_widgets:
        #        concept.resize_widget(width, height)

        
        
    def add_concept(self, inst_prompt_val=None, class_prompt_val=None, inst_data_path_val=None, class_data_path_val=None, do_not_balance_val=False):
        #create a title for the new concept
        concept_title = ctk.CTkLabel(self.data_frame_concepts_subframe, text="Concept " + str(len(self.concept_labels)+1), font=("Helvetica", 10, "bold"), bg_color='#333333')
        concept_title.grid(row=3 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance prompt label
        ins_prompt_label = ctk.CTkLabel(self.data_frame_concepts_subframe, text="Token/Prompt", bg_color='#333333')
        ins_prompt_label_ttp = CreateToolTip(ins_prompt_label, "The token for the concept, will be ignored if use image names as captions is checked.")
        ins_prompt_label.grid(row=4 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance prompt entry
        ins_prompt_entry = ctk.CTkEntry(self.data_frame_concepts_subframe, bg_color='#333333')
        ins_prompt_entry.grid(row=4 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if inst_prompt_val != None:
            ins_prompt_entry.insert(0, inst_prompt_val)
        #create class prompt label
        class_prompt_label = ctk.CTkLabel(self.data_frame_concepts_subframe, text="Class Prompt", bg_color='#333333')
        class_prompt_label_ttp = CreateToolTip(class_prompt_label, "The prompt will be used to generate class images and train the class images if added to dataset")
        class_prompt_label.grid(row=5 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create class prompt entry
        class_prompt_entry = ctk.CTkEntry(self.data_frame_concepts_subframe,width=50, bg_color='#333333')
        class_prompt_entry.grid(row=5 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if class_prompt_val != None:
            class_prompt_entry.insert(0, class_prompt_val)
        #create instance data path label
        ins_data_path_label = ctk.CTkLabel(self.data_frame_concepts_subframe, text="Training Data Directory", bg_color='#333333')
        ins_data_path_label_ttp = CreateToolTip(ins_data_path_label, "The path to the folder containing the concept's images.")
        ins_data_path_label.grid(row=6 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #create instance data path entry
        ins_data_path_entry = ctk.CTkEntry(self.data_frame_concepts_subframe,width=50, bg_color='#333333')
        ins_data_path_entry.bind("<FocusOut>", self.update_preview_image)
        #bind to insert
        ins_data_path_entry.grid(row=6 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if inst_data_path_val != None:
            #focus on the entry
            
            ins_data_path_entry.insert(0, inst_data_path_val)
            ins_data_path_entry.focus_set()
            #focus on main window
            self.focus_set()
        #add a button to open a file dialog to select the instance data path
        ins_data_path_file_dialog_button = ctk.CTkButton(self.data_frame_concepts_subframe, text="...", command=lambda: self.open_file_dialog(ins_data_path_entry), bg_color='#333333')
        ins_data_path_file_dialog_button.grid(row=6 + (len(self.concept_labels)*6), column=2, sticky="nsew")
        #create class data path label
        class_data_path_label = ctk.CTkLabel(self.data_frame_concepts_subframe, text="Class Data Directory", bg_color='#333333')
        class_data_path_label_ttp = CreateToolTip(class_data_path_label, "The path to the folder containing the concept's class images.")
        class_data_path_label.grid(row=7 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        #add a button to open a file dialog to select the class data path
        class_data_path_file_dialog_button = ctk.CTkButton(self.data_frame_concepts_subframe, text="...", command=lambda: self.open_file_dialog(class_data_path_entry), bg_color='#333333')
        class_data_path_file_dialog_button.grid(row=7 + (len(self.concept_labels)*6), column=2, sticky="nsew")
        #create class data path entry
        class_data_path_entry = ctk.CTkEntry(self.data_frame_concepts_subframe, bg_color='#333333')
        class_data_path_entry.grid(row=7 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        if class_data_path_val != None:
            class_data_path_entry.insert(0, class_data_path_val)
        #add a checkbox to do not balance dataset
        do_not_balance_dataset_var = tk.IntVar()
        #label for checkbox
        do_not_balance_dataset_label = ctk.CTkLabel(self.data_frame_concepts_subframe, text="Do not balance dataset", bg_color='#333333')
        do_not_balance_dataset_label_ttp = CreateToolTip(do_not_balance_dataset_label, "If checked, the dataset will not be balanced. this settings overrides the global auto balance setting, if there's a concept you'd like to train without balance while the others will.")
        do_not_balance_dataset_label.grid(row=8 + (len(self.concept_labels)*6), column=0, sticky="nsew")
        do_not_balance_dataset_checkbox = ctk.CTkSwitch(self.data_frame_concepts_subframe, variable=do_not_balance_dataset_var, bg_color='#333333')
        do_not_balance_dataset_checkbox.grid(row=8 + (len(self.concept_labels)*6), column=1, sticky="nsew")
        do_not_balance_dataset_var.set(0)

        #create a preview of the images in the path on the right side of the concept
        #create a frame to hold the images
        #empty column to separate the images from the rest of the concept
        
        #sep = ctk.CTkLabel(self.data_frame_concepts_subframe,padx=3, text="").grid(row=4 + (len(self.concept_labels)*6), column=3, sticky="nsew", bg_color='#333333')

        image_preview_frame = ctk.CTkFrame(self.data_frame_concepts_subframe)
        image_preview_frame.grid(row=4 + (len(self.concept_labels)*6), column=4, rowspan=4, sticky="ne")
        #create a label for the images
        #image_preview_label = ctk.CTkLabel(image_preview_frame, text="Image Preview")
        #image_preview_label.grid(row=0, column=0, sticky="nsew")
        #create a canvas to hold the images
        image_preview_canvas = tk.Canvas(image_preview_frame)
        
        #flat border
        image_preview_canvas.configure(border=0, relief='flat', highlightthickness=0)
        #canvas size is 100x100
        image_preview_canvas.configure(width=150, height=150, bg='#333333')
        image_preview_canvas.grid(row=0, column=0, sticky="nsew")
        #debug test, image preview just white
        #if there's a path in the entry, show the images in the path
        #grab stableTuner_icon.png from the resources folder
        icon = 'resources/stableTuner_icon.png'
        #create a photoimage object of the image in the path
        icon = Image.open(icon)
        #resize the image
        image = icon.resize((150, 150), Image.Resampling.LANCZOS)
        image_preview = ImageTk.PhotoImage(image)
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
                    image_preview = ctk.CTkImage(image)
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
        #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
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
            
            for folder in self.required_folders:
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
                empty_label = ctk.CTkLabel(self.convert_model_dialog, text="")
                empty_label.pack()
                label = ctk.CTkLabel(self.convert_model_dialog, text="Converting CKPT to Diffusers. Please wait...")
                label.pack()
                self.convert_model_dialog.geometry("300x70")
                self.convert_model_dialog.resizable(False, False)
                self.convert_model_dialog.grab_set()
                self.convert_model_dialog.focus_set()
                self.update()
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
        #self.master.focus_set()

    def save_concept_to_json(self,filename=None,preMadeConcepts=None):
        #dialog box to select the file to save to
        if filename == None:
            file = fd.asksaveasfile(mode='w', defaultextension=".json", filetypes=[("JSON", "*.json")])
            #check if file has json extension
            if 'json' not in file.name:
                file.name = file.name + '.json'
        else:
            file = open(filename, 'w')
        if file != None:
            if preMadeConcepts == None:
                concepts = []
                for widget in self.concept_widgets:
                    concept = widget.concept
                    concept_dict = {'instance_prompt' : concept.concept_name, 'class_prompt' : concept.concept_class_name, 'instance_data_dir' : concept.concept_path, 'class_data_dir' : concept.concept_class_path,'flip_p' : concept.flip_p, 'do_not_balance' : concept.concept_do_not_balance, 'use_sub_dirs' : concept.process_sub_dirs}
                    concepts.append(concept_dict)
                if file != None:
                    #write the json to the file
                    json.dump(concepts, file, indent=4)
                    #close the file
                    file.close()
            else:
                json.dump(preMadeConcepts, file, indent=4)
                #close the file
                file.close()
    def load_concept_from_json(self):
        #
        #dialog
        concept_json = fd.askopenfilename(title = "Select file",filetypes = (("json files","*.json"),("all files","*.*")))
        for i in range(len(self.concept_widgets)):
                self.remove_new_concept()
        self.concept_entries = []
        self.concept_labels = []
        self.concepts = []
        with open(concept_json, "r") as f:
            concept_json = json.load(f)
        for concept in concept_json:
            #print(concept)
            if 'flip_p' not in concept:
                concept['flip_p'] = ''
            concept = Concept(concept_name=concept["instance_prompt"], class_name=concept["class_prompt"], concept_path=concept["instance_data_dir"], class_path=concept["class_data_dir"],flip_p=concept['flip_p'],balance_dataset=concept["do_not_balance"], process_sub_dirs=concept["use_sub_dirs"])
            self.add_new_concept(concept)        #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update()
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
            #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def remove_new_concept(self):
        #remove the last concept
        #print(self.concept_widgets)
        if len(self.concept_widgets) > 0:
            
            self.concept_widgets[-1].destroy()
            self.concept_widgets.pop()
            #self.preview_images.pop()
            #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
    def toggle_telegram_settings(self):
        #print(self.send_telegram_updates_var.get())
        if self.send_telegram_updates_var.get() == 1:
            self.telegram_token_label.configure(state="normal")
            self.telegram_token_entry.configure(state="normal")
            self.telegram_chat_id_label.configure(state="normal")
            self.telegram_chat_id_entry.configure(state="normal")
        else:
            self.telegram_token_label.configure(state="disabled")
            self.telegram_token_entry.configure(state="disabled")
            self.telegram_chat_id_label.configure(state="disabled")
            self.telegram_chat_id_entry.configure(state="disabled")
    def add_controlled_seed_sample(self,value=""):
        if len(self.controlled_seed_sample_labels) <= 4:
            self.controlled_seed_sample_labels.append(ctk.CTkLabel(self.sampling_frame_subframe,bg_color='transparent' ,text="Controlled Seed Sample " + str(len(self.controlled_seed_sample_labels)+1)))
            self.controlled_seed_sample_labels[-1].grid(row=self.controlled_sample_row + len(self.sample_prompts) + len(self.controlled_seed_sample_labels), column=2, padx=10, pady=5,sticky="nwes")
            #create entry
            entry = ctk.CTkEntry(self.sampling_frame_subframe,width=250)
            entry.bind("<Button-3>",self.create_right_click_menu)
            self.controlled_seed_sample_entries.append(entry)
            self.controlled_seed_sample_entries[-1].grid(row=self.controlled_sample_row + len(self.sample_prompts) + len(self.controlled_seed_sample_entries), column=3, padx=10, pady=5,sticky="w")
            if value != "":
                self.controlled_seed_sample_entries[-1].insert(0, value)
            self.add_controlled_seed_to_sample.append(value)
            #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def remove_controlled_seed_sample(self):
        #get the entry and label to remove
        if len(self.controlled_seed_sample_labels) > 0:
            self.controlled_seed_sample_labels[-1].destroy()
            self.controlled_seed_sample_labels.pop()
            self.controlled_seed_sample_entries[-1].destroy()
            self.controlled_seed_sample_entries.pop()
            self.add_controlled_seed_to_sample.pop()
            #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
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
            #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            for i in self.controlled_seed_buttons:
                #push to next row
                i.grid(row=i.grid_info()["row"] - 1, column=i.grid_info()["column"], sticky="nsew")
            for i in self.controlled_seed_sample_labels:
                #push to next row
                i.grid(row=i.grid_info()["row"] - 1, column=i.grid_info()["column"], sticky="nsew")
            for i in self.controlled_seed_sample_entries:
                #push to next row
                i.grid(row=i.grid_info()["row"] - 1, column=i.grid_info()["column"], sticky="nsew")


    def add_sample_prompt(self,value=""):
        #add a new label and entry
        if len(self.sample_prompt_entries) <= 4:
            self.sample_prompt_labels.append(ctk.CTkLabel(self.sampling_frame_subframe, text="Sample Prompt " + str(len(self.sample_prompt_labels)+1),bg_color='transparent'))
            self.sample_prompt_labels[-1].grid(row=self.sample_prompt_row + len(self.sample_prompt_labels) - 1, column=2, padx=10, pady=5,sticky="nsew")
            entry = ctk.CTkEntry(self.sampling_frame_subframe,width=250)
            entry.bind("<Button-3>", self.create_right_click_menu)
            self.sample_prompt_entries.append(entry)
            self.sample_prompt_entries[-1].grid(row=self.sample_prompt_row + len(self.sample_prompt_labels) - 1, column=3, padx=10, pady=5,sticky="nsew")
            
            if value != "":
                self.sample_prompt_entries[-1].insert(0, value)
            #update the sample prompts list
            self.sample_prompts.append(value)
            for i in self.controlled_seed_buttons:
                #push to next row
                i.grid(row=i.grid_info()["row"] + 1, column=i.grid_info()["column"], sticky="nsew")
            for i in self.controlled_seed_sample_labels:
                #push to next row
                i.grid(row=i.grid_info()["row"] + 1, column=i.grid_info()["column"], sticky="nsew")
            for i in self.controlled_seed_sample_entries:
                #push to next row
                i.grid(row=i.grid_info()["row"] + 1, column=i.grid_info()["column"], sticky="nsew")
        #print(self.sample_prompts)
        #print(self.sample_prompt_entries)
        #update canvas scroll region
        #self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        
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
        
        self.update()
    def update_concepts(self):
        #update the concepts list
        #if the first index is a dict
        if isinstance(self.concepts, dict):
            return
        self.concepts = []
        for i in range(len(self.concept_widgets)):
            concept = self.concept_widgets[i].concept
            self.concepts.append({'instance_prompt' : concept.concept_name, 'class_prompt' : concept.concept_class_name, 'instance_data_dir' : concept.concept_path, 'class_data_dir' : concept.concept_class_path,'flip_p' : concept.flip_p, 'do_not_balance' : concept.concept_do_not_balance, 'use_sub_dirs' : concept.process_sub_dirs})
    def save_config(self, config_file=None):
        #save the configure file
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
        configure = {}
        self.update_controlled_seed_sample()
        self.update_sample_prompts()
        self.update_concepts()
        configure["concepts"] = self.concepts
        #print(self.concepts)
        configure["sample_prompts"] = self.sample_prompts
        configure['add_controlled_seed_to_sample'] = self.add_controlled_seed_to_sample
        configure["model_path"] = self.input_model_path_entry.get()
        configure["vae_path"] = self.vae_model_path_entry.get()
        configure["output_path"] = self.output_path_entry.get()
        configure["send_telegram_updates"] = self.send_telegram_updates_var.get()
        configure["telegram_token"] = self.telegram_token_entry.get()
        configure["telegram_chat_id"] = self.telegram_chat_id_entry.get()
        configure["resolution"] = self.resolution_var.get()
        configure["batch_size"] = self.train_batch_size_var.get()
        configure["train_epocs"] = self.train_epochs_entry.get()
        configure["mixed_precision"] = self.mixed_precision_var.get()
        configure["use_8bit_adam"] = self.use_8bit_adam_var.get()
        configure["use_gradient_checkpointing"] = self.use_gradient_checkpointing_var.get()
        configure["accumulation_steps"] = self.gradient_accumulation_steps_var.get()
        configure["learning_rate"] = self.learning_rate_entry.get()
        configure["warmup_steps"] = self.num_warmup_steps_entry.get()
        configure["learning_rate_scheduler"] = self.learning_rate_scheduler_var.get()
        #configure["use_latent_cache"] = self.use_latent_cache_var.get()
        #configure["save_latent_cache"] = self.save_latent_cache_var.get()
        configure["regenerate_latent_cache"] = self.regenerate_latent_cache_var.get()
        configure["train_text_encoder"] = self.train_text_encoder_var.get()
        configure["with_prior_loss_preservation"] = self.with_prior_loss_preservation_var.get()
        configure["prior_loss_preservation_weight"] = self.prior_loss_preservation_weight_entry.get()
        configure["use_image_names_as_captions"] = self.use_image_names_as_captions_var.get()
        configure["auto_balance_concept_datasets"] = self.auto_balance_dataset_var.get()
        configure["add_class_images_to_dataset"] = self.add_class_images_to_dataset_var.get()
        configure["number_of_class_images"] = self.number_of_class_images_entry.get()
        configure["save_every_n_epochs"] = self.save_every_n_epochs_entry.get()
        configure["number_of_samples_to_generate"] = self.number_of_samples_to_generate_entry.get()
        configure["sample_height"] = self.sample_height_entry.get()
        configure["sample_width"] = self.sample_width_entry.get()
        configure["sample_random_aspect_ratio"] = self.sample_random_aspect_ratio_var.get()
        configure['sample_on_training_start'] = self.sample_on_training_start_var.get()
        configure['concepts'] = self.concepts
        configure['aspect_ratio_bucketing'] = self.use_aspect_ratio_bucketing_var.get()
        configure['seed'] = self.seed_entry.get()
        configure['dataset_repeats'] = self.dataset_repeats_entry.get()
        configure['limit_text_encoder_training'] = self.limit_text_encoder_entry.get()
        configure['use_text_files_as_captions'] = self.use_text_files_as_captions_var.get()
        configure['ckpt_version'] = self.ckpt_sd_version
        configure['convert_to_ckpt_after_training'] = self.convert_to_ckpt_after_training_var.get()
        configure['execute_post_conversion'] = self.convert_to_ckpt_after_training_var.get()
        configure['disable_cudnn_benchmark'] = self.disable_cudnn_benchmark_var.get()
        configure['sample_step_interval'] = self.sample_step_interval_entry.get()
        configure['conditional_dropout'] = self.conditional_dropout_entry.get()
        configure["clip_penultimate"] = self.clip_penultimate_var.get()
        configure['use_ema'] = self.use_ema_var.get()
        configure['aspect_ratio_bucketing_mode'] = self.aspect_ratio_bucketing_mode_var.get()
        configure['dynamic_bucketing_mode'] = self.dynamic_bucketing_mode_var.get()
        configure['model_variant'] = self.model_variant_var.get()
        #save the configure file
        #if the file exists, delete it
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, "w",encoding='utf-8') as f:
            json.dump(configure, f, indent=4)
            f.close()
    def load_config(self,file_name=None):
        #load the configure file
        #ask the user for a file name
        if file_name == None:
            file_name = fd.askopenfilename(title = "Select file",filetypes = (("json files","*.json"),("all files","*.*")))
        if file_name == "":
            return
        #load the configure file
        with open(file_name, "r",encoding='utf-8') as f:
            configure = json.load(f)

        #load concepts
        try:
            for i in range(len(self.concept_widgets)):
                self.remove_new_concept()
            self.concept_entries = []
            self.concept_labels = []
            self.concepts = []
            for i in range(len(configure["concepts"])):
                inst_prompt = configure["concepts"][i]["instance_prompt"]
                class_prompt = configure["concepts"][i]["class_prompt"]
                inst_data_dir = configure["concepts"][i]["instance_data_dir"]
                class_data_dir = configure["concepts"][i]["class_data_dir"]
                if 'flip_p' not in configure["concepts"][i]:
                    print(configure["concepts"][i].keys())
                    configure["concepts"][i]['flip_p'] = ''
                flip_p = configure["concepts"][i]["flip_p"]
                balance_dataset = configure["concepts"][i]["do_not_balance"]
                process_sub_dirs = configure["concepts"][i]["use_sub_dirs"]
                concept = Concept(concept_name=inst_prompt, class_name=class_prompt, concept_path=inst_data_dir, class_path=class_data_dir,flip_p=flip_p,balance_dataset=balance_dataset,process_sub_dirs=process_sub_dirs)
                self.add_new_concept(concept)
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
        for i in range(len(configure["sample_prompts"])):
            self.add_sample_prompt(value=configure["sample_prompts"][i])
        for i in range(len(configure['add_controlled_seed_to_sample'])):
            self.add_controlled_seed_sample(value=configure['add_controlled_seed_to_sample'][i])
            
        self.input_model_path_entry.delete(0, tk.END)
        self.input_model_path_entry.insert(0, configure["model_path"])
        self.vae_model_path_entry.delete(0, tk.END)
        self.vae_model_path_entry.insert(0, configure["vae_path"])
        self.output_path_entry.delete(0, tk.END)
        self.output_path_entry.insert(0, configure["output_path"])
        self.send_telegram_updates_var.set(configure["send_telegram_updates"])
        if configure["send_telegram_updates"]:
            self.telegram_token_entry.configure(state='normal')
            self.telegram_chat_id_entry.configure(state='normal')
            self.telegram_token_label.configure(state='normal')
            self.telegram_chat_id_label.configure(state='normal')
        self.telegram_token_entry.delete(0, tk.END)
        self.telegram_token_entry.insert(0, configure["telegram_token"])
        self.telegram_chat_id_entry.delete(0, tk.END)
        self.telegram_chat_id_entry.insert(0, configure["telegram_chat_id"])
        self.resolution_var.set(configure["resolution"])
        self.train_batch_size_var.set(configure["batch_size"])
        self.train_epochs_entry.delete(0, tk.END)
        self.train_epochs_entry.insert(0, configure["train_epocs"])
        self.mixed_precision_var.set(configure["mixed_precision"])
        self.use_8bit_adam_var.set(configure["use_8bit_adam"])
        self.use_gradient_checkpointing_var.set(configure["use_gradient_checkpointing"])
        self.gradient_accumulation_steps_var.set(configure["accumulation_steps"])
        self.learning_rate_entry.delete(0, tk.END)
        self.learning_rate_entry.insert(0, configure["learning_rate"])
        self.num_warmup_steps_entry.delete(0, tk.END)
        self.num_warmup_steps_entry.insert(0, configure["warmup_steps"])
        self.learning_rate_scheduler_var.set(configure["learning_rate_scheduler"])
        #self.use_latent_cache_var.set(configure["use_latent_cache"])
        #self.save_latent_cache_var.set(configure["save_latent_cache"])
        self.regenerate_latent_cache_var.set(configure["regenerate_latent_cache"])
        self.train_text_encoder_var.set(configure["train_text_encoder"])
        self.with_prior_loss_preservation_var.set(configure["with_prior_loss_preservation"])
        self.prior_loss_preservation_weight_entry.delete(0, tk.END)
        self.prior_loss_preservation_weight_entry.insert(0, configure["prior_loss_preservation_weight"])
        self.use_image_names_as_captions_var.set(configure["use_image_names_as_captions"])
        self.auto_balance_dataset_var.set(configure["auto_balance_concept_datasets"])
        self.add_class_images_to_dataset_var.set(configure["add_class_images_to_dataset"])
        self.number_of_class_images_entry.delete(0, tk.END)
        self.number_of_class_images_entry.insert(0, configure["number_of_class_images"])
        self.save_every_n_epochs_entry.delete(0, tk.END)
        self.save_every_n_epochs_entry.insert(0, configure["save_every_n_epochs"])
        self.number_of_samples_to_generate_entry.delete(0, tk.END)
        self.number_of_samples_to_generate_entry.insert(0, configure["number_of_samples_to_generate"])
        self.sample_height_entry.delete(0, tk.END)
        self.sample_height_entry.insert(0, configure["sample_height"])
        self.sample_width_entry.delete(0, tk.END)
        self.sample_width_entry.insert(0, configure["sample_width"])
        self.sample_random_aspect_ratio_var.set(configure["sample_random_aspect_ratio"])
        self.sample_on_training_start_var.set(configure["sample_on_training_start"])
        self.use_aspect_ratio_bucketing_var.set(configure["aspect_ratio_bucketing"])
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, configure["seed"])
        self.dataset_repeats_entry.delete(0, tk.END)
        self.dataset_repeats_entry.insert(0, configure["dataset_repeats"])
        self.limit_text_encoder_entry.delete(0, tk.END)
        if configure["limit_text_encoder_training"] != '0':
            self.limit_text_encoder_entry.insert(0, configure["limit_text_encoder_training"])
        self.use_text_files_as_captions_var.set(configure["use_text_files_as_captions"])
        self.convert_to_ckpt_after_training_var.set(configure["convert_to_ckpt_after_training"])
        if configure["execute_post_conversion"]:
            self.execute_post_conversion = True
        self.disable_cudnn_benchmark_var.set(configure["disable_cudnn_benchmark"])
        self.sample_step_interval_entry.delete(0, tk.END)
        self.sample_step_interval_entry.insert(0, configure["sample_step_interval"])
        self.conditional_dropout_entry.delete(0, tk.END)
        self.conditional_dropout_entry.insert(0, configure["conditional_dropout"])
        self.clip_penultimate_var.set(configure["clip_penultimate"])
        self.use_ema_var.set(configure["use_ema"])
        if configure["aspect_ratio_bucketing"]:
            self.aspect_ratio_bucketing_mode_label.configure(state='normal')
            self.aspect_ratio_bucketing_mode_option_menu.configure(state='normal')
            self.dynamic_bucketing_mode_label.configure(state='normal')
            self.dynamic_bucketing_mode_option_menu.configure(state='normal')
        else:
            self.aspect_ratio_bucketing_mode_label.configure(state='disabled')
            self.aspect_ratio_bucketing_mode_option_menu.configure(state='disabled')
            self.dynamic_bucketing_mode_label.configure(state='disabled')
            self.dynamic_bucketing_mode_option_menu.configure(state='disabled')
        self.model_variant_var.set(configure["model_variant"])
        self.aspect_ratio_bucketing_mode_var.set(configure["aspect_ratio_bucketing_mode"])
        self.dynamic_bucketing_mode_var.set(configure["dynamic_bucketing_mode"])
        self.update()
    
    def process_inputs(self,export=None):
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
        #self.use_latent_cache = self.use_latent_cache_var.get()
        #self.save_latent_cache = self.save_latent_cache_var.get()
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
        self.cloud_mode = self.cloud_mode_var.get()
        self.conditional_dropout = self.conditional_dropout_entry.get()
        self.clip_penultimate = self.clip_penultimate_var.get()
        self.use_ema = self.use_ema_var.get()
        self.aspect_ratio_bucketing_mode = self.aspect_ratio_bucketing_mode_var.get()
        self.dynamic_bucketing_mode = self.dynamic_bucketing_mode_var.get()
        self.model_variant = self.model_variant_var.get()
        mode = 'normal'
        if self.cloud_mode == False and export == None:
            #check if output path exists
            if os.path.exists(self.output_path) == True:
                #check if output path is empty
                if len(os.listdir(self.output_path)) > 0:
                    #show a messagebox asking if the user wants to overwrite the output path
                    overwrite = messagebox.askyesno("Overwrite Output Path", "The output path is not empty. Do you want to overwrite it?")
                    if overwrite == False:
                        return
                    else:
                        #delete the contents of the output path but the logs or 0 directory
                        for file in os.listdir(self.output_path):
                            if file != 'logs' and file != '0':
                                if os.path.isdir(self.output_path + '/' + file) == True:
                                    shutil.rmtree(self.output_path + '/' + file)
                                else:
                                    os.remove(self.output_path + '/' + file)

                        
        if self.cloud_mode == True or export == 'LinuxCMD':
            if export == 'LinuxCMD':
                mode = 'LinuxCMD'
            export='Linux'
            #create a sessionName for the cloud based on the output path name and the time
            #format time and date to %month%day%hour%minute
            now = datetime.now()
            dt_string = now.strftime("%m-%d-%H-%M")
            self.export_name = self.output_path.split('/')[-1].split('\\')[-1] + '_' + dt_string
            self.packageForCloud()
        
        if int(self.train_epocs) == 0 or self.train_epocs == '':
            messagebox.showerror("Error", "Number of training epochs must be greater than 0")
            return
        #open stabletune_concept_list.json
        if os.path.exists('stabletune_last_run.json'):
            try:
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
                                
                                messagebox.showinfo("StableTuner", "Configuration changed, regenerating latent cache")
                        except:
                            print("Error trying to see if regenerating latent cache is needed, this means it probably needs to be regenerated and ST was updated recently.")
                            pass
                    else:
                        messagebox.showinfo("StableTuner", "Configuration changed, regenerating latent cache")
                        self.regenerate_latent_cache = True
            except Exception as e:
                print(e)
                print("Error checking last run, regenerating latent cache")
                self.regenerate_latent_cache = True

        #create a bat file to run the training
        if self.mixed_precision == 'fp16' or self.mixed_precision == 'bf16':

            batBase = f'accelerate "launch" "--mixed_precision={self.mixed_precision}" "scripts/trainer.py"'
            if export == 'Linux':
                batBase = f'accelerate launch --mixed_precision="{self.mixed_precision}" scripts/trainer.py'
        else:
            batBase = 'accelerate "launch" "--mixed_precision=no" "scripts/trainer.py"'
            if export == 'Linux':
                batBase = f'accelerate launch --mixed_precision="no" scripts/trainer.py'
        
        if self.model_variant == 'Regular':
            if export == 'Linux':
                batBase += ' --model_variant="base"'
            else:
                batBase += ' "--model_variant=base" '
        elif self.model_variant == 'Inpaint':
            if export == 'Linux':
                batBase += ' --model_variant="inpainting"'
            else:
                batBase += ' "--model_variant=inpainting" '

        if self.disable_cudnn_benchmark == True:
            if export == 'Linux':
                batBase += ' --disable_cudnn_benchmark'
            else:
                batBase += ' "--disable_cudnn_benchmark" '
        if self.use_text_files_as_captions == True:
            if export == 'Linux':
                batBase += ' --use_text_files_as_captions'
            else:
                batBase += ' "--use_text_files_as_captions" '
        if self.sample_step_interval != '0' or self.sample_step_interval != '' or self.sample_step_interval != ' ':
            if export == 'Linux':
                batBase += f' --sample_step_interval={self.sample_step_interval}'
            else:
                batBase += f' "--sample_step_interval={self.sample_step_interval}" '
            try:
                #if limit_text_encoder is a percentage calculate what epoch to stop at
                if '%' in self.limit_text_encoder:
                    percent = float(self.limit_text_encoder.replace('%',''))
                    stop_epoch = int((int(self.train_epocs) * percent) / 100)
                    if export == 'Linux':
                        batBase += f' --stop_text_encoder_training={stop_epoch}'
                    else:
                        batBase += f' "--stop_text_encoder_training={stop_epoch}" '
                elif '%' not in self.limit_text_encoder and self.limit_text_encoder != '' and self.limit_text_encoder != ' ' and self.limit_text_encoder != '0':
                    if export == 'Linux':
                        batBase += f' --stop_text_encoder_training={self.limit_text_encoder}'
                    else:
                        batBase += f' "--stop_text_encoder_training={self.limit_text_encoder}" '
            except:
                pass
        if export=='Linux':
            batBase += f' --pretrained_model_name_or_path="{self.model_path}" '
            batBase += f' --pretrained_vae_name_or_path="{self.vae_path}" '
            batBase += f' --output_dir="{self.output_path}" '
            batBase += f' --seed={self.seed_number} '
            batBase += f' --resolution={self.resolution} '
            batBase += f' --train_batch_size={self.batch_size} '
            batBase += f' --num_train_epochs={self.train_epocs} '
        else:
            batBase += f' "--pretrained_model_name_or_path={self.model_path}" '
            batBase += f' "--pretrained_vae_name_or_path={self.vae_path}" '
            batBase += f' "--output_dir={self.output_path}" '
            batBase += f' "--seed={self.seed_number}" '
            batBase += f' "--resolution={self.resolution}" '
            batBase += f' "--train_batch_size={self.batch_size}" '
            batBase += f' "--num_train_epochs={self.train_epocs}" '

        if self.mixed_precision == 'fp16' or self.mixed_precision == 'bf16':
            if export == 'Linux':
                batBase += f' --mixed_precision="{self.mixed_precision}"'
            else:
                batBase += f' "--mixed_precision={self.mixed_precision}" '
        if self.use_aspect_ratio_bucketing:
            if export == 'Linux':
                batBase += ' --use_bucketing'
            else:
                batBase += f' "--use_bucketing" '
            if self.aspect_ratio_bucketing_mode == 'Dynamic Fill':
                com = 'dynamic'
            if self.aspect_ratio_bucketing_mode == 'Drop Fill':
                com = 'truncate'
            if self.aspect_ratio_bucketing_mode == 'Duplicate Fill':
                com = 'add'
            if export == 'Linux':
                batBase += f' --aspect_mode="{com}"'
            else:
                batBase += f' "--aspect_mode={com}" '
            if self.dynamic_bucketing_mode == 'Duplicate':
                com = 'add'
            if self.dynamic_bucketing_mode == 'Drop':
                com = 'truncate'
            if export == 'Linux':
                batBase += f' --aspect_mode_action_preference="{com}"'
            else:
                batBase += f' "--aspect_mode_action_preference={com}" '
        if self.use_8bit_adam == True:
            if export == 'Linux':
                batBase += ' --use_8bit_adam'
            else:
                batBase += f' "--use_8bit_adam" '
        if self.use_gradient_checkpointing == True:
            if export == 'Linux':
                batBase += ' --gradient_checkpointing'
            else:
                batBase += f' "--gradient_checkpointing" '
        
        if export == 'Linux':
            batBase += f' --gradient_accumulation_steps={self.accumulation_steps}'
            batBase += f' --learning_rate={self.learning_rate}'
            batBase += f' --lr_warmup_steps={self.warmup_steps}'
            batBase += f' --lr_scheduler="{self.learning_rate_scheduler}"'
        else:   
            batBase += f' "--gradient_accumulation_steps={self.accumulation_steps}" '
            batBase += f' "--learning_rate={self.learning_rate}" '
            batBase += f' "--lr_warmup_steps={self.warmup_steps}" '
            batBase += f' "--lr_scheduler={self.learning_rate_scheduler}" '
        if self.regenerate_latent_cache == True:
            if export == 'Linux':
                batBase += ' --regenerate_latent_cache'
            else:
                batBase += f' "--regenerate_latent_cache" '
        if self.train_text_encoder == True:
            if export == 'Linux':
                batBase += ' --train_text_encoder'
            else:
                batBase += f' "--train_text_encoder" '
        if self.with_prior_loss_preservation == True and self.use_aspect_ratio_bucketing == False:
            if export == 'Linux':
                batBase += ' --with_prior_preservation'
                batBase += f' --prior_loss_weight={self.prior_loss_preservation_weight}'
            else:
                batBase += f' "--with_prior_preservation" '
                batBase += f' "--prior_loss_weight={self.prior_loss_preservation_weight}" '
        elif self.with_prior_loss_preservation == True and self.use_aspect_ratio_bucketing == True:
            print('loss preservation isnt supported with aspect ratio bucketing yet, sorry!')
        if self.use_image_names_as_captions == True:
            if export == 'Linux':
                batBase += ' --use_image_names_as_captions'
            else:
                batBase += f' "--use_image_names_as_captions" '
        if self.auto_balance_concept_datasets == True:
            if export == 'Linux':
                batBase += ' --auto_balance_concept_datasets'
            else:
                batBase += f' "--auto_balance_concept_datasets" '
        if self.add_class_images_to_dataset == True and self.with_prior_loss_preservation == False:
            if export == 'Linux':
                batBase += ' --add_class_images_to_dataset'
            else:
                batBase += f' "--add_class_images_to_dataset" '
        if export == 'Linux':
            batBase += f' --concepts_list="{self.concept_list_json_path}"'
            batBase += f' --num_class_images={self.number_of_class_images}'
            batBase += f' --save_every_n_epoch={self.save_every_n_epochs}'
            batBase += f' --n_save_sample={self.number_of_samples_to_generate}'
            batBase += f' --sample_height={self.sample_height}'
            batBase += f' --sample_width={self.sample_width}'
            batBase += f' --dataset_repeats={self.dataset_repeats}'
        else:
            batBase += f' "--concepts_list={self.concept_list_json_path}" '
            batBase += f' "--num_class_images={self.number_of_class_images}" '
            batBase += f' "--save_every_n_epoch={self.save_every_n_epochs}" '
            batBase += f' "--n_save_sample={self.number_of_samples_to_generate}" '
            batBase += f' "--sample_height={self.sample_height}" '
            batBase += f' "--sample_width={self.sample_width}" '
            batBase += f' "--dataset_repeats={self.dataset_repeats}" '
        if self.sample_random_aspect_ratio == True:
            if export == 'Linux':
                batBase += ' --sample_aspect_ratios'
            else:
                batBase += f' "--sample_aspect_ratios" '
        if self.send_telegram_updates == True:
            if export == 'Linux':
                batBase += ' --send_telegram_updates'
                batBase += f' --telegram_token="{self.telegram_token}"'
                batBase += f' --telegram_chat_id="{self.telegram_chat_id}"'
            else:
                batBase += f' "--send_telegram_updates" '
                batBase += f' "--telegram_token={self.telegram_token}" '
                batBase += f' "--telegram_chat_id={self.telegram_chat_id}" '
        #remove duplicates from self.sample_prompts
        
        self.sample_prompts = list(dict.fromkeys(self.sample_prompts))
        #remove duplicates from self.add_controlled_seed_to_sample
        self.add_controlled_seed_to_sample = list(dict.fromkeys(self.add_controlled_seed_to_sample))
        for i in range(len(self.sample_prompts)):
            if export == 'Linux':
                batBase += f' --add_sample_prompt="{self.sample_prompts[i]}"'
            else:
                batBase += f' "--add_sample_prompt={self.sample_prompts[i]}" '
        for i in range(len(self.add_controlled_seed_to_sample)):
            if export == 'Linux':
                batBase += f' --save_sample_controlled_seed={self.add_controlled_seed_to_sample[i]}'
            else:
                batBase += f' "--save_sample_controlled_seed={self.add_controlled_seed_to_sample[i]}" '
        if self.sample_on_training_start == True:
            if export == 'Linux':
                batBase += ' --sample_on_training_start'
            else:
                batBase += f' "--sample_on_training_start" '
        if len(self.conditional_dropout) > 0 and self.conditional_dropout != ' ' and self.conditional_dropout != '0':
            #if % is in the string, remove it
            if '%' in self.conditional_dropout:
                self.conditional_dropout = self.conditional_dropout.replace('%', '')
                #convert to float from percentage string
                self.conditional_dropout = float(self.conditional_dropout) / 100
            else:
                #check if float
                try:
                    self.conditional_dropout = float(self.conditional_dropout)
                except:
                    print('Error: Conditional Dropout must be a percent between 0 and 100, or a decimal between 0 and 1.')
            #print(self.conditional_dropout)
            #if self.coniditional dropout is a float
            if isinstance(self.conditional_dropout, float):
                if export == 'Linux':
                    batBase += f' --conditional_dropout={self.conditional_dropout}'
                else:
                    batBase += f' "--conditional_dropout={self.conditional_dropout}" '
        #save configure
            
        if self.clip_penultimate == True:
            if export == 'Linux':
                batBase += ' --clip_penultimate'
            else:
                batBase += f' "--clip_penultimate" '
        if self.use_ema == True:
            if export == 'Linux':
                batBase += ' --use_ema'
            else:
                batBase += f' "--use_ema" '
        
        self.save_config('stabletune_last_run.json')
        
        if export == False:
            #save the bat file
            with open("scripts/train.bat", "w", encoding="utf-8") as f:
                f.write(batBase)
            #close the window
            self.destroy()
            #run the bat file
            self.quit()
            train = os.system(r".\scripts\train.bat")
            #if exit code is 0, then the training was successful
            if train == 0:
                app = App()
                app.mainloop()
            #if user closed the window or keyboard interrupt, then cancel conversion
            elif train == 1:
                os.system("pause")
            
            #restart the app
        elif export == 'win':
            with open("train.bat", "w", encoding="utf-8") as f:
                f.write(batBase)
            #show message
            messagebox.showinfo("Export", "Exported to train.bat")
        elif mode == 'LinuxCMD':
            #copy batBase to clipboard
            trainer_index = batBase.find('trainer.py')+11
            batStart = batBase[:trainer_index]
            batCommands = batBase[trainer_index:]
            #split on -- and remove the first element
            batCommands = batCommands.split('--')
            batBase = batStart+' \\\n'
            for command in batCommands[1:]:
                #add the -- back
                if command != batCommands[-1]:
                    command = '  --'+command+'\\'+'\n'
                else:
                    command = '  --'+command
                batBase += command
            pyperclip.copy('!'+batBase)
            shutil.rmtree(self.full_export_path)
            messagebox.showinfo("Export", "Copied new training command to clipboard.")
            return
        elif export == 'Linux' and self.cloud_mode == True:
            notebook = 'resources/stableTuner_notebook.ipynb'
            #load the notebook as a dictionary
            with open(notebook) as f:
                nb = json.load(f)
            #get the last cell
            #find the cell with the source that contains changeMe
            #format batBase so it won't be one line
            #find index in batBase of the trainer.py
            trainer_index = batBase.find('trainer.py')+11
            batStart = batBase[:trainer_index]
            batCommands = batBase[trainer_index:]
            #split on -- and remove the first element
            batCommands = batCommands.split('--')
            batBase = batStart+' \\\n'
            for command in batCommands[1:]:
                #add the -- back
                if command != batCommands[-1]:
                    command = '  --'+command+'\\'+'\n'
                else:
                    command = '  --'+command
                batBase += command
            for i in range(len(nb['cells'])):
                if 'changeMe' in nb['cells'][i]['source']:
                    code_cell = nb['cells'][i]
                    index = i
                    code_cell['source'] = '!'+batBase
                    #replace the last cell with the new one
                    nb['cells'][index] = code_cell
                    break
            
            #save the notebook to the export folder
            shutil.copy('requirements.txt', self.full_export_path)
            #zip up everything in export without the folder itself
            shutil.make_archive('payload', 'zip', self.full_export_path)
            #move the zip file to the export folder
            shutil.move('payload.zip', self.full_export_path)
            #save the notebook to the export folder
            with open(self.full_export_path+os.sep+'stableTuner_notebook.ipynb', 'w') as f:
                json.dump(nb, f)
            #delete everything in the export folder except the zip file and the notebook
            for file in os.listdir(self.full_export_path):
                if file.endswith('.zip') or file.endswith('.ipynb'):
                    continue
                else:
                    #if it's a folder, delete it
                    if os.path.isdir(self.full_export_path+os.sep+file):
                        shutil.rmtree(self.full_export_path+os.sep+file)
                    #if it's a file, delete it
                    else:
                        os.remove(self.full_export_path+os.sep+file)
            #show message
            messagebox.showinfo("Success", f"Your cloud\linux payload is ready to go!\nSaved to: {self.full_export_path}\n\nUpload the files and run the notebook to start training.")
        


        
#root = ctk.CTk()
app = App()
app.mainloop()
